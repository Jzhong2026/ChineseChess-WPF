using System.Diagnostics;
using ChineseChess.Encoding;
using ChineseChess.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ChineseChess.Services;

/// <summary>
/// Neural network-based AI service using an ONNX policy-value network.
///
/// Supports two inference modes:
///   1. Pure NN — selects the legal move with the highest policy logit (fast, weaker).
///   2. NN + Alpha-Beta — uses the value network as evaluation function inside the
///      existing Negamax search (stronger, slower).
///
/// The ONNX model is expected to have:
///   Input:  "board_input"   [batch, 1260]  for MLP
///                        or [batch, 14, 10, 9] for CNN
///   Output: "policy_logits" [batch, 8100]
///           "value_pred"    [batch]  (tanh, -1..1 from Red's perspective)
/// </summary>
public sealed class XiangqiNeuralAiService : IDisposable
{
    private readonly InferenceSession _session;
    private readonly XiangqiEngine _engine;
    private readonly bool _isCnnModel;

    // CNN layout constants (planes × rows × cols)
    private const int Planes = BoardEncoder.Planes;          // 14
    private const int BoardRows = XiangqiEngine.BoardRows;   // 10
    private const int BoardCols = XiangqiEngine.BoardCols;   // 9

    public XiangqiNeuralAiService(string onnxModelPath, XiangqiEngine engine, bool isCnnModel = false)
    {
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        _session = new InferenceSession(onnxModelPath, sessionOptions);
        _engine = engine;
        _isCnnModel = isCnnModel;
    }

    /// <summary>
    /// Choose a move using pure neural network policy (no tree search).
    /// Selects the legal move with the highest policy logit.
    /// Fast (~1ms), useful for blitz or as a benchmark.
    /// </summary>
    public Move? ChooseMoveByPolicy(Piece?[,] board, Side side, IReadOnlyList<Move> history)
    {
        var legalMoves = _engine.GetLegalMoves(board, side, history);
        if (legalMoves.Count == 0) return null;

        var (policyLogits, _) = RunInference(board);

        // Find the legal move with the highest logit
        Move? bestMove = null;
        var bestLogit = float.NegativeInfinity;
        foreach (var move in legalMoves)
        {
            var actionId = MoveEncoder.Encode(move);
            var logit = policyLogits[0, actionId];
            if (logit > bestLogit)
            {
                bestLogit = logit;
                bestMove = move;
            }
        }

        return bestMove;
    }

    /// <summary>
    /// Evaluate the board position from the specified side's perspective.
    /// Returns the value network output, adjusted to [side's] perspective.
    /// Range: [-1, 1] (1 = winning, -1 = losing).
    /// </summary>
    public float EvaluatePosition(Piece?[,] board, Side side)
    {
        var (_, valuePred) = RunInference(board);
        // value_pred is from Red's perspective; flip for Black
        var value = valuePred[0];
        return side == Side.Red ? value : -value;
    }

    /// <summary>
    /// Returns (policyLogits, valuePred) tensors.
    /// policyLogits: [1, 8100]
    /// valuePred:    [1]
    /// </summary>
    public (DenseTensor<float> PolicyLogits, DenseTensor<float> ValuePred) RunInference(Piece?[,] board)
    {
        var inputTensor = _isCnnModel ? EncodeCnn(board) : EncodeMlp(board);
        var inputs = new[] { NamedOnnxValue.CreateFromTensor("board_input", inputTensor) };

        using var results = _session.Run(inputs);

        // Policy logits: [1, 8100]
        var policyTensor = results.First(r => r.Name == "policy_logits").AsTensor<float>();
        var policyDense = new DenseTensor<float>(new[] { 1, MoveEncoder.ActionCount });
        for (var i = 0; i < MoveEncoder.ActionCount; i++)
            policyDense[0, i] = policyTensor[0, i];

        // Value pred: [1]
        var valueTensor = results.First(r => r.Name == "value_pred").AsTensor<float>();
        var valueDense = new DenseTensor<float>(new[] { 1 });
        valueDense[0] = valueTensor[0];

        return (policyDense, valueDense);
    }

    /// <summary>
    /// Get policy probability distribution over legal moves (softmax applied).
    /// </summary>
    public IReadOnlyList<(Move Move, float Probability)> GetPolicyDistribution(
        Piece?[,] board, Side side, IReadOnlyList<Move> history)
    {
        var legalMoves = _engine.GetLegalMoves(board, side, history);
        if (legalMoves.Count == 0) return Array.Empty<(Move, float)>();

        var (policyLogits, _) = RunInference(board);

        // Extract logits for legal moves only and apply softmax
        var legalLogits = legalMoves
            .Select(m => (Move: m, Logit: policyLogits[0, MoveEncoder.Encode(m)]))
            .ToList();

        var maxLogit = legalLogits.Max(x => x.Logit);
        var expSum = legalLogits.Sum(x => MathF.Exp(x.Logit - maxLogit));

        return legalLogits
            .Select(x => (x.Move, Probability: MathF.Exp(x.Logit - maxLogit) / expSum))
            .OrderByDescending(x => x.Probability)
            .ToList();
    }

    // ─── Private helpers ────────────────────────────────────────────────────

    /// <summary>MLP input: flat float[1, 1260]</summary>
    private static DenseTensor<float> EncodeMlp(Piece?[,] board)
    {
        var encoded = BoardEncoder.Encode(board);
        var tensor = new DenseTensor<float>(new[] { 1, BoardEncoder.EncodedLength });
        for (var i = 0; i < encoded.Length; i++)
            tensor[0, i] = encoded[i];
        return tensor;
    }

    /// <summary>
    /// CNN input: [1, 14, 10, 9].
    ///
    /// BoardEncoder.Encode() returns float[1260] with layout:
    ///   index = (row * 9 + col) * 14 + plane
    ///
    /// We need to re-pack into [plane, row, col]:
    ///   cnn_index = plane * (10*9) + row * 9 + col
    /// </summary>
    private static DenseTensor<float> EncodeCnn(Piece?[,] board)
    {
        var flat = BoardEncoder.Encode(board);
        var tensor = new DenseTensor<float>(new[] { 1, Planes, BoardRows, BoardCols });

        for (var row = 0; row < BoardRows; row++)
        {
            for (var col = 0; col < BoardCols; col++)
            {
                for (var plane = 0; plane < Planes; plane++)
                {
                    // flat index from BoardEncoder.GetIndex: square * Planes + plane
                    var flatIdx = (row * BoardCols + col) * Planes + plane;
                    tensor[0, plane, row, col] = flat[flatIdx];
                }
            }
        }

        return tensor;
    }

    public void Dispose() => _session.Dispose();
}
