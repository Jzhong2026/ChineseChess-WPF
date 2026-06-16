using System.Diagnostics;
using ChineseChess.Encoding;
using ChineseChess.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ChineseChess.Services;

/// <summary>
/// Neural network-based AI service using an ONNX policy-value network.
///
/// Supports three model architectures (auto-detected from ONNX metadata):
///   1. cnn_factored_v5 — 16 input planes, factored from/to heads, current-player value:
///      Input:  [batch, 16, 10, 9] = 14 piece planes + 2 side-to-move planes
///      Output: from_logits [batch, 90], to_logits [batch, 90], value_pred [batch]
///      Value: from current player's perspective (+1 = I'm winning)
///
///   2. cnn_factored — 14 input planes, factored from/to heads, Red-perspective value:
///      Input:  [batch, 14, 10, 9]
///      Output: from_logits [batch, 90], to_logits [batch, 90], value_pred [batch]
///      Value: from Red's perspective; flip for Black
///
///   3. cnn (legacy) — 14 input planes, flat 8100 policy head:
///      Input:  [batch, 14, 10, 9]
///      Output: policy_logits [batch, 8100], value_pred [batch]
///
/// The model type is auto-detected from ONNX input/output metadata.
/// </summary>
public sealed class XiangqiNeuralAiService : IDisposable
{
    private readonly InferenceSession _session;
    private readonly XiangqiEngine _engine;
    private readonly bool _isCnnModel;
    private readonly bool _isFactoredModel;
    private readonly bool _isV5Model;       // V5: has side-to-move planes + current-player value
    private readonly int _inputPlanes;      // 16 for V5, 14 for legacy

    // CNN layout constants
    private const int BoardPlanes = BoardEncoder.Planes;  // 14
    private const int SidePlanes = 2;                      // Red-to-move + Black-to-move
    private const int BoardRows = XiangqiEngine.BoardRows; // 10
    private const int BoardCols = XiangqiEngine.BoardCols; // 9
    private const int BoardSize = BoardRows * BoardCols;   // 90

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

        // Auto-detect factored model by checking output names
        _isFactoredModel = _session.OutputMetadata.Any(kv => kv.Key == "from_logits");

        // Auto-detect V5 model by checking input shape (16 planes vs 14)
        var inputMeta = _session.InputMetadata.First();
        var inputShape = inputMeta.Value.Dimensions;
        // Input shape: [batch, planes, rows, cols] → planes dimension is index 1
        _inputPlanes = inputShape.Length >= 2 ? inputShape[1] : BoardPlanes;
        _isV5Model = _inputPlanes >= 16;

        Debug.WriteLine($"[NeuralAI] Model: {(IsFactoredModel ? "Factored" : "Flat")}, " +
                        $"InputPlanes={_inputPlanes}, V5={_isV5Model}");
    }

    /// <summary>Whether this model uses factored from/to policy heads.</summary>
    public bool IsFactoredModel => _isFactoredModel;

    /// <summary>Whether this is a V5 model with side-to-move planes.</summary>
    public bool IsV5Model => _isV5Model;

    /// <summary>
    /// Choose a move using pure neural network policy (no tree search).
    /// Selects the legal move with the highest policy logit.
    /// Fast (~1ms), useful for blitz or as a benchmark.
    /// </summary>
    public Move? ChooseMoveByPolicy(Piece?[,] board, Side side, IReadOnlyList<Move> history)
    {
        var legalMoves = _engine.GetLegalMoves(board, side, history);
        if (legalMoves.Count == 0) return null;

        var (policyLogits, _) = RunInference(board, side);

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
    /// Range: [-1, 1] (1 = winning, -1 = losing).
    ///
    /// V5 models: value is already from current player's perspective — no flip needed.
    /// Legacy models: value is from Red's perspective — flip for Black.
    /// </summary>
    public float EvaluatePosition(Piece?[,] board, Side side)
    {
        var (_, valuePred) = RunInference(board, side);
        var value = valuePred[0];

        // V5: value is from current player's perspective already
        // Legacy: value is from Red's perspective; flip for Black
        return _isV5Model ? value : (side == Side.Red ? value : -value);
    }

    /// <summary>
    /// Returns (policyLogits, valuePred) tensors.
    /// policyLogits: [1, 8100] (combined from/to logits for factored models)
    /// valuePred:    [1]
    /// </summary>
    public (DenseTensor<float> PolicyLogits, DenseTensor<float> ValuePred) RunInference(Piece?[,] board, Side side)
    {
        var inputTensor = _isCnnModel ? EncodeCnn(board, side) : EncodeMlp(board);
        var inputs = new[] { NamedOnnxValue.CreateFromTensor("board_input", inputTensor) };

        using var results = _session.Run(inputs);

        if (_isFactoredModel)
        {
            // Factored model: from_logits [1,90] + to_logits [1,90] → combined [1,8100]
            var fromTensor = results.First(r => r.Name == "from_logits").AsTensor<float>();
            var toTensor = results.First(r => r.Name == "to_logits").AsTensor<float>();

            var policyDense = new DenseTensor<float>(new[] { 1, MoveEncoder.ActionCount });

            // Combined: policy_logit[from*90+to] = from_logit[from] + to_logit[to]
            for (var fromIdx = 0; fromIdx < BoardSize; fromIdx++)
            {
                var fromLogit = fromTensor[0, fromIdx];
                for (var toIdx = 0; toIdx < BoardSize; toIdx++)
                {
                    policyDense[0, fromIdx * BoardSize + toIdx] = fromLogit + toTensor[0, toIdx];
                }
            }

            // Value pred: [1]
            var valueTensor = results.First(r => r.Name == "value_pred").AsTensor<float>();
            var valueDense = new DenseTensor<float>(new[] { 1 });
            valueDense[0] = valueTensor[0];

            return (policyDense, valueDense);
        }
        else
        {
            // Legacy flat model: policy_logits [1, 8100]
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
    }

    /// <summary>
    /// Get policy probability distribution over legal moves (softmax applied).
    /// </summary>
    public IReadOnlyList<(Move Move, float Probability)> GetPolicyDistribution(
        Piece?[,] board, Side side, IReadOnlyList<Move> history)
    {
        var legalMoves = _engine.GetLegalMoves(board, side, history);
        if (legalMoves.Count == 0) return Array.Empty<(Move, float)>();

        var (policyLogits, _) = RunInference(board, side);

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
    /// CNN input: [1, inputPlanes, 10, 9].
    ///
    /// V5 models (16 planes): 14 piece planes + 2 side-to-move indicator planes
    /// Legacy models (14 planes): 14 piece planes only
    ///
    /// BoardEncoder.Encode() returns float[1260] with layout:
    ///   index = (row * 9 + col) * 14 + plane  (square-major)
    ///
    /// We re-pack into plane-major [plane, row, col] and add side planes if V5.
    /// </summary>
    private DenseTensor<float> EncodeCnn(Piece?[,] board, Side side)
    {
        var flat = BoardEncoder.Encode(board);
        var tensor = new DenseTensor<float>(new[] { 1, _inputPlanes, BoardRows, BoardCols });

        // Pack 14 piece planes: square-major → plane-major
        for (var row = 0; row < BoardRows; row++)
        {
            for (var col = 0; col < BoardCols; col++)
            {
                for (var plane = 0; plane < BoardPlanes; plane++)
                {
                    var flatIdx = (row * BoardCols + col) * BoardPlanes + plane;
                    tensor[0, plane, row, col] = flat[flatIdx];
                }
            }
        }

        // V5: Add side-to-move indicator planes
        if (_isV5Model)
        {
            // Plane 14: Red-to-move (all 1s if Red's turn)
            // Plane 15: Black-to-move (all 1s if Black's turn)
            var redPlane = side == Side.Red ? 1f : 0f;
            var blackPlane = side == Side.Black ? 1f : 0f;
            for (var row = 0; row < BoardRows; row++)
            {
                for (var col = 0; col < BoardCols; col++)
                {
                    tensor[0, 14, row, col] = redPlane;
                    tensor[0, 15, row, col] = blackPlane;
                }
            }
        }

        return tensor;
    }

    public void Dispose() => _session.Dispose();
}
