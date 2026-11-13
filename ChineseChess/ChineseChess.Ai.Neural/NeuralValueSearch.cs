using ChineseChess.Ai;
using ChineseChess.Evaluation;
using ChineseChess.Models;
using ChineseChess.Search;
using ChineseChess.Services;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ChineseChess.Ai.Neural;

public interface IOnnxSearchEvaluator : IPositionEvaluator, IDisposable
{
    long Evaluations { get; }
    long CacheHits { get; }
    void BeginSearch();
}

/// <summary>Single-search-thread evaluator. Every static score comes from ONNX.</summary>
public sealed class OnnxValueEvaluator : IOnnxSearchEvaluator
{
    private readonly InferenceSession _session;
    private readonly DenseTensor<float> _input = new(new[] { 1, 1260 });
    private readonly DenseTensor<long> _indices = new(new[] { 1, 90 });
    private readonly bool _sparse;
    private readonly NamedOnnxValue[] _inputs;
    private readonly bool _cacheEvaluations;
    private readonly Dictionary<PositionKey, double> _cache = new();
    public long Evaluations { get; private set; }
    public long CacheHits { get; private set; }

    public OnnxValueEvaluator(string modelPath, bool cacheEvaluations = true)
    {
        _cacheEvaluations = cacheEvaluations;
        using var options = new SessionOptions { IntraOpNumThreads = 1, GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        _session = new InferenceSession(modelPath, options);
        var metadata = _session.ModelMetadata.CustomMetadataMap;
        metadata.TryGetValue("encoding", out var encoding);
        var dense = encoding == "canonical-piece-planes-14-v1" && _session.InputMetadata.TryGetValue("board_input", out var input)
            && input.Dimensions.Length == 2 && input.Dimensions[1] == 1260;
        _sparse = encoding == "canonical-active-indices-90-v1" && _session.InputMetadata.TryGetValue("board_indices", out var indices)
            && indices.Dimensions.Length == 2 && indices.Dimensions[1] == 90;
        if ((!dense && !_sparse) || !metadata.TryGetValue("score", out var score) || score != "current-player-centipawns"
            || !_session.OutputMetadata.ContainsKey("value_cp"))
        {
            _session.Dispose();
            throw new ArgumentException("Expected canonical current-player ONNX centipawn evaluator", nameof(modelPath));
        }
        _inputs = new[] { _sparse ? NamedOnnxValue.CreateFromTensor("board_indices", _indices) : NamedOnnxValue.CreateFromTensor("board_input", _input) };
    }

    internal static void EncodeActive(Piece?[,] board, Side perspective, Span<long> target)
    {
        target.Fill(1260);
        var count = 0;
        for (var row = 0; row < 10; row++)
        for (var col = 0; col < 9; col++)
        {
            var piece = board[row, col];
            if (piece is null) continue;
            var square = perspective == Side.Red ? row * 9 + col : (9 - row) * 9 + 8 - col;
            var plane = (piece.Side == perspective ? 0 : 7) + (int)piece.Type;
            target[count++] = plane * 90 + square;
        }
        target[..count].Sort();
    }

    internal static void Encode(Piece?[,] board, Side perspective, Span<float> target)
    {
        target.Clear();
        for (var row = 0; row < 10; row++)
        for (var col = 0; col < 9; col++)
        {
            var piece = board[row, col];
            if (piece is null) continue;
            var square = perspective == Side.Red ? row * 9 + col : (9 - row) * 9 + 8 - col;
            var plane = (piece.Side == perspective ? 0 : 7) + (int)piece.Type;
            target[plane * 90 + square] = 1;
        }
    }

    public double Evaluate(Piece?[,] board, Side perspective, Side aiSide)
    {
        var key = _cacheEvaluations ? MakeKey(board, perspective) : default;
        if (_cacheEvaluations && _cache.TryGetValue(key, out var cached)) { CacheHits++; return cached; }
        if (_sparse) EncodeActive(board, perspective, _indices.Buffer.Span);
        else Encode(board, perspective, _input.Buffer.Span);
        using var outputs = _session.Run(_inputs);
        var value = outputs.First(o => o.Name == "value_cp").AsTensor<float>().GetValue(0);
        Evaluations++;
        if (!float.IsFinite(value)) throw new InvalidOperationException("Non-finite ONNX evaluation");
        var score = Math.Clamp(value, -30000, 30000);
        if (_cacheEvaluations && _cache.Count < 100000) _cache[key] = score;
        return score;
    }

    public void BeginSearch() { _cache.Clear(); CacheHits = 0; }

    // Exact six-word canonical board key. Dictionary equality checks every word;
    // a hash collision cannot cause another position's network score to be used.
    internal readonly record struct PositionKey(ulong A, ulong B, ulong C, ulong D, ulong E, ulong F);
    internal static PositionKey MakeKey(Piece?[,] board, Side perspective)
    {
        Span<ulong> words = stackalloc ulong[6];
        words.Clear();
        for (var row = 0; row < 10; row++)
        for (var col = 0; col < 9; col++)
        {
            var piece = board[row, col];
            if (piece is null) continue;
            var square = perspective == Side.Red ? row * 9 + col : (9 - row) * 9 + 8 - col;
            var code = 1 + (piece.Side == perspective ? 0 : 7) + (int)piece.Type;
            words[square / 16] |= (ulong)code << ((square % 16) * 4);
        }
        return new(words[0], words[1], words[2], words[3], words[4], words[5]);
    }

    public void Dispose() => _session.Dispose();
}

/// <summary>
/// Generic alpha-beta/quiescence with an ONNX evaluator, not the classic AI's
/// handcrafted PositionEvaluator or move-selection service. No teacher process.
/// </summary>
public sealed class NeuralValueAlphaBetaAiEngine : IAiEngine, IDisposable
{
    private readonly IOnnxSearchEvaluator _value;
    private readonly AlphaBetaSearch _search;
    public string Name => "NN Value + AlphaBeta";
    public long LastEvaluations { get; private set; }
    public long LastCacheHits { get; private set; }

    public NeuralValueAlphaBetaAiEngine(string modelPath, XiangqiEngine engine, bool useFairyFeatures = false)
    {
        _value = useFairyFeatures ? new FairyOnnxEvaluator(modelPath) : new OnnxValueEvaluator(modelPath);
        _search = new AlphaBetaSearch(engine, _value);
    }

    public AiMoveResult ChooseMove(AiMoveRequest request)
    {
        var before = _value.Evaluations;
        _value.BeginSearch();
        var result = _search.Search(request.Board, request.Side, 8, request.History,
            request.TimeLimitMs, request.Selection ?? MoveSelectionOptions.Deterministic);
        LastEvaluations = _value.Evaluations - before;
        LastCacheHits = _value.CacheHits;
        return new AiMoveResult(result.Move, result.Stats);
    }

    public void Dispose() => _value.Dispose();
}
