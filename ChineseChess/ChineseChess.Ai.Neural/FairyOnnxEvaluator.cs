using ChineseChess.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ChineseChess.Ai.Neural;

/// <summary>ONNX-only HalfKAv2Variants evaluator. No external engine or search.</summary>
public sealed class FairyOnnxEvaluator : IOnnxSearchEvaluator
{
    private readonly InferenceSession _session;
    private readonly DenseTensor<long> _us = new(new[] { 1, 32 });
    private readonly DenseTensor<long> _them = new(new[] { 1, 32 });
    private readonly DenseTensor<long> _bucket = new(new[] { 1 });
    private readonly NamedOnnxValue[] _inputs;
    private readonly bool _cacheEvaluations;
    private readonly Dictionary<string, double> _cache = new(StringComparer.Ordinal);
    private static readonly int[] Groups = { 6, 1, 5, 4, 0, 2, 3 };
    public long Evaluations { get; private set; }
    public long CacheHits { get; private set; }

    public FairyOnnxEvaluator(string path, bool cacheEvaluations = true)
    {
        _cacheEvaluations = cacheEvaluations;
        using var options = new SessionOptions { IntraOpNumThreads = 1, GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        _session = new InferenceSession(path, options);
        var metadata = _session.ModelMetadata.CustomMetadataMap;
        if (!metadata.TryGetValue("encoding", out var encoding) || encoding != "xiangqi-halfka-v2-10530-v1"
            || !metadata.TryGetValue("score", out var score) || score != "current-player-centipawns"
            || !_session.InputMetadata.ContainsKey("us") || !_session.InputMetadata.ContainsKey("them")
            || !_session.InputMetadata.ContainsKey("bucket") || !_session.OutputMetadata.ContainsKey("value_cp"))
        {
            _session.Dispose();
            throw new ArgumentException("Expected converted xiangqi HalfKAv2 ONNX model", nameof(path));
        }
        _inputs = new[] { NamedOnnxValue.CreateFromTensor("us", _us), NamedOnnxValue.CreateFromTensor("them", _them), NamedOnnxValue.CreateFromTensor("bucket", _bucket) };
    }

    internal static int Encode(Piece?[,] board, Side perspective, Span<long> target)
    {
        var kingRow = -1; var kingCol = -1;
        for (var row = 0; row < 10; row++)
        for (var col = 0; col < 9; col++)
            if (board[row, col] is { Type: PieceType.General } king && king.Side == perspective) { kingRow = row; kingCol = col; }
        var rank = perspective == Side.Red ? 9 - kingRow : kingRow;
        if (kingRow < 0 || rank is < 0 or > 2 || kingCol is < 3 or > 5) throw new ArgumentException("King outside supported palace");
        var offset = (rank * 3 + kingCol - 3) * 1170;
        target.Fill(10530);
        var count = 0;
        for (var row = 0; row < 10; row++)
        for (var col = 0; col < 9; col++)
        {
            var piece = board[row, col];
            if (piece is null) continue;
            if (count >= 32) throw new ArgumentException("Too many pieces for xiangqi network");
            var group = piece.Type == PieceType.General ? 12 : Groups[(int)piece.Type] * 2 + (piece.Side == perspective ? 0 : 1);
            var square = (perspective == Side.Red ? 9 - row : row) * 9 + col;
            target[count++] = offset + group * 90 + square;
        }
        target[..count].Sort();
        return count;
    }

    public double Evaluate(Piece?[,] board, Side perspective, Side aiSide)
    {
        var ownKing = false; var enemyKing = false;
        foreach (var p in board)
            if (p is { Type: PieceType.General }) { if (p.Side == perspective) ownKing = true; else enemyKing = true; }
        if (!ownKing) return -30000;
        if (!enemyKing) return 30000;
        // Absolute board + side: this network uses rank reflection, not the
        // 180-degree canonical symmetry of the smaller distilled MLP.
        var key = _cacheEvaluations ? MctsAiService.PositionKey(board, perspective) : string.Empty;
        if (_cacheEvaluations && _cache.TryGetValue(key, out var cached)) { CacheHits++; return cached; }
        var count = Encode(board, perspective, _us.Buffer.Span);
        Encode(board, perspective == Side.Red ? Side.Black : Side.Red, _them.Buffer.Span);
        _bucket.Buffer.Span[0] = Math.Min((count - 1) * 8 / 32, 7);
        using var outputs = _session.Run(_inputs);
        var value = outputs.First(o => o.Name == "value_cp").AsTensor<float>().GetValue(0);
        Evaluations++;
        if (!float.IsFinite(value)) throw new InvalidOperationException("Non-finite NNUE ONNX value");
        var result = Math.Clamp(value, -30000, 30000);
        if (_cacheEvaluations && _cache.Count < 100000) _cache[key] = result;
        return result;
    }

    public void BeginSearch() { _cache.Clear(); CacheHits = 0; }
    public void Dispose() => _session.Dispose();
}
