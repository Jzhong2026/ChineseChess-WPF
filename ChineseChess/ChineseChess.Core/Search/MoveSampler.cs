using ChineseChess.Models;

namespace ChineseChess.Search;

/// <summary>
/// Near-best move sampling for self-play diversity, including per-seed
/// reproducible RNG. Separated from search so selection policy is testable and
/// swappable (SRP).
/// </summary>
public sealed class MoveSampler
{
    private readonly Random _random = new();
    private readonly Dictionary<int, Random> _seeded = new();

    public Random GetRandom(MoveSelectionOptions options)
    {
        if (!options.Seed.HasValue) return _random;
        if (_seeded.TryGetValue(options.Seed.Value, out var seeded)) return seeded;
        seeded = new Random(options.Seed.Value);
        _seeded[options.Seed.Value] = seeded;
        return seeded;
    }

    public Move PickRandom(IReadOnlyList<Move> moves, MoveSelectionOptions options) =>
        moves[GetRandom(options).Next(moves.Count)];

    public Move Select(IReadOnlyList<(Move Move, double Score)> scoredMoves, Move fallbackMove, double bestScore, MoveSelectionOptions options)
    {
        if (scoredMoves.Count == 0) return fallbackMove;
        if (!options.EnableRandomSelection) return fallbackMove;

        var sorted = scoredMoves.OrderByDescending(item => item.Score).ToList();
        var topK = Math.Clamp(options.TopK, 1, sorted.Count);
        var nearBestWindow = Math.Max(0, options.NearBestWindow);
        var minTopKScore = sorted[topK - 1].Score;
        var threshold = Math.Max(bestScore - nearBestWindow, minTopKScore);
        var candidates = sorted.Where(item => item.Score >= threshold).Select(item => item.Move).ToList();

        return candidates.Count == 0 ? fallbackMove : candidates[GetRandom(options).Next(candidates.Count)];
    }
}