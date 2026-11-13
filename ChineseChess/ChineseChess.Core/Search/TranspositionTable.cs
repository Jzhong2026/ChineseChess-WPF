using ChineseChess.Models;

namespace ChineseChess.Search;

/// <summary>
/// Transposition storage only (SRP): probing with depth/flag bounds and
/// depth-preferred replacement. Counters live in <see cref="SearchState"/>.
/// </summary>
public sealed class TranspositionTable
{
    public const byte Exact = 0;
    public const byte Lower = 1;   // score >= beta (fail-high)
    public const byte Upper = 2;   // score <= alpha (fail-low)

    private readonly Dictionary<ulong, Entry> _entries = new();
    private readonly int _capacity;

    public TranspositionTable(int capacity = 250_000) => _capacity = capacity;

    public sealed record Entry(int Depth, double Score, byte Flag, Move? BestMove);

    public void Clear() => _entries.Clear();

    private const double MateThreshold = AlphaBetaSearch.MateScore - 10_000;

    private static double ToStoredScore(double score, int ply) => score >= MateThreshold ? score + ply
        : score <= -MateThreshold ? score - ply : score;

    private static double FromStoredScore(double score, int ply) => score >= MateThreshold ? score - ply
        : score <= -MateThreshold ? score + ply : score;

    /// <summary>
    /// Returns the stored entry (if any) and whether its score can be used as a
    /// cutoff at the requested depth/window.
    /// </summary>
    public bool TryProbe(ulong hash, int depth, double alpha, double beta, out Entry? entry, out bool cutoff, int depthFromRoot = 0)
    {
        cutoff = false;
        if (!_entries.TryGetValue(hash, out var found))
        {
            entry = null;
            return false;
        }
        // Mate distances are stored relative to this position, not a previous
        // search root. Restore them before testing alpha/beta bounds.
        found = found with { Score = FromStoredScore(found.Score, depthFromRoot) };
        entry = found;
        if (found.Depth < depth) return true;
        cutoff = found.Flag switch
        {
            Exact => true,
            Lower => found.Score >= beta,
            Upper => found.Score <= alpha,
            _ => false,
        };
        return true;
    }

    public void Store(ulong hash, Entry entry, int depthFromRoot = 0)
    {
        entry = entry with { Score = ToStoredScore(entry.Score, depthFromRoot) };
        if (_entries.Count >= _capacity) _entries.Clear();
        if (!_entries.TryGetValue(hash, out var current) || entry.Depth >= current.Depth)
            _entries[hash] = entry;
    }
}
