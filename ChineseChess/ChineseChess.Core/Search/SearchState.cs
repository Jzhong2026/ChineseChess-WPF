using System.Diagnostics;

namespace ChineseChess.Search;

/// <summary>
/// Mutable per-search bookkeeping (clock, node count, abort flag, TT counters).
/// A reference type so recursive search passes one object instead of a long
/// list of ref parameters.
/// </summary>
public sealed class SearchState
{
    public SearchState(Stopwatch watch, long deadline)
    {
        Watch = watch;
        Deadline = deadline;
    }

    public Stopwatch Watch { get; }
    public long Deadline { get; }
    public int Nodes;
    public bool Aborted;
    public int TtHits;
    public int TtStores;
    public int TtBestMoveHits;
    public int TtScoreHits;

    public bool TimeUp => Watch.ElapsedMilliseconds >= Deadline;

    /// <summary>Polls the clock every 64 nodes to keep overhead low.</summary>
    public void PollTime()
    {
        if ((Nodes & 0x3f) == 0 && TimeUp) Aborted = true;
    }
}