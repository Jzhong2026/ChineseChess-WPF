using ChineseChess.AiMatchRunner;
using Xunit;

namespace ChineseChess.Core.Tests;

/// <summary>
/// C5 — a draw requires a position to have occurred THREE times. The previous
/// implementation counted consecutive re-visits of any already-seen position,
/// which invented draws on 3-/4-cycles and missed genuine interleaved repeats.
/// Sequences and expected trigger plies below were established by replaying the
/// old algorithm (see docs/devlog/2026-11-11.md, probe 7).
/// </summary>
public class RepetitionCounterTests
{
    private static int? TriggerPly(string[] keys)
    {
        var counter = new RepetitionCounter();
        for (var i = 0; i < keys.Length; i++)
            if (counter.Add(keys[i])) return i;
        return null;
    }

    [Fact]
    public void PureTwoCycleDrawsOnTheThirdOccurrence()
    {
        Assert.Equal(4, TriggerPly(new[] { "A", "B", "A", "B", "A", "B" }));
    }

    [Fact]
    public void ThreeCycleWithOnlyTwoOccurrencesEachIsNotADraw()
    {
        Assert.Null(TriggerPly(new[] { "A", "B", "C", "A", "B", "C" }));
    }

    [Fact]
    public void FourCycleWithOnlyTwoOccurrencesEachIsNotADraw()
    {
        Assert.Null(TriggerPly(new[] { "A", "B", "C", "D", "A", "B", "C", "D" }));
    }

    [Fact]
    public void InterleavedGenuineThreefoldIsDetected()
    {
        Assert.Equal(6, TriggerPly(new[] { "A", "X", "B", "Y", "A", "Z", "A" }));
    }

    [Fact]
    public void ThreefoldWithInterveningNewPositionsIsDetected()
    {
        Assert.Equal(4, TriggerPly(new[] { "A", "B", "A", "C", "A" }));
    }

    [Fact]
    public void TwoOccurrencesNeverDraw()
    {
        Assert.Null(TriggerPly(new[] { "A", "X", "A", "Y" }));
    }

    [Fact]
    public void MaxOccurrencesTracksTheHottestPosition()
    {
        var counter = new RepetitionCounter();
        foreach (var k in new[] { "A", "B", "A", "C", "A" }) counter.Add(k);
        Assert.Equal(3, counter.MaxOccurrences);
    }
}