using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

/// <summary>
/// A8 — near-best sampling must see exact root scores, so the candidate pool is
/// not silently narrowed to the front of the move ordering. Guards the sampling
/// contract: legal moves, seed determinism, and real diversity across seeds.
/// </summary>
public class MoveSamplingTests
{
    private static readonly XiangqiEngine Engine = new();

    private static MoveSelectionOptions Options(int seed) => new(
        EnableRandomSelection: true, TopK: 4, NearBestWindow: 120, Seed: seed);

    [Fact]
    public void SampledMovesAreAlwaysLegal()
    {
        var ai = new XiangqiAiService(Engine);
        var state = Engine.CreateInitialState();
        var legal = Engine.GetLegalMoves(state.Board, state.Turn, state.History);

        for (var seed = 0; seed < 25; seed++)
        {
            var result = ai.ChooseMove(state.Board, state.Turn, 3, state.History, 400, Options(seed));
            Assert.NotNull(result.Move);
            Assert.Contains(legal, m => m.From == result.Move!.From && m.To == result.Move.To);
            // A full-window root search must yield finite scores; an inverted
            // window leaks +/-Infinity into the JSONL that SelfPlay writes.
            Assert.True(double.IsFinite(result.Stats.BestScore),
                $"seed {seed} produced non-finite score {result.Stats.BestScore}");
        }
    }

    [Fact]
    public void SamplingScoresAreFiniteAtEveryLevel()
    {
        var state = Engine.CreateInitialState();
        for (var level = 1; level <= 4; level++)
        {
            var ai = new XiangqiAiService(Engine);
            var result = ai.ChooseMove(state.Board, state.Turn, level, state.History, 300, Options(3));
            Assert.True(double.IsFinite(result.Stats.BestScore),
                $"level {level} produced non-finite score {result.Stats.BestScore}");
        }
    }

    [Fact]
    public void SameSeedOnAFreshServiceIsReproducible()
    {
        // The seeded Random is cached per service and advances across calls, so
        // reproducibility is defined per fresh service instance, not per call.
        var state = Engine.CreateInitialState();
        var a = new XiangqiAiService(Engine).ChooseMove(state.Board, state.Turn, 3, state.History, 400, Options(7));
        var b = new XiangqiAiService(Engine).ChooseMove(state.Board, state.Turn, 3, state.History, 400, Options(7));
        Assert.Equal(a.Move!.From, b.Move!.From);
        Assert.Equal(a.Move.To, b.Move.To);
    }

    [Fact]
    public void SamplingProducesMoreThanOneDistinctOpening()
    {
        var ai = new XiangqiAiService(Engine);
        var state = Engine.CreateInitialState();
        var seen = new HashSet<(Position, Position)>();

        for (var seed = 0; seed < 25; seed++)
        {
            var result = ai.ChooseMove(state.Board, state.Turn, 3, state.History, 400, Options(seed));
            seen.Add((result.Move!.From, result.Move.To));
        }

        Assert.True(seen.Count >= 3, $"expected a diverse candidate pool, got {seen.Count} distinct moves");
    }

    [Fact]
    public void DeterministicModeIgnoresSamplingOptions()
    {
        var ai = new XiangqiAiService(Engine);
        var state = Engine.CreateInitialState();
        var deterministic = ai.ChooseMove(state.Board, state.Turn, 3, state.History, 400, MoveSelectionOptions.Deterministic);
        Assert.NotNull(deterministic.Move);
        Assert.True(double.IsFinite(deterministic.Stats.BestScore));
    }
}