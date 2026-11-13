using ChineseChess.Ai;
using ChineseChess.Ai.Neural;
using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

public class MctsSelectionTests
{
    [Fact]
    public void LowTemperatureDoesNotSelectFirstOverflowingBranch()
    {
        var engine = new XiangqiEngine();
        var board = engine.CreateInitialBoard();
        var moves = engine.GetLegalMoves(board, Side.Red);
        var root = new MctsNode(null, 0, board, Side.Red)
        {
            ChildMoves = new Move?[] { moves[0], moves[1] },
            ChildN = new[] { 100, 300 }, ChildW = new float[2]
        };
        // P(first) is about 2.9e-10. The former overflow chose it every time.
        var mostVisitedPicks = Enumerable.Range(0, 100).Count(_ =>
            MctsAiService.BuildResult(root, 0.05f, 400).BestMove == moves[1]);
        Assert.InRange(mostVisitedPicks, 99, 100);
    }

    [Fact]
    public void ZeroSimulationsStillReturnsALegalMove()
    {
        var engine = new XiangqiEngine();
        var board = engine.CreateInitialBoard();
        var move = engine.GetLegalMoves(board, Side.Red)[0];
        var root = new MctsNode(null, 0, board, Side.Red)
        {
            ChildMoves = new Move?[] { move }, ChildN = new int[1], ChildW = new float[1]
        };
        Assert.Equal(move, MctsAiService.BuildResult(root, 0, 0).BestMove);
    }

    [Fact]
    public void RequestCanChangeSimulationBudgetWithoutRecreatingEngine()
    {
        var engine = new XiangqiEngine();
        using var neural = new XiangqiNeuralAiService(TestPaths.DeployedOnnx, engine, true);
        var ai = new MctsAiEngine(neural, new MctsAiService(neural, engine), simulations: 2);
        var state = engine.CreateInitialState();
        var request = new AiMoveRequest(state.Board, state.Turn, state.History, int.MaxValue);
        Assert.Equal(2, ai.ChooseMove(request).Stats.Sims);
        Assert.Equal(5, ai.ChooseMove(request with { Simulations = 5 }).Stats.Sims);
        Assert.Equal(9, ai.ChooseMove(request with { Simulations = 9 }).Stats.Sims);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(1f)]
    public void ProvenLossCannotOutvoteASafeMove(float temperature)
    {
        var engine = new XiangqiEngine();
        var board = engine.CreateInitialBoard();
        var moves = engine.GetLegalMoves(board, Side.Red);
        var root = new MctsNode(null, 0, board, Side.Red)
        {
            ChildMoves = new Move?[] { moves[0], moves[1] },
            ChildN = new[] { 1000, 1 }, ChildW = new float[] { 999, -0.9f },
            Children = new MctsNode?[]
            {
                new(null, 0, board, Side.Black) { ProvenValue = 1 },
                new(null, 0, board, Side.Black)
            }
        };
        for (var i = 0; i < 20; i++)
            Assert.Equal(moves[1], MctsAiService.BuildResult(root, temperature, 1001).BestMove);
        root.Children[1]!.ProvenValue = -1;
        MctsAiService.UpdateProof(root);
        Assert.Equal(1, root.ProvenValue);
        Assert.Equal(1f, MctsAiService.BuildResult(root, temperature, 1001).BestQ);
    }

    [Fact]
    public void EvaluationSearchDoesNotAddRandomRootNoise()
    {
        var engine = new XiangqiEngine();
        using var neural = new XiangqiNeuralAiService(TestPaths.DeployedOnnx, engine, true);
        var mcts = new MctsAiService(neural, engine);
        var state = engine.CreateInitialState();
        var first = mcts.Search(state.Board, state.Turn, state.History, simulations: 16);
        var second = mcts.Search(state.Board, state.Turn, state.History, simulations: 16);
        Assert.Equal(first.BestMove, second.BestMove);
        Assert.Equal(first.MoveStats.Select(s => s.Visits), second.MoveStats.Select(s => s.Visits));
    }

    [Fact]
    public void TrainingDirichletNoiseHasTheRequestedConcentration()
    {
        // E[sum(p_i^2)] = (alpha+1)/(K*alpha+1) = 0.325.
        // Omitting the alpha<1 gamma correction gives about 0.164 instead.
        var secondMoment = Enumerable.Range(0, 2000).Average(_ =>
        {
            var sample = MctsAiService.SampleDirichlet(0.3f, 10);
            Assert.InRange(sample.Sum(), 0.9999f, 1.0001f);
            return sample.Sum(p => p * p);
        });
        Assert.InRange(secondMoment, 0.28, 0.37);
    }

    [Fact]
    public void HistoryCountingIncludesTheInitialPositionAndRestoresCaptures()
    {
        var engine = new XiangqiEngine();
        var state = engine.CreateInitialState();
        var expected = new Dictionary<string, int>();
        for (var i = 0; i < 12; i++)
        {
            var key = MctsAiService.PositionKey(state.Board, state.Turn);
            expected[key] = expected.GetValueOrDefault(key) + 1;
            var legal = engine.GetLegalMoves(state.Board, state.Turn, state.History);
            state = engine.MakeMove(state, legal.OrderByDescending(m => m.Captured is not null).First());
        }
        var last = MctsAiService.PositionKey(state.Board, state.Turn);
        expected[last] = expected.GetValueOrDefault(last) + 1;
        var actual = MctsAiService.CountHistoryPositions(state.Board, state.Turn, state.History);
        Assert.Equal(expected.OrderBy(k => k.Key, StringComparer.Ordinal), actual.OrderBy(k => k.Key, StringComparer.Ordinal));
        Assert.Contains(state.History, m => m.Captured is not null);
    }

    [Fact]
    public void AProvenDrawBeatsAProvenLoss()
    {
        var engine = new XiangqiEngine();
        var board = engine.CreateInitialBoard();
        var moves = engine.GetLegalMoves(board, Side.Red);
        var root = new MctsNode(null, 0, board, Side.Red)
        {
            ChildMoves = new Move?[] { moves[0], moves[1] }, ChildN = new[] { 100, 1 }, ChildW = new float[2],
            Children = new MctsNode?[] { new(null, 0, board, Side.Black) { ProvenValue = 1 }, new(null, 0, board, Side.Black) { ProvenValue = 0 } }
        };
        MctsAiService.UpdateProof(root);
        Assert.Equal(0, root.ProvenValue);
        Assert.Equal(moves[1], MctsAiService.BuildResult(root, 0, 101).BestMove);
    }
}
