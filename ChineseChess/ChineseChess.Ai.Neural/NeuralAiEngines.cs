using System.Diagnostics;
using ChineseChess.Ai;
using ChineseChess.Models;

namespace ChineseChess.Ai.Neural;

/// <summary>Pure neural-policy engine (no tree search) behind the IAiEngine seam.</summary>
public sealed class NeuralPolicyAiEngine : IAiEngine, IDisposable
{
    private readonly XiangqiNeuralAiService _neural;

    public NeuralPolicyAiEngine(XiangqiNeuralAiService neural) => _neural = neural;

    public string Name => "NN Policy";

    public AiMoveResult ChooseMove(AiMoveRequest request)
    {
        var watch = Stopwatch.StartNew();
        var move = _neural.ChooseMoveByPolicy(request.Board, request.Side, request.History);
        var value = _neural.EvaluatePosition(request.Board, request.Side);
        return new AiMoveResult(move, new SearchStats(0, 0, watch.Elapsed.TotalMilliseconds, value, Q: value));
    }

    public void Dispose() => _neural.Dispose();
}

/// <summary>MCTS + neural policy/value engine behind the IAiEngine seam.</summary>
public sealed class MctsAiEngine : IAiEngine, IDisposable
{
    private readonly MctsAiService _mcts;
    private readonly XiangqiNeuralAiService _neural;
    private readonly int _simulations;

    public MctsAiEngine(XiangqiNeuralAiService neural, MctsAiService mcts, int simulations = 3000)
    {
        _neural = neural;
        _mcts = mcts;
        _simulations = simulations;
    }

    public string Name => "NN+MCTS";

    public AiMoveResult ChooseMove(AiMoveRequest request)
    {
        var watch = Stopwatch.StartNew();
        var result = _mcts.Search(request.Board, request.Side, request.History,
            simulations: _simulations, temperature: request.Temperature, timeLimitMs: request.TimeLimitMs);
        return new AiMoveResult(result.BestMove, new SearchStats(
            0, result.SimulationsRun, watch.Elapsed.TotalMilliseconds, result.BestQ,
            Sims: result.SimulationsRun, Q: result.BestQ), result.MoveStats);
    }

    public void Dispose() => _neural.Dispose();
}