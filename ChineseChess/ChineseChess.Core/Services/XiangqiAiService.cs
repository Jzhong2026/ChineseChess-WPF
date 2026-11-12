using ChineseChess.Ai;
using ChineseChess.Evaluation;
using ChineseChess.Models;
using ChineseChess.Search;

namespace ChineseChess.Services;

/// <summary>
/// The classic (alpha-beta) AI engine. Implements <see cref="IAiEngine"/> and
/// composes the search / evaluator / sampler collaborators; it owns only the
/// difficulty-level policy (SRP). Deep search lives in <see cref="AlphaBetaSearch"/>.
/// </summary>
public sealed class XiangqiAiService : IAiEngine
{
    private readonly XiangqiEngine _engine;
    private readonly AlphaBetaSearch _search;
    private readonly MoveSampler _sampler;
    private readonly MoveOrderer _orderer;
    private readonly EvaluationWeights _weights;

    public XiangqiAiService(
        XiangqiEngine engine,
        EvaluationWeights? weights = null,
        AlphaBetaSearch? search = null,
        MoveSampler? sampler = null,
        MoveOrderer? orderer = null)
    {
        _engine = engine;
        _weights = weights ?? EvaluationWeights.Default;
        _sampler = sampler ?? new MoveSampler();
        _orderer = orderer ?? new MoveOrderer(_weights);
        _search = search ?? new AlphaBetaSearch(engine, new PositionEvaluator(engine, _weights), _orderer, sampler: _sampler);
    }

    public string Name => "Classic";

    public AiMoveResult ChooseMove(AiMoveRequest request)
    {
        var result = ChooseMove(request.Board, request.Side, request.Difficulty, request.History, request.TimeLimitMs, request.Selection);
        return new AiMoveResult(result.Move, result.Stats);
    }

    public AiSearchResult ChooseMove(
        Piece?[,] board, Side side, int level, IReadOnlyList<Move> history,
        int timeLimitMs, MoveSelectionOptions? selectionOptions = null)
    {
        var options = selectionOptions ?? MoveSelectionOptions.Deterministic;
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var legalMoves = _orderer.Order(board, _engine.GetLegalMoves(board, side, history), depth: 0).ToList();
        if (legalMoves.Count == 0) return new AiSearchResult(null, new SearchStats(0, 0, 0, 0));

        level = Math.Clamp(level, 1, 4);
        if (level == 1)
        {
            var move = options.EnableRandomSelection ? _sampler.PickRandom(legalMoves, options) : legalMoves[0];
            return new AiSearchResult(move, new SearchStats(1, legalMoves.Count, watch.Elapsed.TotalMilliseconds, 0));
        }
        if (level == 2)
        {
            return new AiSearchResult(PickBasicMove(board, legalMoves, side), new SearchStats(2, legalMoves.Count, watch.Elapsed.TotalMilliseconds, 0));
        }

        var maxDepth = level == 3 ? 4 : 8;
        return _search.Search(board, side, maxDepth, history, timeLimitMs, options);
    }

    private Move PickBasicMove(Piece?[,] board, IReadOnlyList<Move> moves, Side side)
    {
        var scored = moves.Select(move =>
        {
            var next = _engine.ApplyMove(board, move);
            var captureValue = move.Captured is null ? 0 : _weights.PieceValues[move.Captured.Type];
            var givesCheck = _engine.IsInCheck(next, _engine.Opposite(side)) ? 50 : 0;
            return (Move: move, Score: captureValue * 10 + givesCheck);
        }).ToList();

        var topScore = scored.Max(item => item.Score);
        var bestMoves = scored.Where(item => item.Score == topScore).Select(item => item.Move).ToList();
        return bestMoves[_sampler.GetRandom(MoveSelectionOptions.Deterministic).Next(bestMoves.Count)];
    }
}