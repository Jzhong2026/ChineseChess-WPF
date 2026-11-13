using System.Diagnostics;
using ChineseChess.Evaluation;
using ChineseChess.Models;
using ChineseChess.Services;

namespace ChineseChess.Search;

/// <summary>
/// Iterative-deepening negamax with alpha-beta, quiescence and a transposition
/// table. Composes evaluator / orderer / table / sampler rather than owning
/// their logic (SRP, DIP).
/// </summary>
public sealed class AlphaBetaSearch
{
    public const double MateScore = 999999;
    private const int MaxQuiescenceDepth = 8;

    private readonly XiangqiEngine _engine;
    private readonly IPositionEvaluator _evaluator;
    private readonly MoveOrderer _orderer;
    private readonly TranspositionTable _table;
    private readonly MoveSampler _sampler;

    public AlphaBetaSearch(
        XiangqiEngine engine,
        IPositionEvaluator? evaluator = null,
        MoveOrderer? orderer = null,
        TranspositionTable? table = null,
        MoveSampler? sampler = null)
    {
        _engine = engine;
        _evaluator = evaluator ?? new PositionEvaluator(engine);
        _orderer = orderer ?? new MoveOrderer();
        _table = table ?? new TranspositionTable();
        _sampler = sampler ?? new MoveSampler();
    }

    public AiSearchResult Search(
        Piece?[,] board, Side side, int maxDepth, IReadOnlyList<Move> history,
        int timeLimitMs, MoveSelectionOptions selectionOptions)
    {
        // A new root may have a different AI side and repetition history.
        _table.Clear();
        var watch = Stopwatch.StartNew();
        var legalMoves = _orderer.Order(board, _engine.GetLegalMoves(board, side, history), depth: 0).ToList();
        if (legalMoves.Count == 0) return new AiSearchResult(null, new SearchStats(0, 0, 0, 0));

        var deadline = watch.ElapsedMilliseconds + Math.Max(30, timeLimitMs);
        var state = new SearchState(watch, deadline);
        var bestMove = legalMoves[0];
        var bestMoveScore = _evaluator.Evaluate(_engine.ApplyMove(board, bestMove), side, side);
        var depthReached = 0;

        _orderer.BeginSearch();

        for (var depth = 1; depth <= maxDepth; depth++)
        {
            if (depth > 1 && watch.ElapsedMilliseconds >= deadline - 10) break;
            state.Aborted = false;
            var iterBestMove = legalMoves[0];
            var iterBestScore = double.NegativeInfinity;
            var alpha = double.NegativeInfinity;
            var beta = double.PositiveInfinity;

            var iterRootScores = new List<(Move Move, double Score)>(legalMoves.Count);
            // Sampling needs EXACT root scores; a rising alpha would make later
            // moves fail low and silently shrink the candidate pool.
            var sampling = selectionOptions.EnableRandomSelection;
            foreach (var move in legalMoves)
            {
                if (state.TimeUp) { state.Aborted = true; break; }

                state.Nodes++;
                var score = -Negamax(
                    _engine.ApplyMove(board, move), _engine.Opposite(side), depth - 1,
                    -beta, sampling ? double.PositiveInfinity : -alpha,
                    _engine.Opposite(side), side, history.Concat(new[] { move }).ToList(),
                    state, depthFromRoot: 1);

                if (!state.Aborted)
                {
                    iterRootScores.Add((move, score));
                    if (score > iterBestScore) { iterBestScore = score; iterBestMove = move; }
                }
                if (!state.Aborted && !sampling) alpha = Math.Max(alpha, score);
            }

            if (state.Aborted) continue;
            bestMove = _sampler.Select(iterRootScores, iterBestMove, iterBestScore, selectionOptions);
            bestMoveScore = iterRootScores.First(item => item.Move.From == bestMove.From && item.Move.To == bestMove.To).Score;
            depthReached = depth;
        }

        return new AiSearchResult(bestMove, new SearchStats(
            Math.Max(depthReached, 0), state.Nodes, watch.Elapsed.TotalMilliseconds, bestMoveScore,
            state.TtHits, state.TtStores, state.TtBestMoveHits, state.TtScoreHits));
    }

    internal double Negamax(
        Piece?[,] board, Side turn, int depth, double alpha, double beta,
        Side perspective, Side aiSide, IReadOnlyList<Move> history, SearchState state, int depthFromRoot)
    {
        var alphaOrig = alpha;
        state.PollTime();
        if (state.Aborted) return 0;

        var hash = ZobristHash.Compute(board, turn);
        Move? ttBestMove = null;
        if (_table.TryProbe(hash, depth, alpha, beta, out var entry, out var cutoff, depthFromRoot))
        {
            state.TtHits++;
            ttBestMove = entry!.BestMove;
            if (cutoff) { state.TtScoreHits++; return entry.Score; }
        }

        var moves = _orderer.Order(board, _engine.GetLegalMoves(board, turn, history), ttBestMove, depthFromRoot).ToList();
        if (moves.Count == 0) return -MateScore + depthFromRoot;
        if (depth <= 0) return Quiescence(board, turn, alpha, beta, perspective, aiSide, history, state, depthFromRoot, 0);

        var best = double.NegativeInfinity;
        Move? bestMove = null;
        foreach (var move in moves)
        {
            if (state.Aborted) break;
            state.Nodes++;
            var score = -Negamax(
                _engine.ApplyMove(board, move), _engine.Opposite(turn), depth - 1,
                -beta, -alpha, _engine.Opposite(perspective), aiSide,
                history.Concat(new[] { move }).ToList(), state, depthFromRoot + 1);

            if (score > best) { best = score; bestMove = move; }
            alpha = Math.Max(alpha, score);
            if (alpha >= beta)
            {
                _orderer.RecordCutoff(move, depth, depthFromRoot);
                break;
            }
        }

        if (!state.Aborted && best > double.NegativeInfinity)
        {
            var flag = best <= alphaOrig ? TranspositionTable.Upper
                : best >= beta ? TranspositionTable.Lower
                : TranspositionTable.Exact;
            _table.Store(hash, new TranspositionTable.Entry(depth, best, flag, bestMove), depthFromRoot);
            state.TtStores++;
            if (ttBestMove is not null && bestMove is not null
                && ttBestMove.From == bestMove.From && ttBestMove.To == bestMove.To) state.TtBestMoveHits++;
        }

        return best;
    }

    internal double Quiescence(
        Piece?[,] board, Side turn, double alpha, double beta,
        Side perspective, Side aiSide, IReadOnlyList<Move> history, SearchState state, int depthFromRoot, int qDepth)
    {
        state.PollTime();
        if (state.Aborted) return 0;

        var inCheck = _engine.IsInCheck(board, turn);
        var legalMoves = _engine.GetLegalMoves(board, turn, history);
        if (legalMoves.Count == 0) return -MateScore + depthFromRoot;

        if (inCheck)
        {
            // Passing is illegal in check, so the static score is not a lower
            // bound; being mated is the valid floor.
            var mateFloor = -MateScore + depthFromRoot;
            if (alpha < mateFloor) alpha = mateFloor;
        }
        else
        {
            var standPat = _evaluator.Evaluate(board, perspective, aiSide);
            if (standPat >= beta) return beta;
            if (alpha < standPat) alpha = standPat;
        }
        if (qDepth >= MaxQuiescenceDepth) return inCheck ? _evaluator.Evaluate(board, perspective, aiSide) : alpha;

        var tacticalMoves = _orderer.Order(board,
            inCheck ? legalMoves : legalMoves.Where(move => move.Captured is not null),
            depth: depthFromRoot);
        foreach (var move in tacticalMoves)
        {
            state.PollTime();
            if (state.Aborted) break;
            state.Nodes++;
            var score = -Quiescence(
                _engine.ApplyMove(board, move), _engine.Opposite(turn), -beta, -alpha,
                _engine.Opposite(perspective), aiSide, history.Concat(new[] { move }).ToList(),
                state, depthFromRoot + 1, qDepth + 1);

            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }

        return alpha;
    }
}
