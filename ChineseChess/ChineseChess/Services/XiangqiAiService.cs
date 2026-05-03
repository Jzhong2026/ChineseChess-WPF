using System.Diagnostics;
using ChineseChess.Models;

namespace ChineseChess.Services;

public sealed class XiangqiAiService
{
    private readonly XiangqiEngine _engine;
    private readonly Random _random = new();

    private static readonly IReadOnlyDictionary<PieceType, int> PieceValues = new Dictionary<PieceType, int>
    {
        [PieceType.General] = 10000,
        [PieceType.Rook] = 900,
        [PieceType.Cannon] = 450,
        [PieceType.Horse] = 400,
        [PieceType.Elephant] = 200,
        [PieceType.Advisor] = 200,
        [PieceType.Soldier] = 120
    };

    public XiangqiAiService(XiangqiEngine engine)
    {
        _engine = engine;
    }

    public AiSearchResult ChooseMove(Piece?[,] board, Side side, int level, IReadOnlyList<Move> history, int timeLimitMs)
    {
        var watch = Stopwatch.StartNew();
        var legalMoves = OrderMoves(board, _engine.GetLegalMoves(board, side, history)).ToList();
        if (legalMoves.Count == 0)
        {
            return new AiSearchResult(null, new SearchStats(0, 0, 0, 0));
        }

        level = Math.Clamp(level, 1, 4);
        if (level == 1)
        {
            return new AiSearchResult(legalMoves[_random.Next(legalMoves.Count)], new SearchStats(1, legalMoves.Count, watch.Elapsed.TotalMilliseconds, 0));
        }

        if (level == 2)
        {
            return new AiSearchResult(PickBasicMove(board, legalMoves, side), new SearchStats(2, legalMoves.Count, watch.Elapsed.TotalMilliseconds, 0));
        }

        var maxDepth = level == 3 ? 4 : 6;
        var deadline = watch.ElapsedMilliseconds + Math.Max(30, timeLimitMs);
        var nodes = 0;
        var bestMove = legalMoves[0];
        var bestScore = EvaluateBoard(_engine.ApplyMove(board, bestMove), side, side);
        var depthReached = 0;

        for (var depth = 1; depth <= maxDepth; depth++)
        {
            if (depth > 1 && watch.ElapsedMilliseconds >= deadline - 10) break;
            var aborted = false;
            var iterBestMove = legalMoves[0];
            var iterBestScore = double.NegativeInfinity;
            var alpha = double.NegativeInfinity;
            var beta = double.PositiveInfinity;

            foreach (var move in legalMoves)
            {
                if (watch.ElapsedMilliseconds >= deadline)
                {
                    aborted = true;
                    break;
                }

                nodes++;
                var score = -Negamax(
                    _engine.ApplyMove(board, move),
                    _engine.Opposite(side),
                    depth - 1,
                    -beta,
                    -alpha,
                    _engine.Opposite(side),
                    side,
                    history.Concat(new[] { move }).ToList(),
                    watch,
                    deadline,
                    ref nodes,
                    ref aborted);

                if (!aborted && score > iterBestScore)
                {
                    iterBestScore = score;
                    iterBestMove = move;
                }

                if (!aborted) alpha = Math.Max(alpha, score);
            }

            if (!aborted)
            {
                bestMove = iterBestMove;
                bestScore = iterBestScore;
                depthReached = depth;
            }
        }

        return new AiSearchResult(bestMove, new SearchStats(Math.Max(depthReached, 0), nodes, watch.Elapsed.TotalMilliseconds, bestScore));
    }

    private double Negamax(
        Piece?[,] board,
        Side turn,
        int depth,
        double alpha,
        double beta,
        Side perspective,
        Side aiSide,
        IReadOnlyList<Move> history,
        Stopwatch watch,
        long deadline,
        ref int nodes,
        ref bool aborted)
    {
        if ((nodes & 0x3f) == 0 && watch.ElapsedMilliseconds >= deadline) aborted = true;
        if (aborted) return 0;

        var status = _engine.GetGameStatus(board, turn);
        if (status == GameStatus.RedWins) return perspective == Side.Red ? 999999 : -999999;
        if (status == GameStatus.BlackWins) return perspective == Side.Black ? 999999 : -999999;
        if (status == GameStatus.Draw) return 0;
        if (depth == 0) return Quiescence(board, turn, alpha, beta, perspective, aiSide, history, watch, deadline, ref nodes, ref aborted, 0);

        var best = double.NegativeInfinity;
        var moves = OrderMoves(board, _engine.GetLegalMoves(board, turn, history));
        foreach (var move in moves)
        {
            if (aborted) break;
            nodes++;
            var score = -Negamax(
                _engine.ApplyMove(board, move),
                _engine.Opposite(turn),
                depth - 1,
                -beta,
                -alpha,
                _engine.Opposite(perspective),
                aiSide,
                history.Concat(new[] { move }).ToList(),
                watch,
                deadline,
                ref nodes,
                ref aborted);

            best = Math.Max(best, score);
            alpha = Math.Max(alpha, score);
            if (alpha >= beta) break;
        }

        return best;
    }

    private double Quiescence(
        Piece?[,] board,
        Side turn,
        double alpha,
        double beta,
        Side perspective,
        Side aiSide,
        IReadOnlyList<Move> history,
        Stopwatch watch,
        long deadline,
        ref int nodes,
        ref bool aborted,
        int depth)
    {
        var standPat = EvaluateBoard(board, perspective, aiSide);
        if (standPat >= beta) return beta;
        if (alpha < standPat) alpha = standPat;
        if (depth >= 6) return alpha;

        var tacticalMoves = OrderMoves(board, _engine.GetLegalMoves(board, turn, history).Where(move => move.Captured is not null));
        foreach (var move in tacticalMoves)
        {
            if ((nodes & 0x3f) == 0 && watch.ElapsedMilliseconds >= deadline) aborted = true;
            if (aborted) break;
            nodes++;
            var score = -Quiescence(
                _engine.ApplyMove(board, move),
                _engine.Opposite(turn),
                -beta,
                -alpha,
                _engine.Opposite(perspective),
                aiSide,
                history.Concat(new[] { move }).ToList(),
                watch,
                deadline,
                ref nodes,
                ref aborted,
                depth + 1);

            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }

        return alpha;
    }

    private Move PickBasicMove(Piece?[,] board, IReadOnlyList<Move> moves, Side side)
    {
        var scoredMoves = moves.Select(move =>
        {
            var next = _engine.ApplyMove(board, move);
            var captureValue = move.Captured is null ? 0 : PieceValues[move.Captured.Type];
            var givesCheck = _engine.IsInCheck(next, _engine.Opposite(side)) ? 50 : 0;
            var leavesSelfInCheck = _engine.IsInCheck(next, side) ? -5000 : 0;
            return (Move: move, Score: captureValue * 10 + givesCheck + leavesSelfInCheck);
        }).ToList();

        var topScore = scoredMoves.Max(item => item.Score);
        var bestMoves = scoredMoves.Where(item => item.Score == topScore).Select(item => item.Move).ToList();
        return bestMoves[_random.Next(bestMoves.Count)];
    }

    private IEnumerable<Move> OrderMoves(Piece?[,] board, IEnumerable<Move> moves)
    {
        return moves.OrderByDescending(move =>
        {
            var captured = board[move.To.Row, move.To.Col];
            return captured is null ? 0 : PieceValues[captured.Type] - PieceValues[move.Piece.Type] / 20.0;
        });
    }

    private double EvaluateBoard(Piece?[,] board, Side perspective, Side aiSide)
    {
        var score = 0.0;
        for (var row = 0; row < XiangqiEngine.BoardRows; row++)
        {
            for (var col = 0; col < XiangqiEngine.BoardCols; col++)
            {
                var piece = board[row, col];
                if (piece is null) continue;
                var sideSign = piece.Side == perspective ? 1 : -1;
                score += sideSign * (PieceValues[piece.Type] + PositionalBonus(piece.Type, piece.Side, row, col));
            }
        }

        if (_engine.IsInCheck(board, _engine.Opposite(aiSide))) score += perspective == aiSide ? 35 : -35;
        if (_engine.IsInCheck(board, aiSide)) score += perspective == aiSide ? -45 : 45;
        return score;
    }

    private static double PositionalBonus(PieceType type, Side side, int row, int col)
    {
        var forwardProgress = side == Side.Red ? 9 - row : row;
        var centerDistance = Math.Abs(col - 4);
        return type switch
        {
            PieceType.Soldier => forwardProgress * 14 - centerDistance * 3,
            PieceType.Horse => 25 - centerDistance * 4 - Math.Abs(row - 4.5) * 2,
            PieceType.Cannon => 16 - centerDistance * 2,
            PieceType.Rook => 10 - centerDistance,
            _ => 0
        };
    }
}
