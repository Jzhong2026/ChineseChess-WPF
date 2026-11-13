using System.Diagnostics;
using ChineseChess.Models;
using ChineseChess.Services;

namespace ChineseChess.Ai.Neural;

/// <summary>
/// Bounded capture/evasion extension for a neural leaf. This supplies a tactical
/// estimate, never a proof, and never calls the classic opponent's move selector.
/// </summary>
internal sealed class MaterialQuiescence(XiangqiEngine engine)
{
    internal static int Material(Piece?[,] board, Side side)
    {
        var score = 0;
        foreach (var piece in board)
            if (piece is not null) score += (piece.Side == side ? 1 : -1) * PieceValue(piece.Type);
        return score;
    }

    private static int PieceValue(PieceType type) => type switch
    {
        PieceType.Rook => 900, PieceType.Cannon => 450, PieceType.Horse => 400,
        PieceType.Advisor or PieceType.Elephant => 200, PieceType.Soldier => 120, _ => 0
    };

    internal float Evaluate(Piece?[,] board, Side side, IReadOnlyList<Move> history, int depth,
        Stopwatch watch, int deadlineMs, IReadOnlyList<Move>? rootLegal = null)
    {
        var remaining = 48;
        return Search(board, side, history, depth, -30000, 30000, watch, deadlineMs, ref remaining, rootLegal);
    }

    private float Search(Piece?[,] board, Side side, IReadOnlyList<Move> history, int depth,
        float alpha, float beta, Stopwatch watch, int deadlineMs, ref int remaining, IReadOnlyList<Move>? legal = null)
    {
        var standPat = Material(board, side);
        if (depth <= 0 || --remaining <= 0 || watch.ElapsedMilliseconds >= deadlineMs) return standPat;
        legal ??= engine.GetLegalMoves(board, side, history);
        if (legal.Count == 0) return -30000;
        var inCheck = engine.IsInCheck(board, side);
        if (!inCheck)
        {
            if (standPat >= beta) return standPat;
            alpha = Math.Max(alpha, standPat);
        }
        foreach (var move in legal.Where(m => inCheck || m.Captured is not null)
                     .OrderByDescending(m => m.Captured is null ? 0 : PieceValue(m.Captured.Type) * 10 - PieceValue(m.Piece.Type)))
        {
            if (remaining <= 0 || watch.ElapsedMilliseconds >= deadlineMs) break;
            var score = -Search(engine.ApplyMove(board, move), engine.Opposite(side), history.Append(move).ToList(),
                depth - 1, -beta, -alpha, watch, deadlineMs, ref remaining);
            if (score >= beta) return score;
            alpha = Math.Max(alpha, score);
        }
        // An interrupted evasion search is an estimate, not an artificial mate.
        return alpha == -30000 ? standPat : alpha;
    }
}

public sealed record MctsSearchOptions(int TacticalDepth = 0, float TacticalBlend = 0.5f,
    float PolicyTemperature = 1f, float CPuct = 1.5f, float? FirstPlayReduction = null,
    int RepetitionDrawCount = 0);
