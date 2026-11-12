using ChineseChess.Geometry;
using ChineseChess.Models;
using ChineseChess.Services;

namespace ChineseChess.Evaluation;

public sealed class PositionEvaluator : IPositionEvaluator
{
    private readonly XiangqiEngine _engine;
    private readonly EvaluationWeights _weights;

    public PositionEvaluator(XiangqiEngine engine, EvaluationWeights? weights = null)
    {
        _engine = engine;
        _weights = weights ?? EvaluationWeights.Default;
    }

    public double Evaluate(Piece?[,] board, Side perspective, Side aiSide)
    {
        var material = 0.0;
        var positional = 0.0;
        var mobility = 0.0;
        var kingSafety = 0.0;
        var pawnStructure = 0.0;
        var myKingRow = -1;
        var myKingCol = -1;
        var myAdvisorCount = 0;
        var myElephantCount = 0;
        var myAdvisorInPlace = 0;
        var myElephantInPlace = 0;

        for (var row = 0; row < XiangqiGeometry.Rows; row++)
        {
            for (var col = 0; col < XiangqiGeometry.Cols; col++)
            {
                var piece = board[row, col];
                if (piece is null) continue;

                var isMine = piece.Side == perspective;
                var sideSign = isMine ? 1 : -1;

                material += sideSign * _weights.PieceValues[piece.Type];
                positional += sideSign * PositionalBonus(piece.Type, piece.Side, row, col);
                mobility += sideSign * _weights.MobilityWeight(piece.Type) * XiangqiGeometry.CountMobility(board, row, col, piece);

                if (piece.Type == PieceType.Soldier)
                    pawnStructure += sideSign * PawnStructureScore(piece.Side, row, col);

                if (!isMine) continue;
                if (piece.Type == PieceType.General)
                {
                    myKingRow = row;
                    myKingCol = col;
                }
                else if (piece.Type == PieceType.Advisor)
                {
                    myAdvisorCount++;
                    if (XiangqiGeometry.IsAdvisorSlot(piece.Side, row, col)) myAdvisorInPlace++;
                }
                else if (piece.Type == PieceType.Elephant)
                {
                    myElephantCount++;
                    if (XiangqiGeometry.IsElephantSlot(piece.Side, row, col)) myElephantInPlace++;
                }
            }
        }

        if (myKingRow >= 0)
        {
            kingSafety += (myAdvisorCount + myElephantCount) * _weights.DefenderPresence;
            kingSafety += (myAdvisorInPlace + myElephantInPlace) * _weights.DefenderInPlace;
            kingSafety -= CountAttackersNearKing(board, myKingRow, myKingCol, _engine.Opposite(perspective)) * _weights.EnemyAttackerPenalty;
        }

        if (_engine.IsInCheck(board, _engine.Opposite(aiSide))) material += perspective == aiSide ? _weights.GivingCheck : -_weights.GivingCheck;
        if (_engine.IsInCheck(board, aiSide)) material += perspective == aiSide ? -_weights.BeingChecked : _weights.BeingChecked;

        return material + positional + mobility + kingSafety + pawnStructure;
    }

    private double PawnStructureScore(Side side, int row, int col)
    {
        var forwardProgress = side == Side.Red ? 9 - row : row;
        var score = forwardProgress * _weights.PawnAdvance;
        if (XiangqiGeometry.CrossedRiver(side, row))
        {
            score += _weights.PawnCrossedRiver;
            if (forwardProgress >= _weights.PawnDeepThreshold) score += _weights.PawnDeep;
        }
        return score;
    }

    private double PositionalBonus(PieceType type, Side side, int row, int col)
    {
        var forwardProgress = side == Side.Red ? 9 - row : row;
        var centerDistance = Math.Abs(col - 4);
        return type switch
        {
            PieceType.Soldier => forwardProgress * _weights.SoldierAdvance - centerDistance * _weights.SoldierCenter,
            PieceType.Horse => _weights.HorseBase - centerDistance * _weights.HorseCenter - Math.Abs(row - 4.5) * _weights.HorseRow,
            PieceType.Cannon => _weights.CannonBase - centerDistance * _weights.CannonCenter,
            PieceType.Rook => _weights.RookBase - centerDistance * _weights.RookCenter,
            _ => 0,
        };
    }

    private static int CountAttackersNearKing(Piece?[,] board, int kingRow, int kingCol, Side enemySide)
    {
        var attackers = 0;
        for (var row = 0; row < XiangqiGeometry.Rows; row++)
        {
            for (var col = 0; col < XiangqiGeometry.Cols; col++)
            {
                var piece = board[row, col];
                if (piece is null || piece.Side != enemySide || piece.Type == PieceType.General) continue;
                if (CanReachKingArea(board, row, col, piece, kingRow, kingCol)) attackers++;
            }
        }
        return attackers;
    }

    private static bool CanReachKingArea(Piece?[,] board, int r, int c, Piece piece, int kingRow, int kingCol)
    {
        for (var dr = -1; dr <= 1; dr++)
        {
            for (var dc = -1; dc <= 1; dc++)
            {
                if (dr == 0 && dc == 0) continue;
                var tr = kingRow + dr;
                var tc = kingCol + dc;
                if (!XiangqiGeometry.InBounds(tr, tc)) continue;
                if (XiangqiGeometry.PieceAttacksSquare(board, r, c, piece, tr, tc)) return true;
            }
        }
        return false;
    }
}