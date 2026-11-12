using ChineseChess.Models;

namespace ChineseChess.Evaluation;

/// <summary>
/// Static position scoring, independent of any search algorithm (SRP). The
/// score is from <c>perspective</c>'s point of view; <c>aiSide</c> only selects
/// which side the check-tempo bonuses are attributed to.
/// </summary>
public interface IPositionEvaluator
{
    double Evaluate(Piece?[,] board, Side perspective, Side aiSide);
}