using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

/// <summary>
/// A4 — Negamax used to call GetGameStatus (which generates every legal move and
/// throws the list away) and then GetLegalMoves again on the same position.
/// Terminal detection now reuses the single generated list, so these tests guard
/// the mate/stalemate path that the change touches.
/// </summary>
public class ClassicSearchTacticsTests
{
    private static readonly XiangqiEngine Engine = new();

    /// <summary>
    /// Roughly balanced position with a mate-in-1: R(5,0)-(0,0) seals row 0 while
    /// R(1,8) seals row 1. Same fixture as the MCTS test, hand-verified.
    /// </summary>
    private static Piece?[,] MateInOneBoard()
    {
        var board = Engine.EmptyBoard();
        void Put(int r, int c, Side s, PieceType t, int i) =>
            board[r, c] = new Piece($"{s.ToString().ToLowerInvariant()}-{t.ToString().ToLowerInvariant()}-{i}", s, t);

        Put(0, 4, Side.Black, PieceType.General, 0);
        Put(2, 3, Side.Black, PieceType.Advisor, 0);
        Put(2, 5, Side.Black, PieceType.Advisor, 1);
        Put(4, 2, Side.Black, PieceType.Elephant, 0);
        Put(4, 6, Side.Black, PieceType.Elephant, 1);
        Put(5, 8, Side.Black, PieceType.Rook, 0);
        Put(2, 7, Side.Black, PieceType.Cannon, 0);
        Put(4, 4, Side.Black, PieceType.Cannon, 1);
        foreach (var c in new[] { 2, 4, 6, 8 }) Put(3, c, Side.Black, PieceType.Soldier, c);

        Put(9, 3, Side.Red, PieceType.General, 0);
        Put(1, 8, Side.Red, PieceType.Rook, 0);
        Put(5, 0, Side.Red, PieceType.Rook, 1);
        Put(7, 1, Side.Red, PieceType.Cannon, 0);
        Put(7, 7, Side.Red, PieceType.Cannon, 1);
        Put(9, 1, Side.Red, PieceType.Horse, 0);
        Put(9, 7, Side.Red, PieceType.Horse, 1);
        return board;
    }

    [Fact]
    public void LevelFourFindsMateInOne()
    {
        var ai = new XiangqiAiService(Engine);
        var board = MateInOneBoard();

        var result = ai.ChooseMove(board, Side.Red, 4, Array.Empty<Move>(), 3000);

        Assert.NotNull(result.Move);
        Assert.Equal(new Position(5, 0), result.Move!.From);
        Assert.Equal(new Position(0, 0), result.Move.To);
    }

    [Fact]
    public void SearchReturnsLegalMovesWithFiniteScoresForBothSides()
    {
        // Black is not mated here and can even capture the attacking rook via
        // R(5,8)x(5,0), so no mate is asserted for Black. What must hold is that
        // both sides get a legal move and a finite score (no -Infinity leak from
        // the terminal path).
        var ai = new XiangqiAiService(Engine);
        var board = MateInOneBoard();

        foreach (var side in new[] { Side.Red, Side.Black })
        {
            var result = ai.ChooseMove(board, side, 4, Array.Empty<Move>(), 1500);
            Assert.NotNull(result.Move);
            Assert.Contains(Engine.GetLegalMoves(board, side),
                m => m.From == result.Move!.From && m.To == result.Move.To);
            Assert.True(double.IsFinite(result.Stats.BestScore),
                $"{side} produced a non-finite score: {result.Stats.BestScore}");
        }
    }

    [Fact]
    public void StalemateIsALossForTheSideToMove()
    {
        // Black general (0,4) is NOT in check but has no legal move:
        //  - Red horse (2,4) covers (0,3) and (0,5) (leg (1,4) is empty)
        //  - Red rook (1,8) covers (1,4) along row 1
        // Xiangqi scores stalemate as a loss for the side to move.
        var board = Engine.EmptyBoard();
        board[0, 4] = new Piece("black-general-0", Side.Black, PieceType.General);
        board[9, 3] = new Piece("red-general-0", Side.Red, PieceType.General);
        board[1, 8] = new Piece("red-rook-0", Side.Red, PieceType.Rook);
        board[2, 4] = new Piece("red-horse-0", Side.Red, PieceType.Horse);

        Assert.False(Engine.IsInCheck(board, Side.Black));
        Assert.Empty(Engine.GetLegalMoves(board, Side.Black));
        Assert.Equal(GameStatus.RedWins, Engine.GetGameStatus(board, Side.Black));

        var ai = new XiangqiAiService(Engine);
        var result = ai.ChooseMove(board, Side.Black, 4, Array.Empty<Move>(), 500);
        Assert.Null(result.Move);
    }
}