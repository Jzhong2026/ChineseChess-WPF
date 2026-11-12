using ChineseChess.Geometry;
using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

/// <summary>
/// A9 — the "correctly positioned" king-safety bonus must use the squares an
/// advisor/elephant can ACTUALLY occupy. Before the fix IsElephantSlot demanded
/// col in {2,6}, which awards (7,2)/(7,6) — unreachable for an elephant — and
/// excludes the key central square (7,4). IsAdvisorSlot missed (8,4).
/// A10 — horse mobility must not count a landing square occupied by its own piece.
/// Expected sets below are derived from the movement rules, not from the code.
/// </summary>
public class EvaluationGeometryTests
{
    private static readonly XiangqiEngine Engine = new();

    // Elephant starts at (9,2)/(9,6) for Red and moves 2 diagonally without
    // crossing the river, giving exactly these seven squares.
    private static readonly (int r, int c)[] RedElephantSquares =
        { (9, 2), (9, 6), (7, 0), (7, 4), (7, 8), (5, 2), (5, 6) };
    private static readonly (int r, int c)[] BlackElephantSquares =
        { (0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6) };

    // Advisor stays inside the palace and moves 1 diagonally: five squares.
    private static readonly (int r, int c)[] RedAdvisorSquares =
        { (9, 3), (9, 5), (8, 4), (7, 3), (7, 5) };
    private static readonly (int r, int c)[] BlackAdvisorSquares =
        { (0, 3), (0, 5), (1, 4), (2, 3), (2, 5) };

    [Fact]
    public void ElephantSlotsAreExactlyTheReachableSquares()
    {
        foreach (var side in new[] { Side.Red, Side.Black })
        {
            var expected = side == Side.Red ? RedElephantSquares : BlackElephantSquares;
            var actual = new List<(int, int)>();
            for (var r = 0; r < 10; r++)
                for (var c = 0; c < 9; c++)
                    if (XiangqiGeometry.IsElephantSlot(side, r, c)) actual.Add((r, c));

            Assert.Equal(expected.OrderBy(x => x).ToList(), actual.OrderBy(x => x).ToList());
        }
    }

    [Fact]
    public void AdvisorSlotsAreExactlyTheReachableSquares()
    {
        foreach (var side in new[] { Side.Red, Side.Black })
        {
            var expected = side == Side.Red ? RedAdvisorSquares : BlackAdvisorSquares;
            var actual = new List<(int, int)>();
            for (var r = 0; r < 10; r++)
                for (var c = 0; c < 9; c++)
                    if (XiangqiGeometry.IsAdvisorSlot(side, r, c)) actual.Add((r, c));

            Assert.Equal(expected.OrderBy(x => x).ToList(), actual.OrderBy(x => x).ToList());
        }
    }

    [Fact]
    public void CentralElephantAndAdvisorSquaresCountAsInPlace()
    {
        Assert.True(XiangqiGeometry.IsElephantSlot(Side.Red, 7, 4));
        Assert.True(XiangqiGeometry.IsElephantSlot(Side.Black, 2, 4));
        Assert.True(XiangqiGeometry.IsAdvisorSlot(Side.Red, 8, 4));
        Assert.True(XiangqiGeometry.IsAdvisorSlot(Side.Black, 1, 4));
    }

    [Fact]
    public void HorseMobilityExcludesSquaresOccupiedByOwnPieces()
    {
        var board = Engine.EmptyBoard();
        var horse = new Piece("red-horse-0", Side.Red, PieceType.Horse);
        board[5, 4] = horse;
        // From (5,4) all eight jumps are on-board and all legs are empty.
        Assert.Equal(8, XiangqiGeometry.CountHorseMoves(board, 5, 4, horse));

        // Block one landing square with a OWN piece: mobility must drop to 7.
        board[3, 3] = new Piece("red-soldier-0", Side.Red, PieceType.Soldier);
        Assert.Equal(7, XiangqiGeometry.CountHorseMoves(board, 5, 4, horse));

        // An ENEMY piece on a landing square is a capture and still counts.
        board[3, 3] = new Piece("black-soldier-0", Side.Black, PieceType.Soldier);
        Assert.Equal(8, XiangqiGeometry.CountHorseMoves(board, 5, 4, horse));
    }

    [Fact]
    public void HorseMobilityRespectsTheLeg()
    {
        var board = Engine.EmptyBoard();
        var horse = new Piece("red-horse-0", Side.Red, PieceType.Horse);
        board[5, 4] = horse;
        // Leg (4,4) blocks BOTH (3,3) and (3,5).
        board[4, 4] = new Piece("red-soldier-0", Side.Red, PieceType.Soldier);
        Assert.Equal(6, XiangqiGeometry.CountHorseMoves(board, 5, 4, horse));
    }
}