using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

/// <summary>
/// Hand-verified check geometry. Every expected value below is reasoned from the
/// rules of xiangqi, not computed from the implementation — this is the oracle
/// that makes an allocation-free IsInCheck safe to write.
/// Target square is the RED general; we ask "is Red in check?".
/// </summary>
public class EngineCheckDetectionTests
{
    private static readonly XiangqiEngine Engine = new();

    private static Piece?[,] Board(int gr, int gc, params (int r, int c, Side s, PieceType t)[] pieces)
    {
        var b = Engine.EmptyBoard();
        b[gr, gc] = new Piece("red-general-0", Side.Red, PieceType.General);
        var i = 0;
        foreach (var (r, c, s, t) in pieces)
            b[r, c] = new Piece($"{s}-{t}-{i++}", s, t);
        // Keep the black general somewhere harmless so positions stay well-formed.
        if (!pieces.Any(p => p.s == Side.Black && p.t == PieceType.General))
            b[0, 0] = new Piece("black-general-0", Side.Black, PieceType.General);
        return b;
    }

    [Theory]
    // ── Rook: open line attacks, any intervening piece blocks ──
    [InlineData(9, 4, 5, 4, PieceType.Rook, true)]    // same file, nothing between
    [InlineData(9, 4, 9, 0, PieceType.Rook, true)]    // same rank, nothing between
    [InlineData(9, 4, 0, 4, PieceType.Rook, true)]    // long open file
    // ── Cannon: needs EXACTLY one screen ──
    [InlineData(9, 4, 5, 4, PieceType.Cannon, false)] // zero screens -> no check
    // ── Horse: all eight jumps, leg must be free ──
    [InlineData(9, 4, 7, 3, PieceType.Horse, true)]   // (-2,-1) leg (8,4) free
    [InlineData(9, 4, 7, 5, PieceType.Horse, true)]   // (-2,+1) leg (8,4) free
    [InlineData(9, 4, 8, 2, PieceType.Horse, true)]   // (-1,-2) leg (9,3) free
    [InlineData(9, 4, 8, 6, PieceType.Horse, true)]   // (-1,+2) leg (9,5) free
    // ── Soldier: forward only before the river, sideways after ──
    [InlineData(9, 4, 8, 4, PieceType.Soldier, true)]  // black soldier steps down to (9,4)
    [InlineData(9, 4, 9, 3, PieceType.Soldier, true)]  // black soldier on row 9 has crossed -> sideways
    [InlineData(9, 4, 9, 5, PieceType.Soldier, true)]
    public void CheckGeometry(int gr, int gc, int pr, int pc, PieceType type, bool expected)
    {
        if (pr == gr && pc == gc) return; // skip placeholder row
        var board = Board(gr, gc, (pr, pc, Side.Black, type));
        Assert.Equal(expected, Engine.IsInCheck(board, Side.Red));
    }

    [Fact]
    public void RookIsBlockedByAnInterveningPiece()
    {
        var board = Board(9, 4, (5, 4, Side.Black, PieceType.Rook), (7, 4, Side.Red, PieceType.Soldier));
        Assert.False(Engine.IsInCheck(board, Side.Red));
    }

    [Fact]
    public void CannonChecksWithExactlyOneScreen()
    {
        var oneScreen = Board(9, 4, (5, 4, Side.Black, PieceType.Cannon), (7, 4, Side.Red, PieceType.Soldier));
        Assert.True(Engine.IsInCheck(oneScreen, Side.Red));

        var twoScreens = Board(9, 4,
            (5, 4, Side.Black, PieceType.Cannon), (7, 4, Side.Red, PieceType.Soldier), (6, 4, Side.Black, PieceType.Soldier));
        Assert.False(Engine.IsInCheck(twoScreens, Side.Red));

        var noScreen = Board(9, 4, (5, 4, Side.Black, PieceType.Cannon));
        Assert.False(Engine.IsInCheck(noScreen, Side.Red));
    }

    [Fact]
    public void HorseIsBlockedByItsLeg()
    {
        // Horse (7,3) jumps to (9,4); leg is (8,3).
        var free = Board(9, 4, (7, 3, Side.Black, PieceType.Horse));
        Assert.True(Engine.IsInCheck(free, Side.Red));

        var blocked = Board(9, 4, (7, 3, Side.Black, PieceType.Horse), (8, 3, Side.Red, PieceType.Soldier));
        Assert.False(Engine.IsInCheck(blocked, Side.Red));

        // Horse (8,2) jumps to (9,4); leg is (8,3) — same leg square, different jump.
        var free2 = Board(9, 4, (8, 2, Side.Black, PieceType.Horse));
        Assert.True(Engine.IsInCheck(free2, Side.Red));
        var blocked2 = Board(9, 4, (8, 2, Side.Black, PieceType.Horse), (8, 3, Side.Black, PieceType.Soldier));
        Assert.False(Engine.IsInCheck(blocked2, Side.Red));
    }

    [Fact]
    public void SoldierCannotMoveSidewaysBeforeCrossingTheRiver()
    {
        // Black soldier on row 4 has NOT crossed (black crosses at row >= 5).
        var board = Board(9, 4, (4, 3, Side.Black, PieceType.Soldier));
        Assert.False(Engine.IsInCheck(board, Side.Red));
    }

    [Fact]
    public void FlyingGeneralsCheckAlongAnOpenFile()
    {
        var open = Engine.EmptyBoard();
        open[9, 4] = new Piece("red-general-0", Side.Red, PieceType.General);
        open[0, 4] = new Piece("black-general-0", Side.Black, PieceType.General);
        Assert.True(Engine.IsInCheck(open, Side.Red));
        Assert.True(Engine.IsInCheck(open, Side.Black));

        var blocked = Engine.CloneBoard(open);
        blocked[5, 4] = new Piece("red-soldier-0", Side.Red, PieceType.Soldier);
        Assert.False(Engine.IsInCheck(blocked, Side.Red));
    }

    [Fact]
    public void AdvisorsAndElephantsCanNeverCheckTheEnemyGeneral()
    {
        // Neither piece can leave its own half/palace, so they cannot attack (9,4).
        var advisor = Board(9, 4, (2, 4, Side.Black, PieceType.Advisor));
        Assert.False(Engine.IsInCheck(advisor, Side.Red));
        var elephant = Board(9, 4, (4, 4, Side.Black, PieceType.Elephant));
        Assert.False(Engine.IsInCheck(elephant, Side.Red));
    }

    [Fact]
    public void OwnPiecesNeverGiveCheck()
    {
        var board = Board(9, 4, (5, 4, Side.Red, PieceType.Rook), (7, 3, Side.Red, PieceType.Horse));
        Assert.False(Engine.IsInCheck(board, Side.Red));
    }

    [Fact]
    public void MissingGeneralCountsAsInCheck()
    {
        var board = Engine.EmptyBoard();
        board[0, 0] = new Piece("black-general-0", Side.Black, PieceType.General);
        Assert.True(Engine.IsInCheck(board, Side.Red));
    }
}