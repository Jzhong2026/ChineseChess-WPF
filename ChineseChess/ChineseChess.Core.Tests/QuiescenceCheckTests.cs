using System.Diagnostics;
using ChineseChess.Models;
using ChineseChess.Search;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

/// <summary>
/// A7 — quiescence must not "stand pat" while in check: passing is not a legal
/// option, so the static evaluation is not a lower bound. Fixture: Red is
/// checkmated yet stands ~850cp ahead on material, so a stand-pat return is
/// clearly positive while the true score is a mate loss.
///   Red: G(9,4) R(2,8) R(2,6) H(0,7) C(4,6)   Black: G(0,3) R(9,0) R(8,8)
/// R(9,0) checks along row 9; (9,3)/(9,5) are covered by it and (8,4) by R(8,8).
/// No red piece can interpose on (9,1)/(9,2)/(9,3) or capture R(9,0).
/// </summary>
public class QuiescenceCheckTests
{
    private static readonly XiangqiEngine Engine = new();

    private static Piece?[,] RedMatedButAheadOnMaterial()
    {
        var b = Engine.EmptyBoard();
        void Put(int r, int c, Side s, PieceType t, int i) =>
            b[r, c] = new Piece($"{s}-{t}-{i}", s, t);

        Put(9, 4, Side.Red, PieceType.General, 0);
        Put(2, 8, Side.Red, PieceType.Rook, 0);
        Put(2, 6, Side.Red, PieceType.Rook, 1);
        Put(0, 7, Side.Red, PieceType.Horse, 0);
        Put(4, 6, Side.Red, PieceType.Cannon, 0);

        Put(0, 3, Side.Black, PieceType.General, 0);
        Put(9, 0, Side.Black, PieceType.Rook, 0);
        Put(8, 8, Side.Black, PieceType.Rook, 1);
        return b;
    }

    private static double Q(Piece?[,] board, Side side)
    {
        var search = new AlphaBetaSearch(Engine);
        var state = new SearchState(Stopwatch.StartNew(), long.MaxValue);
        return search.Quiescence(board, side, double.NegativeInfinity, double.PositiveInfinity,
            side, side, Array.Empty<Move>(), state, depthFromRoot: 0, qDepth: 0);
    }

    [Fact]
    public void FixtureIsMateAndRedIsAheadOnMaterial()
    {
        var board = RedMatedButAheadOnMaterial();
        Assert.True(Engine.IsInCheck(board, Side.Red));
        Assert.Empty(Engine.GetLegalMoves(board, Side.Red));
        Assert.False(Engine.IsInCheck(board, Side.Black));
    }

    [Fact]
    public void QuiescenceScoresAMatedSideAsLostNotAsStandingPat()
    {
        var score = Q(RedMatedButAheadOnMaterial(), Side.Red);
        Assert.True(score < -900000, $"expected a mate-loss score, got {score}");
    }

    [Fact]
    public void QuiescenceStillStandsPatWhenNotInCheck()
    {
        // Same material minus the checking rook: Red is not in check and ahead,
        // so standing pat is legitimate and the score must be positive.
        var board = RedMatedButAheadOnMaterial();
        board[9, 0] = null;
        Assert.False(Engine.IsInCheck(board, Side.Red));
        Assert.True(Engine.GetLegalMoves(board, Side.Red).Count > 0);

        var score = Q(board, Side.Red);
        Assert.True(score > 0, $"expected a positive stand-pat score, got {score}");
    }
}