using System.Diagnostics;
using ChineseChess.Models;
using ChineseChess.Search;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

public class SearchRegressionTests
{
    [Theory]
    [InlineData(999997, 999995)]
    [InlineData(-999997, -999995)]
    [InlineData(123, 123)]
    public void CachedScoreIsRestoredForCurrentPly(double stored, double expected)
    {
        var table = new TranspositionTable();
        table.Store(1, new TranspositionTable.Entry(4, stored, TranspositionTable.Exact, null), depthFromRoot: 1);
        Assert.True(table.TryProbe(1, 4, double.NegativeInfinity, double.PositiveInfinity,
            out var entry, out var cutoff, depthFromRoot: 3));
        Assert.True(cutoff);
        Assert.Equal(expected, entry!.Score);
    }

    [Theory]
    [InlineData(TranspositionTable.Lower, 999996, 999998)]
    [InlineData(TranspositionTable.Upper, -999998, -999996)]
    public void BoundCutoffsUseRestoredMateDistance(byte flag, double alpha, double beta)
    {
        var table = new TranspositionTable();
        var score = flag == TranspositionTable.Lower ? 999997 : -999997;
        table.Store(1, new TranspositionTable.Entry(4, score, flag, null), depthFromRoot: 1);
        table.TryProbe(1, 4, alpha, beta, out _, out var cutoff, depthFromRoot: 3);
        Assert.False(cutoff);
    }

    [Fact]
    public void CachedMateMatchesFreshSearchAtDifferentRootDistance()
    {
        var engine = new XiangqiEngine();
        var board = engine.EmptyBoard();
        board[0, 4] = new Piece("bg", Side.Black, PieceType.General);
        board[9, 3] = new Piece("rg", Side.Red, PieceType.General);
        board[1, 8] = new Piece("rr0", Side.Red, PieceType.Rook);
        board[5, 0] = new Piece("rr1", Side.Red, PieceType.Rook);

        double Score(AlphaBetaSearch search, int ply) => search.Negamax(board, Side.Red, 1,
            double.NegativeInfinity, double.PositiveInfinity, Side.Red, Side.Red,
            Array.Empty<Move>(), new SearchState(Stopwatch.StartNew(), 10000), ply);
        var cached = new AlphaBetaSearch(engine);
        Assert.Equal(999997, Score(cached, 1));
        Assert.Equal(999995, Score(cached, 3));
        Assert.Equal(Score(new AlphaBetaSearch(engine), 3), Score(cached, 3));
    }
}
