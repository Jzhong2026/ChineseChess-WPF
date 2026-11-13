using ChineseChess.Ai.Neural;
using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

public class FairyOnnxEncodingTests
{
    [Fact]
    public void HalfKaUsesPalaceBucketAndRankReflection()
    {
        var board = new XiangqiEngine().EmptyBoard();
        board[9, 4] = new Piece("red-king", Side.Red, PieceType.General);
        board[0, 4] = new Piece("black-king", Side.Black, PieceType.General);
        board[8, 0] = new Piece("red-rook", Side.Red, PieceType.Rook);
        board[2, 7] = new Piece("black-horse", Side.Black, PieceType.Horse);
        var red = new long[32]; var black = new long[32];
        Assert.Equal(4, FairyOnnxEvaluator.Encode(board, Side.Red, red));
        Assert.Equal(4, FairyOnnxEvaluator.Encode(board, Side.Black, black));
        Assert.Contains(1179L, red); // own rook, rank1/file0, own king bucket1.
        Assert.Contains(1332L, black); // enemy rook, rank8/file0 (no file reflection).
        Assert.Contains(1915L, black); // own horse, rank2/file7.
        Assert.Contains(2254L, red); Assert.Contains(2335L, red);
        Assert.Equal(28, red.Count(i => i == 10530));
    }

    [Fact]
    public void InvalidKingPositionCannotSilentlyUseAnotherBucket()
    {
        var board = new XiangqiEngine().EmptyBoard();
        board[5, 4] = new Piece("king", Side.Red, PieceType.General);
        Assert.Throws<ArgumentException>(() => FairyOnnxEvaluator.Encode(board, Side.Red, new long[32]));
    }
}
