using ChineseChess.Ai.Neural;
using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

public class NeuralValueEncodingTests
{
    [Fact]
    public void CanonicalEncodingPreservesOwnPiecesUnderColorRotation()
    {
        var engine = new XiangqiEngine();
        var board = engine.EmptyBoard();
        board[8, 2] = new Piece("own", Side.Red, PieceType.Horse);
        board[2, 7] = new Piece("enemy", Side.Black, PieceType.Cannon);
        var rotated = engine.EmptyBoard();
        foreach (var row in Enumerable.Range(0, 10))
        foreach (var col in Enumerable.Range(0, 9))
            if (board[row, col] is { } p) rotated[9-row, 8-col] = p with { Side = engine.Opposite(p.Side) };
        var red = new float[1260]; var black = new float[1260];
        OnnxValueEvaluator.Encode(board, Side.Red, red);
        OnnxValueEvaluator.Encode(rotated, Side.Black, black);
        Assert.Equal(red, black);
        var redActive = new long[90]; var blackActive = new long[90];
        OnnxValueEvaluator.EncodeActive(board, Side.Red, redActive);
        OnnxValueEvaluator.EncodeActive(rotated, Side.Black, blackActive);
        Assert.Equal(redActive, blackActive);
        var reconstructed = new float[1260];
        foreach (var index in redActive.Where(i => i != 1260)) reconstructed[index] = 1;
        Assert.Equal(red, reconstructed);
        Assert.Equal(OnnxValueEvaluator.MakeKey(board, Side.Red), OnnxValueEvaluator.MakeKey(rotated, Side.Black));
        Assert.NotEqual(OnnxValueEvaluator.MakeKey(board, Side.Red), OnnxValueEvaluator.MakeKey(board, Side.Black));
        var oldKey = OnnxValueEvaluator.MakeKey(board, Side.Red);
        board[8, 2] = null;
        Assert.NotEqual(oldKey, OnnxValueEvaluator.MakeKey(board, Side.Red));
        Assert.Equal(1f, red[3 * 90 + 8 * 9 + 2]);
        Assert.Equal(1f, red[12 * 90 + 2 * 9 + 7]);
        Assert.Equal(2f, red.Sum());
    }

    [Fact]
    public void ValueEvaluatorRejectsAPlainPolicyModel()
    {
        Assert.Throws<ArgumentException>(() => new OnnxValueEvaluator(TestPaths.DeployedOnnx));
    }
}
