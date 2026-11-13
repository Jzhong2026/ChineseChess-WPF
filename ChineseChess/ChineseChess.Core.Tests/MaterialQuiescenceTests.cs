using System.Diagnostics;
using ChineseChess.Ai.Neural;
using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

public class MaterialQuiescenceTests
{
    [Fact]
    public void PoisonedPawnCaptureSeesTheImmediateRookRecapture()
    {
        var engine = new XiangqiEngine();
        var board = engine.EmptyBoard();
        void Put(int row, int col, Side side, PieceType type) => board[row, col] = new Piece($"{row}-{col}", side, type);
        Put(9, 4, Side.Red, PieceType.General);
        Put(0, 4, Side.Black, PieceType.General);
        Put(5, 4, Side.Red, PieceType.Soldier);
        Put(5, 0, Side.Red, PieceType.Rook);
        Put(4, 0, Side.Black, PieceType.Soldier);
        Put(0, 0, Side.Black, PieceType.Rook);
        var capture = engine.GetLegalMoves(board, Side.Red).Single(m => m.From == new Position(5, 0) && m.To == new Position(4, 0));
        var after = engine.ApplyMove(board, capture);
        Assert.Equal(-120, MaterialQuiescence.Material(after, Side.Black));
        var score = new MaterialQuiescence(engine).Evaluate(after, Side.Black, new[] { capture }, 3,
            Stopwatch.StartNew(), 10000);
        Assert.True(score >= 780, $"Expected the rook recapture, got {score}");
    }

    [Fact]
    public void ZeroDepthUsesTheDeclaredMaterialEstimate()
    {
        var engine = new XiangqiEngine();
        var state = engine.CreateInitialState();
        var tactics = new MaterialQuiescence(engine);
        Assert.Equal(0, tactics.Evaluate(state.Board, Side.Red, state.History, 0, Stopwatch.StartNew(), 10000));
    }
}
