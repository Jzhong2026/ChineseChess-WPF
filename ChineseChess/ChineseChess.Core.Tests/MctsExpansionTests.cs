using ChineseChess.Encoding;
using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

/// <summary>
/// A3 — ExpandNode must size child storage by the number of LEGAL moves (~40),
/// not by MoveEncoder.ActionCount (8100). Five dense 8100 arrays cost 221.6 KB
/// per expanded node (measured), which is what drives ~1.93 MB per simulation.
/// </summary>
public class MctsExpansionTests
{
    private static readonly XiangqiEngine Engine = new();

    [Fact]
    public void ChildStorageIsSizedToLegalMoveCount()
    {
        var state = Engine.CreateInitialState();
        var legal = Engine.GetLegalMoves(state.Board, state.Turn, state.History);
        using var neural = new XiangqiNeuralAiService(TestPaths.DeployedOnnx, Engine, isCnnModel: true);
        var mcts = new MctsAiService(neural, Engine);
        var node = new MctsNode(null, 0f, state.Board, state.Turn);

        mcts.ExpandNode(node, state.History);

        Assert.False(node.IsLeaf);
        Assert.Equal(legal.Count, node.Children!.Length);
        Assert.Equal(legal.Count, node.ChildMoves!.Length);
        Assert.Equal(legal.Count, node.ChildN!.Length);
        Assert.Equal(legal.Count, node.ChildW!.Length);
        Assert.Equal(legal.Count, node.ChildP!.Length);
    }

    [Fact]
    public void EveryLegalMoveGetsANonZeroPriorAndProbabilitiesSumToOne()
    {
        var state = Engine.CreateInitialState();
        var legal = Engine.GetLegalMoves(state.Board, state.Turn, state.History);
        using var neural = new XiangqiNeuralAiService(TestPaths.DeployedOnnx, Engine, isCnnModel: true);
        var mcts = new MctsAiService(neural, Engine);
        var node = new MctsNode(null, 0f, state.Board, state.Turn);

        mcts.ExpandNode(node, state.History);

        Assert.All(node.ChildMoves!, m => Assert.NotNull(m));
        Assert.Equal(
            legal.Select(MoveEncoder.Encode).OrderBy(x => x).ToList(),
            node.ChildMoves!.Select(m => MoveEncoder.Encode(m!)).OrderBy(x => x).ToList());
        Assert.Equal(1f, node.ChildP!.Sum(), 3);
        Assert.All(node.ChildP!, p => Assert.True(p >= 0f));
        Assert.True(node.ChildP!.Count(p => p > 0f) >= legal.Count - 2,
            "softmax underflow should not zero out more than a couple of priors");
    }

    [Fact]
    public void TerminalNodeIsMarkedWithoutAllocatingChildStorage()
    {
        // Black general (0,4) mated by Red rooks on (0,0) and (1,8); Red general (9,3).
        var board = Engine.EmptyBoard();
        board[0, 4] = new Piece("black-general-0", Side.Black, PieceType.General);
        board[9, 3] = new Piece("red-general-0", Side.Red, PieceType.General);
        board[0, 0] = new Piece("red-rook-0", Side.Red, PieceType.Rook);
        board[1, 8] = new Piece("red-rook-1", Side.Red, PieceType.Rook);
        Assert.Empty(Engine.GetLegalMoves(board, Side.Black));

        using var neural = new XiangqiNeuralAiService(TestPaths.DeployedOnnx, Engine, isCnnModel: true);
        var mcts = new MctsAiService(neural, Engine);
        var node = new MctsNode(null, 0f, board, Side.Black);

        mcts.ExpandNode(node, Array.Empty<Move>());

        Assert.True(node.IsTerminal);
        Assert.Null(node.Children);
    }
}