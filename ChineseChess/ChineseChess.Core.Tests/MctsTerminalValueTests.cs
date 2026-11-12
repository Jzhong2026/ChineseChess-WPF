using ChineseChess.Models;
using ChineseChess.Ai.Neural;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

/// <summary>
/// A1 — MCTS must score a checkmated side as LOSING.
/// Hand-constructed roughly-balanced position (Red ~3500 vs Black ~3080 material)
/// so the value head is NOT saturated: a saturated Q~0.99 would drown the PUCT
/// exploration term and the mating move would never be visited at all.
/// Black general (0,4); Red rook (1,8) seals row 1; Red rook (5,0) slides up the
/// empty a-file to (0,0) sealing row 0. Escapes (0,3)/(0,5) are covered by that
/// rook, (1,4) by the rook on row 1, and no black piece can interpose on
/// (0,1)/(0,2)/(0,3) or capture the checker. Mate-in-1 is objectively winning.
/// </summary>
public class MctsTerminalValueTests
{
    private static readonly XiangqiEngine Engine = new();

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
    public void MateInOneIsObjectivelyWinning()
    {
        var board = MateInOneBoard();
        var mating = new Move(new Position(5, 0), new Position(0, 0), board[5, 0]!);

        // The fixture must be a LEGAL mate-in-1, not just a mate-looking position.
        Assert.Contains(Engine.GetLegalMoves(board, Side.Red), m => m.From == mating.From && m.To == mating.To);
        Assert.False(Engine.IsInCheck(board, Side.Black));
        Assert.False(Engine.IsInCheck(board, Side.Red));

        var after = Engine.ApplyMove(board, mating);

        Assert.True(Engine.IsInCheck(after, Side.Black));
        Assert.Empty(Engine.GetLegalMoves(after, Side.Black));
        Assert.Equal(GameStatus.RedWins, Engine.GetGameStatus(after, Side.Black));
    }

    [Fact]
    public void SearchPrefersTheMatingMove()
    {
        var board = MateInOneBoard();
        using var neural = new XiangqiNeuralAiService(TestPaths.DeployedOnnx, Engine, isCnnModel: true);
        var mcts = new MctsAiService(neural, Engine);

        var result = mcts.Search(board, Side.Red, Array.Empty<Move>(), simulations: 400, temperature: 0f, timeLimitMs: 60000);

        Assert.NotNull(result.BestMove);
        Assert.Equal(new Position(5, 0), result.BestMove!.From);
        Assert.Equal(new Position(0, 0), result.BestMove.To);
    }
}