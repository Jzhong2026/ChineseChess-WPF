using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

/// <summary>
/// A5 — history-aware legal move generation must not rebuild the whole game as
/// cloned boards + 90-field string keys on every call. Measured before the fix:
/// 795.6 KB and 574.9 us per call, versus 263 us with no history at all.
/// A6 — the repetition key must depend on (side, pieceType) placement, not on
/// piece.Id, so two interchangeable same-type pieces swapping squares still
/// counts as the same position.
/// </summary>
public class EngineMoveGenerationTests
{
    private static readonly XiangqiEngine Engine = new();

    private static (Piece?[,] Board, Side Side, IReadOnlyList<Move> History) MidGame(int plies)
    {
        var rng = new Random(42);
        var state = Engine.CreateInitialState();
        for (var i = 0; i < plies; i++)
        {
            var moves = Engine.GetLegalMoves(state.Board, state.Turn, state.History);
            if (moves.Count == 0) break;
            state = Engine.MakeMove(state, moves[rng.Next(moves.Count)]);
            if (state.Status is GameStatus.RedWins or GameStatus.BlackWins) break;
        }
        return (state.Board, state.Turn, state.History);
    }

    private static long AllocPerCall(Action action, int iterations)
    {
        action();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        var before = GC.GetTotalAllocatedBytes(precise: true);
        for (var i = 0; i < iterations; i++) action();
        return (GC.GetTotalAllocatedBytes(precise: true) - before) / iterations;
    }

    [Fact]
    public void HistoryAwareGenerationDoesNotDominateAllocation()
    {
        var (board, side, history) = MidGame(40);
        Assert.True(history.Count >= 30, "fixture needs a real history");

        var withoutHistory = AllocPerCall(() => Engine.GetLegalMoves(board, side, null), 100);
        var withHistory = AllocPerCall(() => Engine.GetLegalMoves(board, side, history), 100);

        Assert.True(withHistory < withoutHistory * 3 / 2,
            $"history overhead too large: withHistory={withHistory / 1024.0:F1} KB, " +
            $"withoutHistory={withoutHistory / 1024.0:F1} KB (ratio {(double)withHistory / withoutHistory:F2})");
    }

    /// <summary>
    /// Regression guard for the repetition machinery. Hand-verified line:
    ///   R(5,0)-(0,0)+ G(0,4)-(1,4) R(0,0)-(1,0)+ G(1,4)-(0,4)
    /// After move 2 the shuttle R(0,0)-(1,0)+ checks a NEW position, so it is legal.
    /// After move 4 the position "rook(0,0), general(0,4), Black to move" has
    /// already occurred (right after move 1), so R(1,0)-(0,0)+ is a forbidden
    /// perpetual check. Note the engine forbids on the 2nd occurrence of a
    /// checked position, not the 3rd.
    /// </summary>
    [Fact]
    public void PerpetualCheckIsForbiddenOnceThePositionRepeats()
    {
        var board = Engine.EmptyBoard();
        board[0, 4] = new Piece("black-general-0", Side.Black, PieceType.General);
        board[9, 3] = new Piece("red-general-0", Side.Red, PieceType.General);
        board[5, 0] = new Piece("red-rook-0", Side.Red, PieceType.Rook);

        var state = new GameState(board, Side.Red, GameStatus.Playing, Array.Empty<Move>());
        Position P(int r, int c) => new(r, c);
        Move Step(GameState s, int fr, int fc, int tr, int tc) =>
            new(P(fr, fc), P(tr, tc), s.Board[fr, fc]!);

        state = Engine.MakeMove(state, Step(state, 5, 0, 0, 0));   // m1 check
        state = Engine.MakeMove(state, Step(state, 0, 4, 1, 4));   // m2

        // The next check reaches a position that has not occurred yet.
        var fresh = Engine.GetLegalMoves(state.Board, state.Turn, state.History);
        Assert.Contains(fresh, m => m.From == P(0, 0) && m.To == P(1, 0));
        Assert.False(Engine.IsRepeatCheckMove(state.Board, Step(state, 0, 0, 1, 0), state.Turn, state.History));

        state = Engine.MakeMove(state, Step(state, 0, 0, 1, 0));   // m3 check
        state = Engine.MakeMove(state, Step(state, 1, 4, 0, 4));   // m4

        // Now the same check would recreate the post-m1 position.
        var repeating = Step(state, 1, 0, 0, 0);
        Assert.True(Engine.IsRepeatCheckMove(state.Board, repeating, state.Turn, state.History));
        var filtered = Engine.GetLegalMoves(state.Board, state.Turn, state.History);
        Assert.DoesNotContain(filtered, m => m.From == P(1, 0) && m.To == P(0, 0));
        // ...and the unfiltered list still contains it, proving history did the work.
        Assert.Contains(Engine.GetLegalMoves(state.Board, state.Turn, null),
            m => m.From == P(1, 0) && m.To == P(0, 0));
    }

    /// <summary>
    /// A6 — repetition identity must come from (side, pieceType) placement, not
    /// piece.Id. Same fixture as above, but the rook standing on the board is
    /// re-labelled with a different Id while the history still carries the old
    /// one. A string key built from Ids would no longer match; a type-based key
    /// must still recognise the repeated check.
    /// </summary>
    [Fact]
    public void RepetitionIsRecognisedAcrossInterchangeablePieceIds()
    {
        var board = Engine.EmptyBoard();
        board[0, 4] = new Piece("black-general-0", Side.Black, PieceType.General);
        board[9, 3] = new Piece("red-general-0", Side.Red, PieceType.General);
        board[5, 0] = new Piece("red-rook-0", Side.Red, PieceType.Rook);

        var state = new GameState(board, Side.Red, GameStatus.Playing, Array.Empty<Move>());
        Move Step(GameState s, int fr, int fc, int tr, int tc) =>
            new(new Position(fr, fc), new Position(tr, tc), s.Board[fr, fc]!);

        state = Engine.MakeMove(state, Step(state, 5, 0, 0, 0));
        state = Engine.MakeMove(state, Step(state, 0, 4, 1, 4));
        state = Engine.MakeMove(state, Step(state, 0, 0, 1, 0));
        state = Engine.MakeMove(state, Step(state, 1, 4, 0, 4));

        // Re-label the rook: same square, same side, same type, different identity.
        var relabelled = Engine.CloneBoard(state.Board);
        relabelled[1, 0] = new Piece("red-rook-RENAMED", Side.Red, PieceType.Rook);
        var repeating = new Move(new Position(1, 0), new Position(0, 0), relabelled[1, 0]!);

        Assert.True(Engine.IsRepeatCheckMove(relabelled, repeating, state.Turn, state.History),
            "repetition must not depend on piece.Id");
    }
}