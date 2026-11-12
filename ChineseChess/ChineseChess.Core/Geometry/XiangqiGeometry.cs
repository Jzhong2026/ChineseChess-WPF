using ChineseChess.Models;
using ChineseChess.Services;

namespace ChineseChess.Geometry;

/// <summary>
/// Single source of truth for piece attack / mobility geometry, shared by the
/// rules engine (check detection) and the position evaluator. Keeping it in one
/// place prevents the two from drifting apart.
/// </summary>
public static class XiangqiGeometry
{
    public const int Rows = XiangqiEngine.BoardRows;
    public const int Cols = XiangqiEngine.BoardCols;

    public static readonly (int dr, int dc)[] Orthogonal = { (-1, 0), (1, 0), (0, -1), (0, 1) };

    /// <summary>Horse jump offsets (piece -> target) with the leg offset relative to the piece.</summary>
    public static readonly (int dr, int dc, int lr, int lc)[] HorseJumps =
    {
        (-2, -1, -1, 0), (-2, 1, -1, 0),
        (2, -1, 1, 0), (2, 1, 1, 0),
        (-1, -2, 0, -1), (1, -2, 0, -1),
        (-1, 2, 0, 1), (1, 2, 0, 1),
    };

    /// <summary>Offsets from a TARGET square to an attacking horse, and to that horse's leg.</summary>
    private static readonly (int dr, int dc, int lr, int lc)[] HorseAttackOffsets =
    {
        (-2, -1, -1, -1), (-2, 1, -1, 1),
        (2, -1, 1, -1), (2, 1, 1, 1),
        (-1, -2, -1, -1), (1, -2, 1, -1),
        (-1, 2, -1, 1), (1, 2, 1, 1),
    };

    public static bool InBounds(int row, int col) => row >= 0 && row < Rows && col >= 0 && col < Cols;

    public static int Forward(Side side) => side == Side.Red ? -1 : 1;

    public static bool CrossedRiver(Side side, int row) => side == Side.Red ? row <= 4 : row >= 5;

    public static int SquareIndex(int row, int col) => row * Cols + col;

    /// <summary>
    /// Allocation-free attack scan on a general's square. Advisors and elephants
    /// are skipped: neither can leave its own palace/half, so neither can ever
    /// attack the enemy general.
    /// </summary>
    public static bool IsSquareAttacked(Piece?[,] board, Position target, Side attacker)
    {
        var tr = target.Row;
        var tc = target.Col;

        foreach (var (dr, dc) in Orthogonal)
        {
            var r = tr + dr;
            var c = tc + dc;
            var distance = 1;
            var screens = 0;
            while (InBounds(r, c))
            {
                var piece = board[r, c];
                if (piece is not null)
                {
                    if (piece.Side == attacker)
                    {
                        if (screens == 0)
                        {
                            if (piece.Type == PieceType.Rook) return true;
                            if (piece.Type == PieceType.General && (dc == 0 || distance == 1)) return true;
                        }
                        else if (piece.Type == PieceType.Cannon)
                        {
                            return true;
                        }
                        else
                        {
                            break;
                        }
                    }
                    screens++;
                    if (screens >= 2) break;
                }
                r += dr;
                c += dc;
                distance++;
            }
        }

        foreach (var (dr, dc, lr, lc) in HorseAttackOffsets)
        {
            var hr = tr + dr;
            var hc = tc + dc;
            if (!InBounds(hr, hc)) continue;
            var piece = board[hr, hc];
            if (piece is null || piece.Side != attacker || piece.Type != PieceType.Horse) continue;
            if (board[tr + lr, tc + lc] is null) return true;
        }

        var soldierRow = tr - Forward(attacker);
        if (InBounds(soldierRow, tc))
        {
            var piece = board[soldierRow, tc];
            if (piece is not null && piece.Side == attacker && piece.Type == PieceType.Soldier) return true;
        }
        for (var dc = -1; dc <= 1; dc += 2)
        {
            var c = tc + dc;
            if (!InBounds(tr, c)) continue;
            var piece = board[tr, c];
            if (piece is null || piece.Side != attacker || piece.Type != PieceType.Soldier) continue;
            if (CrossedRiver(attacker, tr)) return true;
        }

        return false;
    }

    /// <summary>Whether a specific piece attacks a specific square (used by king-safety).</summary>
    public static bool PieceAttacksSquare(Piece?[,] board, int r, int c, Piece piece, int tr, int tc) => piece.Type switch
    {
        PieceType.Rook => LineAttacksSquare(board, r, c, tr, tc),
        PieceType.Cannon => CannonAttacksSquare(board, r, c, tr, tc),
        PieceType.Horse => HorseAttacksSquare(board, r, c, tr, tc),
        PieceType.Soldier => SoldierAttacksSquare(piece.Side, r, c, tr, tc),
        _ => false,
    };

    /// <summary>Pseudo-mobility: empty squares plus captures, no check/repetition filtering.</summary>
    public static int CountMobility(Piece?[,] board, int r, int c, Piece piece) => piece.Type switch
    {
        PieceType.Rook => CountLineMoves(board, r, c, piece),
        PieceType.Cannon => CountCannonMoves(board, r, c, piece),
        PieceType.Horse => CountHorseMoves(board, r, c, piece),
        PieceType.Soldier => CountSoldierMoves(board, r, c, piece),
        _ => 0,
    };

    public static int CountHorseMoves(Piece?[,] board, int r, int c, Piece piece)
    {
        var count = 0;
        foreach (var (dr, dc, lr, lc) in HorseJumps)
        {
            if (!InBounds(r + lr, c + lc) || board[r + lr, c + lc] is not null) continue;
            var nr = r + dr;
            var nc = c + dc;
            if (!InBounds(nr, nc)) continue;
            var landing = board[nr, nc];
            if (landing is not null && landing.Side == piece.Side) continue;
            count++;
        }
        return count;
    }

    private static int CountLineMoves(Piece?[,] board, int r, int c, Piece piece)
    {
        var count = 0;
        foreach (var (dr, dc) in Orthogonal)
        {
            var nr = r + dr;
            var nc = c + dc;
            while (InBounds(nr, nc))
            {
                var target = board[nr, nc];
                if (target is null) count++;
                else
                {
                    if (target.Side != piece.Side) count++;
                    break;
                }
                nr += dr;
                nc += dc;
            }
        }
        return count;
    }

    private static int CountCannonMoves(Piece?[,] board, int r, int c, Piece piece)
    {
        var count = 0;
        foreach (var (dr, dc) in Orthogonal)
        {
            var nr = r + dr;
            var nc = c + dc;
            var seenScreen = false;
            while (InBounds(nr, nc))
            {
                var target = board[nr, nc];
                if (!seenScreen)
                {
                    if (target is null) count++;
                    else seenScreen = true;
                }
                else if (target is not null)
                {
                    if (target.Side != piece.Side) count++;
                    break;
                }
                nr += dr;
                nc += dc;
            }
        }
        return count;
    }

    private static int CountSoldierMoves(Piece?[,] board, int r, int c, Piece piece)
    {
        var count = 0;
        var fr = r + Forward(piece.Side);
        if (InBounds(fr, c))
        {
            var p = board[fr, c];
            if (p is null || p.Side != piece.Side) count++;
        }
        if (CrossedRiver(piece.Side, r))
        {
            if (c > 0)
            {
                var p = board[r, c - 1];
                if (p is null || p.Side != piece.Side) count++;
            }
            if (c < Cols - 1)
            {
                var p = board[r, c + 1];
                if (p is null || p.Side != piece.Side) count++;
            }
        }
        return count;
    }

    private static bool LineAttacksSquare(Piece?[,] board, int r, int c, int tr, int tc)
    {
        foreach (var (dr, dc) in Orthogonal)
        {
            var nr = r + dr;
            var nc = c + dc;
            while (InBounds(nr, nc))
            {
                if (nr == tr && nc == tc) return true;
                if (board[nr, nc] is not null) break;
                nr += dr;
                nc += dc;
            }
        }
        return false;
    }

    private static bool CannonAttacksSquare(Piece?[,] board, int r, int c, int tr, int tc)
    {
        foreach (var (dr, dc) in Orthogonal)
        {
            var nr = r + dr;
            var nc = c + dc;
            var seenScreen = false;
            while (InBounds(nr, nc))
            {
                if (nr == tr && nc == tc) return seenScreen;
                if (board[nr, nc] is not null)
                {
                    if (seenScreen) break;
                    seenScreen = true;
                }
                nr += dr;
                nc += dc;
            }
        }
        return false;
    }

    private static bool HorseAttacksSquare(Piece?[,] board, int r, int c, int tr, int tc)
    {
        foreach (var (dr, dc, lr, lc) in HorseJumps)
        {
            if (r + dr != tr || c + dc != tc) continue;
            return InBounds(r + lr, c + lc) && board[r + lr, c + lc] is null;
        }
        return false;
    }

    private static bool SoldierAttacksSquare(Side side, int r, int c, int tr, int tc)
    {
        if (tr == r + Forward(side) && tc == c) return true;
        return CrossedRiver(side, r) && tr == r && (tc == c - 1 || tc == c + 1);
    }

    /// <summary>The five squares an advisor can occupy (palace corners + centre).</summary>
    public static bool IsAdvisorSlot(Side side, int row, int col)
    {
        var backRow = side == Side.Red ? 9 : 0;
        var frontRow = side == Side.Red ? 7 : 2;
        var centreRow = side == Side.Red ? 8 : 1;
        if (col == 4) return row == centreRow;
        if (col != 3 && col != 5) return false;
        return row == backRow || row == frontRow;
    }

    /// <summary>The seven squares an elephant can occupy (never crosses the river).</summary>
    public static bool IsElephantSlot(Side side, int row, int col)
    {
        var backRow = side == Side.Red ? 9 : 0;
        var midRow = side == Side.Red ? 7 : 2;
        var bankRow = side == Side.Red ? 5 : 4;
        if (row == backRow || row == bankRow) return col == 2 || col == 6;
        if (row == midRow) return col == 0 || col == 4 || col == 8;
        return false;
    }
}