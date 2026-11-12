using ChineseChess.Models;

namespace ChineseChess.Services;

/// <summary>
/// Zobrist keys over (square, side, pieceType) plus side-to-move.
/// Piece identity is deliberately excluded: two same-side pieces of the same
/// type are interchangeable for repetition purposes.
/// </summary>
internal static class ZobristHash
{
    private const ulong Seed = 0x9E3779B97F4A7C15UL;
    private static readonly ulong[,,,] Pieces = CreatePieces();
    private static readonly ulong[] TurnKeys = { Next(0), Next(1) };

    public static ulong PieceKey(int row, int col, Side side, PieceType type) => Pieces[row, col, (int)side, (int)type];

    public static ulong TurnKey(Side side) => TurnKeys[(int)side];

    public static ulong Compute(Piece?[,] board, Side turn)
    {
        var hash = TurnKeys[(int)turn];
        for (var row = 0; row < XiangqiEngine.BoardRows; row++)
        {
            for (var col = 0; col < XiangqiEngine.BoardCols; col++)
            {
                var piece = board[row, col];
                if (piece is not null) hash ^= Pieces[row, col, (int)piece.Side, (int)piece.Type];
            }
        }
        return hash;
    }

    private static ulong[,,,] CreatePieces()
    {
        var values = new ulong[XiangqiEngine.BoardRows, XiangqiEngine.BoardCols, 2, 7];
        var index = 2;
        for (var row = 0; row < XiangqiEngine.BoardRows; row++)
        {
            for (var col = 0; col < XiangqiEngine.BoardCols; col++)
            {
                for (var side = 0; side < 2; side++)
                {
                    for (var type = 0; type < 7; type++)
                    {
                        values[row, col, side, type] = Next(index++);
                    }
                }
            }
        }
        return values;
    }

    private static ulong Next(int index)
    {
        var value = Seed + (ulong)index * 0x9E3779B97F4A7C15UL;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
        return value ^ (value >> 31);
    }
}