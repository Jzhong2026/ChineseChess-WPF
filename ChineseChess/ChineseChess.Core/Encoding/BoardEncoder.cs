using ChineseChess.Models;
using ChineseChess.Services;

namespace ChineseChess.Encoding;

public static class BoardEncoder
{
    public const int Planes = 14;
    public const int EncodedLength = XiangqiEngine.BoardRows * XiangqiEngine.BoardCols * Planes;

    public static float[] Encode(Piece?[,] board)
    {
        if (board.GetLength(0) != XiangqiEngine.BoardRows || board.GetLength(1) != XiangqiEngine.BoardCols)
        {
            throw new ArgumentException("Board must be 10x9.", nameof(board));
        }

        var encoded = new float[EncodedLength];
        for (var row = 0; row < XiangqiEngine.BoardRows; row++)
        {
            for (var col = 0; col < XiangqiEngine.BoardCols; col++)
            {
                var piece = board[row, col];
                if (piece is null) continue;

                encoded[GetIndex(row, col, piece.Side, piece.Type)] = 1f;
            }
        }

        return encoded;
    }

    public static int GetIndex(int row, int col, Side side, PieceType type)
    {
        if (row < 0 || row >= XiangqiEngine.BoardRows) throw new ArgumentOutOfRangeException(nameof(row));
        if (col < 0 || col >= XiangqiEngine.BoardCols) throw new ArgumentOutOfRangeException(nameof(col));

        var square = row * XiangqiEngine.BoardCols + col;
        var plane = (int)side * 7 + (int)type;
        return square * Planes + plane;
    }
}
