using ChineseChess.Models;
using ChineseChess.Services;

namespace ChineseChess.Encoding;

public static class MoveEncoder
{
    public const int BoardSquareCount = XiangqiEngine.BoardRows * XiangqiEngine.BoardCols;
    public const int ActionCount = BoardSquareCount * BoardSquareCount;
    public const int MinActionId = 0;
    public const int MaxActionId = ActionCount - 1;

    public static int Encode(Move move) => Encode(move.From, move.To);

    public static int Encode(Position from, Position to)
    {
        var fromIndex = GetSquareIndex(from);
        var toIndex = GetSquareIndex(to);
        return fromIndex * BoardSquareCount + toIndex;
    }

    public static (Position From, Position To) Decode(int actionId)
    {
        if (actionId < MinActionId || actionId > MaxActionId) throw new ArgumentOutOfRangeException(nameof(actionId));

        var fromIndex = actionId / BoardSquareCount;
        var toIndex = actionId % BoardSquareCount;
        return (GetPosition(fromIndex), GetPosition(toIndex));
    }

    private static int GetSquareIndex(Position position)
    {
        if (position.Row < 0 || position.Row >= XiangqiEngine.BoardRows) throw new ArgumentOutOfRangeException(nameof(position));
        if (position.Col < 0 || position.Col >= XiangqiEngine.BoardCols) throw new ArgumentOutOfRangeException(nameof(position));
        return position.Row * XiangqiEngine.BoardCols + position.Col;
    }

    private static Position GetPosition(int squareIndex)
    {
        if (squareIndex < 0 || squareIndex >= BoardSquareCount) throw new ArgumentOutOfRangeException(nameof(squareIndex));
        return new Position(squareIndex / XiangqiEngine.BoardCols, squareIndex % XiangqiEngine.BoardCols);
    }
}
