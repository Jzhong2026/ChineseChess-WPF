using System;
using System.Collections.Generic;
using System.Linq;

namespace Learning.Demo4NeuralEncoding;

// Standalone reference code for the neural encoding path.
// It mirrors the production classes:
//   BoardEncoder, MoveEncoder, XiangqiNeuralAiService.EncodeCnn().
public static class NeuralEncodingDemo
{
    private const int Rows = 10;
    private const int Cols = 9;
    private const int BoardSize = Rows * Cols;
    private const int PiecePlanes = 14;
    private const int SidePlanes = 2;
    private const int CnnPlanes = PiecePlanes + SidePlanes;
    private const int ActionCount = BoardSize * BoardSize;

    public enum Side
    {
        Red = 0,
        Black = 1,
    }

    public enum PieceType
    {
        General = 0,
        Advisor = 1,
        Elephant = 2,
        Horse = 3,
        Rook = 4,
        Cannon = 5,
        Soldier = 6,
    }

    public readonly record struct Position(int Row, int Col);
    public readonly record struct Piece(Side Side, PieceType Type);
    public readonly record struct Move(Position From, Position To);

    public static int GetBoardIndex(int row, int col, Side side, PieceType type)
    {
        var square = GetSquareIndex(new Position(row, col));
        var plane = (int)side * 7 + (int)type;
        return square * PiecePlanes + plane;
    }

    public static float[] EncodeBoard(Piece?[,] board)
    {
        if (board.GetLength(0) != Rows || board.GetLength(1) != Cols)
        {
            throw new ArgumentException("Board must be 10x9.", nameof(board));
        }

        var encoded = new float[BoardSize * PiecePlanes];
        for (var row = 0; row < Rows; row++)
        {
            for (var col = 0; col < Cols; col++)
            {
                var piece = board[row, col];
                if (piece is null) continue;

                encoded[GetBoardIndex(row, col, piece.Value.Side, piece.Value.Type)] = 1f;
            }
        }

        return encoded;
    }

    public static float[,,,] PackCnnInput(float[] flatBoard, Side sideToMove)
    {
        if (flatBoard.Length != BoardSize * PiecePlanes)
        {
            throw new ArgumentException("Encoded board must have length 1260.", nameof(flatBoard));
        }

        var tensor = new float[1, CnnPlanes, Rows, Cols];

        for (var row = 0; row < Rows; row++)
        {
            for (var col = 0; col < Cols; col++)
            {
                for (var plane = 0; plane < PiecePlanes; plane++)
                {
                    var flatIndex = (row * Cols + col) * PiecePlanes + plane;
                    tensor[0, plane, row, col] = flatBoard[flatIndex];
                }
            }
        }

        var redPlane = sideToMove == Side.Red ? 1f : 0f;
        var blackPlane = sideToMove == Side.Black ? 1f : 0f;
        for (var row = 0; row < Rows; row++)
        {
            for (var col = 0; col < Cols; col++)
            {
                tensor[0, 14, row, col] = redPlane;
                tensor[0, 15, row, col] = blackPlane;
            }
        }

        return tensor;
    }

    public static int EncodeMove(Move move)
    {
        var fromIndex = GetSquareIndex(move.From);
        var toIndex = GetSquareIndex(move.To);
        return fromIndex * BoardSize + toIndex;
    }

    public static Move DecodeMove(int actionId)
    {
        if (actionId < 0 || actionId >= ActionCount)
        {
            throw new ArgumentOutOfRangeException(nameof(actionId));
        }

        var fromIndex = actionId / BoardSize;
        var toIndex = actionId % BoardSize;
        return new Move(GetPosition(fromIndex), GetPosition(toIndex));
    }

    public static float[] CombineFactoredPolicy(float[] fromLogits, float[] toLogits)
    {
        if (fromLogits.Length != BoardSize) throw new ArgumentException("Expected 90 values.", nameof(fromLogits));
        if (toLogits.Length != BoardSize) throw new ArgumentException("Expected 90 values.", nameof(toLogits));

        var combined = new float[ActionCount];
        for (var fromIndex = 0; fromIndex < BoardSize; fromIndex++)
        {
            for (var toIndex = 0; toIndex < BoardSize; toIndex++)
            {
                combined[fromIndex * BoardSize + toIndex] = fromLogits[fromIndex] + toLogits[toIndex];
            }
        }

        return combined;
    }

    public static Move ChooseBestLegalMove(float[] policyLogits, IReadOnlyList<Move> legalMoves)
    {
        if (policyLogits.Length != ActionCount)
        {
            throw new ArgumentException("Expected 8100 action logits.", nameof(policyLogits));
        }

        if (legalMoves.Count == 0)
        {
            throw new ArgumentException("Legal moves must not be empty.", nameof(legalMoves));
        }

        return legalMoves
            .OrderByDescending(move => policyLogits[EncodeMove(move)])
            .First();
    }

    public static Move FlipMoveHorizontal(Move move)
    {
        return new Move(
            new Position(move.From.Row, Cols - 1 - move.From.Col),
            new Position(move.To.Row, Cols - 1 - move.To.Col));
    }

    private static int GetSquareIndex(Position position)
    {
        if (position.Row < 0 || position.Row >= Rows) throw new ArgumentOutOfRangeException(nameof(position));
        if (position.Col < 0 || position.Col >= Cols) throw new ArgumentOutOfRangeException(nameof(position));
        return position.Row * Cols + position.Col;
    }

    private static Position GetPosition(int squareIndex)
    {
        if (squareIndex < 0 || squareIndex >= BoardSize) throw new ArgumentOutOfRangeException(nameof(squareIndex));
        return new Position(squareIndex / Cols, squareIndex % Cols);
    }
}

