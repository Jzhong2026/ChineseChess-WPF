namespace ChineseChess.Models;

public enum Side
{
    Red,
    Black
}

public enum PieceType
{
    General,
    Advisor,
    Elephant,
    Horse,
    Rook,
    Cannon,
    Soldier
}

public enum GameStatus
{
    Playing,
    RedCheck,
    BlackCheck,
    RedWins,
    BlackWins,
    Draw
}

public readonly record struct Position(int Row, int Col);

public sealed record Piece(string Id, Side Side, PieceType Type);

public sealed record Move(Position From, Position To, Piece Piece, Piece? Captured = null);

public sealed record GameState(Piece?[,] Board, Side Turn, GameStatus Status, IReadOnlyList<Move> History);

public sealed record SearchStats(int DepthReached, int Nodes, double TimeMs, double BestScore, int TtHits = 0, int TtStores = 0, int TtBestMoveHits = 0, int TtScoreHits = 0);

public sealed record AiSearchResult(Move? Move, SearchStats Stats);

public sealed record MoveSelectionOptions(
    bool EnableRandomSelection = false,
    int TopK = 1,
    double NearBestWindow = 0,
    int? Seed = null,
    int RandomOpeningPlies = 0)
{
    public static MoveSelectionOptions Deterministic { get; } = new();
}
