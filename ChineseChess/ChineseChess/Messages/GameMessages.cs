using ChineseChess.Models;

namespace ChineseChess.Messages;

public sealed record SquareSelectedMessage(Position Position);

public sealed record BoardStateChangedMessage(GameState State, Position? Selected, IReadOnlySet<Position> LegalTargets, Move? LastMove);

public sealed record SideChangedMessage(Side Side);

public sealed record AiLevelChangedMessage(int Level);

public sealed record AiTimeLimitChangedMessage(int TimeLimitMs);

public sealed record RestartRequestedMessage;

public sealed record UndoRequestedMessage;
