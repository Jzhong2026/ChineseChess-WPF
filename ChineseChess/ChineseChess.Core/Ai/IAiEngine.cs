using ChineseChess.Models;

namespace ChineseChess.Ai;

/// <summary>
/// Uniform input for any move-selecting engine (classic search, MCTS, pure
/// policy). Engine-specific tuning lives in each engine's constructor/config,
/// not in this request (ISP).
/// </summary>
public sealed record AiMoveRequest(
    Piece?[,] Board,
    Side Side,
    IReadOnlyList<Move> History,
    int TimeLimitMs,
    MoveSelectionOptions? Selection = null,
    int Difficulty = 4,
    float Temperature = 0f,
    int? Simulations = null);

/// <summary>
/// The abstraction every AI engine implements so callers (UI, match runner,
/// self-play) depend on it rather than on concrete engines (DIP / OCP): adding
/// a new engine requires no change to any caller.
/// </summary>
public interface IAiEngine
{
    string Name { get; }

    AiMoveResult ChooseMove(AiMoveRequest request);
}

public sealed record AiMoveResult(
    Move? Move,
    SearchStats Stats,
    IReadOnlyList<(Move Move, int Visits, float Q)>? MoveStats = null);

/// <summary>Which engine implementation to construct. Lives in Core so callers
/// can name a kind without referencing the neural assembly.</summary>
public enum AiEngineKind
{
    Classic,
    NeuralPolicy,
    NeuralMcts,
}
