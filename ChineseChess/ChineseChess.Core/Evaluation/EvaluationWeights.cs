using ChineseChess.Models;

namespace ChineseChess.Evaluation;

/// <summary>
/// All tunable evaluation terms in one place (no magic numbers scattered through
/// the search). Defaults reproduce the previously hard-coded behaviour.
/// </summary>
public sealed record EvaluationWeights
{
    public IReadOnlyDictionary<PieceType, int> PieceValues { get; init; } = new Dictionary<PieceType, int>
    {
        [PieceType.General] = 10000,
        [PieceType.Rook] = 900,
        [PieceType.Cannon] = 450,
        [PieceType.Horse] = 400,
        [PieceType.Elephant] = 200,
        [PieceType.Advisor] = 200,
        [PieceType.Soldier] = 120,
    };

    // Mobility per free square / capture.
    public double RookMobility { get; init; } = 2.0;
    public double CannonMobility { get; init; } = 1.6;
    public double HorseMobility { get; init; } = 3.0;
    public double SoldierMobility { get; init; } = 2.5;

    // King safety.
    public double DefenderPresence { get; init; } = 8;
    public double DefenderInPlace { get; init; } = 6;
    public double EnemyAttackerPenalty { get; init; } = 14;

    // Check tempo bonuses.
    public double GivingCheck { get; init; } = 35;
    public double BeingChecked { get; init; } = 45;

    // Pawn structure.
    public double PawnAdvance { get; init; } = 4;
    public double PawnCrossedRiver { get; init; } = 18;
    public double PawnDeep { get; init; } = 22;
    public int PawnDeepThreshold { get; init; } = 7;

    // Positional tables.
    public double SoldierAdvance { get; init; } = 6;
    public double SoldierCenter { get; init; } = 2;
    public double HorseBase { get; init; } = 18;
    public double HorseCenter { get; init; } = 4;
    public double HorseRow { get; init; } = 2;
    public double CannonBase { get; init; } = 10;
    public double CannonCenter { get; init; } = 2;
    public double RookBase { get; init; } = 8;
    public double RookCenter { get; init; } = 1;

    public static EvaluationWeights Default { get; } = new();

    public double MobilityWeight(PieceType type) => type switch
    {
        PieceType.Rook => RookMobility,
        PieceType.Cannon => CannonMobility,
        PieceType.Horse => HorseMobility,
        PieceType.Soldier => SoldierMobility,
        _ => 0.0,
    };
}