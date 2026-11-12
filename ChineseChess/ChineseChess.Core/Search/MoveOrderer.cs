using ChineseChess.Evaluation;
using ChineseChess.Geometry;
using ChineseChess.Models;

namespace ChineseChess.Search;

/// <summary>
/// Move-ordering heuristics (TT best &gt; MVV-LVA &gt; killers &gt; history) plus the
/// history/killer tables they read and write. Isolated from the search so the
/// ordering policy can evolve independently (SRP / OCP).
/// </summary>
public sealed class MoveOrderer
{
    private const int MaxKillerDepth = 32;
    private const int BoardSize = XiangqiGeometry.Rows * XiangqiGeometry.Cols;

    private readonly int[] _history = new int[BoardSize * BoardSize];
    private readonly Move?[,] _killers = new Move?[MaxKillerDepth, 2];
    private readonly EvaluationWeights _weights;

    public MoveOrderer(EvaluationWeights? weights = null) => _weights = weights ?? EvaluationWeights.Default;

    /// <summary>Killers are ply-relative so reset per search; history decays softly.</summary>
    public void BeginSearch()
    {
        Array.Clear(_killers);
        for (var i = 0; i < _history.Length; i++) _history[i] /= 2;
    }

    public IEnumerable<Move> Order(Piece?[,] board, IEnumerable<Move> moves, Move? ttBestMove = null, int depth = 0)
    {
        return moves.OrderByDescending(move =>
        {
            if (ttBestMove is not null && SameMove(move, ttBestMove)) return 10_000_000.0;

            var captured = board[move.To.Row, move.To.Col];
            if (captured is not null)
                return 1_000_000.0 + _weights.PieceValues[captured.Type] - _weights.PieceValues[move.Piece.Type] / 20.0;

            if (depth < MaxKillerDepth)
            {
                if (SameMoveNullable(_killers[depth, 0], move)) return 500_000.0 - depth;
                if (SameMoveNullable(_killers[depth, 1], move)) return 490_000.0 - depth;
            }

            return _history[XiangqiGeometry.SquareIndex(move.From.Row, move.From.Col) * BoardSize
                            + XiangqiGeometry.SquareIndex(move.To.Row, move.To.Col)];
        });
    }

    /// <summary>Records a quiet beta-cutoff for history + killer heuristics.</summary>
    public void RecordCutoff(Move move, int depth, int depthFromRoot)
    {
        if (move.Captured is not null) return;
        var idx = XiangqiGeometry.SquareIndex(move.From.Row, move.From.Col) * BoardSize
                  + XiangqiGeometry.SquareIndex(move.To.Row, move.To.Col);
        _history[idx] += depth * depth;
        if (_history[idx] > 1_000_000) ScaleHistory();

        if (depthFromRoot < MaxKillerDepth && !SameMoveNullable(_killers[depthFromRoot, 0], move))
        {
            _killers[depthFromRoot, 1] = _killers[depthFromRoot, 0];
            _killers[depthFromRoot, 0] = move;
        }
    }

    private void ScaleHistory()
    {
        for (var i = 0; i < _history.Length; i++) _history[i] /= 2;
    }

    private static bool SameMove(Move left, Move right) => left.From == right.From && left.To == right.To;

    private static bool SameMoveNullable(Move? left, Move right) => left is Move m && m.From == right.From && m.To == right.To;
}