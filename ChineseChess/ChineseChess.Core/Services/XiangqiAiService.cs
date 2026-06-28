using System.Diagnostics;
using ChineseChess.Models;

namespace ChineseChess.Services;

public sealed class XiangqiAiService
{
    private readonly XiangqiEngine _engine;
    private readonly Random _random = new();
    private readonly Dictionary<int, Random> _seededRandoms = new();
    private readonly Dictionary<ulong, TranspositionEntry> _transpositionTable = new();

    // ── Transposition table flags ──────────────────────────────────────
    private const byte TT_EXACT = 0;
    private const byte TT_LOWER = 1;   // score >= beta (fail-high)
    private const byte TT_UPPER = 2;   // score <= alpha (fail-low)

    // ── Move ordering heuristics ──────────────────────────────────────
    // History table indexed by [from*90 + to]. Long-lived (decayed at the start
    // of each ChooseMove so it doesn't grow unbounded).
    private readonly int[] _historyTable = new int[BoardSize * BoardSize];
    // Killer moves: best 2 quiet moves that caused a beta cut-off at each depth.
    // Cleared at the start of each ChooseMove (depth-index changes between searches).
    private readonly Move?[,] _killerMoves = new Move?[MaxKillerDepth, 2];

    private const int BoardSize = XiangqiEngine.BoardRows * XiangqiEngine.BoardCols;  // 90
    private const int MaxKillerDepth = 32;        // covers Negamax + Quiescence

    private const ulong ZobristSeed = 0x9E3779B97F4A7C15UL;
    private const int MaxTranspositionEntries = 250_000;
    private static readonly ulong[,,,] ZobristPieces = CreateZobristPieces();
    private static readonly ulong[] ZobristTurn = { NextZobristValue(0), NextZobristValue(1) };

    private static readonly IReadOnlyDictionary<PieceType, int> PieceValues = new Dictionary<PieceType, int>
    {
        [PieceType.General] = 10000,
        [PieceType.Rook] = 900,
        [PieceType.Cannon] = 450,
        [PieceType.Horse] = 400,
        [PieceType.Elephant] = 200,
        [PieceType.Advisor] = 200,
        [PieceType.Soldier] = 120
    };

    public XiangqiAiService(XiangqiEngine engine)
    {
        _engine = engine;
    }

    public AiSearchResult ChooseMove(Piece?[,] board, Side side, int level, IReadOnlyList<Move> history, int timeLimitMs, MoveSelectionOptions? selectionOptions = null)
    {
        var watch = Stopwatch.StartNew();
        var effectiveSelectionOptions = selectionOptions ?? MoveSelectionOptions.Deterministic;
        var legalMoves = OrderMoves(board, _engine.GetLegalMoves(board, side, history), depth: 0).ToList();
        if (legalMoves.Count == 0)
        {
            return new AiSearchResult(null, new SearchStats(0, 0, 0, 0));
        }

        level = Math.Clamp(level, 1, 4);
        if (level == 1)
        {
            var selectedLevelOneMove = effectiveSelectionOptions.EnableRandomSelection
                ? legalMoves[GetRandom(effectiveSelectionOptions).Next(legalMoves.Count)]
                : legalMoves[0];
            return new AiSearchResult(selectedLevelOneMove, new SearchStats(1, legalMoves.Count, watch.Elapsed.TotalMilliseconds, 0));
        }

        if (level == 2)
        {
            return new AiSearchResult(PickBasicMove(board, legalMoves, side), new SearchStats(2, legalMoves.Count, watch.Elapsed.TotalMilliseconds, 0));
        }

        var maxDepth = level == 3 ? 4 : 8;
        var deadline = watch.ElapsedMilliseconds + Math.Max(30, timeLimitMs);
        var nodes = 0;
        var ttHits = 0;
        var ttStores = 0;
        var ttBestMoveHits = 0;
        var ttScoreHits = 0;
        var bestMove = legalMoves[0];
        var bestScore = EvaluateBoard(_engine.ApplyMove(board, bestMove), side, side);
        var bestMoveScore = bestScore;
        var depthReached = 0;

        // Don't clear the TT between moves — preserving it lets iterative-deepening
        // benefit from previous searches, and the table now stores scores+flags so
        // they're actually reusable. Soft-decay history to avoid stale domination.
        Array.Clear(_killerMoves);
        for (var i = 0; i < _historyTable.Length; i++)
        {
            _historyTable[i] = _historyTable[i] / 2;
        }

        for (var depth = 1; depth <= maxDepth; depth++)
        {
            if (depth > 1 && watch.ElapsedMilliseconds >= deadline - 10) break;
            var aborted = false;
            var iterBestMove = legalMoves[0];
            var iterBestScore = double.NegativeInfinity;
            var alpha = double.NegativeInfinity;
            var beta = double.PositiveInfinity;

            var iterRootScores = new List<(Move Move, double Score)>(legalMoves.Count);
            foreach (var move in legalMoves)
            {
                if (watch.ElapsedMilliseconds >= deadline)
                {
                    aborted = true;
                    break;
                }

                nodes++;
                var score = -Negamax(
                    _engine.ApplyMove(board, move),
                    _engine.Opposite(side),
                    depth - 1,
                    -beta,
                    -alpha,
                    _engine.Opposite(side),
                    side,
                    history.Concat(new[] { move }).ToList(),
                    watch,
                    deadline,
                    ref nodes,
                    ref aborted,
                    ref ttHits,
                    ref ttStores,
                    ref ttBestMoveHits,
                    ref ttScoreHits,
                    depthFromRoot: 1);

                if (!aborted)
                {
                    iterRootScores.Add((move, score));
                    if (score > iterBestScore)
                    {
                        iterBestScore = score;
                        iterBestMove = move;
                    }
                }

                if (!aborted) alpha = Math.Max(alpha, score);
            }

            if (!aborted)
            {
                bestMove = SelectMove(iterRootScores, iterBestMove, iterBestScore, effectiveSelectionOptions);
                bestScore = iterBestScore;
                bestMoveScore = iterRootScores.First(item => SameMove(item.Move, bestMove)).Score;
                depthReached = depth;
            }
        }

        return new AiSearchResult(bestMove, new SearchStats(Math.Max(depthReached, 0), nodes, watch.Elapsed.TotalMilliseconds, bestMoveScore, ttHits, ttStores, ttBestMoveHits, ttScoreHits));
    }

    private Move SelectMove(IReadOnlyList<(Move Move, double Score)> scoredMoves, Move fallbackMove, double bestScore, MoveSelectionOptions options)
    {
        if (scoredMoves.Count == 0) return fallbackMove;
        if (!options.EnableRandomSelection) return fallbackMove;

        var sorted = scoredMoves.OrderByDescending(item => item.Score).ToList();
        var topK = Math.Clamp(options.TopK, 1, sorted.Count);
        var nearBestWindow = Math.Max(0, options.NearBestWindow);
        var minTopKScore = sorted[topK - 1].Score;
        var threshold = Math.Max(bestScore - nearBestWindow, minTopKScore);
        var candidateMoves = sorted.Where(item => item.Score >= threshold).Select(item => item.Move).ToList();

        if (candidateMoves.Count == 0) return fallbackMove;
        return candidateMoves[GetRandom(options).Next(candidateMoves.Count)];
    }

    private Random GetRandom(MoveSelectionOptions options)
    {
        if (!options.Seed.HasValue) return _random;
        if (_seededRandoms.TryGetValue(options.Seed.Value, out var seededRandom)) return seededRandom;

        seededRandom = new Random(options.Seed.Value);
        _seededRandoms[options.Seed.Value] = seededRandom;
        return seededRandom;
    }

    private double Negamax(
        Piece?[,] board,
        Side turn,
        int depth,
        double alpha,
        double beta,
        Side perspective,
        Side aiSide,
        IReadOnlyList<Move> history,
        Stopwatch watch,
        long deadline,
        ref int nodes,
        ref bool aborted,
        ref int ttHits,
        ref int ttStores,
        ref int ttBestMoveHits,
        ref int ttScoreHits,
        int depthFromRoot)
    {
        var alphaOrig = alpha;  // capture entry alpha for correct TT flag at exit

        if ((nodes & 0x3f) == 0 && watch.ElapsedMilliseconds >= deadline) aborted = true;
        if (aborted) return 0;

        var hash = ComputeZobristHash(board, turn);
        Move? ttBestMove = null;
        if (_transpositionTable.TryGetValue(hash, out var entry))
        {
            ttHits++;
            ttBestMove = entry.BestMove;
            // TT score cutoff: only trust entries searched at sufficient depth.
            // NOTE: Score reuse is correct only when the entry's depth bound
            // covers what we'd recompute — i.e. entry.Depth >= remaining depth.
            if (entry.Depth >= depth)
            {
                if (entry.Flag == TT_EXACT)
                {
                    ttScoreHits++;
                    return entry.Score;
                }
                if (entry.Flag == TT_LOWER && entry.Score >= beta)
                {
                    ttScoreHits++;
                    return entry.Score;
                }
                if (entry.Flag == TT_UPPER && entry.Score <= alpha)
                {
                    ttScoreHits++;
                    return entry.Score;
                }
            }
        }

        var status = _engine.GetGameStatus(board, turn);
        if (status == GameStatus.RedWins) return perspective == Side.Red ? 999999 - depthFromRoot : -999999 + depthFromRoot;
        if (status == GameStatus.BlackWins) return perspective == Side.Black ? 999999 - depthFromRoot : -999999 + depthFromRoot;
        if (status == GameStatus.Draw) return 0;
        if (depth <= 0) return Quiescence(board, turn, alpha, beta, perspective, aiSide, history, watch, deadline, ref nodes, ref aborted, depthFromRoot, 0);

        var best = double.NegativeInfinity;
        Move? bestMove = null;
        var moves = OrderMoves(board, _engine.GetLegalMoves(board, turn, history), ttBestMove, depthFromRoot);
        foreach (var move in moves)
        {
            if (aborted) break;
            nodes++;
            var score = -Negamax(
                _engine.ApplyMove(board, move),
                _engine.Opposite(turn),
                depth - 1,
                -beta,
                -alpha,
                _engine.Opposite(perspective),
                aiSide,
                history.Concat(new[] { move }).ToList(),
                watch,
                deadline,
                ref nodes,
                ref aborted,
                ref ttHits,
                ref ttStores,
                ref ttBestMoveHits,
                ref ttScoreHits,
                depthFromRoot + 1);

            if (score > best)
            {
                best = score;
                bestMove = move;
            }

            alpha = Math.Max(alpha, score);
            if (alpha >= beta)
            {
                // Beta cut-off: update history + killer (quiet moves only).
                if (move.Captured is null)
                {
                    var fromIdx = move.From.Row * XiangqiEngine.BoardCols + move.From.Col;
                    var toIdx = move.To.Row * XiangqiEngine.BoardCols + move.To.Col;
                    _historyTable[fromIdx * BoardSize + toIdx] += depth * depth;
                    if (_historyTable[fromIdx * BoardSize + toIdx] > 1_000_000) ScaleHistory();

                    // Slot 0 = most recent killer; shift previous killer to slot 1.
                    if (depthFromRoot < MaxKillerDepth && !SameMoveNullable(_killerMoves[depthFromRoot, 0], move))
                    {
                        _killerMoves[depthFromRoot, 1] = _killerMoves[depthFromRoot, 0];
                        _killerMoves[depthFromRoot, 0] = move;
                    }
                }
                break;
            }
        }

        if (!aborted && best > double.NegativeInfinity)
        {
            byte flag;
            if (best <= alphaOrig) flag = TT_UPPER;
            else if (best >= beta) flag = TT_LOWER;
            else flag = TT_EXACT;
            StoreTransposition(hash, new TranspositionEntry(depth, best, flag, bestMove));
            ttStores++;
            if (ttBestMove is not null && bestMove is not null && SameMove(ttBestMove, bestMove)) ttBestMoveHits++;
        }

        return best;
    }

    private double Quiescence(
        Piece?[,] board,
        Side turn,
        double alpha,
        double beta,
        Side perspective,
        Side aiSide,
        IReadOnlyList<Move> history,
        Stopwatch watch,
        long deadline,
        ref int nodes,
        ref bool aborted,
        int depthFromRoot,
        int qDepth)
    {
        var standPat = EvaluateBoard(board, perspective, aiSide);
        if (standPat >= beta) return beta;
        if (alpha < standPat) alpha = standPat;
        if (qDepth >= 8) return alpha;

        var tacticalMoves = OrderMoves(board, _engine.GetLegalMoves(board, turn, history).Where(move => move.Captured is not null), depth: depthFromRoot);
        foreach (var move in tacticalMoves)
        {
            if ((nodes & 0x3f) == 0 && watch.ElapsedMilliseconds >= deadline) aborted = true;
            if (aborted) break;
            nodes++;
            var score = -Quiescence(
                _engine.ApplyMove(board, move),
                _engine.Opposite(turn),
                -beta,
                -alpha,
                _engine.Opposite(perspective),
                aiSide,
                history.Concat(new[] { move }).ToList(),
                watch,
                deadline,
                ref nodes,
                ref aborted,
                depthFromRoot,
                qDepth + 1);

            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }

        return alpha;
    }

    private Move PickBasicMove(Piece?[,] board, IReadOnlyList<Move> moves, Side side)
    {
        var scoredMoves = moves.Select(move =>
        {
            var next = _engine.ApplyMove(board, move);
            var captureValue = move.Captured is null ? 0 : PieceValues[move.Captured.Type];
            var givesCheck = _engine.IsInCheck(next, _engine.Opposite(side)) ? 50 : 0;
            var leavesSelfInCheck = _engine.IsInCheck(next, side) ? -5000 : 0;
            return (Move: move, Score: captureValue * 10 + givesCheck + leavesSelfInCheck);
        }).ToList();

        var topScore = scoredMoves.Max(item => item.Score);
        var bestMoves = scoredMoves.Where(item => item.Score == topScore).Select(item => item.Move).ToList();
        return bestMoves[_random.Next(bestMoves.Count)];
    }

    private IEnumerable<Move> OrderMoves(Piece?[,] board, IEnumerable<Move> moves, Move? ttBestMove = null, int depth = 0)
    {
        return moves.OrderByDescending(move =>
        {
            if (ttBestMove is not null && SameMove(move, ttBestMove)) return 10_000_000.0;

            var captured = board[move.To.Row, move.To.Col];
            if (captured is not null)
            {
                // MVV-LVA: most-valuable victim, least-valuable attacker.
                return 1_000_000.0 + PieceValues[captured.Type] - PieceValues[move.Piece.Type] / 20.0;
            }

            // Killer move heuristic (quiet moves only).
            if (depth < MaxKillerDepth)
            {
                if (SameMoveNullable(_killerMoves[depth, 0], move)) return 500_000.0 - depth;
                if (SameMoveNullable(_killerMoves[depth, 1], move)) return 490_000.0 - depth;
            }

            // History heuristic.
            var fromIdx = move.From.Row * XiangqiEngine.BoardCols + move.From.Col;
            var toIdx = move.To.Row * XiangqiEngine.BoardCols + move.To.Col;
            return _historyTable[fromIdx * BoardSize + toIdx];
        });
    }

    private void ScaleHistory()
    {
        for (var i = 0; i < _historyTable.Length; i++)
        {
            _historyTable[i] = _historyTable[i] / 2;
        }
    }

    private void StoreTransposition(ulong hash, TranspositionEntry entry)
    {
        if (_transpositionTable.Count >= MaxTranspositionEntries) _transpositionTable.Clear();
        if (!_transpositionTable.TryGetValue(hash, out var current) || entry.Depth >= current.Depth)
        {
            _transpositionTable[hash] = entry;
        }
    }

    private static bool SameMove(Move left, Move right) => left.From == right.From && left.To == right.To;

    private static bool SameMoveNullable(Move? left, Move right) => left is Move m && m.From == right.From && m.To == right.To;

    private static ulong ComputeZobristHash(Piece?[,] board, Side turn)
    {
        var hash = ZobristTurn[(int)turn];
        for (var row = 0; row < XiangqiEngine.BoardRows; row++)
        {
            for (var col = 0; col < XiangqiEngine.BoardCols; col++)
            {
                var piece = board[row, col];
                if (piece is null) continue;
                hash ^= ZobristPieces[row, col, (int)piece.Side, (int)piece.Type];
            }
        }

        return hash;
    }

    private static ulong[,,,] CreateZobristPieces()
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
                        values[row, col, side, type] = NextZobristValue(index++);
                    }
                }
            }
        }

        return values;
    }

    private static ulong NextZobristValue(int index)
    {
        var value = ZobristSeed + (ulong)index * 0x9E3779B97F4A7C15UL;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
        return value ^ (value >> 31);
    }

    private double EvaluateBoard(Piece?[,] board, Side perspective, Side aiSide)
    {
        var material = 0.0;
        var positional = 0.0;
        var mobility = 0.0;
        var kingSafety = 0.0;
        var pawnStructure = 0.0;
        var myKingRow = -1;
        var myKingCol = -1;
        var myAdvisorCount = 0;
        var myElephantCount = 0;
        var myAdvisorInPlace = 0;     // advisors on their correct diagonal slots
        var myElephantInPlace = 0;    // elephants on their correct diagonal slots

        for (var row = 0; row < XiangqiEngine.BoardRows; row++)
        {
            for (var col = 0; col < XiangqiEngine.BoardCols; col++)
            {
                var piece = board[row, col];
                if (piece is null) continue;

                var isMine = piece.Side == perspective;
                var sideSign = isMine ? 1 : -1;

                // Material + positional
                material += sideSign * PieceValues[piece.Type];
                positional += sideSign * PositionalBonus(piece.Type, piece.Side, row, col);

                // Mobility (cheap pseudo-mobility, no check/repetition filtering)
                mobility += sideSign * MobilityWeight(piece.Type) * CountMobility(board, row, col, piece);

                // Pawn structure (only meaningful for soldiers)
                if (piece.Type == PieceType.Soldier)
                {
                    pawnStructure += sideSign * PawnStructureScore(piece.Side, row, col);
                }

                // King-safety data collection (for my king only)
                if (isMine)
                {
                    if (piece.Type == PieceType.General)
                    {
                        myKingRow = row;
                        myKingCol = col;
                    }
                    else if (piece.Type == PieceType.Advisor)
                    {
                        myAdvisorCount++;
                        if (IsAdvisorSlot(piece.Side, row, col)) myAdvisorInPlace++;
                    }
                    else if (piece.Type == PieceType.Elephant)
                    {
                        myElephantCount++;
                        if (IsElephantSlot(piece.Side, row, col)) myElephantInPlace++;
                    }
                }
            }
        }

        // King safety: defenders on the board + in correct positions; opponent attackers.
        if (myKingRow >= 0)
        {
            kingSafety += (myAdvisorCount + myElephantCount) * 8;       // piece-presence bonus
            kingSafety += (myAdvisorInPlace + myElephantInPlace) * 6;  // correctly-positioned bonus

            // Penalty for enemy pieces attacking the king square (or adjacent squares).
            var enemyAttackers = CountAttackersNearKing(board, myKingRow, myKingCol, _engine.Opposite(perspective));
            kingSafety -= enemyAttackers * 14;
        }

        // Existing check bonuses (kept for tactical clarity).
        if (_engine.IsInCheck(board, _engine.Opposite(aiSide))) material += perspective == aiSide ? 35 : -35;
        if (_engine.IsInCheck(board, aiSide)) material += perspective == aiSide ? -45 : 45;

        return material + positional + mobility + kingSafety + pawnStructure;
    }

    // ── Evaluation helpers ──────────────────────────────────────────────

    private static double MobilityWeight(PieceType type) => type switch
    {
        PieceType.Rook => 2.0,
        PieceType.Cannon => 1.6,
        PieceType.Horse => 3.0,
        PieceType.Soldier => 2.5,
        _ => 0.0,  // General / Advisor / Elephant: static, skip
    };

    private static int CountMobility(Piece?[,] board, int r, int c, Piece piece)
    {
        return piece.Type switch
        {
            PieceType.Rook => CountLineMoves(board, r, c, piece, RookDirections),
            PieceType.Cannon => CountCannonMoves(board, r, c, piece),
            PieceType.Horse => CountHorseMoves(board, r, c, piece),
            PieceType.Soldier => CountSoldierMoves(board, r, c, piece),
            _ => 0,
        };
    }

    private static readonly (int dr, int dc)[] RookDirections = { (-1, 0), (1, 0), (0, -1), (0, 1) };

    private static int CountLineMoves(Piece?[,] board, int r, int c, Piece piece, (int dr, int dc)[] dirs)
    {
        var count = 0;
        foreach (var (dr, dc) in dirs)
        {
            var nr = r + dr;
            var nc = c + dc;
            while (nr >= 0 && nr < XiangqiEngine.BoardRows && nc >= 0 && nc < XiangqiEngine.BoardCols)
            {
                var target = board[nr, nc];
                if (target is null)
                {
                    count++;
                }
                else
                {
                    if (target.Side != piece.Side) count++;  // capture
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
        foreach (var (dr, dc) in RookDirections)
        {
            var nr = r + dr;
            var nc = c + dc;
            var seenScreen = false;
            while (nr >= 0 && nr < XiangqiEngine.BoardRows && nc >= 0 && nc < XiangqiEngine.BoardCols)
            {
                var target = board[nr, nc];
                if (!seenScreen)
                {
                    if (target is null)
                    {
                        count++;  // empty-square slide
                    }
                    else
                    {
                        seenScreen = true;  // screen piece found
                    }
                }
                else if (target is not null)
                {
                    if (target.Side != piece.Side) count++;  // jump-capture
                    break;
                }
                nr += dr;
                nc += dc;
            }
        }
        return count;
    }

    private static readonly (int dr, int dc, int lr, int lc)[] HorseMoves =
    {
        (-2, -1, -1, 0), (-2,  1, -1, 0),
        ( 2, -1,  1, 0), ( 2,  1,  1, 0),
        (-1, -2,  0, -1), ( 1, -2,  0, -1),
        (-1,  2,  0,  1), ( 1,  2,  0,  1),
    };

    private static int CountHorseMoves(Piece?[,] board, int r, int c, Piece piece)
    {
        var count = 0;
        foreach (var (dr, dc, lr, lc) in HorseMoves)
        {
            var legR = r + lr;
            var legC = c + lc;
            if (legR < 0 || legR >= XiangqiEngine.BoardRows || legC < 0 || legC >= XiangqiEngine.BoardCols) continue;
            if (board[legR, legC] is not null) continue;  // leg blocked
            var nr = r + dr;
            var nc = c + dc;
            if (nr < 0 || nr >= XiangqiEngine.BoardRows || nc < 0 || nc >= XiangqiEngine.BoardCols) continue;
            count++;  // horse always lands on empty or enemy square; both are "mobile"
        }
        return count;
    }

    private static int CountSoldierMoves(Piece?[,] board, int r, int c, Piece piece)
    {
        var count = 0;
        var forward = piece.Side == Side.Red ? -1 : 1;
        var fr = r + forward;
        if (fr >= 0 && fr < XiangqiEngine.BoardRows)
        {
            var p = board[fr, c];
            if (p is null || p.Side != piece.Side) count++;
        }
        var crossedRiver = piece.Side == Side.Red ? r <= 4 : r >= 5;
        if (crossedRiver)
        {
            if (c > 0)
            {
                var p = board[r, c - 1];
                if (p is null || p.Side != piece.Side) count++;
            }
            if (c < XiangqiEngine.BoardCols - 1)
            {
                var p = board[r, c + 1];
                if (p is null || p.Side != piece.Side) count++;
            }
        }
        return count;
    }

    private static double PawnStructureScore(Side side, int row, int col)
    {
        // Red advances from row 9 toward 0; Black from row 0 toward 9.
        var forwardProgress = side == Side.Red ? 9 - row : row;
        var crossedRiver = side == Side.Red ? row <= 4 : row >= 5;
        var score = forwardProgress * 4.0;  // base advancement reward

        if (crossedRiver)
        {
            score += 18;                              // crossed-river flat bonus
            if (forwardProgress >= 7) score += 22;   // deep into enemy territory / near palace
        }
        return score;
    }

    private static bool IsAdvisorSlot(Side side, int row, int col)
    {
        // Advisor legal slots: (3,3), (3,5), (5,3), (5,5) for Red; mirrored for Black.
        if (col != 3 && col != 5) return false;
        return side == Side.Red ? (row == 7 || row == 9) : (row == 0 || row == 2);
    }

    private static bool IsElephantSlot(Side side, int row, int col)
    {
        // Elephant legal slots: (0,2), (0,6), (2,4), (4,2), (4,6), (6,4) etc. — keep it simple:
        // elephants always start on row 9 (Red) / row 0 (Black) at cols 2 and 6.
        if (col != 2 && col != 6) return false;
        return side == Side.Red ? (row == 9 || row == 7 || row == 5) : (row == 0 || row == 2 || row == 4);
    }

    private static int CountAttackersNearKing(Piece?[,] board, int kingRow, int kingCol, Side enemySide)
    {
        // Count distinct enemy pieces that can reach the 3x3 area around the king.
        var attackers = 0;
        for (var row = 0; row < XiangqiEngine.BoardRows; row++)
        {
            for (var col = 0; col < XiangqiEngine.BoardCols; col++)
            {
                var piece = board[row, col];
                if (piece is null || piece.Side != enemySide) continue;
                if (piece.Type == PieceType.General) continue;  // don't count opposing kings
                if (CanReachArea(board, row, col, piece, kingRow, kingCol)) attackers++;
            }
        }
        return attackers;
    }

    private static bool CanReachArea(Piece?[,] board, int r, int c, Piece piece, int kingRow, int kingCol)
    {
        // The "area" around the king is the 3x3 block centered on (kingRow, kingCol).
        // We treat it as "reachable" if any of the king's adjacent squares is a legal move target.
        for (var dr = -1; dr <= 1; dr++)
        {
            for (var dc = -1; dc <= 1; dc++)
            {
                if (dr == 0 && dc == 0) continue;
                var tr = kingRow + dr;
                var tc = kingCol + dc;
                if (tr < 0 || tr >= XiangqiEngine.BoardRows || tc < 0 || tc >= XiangqiEngine.BoardCols) continue;
                if (CanPieceAttackSquare(board, r, c, piece, tr, tc)) return true;
            }
        }
        return false;
    }

    private static bool CanPieceAttackSquare(Piece?[,] board, int r, int c, Piece piece, int tr, int tc)
    {
        return piece.Type switch
        {
            PieceType.Rook => LineAttacksSquare(board, r, c, piece, tr, tc, RookDirections),
            PieceType.Cannon => CannonAttacksSquare(board, r, c, piece, tr, tc),
            PieceType.Horse => HorseAttacksSquare(board, r, c, tr, tc),
            PieceType.Soldier => SoldierAttacksSquare(piece.Side, r, c, tr, tc),
            _ => false,
        };
    }

    private static bool LineAttacksSquare(Piece?[,] board, int r, int c, Piece piece, int tr, int tc, (int dr, int dc)[] dirs)
    {
        foreach (var (dr, dc) in dirs)
        {
            var nr = r + dr;
            var nc = c + dc;
            while (nr >= 0 && nr < XiangqiEngine.BoardRows && nc >= 0 && nc < XiangqiEngine.BoardCols)
            {
                if (nr == tr && nc == tc) return true;
                if (board[nr, nc] is not null) break;
                nr += dr;
                nc += dc;
            }
        }
        return false;
    }

    private static bool CannonAttacksSquare(Piece?[,] board, int r, int c, Piece piece, int tr, int tc)
    {
        foreach (var (dr, dc) in RookDirections)
        {
            var nr = r + dr;
            var nc = c + dc;
            var seenScreen = false;
            while (nr >= 0 && nr < XiangqiEngine.BoardRows && nc >= 0 && nc < XiangqiEngine.BoardCols)
            {
                if (nr == tr && nc == tc)
                {
                    // Cannon reaches this square only if it has hopped exactly one screen piece.
                    return seenScreen;
                }
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
        foreach (var (dr, dc, lr, lc) in HorseMoves)
        {
            if (r + dr == tr && c + dc == tc)
            {
                // Verify the horse's leg is not blocked.
                var legR = r + lr;
                var legC = c + lc;
                if (legR >= 0 && legR < XiangqiEngine.BoardRows && legC >= 0 && legC < XiangqiEngine.BoardCols)
                {
                    if (board[legR, legC] is null) return true;
                }
                return false;
            }
        }
        return false;
    }

    private static bool SoldierAttacksSquare(Side side, int r, int c, int tr, int tc)
    {
        var forward = side == Side.Red ? -1 : 1;
        if (tr == r + forward && tc == c) return true;
        var crossed = side == Side.Red ? r <= 4 : r >= 5;
        if (crossed && tr == r && (tc == c - 1 || tc == c + 1)) return true;
        return false;
    }

    private static double PositionalBonus(PieceType type, Side side, int row, int col)
    {
        var forwardProgress = side == Side.Red ? 9 - row : row;
        var centerDistance = Math.Abs(col - 4);
        return type switch
        {
            PieceType.Soldier => forwardProgress * 6 - centerDistance * 2,  // reduced: pawn-structure term now handles advancement
            PieceType.Horse => 18 - centerDistance * 4 - Math.Abs(row - 4.5) * 2,
            PieceType.Cannon => 10 - centerDistance * 2,
            PieceType.Rook => 8 - centerDistance,
            _ => 0
        };
    }

    private sealed record TranspositionEntry(int Depth, double Score, byte Flag, Move? BestMove);
}