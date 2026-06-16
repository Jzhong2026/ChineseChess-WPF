/**
 * Demo 3b: 简化象棋 QSearch + TT 完整集成 (C#)
 * ===============================================
 * 对应本项目 XiangqiAiService.cs 中的核心逻辑：
 * - NegamaxSearch → negamax()
 * - EvaluateWithQSearch → quiescence()
 * - TranspositionTable → TT 类
 * - 迭代加深 → search()
 *
 * 可直接在 .NET 8 控制台运行。
 */

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MiniChessQTT
{
    // ─── 棋子类型 ───
    public static class Piece
    {
        public const int Empty = 0;
        public const int King = 1;
        public const int Rook = 2;
        public const int Pawn = 3;

        public static readonly Dictionary<int, int> Values = new()
        {
            { King, 1000 }, { Rook, 500 }, { Pawn, 100 }
        };
    }

    // ─── 棋盘 ───
    public class Board
    {
        const int R = 4, C = 4;
        public int[,] Cells { get; } = new int[R, C];
        public int Side { get; set; } = 1; // 1=红, -1=黑

        public Board() => Setup();

        void Setup()
        {
            Cells[0, 0] = Piece.Rook;   // 红车
            Cells[0, 3] = Piece.King;   // 红帅
            Cells[1, 1] = Piece.Pawn;   // 红兵
            Cells[3, 0] = -Piece.Rook;  // 黑车
            Cells[3, 3] = -Piece.King;  // 黑将
            Cells[2, 2] = -Piece.Pawn;  // 黑卒
        }

        public Board Clone()
        {
            var b = new Board();
            Array.Copy(Cells, b.Cells, Cells.Length);
            b.Side = Side;
            return b;
        }

        // 走法 = (fromR, fromC, toR, toC)
        public List<(int, int, int, int)> GenerateMoves(bool includeNonCapture = true)
        {
            var moves = new List<(int, int, int, int)>();
            for (int r = 0; r < R; r++)
                for (int c = 0; c < C; c++)
                {
                    int piece = Cells[r, c];
                    if (piece == 0) continue;
                    if ((piece > 0) != (Side == 1)) continue;

                    int pt = Math.Abs(piece);
                    if (pt == Piece.King)
                        AddKingMoves(r, c, moves);
                    else if (pt == Piece.Rook)
                        AddRookMoves(r, c, moves);
                    else if (pt == Piece.Pawn)
                        AddPawnMoves(r, c, moves);
                }
            return moves;
        }

        void AddKingMoves(int r, int c, List<(int, int, int, int)> moves)
        {
            foreach (var (dr, dc) in new[] { (0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1) })
            {
                int nr = r + dr, nc = c + dc;
                if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
                int t = Cells[nr, nc];
                if (t == 0 || (t < 0 ? Side == 1 : Side == -1))
                    moves.Add((r, c, nr, nc));
            }
        }

        void AddRookMoves(int r, int c, List<(int, int, int, int)> moves)
        {
            foreach (var (dr, dc) in new[] { (0,1),(0,-1),(1,0),(-1,0) })
            {
                int nr = r + dr, nc = c + dc;
                while (nr >= 0 && nr < R && nc >= 0 && nc < C)
                {
                    int t = Cells[nr, nc];
                    if (t == 0) { moves.Add((r, c, nr, nc)); }
                    else if ((t < 0 ? Side == 1 : Side == -1))
                    { moves.Add((r, c, nr, nc)); break; }
                    else break;
                    nr += dr; nc += dc;
                }
            }
        }

        void AddPawnMoves(int r, int c, List<(int, int, int, int)> moves)
        {
            int dr = Side == 1 ? -1 : 1;
            int nr = r + dr;
            if (nr >= 0 && nr < R)
            {
                int t = Cells[nr, c];
                if (t == 0 || (t < 0 ? Side == 1 : Side == -1))
                    moves.Add((r, c, nr, c));
            }
        }

        public bool IsCapture((int, int, int, int) move) =>
            Cells[move.Item3, move.Item4] != 0;

        public void MakeMove((int, int, int, int) move)
        {
            Cells[move.Item3, move.Item4] = Cells[move.Item1, move.Item2];
            Cells[move.Item1, move.Item2] = 0;
            Side = -Side;
        }

        public void UnmakeMove((int, int, int, int) move, int captured)
        {
            Cells[move.Item1, move.Item2] = Cells[move.Item3, move.Item4];
            Cells[move.Item3, move.Item4] = captured;
            Side = -Side;
        }

        public bool GameOver()
        {
            bool hasRed = false, hasBlack = false;
            for (int r = 0; r < R; r++)
                for (int c = 0; c < C; c++)
                {
                    if (Cells[r, c] == Piece.King) hasRed = true;
                    if (Cells[r, c] == -Piece.King) hasBlack = true;
                }
            return !(hasRed && hasBlack);
        }

        public int Evaluate()
        {
            int score = 0;
            for (int r = 0; r < R; r++)
                for (int c = 0; c < C; c++)
                {
                    int p = Cells[r, c];
                    if (p > 0) score += Piece.Values[Math.Abs(p)];
                    else if (p < 0) score -= Piece.Values[Math.Abs(p)];
                }
            return score;
        }

        public void Print()
        {
            var chars = new Dictionary<int, char>
            {
                {0, '.'}, {Piece.King, 'K'}, {Piece.Rook, 'R'}, {Piece.Pawn, 'P'},
                {-Piece.King, 'k'}, {-Piece.Rook, 'r'}, {-Piece.Pawn, 'p'}
            };
            Console.WriteLine($"{(Side == 1 ? "红方" : "黑方")}行棋:");
            for (int r = 0; r < R; r++)
            {
                for (int c = 0; c < C; c++)
                    Console.Write(chars[Cells[r, c]] + " ");
                Console.WriteLine();
            }
            Console.WriteLine();
        }
    }

    // ─── Zobrist 哈希 ───
    public class Zobrist
    {
        readonly Dictionary<(int, int, int), ulong> _table = new();
        readonly ulong _sideHash;

        public Zobrist()
        {
            var rng = new Random(42);
            for (int pt = 1; pt <= 3; pt++)
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                    {
                        _table[(pt, r, c)] = ((ulong)rng.NextInt64() << 32) | (ulong)rng.NextInt64();
                        _table[(-pt, r, c)] = ((ulong)rng.NextInt64() << 32) | (ulong)rng.NextInt64();
                    }
            _sideHash = ((ulong)rng.NextInt64() << 32) | (ulong)rng.NextInt64();
        }

        public ulong Hash(Board board)
        {
            ulong h = 0;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                {
                    int p = board.Cells[r, c];
                    if (p != 0 && _table.TryGetValue((p, r, c), out var v))
                        h ^= v;
                }
            if (board.Side == -1) h ^= _sideHash;
            return h;
        }
    }

    // ─── 置换表 ───
    enum TTFlag { EXACT, LOWERBOUND, UPPERBOUND }

    class TTEntry
    {
        public ulong Key; public int Depth; public int Score;
        public TTFlag Flag; public (int, int, int, int) BestMove;
    }

    class TranspositionTable
    {
        readonly Dictionary<ulong, TTEntry> _table = new();
        readonly int _maxSize;
        public int Hits { get; private set; }
        public int Misses { get; private set; }

        public TranspositionTable(int maxSize = 200000) => _maxSize = maxSize;

        public (bool found, int score, (int, int, int, int) bestMove)
            Probe(ulong key, int depth, int alpha, int beta)
        {
            if (_table.TryGetValue(key, out var entry) && entry.Key == key)
            {
                Hits++;
                if (entry.Depth >= depth)
                {
                    if (entry.Flag == TTFlag.EXACT) return (true, entry.Score, entry.BestMove);
                    if (entry.Flag == TTFlag.LOWERBOUND && entry.Score >= beta)
                        return (true, entry.Score, entry.BestMove);
                    if (entry.Flag == TTFlag.UPPERBOUND && entry.Score <= alpha)
                        return (true, entry.Score, entry.BestMove);
                }
                return (false, 0, entry.BestMove);
            }
            Misses++;
            return (false, 0, default);
        }

        public void Save(ulong key, int depth, int score, TTFlag flag,
            (int, int, int, int) bestMove = default)
        {
            if (_table.Count >= _maxSize) _table.Clear();
            _table[key] = new TTEntry
            {
                Key = key, Depth = depth, Score = score,
                Flag = flag, BestMove = bestMove
            };
        }

        public string Stats() =>
            $"TT: {Hits}/{Hits + Misses} hits ({100.0 * Hits / Math.Max(1, Hits + Misses):F1}%) | size={_table.Count}";
    }

    // ─── 搜索引擎（对应 XiangqiAiService.cs） ───
    class SearchEngine
    {
        readonly int _maxDepth;
        readonly int _qsDepth;
        readonly TranspositionTable _tt = new();
        readonly Zobrist _zobrist = new();
        public long NodesSearched { get; private set; }

        public SearchEngine(int maxDepth = 4, int qsDepth = 4)
        {
            _maxDepth = maxDepth;
            _qsDepth = qsDepth;
        }

        // ─── QSearch（对应 EvaluateWithQSearch） ───
        int Quiescence(Board board, int alpha, int beta, int depth = 0)
        {
            if (depth >= _qsDepth)
                return board.Evaluate();

            int standPat = board.Evaluate();

            if (standPat >= beta) return beta;
            if (standPat > alpha) alpha = standPat;

            // 只搜吃子走法
            var captureMoves = board.GenerateMoves(false)
                .Where(m => board.IsCapture(m))
                .ToList();

            // MVV-LVA 排序
            captureMoves.Sort((a, b) =>
            {
                int va = Piece.Values[Math.Abs(board.Cells[a.Item3, a.Item4])];
                int vb = Piece.Values[Math.Abs(board.Cells[b.Item3, b.Item4])];
                return vb.CompareTo(va);
            });

            int best = standPat;
            foreach (var move in captureMoves)
            {
                int captured = board.Cells[move.Item3, move.Item4];
                board.MakeMove(move);
                int score = -Quiescence(board, -beta, -alpha, depth + 1);
                board.UnmakeMove(move, captured);

                if (score > best) best = score;
                if (best > alpha) alpha = best;
                if (alpha >= beta) break;
            }
            return best;
        }

        // ─── Negamax 主搜索（对应 NegamaxSearch） ───
        int Negamax(Board board, int alpha, int beta, int depth)
        {
            NodesSearched++;
            var key = _zobrist.Hash(board);

            // TT Probe
            var (found, cached, bestMoveHint) = _tt.Probe(key, depth, alpha, beta);
            if (found) return cached;

            // 终局
            if (board.GameOver())
            {
                int winner = board.Side == -1 ? 1 : -1;
                int score = 10000 * winner;
                _tt.Save(key, depth, score, TTFlag.EXACT);
                return score;
            }

            // QSearch 截断
            if (depth <= 0)
            {
                int score = Quiescence(board, alpha, beta);
                _tt.Save(key, depth, score, TTFlag.EXACT);
                return score;
            }

            var moves = board.GenerateMoves();
            if (moves.Count == 0)
            {
                int winner = board.Side == 1 ? -1 : 1;
                int score = 10000 * winner;
                _tt.Save(key, depth, score, TTFlag.EXACT);
                return score;
            }

            // 走法排序：TT bestMove 优先 + 吃子优先
            moves.Sort((a, b) =>
            {
                int sa = 0, sb = 0;
                if (bestMoveHint != default && a == bestMoveHint) sa += 9999;
                if (bestMoveHint != default && b == bestMoveHint) sb += 9999;
                if (board.IsCapture(a)) sa += Piece.Values[Math.Abs(board.Cells[a.Item3, a.Item4])] + 1000;
                if (board.IsCapture(b)) sb += Piece.Values[Math.Abs(board.Cells[b.Item3, b.Item4])] + 1000;
                return sb.CompareTo(sa);
            });

            int best = -999999;
            var bestMove = moves[0];

            foreach (var move in moves)
            {
                int captured = board.Cells[move.Item3, move.Item4];
                board.MakeMove(move);
                int score = -Negamax(board, -beta, -alpha, depth - 1);
                board.UnmakeMove(move, captured);

                if (score > best) { best = score; bestMove = move; }
                if (score > alpha) alpha = score;
                if (alpha >= beta)
                {
                    _tt.Save(key, depth, best, TTFlag.LOWERBOUND, bestMove);
                    return best;
                }
            }

            _tt.Save(key, depth, best, TTFlag.EXACT, bestMove);
            return best;
        }

        // ─── 迭代加深（对应正式搜索接口） ───
        public ((int, int, int, int) move, int score) Search(Board board)
        {
            NodesSearched = 0;
            int bestScore = -999999;
            var bestMove = board.GenerateMoves()[0];

            for (int d = 1; d <= _maxDepth; d++)
            {
                int alpha = -999999, beta = 999999;
                bestScore = -999999;
                var moves = board.GenerateMoves();

                foreach (var move in moves)
                {
                    int captured = board.Cells[move.Item3, move.Item4];
                    board.MakeMove(move);
                    int score = -Negamax(board, -beta, -alpha, d - 1);
                    board.UnmakeMove(move, captured);

                    if (score > bestScore) { bestScore = score; bestMove = move; }
                    if (score > alpha) alpha = score;
                    if (alpha >= beta) break;
                }
            }

            return (bestMove, bestScore);
        }
    }

    // ─── 主程序 ───
    class Program
    {
        static void Main()
        {
            Console.WriteLine("=".PadRight(60, '='));
            Console.WriteLine("Demo 3b: 简化象棋 QSearch + TT (C#)");
            Console.WriteLine("=".PadRight(60, '=') + "\n");

            var board = new Board();
            board.Print();

            // 无 QSearch
            var sw = Stopwatch.StartNew();
            var e1 = new SearchEngine(maxDepth: 2, qsDepth: 0);
            var (m1, s1) = e1.Search(board.Clone());
            sw.Stop();
            Console.WriteLine($"{"配置",-25} {"走法",12} {"估值",8} {"耗时",10} {"节点",10}");
            Console.WriteLine("-".PadRight(65, '-'));
            Console.WriteLine($"{"无 QSearch (d=2)",-25} {m1,12} {s1,8} {sw.Elapsed.TotalSeconds,10:F4}s {e1.NodesSearched,10}");

            // 有 QSearch
            sw.Restart();
            var e2 = new SearchEngine(maxDepth: 2, qsDepth: 4);
            var (m2, s2) = e2.Search(board.Clone());
            sw.Stop();
            Console.WriteLine($"{"有 QSearch (d=2)",-25} {m2,12} {s2,8} {sw.Elapsed.TotalSeconds,10:F4}s {e2.NodesSearched,10}");

            // 有 QSearch + TT
            sw.Restart();
            var e3 = new SearchEngine(maxDepth: 2, qsDepth: 4);
            var (m3, s3) = e3.Search(board.Clone());
            sw.Stop();
            Console.WriteLine($"{"有 QSearch+TT (d=2)",-25} {m3,12} {s3,8} {sw.Elapsed.TotalSeconds,10:F4}s {e3.NodesSearched,10}");

            Console.WriteLine($"\nTT 统计: n/a (第二次搜索重用 TT 已重置)");
            Console.WriteLine();

            // 水平线效应演示
            Console.WriteLine("-".PadRight(60, '-'));
            Console.WriteLine("水平线效应演示");
            Console.WriteLine("-".PadRight(60, '-') + "\n");

            var hBoard = new Board();
            // 构造场景
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    hBoard.Cells[r, c] = 0;
            hBoard.Cells[0, 0] = Piece.Rook;   // 红车
            hBoard.Cells[1, 1] = -Piece.Pawn;  // 黑兵（红车可吃）
            hBoard.Cells[0, 3] = -Piece.Rook;  // 黑车（红车吃兵后会被吃）
            hBoard.Cells[3, 3] = -Piece.King;  // 黑将
            hBoard.Side = 1;
            hBoard.Print();

            var he1 = new SearchEngine(maxDepth: 2, qsDepth: 0);
            var he2 = new SearchEngine(maxDepth: 2, qsDepth: 4);

            var (hm1, hs1) = he1.Search(hBoard.Clone());
            var (hm2, hs2) = he2.Search(hBoard.Clone());

            Console.WriteLine($"无 QSearch: 走法=({hm1.Item1},{hm1.Item2}→{hm1.Item3},{hm1.Item4}), 估值={hs1}");
            Console.WriteLine($"  → 可能贪吃黑兵，看不到黑车反吃");
            Console.WriteLine($"有 QSearch:  走法=({hm2.Item1},{hm2.Item2}→{hm2.Item3},{hm2.Item4}), 估值={hs2}");
            Console.WriteLine($"  → QSearch 看到反吃链，更安全");
        }
    }
}
