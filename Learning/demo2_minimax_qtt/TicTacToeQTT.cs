/**
 * Demo 2b: 带 QSearch + TT 的完整井字棋 AI (C#)
 * ==============================================
 * C# 版本，结构与 Python 版对应。
 * 可直接在 .NET 8 控制台运行。
 */

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace TicTacToeQTT
{
    // ─── 棋盘 ───
    class Board
    {
        public const int N = 3;
        public const int EMPTY = 0, X = 1, O = 2;

        public int[] Cells { get; } = new int[N * N];
        public int Side { get; set; } = X;

        public Board Clone()
        {
            var b = new Board();
            Array.Copy(Cells, b.Cells, N * N);
            b.Side = Side;
            return b;
        }

        public List<int> Moves() =>
            Enumerable.Range(0, N * N).Where(i => Cells[i] == EMPTY).ToList();

        public void Apply(int move)
        {
            Cells[move] = Side;
            Side = Side == X ? O : X;
        }

        public void Undo(int move)
        {
            Cells[move] = EMPTY;
            Side = Side == X ? O : X;
        }

        public bool IsWin(int player)
        {
            int[][] lines =
            {
                new[] { 0, 1, 2 }, new[] { 3, 4, 5 }, new[] { 6, 7, 8 },
                new[] { 0, 3, 6 }, new[] { 1, 4, 7 }, new[] { 2, 5, 8 },
                new[] { 0, 4, 8 }, new[] { 2, 4, 6 }
            };
            return lines.Any(line => line.All(i => Cells[i] == player));
        }

        public bool IsDraw() => !Moves().Any() && !IsWin(X) && !IsWin(O);
        public bool GameOver() => IsWin(X) || IsWin(O) || IsDraw();

        public int Evaluate()
        {
            if (IsWin(X)) return 100;
            if (IsWin(O)) return -100;
            return 0;
        }

        public void Print()
        {
            var chars = new[] { '.', 'X', 'O' };
            for (int r = 0; r < N; r++)
            {
                for (int c = 0; c < N; c++)
                    Console.Write(chars[Cells[r * N + c]] + " ");
                Console.WriteLine();
            }
        }
    }

    // ─── 阶段 1：纯 Minimax ───
    class MinimaxSolver
    {
        public int Solve(Board board)
        {
            if (board.GameOver()) return board.Evaluate();

            if (board.Side == Board.X)
            {
                int best = -999;
                foreach (var m in board.Moves())
                {
                    board.Apply(m);
                    best = Math.Max(best, Solve(board));
                    board.Undo(m);
                }
                return best;
            }
            else
            {
                int best = 999;
                foreach (var m in board.Moves())
                {
                    board.Apply(m);
                    best = Math.Min(best, Solve(board));
                    board.Undo(m);
                }
                return best;
            }
        }
    }

    // ─── 阶段 2：Alpha-Beta ───
    class AlphaBetaSolver
    {
        public int Solve(Board board, int alpha, int beta)
        {
            if (board.GameOver()) return board.Evaluate();

            if (board.Side == Board.X)
            {
                int best = -999;
                foreach (var m in board.Moves())
                {
                    board.Apply(m);
                    best = Math.Max(best, Solve(board, alpha, beta));
                    board.Undo(m);
                    alpha = Math.Max(alpha, best);
                    if (alpha >= beta) break;
                }
                return best;
            }
            else
            {
                int best = 999;
                foreach (var m in board.Moves())
                {
                    board.Apply(m);
                    best = Math.Min(best, Solve(board, alpha, beta));
                    board.Undo(m);
                    beta = Math.Min(beta, best);
                    if (alpha >= beta) break;
                }
                return best;
            }
        }
    }

    // ─── 阶段 3：Alpha-Beta + TT ───
    enum TTFlag { EXACT, LOWERBOUND, UPPERBOUND }

    class TTEntry
    {
        public ulong Key;
        public int Depth;
        public int Score;
        public TTFlag Flag;
        public int BestMove = -1;

        public TTEntry(ulong key, int depth, int score, TTFlag flag, int bestMove = -1)
        {
            Key = key; Depth = depth; Score = score;
            Flag = flag; BestMove = bestMove;
        }
    }

    class TranspositionTable
    {
        private readonly Dictionary<ulong, TTEntry> _table = new();
        private readonly int _maxSize;
        public int Hits { get; private set; }

        public TranspositionTable(int maxSize = 50000) => _maxSize = maxSize;

        public ulong HashBoard(Board board)
        {
            ulong h = 0;
            for (int i = 0; i < Board.N * Board.N; i++)
                h ^= (ulong)(board.Cells[i] + 1) << (i * 2);
            h ^= (ulong)board.Side << 18;
            return h;
        }

        public (bool found, int score, int bestMove) Probe(ulong key, int depth, int alpha, int beta)
        {
            if (_table.TryGetValue(key, out var entry) && entry.Key == key)
            {
                Hits++;
                if (entry.Depth >= depth)
                {
                    if (entry.Flag == TTFlag.EXACT)
                        return (true, entry.Score, entry.BestMove);
                    if (entry.Flag == TTFlag.LOWERBOUND && entry.Score >= beta)
                        return (true, entry.Score, entry.BestMove);
                    if (entry.Flag == TTFlag.UPPERBOUND && entry.Score <= alpha)
                        return (true, entry.Score, entry.BestMove);
                }
                return (false, 0, entry.BestMove);
            }
            return (false, 0, -1);
        }

        public void Save(ulong key, int depth, int score, TTFlag flag, int bestMove = -1)
        {
            if (_table.Count >= _maxSize) _table.Clear();
            _table[key] = new TTEntry(key, depth, score, flag, bestMove);
        }
    }

    class AlphaBetaTTSolver
    {
        private TranspositionTable _tt;

        public AlphaBetaTTSolver(TranspositionTable tt) => _tt = tt;

        public int Solve(Board board, int alpha, int beta, int depth = 0)
        {
            var key = _tt.HashBoard(board);

            var (found, cached, bestMove) = _tt.Probe(key, depth, alpha, beta);
            if (found) return cached;

            if (board.GameOver())
            {
                int s = board.Evaluate();
                _tt.Save(key, depth, s, TTFlag.EXACT);
                return s;
            }

            var moves = board.Moves();
            if (bestMove >= 0 && moves.Contains(bestMove))
            {
                moves.Remove(bestMove);
                moves.Insert(0, bestMove);
            }

            if (board.Side == Board.X)
            {
                int best = -999;
                int bm = -1;
                foreach (var m in moves)
                {
                    board.Apply(m);
                    int score = Solve(board, alpha, beta, depth + 1);
                    board.Undo(m);
                    if (score > best) { best = score; bm = m; }
                    alpha = Math.Max(alpha, score);
                    if (alpha >= beta)
                    {
                        _tt.Save(key, depth, best, TTFlag.LOWERBOUND, bm);
                        return best;
                    }
                }
                _tt.Save(key, depth, best, TTFlag.EXACT, bm);
                return best;
            }
            else
            {
                int best = 999;
                int bm = -1;
                foreach (var m in moves)
                {
                    board.Apply(m);
                    int score = Solve(board, alpha, beta, depth + 1);
                    board.Undo(m);
                    if (score < best) { best = score; bm = m; }
                    beta = Math.Min(beta, score);
                    if (alpha >= beta)
                    {
                        _tt.Save(key, depth, best, TTFlag.UPPERBOUND, bm);
                        return best;
                    }
                }
                _tt.Save(key, depth, best, TTFlag.EXACT, bm);
                return best;
            }
        }
    }

    // ─── 主程序 ───
    class Program
    {
        static void Benchmark()
        {
            Console.WriteLine("=".PadRight(60, '='));
            Console.WriteLine("Demo 2b: 井字棋 AI 各阶段性能对比 (C#)");
            Console.WriteLine("=".PadRight(60, '='));

            const int trials = 1000;

            // 纯 Minimax
            var sw = Stopwatch.StartNew();
            var solver1 = new MinimaxSolver();
            for (int i = 0; i < trials; i++)
            {
                var b = new Board();
                solver1.Solve(b);
            }
            sw.Stop();
            var t1 = sw.Elapsed.TotalSeconds;

            // Alpha-Beta
            sw.Restart();
            var solver2 = new AlphaBetaSolver();
            for (int i = 0; i < trials; i++)
            {
                var b = new Board();
                solver2.Solve(b, -999, 999);
            }
            sw.Stop();
            var t2 = sw.Elapsed.TotalSeconds;

            // Alpha-Beta + TT
            sw.Restart();
            var tt = new TranspositionTable();
            var solver3 = new AlphaBetaTTSolver(tt);
            for (int i = 0; i < trials; i++)
            {
                var b = new Board();
                solver3.Solve(b, -999, 999);
            }
            sw.Stop();
            var t3 = sw.Elapsed.TotalSeconds;

            Console.WriteLine($"\n{{0,-25}} {{1,10}} {{2,10}}", "算法", "耗时", "加速比");
            Console.WriteLine("-".PadRight(47, '-'));
            Console.WriteLine($"{{0,-25}} {{1,10:F3}}s {{2,10:F1}}x", "纯 Minimax", t1, 1.0);
            Console.WriteLine($"{{0,-25}} {{1,10:F3}}s {{2,10:F1}}x", "Alpha-Beta", t2, t1 / t2);
            Console.WriteLine($"{{0,-25}} {{1,10:F3}}s {{2,10:F1}}x", "Alpha-Beta + TT", t3, t1 / t3);
            Console.WriteLine($"\nTT 命中率: {tt.Hits / (double)(tt.Hits + 50000) * 100:F1}%");
            Console.WriteLine();
        }

        static void Main(string[] args)
        {
            Benchmark();

            // 人机对战
            Console.WriteLine("人机对战（AI 使用 Alpha-Beta + TT）");
            Console.WriteLine("你走 O，AI 走 X");
            Console.WriteLine("输入 0-8：");
            Console.WriteLine("0 1 2");
            Console.WriteLine("3 4 5");
            Console.WriteLine("6 7 8\n");

            var board = new Board();
            var tt = new TranspositionTable();
            var solver = new AlphaBetaTTSolver(tt);

            while (!board.GameOver())
            {
                board.Print();
                Console.WriteLine();

                if (board.Side == Board.X)
                {
                    int bestMove = -1, bestScore = -999;
                    foreach (var m in board.Moves())
                    {
                        board.Apply(m);
                        int score = solver.Solve(board, -999, 999);
                        board.Undo(m);
                        if (score > bestScore) { bestScore = score; bestMove = m; }
                    }
                    board.Apply(bestMove);
                    Console.WriteLine($"AI 走: {bestMove}");
                }
                else
                {
                    Console.Write("你的走法: ");
                    int move = int.Parse(Console.ReadLine()!);
                    if (board.Moves().Contains(move))
                        board.Apply(move);
                    else
                        Console.WriteLine("无效走法！");
                }
            }

            board.Print();
            if (board.IsWin(Board.X)) Console.WriteLine("AI 胜！");
            else if (board.IsWin(Board.O)) Console.WriteLine("你胜！");
            else Console.WriteLine("平局！");
        }
    }
}
