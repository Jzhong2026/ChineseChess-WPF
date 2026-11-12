using System.Diagnostics;
using ChineseChess.Encoding;
using ChineseChess.Models;
using ChineseChess.Services;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ChineseChess.Ai.Neural;

/// <summary>
/// Monte Carlo Tree Search (MCTS) combined with a neural network policy-value network.
///
/// Algorithm: PUCT (Polynomial Upper Confidence Trees) — same as AlphaGo/AlphaZero.
///
/// Each node stores:
///   N  = visit count
///   W  = total action value (sum of value estimates from rollouts through this node)
///   Q  = mean action value = W / N
///   P  = prior probability from NN policy
///
/// Selection:  argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a)) ]
/// Expansion:  add all legal children, assign P from NN policy
/// Evaluation: use NN value head (no rollout needed)
/// Backup:     propagate value up the tree
///
/// Move selection: proportional to visit count^(1/temperature)
///   temperature=1 → exploration; temperature→0 → greedy (exploit best move)
/// </summary>
public sealed class MctsAiService
{
    private readonly XiangqiNeuralAiService _neural;
    private readonly XiangqiEngine _engine;

    // PUCT exploration constant (higher = more exploration)
    private const float CPuct = 1.5f;

    // Dirichlet noise for root node exploration (AlphaZero style)
    private const float DirichletAlpha = 0.3f;
    private const float DirichletEpsilon = 0.25f;   // fraction of noise at root

    public MctsAiService(XiangqiNeuralAiService neural, XiangqiEngine engine)
    {
        _neural = neural;
        _engine = engine;
    }

    /// <summary>
    /// Run MCTS and return the best move.
    /// </summary>
    /// <param name="board">Current board position.</param>
    /// <param name="side">Side to move.</param>
    /// <param name="history">Move history (for repetition detection).</param>
    /// <param name="simulations">Number of MCTS simulations to run.</param>
    /// <param name="temperature">
    ///   Move selection temperature.
    ///   1.0 = sample proportional to visits (good for self-play training data).
    ///   0.0 = greedy (always pick most-visited move, good for play).
    /// </param>
    /// <param name="timeLimitMs">Optional time limit in milliseconds.</param>
    public MctsResult Search(
        Piece?[,] board,
        Side side,
        IReadOnlyList<Move> history,
        int simulations = 800,
        float temperature = 0.0f,
        int timeLimitMs = int.MaxValue)
    {
        var root = new MctsNode(null, 0f, board, side);
        ExpandNode(root, history);
        AddDirichletNoise(root);

        var watch = Stopwatch.StartNew();
        var sim = 0;

        for (; sim < simulations; sim++)
        {
            if (watch.ElapsedMilliseconds >= timeLimitMs) break;

            // 1. Selection — traverse to a leaf
            var path = new List<(MctsNode Node, int ChildIndex)>();
            var node = root;
            var currentHistory = new List<Move>(history);

            while (!node.IsLeaf)
            {
                var (bestChild, bestIndex) = SelectChild(node);
                path.Add((node, bestIndex));
                currentHistory.Add(node.ChildMoves![bestIndex]!);
                node = bestChild;
            }

            // 2. Expansion & Evaluation — ExpandNode already generates the legal
            // list, so terminal detection reuses it instead of a second
            // GetGameStatus/GetLegalMoves pass over the same position.
            if (node.IsLeaf && !node.IsTerminal) ExpandNode(node, currentHistory);
            float value;
            if (node.IsTerminal)
            {
                // No legal move for the side to move means it lost (mate or stalemate).
                value = -1f;
            }
            else
            {
                // CachedValue is already from current player's perspective for V5 models.
                // For legacy models, flip for Black.
                value = _neural.IsV5Model
                    ? node.CachedValue
                    : (node.Side == Side.Red ? node.CachedValue : -node.CachedValue);
            }

            // 3. Backpropagation
            BackupValue(node, path, value);
        }

        return BuildResult(root, temperature, sim);
    }

    /// <summary>
    /// Propagates a leaf evaluation up the path. <paramref name="value"/> is from
    /// <paramref name="leaf"/>'s side-to-move perspective, so it flips at every ply.
    /// </summary>
    internal static void BackupValue(MctsNode leaf, List<(MctsNode Node, int ChildIndex)> path, float value)
    {
        // SelectChild reads VisitCount at every depth, so every node on the path
        // must be counted — not just the root — or sqrt(N) collapses to 0 below it.
        leaf.VisitCount++;
        var backValue = value;
        for (var i = path.Count - 1; i >= 0; i--)
        {
            var (parent, childIndex) = path[i];
            backValue = -backValue;  // flip perspective
            parent.ChildN![childIndex]++;
            parent.ChildW![childIndex] += backValue;
            parent.VisitCount++;
        }
    }

    // ─── Private helpers ────────────────────────────────────────────────────

    internal void ExpandNode(MctsNode node, IReadOnlyList<Move> history)
    {
        if (!node.IsLeaf) return;

        var legalMoves = _engine.GetLegalMoves(node.Board, node.Side, history);
        if (legalMoves.Count == 0)
        {
            node.IsTerminal = true;
            return;
        }

        // Get BOTH policy and value from a single NN inference (avoids duplicate RunInference call)
        var (policyLogits, valuePred) = _neural.RunInference(node.Board, node.Side);
        node.CachedValue = valuePred[0];  // Cache value for use in Search loop

        // Softmax over legal moves only; child storage is indexed by legal-move
        // ordinal, NOT by the 8100-wide action id (five 8100 arrays cost 221 KB/node).
        var count = legalMoves.Count;
        var probs = new float[count];
        var maxLogit = float.NegativeInfinity;
        for (var i = 0; i < count; i++)
        {
            probs[i] = policyLogits[0, MoveEncoder.Encode(legalMoves[i])];
            if (probs[i] > maxLogit) maxLogit = probs[i];
        }
        var expSum = 0f;
        for (var i = 0; i < count; i++)
        {
            probs[i] = MathF.Exp(probs[i] - maxLogit);
            expSum += probs[i];
        }

        node.Children = new MctsNode?[count];
        node.ChildMoves = new Move?[count];
        node.ChildN = new int[count];
        node.ChildW = new float[count];
        node.ChildP = new float[count];

        for (var i = 0; i < count; i++)
        {
            var move = legalMoves[i];
            var prior = probs[i] / expSum;
            node.Children[i] = new MctsNode(node, prior, _engine.ApplyMove(node.Board, move), _engine.Opposite(node.Side));
            node.ChildMoves[i] = move;
            node.ChildP[i] = prior;
        }

        node.IsLeaf = false;
    }

    private static (MctsNode Child, int ChildIndex) SelectChild(MctsNode node)
    {
        var sqrtN = MathF.Sqrt(node.VisitCount);
        var bestScore = float.NegativeInfinity;
        MctsNode? bestChild = null;
        var bestIndex = 0;

        for (var i = 0; i < node.ChildN!.Length; i++)
        {
            var n = node.ChildN[i];
            var q = n > 0 ? node.ChildW![i] / n : 0f;
            var u = CPuct * node.ChildP![i] * sqrtN / (1 + n);
            var score = q + u;

            if (score > bestScore)
            {
                bestScore = score;
                bestChild = node.Children![i];
                bestIndex = i;
            }
        }

        return (bestChild!, bestIndex);
    }

    private static void AddDirichletNoise(MctsNode root)
    {
        if (root.ChildP is null || root.ChildP.Length == 0) return;

        var noise = SampleDirichlet(DirichletAlpha, root.ChildP.Length);
        for (var i = 0; i < root.ChildP.Length; i++)
            root.ChildP[i] = (1 - DirichletEpsilon) * root.ChildP[i] + DirichletEpsilon * noise[i];
    }

    private MctsResult BuildResult(MctsNode root, float temperature, int simulationsRun)
    {
        if (root.ChildMoves is null || root.ChildMoves.Length == 0)
            return new MctsResult(null, Array.Empty<(Move, int, float)>(), simulationsRun, 0f);

        Move? bestMove = null;
        var stats = new List<(Move Move, int Visits, float Q)>();
        var maxVisits = 0;

        for (var i = 0; i < root.ChildMoves.Length; i++)
        {
            var move = root.ChildMoves[i];
            if (move is null) continue;
            var n = root.ChildN![i];
            var q = n > 0 ? root.ChildW![i] / n : 0f;
            stats.Add((move, n, q));
            if (n > maxVisits) { maxVisits = n; bestMove = move; }
        }

        if (temperature > 0.01f && stats.Count > 1)
        {
            // Sample proportional to N^(1/temperature)
            var powered = stats.Select(s => MathF.Pow(s.Visits, 1f / temperature)).ToList();
            var total = powered.Sum();
            if (total > 0)
            {
                var r = (float)Random.Shared.NextDouble() * total;
                var cumulative = 0f;
                for (var i = 0; i < stats.Count; i++)
                {
                    cumulative += powered[i];
                    if (r <= cumulative) { bestMove = stats[i].Move; break; }
                }
            }
        }

        var bestQ = bestMove is not null
            ? stats.FirstOrDefault(s => SameMove(s.Move, bestMove)).Q
            : 0f;

        return new MctsResult(bestMove, stats, simulationsRun, bestQ);
    }

    private static bool SameMove(Move a, Move b) => a.From == b.From && a.To == b.To;

    private static float[] SampleDirichlet(float alpha, int size)
    {
        // Sample from Dirichlet(alpha, alpha, ..., alpha) using Gamma distribution
        // Gamma(alpha, 1) approximated as: if alpha < 1, use Marsaglia-Tsang
        var samples = new float[size];
        var sum = 0f;
        for (var i = 0; i < size; i++)
        {
            samples[i] = SampleGamma(alpha);
            sum += samples[i];
        }
        if (sum > 0)
            for (var i = 0; i < size; i++)
                samples[i] /= sum;
        return samples;
    }

    private static float SampleGamma(float alpha)
    {
        // Marsaglia-Tsang "A Simple Method for Generating Gamma Variables" (2000)
        if (alpha < 1f) alpha = alpha + 1f;
        var d = alpha - 1f / 3f;
        var c = 1f / MathF.Sqrt(9f * d);
        while (true)
        {
            float x, v;
            do
            {
                x = SampleNormal();
                v = 1f + c * x;
            } while (v <= 0);
            v = v * v * v;
            var u = (float)Random.Shared.NextDouble();
            if (u < 1f - 0.0331f * (x * x) * (x * x)) return d * v;
            if (MathF.Log(u) < 0.5f * x * x + d * (1f - v + MathF.Log(v))) return d * v;
        }
    }

    private static float SampleNormal()
    {
        // Box-Muller
        var u1 = (float)(1.0 - Random.Shared.NextDouble());
        var u2 = (float)(1.0 - Random.Shared.NextDouble());
        return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
    }
}

/// <summary>A single node in the MCTS tree.</summary>
internal sealed class MctsNode
{
    public MctsNode? Parent { get; }
    public float Prior { get; }
    public Piece?[,] Board { get; }
    public Side Side { get; }
    public int VisitCount { get; set; }
    public bool IsLeaf { get; set; } = true;
    public bool IsTerminal { get; set; }

    // Per-action arrays, indexed by legal-move ordinal (0..legalCount-1).
    public MctsNode?[]? Children { get; set; }
    public Move?[]? ChildMoves { get; set; }
    public int[]? ChildN { get; set; }
    public float[]? ChildW { get; set; }
    public float[]? ChildP { get; set; }

    /// <summary>
    /// Cached value prediction from NN (current player's perspective).
    /// Set during ExpandNode to avoid a second NN inference call.
    /// </summary>
    public float CachedValue { get; set; }

    public MctsNode(MctsNode? parent, float prior, Piece?[,] board, Side side)
    {
        Parent = parent;
        Prior = prior;
        Board = board;
        Side = side;
        VisitCount = 0;
    }
}

/// <summary>Result of an MCTS search.</summary>
public sealed record MctsResult(
    Move? BestMove,
    IReadOnlyList<(Move Move, int Visits, float Q)> MoveStats,
    int SimulationsRun,
    float BestQ);
