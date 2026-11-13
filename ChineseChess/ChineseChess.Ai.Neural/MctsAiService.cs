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
    private readonly MctsSearchOptions _options;
    private readonly MaterialQuiescence _tactics;

    // Dirichlet noise for root node exploration (AlphaZero style)
    private const float DirichletAlpha = 0.3f;
    private const float DirichletEpsilon = 0.25f;   // fraction of noise at root

    public MctsAiService(XiangqiNeuralAiService neural, XiangqiEngine engine, MctsSearchOptions? options = null)
    {
        _neural = neural;
        _engine = engine;
        _options = options ?? new MctsSearchOptions();
        if (_options.TacticalDepth is < 0 or > 8 || !float.IsFinite(_options.TacticalBlend)
            || _options.TacticalBlend is < 0 or > 1 || !float.IsFinite(_options.PolicyTemperature)
            || _options.PolicyTemperature <= 0 || !float.IsFinite(_options.CPuct) || _options.CPuct <= 0
            || _options.RepetitionDrawCount is < 0 or 1
            || (_options.FirstPlayReduction.HasValue && (!float.IsFinite(_options.FirstPlayReduction.Value) || _options.FirstPlayReduction < 0)))
            throw new ArgumentOutOfRangeException(nameof(options));
        _tactics = new MaterialQuiescence(engine);
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
        int timeLimitMs = int.MaxValue,
        bool addExplorationNoise = false)
    {
        var watch = Stopwatch.StartNew();
        var root = new MctsNode(null, 0f, board, side);
        var previousPositions = _options.RepetitionDrawCount > 0 ? CountHistoryPositions(board, side, history) : new Dictionary<string, int>();
        ExpandNode(root, history);
        if (addExplorationNoise) AddDirichletNoise(root);
        // A network estimate must never override a verified mate/stalemate.
        // The guard shares the move's clock; incomplete checks prove nothing.
        CheckRootTactics(root, history, watch, Math.Min(200, timeLimitMs / 4));
        var sim = 0;

        for (; sim < simulations; sim++)
        {
            if (watch.ElapsedMilliseconds >= timeLimitMs) break;
            if (root.ProvenValue.HasValue) break;

            // 1. Selection — traverse to a leaf
            var path = new List<(MctsNode Node, int ChildIndex)>();
            var node = root;
            var currentHistory = new List<Move>(history);
            var pathPositions = new Dictionary<string, int>();

            while (!node.IsLeaf && !node.ProvenValue.HasValue)
            {
                var (bestChild, bestIndex) = SelectChild(node);
                path.Add((node, bestIndex));
                currentHistory.Add(node.ChildMoves![bestIndex]!);
                node = bestChild;
                if (_options.RepetitionDrawCount > 0)
                {
                    var key = node.RepetitionKey ??= PositionKey(node.Board, node.Side);
                    pathPositions[key] = pathPositions.GetValueOrDefault(key) + 1;
                    if (previousPositions.GetValueOrDefault(key) + pathPositions[key] >= _options.RepetitionDrawCount)
                    {
                        // Opt in only when the caller adjudicates repetition draws.
                        // Mate/stalemate takes precedence, matching the arena.
                        node.ProvenValue = _engine.GetLegalMoves(node.Board, node.Side, currentHistory).Count == 0 ? -1 : 0;
                        node.IsTerminal = true;
                        break;
                    }
                }
            }

            // 2. Expansion & Evaluation — ExpandNode already generates the legal
            // list, so terminal detection reuses it instead of a second
            // GetGameStatus/GetLegalMoves pass over the same position.
            if (node.IsLeaf && !node.ProvenValue.HasValue) ExpandNode(node, currentHistory);
            float value;
            if (node.ProvenValue.HasValue)
            {
                value = node.ProvenValue.Value;
            }
            else
            {
                // CachedValue is already from current player's perspective for V5 models.
                // For legacy models, flip for Black.
                value = _neural.IsV5Model
                    ? node.CachedValue
                    : (node.Side == Side.Red ? node.CachedValue : -node.CachedValue);
                if (_options.TacticalDepth > 0)
                {
                    var tactical = _tactics.Evaluate(node.Board, node.Side, currentHistory, _options.TacticalDepth,
                        watch, timeLimitMs, node.ChildMoves!.Select(m => m!).ToArray());
                    value = (1 - _options.TacticalBlend) * value + _options.TacticalBlend * MathF.Tanh(tactical / 600f);
                }
            }

            // 3. Backpropagation
            BackupValue(node, path, value);
            for (var i = path.Count - 1; i >= 0; i--) UpdateProof(path[i].Node);
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
            node.ProvenValue = -1;
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
            probs[i] = policyLogits.GetValue(MoveEncoder.Encode(legalMoves[i])) / _options.PolicyTemperature;
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

    private (MctsNode Child, int ChildIndex) SelectChild(MctsNode node)
    {
        var sqrtN = MathF.Sqrt(Math.Max(1, node.VisitCount));
        var bestScore = float.NegativeInfinity;
        MctsNode? bestChild = null;
        var bestIndex = 0;
        // Eligibility is a property of the sibling set. Compute it once, rather
        // than rescanning all siblings for every candidate (quadratic work).
        var children = node.Children!;
        var hasWinningChild = children.Any(c => c?.ProvenValue == -1);
        var hasNonLosingChild = children.Any(c => c?.ProvenValue != 1);
        var priorValue = _neural.IsV5Model || node.Side == Side.Red ? node.CachedValue : -node.CachedValue;

        for (var i = 0; i < node.ChildN!.Length; i++)
        {
            var proof = children[i]?.ProvenValue;
            if (hasWinningChild ? proof != -1 : hasNonLosingChild && proof == 1) continue;
            var n = node.ChildN[i];
            var q = n > 0 ? node.ChildW![i] / n : _options.FirstPlayReduction.HasValue ? priorValue - _options.FirstPlayReduction.Value : 0f;
            var u = _options.CPuct * node.ChildP![i] * sqrtN / (1 + n);
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

    internal static void UpdateProof(MctsNode node)
    {
        if (node.Children is not { Length: > 0 }) return;
        if (node.Children.Any(c => c?.ProvenValue == -1)) node.ProvenValue = 1;
        else if (node.Children.All(c => c?.ProvenValue.HasValue == true))
            node.ProvenValue = node.Children.Any(c => c!.ProvenValue == 0) ? 0 : -1;
    }

    internal static string PositionKey(Piece?[,] board, Side side) => string.Create(91, (board, side), static (span, state) =>
    {
        span[0] = (char)state.side;
        for (var row = 0; row < 10; row++)
        for (var col = 0; col < 9; col++)
        {
            var piece = state.board[row, col];
            span[1 + row * 9 + col] = piece is null ? '\0' : (char)(1 + (int)piece.Side * 7 + (int)piece.Type);
        }
    });

    internal static Dictionary<string, int> CountHistoryPositions(Piece?[,] board, Side side, IReadOnlyList<Move> history)
    {
        var scratch = (Piece?[,])board.Clone();
        var counts = new Dictionary<string, int>();
        for (var index = history.Count; index >= 0; index--)
        {
            var hash = PositionKey(scratch, side);
            counts[hash] = counts.GetValueOrDefault(hash) + 1;
            if (index == 0) break;
            var move = history[index - 1];
            scratch[move.From.Row, move.From.Col] = move.Piece;
            scratch[move.To.Row, move.To.Col] = move.Captured;
            side = side == Side.Red ? Side.Black : Side.Red;
        }
        return counts;
    }

    private static bool IsEligible(MctsNode node, int index)
    {
        if (node.Children is null) return true;
        if (node.Children.Any(c => c?.ProvenValue == -1)) return node.Children[index]?.ProvenValue == -1;
        return node.Children[index]?.ProvenValue != 1 || node.Children.All(c => c?.ProvenValue == 1);
    }

    private void CheckRootTactics(MctsNode root, IReadOnlyList<Move> history, Stopwatch watch, int budgetMs)
    {
        if (root.Children is null) return;
        // First inspect every immediate win before spending time on opponent replies.
        var replies = new List<Move>?[root.Children.Length];
        for (var i = 0; i < root.Children.Length; i++)
        {
            if (watch.ElapsedMilliseconds >= budgetMs) return;
            var child = root.Children[i]!;
            var childHistory = history.Append(root.ChildMoves![i]!).ToList();
            replies[i] = _engine.GetLegalMoves(child.Board, child.Side, childHistory).ToList();
            if (replies[i]!.Count == 0) { child.ProvenValue = -1; child.IsTerminal = true; UpdateProof(root); return; }
        }
        for (var i = 0; i < root.Children.Length; i++)
        {
            var child = root.Children[i]!;
            var childHistory = history.Append(root.ChildMoves![i]!).ToList();
            foreach (var reply in replies[i]!)
            {
                if (watch.ElapsedMilliseconds >= budgetMs) return;
                var next = _engine.ApplyMove(child.Board, reply);
                if (_engine.GetLegalMoves(next, root.Side, childHistory.Append(reply).ToList()).Count != 0) continue;
                child.ProvenValue = 1;
                break;
            }
        }
        UpdateProof(root);
    }

    private static void AddDirichletNoise(MctsNode root)
    {
        if (root.ChildP is null || root.ChildP.Length == 0) return;

        var noise = SampleDirichlet(DirichletAlpha, root.ChildP.Length);
        for (var i = 0; i < root.ChildP.Length; i++)
            root.ChildP[i] = (1 - DirichletEpsilon) * root.ChildP[i] + DirichletEpsilon * noise[i];
    }

    internal static MctsResult BuildResult(MctsNode root, float temperature, int simulationsRun)
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
            var q = root.Children?[i]?.ProvenValue is int proof ? -proof : n > 0 ? root.ChildW![i] / n : 0f;
            stats.Add((move, n, q));
            if (IsEligible(root, i) && (bestMove is null || n > maxVisits)) { maxVisits = n; bestMove = move; }
        }

        bestMove ??= stats[0].Move;
        if (temperature > 0.01f && stats.Count > 1 && maxVisits > 0)
        {
            // Scaling every visit count by the maximum preserves probabilities
            // and keeps weights in [0,1], even for very low temperatures.
            var powered = stats.Select((s, i) => IsEligible(root, i) ? Math.Pow((double)s.Visits / maxVisits, 1.0 / temperature) : 0).ToList();
            var total = powered.Sum();
            if (total > 0)
            {
                var r = Random.Shared.NextDouble() * total;
                var cumulative = 0.0;
                for (var i = 0; i < stats.Count; i++)
                {
                    cumulative += powered[i];
                    if (r < cumulative) { bestMove = stats[i].Move; break; }
                }
            }
        }

        var bestQ = bestMove is not null
            ? stats.FirstOrDefault(s => SameMove(s.Move, bestMove)).Q
            : 0f;

        return new MctsResult(bestMove, stats, simulationsRun, bestQ);
    }

    private static bool SameMove(Move a, Move b) => a.From == b.From && a.To == b.To;

    internal static float[] SampleDirichlet(float alpha, int size)
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
        if (alpha < 1f)
            return SampleGamma(alpha + 1f) * MathF.Pow((float)(1.0 - Random.Shared.NextDouble()), 1f / alpha);
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
    /// <summary>Exact game-theoretic result from this node's side-to-move perspective.</summary>
    public int? ProvenValue { get; set; }
    internal string? RepetitionKey { get; set; }

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
