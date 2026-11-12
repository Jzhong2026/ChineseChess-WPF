using ChineseChess.Encoding;
using ChineseChess.Models;
using ChineseChess.Ai.Neural;
using ChineseChess.Services;

internal static class Program
{
    private static int Main()
    {
        try
        {
            VerifyEncoders();
            VerifyNeuralAi();
            Console.WriteLine("Core verification passed.");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Core verification FAILED: {ex}");
            return 1;
        }
    }

    private static void VerifyEncoders()
    {
        var engine = new XiangqiEngine();
        var boardEncoding = BoardEncoder.Encode(engine.CreateInitialBoard());
        Require(boardEncoding.Length == 1260, $"board encoding length expected 1260, got {boardEncoding.Length}");

        var minActionId = int.MaxValue;
        var maxActionId = int.MinValue;
        for (var fromRow = 0; fromRow < XiangqiEngine.BoardRows; fromRow++)
        {
            for (var fromCol = 0; fromCol < XiangqiEngine.BoardCols; fromCol++)
            {
                for (var toRow = 0; toRow < XiangqiEngine.BoardRows; toRow++)
                {
                    for (var toCol = 0; toCol < XiangqiEngine.BoardCols; toCol++)
                    {
                        var actionId = MoveEncoder.Encode(new Position(fromRow, fromCol), new Position(toRow, toCol));
                        minActionId = Math.Min(minActionId, actionId);
                        maxActionId = Math.Max(maxActionId, actionId);
                    }
                }
            }
        }

        Require(minActionId == 0, $"move action id min expected 0, got {minActionId}");
        Require(maxActionId == 8099, $"move action id max expected 8099, got {maxActionId}");
        Require(MoveEncoder.ActionCount == 8100, $"move action count expected 8100, got {MoveEncoder.ActionCount}");

        // Round-trip every action id: encode/decode must be exact inverses.
        for (var actionId = 0; actionId < MoveEncoder.ActionCount; actionId++)
        {
            var (from, to) = MoveEncoder.Decode(actionId);
            Require(MoveEncoder.Encode(from, to) == actionId, $"action {actionId} did not round-trip");
        }

        Console.WriteLine("Encoders OK: board length 1260, action ids 0..8099, round-trip exact.");
    }

    private static void VerifyNeuralAi()
    {
        var onnxPath = LocateModel();
        var engine = new XiangqiEngine();
        using var neuralAi = new XiangqiNeuralAiService(onnxPath, engine, isCnnModel: true);
        Console.WriteLine($"ONNX model loaded: {onnxPath}");
        Console.WriteLine($"  factored={neuralAi.IsFactoredModel} v5planes={neuralAi.IsV5Model}");

        var state = engine.CreateInitialState();
        var legalMoves = engine.GetLegalMoves(state.Board, Side.Red, state.History);
        Require(legalMoves.Count > 0, "initial position must have legal moves");

        var policyMove = neuralAi.ChooseMoveByPolicy(state.Board, Side.Red, state.History);
        Require(policyMove is not null, "policy returned no move");
        var chosenPolicy = policyMove!;
        Require(legalMoves.Any(m => m.From == chosenPolicy.From && m.To == chosenPolicy.To),
            $"policy move {chosenPolicy} is not legal");

        var value = neuralAi.EvaluatePosition(state.Board, Side.Red);
        Require(float.IsFinite(value) && value >= -1f && value <= 1f,
            $"value prediction out of range: {value}");

        var dist = neuralAi.GetPolicyDistribution(state.Board, Side.Red, state.History);
        Require(dist.Count == legalMoves.Count, $"policy distribution covers {dist.Count} of {legalMoves.Count} legal moves");
        Require(dist.All(p => float.IsFinite(p.Probability) && p.Probability >= 0f), "policy probabilities must be finite and non-negative");
        var total = dist.Sum(p => p.Probability);
        Require(Math.Abs(total - 1f) < 1e-3f, $"policy probabilities sum to {total}, expected ~1");

        var mctsAi = new MctsAiService(neuralAi, engine);
        var mctsResult = mctsAi.Search(state.Board, Side.Red, state.History, simulations: 50, temperature: 0.0f);
        Require(mctsResult.BestMove is not null, "MCTS returned no move");
        var chosenMcts = mctsResult.BestMove!;
        Require(legalMoves.Any(m => m.From == chosenMcts.From && m.To == chosenMcts.To),
            "MCTS move is not legal");
        Require(mctsResult.SimulationsRun == 50, $"MCTS ran {mctsResult.SimulationsRun} of 50 simulations");

        Console.WriteLine($"Neural AI OK: policy={chosenPolicy.From}->{chosenPolicy.To}, value={value:F4}, MCTS Q={mctsResult.BestQ:F3}");
    }

    /// <summary>
    /// Walks up from the assembly to the repository root, which is the directory
    /// holding artifacts/. The previous fixed four-level hop landed on
    /// ChineseChess/artifacts and silently skipped every neural check.
    /// </summary>
    private static string LocateModel()
    {
        var overridePath = Environment.GetEnvironmentVariable("CHESS_MODEL");
        if (!string.IsNullOrEmpty(overridePath))
        {
            Require(File.Exists(overridePath), $"CHESS_MODEL points at a missing file: {overridePath}");
            return overridePath;
        }

        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "artifacts", "cnn_policy_value.onnx");
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }

        var local = Path.Combine(AppContext.BaseDirectory, "cnn_policy_value.onnx");
        Require(File.Exists(local),
            $"cnn_policy_value.onnx not found in any parent artifacts/ directory nor at {local}. Set CHESS_MODEL to override.");
        return local;
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
}