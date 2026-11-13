using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using ChineseChess.Ai;
using ChineseChess.Ai.Neural;
using ChineseChess.Encoding;
using ChineseChess.Models;
using ChineseChess.Services;

namespace ChineseChess.AiMatchRunner;

/// <summary>Paired, auditable strength evaluation. No external teacher process.</summary>
internal static class ArenaCommand
{
    public static void Run(string[] args)
    {
        if (args.Length < 5)
            throw new ArgumentException("--arena <mcts|policy|value|nnue|nnue-greedy> <even games> <output-dir> <model.onnx> <opening.jsonl> [simulations=3000] [ms=1200] [settings.json]");
        var mode = args[0];
        if (mode is not ("mcts" or "policy" or "value" or "nnue" or "nnue-greedy")) throw new ArgumentException("Unknown mode");
        var games = int.Parse(args[1]);
        var output = Path.GetFullPath(args[2]);
        var model = Path.GetFullPath(args[3]);
        var openingFile = Path.GetFullPath(args[4]);
        var simulations = args.Length > 5 ? int.Parse(args[5]) : 3000;
        var budget = args.Length > 6 ? int.Parse(args[6]) : 1200;
        var settings = args.Length > 7 ? JsonSerializer.Deserialize<MctsSearchOptions>(File.ReadAllText(args[7]))
            ?? throw new ArgumentException("Invalid search settings") : new MctsSearchOptions();
        if (games < 2 || games % 2 != 0 || simulations < 1 || budget < 1) throw new ArgumentException("Invalid match budgets");
        var openings = File.ReadLines(openingFile).Select(line => JsonSerializer.Deserialize<OpeningRow>(line)!)
            .Where(row => row.MoveIndex == 8).ToArray();
        if (openings.Length * 2 < games) throw new ArgumentException("Need one distinct eight-ply opening per color pair");
        if (openings.Select(o => string.Join(',', o.History)).Distinct().Count() != openings.Length)
            throw new ArgumentException("Duplicate opening sequences");
        Directory.CreateDirectory(output);
        using var trace = new StreamWriter(new FileStream(Path.Combine(output, "moves.jsonl"), FileMode.CreateNew)) { AutoFlush = true };
        var metadata = new
        {
            Mode = mode, Games = games, Model = model, ModelSha256 = Hash(model),
            OpeningFile = openingFile, OpeningSha256 = Hash(openingFile),
            CoreSha256 = Hash(typeof(XiangqiEngine).Assembly.Location),
            NeuralSha256 = Hash(typeof(MctsAiService).Assembly.Location),
            ArenaSha256 = Hash(typeof(ArenaCommand).Assembly.Location),
            BudgetMs = budget, Simulations = simulations, Temperature = 0, ClassicLevel = 4,
            SearchSettings = settings,
            MaxPlies = 300, StartedUtc = DateTime.UtcNow,
            Runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
            ProcessorCount = Environment.ProcessorCount
        };
        File.WriteAllText(Path.Combine(output, "metadata.json"), JsonSerializer.Serialize(metadata, Pretty));
        Console.WriteLine(JsonSerializer.Serialize(metadata));
        var engine = new XiangqiEngine();
        var summaries = new List<GameResult>();
        for (var game = 0; game < games; game++)
        {
            // Fresh search state and caches on BOTH sides of every game.
            IAiEngine neural;
            if (mode == "nnue-greedy") neural = new GreedyOnnxEngine(model, engine);
            else if (mode is "value" or "nnue") neural = new NeuralValueAlphaBetaAiEngine(model, engine, useFairyFeatures: mode == "nnue");
            else
            {
                var neuralService = new XiangqiNeuralAiService(model, engine, true);
                neural = mode == "mcts"
                    ? new MctsAiEngine(neuralService, new MctsAiService(neuralService, engine, settings), simulations)
                    : new NeuralPolicyAiEngine(neuralService);
            }
            using var neuralLifetime = (IDisposable)neural;
            IAiEngine classic = new XiangqiAiService(engine);
            var nnSide = game % 2 == 0 ? Side.Red : Side.Black;
            var state = engine.CreateInitialState();
            var repetitions = new Dictionary<string, int>();
            string Key() => ((int)state.Turn) + ":" + string.Join(',', state.Board.Cast<Piece?>().Select(p => p is null ? "." : $"{(int)p.Side}{(int)p.Type}"));
            void CountPosition() { var key = Key(); repetitions[key] = repetitions.GetValueOrDefault(key) + 1; }
            CountPosition();
            foreach (var action in openings[game / 2].History)
            {
                var move = engine.GetLegalMoves(state.Board, state.Turn, state.History).Single(m => MoveEncoder.Encode(m) == action);
                state = engine.MakeMove(state, move);
                CountPosition();
            }
            var nnTimes = new List<double>();
            var classicTimes = new List<double>();
            var classicDepths = new List<int>();
            var counts = new List<int>();
            var outcome = "Draw";
            var reason = "PlyLimit";
            var plies = 0;
            var gameWatch = Stopwatch.StartNew();
            Console.WriteLine($"START game={game + 1}/{games} opening={openings[game / 2].GameId} NN={nnSide}");
            while (plies < 300)
            {
                var isNn = state.Turn == nnSide;
                var legal = engine.GetLegalMoves(state.Board, state.Turn, state.History);
                if (legal.Count == 0) { outcome = isNn ? "Loss" : "Win"; reason = "NoLegalMoves"; break; }
                var watch = Stopwatch.StartNew();
                var result = (isNn ? neural : classic).ChooseMove(new AiMoveRequest(state.Board, state.Turn, state.History,
                    budget, Difficulty: 4, Temperature: 0, Simulations: simulations));
                var elapsed = watch.Elapsed.TotalMilliseconds;
                var move = result.Move;
                if (move is null || !legal.Any(m => m.From == move.From && m.To == move.To))
                    throw new InvalidOperationException($"Illegal/null move in game {game + 1} ply {plies}");
                if (isNn) { nnTimes.Add(elapsed); counts.Add(result.Stats.Sims); }
                else { classicTimes.Add(elapsed); classicDepths.Add(result.Stats.DepthReached); }
                trace.WriteLine(JsonSerializer.Serialize(new
                {
                    Game = game + 1, Opening = openings[game / 2].GameId, Ply = plies + 1,
                    Side = state.Turn.ToString(), Actor = isNn ? mode : "classic",
                    NeuralEvaluations = !isNn ? null : neural is NeuralValueAlphaBetaAiEngine valueEngine ? valueEngine.LastEvaluations
                        : neural is GreedyOnnxEngine greedyEngine ? greedyEngine.LastEvaluations : (long?)null,
                    NeuralCacheHits = isNn && neural is NeuralValueAlphaBetaAiEngine cachedValueEngine ? cachedValueEngine.LastCacheHits : (long?)null,
                    Action = MoveEncoder.Encode(move), move.From, move.To, TimeMs = elapsed, result.Stats
                }));
                state = engine.MakeMove(state, move);
                plies++;
                var status = engine.GetGameStatus(state.Board, state.Turn, state.History);
                if (status is GameStatus.RedWins or GameStatus.BlackWins)
                {
                    var winner = status == GameStatus.RedWins ? Side.Red : Side.Black;
                    outcome = winner == nnSide ? "Win" : "Loss";
                    reason = engine.IsInCheck(state.Board, state.Turn) ? "Checkmate" : "NoLegalMoves";
                    break;
                }
                CountPosition();
                if (repetitions[Key()] >= 3) { reason = "ThreefoldRepetition"; break; }
                if (plies % 20 == 0) Console.WriteLine($"PROGRESS game={game + 1} ply={plies} seconds={gameWatch.Elapsed.TotalSeconds:F0}");
            }
            var summary = new GameResult(game + 1, openings[game / 2].GameId, nnSide.ToString(), outcome, reason, plies,
                nnTimes.DefaultIfEmpty().Average(), classicTimes.DefaultIfEmpty().Average(),
                nnTimes.DefaultIfEmpty().Max(), classicTimes.DefaultIfEmpty().Max(),
                counts.DefaultIfEmpty().Average(), classicDepths.DefaultIfEmpty().Average());
            summaries.Add(summary);
            File.WriteAllText(Path.Combine(output, "games.json"), JsonSerializer.Serialize(summaries, Pretty));
            Console.WriteLine("RESULT " + JsonSerializer.Serialize(summary));
        }
        var wins = summaries.Count(s => s.Outcome == "Win");
        var losses = summaries.Count(s => s.Outcome == "Loss");
        var draws = games - wins - losses;
        var score = (wins + draws * 0.5) / games;
        var resultSummary = new { Wins = wins, Losses = losses, Draws = draws, ScoreRate = score,
            MeetsNumericalGate = games >= 40 && wins > losses && score >= 0.60,
            Note = "Numerical gate alone does not certify isolated load, frozen code or absence of training leakage" };
        File.WriteAllText(Path.Combine(output, "summary.json"), JsonSerializer.Serialize(resultSummary, Pretty));
        Console.WriteLine("COMPLETE " + JsonSerializer.Serialize(resultSummary));
    }

    private static string Hash(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)));
    // Ablation only: same NNUE weights, evaluate legal successors, no deeper tree.
    private sealed class GreedyOnnxEngine(string model, XiangqiEngine engine) : IAiEngine, IDisposable
    {
        private readonly FairyOnnxEvaluator _value = new(model);
        public string Name => "NNUE one-ply value";
        public long LastEvaluations { get; private set; }
        public AiMoveResult ChooseMove(AiMoveRequest request)
        {
            var watch = Stopwatch.StartNew();
            var before = _value.Evaluations;
            _value.BeginSearch();
            var legal = engine.GetLegalMoves(request.Board, request.Side, request.History);
            Move? best = legal.FirstOrDefault();
            var score = double.NegativeInfinity;
            var visited = 0;
            foreach (var move in legal)
            {
                if (visited > 0 && watch.ElapsedMilliseconds >= request.TimeLimitMs) break;
                var board = engine.ApplyMove(request.Board, move);
                var nextSide = engine.Opposite(request.Side);
                var status = engine.GetGameStatus(board, nextSide, request.History.Append(move).ToList());
                var candidate = status is GameStatus.RedWins or GameStatus.BlackWins ? 999998 : -_value.Evaluate(board, nextSide, request.Side);
                visited++;
                if (candidate > score) { score = candidate; best = move; }
            }
            LastEvaluations = _value.Evaluations - before;
            return new AiMoveResult(best, new SearchStats(1, visited, watch.Elapsed.TotalMilliseconds, best is null ? 0 : score));
        }
        public void Dispose() => _value.Dispose();
    }
    private static readonly JsonSerializerOptions Pretty = new() { WriteIndented = true };
    private sealed record OpeningRow(string GameId, int MoveIndex, int[] History);
    private sealed record GameResult(int Game, string Opening, string NnSide, string Outcome, string Reason, int Plies,
        double NnAvgMs, double ClassicAvgMs, double NnMaxMs, double ClassicMaxMs, double AvgSimulations, double AvgClassicDepth);
}
