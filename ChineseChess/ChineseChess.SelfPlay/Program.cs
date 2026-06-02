using System.Text.Json;
using System.Text.Json.Serialization;
using ChineseChess.Encoding;
using ChineseChess.Models;
using ChineseChess.Services;

internal static class Program
{
    private static async Task Main(string[] args)
    {
        var options = SelfPlayOptions.Parse(args);
        var outputPath = Path.GetFullPath(options.OutputPath);
        var outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        var jsonOptions = new JsonSerializerOptions
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.Never
        };

        var engine = new XiangqiEngine();
        var ai = new XiangqiAiService(engine);
        var redWins = 0;
        var blackWins = 0;
        var draws = 0;
        var totalRows = 0;
        var uniqueSignatures = new HashSet<string>(StringComparer.Ordinal);
        var duplicateGames = 0;
        var random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();

        using var writer = new StreamWriter(outputPath, append: false);
        for (var gameId = 1; gameId <= options.Games; gameId++)
        {
            var state = engine.CreateInitialState();
            var rows = new List<SelfPlayRow>();
            var selectedMoves = new List<int>();

            for (var moveIndex = 0; moveIndex < options.MaxMoves && IsPlaying(state.Status); moveIndex++)
            {
                var legalMoves = engine.GetLegalMoves(state.Board, state.Turn, state.History);
                if (legalMoves.Count == 0) break;

                var forceRandomMove = moveIndex < options.RandomOpeningPlies || random.NextDouble() < options.RandomMoveProbability;
                var result = forceRandomMove
                    ? PickRandomMove(legalMoves, random)
                    : ai.ChooseMove(state.Board, state.Turn, options.Level, state.History, options.TimeMs, BuildMoveSelectionOptions(options));

                if (result.Move is null) break;
                var selectedMove = MoveEncoder.Encode(result.Move);

                rows.Add(new SelfPlayRow(
                    gameId,
                    moveIndex,
                    state.Turn.ToString(),
                    BoardEncoder.Encode(state.Board),
                    legalMoves.Select(MoveEncoder.Encode).ToArray(),
                    selectedMove,
                    0,
                    result.Stats.BestScore,
                    ToRedPerspectiveScore(result.Stats.BestScore, state.Turn),
                    result.Stats.DepthReached,
                    result.Stats.Nodes,
                    result.Stats.TimeMs));

                selectedMoves.Add(selectedMove);
                state = engine.MakeMove(state, result.Move);
            }

            var signature = string.Join(',', selectedMoves);
            if (!uniqueSignatures.Add(signature)) duplicateGames++;

            var unfinished = IsPlaying(state.Status);
            var finalResult = GetRedPerspectiveResult(state.Status);
            if (finalResult == 1) redWins++;
            else if (finalResult == -1) blackWins++;
            else draws++;

            foreach (var row in rows)
            {
                var finalizedRow = row with { Result = finalResult, Unfinished = unfinished };
                ValidateRow(finalizedRow);
                await writer.WriteLineAsync(JsonSerializer.Serialize(finalizedRow, jsonOptions));
            }

            totalRows += rows.Count;
            Console.WriteLine($"Game {gameId}/{options.Games}: {FormatResult(finalResult)}, rows {rows.Count}");
        }

        Console.WriteLine("Self-play complete.");
        Console.WriteLine($"Games played: {options.Games}");
        Console.WriteLine($"Red wins: {redWins}");
        Console.WriteLine($"Black wins: {blackWins}");
        Console.WriteLine($"Draws: {draws}");
        Console.WriteLine($"Total rows: {totalRows}");
        Console.WriteLine($"Duplicate games: {duplicateGames}");
        Console.WriteLine($"Unique signatures: {uniqueSignatures.Count}");
        Console.WriteLine($"Output: {outputPath}");
    }

    private static AiSearchResult PickRandomMove(IReadOnlyList<Move> legalMoves, Random random)
    {
        var move = legalMoves[random.Next(legalMoves.Count)];
        return new AiSearchResult(move, new SearchStats(0, legalMoves.Count, 0, 0));
    }

    private static MoveSelectionOptions BuildMoveSelectionOptions(SelfPlayOptions options)
    {
        var enableRandomSelection = options.TopK > 1 || options.NearBestWindow > 0;
        return new MoveSelectionOptions(
            EnableRandomSelection: enableRandomSelection,
            TopK: options.TopK,
            NearBestWindow: options.NearBestWindow,
            Seed: options.Seed,
            RandomOpeningPlies: options.RandomOpeningPlies);
    }

    private static void ValidateRow(SelfPlayRow row)
    {
        if (row.BoardEncoding.Length != 1260) throw new InvalidOperationException($"Invalid BoardEncoding length: {row.BoardEncoding.Length}");
        if (row.LegalMoves.Length == 0) throw new InvalidOperationException("LegalMoves must not be empty.");
        if (!row.LegalMoves.Contains(row.SelectedMove)) throw new InvalidOperationException($"SelectedMove {row.SelectedMove} not found in LegalMoves.");
        if (row.SelectedMove is < 0 or > 8099) throw new InvalidOperationException($"SelectedMove out of range: {row.SelectedMove}");
    }

    private static bool IsPlaying(GameStatus status) => status is GameStatus.Playing or GameStatus.RedCheck or GameStatus.BlackCheck;

    private static int GetRedPerspectiveResult(GameStatus status) => status switch
    {
        GameStatus.RedWins => 1,
        GameStatus.BlackWins => -1,
        _ => 0
    };

    private static string FormatResult(int result) => result switch
    {
        1 => "Red win",
        -1 => "Black win",
        _ => "Draw/unfinished"
    };

    private static double ToRedPerspectiveScore(double score, Side side) => side == Side.Red ? score : -score;
}

internal sealed record SelfPlayOptions(
    int Games,
    string OutputPath,
    int Level,
    int TimeMs,
    int MaxMoves,
    int TopK,
    double NearBestWindow,
    int RandomOpeningPlies,
    double RandomMoveProbability,
    int? Seed)
{
    public static SelfPlayOptions Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["--games"] = "100",
            ["--out"] = "data/selfplay/games.jsonl",
            ["--level"] = "4",
            ["--timeMs"] = "300",
            ["--maxMoves"] = "300",
            ["--topK"] = "1",
            ["--nearBestWindow"] = "0",
            ["--randomOpeningPlies"] = "0",
            ["--randomMoveProbability"] = "0",
            ["--seed"] = string.Empty
        };

        for (var i = 0; i < args.Length; i++)
        {
            var key = args[i];
            if (!values.ContainsKey(key))
            {
                throw new ArgumentException($"Unknown argument '{key}'.");
            }

            if (i + 1 >= args.Length)
            {
                throw new ArgumentException($"Missing value for '{key}'.");
            }

            values[key] = args[++i];
        }

        return new SelfPlayOptions(
            ParsePositiveInt(values["--games"], "--games"),
            values["--out"],
            ParseRangedInt(values["--level"], "--level", 1, 4),
            ParsePositiveInt(values["--timeMs"], "--timeMs"),
            ParsePositiveInt(values["--maxMoves"], "--maxMoves"),
            ParsePositiveInt(values["--topK"], "--topK"),
            ParseNonNegativeDouble(values["--nearBestWindow"], "--nearBestWindow"),
            ParseNonNegativeInt(values["--randomOpeningPlies"], "--randomOpeningPlies"),
            ParseProbability(values["--randomMoveProbability"], "--randomMoveProbability"),
            ParseOptionalInt(values["--seed"], "--seed"));
    }

    private static int ParsePositiveInt(string value, string name)
    {
        if (!int.TryParse(value, out var parsed) || parsed <= 0)
        {
            throw new ArgumentException($"{name} must be a positive integer.");
        }

        return parsed;
    }

    private static int ParseRangedInt(string value, string name, int min, int max)
    {
        if (!int.TryParse(value, out var parsed) || parsed < min || parsed > max)
        {
            throw new ArgumentException($"{name} must be an integer from {min} to {max}.");
        }

        return parsed;
    }

    private static int ParseNonNegativeInt(string value, string name)
    {
        if (!int.TryParse(value, out var parsed) || parsed < 0)
        {
            throw new ArgumentException($"{name} must be a non-negative integer.");
        }

        return parsed;
    }

    private static double ParseNonNegativeDouble(string value, string name)
    {
        if (!double.TryParse(value, out var parsed) || parsed < 0)
        {
            throw new ArgumentException($"{name} must be a non-negative number.");
        }

        return parsed;
    }

    private static double ParseProbability(string value, string name)
    {
        if (!double.TryParse(value, out var parsed) || parsed < 0 || parsed > 1)
        {
            throw new ArgumentException($"{name} must be a number from 0 to 1.");
        }

        return parsed;
    }

    private static int? ParseOptionalInt(string value, string name)
    {
        if (string.IsNullOrWhiteSpace(value)) return null;
        if (!int.TryParse(value, out var parsed))
        {
            throw new ArgumentException($"{name} must be an integer when provided.");
        }

        return parsed;
    }
}

internal sealed record SelfPlayRow(
    int GameId,
    int MoveIndex,
    string Side,
    float[] BoardEncoding,
    int[] LegalMoves,
    int SelectedMove,
    int Result,
    double SearchScoreSidePerspective,
    double SearchScoreRedPerspective,
    int DepthReached,
    int Nodes,
    double TimeMs,
    bool Unfinished = false);
