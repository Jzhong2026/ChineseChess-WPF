using System.Text.Json;
using System.Text.Json.Serialization;
using ChineseChess.Encoding;
using ChineseChess.Models;
using ChineseChess.Services;

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

using var writer = new StreamWriter(outputPath, append: false);
for (var gameId = 1; gameId <= options.Games; gameId++)
{
    var state = engine.CreateInitialState();
    var rows = new List<SelfPlayRow>();

    for (var moveIndex = 0; moveIndex < options.MaxMoves && IsPlaying(state.Status); moveIndex++)
    {
        var legalMoves = engine.GetLegalMoves(state.Board, state.Turn, state.History);
        if (legalMoves.Count == 0) break;

        var result = ai.ChooseMove(state.Board, state.Turn, options.Level, state.History, options.TimeMs);
        if (result.Move is null) break;

        rows.Add(new SelfPlayRow(
            gameId,
            moveIndex,
            state.Turn.ToString(),
            BoardEncoder.Encode(state.Board),
            legalMoves.Select(MoveEncoder.Encode).ToArray(),
            MoveEncoder.Encode(result.Move),
            0,
            result.Stats.BestScore,
            ToRedPerspectiveScore(result.Stats.BestScore, state.Turn),
            result.Stats.DepthReached,
            result.Stats.Nodes,
            result.Stats.TimeMs));

        state = engine.MakeMove(state, result.Move);
    }

    var unfinished = IsPlaying(state.Status);
    var finalResult = GetRedPerspectiveResult(state.Status);
    if (finalResult == 1) redWins++;
    else if (finalResult == -1) blackWins++;
    else draws++;

    foreach (var row in rows)
    {
        await writer.WriteLineAsync(JsonSerializer.Serialize(row with { Result = finalResult, Unfinished = unfinished }, jsonOptions));
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
Console.WriteLine($"Output: {outputPath}");

static bool IsPlaying(GameStatus status) => status is GameStatus.Playing or GameStatus.RedCheck or GameStatus.BlackCheck;

static int GetRedPerspectiveResult(GameStatus status) => status switch
{
    GameStatus.RedWins => 1,
    GameStatus.BlackWins => -1,
    _ => 0
};

static string FormatResult(int result) => result switch
{
    1 => "Red win",
    -1 => "Black win",
    _ => "Draw/unfinished"
};

internal sealed record SelfPlayOptions(int Games, string OutputPath, int Level, int TimeMs, int MaxMoves)
{
    public static SelfPlayOptions Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["--games"] = "100",
            ["--out"] = "data/selfplay/games.jsonl",
            ["--level"] = "4",
            ["--timeMs"] = "300",
            ["--maxMoves"] = "300"
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
            ParsePositiveInt(values["--maxMoves"], "--maxMoves"));
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

static double ToRedPerspectiveScore(double score, Side side) => side == Side.Red ? score : -score;
