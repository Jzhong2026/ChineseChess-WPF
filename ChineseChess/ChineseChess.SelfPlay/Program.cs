using System.Globalization;
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
var unfinishedGames = 0;
var adjudicatedGames = 0;
var skippedDrawGames = 0;
var skippedUnfinishedGames = 0;
var totalRows = 0;
var uniqueSignatures = new HashSet<string>(StringComparer.Ordinal);
var duplicateGames = 0;
var writePath = options.AtomicOutput ? GetAtomicWritePath(outputPath) : outputPath;

if (options.AtomicOutput && File.Exists(writePath))
{
    File.Delete(writePath);
}

var writer = new StreamWriter(writePath, append: false);
try
{
    for (var gameId = 1; gameId <= options.Games; gameId++)
    {
        var state = engine.CreateInitialState();
        var rows = new List<SelfPlayRow>();
        var selectedMoves = new List<int>();
        var noCapturePlies = 0;
        var adjudicatedResult = 0;
        var endReason = "unfinished";

        for (var moveIndex = 0; moveIndex < options.MaxMoves && IsPlaying(state.Status); moveIndex++)
        {
            var legalMoves = engine.GetLegalMoves(state.Board, state.Turn, state.History);
            if (legalMoves.Count == 0) break;

            var selectionOptions = BuildMoveSelectionOptions(options, moveIndex);
            var result = ai.ChooseMove(state.Board, state.Turn, options.Level, state.History, options.TimeMs, selectionOptions);
            if (result.Move is null) break;

            var selectedMove = MoveEncoder.Encode(result.Move);

            rows.Add(new SelfPlayRow(
                gameId,
                moveIndex,
                state.Turn.ToString(),
                SideToMoveValue(state.Turn),
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
            var captured = state.Board[result.Move.To.Row, result.Move.To.Col];
            state = engine.MakeMove(state, result.Move);
            noCapturePlies = captured is null ? noCapturePlies + 1 : 0;

            if (TryAdjudicateByMaterial(state, moveIndex + 1, noCapturePlies, options, out adjudicatedResult))
            {
                endReason = "material-adjudication";
                break;
            }
        }

        var signature = string.Join(',', selectedMoves);
        if (!uniqueSignatures.Add(signature)) duplicateGames++;

        var finalResult = adjudicatedResult != 0 ? adjudicatedResult : GetRedPerspectiveResult(state.Status);
        var unfinished = false;
        if (adjudicatedResult != 0)
        {
            adjudicatedGames++;
        }
        else if (IsPlaying(state.Status))
        {
            unfinished = true;
            endReason = rows.Count >= options.MaxMoves ? "max-moves" : "interrupted";
            if (options.AdjudicateAtMaxMoves && TryAdjudicateAtMaxMoves(state, rows.Count, options, out finalResult))
            {
                unfinished = false;
                adjudicatedGames++;
                endReason = "max-moves-material-adjudication";
            }

            if (unfinished) unfinishedGames++;
        }
        else
        {
            endReason = state.Status.ToString();
        }

        if (finalResult == 1) redWins++;
        else if (finalResult == -1) blackWins++;
        else draws++;

        if (options.SkipUnfinished && unfinished)
        {
            skippedUnfinishedGames++;
            Console.WriteLine($"Game {gameId}/{options.Games}: skipped unfinished {FormatResult(finalResult)} ({endReason}), rows {rows.Count}");
            continue;
        }

        if (options.SkipDraws && finalResult == 0)
        {
            skippedDrawGames++;
            Console.WriteLine($"Game {gameId}/{options.Games}: skipped {FormatResult(finalResult)} ({endReason}), rows {rows.Count}");
            continue;
        }

        foreach (var row in rows)
        {
            var valueWeight = GetValueWeight(finalResult, unfinished, endReason);
            var policyWeight = GetPolicyWeight(row.DepthReached);
            var finalizedRow = row with
            {
                Result = finalResult,
                ValueWeight = valueWeight,
                PolicyWeight = policyWeight,
                UseForValueTraining = valueWeight > 0,
                UseForPolicyTraining = policyWeight > 0,
                Unfinished = unfinished,
                EndReason = endReason
            };
            ValidateRow(finalizedRow);
            await writer.WriteLineAsync(JsonSerializer.Serialize(finalizedRow, jsonOptions));
        }

        totalRows += rows.Count;
        Console.WriteLine($"Game {gameId}/{options.Games}: {FormatResult(finalResult)} ({endReason}), rows {rows.Count}");
    }
}
finally
{
    writer.Dispose();
}

if (options.AtomicOutput)
{
    File.Move(writePath, outputPath, overwrite: true);
}

Console.WriteLine("Self-play complete.");
Console.WriteLine($"Games requested: {options.Games}");
Console.WriteLine($"Red wins: {redWins}");
Console.WriteLine($"Black wins: {blackWins}");
Console.WriteLine($"Draws: {draws}");
Console.WriteLine($"Unfinished games: {unfinishedGames}");
Console.WriteLine($"Adjudicated games: {adjudicatedGames}");
Console.WriteLine($"Skipped draw games: {skippedDrawGames}");
Console.WriteLine($"Skipped unfinished games: {skippedUnfinishedGames}");
Console.WriteLine($"Total rows written: {totalRows}");
Console.WriteLine($"Duplicate games: {duplicateGames}");
Console.WriteLine($"Unique signatures: {uniqueSignatures.Count}");
Console.WriteLine($"Output: {outputPath}");

static MoveSelectionOptions BuildMoveSelectionOptions(SelfPlayOptions options, int moveIndex)
{
    var openingRandom = moveIndex < options.RandomOpeningPlies;
    var enableRandomSelection = openingRandom || (options.TopK > 1 || options.NearBestWindow > 0);
    return new MoveSelectionOptions(
        EnableRandomSelection: enableRandomSelection,
        TopK: openingRandom ? Math.Max(options.TopK, options.OpeningTopK) : options.TopK,
        NearBestWindow: openingRandom ? Math.Max(options.NearBestWindow, options.OpeningNearBestWindow) : options.NearBestWindow,
        Seed: options.Seed,
        RandomOpeningPlies: options.RandomOpeningPlies);
}

static bool TryAdjudicateByMaterial(GameState state, int playedPlies, int noCapturePlies, SelfPlayOptions options, out int result)
{
    result = 0;
    if (!options.EnableMaterialAdjudication) return false;
    if (playedPlies < options.AdjudicateAfterMoves) return false;
    if (noCapturePlies < options.AdjudicateNoCapturePlies) return false;
    return TryAdjudicateMaterialMargin(state.Board, options.AdjudicateMaterialMargin, out result);
}

static bool TryAdjudicateAtMaxMoves(GameState state, int playedPlies, SelfPlayOptions options, out int result)
{
    result = 0;
    if (!options.AdjudicateAtMaxMoves) return false;
    if (playedPlies < options.MaxMoves) return false;
    return TryAdjudicateMaterialMargin(state.Board, options.AdjudicateMaterialMargin, out result);
}

static bool TryAdjudicateMaterialMargin(Piece?[,] board, int threshold, out int result)
{
    var margin = MaterialScore(board);
    if (Math.Abs(margin) < threshold)
    {
        result = 0;
        return false;
    }

    result = margin > 0 ? 1 : -1;
    return true;
}

static int MaterialScore(Piece?[,] board)
{
    var score = 0;
    for (var row = 0; row < XiangqiEngine.BoardRows; row++)
    {
        for (var col = 0; col < XiangqiEngine.BoardCols; col++)
        {
            var piece = board[row, col];
            if (piece is null) continue;
            var value = PieceMaterialValue(piece.Type);
            score += piece.Side == Side.Red ? value : -value;
        }
    }

    return score;
}

static int PieceMaterialValue(PieceType type) => type switch
{
    PieceType.General => 10000,
    PieceType.Rook => 900,
    PieceType.Cannon => 450,
    PieceType.Horse => 400,
    PieceType.Elephant => 200,
    PieceType.Advisor => 200,
    PieceType.Soldier => 120,
    _ => 0
};

static void ValidateRow(SelfPlayRow row)
{
    if (row.BoardEncoding.Length != BoardEncoder.EncodedLength) throw new InvalidOperationException($"Invalid BoardEncoding length: {row.BoardEncoding.Length}");
    if (row.SideToMove is not 1 and not -1) throw new InvalidOperationException($"Invalid SideToMove value: {row.SideToMove}");
    if (row.LegalMoves.Length == 0) throw new InvalidOperationException("LegalMoves must not be empty.");
    if (!row.LegalMoves.Contains(row.SelectedMove)) throw new InvalidOperationException($"SelectedMove {row.SelectedMove} not found in LegalMoves.");
    if (row.SelectedMove is < 0 or > 8099) throw new InvalidOperationException($"SelectedMove out of range: {row.SelectedMove}");
    if (row.ValueWeight is < 0 or > 1) throw new InvalidOperationException($"ValueWeight out of range: {row.ValueWeight}");
    if (row.PolicyWeight is < 0 or > 1) throw new InvalidOperationException($"PolicyWeight out of range: {row.PolicyWeight}");
}

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

static double ToRedPerspectiveScore(double score, Side side) => side == Side.Red ? score : -score;

static int SideToMoveValue(Side side) => side == Side.Red ? 1 : -1;

static double GetValueWeight(int result, bool unfinished, string endReason)
{
    if (unfinished) return 0;

    return endReason switch
    {
        nameof(GameStatus.RedWins) or nameof(GameStatus.BlackWins) or nameof(GameStatus.Draw) => 1,
        "material-adjudication" => 0.75,
        "max-moves-material-adjudication" => 0.5,
        _ => result == 0 ? 0.25 : 0.5
    };
}

static double GetPolicyWeight(int depthReached) => depthReached switch
{
    <= 0 => 0.1,
    1 => 0.4,
    2 => 0.7,
    _ => 1
};

static string GetAtomicWritePath(string outputPath)
{
    var directory = Path.GetDirectoryName(outputPath);
    var fileName = Path.GetFileName(outputPath);
    return Path.Combine(directory ?? ".", $".{fileName}.{Environment.ProcessId}.tmp");
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
    int OpeningTopK,
    double OpeningNearBestWindow,
    int? Seed,
    bool EnableMaterialAdjudication,
    int AdjudicateAfterMoves,
    int AdjudicateNoCapturePlies,
    int AdjudicateMaterialMargin,
    bool AdjudicateAtMaxMoves,
    bool SkipDraws,
    bool SkipUnfinished,
    bool AtomicOutput)
{
    public static SelfPlayOptions Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["--games"] = "100",
            ["--out"] = "data/selfplay/games.jsonl",
            ["--level"] = "4",
            ["--timeMs"] = "300",
            ["--maxMoves"] = "220",
            ["--topK"] = "2",
            ["--nearBestWindow"] = "80",
            ["--randomOpeningPlies"] = "12",
            ["--openingTopK"] = "6",
            ["--openingNearBestWindow"] = "180",
            ["--seed"] = string.Empty,
            ["--materialAdjudication"] = "true",
            ["--adjudicateAfterMoves"] = "120",
            ["--adjudicateNoCapturePlies"] = "60",
            ["--adjudicateMaterialMargin"] = "450",
            ["--adjudicateAtMaxMoves"] = "true",
            ["--skipDraws"] = "false",
            ["--skipUnfinished"] = "true",
            ["--atomicOutput"] = "true"
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
            ParsePositiveInt(values["--openingTopK"], "--openingTopK"),
            ParseNonNegativeDouble(values["--openingNearBestWindow"], "--openingNearBestWindow"),
            ParseOptionalInt(values["--seed"], "--seed"),
            ParseBool(values["--materialAdjudication"], "--materialAdjudication"),
            ParsePositiveInt(values["--adjudicateAfterMoves"], "--adjudicateAfterMoves"),
            ParsePositiveInt(values["--adjudicateNoCapturePlies"], "--adjudicateNoCapturePlies"),
            ParsePositiveInt(values["--adjudicateMaterialMargin"], "--adjudicateMaterialMargin"),
            ParseBool(values["--adjudicateAtMaxMoves"], "--adjudicateAtMaxMoves"),
            ParseBool(values["--skipDraws"], "--skipDraws"),
            ParseBool(values["--skipUnfinished"], "--skipUnfinished"),
            ParseBool(values["--atomicOutput"], "--atomicOutput"));
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
        if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) || parsed < 0)
        {
            throw new ArgumentException($"{name} must be a non-negative number.");
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

    private static bool ParseBool(string value, string name)
    {
        if (!bool.TryParse(value, out var parsed))
        {
            throw new ArgumentException($"{name} must be true or false.");
        }

        return parsed;
    }
}

internal sealed record SelfPlayRow(
    int GameId,
    int MoveIndex,
    string Side,
    int SideToMove,
    float[] BoardEncoding,
    int[] LegalMoves,
    int SelectedMove,
    int Result,
    double SearchScoreSidePerspective,
    double SearchScoreRedPerspective,
    int DepthReached,
    int Nodes,
    double TimeMs,
    double ValueWeight = 1,
    double PolicyWeight = 1,
    bool UseForValueTraining = true,
    bool UseForPolicyTraining = true,
    bool Unfinished = false,
    string EndReason = "unfinished");
