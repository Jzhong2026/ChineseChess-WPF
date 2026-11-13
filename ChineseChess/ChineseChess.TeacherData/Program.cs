using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using ChineseChess.Encoding;
using ChineseChess.Models;
using ChineseChess.Services;

// The teacher is an external, separately licensed OFFLINE data source.
// Neither the WPF application nor the arena references this project.
if (args.Length < 4)
{
    Console.WriteLine("Usage: TeacherData <engine.exe> <output.jsonl> <games> <seed> [nodes=20000] [maxPlies=220]");
    return;
}
var executable = Path.GetFullPath(args[0]);
var output = Path.GetFullPath(args[1]);
var games = int.Parse(args[2]);
var seed = int.Parse(args[3]);
var nodes = args.Length > 4 ? int.Parse(args[4]) : 20000;
var maxPlies = args.Length > 5 ? int.Parse(args[5]) : 220;
if (games < 1 || nodes < 1 || maxPlies < 1) throw new ArgumentException("Positive budgets required");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
// Fail rather than silently replacing another experiment's evidence.
using var writer = new StreamWriter(new FileStream(output, FileMode.CreateNew), new UTF8Encoding(false)) { AutoFlush = true };
var metadata = new
{
    Executable = executable, EngineSha256 = Hash(executable),
    NnueSha256 = Hash(Path.Combine(Path.GetDirectoryName(executable)!, "pikafish.nnue")),
    Games = games, Seed = seed, Nodes = nodes, MaxPlies = maxPlies,
    StartedUtc = DateTime.UtcNow, Threads = 1, HashMiB = 128, MultiPV = 3,
    ValueDefinition = "tanh(teacher_cp/600), side-to-move search estimate, not game outcome",
    HistoryLimitation = "Teacher FEN omits history; searchmoves uses project history-aware legal moves"
};
File.WriteAllText(output + ".metadata.json", JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));
using var process = Process.Start(new ProcessStartInfo(executable)
{
    WorkingDirectory = Path.GetDirectoryName(executable), UseShellExecute = false,
    RedirectStandardInput = true, RedirectStandardOutput = true, CreateNoWindow = true
}) ?? throw new InvalidOperationException("Teacher failed to start");
void Send(string command) { process.StandardInput.WriteLine(command); process.StandardInput.Flush(); }
async Task<string> Read() => await process.StandardOutput.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(30))
    ?? throw new InvalidOperationException("Teacher exited before bestmove");
var engine = new XiangqiEngine();
var random = new Random(seed);
var total = 0;
try
{
    Send("uci");
    while (!(await Read()).StartsWith("uciok")) { }
    Send("setoption name Threads value 1");
    Send("setoption name Hash value 128");
    Send("setoption name MultiPV value 3");
    Send("isready");
    while (!(await Read()).StartsWith("readyok")) { }
    for (var game = 0; game < games; game++)
    {
        var state = engine.CreateInitialState();
        var seen = new Dictionary<string, int>();
        Send("ucinewgame");
        for (var ply = 0; ply < maxPlies; ply++)
        {
            var legal = engine.GetLegalMoves(state.Board, state.Turn, state.History);
            if (legal.Count == 0) break;
            var fen = Fen(state);
            seen[fen] = seen.GetValueOrDefault(fen) + 1;
            if (seen[fen] >= 3) break;
            var byUci = legal.ToDictionary(Uci);
            Send("position fen " + fen);
            Send($"go nodes {nodes} searchmoves " + string.Join(' ', byUci.Keys));
            var scores = new Dictionary<string, (int Cp, int Depth)>();
            var ranks = new Dictionary<int, string>();
            long actualNodes = 0;
            var searchMs = 0;
            string best;
            while (true)
            {
                var line = await Read();
                if (game == 0 && ply < 2) File.AppendAllText(output + ".protocol.txt", line + Environment.NewLine);
                if (line.StartsWith("bestmove ")) { best = line.Split(' ')[1]; break; }
                var pv = Regex.Match(line, @"\bpv ([a-i][0-9][a-i][0-9])");
                var score = Regex.Match(line, @"\bscore (cp|mate) (-?\d+)");
                var depth = Regex.Match(line, @"\bdepth (\d+)");
                if (!pv.Success || !score.Success || !depth.Success) continue;
                var cp = int.Parse(score.Groups[2].Value);
                if (score.Groups[1].Value == "mate") cp = Math.Sign(cp) * 30000;
                scores[pv.Groups[1].Value] = (cp, int.Parse(depth.Groups[1].Value));
                var rank = Regex.Match(line, @"\bmultipv (\d+)");
                if (rank.Success) ranks[int.Parse(rank.Groups[1].Value)] = pv.Groups[1].Value;
                var n = Regex.Match(line, @"\bnodes (\d+)");
                if (n.Success) actualNodes = long.Parse(n.Groups[1].Value);
                var t = Regex.Match(line, @"\btime (\d+)");
                if (t.Success) searchMs = int.Parse(t.Groups[1].Value);
            }
            if (!byUci.TryGetValue(best, out var bestMove) || !scores.TryGetValue(best, out var label))
                throw new InvalidOperationException($"Invalid teacher response {best} at {fen}");
            var candidates = scores.Where(k => ranks.Values.Contains(k.Key) && byUci.ContainsKey(k.Key)).ToArray();
            var next = bestMove;
            if (ply < 12)
            {
                var nearBest = candidates.Where(k => k.Value.Cp >= label.Cp - 100).ToArray();
                if (nearBest.Length > 0) next = byUci[nearBest[random.Next(nearBest.Length)].Key];
            }
            else if (random.NextDouble() < 0.06) next = legal[random.Next(legal.Count)];
            writer.WriteLine(JsonSerializer.Serialize(new
            {
                GameId = $"teacher-{seed}-{game}", MoveIndex = ply, Fen = fen,
                SideToMove = state.Turn == Side.Red ? 1 : -1,
                BoardEncoding = BoardEncoder.Encode(state.Board),
                History = state.History.Select(MoveEncoder.Encode).ToArray(),
                LegalMoves = legal.Select(MoveEncoder.Encode).ToArray(),
                SelectedMove = MoveEncoder.Encode(bestMove), PlayedMove = MoveEncoder.Encode(next),
                TeacherCp = label.Cp, TeacherValue = Math.Tanh(label.Cp / 600.0), Depth = label.Depth,
                NodesBudget = nodes, ActualNodes = actualNodes, SearchMs = searchMs,
                Candidates = candidates.Select(k => new { Action = MoveEncoder.Encode(byUci[k.Key]), Score = k.Value.Cp, Depth = k.Value.Depth }).ToArray()
            }));
            total++;
            state = engine.MakeMove(state, next);
            if (total % 100 == 0) Console.WriteLine($"rows={total} game={game} ply={ply} depth={label.Depth}");
        }
        Console.WriteLine($"game_done={game} rows={total}");
    }
}
finally
{
    if (!process.HasExited)
    {
        Send("quit");
        if (!process.WaitForExit(5000)) process.Kill();
    }
}
Console.WriteLine($"COMPLETE games={games} rows={total}");

static string Hash(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)));
static string Uci(Move move) => $"{(char)('a' + move.From.Col)}{9 - move.From.Row}{(char)('a' + move.To.Col)}{9 - move.To.Row}";
static string Fen(GameState state)
{
    var text = new StringBuilder();
    for (var row = 0; row < 10; row++)
    {
        if (row > 0) text.Append('/');
        var empty = 0;
        for (var col = 0; col < 9; col++)
        {
            var piece = state.Board[row, col];
            if (piece is null) { empty++; continue; }
            if (empty > 0) { text.Append(empty); empty = 0; }
            var ch = piece.Type switch
            {
                PieceType.General => 'k', PieceType.Advisor => 'a', PieceType.Elephant => 'b',
                PieceType.Horse => 'n', PieceType.Rook => 'r', PieceType.Cannon => 'c', _ => 'p'
            };
            text.Append(piece.Side == Side.Red ? char.ToUpperInvariant(ch) : ch);
        }
        if (empty > 0) text.Append(empty);
    }
    return text + (state.Turn == Side.Red ? " w - - 0 1" : " b - - 0 1");
}
