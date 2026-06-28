using System.Diagnostics;
using ChineseChess.Encoding;
using ChineseChess.Models;
using ChineseChess.Services;

// ─── Configuration (overridable via env vars or args) ────────────────────────

int gamesToPlay = ParseInt("CHESS_GAMES", 10);
string onnxModelPath = Environment.GetEnvironmentVariable("CHESS_MODEL") ?? "cnn_policy_value.onnx";
int hardAiLevel = ParseInt("CHESS_LEVEL", 4);     // Level 4 = 8-ply Alpha-Beta + QSearch
int hardAiTimeMs = ParseInt("CHESS_CLASSIC_MS", 5000);
bool useMcts = ParseBool("CHESS_USE_MCTS", true);
int mctsSimulations = ParseInt("CHESS_MCTS_SIMS", 3000);
int mctsTimeLimitMs = ParseInt("CHESS_MCTS_MS", 5000);
bool verbosePerMove = ParseBool("CHESS_VERBOSE", false);
int maxMoves = ParseInt("CHESS_MAX_MOVES", 300);

// ─── Results ─────────────────────────────────────────────────────────────────

int onnxWins = 0, classicWins = 0, draws = 0, totalGames = 0;
var allResults = new List<(int Game, string OnnxSide, string Result, int Moves, string Reason)>();
var diag = new List<DiagRow>();

Console.WriteLine("═".PadRight(72, '═'));
Console.WriteLine("  中国象棋 AI 对弈测试 — qsearch+TT(经典) vs MCTS+NN");
Console.WriteLine("═".PadRight(72, '═'));
Console.WriteLine();
Console.WriteLine($"  ONNX 模型: {onnxModelPath}");
Console.WriteLine($"  经典 AI:   Level {hardAiLevel} ({(hardAiLevel == 4 ? 8 : 4)}-ply Alpha-Beta + QSearch + TT)");
Console.WriteLine($"  思考时间:   经典 AI 每步 {hardAiTimeMs}ms");
Console.WriteLine($"  NN 模式:    {(useMcts ? $"MCTS + 神经网络 ({mctsSimulations} sims, {mctsTimeLimitMs}ms)" : "纯神经网络策略")}");
Console.WriteLine($"  总局数:     {gamesToPlay} 局(轮流执红/黑)");
Console.WriteLine();

// ─── Setup ────────────────────────────────────────────────────────────────────

var engine = new XiangqiEngine();
var classicAi = new XiangqiAiService(engine);

if (!File.Exists(onnxModelPath))
{
    Console.WriteLine($"❌ ONNX 模型文件未找到: {onnxModelPath}");
    Console.WriteLine($"   当前目录: {Environment.CurrentDirectory}");
    return;
}

Console.Write("正在加载 ONNX 模型... ");
var neuralAi = new XiangqiNeuralAiService(onnxModelPath, engine, isCnnModel: true);
MctsAiService? mctsAi = useMcts ? new MctsAiService(neuralAi, engine) : null;
Console.WriteLine($"✅ ({(neuralAi.IsFactoredModel ? "Factored" : "Flat")}, {(neuralAi.IsV5Model ? "V5/16平面" : "Legacy/14平面")})");
Console.WriteLine();

// ─── Main game loop ──────────────────────────────────────────────────────────

for (int gameIdx = 0; gameIdx < gamesToPlay; gameIdx++)
{
    var onnxSide = gameIdx % 2 == 0 ? Side.Red : Side.Black;
    var classicSide = engine.Opposite(onnxSide);
    var onnxName = onnxSide == Side.Red ? "MCTS(红)" : "MCTS(黑)";
    var classicName = classicSide == Side.Red ? "Classic(红)" : "Classic(黑)";

    Console.Write($"第 {gameIdx + 1,2} 局 — {onnxName} vs {classicName} ... ");

    var state = engine.CreateInitialState();
    int moveCount = 0;
    string? gameResult = null;
    string? endReason = null;
    var moveTimes = new List<(int MoveNum, string Player, double TimeMs)>();
    var classicStats = new List<(int Depth, int Nodes, double TimeMs, int TtHits, int TtStores, int TtBestHits)>();
    var mctsStats = new List<(int Sims, double TimeMs, float BestQ)>();

    var stopwatch = Stopwatch.StartNew();
    var recentBoards = new HashSet<string>();
    int repeatCount = 0;

    while (moveCount < maxMoves && gameResult is null)
    {
        Move? chosenMove;
        var turnStart = Stopwatch.StartNew();

        if (state.Turn == onnxSide)
        {
            // ── MCTS+NN side ──
            if (mctsAi is not null)
            {
                var result = mctsAi.Search(state.Board, state.Turn, state.History,
                    simulations: mctsSimulations, temperature: 0.05f, timeLimitMs: mctsTimeLimitMs);
                chosenMove = result.BestMove;
                mctsStats.Add((result.SimulationsRun, turnStart.Elapsed.TotalMilliseconds, result.BestQ));
                if (verbosePerMove)
                {
                    Console.WriteLine($"    [MCTS 步 {moveCount}] sims={result.SimulationsRun} time={turnStart.Elapsed.TotalMilliseconds:F0}ms Q={result.BestQ:F3}");
                    foreach (var s in result.MoveStats.OrderByDescending(s => s.Visits).Take(3))
                        Console.WriteLine($"        → {s.Move.From.Row},{s.Move.From.Col}->{s.Move.To.Row},{s.Move.To.Col} visits={s.Visits} Q={s.Q:F3}");
                }
            }
            else
            {
                chosenMove = neuralAi.ChooseMoveByPolicy(state.Board, state.Turn, state.History);
            }
        }
        else
        {
            // ── Classic AI side ──
            var result = classicAi.ChooseMove(state.Board, state.Turn, hardAiLevel, state.History, hardAiTimeMs);
            chosenMove = result.Move;
            var s = result.Stats;
            classicStats.Add((s.DepthReached, s.Nodes, turnStart.Elapsed.TotalMilliseconds, s.TtHits, s.TtStores, s.TtBestMoveHits));
            if (verbosePerMove)
            {
                Console.WriteLine($"    [Cls 步 {moveCount}] depth={s.DepthReached} nodes={s.Nodes} time={turnStart.Elapsed.TotalMilliseconds:F0}ms ttHits={s.TtHits}");
            }
        }

        if (chosenMove is null)
        {
            gameResult = state.Turn == onnxSide ? "Classic" : "MCTS";
            endReason = $"{state.Turn} 无合法着法";
            break;
        }

        state = engine.MakeMove(state, chosenMove);
        moveCount++;

        if (state.Status is GameStatus.RedWins or GameStatus.BlackWins)
        {
            var winner = state.Status == GameStatus.RedWins ? Side.Red : Side.Black;
            gameResult = winner == onnxSide ? "MCTS" : "Classic";
            endReason = winner == Side.Red ? "红方将杀" : "黑方将杀";
            break;
        }

        if (state.Status == GameStatus.Draw)
        {
            gameResult = "Draw";
            endReason = "和棋";
            break;
        }

        // 三次重复局面
        var boardKey = BoardFingerprint(state.Board, state.Turn);
        if (!recentBoards.Add(boardKey))
        {
            repeatCount++;
            if (repeatCount >= 3)
            {
                gameResult = "Draw";
                endReason = "三次重复局面";
                break;
            }
        }
        else repeatCount = 0;
    }

    if (gameResult is null)
    {
        gameResult = "Draw";
        endReason = $"超过 {maxMoves} 步限制";
    }

    stopwatch.Stop();

    totalGames++;
    if (gameResult == "MCTS") onnxWins++;
    else if (gameResult == "Classic") classicWins++;
    else draws++;

    allResults.Add((gameIdx + 1, onnxName, gameResult, moveCount, endReason!));

    // 诊断汇总
    double avgClsMs = classicStats.Count == 0 ? 0 : classicStats.Average(s => s.TimeMs);
    double avgMctsMs = mctsStats.Count == 0 ? 0 : mctsStats.Average(s => s.TimeMs);
    double avgDepth = classicStats.Count == 0 ? 0 : classicStats.Average(s => s.Depth);
    double avgNodes = classicStats.Count == 0 ? 0 : classicStats.Average(s => s.Nodes);
    int totalTtHits = classicStats.Sum(s => s.TtHits);
    int totalTtStores = classicStats.Sum(s => s.TtStores);
    int totalTtBestHits = classicStats.Sum(s => s.TtBestHits);
    int totalNodes = classicStats.Sum(s => s.Nodes);
    double ttHitRate = totalNodes == 0 ? 0 : 100.0 * totalTtHits / totalNodes;
    double ttBestRate = totalNodes == 0 ? 0 : 100.0 * totalTtBestHits / totalNodes;
    double avgMctsSims = mctsStats.Count == 0 ? 0 : mctsStats.Average(s => s.Sims);

    diag.Add(new DiagRow(
        gameIdx + 1, onnxSide == Side.Red, gameResult, moveCount, endReason!,
        classicStats.Count, avgDepth, avgNodes, avgClsMs,
        totalTtHits, totalTtStores, ttHitRate, ttBestRate,
        mctsStats.Count, avgMctsSims, avgMctsMs));

    var sym = gameResult switch { "MCTS" => "✅M", "Classic" => "✅C", _ => "🔷D" };
    Console.WriteLine($"{sym}  {endReason,-14} | {moveCount,3}步 | {stopwatch.Elapsed.TotalSeconds:F0}s");
    Console.WriteLine($"      Classic: 平均深度 {avgDepth:F1} | 平均节点 {avgNodes:F0} | {avgClsMs:F0}ms/步 | TT命中率 {ttHitRate:F1}% | TT最佳着命中率 {ttBestRate:F1}%");
    Console.WriteLine($"      MCTS:    平均 sims {avgMctsSims:F0} | {avgMctsMs:F0}ms/步");
}

// ─── Summary ──────────────────────────────────────────────────────────────────

Console.WriteLine();
Console.WriteLine("═".PadRight(72, '═'));
Console.WriteLine("  对局结果汇总");
Console.WriteLine("═".PadRight(72, '═'));
Console.WriteLine();
Console.WriteLine($"  总局数:      {totalGames}");
Console.WriteLine($"  MCTS 胜:     {onnxWins,2}  ({onnxWins * 100.0 / totalGames,5:F1}%)");
Console.WriteLine($"  Classic 胜:  {classicWins,2}  ({classicWins * 100.0 / totalGames,5:F1}%)");
Console.WriteLine($"  和棋:        {draws,2}  ({draws * 100.0 / totalGames,5:F1}%)");
Console.WriteLine();

Console.WriteLine($"  MCTS 执红:   {allResults.Count(r => r.OnnxSide == "MCTS(红)" && r.Result == "MCTS")}胜 / " +
                  $"{allResults.Count(r => r.OnnxSide == "MCTS(红)" && r.Result == "Classic")}负 / " +
                  $"{allResults.Count(r => r.OnnxSide == "MCTS(红)" && r.Result == "Draw")}和");
Console.WriteLine($"  MCTS 执黑:   {allResults.Count(r => r.OnnxSide == "MCTS(黑)" && r.Result == "MCTS")}胜 / " +
                  $"{allResults.Count(r => r.OnnxSide == "MCTS(黑)" && r.Result == "Classic")}负 / " +
                  $"{allResults.Count(r => r.OnnxSide == "MCTS(黑)" && r.Result == "Draw")}和");
Console.WriteLine();

// 详细诊断
Console.WriteLine("  每局详细诊断:");
Console.WriteLine($"  {"局",-3} {"MCTS方",-10} {"结果",-8} {"步数",-4} {"Cls深度",-7} {"Cls节点",-9} {"Cls时间ms",-10} {"TT命中率",-9} {"TT最佳",-8} {"MCTS sims",-10} {"MCTS ms",-8}");
Console.WriteLine($"  {new string('─', 95)}");
foreach (var d in diag)
{
    var sym = d.Result switch { "MCTS" => "✅M", "Classic" => "✅C", _ => "🔷D" };
    Console.WriteLine($"  {d.Game,-3} {(d.MctsIsRed ? "红" : "黑"),-10} {sym,-8} {d.Moves,-4} {d.AvgDepth,-7:F1} {d.AvgNodes,-9:F0} {d.AvgClsMs,-10:F0} {d.TtHitRate,-9:F1}% {d.TtBestRate,-8:F1}% {d.AvgMctsSims,-10:F0} {d.AvgMctsMs,-8:F0}");
}

// 总体诊断
double oDepth = diag.Where(d => d.ClsMoves > 0).Select(d => d.AvgDepth).DefaultIfEmpty(0).Average();
double oNodes = diag.Where(d => d.ClsMoves > 0).Select(d => d.AvgNodes).DefaultIfEmpty(0).Average();
double oTtHit = diag.Where(d => d.ClsMoves > 0).Select(d => d.TtHitRate).DefaultIfEmpty(0).Average();
double oTtBest = diag.Where(d => d.ClsMoves > 0).Select(d => d.TtBestRate).DefaultIfEmpty(0).Average();
double oMctsSims = diag.Where(d => d.MctsMoves > 0).Select(d => d.AvgMctsSims).DefaultIfEmpty(0).Average();
double oClsMs = diag.Where(d => d.ClsMoves > 0).Select(d => d.AvgClsMs).DefaultIfEmpty(0).Average();
double oMctsMs = diag.Where(d => d.MctsMoves > 0).Select(d => d.AvgMctsMs).DefaultIfEmpty(0).Average();

Console.WriteLine();
Console.WriteLine("  总体诊断:");
Console.WriteLine($"    Classic 平均深度 {oDepth:F1} | 平均节点/步 {oNodes:F0} | 平均思考 {oClsMs:F0}ms");
Console.WriteLine($"    Classic TT 命中率 {oTtHit:F1}% | TT 最佳着命中率 {oTtBest:F1}%");
Console.WriteLine($"    MCTS    平均 sims {oMctsSims:F0} | 平均思考 {oMctsMs:F0}ms");

Console.WriteLine();
Console.WriteLine("═".PadRight(72, '═'));
Console.WriteLine("  对局完成!");
Console.WriteLine("═".PadRight(72, '═'));

// ─── Helpers ─────────────────────────────────────────────────────────────────

static string BoardFingerprint(Piece?[,] board, Side turn)
{
    var sb = new System.Text.StringBuilder();
    sb.Append((int)turn);
    for (int r = 0; r < XiangqiEngine.BoardRows; r++)
        for (int c = 0; c < XiangqiEngine.BoardCols; c++)
        {
            var p = board[r, c];
            if (p is null) sb.Append('.');
            else sb.Append($"{(int)p.Side}{(int)p.Type}");
        }
    return sb.ToString();
}

static int ParseInt(string envName, int fallback)
{
    var v = Environment.GetEnvironmentVariable(envName);
    return int.TryParse(v, out var i) ? i : fallback;
}

static bool ParseBool(string envName, bool fallback)
{
    var v = Environment.GetEnvironmentVariable(envName);
    if (string.IsNullOrEmpty(v)) return fallback;
    return v == "1" || v.Equals("true", StringComparison.OrdinalIgnoreCase);
}

record DiagRow(
    int Game,
    bool MctsIsRed,
    string Result,
    int Moves,
    string Reason,
    int ClsMoves,
    double AvgDepth,
    double AvgNodes,
    double AvgClsMs,
    int TtHits,
    int TtStores,
    double TtHitRate,
    double TtBestRate,
    int MctsMoves,
    double AvgMctsSims,
    double AvgMctsMs);