using System.Diagnostics;
using ChineseChess.Encoding;
using ChineseChess.Models;
using ChineseChess.Services;

// ─── Configuration ───────────────────────────────────────────────────────────

var gamesToPlay = 20;        // Number of games (half as Red, half as Black)
var onnxModelPath = "cnn_policy_value.onnx";
var hardAiLevel = 4;         // Level 4 = 8-ply Alpha-Beta + QSearch
var hardAiTimeMs = 5000;     // 5 seconds per move for classic AI
var useMcts = true;          // true = MCTS+NN, false = pure NN policy
var mctsSimulations = 600;   // MCTS simulations per move
var mctsTimeLimitMs = 5000;  // MCTS time limit per move

// ─── Results ─────────────────────────────────────────────────────────────────

var onnxWins = 0;
var classicWins = 0;
var draws = 0;
var totalGames = 0;

var allResults = new List<(int Game, string OnnxSide, string Result, int Moves, string Reason)>();

Console.WriteLine("═" .PadRight(70, '═'));
Console.WriteLine("  中国象棋 AI 对弈测试 — ONNX 神经网络 vs 经典困难模式 (Level 4)");
Console.WriteLine("═" .PadRight(70, '═'));
Console.WriteLine();
Console.WriteLine($"  ONNX 模型: {onnxModelPath}");
Console.WriteLine($"  经典 AI:   深度 {(hardAiLevel == 4 ? 8 : 4)} 层 Alpha-Beta + QSearch");
Console.WriteLine($"  思考时间:   经典 AI 每步 {hardAiTimeMs}ms");
Console.WriteLine($"  NN 模式:    {(useMcts ? $"MCTS + 神经网络 ({mctsSimulations} sims)" : "纯神经网络策略 (最高 Policy)")}");
Console.WriteLine($"  总局数:     {gamesToPlay} 局（轮流执红/黑）");
Console.WriteLine();

// ─── Setup ────────────────────────────────────────────────────────────────────

var engine = new XiangqiEngine();
var classicAi = new XiangqiAiService(engine);

if (!File.Exists(onnxModelPath))
{
    Console.WriteLine($"❌ 错误: ONNX 模型文件未找到: {onnxModelPath}");
    Console.WriteLine($"   当前目录: {Environment.CurrentDirectory}");
    return;
}

Console.WriteLine("正在加载 ONNX 模型...");
var neuralAi = new XiangqiNeuralAiService(onnxModelPath, engine, isCnnModel: true);
MctsAiService? mctsAi = useMcts ? new MctsAiService(neuralAi, engine) : null;
Console.WriteLine($"  ✅ 模型加载成功 (架构: {(neuralAi.IsFactoredModel ? "Factored" : "Flat")}, {(neuralAi.IsV5Model ? "V5/16平面" : "Legacy/14平面")})");
Console.WriteLine();

// ─── Main game loop ──────────────────────────────────────────────────────────

for (var gameIdx = 0; gameIdx < gamesToPlay; gameIdx++)
{
    var onnxSide = gameIdx % 2 == 0 ? Side.Red : Side.Black;
    var classicSide = engine.Opposite(onnxSide);
    var onnxName = onnxSide == Side.Red ? "ONNX(红)" : "ONNX(黑)";
    var classicName = classicSide == Side.Red ? "Classic(红)" : "Classic(黑)";

    Console.Write($"\n第 {gameIdx + 1,2} 局 — {onnxName} vs {classicName} ... ");

    var state = engine.CreateInitialState();
    var moveCount = 0;
    var maxMoves = 300;        // Safety limit
    var recentBoards = new HashSet<string>();
    var repeatCount = 0;
    string? gameResult = null;
    string? endReason = null;
    var moveTimes = new List<(int MoveNum, string Player, double TimeMs)>();

    var stopwatch = Stopwatch.StartNew();

    while (moveCount < maxMoves && gameResult is null)
    {
        Move? chosenMove;

        if (state.Turn == onnxSide)
        {
            // ── ONNX side ──
            if (mctsAi is not null)
            {
                var sw = Stopwatch.StartNew();
                var result = mctsAi.Search(state.Board, state.Turn, state.History,
                    simulations: mctsSimulations, temperature: 0.1f, timeLimitMs: mctsTimeLimitMs);
                sw.Stop();
                chosenMove = result.BestMove;
                moveTimes.Add((moveCount, "ONNX", sw.Elapsed.TotalMilliseconds));
            }
            else
            {
                var sw = Stopwatch.StartNew();
                chosenMove = neuralAi.ChooseMoveByPolicy(state.Board, state.Turn, state.History);
                sw.Stop();
                moveTimes.Add((moveCount, "ONNX", sw.Elapsed.TotalMilliseconds));
            }
        }
        else
        {
            // ── Classic AI side ──
            var sw = Stopwatch.StartNew();
            var result = classicAi.ChooseMove(state.Board, state.Turn, hardAiLevel, state.History, hardAiTimeMs);
            sw.Stop();
            chosenMove = result.Move;
            moveTimes.Add((moveCount, "Classic", sw.Elapsed.TotalMilliseconds));
        }

        if (chosenMove is null)
        {
            gameResult = state.Turn == onnxSide ? "Classic" : "ONNX";
            endReason = $"{state.Turn} 无合法着法";
            break;
        }

        state = engine.MakeMove(state, chosenMove);
        moveCount++;

        if (state.Status is GameStatus.RedWins or GameStatus.BlackWins)
        {
            var winner = state.Status == GameStatus.RedWins ? Side.Red : Side.Black;
            gameResult = winner == onnxSide ? "ONNX" : "Classic";
            endReason = winner == Side.Red ? "红方将杀" : "黑方将杀";
            break;
        }

        if (state.Status == GameStatus.Draw)
        {
            gameResult = "Draw";
            endReason = "和棋";
            break;
        }

        // Detect three-fold repetition
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
        else
        {
            repeatCount = 0;
        }
    }

    if (gameResult is null)
    {
        gameResult = "Draw";
        endReason = $"超过 {maxMoves} 步限制";
    }

    stopwatch.Stop();

    // ── Tally ──
    totalGames++;
    if (gameResult == "ONNX") onnxWins++;
    else if (gameResult == "Classic") classicWins++;
    else draws++;

    allResults.Add((gameIdx + 1, onnxName, gameResult, moveCount, endReason!));

    var avgOnnxMs = moveTimes.Where(m => m.Player == "ONNX").Select(m => m.TimeMs).DefaultIfEmpty().Average();
    var avgClassicMs = moveTimes.Where(m => m.Player == "Classic").Select(m => m.TimeMs).DefaultIfEmpty().Average();

    Console.WriteLine($"{endReason} | {moveCount}步 | {stopwatch.Elapsed.TotalSeconds:F0}s");
    Console.WriteLine($"          ONNX avg: {avgOnnxMs:F1}ms, Classic avg: {avgClassicMs:F1}ms");
}

// ─── Summary ──────────────────────────────────────────────────────────────────

Console.WriteLine();
Console.WriteLine("═" .PadRight(70, '═'));
Console.WriteLine("  对局结果汇总");
Console.WriteLine("═" .PadRight(70, '═'));
Console.WriteLine();
Console.WriteLine($"  总局数:     {totalGames}");
Console.WriteLine($"  ONNX 胜:    {onnxWins,2}  ({onnxWins * 100.0 / totalGames,5:F1}%)");
Console.WriteLine($"  Classic 胜: {classicWins,2}  ({classicWins * 100.0 / totalGames,5:F1}%)");
Console.WriteLine($"  和棋:       {draws,2}  ({draws * 100.0 / totalGames,5:F1}%)");
Console.WriteLine();

Console.WriteLine($"  ONNX 执红: {allResults.Count(r => r.OnnxSide == "ONNX(红)" && r.Result == "ONNX")}胜 / " +
                  $"{allResults.Count(r => r.OnnxSide == "ONNX(红)" && r.Result == "Classic")}负 / " +
                  $"{allResults.Count(r => r.OnnxSide == "ONNX(红)" && r.Result == "Draw")}和");
Console.WriteLine($"  ONNX 执黑: {allResults.Count(r => r.OnnxSide == "ONNX(黑)" && r.Result == "ONNX")}胜 / " +
                  $"{allResults.Count(r => r.OnnxSide == "ONNX(黑)" && r.Result == "Classic")}负 / " +
                  $"{allResults.Count(r => r.OnnxSide == "ONNX(黑)" && r.Result == "Draw")}和");
Console.WriteLine();

// Detailed game log
Console.WriteLine("  详细对局日志:");
Console.WriteLine($"  {"局号",4} {"ONNX方",12} {"结果",10} {"步数",5} {"原因",-20}");
Console.WriteLine($"  {"".PadRight(55, '─')}");
foreach (var r in allResults)
{
    var resultSymbol = r.Result switch
    {
        "ONNX" => "✅",
        "Classic" => "❌",
        _ => "🔷"
    };
    Console.WriteLine($"  {r.Game,4} {r.OnnxSide,12} {resultSymbol + " " + r.Result,-10} {r.Moves,5} {r.Reason,-20}");
}

Console.WriteLine();
Console.WriteLine("═" .PadRight(70, '═'));
Console.WriteLine("  对局完成!");
Console.WriteLine("═" .PadRight(70, '═'));

// ─── Helpers ─────────────────────────────────────────────────────────────────

static string BoardFingerprint(Piece?[,] board, Side turn)
{
    // Simple fingerprint: piece layout + turn
    var sb = new System.Text.StringBuilder();
    sb.Append((int)turn);
    for (var r = 0; r < XiangqiEngine.BoardRows; r++)
    {
        for (var c = 0; c < XiangqiEngine.BoardCols; c++)
        {
            var p = board[r, c];
            if (p is null) sb.Append('.');
            else sb.Append($"{(int)p.Side}{(int)p.Type}");
        }
    }
    return sb.ToString();
}
