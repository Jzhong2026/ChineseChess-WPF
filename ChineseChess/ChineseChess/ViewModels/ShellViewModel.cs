using System.IO;
using System.Windows;
using Caliburn.Micro;
using ChineseChess.Ai;
using ChineseChess.Ai.Neural;
using ChineseChess.Messages;
using ChineseChess.Models;
using ChineseChess.Services;

namespace ChineseChess.ViewModels;

public sealed class ShellViewModel :
    Screen,
    IHandle<SquareSelectedMessage>,
    IHandle<SideChangedMessage>,
    IHandle<AiLevelChangedMessage>,
    IHandle<AiTimeLimitChangedMessage>,
    IHandle<RestartRequestedMessage>,
    IHandle<UndoRequestedMessage>,
    IHandle<AiEngineModeChangedMessage>
{
    private readonly IEventAggregator _events;
    private readonly XiangqiEngine _engine;
    private readonly XiangqiAiService _ai;
    private readonly SoundService _sound;
    private IAiEngine? _neuralEngine;
    private GameState _state;
    private Position? _selected;
    private IReadOnlyList<Move> _legalMoves = Array.Empty<Move>();
    private CancellationTokenSource? _aiCancellation;

    // ONNX model path — relative to the application's directory
    private const string OnnxModelPath = "cnn_policy_value.onnx";

    public ShellViewModel(IEventAggregator events, XiangqiEngine engine, XiangqiAiService ai, SoundService sound, BoardViewModel board, SidePanelViewModel sidePanel)
    {
        _events = events;
        _engine = engine;
        _ai = ai;
        _sound = sound;
        Board = board;
        SidePanel = sidePanel;
        _state = _engine.CreateInitialState();
        DisplayName = "中国象棋 WPF";
        events.SubscribeOnUIThread(this);
    }

    public BoardViewModel Board { get; }

    public SidePanelViewModel SidePanel { get; }

    protected override async Task OnActivatedAsync(CancellationToken cancellationToken)
    {
        await base.OnActivatedAsync(cancellationToken);
        await PublishBoardState();
    }

    public async Task HandleAsync(SquareSelectedMessage message, CancellationToken cancellationToken)
    {
        if (IsGameOver || SidePanel.AiThinking || _state.Turn != SidePanel.HumanSide) return;

        var piece = _state.Board[message.Position.Row, message.Position.Col];
        var move = _legalMoves.FirstOrDefault(item => item.To == message.Position);
        if (move is not null)
        {
            await MakeMove(move);
            return;
        }

        if (piece?.Side == SidePanel.HumanSide)
        {
            _selected = message.Position;
            _legalMoves = _engine.GetLegalMovesForPiece(_state.Board, message.Position, _state.History);
            SidePanel.RuleNotice = null;
            await PublishBoardState();
            return;
        }

        if (_selected is not null)
        {
            var selectedPiece = _state.Board[_selected.Value.Row, _selected.Value.Col];
            if (selectedPiece is not null)
            {
                var candidateMove = new Move(_selected.Value, message.Position, selectedPiece, piece);
                if (_engine.IsRepeatCheckMove(_state.Board, candidateMove, SidePanel.HumanSide, _state.History))
                {
                    SidePanel.RuleNotice = "重复将军不允许，请改走其他着法。";
                    await PublishBoardState();
                    return;
                }
            }
        }

        _selected = null;
        _legalMoves = Array.Empty<Move>();
        SidePanel.RuleNotice = null;
        await PublishBoardState();
    }

    public async Task HandleAsync(SideChangedMessage message, CancellationToken cancellationToken)
    {
        SidePanel.HumanSide = message.Side;
        await ResetGame();
    }

    public async Task HandleAsync(AiLevelChangedMessage message, CancellationToken cancellationToken)
    {
        SidePanel.AiLevel = message.Level;
        await ResetGame();
    }

    public Task HandleAsync(AiTimeLimitChangedMessage message, CancellationToken cancellationToken)
    {
        SidePanel.AiTimeLimitMs = message.TimeLimitMs;
        return Task.CompletedTask;
    }

    public async Task HandleAsync(RestartRequestedMessage message, CancellationToken cancellationToken)
    {
        await ResetGame();
    }

    public async Task HandleAsync(AiEngineModeChangedMessage message, CancellationToken cancellationToken)
    {
        SidePanel.AiEngineMode = message.Mode;
        DisposeNeuralEngine();
        if (message.Mode != AiEngineMode.Classic)
        {
            await TryInitNeuralAiAsync();
        }
        await ResetGame();
    }

    protected override async Task OnDeactivateAsync(bool close, CancellationToken cancellationToken)
    {
        DisposeNeuralEngine();
        await base.OnDeactivateAsync(close, cancellationToken);
    }

    public async Task HandleAsync(UndoRequestedMessage message, CancellationToken cancellationToken)
    {
        if (_state.History.Count == 0 || SidePanel.AiThinking) return;

        CancelAi();
        var steps = _state.Turn == SidePanel.HumanSide ? 2 : 1;
        var keep = Math.Max(0, _state.History.Count - steps);
        var replay = _state.History.Take(keep).ToList();
        var next = _engine.CreateInitialState();
        foreach (var move in replay)
        {
            var currentPiece = next.Board[move.From.Row, move.From.Col];
            if (currentPiece is not null)
            {
                next = _engine.MakeMove(next, move with { Piece = currentPiece, Captured = next.Board[move.To.Row, move.To.Col] });
            }
        }

        _state = next;
        _selected = null;
        _legalMoves = Array.Empty<Move>();
        SidePanel.RuleNotice = null;
        SidePanel.SearchStats = null;
        await PublishBoardState();
        await MaybeStartAiTurn();
    }

    private bool IsGameOver => _state.Status is GameStatus.RedWins or GameStatus.BlackWins or GameStatus.Draw;

    private async Task TryInitNeuralAiAsync()
    {
        var kind = ToEngineKind(SidePanel.AiEngineMode);
        if (kind == AiEngineKind.Classic) return;
        var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, OnnxModelPath);
        if (!File.Exists(modelPath))
        {
            SidePanel.NeuralAiInfo = $"⚠ 模型文件不存在: {OnnxModelPath}";
            return;
        }

        SidePanel.NeuralAiInfo = "神经网络加载中…";
        try
        {
            // InferenceSession construction does native model loading and would
            // freeze the UI if it ran on the dispatcher thread.
            var engine = _engine;
            var simulations = SidePanel.MctsSimulations;
            _neuralEngine = await Task.Run(() => AiEngineFactory.Create(kind, engine, modelPath, simulations));
            SidePanel.NeuralAiInfo = "神经网络已加载";
        }
        catch (Exception ex)
        {
            SidePanel.NeuralAiInfo = $"⚠ 加载失败: {ex.Message}";
        }
    }

    private static AiEngineKind ToEngineKind(AiEngineMode mode) => mode switch
    {
        AiEngineMode.Neural => AiEngineKind.NeuralPolicy,
        AiEngineMode.NeuralMcts => AiEngineKind.NeuralMcts,
        _ => AiEngineKind.Classic,
    };

    private void DisposeNeuralEngine()
    {
        (_neuralEngine as IDisposable)?.Dispose();
        _neuralEngine = null;
    }

    private async Task ResetGame()
    {
        CancelAi();
        _state = _engine.CreateInitialState();
        _selected = null;
        _legalMoves = Array.Empty<Move>();
        SidePanel.AiThinking = false;
        SidePanel.AiCountdown = 0;
        SidePanel.RuleNotice = null;
        SidePanel.SearchStats = null;
        await PublishBoardState();
        await MaybeStartAiTurn();
    }

    private async Task MakeMove(Move move)
    {
        _state = _engine.MakeMove(_state, move);
        _sound.PlayForStatus(_state.Status);
        _selected = null;
        _legalMoves = Array.Empty<Move>();
        SidePanel.RuleNotice = null;
        await PublishBoardState();
        await MaybeStartAiTurn();
    }

    private async Task MaybeStartAiTurn()
    {
        if (IsGameOver || _state.Turn == SidePanel.HumanSide) return;

        CancelAi();
        var tokenSource = new CancellationTokenSource();
        _aiCancellation = tokenSource;
        var token = tokenSource.Token;
        var snapshot = _state;
        var thinkMs = Math.Max(100, SidePanel.AiTimeLimitMs);
        SidePanel.AiThinking = true;
        SidePanel.AiCountdown = thinkMs;
        await PublishBoardState();

        try
        {
            // Search starts immediately and the countdown runs against the same
            // budget, so the displayed time reflects real thinking time. The
            // countdown only touches its own bound property — republishing the
            // whole board every 100ms rebuilt the move list and all 90 squares.
            var searchTask = Task.Run(() => RunAiSearch(snapshot), token);
            for (var remaining = thinkMs; remaining > 0 && !searchTask.IsCompleted; remaining -= 100)
            {
                token.ThrowIfCancellationRequested();
                SidePanel.AiCountdown = remaining;
                await Task.Delay(100, token);
            }

            var outcome = await searchTask;
            token.ThrowIfCancellationRequested();

            SidePanel.SearchStats = outcome.Stats;
            SidePanel.NeuralAiInfo = outcome.NeuralInfo;
            SidePanel.AiCountdown = 0;

            if (outcome.Move is not null)
            {
                _state = _engine.MakeMove(snapshot, outcome.Move);
                _sound.PlayForStatus(_state.Status);
            }
        }
        catch (OperationCanceledException)
        {
            return;
        }
        finally
        {
            if (_aiCancellation == tokenSource)
            {
                SidePanel.AiThinking = false;
                SidePanel.AiCountdown = 0;
                _aiCancellation = null;
                tokenSource.Dispose();
            }
        }

        await PublishBoardState();
        if (IsGameOver)
        {
            MessageBox.Show(GameOverText(), "对局结束", MessageBoxButton.OK, MessageBoxImage.Information);
        }
    }

    private sealed record AiTurnOutcome(Move? Move, SearchStats? Stats, string? NeuralInfo);

    /// <summary>Runs on a worker thread; returns results so the UI thread applies them.</summary>
    private AiTurnOutcome RunAiSearch(GameState snapshot)
    {
        var mode = SidePanel.AiEngineMode;
        // Depend on the abstraction: any IAiEngine works here, no per-engine branch.
        IAiEngine engine = mode == AiEngineMode.Classic || _neuralEngine is null ? _ai : _neuralEngine;
        var request = new AiMoveRequest(snapshot.Board, snapshot.Turn, snapshot.History,
            SidePanel.AiTimeLimitMs, Difficulty: SidePanel.AiLevel);
        var result = engine.ChooseMove(request);

        string? info = mode switch
        {
            AiEngineMode.Neural when _neuralEngine is not null => $"NN Policy | Value: {result.Stats.Q:+0.00;-0.00;0.00}",
            AiEngineMode.NeuralMcts when _neuralEngine is not null => $"MCTS {result.Stats.Sims} sims | Q: {result.Stats.Q:+0.00;-0.00;0.00}",
            AiEngineMode.Classic => null,
            // A neural mode was requested but is unavailable: keep the reason visible.
            _ => "⚠ 神经网络不可用，已回退传统搜索",
        };
        var stats = mode == AiEngineMode.Classic ? result.Stats : null;
        return new AiTurnOutcome(result.Move, stats, info);
    }

    private string GameOverText() => _state.Status switch
    {
        GameStatus.RedWins => SidePanel.HumanSide == Side.Red ? "你赢了，红方胜。" : "AI 获胜，红方胜。",
        GameStatus.BlackWins => SidePanel.HumanSide == Side.Black ? "你赢了，黑方胜。" : "AI 获胜，黑方胜。",
        GameStatus.Draw => "本局和棋。",
        _ => ""
    };

    private async Task PublishBoardState()
    {
        var legalTargets = _legalMoves.Select(move => move.To).ToHashSet();
        var lastMove = _state.History.LastOrDefault();
        await _events.PublishOnUIThreadAsync(new BoardStateChangedMessage(_state, _selected, legalTargets, lastMove));
    }

    private void CancelAi()
    {
        _aiCancellation?.Cancel();
        _aiCancellation?.Dispose();
        _aiCancellation = null;
    }
}
