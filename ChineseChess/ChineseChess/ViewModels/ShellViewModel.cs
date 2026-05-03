using System.Windows;
using Caliburn.Micro;
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
    IHandle<UndoRequestedMessage>
{
    private readonly IEventAggregator _events;
    private readonly XiangqiEngine _engine;
    private readonly XiangqiAiService _ai;
    private readonly SoundService _sound;
    private GameState _state;
    private Position? _selected;
    private IReadOnlyList<Move> _legalMoves = Array.Empty<Move>();
    private CancellationTokenSource? _aiCancellation;

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
        SidePanel.AiThinking = true;
        SidePanel.AiCountdown = 1200;
        await PublishBoardState();

        try
        {
            for (var remaining = 1200; remaining > 0; remaining -= 100)
            {
                token.ThrowIfCancellationRequested();
                SidePanel.AiCountdown = remaining;
                await PublishBoardState();
                await Task.Delay(100, token);
            }

            var snapshot = _state;
            var result = await Task.Run(() => _ai.ChooseMove(snapshot.Board, snapshot.Turn, SidePanel.AiLevel, snapshot.History, SidePanel.AiTimeLimitMs), token);
            token.ThrowIfCancellationRequested();
            SidePanel.SearchStats = result.Stats;
            if (result.Move is not null)
            {
                _state = _engine.MakeMove(snapshot, result.Move);
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
            }
        }

        await PublishBoardState();
        if (IsGameOver)
        {
            MessageBox.Show(GameOverText(), "对局结束", MessageBoxButton.OK, MessageBoxImage.Information);
        }
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
