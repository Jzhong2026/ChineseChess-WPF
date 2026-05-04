using System.Collections.ObjectModel;
using Caliburn.Micro;
using ChineseChess.Messages;
using ChineseChess.Models;
using ChineseChess.Services;

namespace ChineseChess.ViewModels;

public sealed class SidePanelViewModel : Screen, IHandle<BoardStateChangedMessage>
{
    private readonly IEventAggregator _events;
    private readonly XiangqiEngine _engine;
    private Side _humanSide = Side.Red;
    private int _aiLevel = 2;
    private int _aiTimeLimitMs = 1200;
    private bool _aiThinking;
    private int _aiCountdown;
    private string? _ruleNotice;
    private SearchStats? _searchStats;
    private string _statusText = "";
    private string _statusDetail = "";
    private bool _isRedTurn = true;

    public SidePanelViewModel(IEventAggregator events, XiangqiEngine engine)
    {
        _events = events;
        _engine = engine;
        History = new ObservableCollection<MoveHistoryItemViewModel>();
        events.SubscribeOnUIThread(this);
    }

    public ObservableCollection<MoveHistoryItemViewModel> History { get; }

    public Side HumanSide
    {
        get => _humanSide;
        set
        {
            if (Set(ref _humanSide, value))
            {
                NotifyOfPropertyChange(nameof(IsHumanRed));
                NotifyOfPropertyChange(nameof(IsHumanBlack));
            }
        }
    }

    public int AiLevel
    {
        get => _aiLevel;
        set
        {
            if (Set(ref _aiLevel, value))
            {
                NotifyOfPropertyChange(nameof(IsLevel1));
                NotifyOfPropertyChange(nameof(IsLevel2));
                NotifyOfPropertyChange(nameof(IsLevel3));
                NotifyOfPropertyChange(nameof(IsLevel4));
            }
        }
    }

    public int AiTimeLimitMs
    {
        get => _aiTimeLimitMs;
        set => Set(ref _aiTimeLimitMs, value);
    }

    public double AiTimeLimitSeconds
    {
        get => AiTimeLimitMs / 1000.0;
        set
        {
            var ms = (int)Math.Round(value * 1000);
            if (ms == AiTimeLimitMs) return;
            AiTimeLimitMs = ms;
            NotifyOfPropertyChange();
            _ = _events.PublishOnUIThreadAsync(new AiTimeLimitChangedMessage(ms));
        }
    }

    public bool AiThinking
    {
        get => _aiThinking;
        set => Set(ref _aiThinking, value);
    }

    public int AiCountdown
    {
        get => _aiCountdown;
        set => Set(ref _aiCountdown, value);
    }

    public string? RuleNotice
    {
        get => _ruleNotice;
        set => Set(ref _ruleNotice, value);
    }

    public SearchStats? SearchStats
    {
        get => _searchStats;
        set
        {
            if (Set(ref _searchStats, value))
            {
                NotifyOfPropertyChange(nameof(HasSearchStats));
                NotifyOfPropertyChange(nameof(SearchDepthText));
                NotifyOfPropertyChange(nameof(SearchNodesText));
                NotifyOfPropertyChange(nameof(SearchTimeText));
                NotifyOfPropertyChange(nameof(SearchScoreText));
                NotifyOfPropertyChange(nameof(SearchTtHitsText));
                NotifyOfPropertyChange(nameof(SearchTtStoresText));
                NotifyOfPropertyChange(nameof(SearchTtBestMoveHitsText));
                NotifyOfPropertyChange(nameof(SearchTtScoreHitsText));
            }
        }
    }

    public string StatusText
    {
        get => _statusText;
        set => Set(ref _statusText, value);
    }

    public string StatusDetail
    {
        get => _statusDetail;
        set => Set(ref _statusDetail, value);
    }

    public bool IsRedTurn
    {
        get => _isRedTurn;
        set => Set(ref _isRedTurn, value);
    }

    public bool IsHumanRed => HumanSide == Side.Red;
    public bool IsHumanBlack => HumanSide == Side.Black;
    public bool IsLevel1 => AiLevel == 1;
    public bool IsLevel2 => AiLevel == 2;
    public bool IsLevel3 => AiLevel == 3;
    public bool IsLevel4 => AiLevel == 4;
    public bool HasSearchStats => SearchStats is not null;
    public string SearchDepthText => SearchStats?.DepthReached.ToString() ?? "-";
    public string SearchNodesText => SearchStats?.Nodes.ToString("N0") ?? "-";
    public string SearchTimeText => SearchStats is null ? "-" : $"{Math.Round(SearchStats.TimeMs)} ms";
    public string SearchScoreText => SearchStats is null ? "-" : Math.Round(SearchStats.BestScore).ToString();
    public string SearchTtHitsText => SearchStats?.TtHits.ToString("N0") ?? "-";
    public string SearchTtStoresText => SearchStats?.TtStores.ToString("N0") ?? "-";
    public string SearchTtBestMoveHitsText => SearchStats?.TtBestMoveHits.ToString("N0") ?? "-";
    public string SearchTtScoreHitsText => SearchStats?.TtScoreHits.ToString("N0") ?? "-";

    public async Task ChooseRed() => await _events.PublishOnUIThreadAsync(new SideChangedMessage(Side.Red));
    public async Task ChooseBlack() => await _events.PublishOnUIThreadAsync(new SideChangedMessage(Side.Black));
    public async Task ChooseLevel1() => await _events.PublishOnUIThreadAsync(new AiLevelChangedMessage(1));
    public async Task ChooseLevel2() => await _events.PublishOnUIThreadAsync(new AiLevelChangedMessage(2));
    public async Task ChooseLevel3() => await _events.PublishOnUIThreadAsync(new AiLevelChangedMessage(3));
    public async Task ChooseLevel4() => await _events.PublishOnUIThreadAsync(new AiLevelChangedMessage(4));
    public async Task Restart() => await _events.PublishOnUIThreadAsync(new RestartRequestedMessage());
    public async Task Undo() => await _events.PublishOnUIThreadAsync(new UndoRequestedMessage());

    public Task HandleAsync(BoardStateChangedMessage message, CancellationToken cancellationToken)
    {
        IsRedTurn = message.State.Turn == Side.Red;
        StatusText = message.State.Status switch
        {
            GameStatus.RedCheck => "红方被将军",
            GameStatus.BlackCheck => "黑方被将军",
            GameStatus.RedWins => "红方胜",
            GameStatus.BlackWins => "黑方胜",
            GameStatus.Draw => "和棋",
            _ => message.State.Turn == Side.Red ? "红方行棋" : "黑方行棋"
        };
        StatusDetail = AiThinking
            ? $"AI 思考中，{AiCountdown / 1000.0:0.0} 秒"
            : message.State.Turn == HumanSide ? "轮到你走棋" : "等待 AI 落子";

        History.Clear();
        var numbered = message.State.History
            .Select((move, index) => new MoveHistoryItemViewModel(index + 1, _engine.MoveNotation(move)))
            .Reverse();
        foreach (var item in numbered) History.Add(item);
        return Task.CompletedTask;
    }
}
