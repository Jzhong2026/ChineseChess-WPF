using System.Collections.ObjectModel;
using Caliburn.Micro;
using ChineseChess.Messages;
using ChineseChess.Models;
using ChineseChess.Services;

namespace ChineseChess.ViewModels;

public sealed class BoardViewModel : Screen, IHandle<BoardStateChangedMessage>
{
    private readonly XiangqiEngine _engine;
    private string? _gameResultText;

    public BoardViewModel(IEventAggregator events, XiangqiEngine engine)
    {
        _engine = engine;
        Squares = new ObservableCollection<SquareViewModel>(
            Enumerable.Range(0, XiangqiEngine.BoardRows)
                .SelectMany(row => Enumerable.Range(0, XiangqiEngine.BoardCols).Select(col => new SquareViewModel(events, row, col))));
        events.SubscribeOnUIThread(this);
    }

    public ObservableCollection<SquareViewModel> Squares { get; }

    public string? GameResultText
    {
        get => _gameResultText;
        private set
        {
            _gameResultText = value;
            NotifyOfPropertyChange();
            NotifyOfPropertyChange(nameof(ShowGameResult));
        }
    }

    public bool ShowGameResult => !string.IsNullOrWhiteSpace(GameResultText);

    public Task HandleAsync(BoardStateChangedMessage message, CancellationToken cancellationToken)
    {
        foreach (var square in Squares)
        {
            var piece = message.State.Board[square.Position.Row, square.Position.Col];
            square.Update(
                piece is null ? null : _engine.PieceLabel(piece),
                piece?.Side,
                message.Selected == square.Position,
                message.LegalTargets.Contains(square.Position),
                message.LastMove?.From == square.Position,
                message.LastMove?.To == square.Position);
        }

        GameResultText = message.State.Status switch
        {
            GameStatus.RedWins => "红方胜",
            GameStatus.BlackWins => "黑方胜",
            GameStatus.Draw => "和棋",
            _ => null
        };

        return Task.CompletedTask;
    }
}
