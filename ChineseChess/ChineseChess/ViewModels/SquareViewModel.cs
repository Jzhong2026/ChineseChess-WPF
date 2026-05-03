using Caliburn.Micro;
using ChineseChess.Messages;
using ChineseChess.Models;

namespace ChineseChess.ViewModels;

public sealed class SquareViewModel : PropertyChangedBase
{
    private const double PointSpacing = 78;
    private const double PieceHitSize = 78;
    private readonly IEventAggregator _events;
    private string? _pieceText;
    private Side? _pieceSide;
    private bool _isSelected;
    private bool _isLegalTarget;
    private bool _isLastFrom;
    private bool _isLastTo;

    public SquareViewModel(IEventAggregator events, int row, int col)
    {
        _events = events;
        Position = new Position(row, col);
    }

    public Position Position { get; }

    public string CoordinateText => $"{Position.Col + 1}路 {10 - Position.Row}线";

    public double BoardX => Position.Col * PointSpacing;

    public double BoardY => Position.Row * PointSpacing;

    public double HitLeft => BoardX - PieceHitSize / 2;

    public double HitTop => BoardY - PieceHitSize / 2;

    public string? PieceText
    {
        get => _pieceText;
        private set => Set(ref _pieceText, value);
    }

    public Side? PieceSide
    {
        get => _pieceSide;
        private set => Set(ref _pieceSide, value);
    }

    public bool HasPiece => PieceText is not null;

    public bool IsRedPiece => PieceSide == Side.Red;

    public bool IsBlackPiece => PieceSide == Side.Black;

    public bool IsSelected
    {
        get => _isSelected;
        private set => Set(ref _isSelected, value);
    }

    public bool IsLegalTarget
    {
        get => _isLegalTarget;
        private set => Set(ref _isLegalTarget, value);
    }

    public bool IsLastFrom
    {
        get => _isLastFrom;
        private set => Set(ref _isLastFrom, value);
    }

    public bool IsLastTo
    {
        get => _isLastTo;
        private set => Set(ref _isLastTo, value);
    }

    public void Update(string? pieceText, Side? pieceSide, bool isSelected, bool isLegalTarget, bool isLastFrom, bool isLastTo)
    {
        PieceText = pieceText;
        PieceSide = pieceSide;
        NotifyOfPropertyChange(nameof(HasPiece));
        NotifyOfPropertyChange(nameof(IsRedPiece));
        NotifyOfPropertyChange(nameof(IsBlackPiece));
        IsSelected = isSelected;
        IsLegalTarget = isLegalTarget;
        IsLastFrom = isLastFrom;
        IsLastTo = isLastTo;
    }

    public async Task Select()
    {
        await _events.PublishOnUIThreadAsync(new SquareSelectedMessage(Position));
    }
}
