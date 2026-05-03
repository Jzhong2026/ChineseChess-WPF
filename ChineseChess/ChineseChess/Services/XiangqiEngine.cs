using ChineseChess.Models;

namespace ChineseChess.Services;

public sealed class XiangqiEngine
{
    public const int BoardRows = 10;
    public const int BoardCols = 9;

    public string PieceLabel(Piece piece) => piece.Side == Side.Red
        ? piece.Type switch
        {
            PieceType.General => "帅",
            PieceType.Advisor => "仕",
            PieceType.Elephant => "相",
            PieceType.Horse => "马",
            PieceType.Rook => "车",
            PieceType.Cannon => "炮",
            PieceType.Soldier => "兵",
            _ => ""
        }
        : piece.Type switch
        {
            PieceType.General => "将",
            PieceType.Advisor => "士",
            PieceType.Elephant => "象",
            PieceType.Horse => "马",
            PieceType.Rook => "车",
            PieceType.Cannon => "炮",
            PieceType.Soldier => "卒",
            _ => ""
        };

    public Side Opposite(Side side) => side == Side.Red ? Side.Black : Side.Red;

    public GameState CreateInitialState() => new(CreateInitialBoard(), Side.Red, GameStatus.Playing, Array.Empty<Move>());

    public Piece?[,] CreateInitialBoard()
    {
        var board = EmptyBoard();
        void Place(int row, int col, Side side, PieceType type, int index)
        {
            board[row, col] = new Piece($"{side.ToString().ToLowerInvariant()}-{type.ToString().ToLowerInvariant()}-{index}", side, type);
        }

        var backRank = new[]
        {
            PieceType.Rook, PieceType.Horse, PieceType.Elephant, PieceType.Advisor, PieceType.General,
            PieceType.Advisor, PieceType.Elephant, PieceType.Horse, PieceType.Rook
        };

        for (var col = 0; col < backRank.Length; col++)
        {
            Place(0, col, Side.Black, backRank[col], col);
            Place(9, col, Side.Red, backRank[col], col);
        }

        Place(2, 1, Side.Black, PieceType.Cannon, 0);
        Place(2, 7, Side.Black, PieceType.Cannon, 1);
        Place(7, 1, Side.Red, PieceType.Cannon, 0);
        Place(7, 7, Side.Red, PieceType.Cannon, 1);

        var soldierCols = new[] { 0, 2, 4, 6, 8 };
        for (var i = 0; i < soldierCols.Length; i++)
        {
            Place(3, soldierCols[i], Side.Black, PieceType.Soldier, i);
            Place(6, soldierCols[i], Side.Red, PieceType.Soldier, i);
        }

        return board;
    }

    public Piece?[,] EmptyBoard() => new Piece?[BoardRows, BoardCols];

    public Piece?[,] CloneBoard(Piece?[,] board)
    {
        var next = new Piece?[BoardRows, BoardCols];
        Array.Copy(board, next, board.Length);
        return next;
    }

    public Piece?[,] ApplyMove(Piece?[,] board, Move move)
    {
        var next = CloneBoard(board);
        next[move.To.Row, move.To.Col] = next[move.From.Row, move.From.Col];
        next[move.From.Row, move.From.Col] = null;
        return next;
    }

    public GameState MakeMove(GameState state, Move move)
    {
        var captured = state.Board[move.To.Row, move.To.Col];
        var completeMove = move with { Captured = captured };
        var board = ApplyMove(state.Board, completeMove);
        var turn = Opposite(state.Turn);
        var history = state.History.Concat(new[] { completeMove }).ToList();
        return new GameState(board, turn, GetGameStatus(board, turn), history);
    }

    public IReadOnlyList<Move> GetLegalMoves(Piece?[,] board, Side side, IReadOnlyList<Move>? history = null)
    {
        var moves = new List<Move>();
        var seenPositions = history is { Count: > 0 } ? GetPositionHistory(history, board) : null;
        ForEachPiece(board, (piece, from) =>
        {
            if (piece.Side != side) return;
            foreach (var move in GetPseudoMoves(board, from, piece))
            {
                if (!WouldLeaveGeneralInCheck(board, move, side) && !WouldRepeatCheck(board, move, side, seenPositions))
                {
                    moves.Add(move);
                }
            }
        });
        return moves;
    }

    public IReadOnlyList<Move> GetLegalMovesForPiece(Piece?[,] board, Position from, IReadOnlyList<Move>? history = null)
    {
        var piece = GetPiece(board, from);
        if (piece is null) return Array.Empty<Move>();
        var seenPositions = history is { Count: > 0 } ? GetPositionHistory(history, board) : null;
        return GetPseudoMoves(board, from, piece)
            .Where(move => !WouldLeaveGeneralInCheck(board, move, piece.Side) && !WouldRepeatCheck(board, move, piece.Side, seenPositions))
            .ToList();
    }

    public bool IsLegalMove(Piece?[,] board, Move move, Side side, IReadOnlyList<Move>? history = null)
    {
        var piece = GetPiece(board, move.From);
        return piece is not null &&
               piece.Side == side &&
               piece.Id == move.Piece.Id &&
               GetLegalMovesForPiece(board, move.From, history).Any(legalMove => legalMove.To == move.To);
    }

    public bool IsRepeatCheckMove(Piece?[,] board, Move move, Side side, IReadOnlyList<Move>? history = null)
    {
        var seenPositions = history is { Count: > 0 } ? GetPositionHistory(history, board) : null;
        return WouldRepeatCheck(board, move, side, seenPositions);
    }

    public GameStatus GetGameStatus(Piece?[,] board, Side turn, IReadOnlyList<Move>? history = null)
    {
        var inCheck = IsInCheck(board, turn);
        var moves = GetLegalMoves(board, turn, history);
        if (moves.Count > 0)
        {
            if (inCheck) return turn == Side.Red ? GameStatus.RedCheck : GameStatus.BlackCheck;
            return GameStatus.Playing;
        }

        return turn == Side.Red ? GameStatus.BlackWins : GameStatus.RedWins;
    }

    public bool IsInCheck(Piece?[,] board, Side side)
    {
        var general = FindGeneral(board, side);
        if (general is null) return true;
        var enemy = Opposite(side);
        return GetAllPseudoMovesWithoutCheckFilter(board, enemy).Any(move => move.To == general.Value);
    }

    public Position? FindGeneral(Piece?[,] board, Side side)
    {
        for (var row = 0; row < BoardRows; row++)
        {
            for (var col = 0; col < BoardCols; col++)
            {
                var piece = board[row, col];
                if (piece?.Side == side && piece.Type == PieceType.General) return new Position(row, col);
            }
        }

        return null;
    }

    public string MoveNotation(Move move)
    {
        var from = $"{move.From.Col + 1}{10 - move.From.Row}";
        var to = $"{move.To.Col + 1}{10 - move.To.Row}";
        var capture = move.Captured is null ? "" : $" x {PieceLabel(move.Captured)}";
        return $"{PieceLabel(move.Piece)} {from}-{to}{capture}";
    }

    private bool WouldLeaveGeneralInCheck(Piece?[,] board, Move move, Side side) => IsInCheck(ApplyMove(board, move), side);

    private bool WouldRepeatCheck(Piece?[,] board, Move move, Side side, HashSet<string>? seenPositions)
    {
        if (seenPositions is null) return false;
        var nextBoard = ApplyMove(board, move);
        var enemy = Opposite(side);
        return IsInCheck(nextBoard, enemy) && seenPositions.Contains(GetPositionKey(nextBoard, enemy));
    }

    private Piece?[,] UndoMove(Piece?[,] board, Move move)
    {
        var next = CloneBoard(board);
        next[move.From.Row, move.From.Col] = move.Piece;
        next[move.To.Row, move.To.Col] = move.Captured;
        return next;
    }

    private HashSet<string> GetPositionHistory(IReadOnlyList<Move> history, Piece?[,] currentBoard)
    {
        var board = currentBoard;
        var boards = new List<Piece?[,]> { board };
        for (var i = history.Count - 1; i >= 0; i--)
        {
            board = UndoMove(board, history[i]);
            boards.Add(board);
        }

        boards.Reverse();
        var seen = new HashSet<string>();
        var turn = Side.Red;
        foreach (var item in boards)
        {
            seen.Add(GetPositionKey(item, turn));
            turn = Opposite(turn);
        }

        return seen;
    }

    private string GetPositionKey(Piece?[,] board, Side turn)
    {
        var parts = new List<string> { turn.ToString() };
        for (var row = 0; row < BoardRows; row++)
        {
            for (var col = 0; col < BoardCols; col++)
            {
                parts.Add(board[row, col]?.Id ?? ".");
            }
        }

        return string.Join("|", parts);
    }

    private void ForEachPiece(Piece?[,] board, Action<Piece, Position> callback)
    {
        for (var row = 0; row < BoardRows; row++)
        {
            for (var col = 0; col < BoardCols; col++)
            {
                var piece = board[row, col];
                if (piece is not null) callback(piece, new Position(row, col));
            }
        }
    }

    private IReadOnlyList<Move> GetAllPseudoMovesWithoutCheckFilter(Piece?[,] board, Side side)
    {
        var moves = new List<Move>();
        ForEachPiece(board, (piece, from) =>
        {
            if (piece.Side == side) moves.AddRange(GetPseudoMoves(board, from, piece));
        });
        return moves;
    }

    private IReadOnlyList<Move> GetPseudoMoves(Piece?[,] board, Position from, Piece piece) => piece.Type switch
    {
        PieceType.General => GeneralMoves(board, from, piece),
        PieceType.Advisor => AdvisorMoves(board, from, piece),
        PieceType.Elephant => ElephantMoves(board, from, piece),
        PieceType.Horse => HorseMoves(board, from, piece),
        PieceType.Rook => LineMoves(board, from, piece, false),
        PieceType.Cannon => LineMoves(board, from, piece, true),
        PieceType.Soldier => SoldierMoves(board, from, piece),
        _ => Array.Empty<Move>()
    };

    private void PushIfAvailable(ICollection<Move> moves, Piece?[,] board, Position from, Position to, Piece piece)
    {
        if (!IsInsideBoard(to)) return;
        var target = board[to.Row, to.Col];
        if (target is null || target.Side != piece.Side) moves.Add(new Move(from, to, piece, target));
    }

    private bool IsInsideBoard(Position pos) => pos.Row >= 0 && pos.Row < BoardRows && pos.Col >= 0 && pos.Col < BoardCols;

    private bool InPalace(Position pos, Side side)
    {
        var minRow = side == Side.Red ? 7 : 0;
        var maxRow = side == Side.Red ? 9 : 2;
        return pos.Row >= minRow && pos.Row <= maxRow && pos.Col >= 3 && pos.Col <= 5;
    }

    private IReadOnlyList<Move> GeneralMoves(Piece?[,] board, Position from, Piece piece)
    {
        var moves = new List<Move>();
        foreach (var to in new[] { new Position(from.Row - 1, from.Col), new Position(from.Row + 1, from.Col), new Position(from.Row, from.Col - 1), new Position(from.Row, from.Col + 1) })
        {
            if (InPalace(to, piece.Side)) PushIfAvailable(moves, board, from, to, piece);
        }

        var step = piece.Side == Side.Red ? -1 : 1;
        var row = from.Row + step;
        while (row >= 0 && row < BoardRows)
        {
            var target = board[row, from.Col];
            if (target is not null)
            {
                if (target.Type == PieceType.General && target.Side != piece.Side) moves.Add(new Move(from, new Position(row, from.Col), piece, target));
                break;
            }

            row += step;
        }

        return moves;
    }

    private IReadOnlyList<Move> AdvisorMoves(Piece?[,] board, Position from, Piece piece)
    {
        var moves = new List<Move>();
        foreach (var to in new[] { new Position(from.Row - 1, from.Col - 1), new Position(from.Row - 1, from.Col + 1), new Position(from.Row + 1, from.Col - 1), new Position(from.Row + 1, from.Col + 1) })
        {
            if (InPalace(to, piece.Side)) PushIfAvailable(moves, board, from, to, piece);
        }

        return moves;
    }

    private IReadOnlyList<Move> ElephantMoves(Piece?[,] board, Position from, Piece piece)
    {
        var moves = new List<Move>();
        foreach (var delta in new[] { new Position(-2, -2), new Position(-2, 2), new Position(2, -2), new Position(2, 2) })
        {
            var eye = new Position(from.Row + delta.Row / 2, from.Col + delta.Col / 2);
            var to = new Position(from.Row + delta.Row, from.Col + delta.Col);
            var crossedRiver = piece.Side == Side.Red ? to.Row < 5 : to.Row > 4;
            if (!IsInsideBoard(to) || crossedRiver || board[eye.Row, eye.Col] is not null) continue;
            PushIfAvailable(moves, board, from, to, piece);
        }

        return moves;
    }

    private IReadOnlyList<Move> HorseMoves(Piece?[,] board, Position from, Piece piece)
    {
        var moves = new List<Move>();
        var candidates = new[]
        {
            (To: new Position(from.Row - 2, from.Col - 1), Leg: new Position(from.Row - 1, from.Col)),
            (To: new Position(from.Row - 2, from.Col + 1), Leg: new Position(from.Row - 1, from.Col)),
            (To: new Position(from.Row + 2, from.Col - 1), Leg: new Position(from.Row + 1, from.Col)),
            (To: new Position(from.Row + 2, from.Col + 1), Leg: new Position(from.Row + 1, from.Col)),
            (To: new Position(from.Row - 1, from.Col - 2), Leg: new Position(from.Row, from.Col - 1)),
            (To: new Position(from.Row + 1, from.Col - 2), Leg: new Position(from.Row, from.Col - 1)),
            (To: new Position(from.Row - 1, from.Col + 2), Leg: new Position(from.Row, from.Col + 1)),
            (To: new Position(from.Row + 1, from.Col + 2), Leg: new Position(from.Row, from.Col + 1))
        };

        foreach (var candidate in candidates)
        {
            if (!IsInsideBoard(candidate.To) || board[candidate.Leg.Row, candidate.Leg.Col] is not null) continue;
            PushIfAvailable(moves, board, from, candidate.To, piece);
        }

        return moves;
    }

    private IReadOnlyList<Move> LineMoves(Piece?[,] board, Position from, Piece piece, bool isCannon)
    {
        var moves = new List<Move>();
        foreach (var direction in new[] { new Position(-1, 0), new Position(1, 0), new Position(0, -1), new Position(0, 1) })
        {
            var row = from.Row + direction.Row;
            var col = from.Col + direction.Col;
            var screenSeen = false;
            while (IsInsideBoard(new Position(row, col)))
            {
                var target = board[row, col];
                if (!isCannon)
                {
                    if (target is null) moves.Add(new Move(from, new Position(row, col), piece));
                    else
                    {
                        if (target.Side != piece.Side) moves.Add(new Move(from, new Position(row, col), piece, target));
                        break;
                    }
                }
                else if (!screenSeen)
                {
                    if (target is null) moves.Add(new Move(from, new Position(row, col), piece));
                    else screenSeen = true;
                }
                else if (target is not null)
                {
                    if (target.Side != piece.Side) moves.Add(new Move(from, new Position(row, col), piece, target));
                    break;
                }

                row += direction.Row;
                col += direction.Col;
            }
        }

        return moves;
    }

    private IReadOnlyList<Move> SoldierMoves(Piece?[,] board, Position from, Piece piece)
    {
        var moves = new List<Move>();
        var forward = piece.Side == Side.Red ? -1 : 1;
        PushIfAvailable(moves, board, from, new Position(from.Row + forward, from.Col), piece);
        var crossedRiver = piece.Side == Side.Red ? from.Row <= 4 : from.Row >= 5;
        if (crossedRiver)
        {
            PushIfAvailable(moves, board, from, new Position(from.Row, from.Col - 1), piece);
            PushIfAvailable(moves, board, from, new Position(from.Row, from.Col + 1), piece);
        }

        return moves;
    }

    private Piece? GetPiece(Piece?[,] board, Position position) => IsInsideBoard(position) ? board[position.Row, position.Col] : null;
}
