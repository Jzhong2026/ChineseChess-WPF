using ChineseChess.Encoding;
using ChineseChess.Models;
using ChineseChess.Services;

var engine = new XiangqiEngine();
var boardEncoding = BoardEncoder.Encode(engine.CreateInitialBoard());
Require(boardEncoding.Length == 1260, $"board encoding length expected 1260, got {boardEncoding.Length}");

var minActionId = int.MaxValue;
var maxActionId = int.MinValue;
for (var fromRow = 0; fromRow < XiangqiEngine.BoardRows; fromRow++)
{
    for (var fromCol = 0; fromCol < XiangqiEngine.BoardCols; fromCol++)
    {
        for (var toRow = 0; toRow < XiangqiEngine.BoardRows; toRow++)
        {
            for (var toCol = 0; toCol < XiangqiEngine.BoardCols; toCol++)
            {
                var actionId = MoveEncoder.Encode(new Position(fromRow, fromCol), new Position(toRow, toCol));
                minActionId = Math.Min(minActionId, actionId);
                maxActionId = Math.Max(maxActionId, actionId);
            }
        }
    }
}

Require(minActionId == 0, $"move action id min expected 0, got {minActionId}");
Require(maxActionId == 8099, $"move action id max expected 8099, got {maxActionId}");
Require(MoveEncoder.ActionCount == 8100, $"move action count expected 8100, got {MoveEncoder.ActionCount}");

Console.WriteLine("Core encoder verification passed.");
Console.WriteLine($"Board encoding length: {boardEncoding.Length}");
Console.WriteLine($"Move action id range: {minActionId}..{maxActionId}");

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}
