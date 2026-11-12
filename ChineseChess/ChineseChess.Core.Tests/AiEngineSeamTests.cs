using ChineseChess.Ai;
using ChineseChess.Ai.Neural;
using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

/// <summary>
/// Locks in the IAiEngine seam: every engine is interchangeable through the
/// abstraction (LSP), the factory is the single composition point (OCP/DIP),
/// and callers never need concrete engine types.
/// </summary>
public class AiEngineSeamTests
{
    private static readonly XiangqiEngine Engine = new();

    [Theory]
    [InlineData(AiEngineKind.Classic)]
    [InlineData(AiEngineKind.NeuralPolicy)]
    [InlineData(AiEngineKind.NeuralMcts)]
    public void EveryEngineReturnsALegalMoveThroughTheInterface(AiEngineKind kind)
    {
        var state = Engine.CreateInitialState();
        var legal = Engine.GetLegalMoves(state.Board, state.Turn, state.History);
        IAiEngine ai = AiEngineFactory.Create(kind, Engine, TestPaths.DeployedOnnx, mctsSimulations: 40);
        using var disposable = ai as IDisposable;

        var result = ai.ChooseMove(new AiMoveRequest(state.Board, state.Turn, state.History, 300, Difficulty: 3));

        Assert.False(string.IsNullOrWhiteSpace(ai.Name));
        Assert.NotNull(result.Move);
        Assert.Contains(legal, m => m.From == result.Move!.From && m.To == result.Move.To);
    }

    [Fact]
    public void FactoryRejectsUnknownKind()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            AiEngineFactory.Create((AiEngineKind)99, Engine, TestPaths.DeployedOnnx));
    }

    [Fact]
    public void NeuralFactoryRequiresModelPath()
    {
        Assert.Throws<ArgumentException>(() =>
            AiEngineFactory.Create(AiEngineKind.NeuralPolicy, Engine, onnxModelPath: null));
    }

    [Fact]
    public void ClassicEngineNeedsNoModel()
    {
        IAiEngine ai = AiEngineFactory.Create(AiEngineKind.Classic, Engine);
        var state = Engine.CreateInitialState();
        var result = ai.ChooseMove(new AiMoveRequest(state.Board, state.Turn, state.History, 200, Difficulty: 2));
        Assert.NotNull(result.Move);
    }
}