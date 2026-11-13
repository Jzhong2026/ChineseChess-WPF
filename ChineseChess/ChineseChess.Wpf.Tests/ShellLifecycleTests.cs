using System.Reflection;
using Caliburn.Micro;
using ChineseChess.Ai;
using ChineseChess.Messages;
using ChineseChess.Models;
using ChineseChess.Services;
using ChineseChess.ViewModels;
using Xunit;

[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace ChineseChess.Wpf.Tests;

public class ShellLifecycleTests
{
    private static ShellViewModel Create(IAiEngine ai)
    {
        PlatformProvider.Current = new DefaultPlatformProvider();
        var events = new NoopEvents();
        var engine = new XiangqiEngine();
        var side = new SidePanelViewModel(events, engine) { AiEngineMode = AiEngineMode.NeuralMcts };
        var shell = new ShellViewModel(events, engine, new XiangqiAiService(engine),
            new SoundService(), new BoardViewModel(events, engine), side);
        Set(shell, "_neuralEngine", ai);
        Set(shell, "_state", engine.CreateInitialState() with { Turn = Side.Black });
        return shell;
    }

    private static void Set(ShellViewModel shell, string name, object value) =>
        typeof(ShellViewModel).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(shell, value);

    private static Task Invoke(ShellViewModel shell, string method, params object[] args) =>
        (Task)typeof(ShellViewModel).GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(shell, args)!;

    [Fact]
    public async Task ModeSwitchWaitsForWorkerBeforeDisposingAndDiscardsItsMove()
    {
        using var ai = new BlockingEngine();
        var shell = Create(ai);
        var turn = Invoke(shell, "MaybeStartAiTurn");
        await ai.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        var switching = shell.HandleAsync(new AiEngineModeChangedMessage(AiEngineMode.Classic), CancellationToken.None);
        try
        {
            Assert.False(switching.IsCompleted);
            Assert.False(ai.Disposed);
        }
        finally { ai.Release.Set(); }
        await Task.WhenAll(turn, switching).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(ai.Disposed);
        Assert.False(ai.DisposedWhileSearching);
        var state = (GameState)typeof(ShellViewModel).GetField("_state", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(shell)!;
        Assert.Empty(state.History);
        Assert.Equal(Side.Red, state.Turn);
    }

    [Fact]
    public async Task DeactivationWaitsForWorkerBeforeDisposing()
    {
        using var ai = new BlockingEngine();
        var shell = Create(ai);
        var turn = Invoke(shell, "MaybeStartAiTurn");
        await ai.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        var closing = Invoke(shell, "OnDeactivateAsync", true, CancellationToken.None);
        try
        {
            Assert.False(closing.IsCompleted);
            Assert.False(ai.Disposed);
        }
        finally { ai.Release.Set(); }
        await Task.WhenAll(turn, closing).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(ai.Disposed);
        Assert.False(ai.DisposedWhileSearching);
    }

    [Fact]
    public async Task RestartWaitsForWorkerAndDoesNotApplyCancelledMove()
    {
        using var ai = new BlockingEngine();
        var shell = Create(ai);
        var turn = Invoke(shell, "MaybeStartAiTurn");
        await ai.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        var restart = shell.HandleAsync(new RestartRequestedMessage(), CancellationToken.None);
        try { Assert.False(restart.IsCompleted); }
        finally { ai.Release.Set(); }
        await Task.WhenAll(turn, restart).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.False(ai.Disposed);
        Assert.False(shell.SidePanel.AiThinking);
        var state = (GameState)typeof(ShellViewModel).GetField("_state", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(shell)!;
        Assert.Empty(state.History);
    }

    [Fact]
    public async Task SliderChangesAreCapturedForEachTurn()
    {
        var ai = new RecordingEngine();
        var shell = Create(ai);
        shell.SidePanel.MctsSimulations = 100;
        await Invoke(shell, "MaybeStartAiTurn");
        shell.SidePanel.MctsSimulations = 700;
        await Invoke(shell, "MaybeStartAiTurn");
        Assert.Equal(new int?[] { 100, 700 }, ai.Budgets);
    }

    [Fact]
    public async Task RealNeuralEngineUsesChangedBudgetAfterRestart()
    {
        var shell = Create(new RecordingEngine());
        shell.SidePanel.HumanSide = Side.Black;
        shell.SidePanel.MctsSimulations = 3;
        shell.SidePanel.AiTimeLimitMs = 10000;
        try
        {
            await shell.HandleAsync(new AiEngineModeChangedMessage(AiEngineMode.NeuralMcts), CancellationToken.None);
            Assert.StartsWith("MCTS 3 sims", shell.SidePanel.NeuralAiInfo);
            shell.SidePanel.MctsSimulations = 7;
            await shell.HandleAsync(new RestartRequestedMessage(), CancellationToken.None);
            Assert.StartsWith("MCTS 7 sims", shell.SidePanel.NeuralAiInfo);
        }
        finally
        {
            await Invoke(shell, "OnDeactivateAsync", true, CancellationToken.None);
        }
    }

    [Fact]
    public async Task LatestModeWinsWhenLoadingAndSwitchingOverlap()
    {
        var shell = Create(new RecordingEngine());
        var loading = shell.HandleAsync(new AiEngineModeChangedMessage(AiEngineMode.NeuralMcts), CancellationToken.None);
        var switching = shell.HandleAsync(new AiEngineModeChangedMessage(AiEngineMode.Classic), CancellationToken.None);
        await Task.WhenAll(loading, switching).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(AiEngineMode.Classic, shell.SidePanel.AiEngineMode);
        Assert.Null(typeof(ShellViewModel).GetField("_neuralEngine", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(shell));
    }

    [Fact]
    public async Task StrongOnnxModeDoesNotPlayAClassicMoveWhenModelIsUnavailable()
    {
        var shell = Create(new RecordingEngine());
        Set(shell, "_neuralEngine", null!);
        shell.SidePanel.AiEngineMode = AiEngineMode.NnueOnnx;
        var before = typeof(ShellViewModel).GetField("_state", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(shell);
        await Invoke(shell, "MaybeStartAiTurn");
        Assert.Same(before, typeof(ShellViewModel).GetField("_state", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(shell));
        Assert.Null(typeof(ShellViewModel).GetField("_aiCancellation", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(shell));
        Assert.False(shell.SidePanel.AiThinking);
        Assert.Contains("未加载", shell.SidePanel.NeuralAiInfo);
    }

    [Fact]
    public async Task StrongOnnxModeLoadsTheRealModelAndSearchesThroughTheShell()
    {
        var shell = Create(new RecordingEngine());
        shell.SidePanel.HumanSide = Side.Black;
        shell.SidePanel.AiTimeLimitMs = 100;
        try
        {
            await shell.HandleAsync(new AiEngineModeChangedMessage(AiEngineMode.NnueOnnx), CancellationToken.None);
            var actual = typeof(ShellViewModel).GetField("_neuralEngine", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(shell);
            var neural = Assert.IsType<ChineseChess.Ai.Neural.NeuralValueAlphaBetaAiEngine>(actual);
            Assert.True(neural.LastEvaluations > 0);
            Assert.StartsWith("ONNX 加强", shell.SidePanel.NeuralAiInfo);
            var state = (GameState)typeof(ShellViewModel).GetField("_state", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(shell)!;
            Assert.Single(state.History);
            Assert.Equal(Side.Black, state.Turn);
        }
        finally { await Invoke(shell, "OnDeactivateAsync", true, CancellationToken.None); }
    }

    private sealed class RecordingEngine : IAiEngine
    {
        public string Name => "Recording";
        public List<int?> Budgets { get; } = new();
        public AiMoveResult ChooseMove(AiMoveRequest request)
        {
            Budgets.Add(request.Simulations);
            return new(null, new SearchStats(0, 0, 0, 0));
        }
    }

    private sealed class BlockingEngine : IAiEngine, IDisposable
    {
        public string Name => "Blocking";
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public ManualResetEventSlim Release { get; } = new(false);
        public bool Disposed { get; private set; }
        public bool DisposedWhileSearching { get; private set; }
        private bool _searching;
        public AiMoveResult ChooseMove(AiMoveRequest request)
        {
            _searching = true;
            Started.SetResult();
            try
            {
                if (!Release.Wait(TimeSpan.FromSeconds(10))) throw new TimeoutException();
                var move = new XiangqiEngine().GetLegalMoves(request.Board, request.Side, request.History)[0];
                return new(move, new SearchStats(1, 1, 0, 0));
            }
            finally { _searching = false; }
        }
        public void Dispose()
        {
            DisposedWhileSearching |= _searching;
            Disposed = true;
            Release.Set();
        }
    }

    private sealed class NoopEvents : IEventAggregator
    {
        public bool HandlerExistsFor(Type messageType) => false;
        public void Subscribe(object subscriber, Func<Func<Task>, Task> marshal) { }
        public void Unsubscribe(object subscriber) { }
        public Task PublishAsync(object message, Func<Func<Task>, Task> marshal, CancellationToken cancellationToken = default) => Task.CompletedTask;
    }
}
