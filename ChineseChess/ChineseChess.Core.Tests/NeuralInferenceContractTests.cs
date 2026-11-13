using ChineseChess.Ai.Neural;
using ChineseChess.Encoding;
using ChineseChess.Models;
using ChineseChess.Services;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace ChineseChess.Core.Tests;

public class NeuralInferenceContractTests
{
    [Theory]
    [InlineData(Side.Red)]
    [InlineData(Side.Black)]
    public void ServiceMatchesRawOnnxOutputsForEveryAction(Side side)
    {
        var model = Environment.GetEnvironmentVariable("CHESS_CONTRACT_MODEL") ?? TestPaths.DeployedOnnx;
        var engine = new XiangqiEngine();
        var state = engine.CreateInitialState();
        for (var i = 0; i < 5; i++)
        {
            var legal = engine.GetLegalMoves(state.Board, state.Turn, state.History);
            state = engine.MakeMove(state, legal[(i * 7) % legal.Count]);
        }
        using var rawSession = new InferenceSession(model);
        var planes = rawSession.InputMetadata["board_input"].Dimensions[1];
        var input = new DenseTensor<float>(new[] { 1, planes, 10, 9 });
        var flat = BoardEncoder.Encode(state.Board);
        for (var row = 0; row < 10; row++)
        for (var col = 0; col < 9; col++)
        {
            for (var plane = 0; plane < 14; plane++) input[0, plane, row, col] = flat[(row * 9 + col) * 14 + plane];
            if (planes >= 16) input[0, side == Side.Red ? 14 : 15, row, col] = 1;
        }
        using var raw = rawSession.Run(new[] { NamedOnnxValue.CreateFromTensor("board_input", input) });
        using var service = new XiangqiNeuralAiService(model, engine, true);
        var (policy, value) = service.RunInference(state.Board, side);
        var factored = rawSession.OutputMetadata.ContainsKey("from_logits");
        for (var action = 0; action < 8100; action++)
        {
            var expected = factored
                ? raw.First(r => r.Name == "from_logits").AsTensor<float>()[0, action / 90]
                    + raw.First(r => r.Name == "to_logits").AsTensor<float>()[0, action % 90]
                : raw.First(r => r.Name == "policy_logits").AsTensor<float>()[0, action];
            Assert.InRange(Math.Abs(policy.GetValue(action) - expected), 0, 1e-5);
        }
        Assert.InRange(Math.Abs(value.GetValue(0) - raw.First(r => r.Name == "value_pred").AsTensor<float>()[0]), 0, 1e-5);
    }
}
