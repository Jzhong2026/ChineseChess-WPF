using ChineseChess.Ai;
using ChineseChess.Services;

namespace ChineseChess.Ai.Neural;

/// <summary>
/// Composition root for engines (DIP): callers ask for an <see cref="IAiEngine"/>
/// by kind and never construct concrete engines or touch ONNX directly.
/// </summary>
public static class AiEngineFactory
{
    public static IAiEngine Create(AiEngineKind kind, XiangqiEngine engine, string? onnxModelPath = null, int mctsSimulations = 3000)
    {
        switch (kind)
        {
            case AiEngineKind.Classic:
                return new XiangqiAiService(engine);
            case AiEngineKind.NeuralPolicy:
                return new NeuralPolicyAiEngine(LoadNeural(engine, onnxModelPath));
            case AiEngineKind.NeuralMcts:
                var neural = LoadNeural(engine, onnxModelPath);
                return new MctsAiEngine(neural, new MctsAiService(neural, engine), mctsSimulations);
            default:
                throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown engine kind");
        }
    }

    private static XiangqiNeuralAiService LoadNeural(XiangqiEngine engine, string? onnxModelPath)
    {
        if (string.IsNullOrEmpty(onnxModelPath))
            throw new ArgumentException("A neural engine requires an ONNX model path.", nameof(onnxModelPath));
        return new XiangqiNeuralAiService(onnxModelPath, engine, isCnnModel: true);
    }
}