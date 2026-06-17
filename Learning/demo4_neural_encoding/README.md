# Demo 4: 神经网络编码与策略头解码

这个 demo 对应项目里的神经网络 AI 链路：

- `ChineseChess.Core/Encoding/BoardEncoder.cs`
- `ChineseChess.Core/Encoding/MoveEncoder.cs`
- `ChineseChess.Core/Services/XiangqiNeuralAiService.cs`
- `train_cnn_policy_value.py`
- `export_cnn_onnx.py`

核心目标是把下面几件事跑清楚：

1. 棋盘如何编码成 `float[1260]`
2. `float[1260]` 如何重排成 CNN/ONNX 需要的 `[1,16,10,9]`
3. 走法如何编码成 `0..8099` 的 action id
4. factored policy head 的 `from_logits[90] + to_logits[90]` 如何合并并只在合法走法中选最大值
5. 左右翻转数据增强时，棋盘、走法、legal mask 应该如何同步映射

## 运行

```powershell
python Learning\demo4_neural_encoding\neural_encoding_demo.py
```

这个脚本不依赖 PyTorch 或 ONNX Runtime，只用 Python 标准库模拟张量形状和策略头逻辑。它适合在训练/导出/接入 ONNX 前先验证数据编码思路。

## 和项目代码的对应关系

| Demo 概念 | 项目代码 |
|---|---|
| `encode_board_square_major()` | `BoardEncoder.Encode()` |
| `get_board_index()` | `BoardEncoder.GetIndex()` |
| `encode_move()` / `decode_move()` | `MoveEncoder.Encode()` / `MoveEncoder.Decode()` |
| `pack_cnn_input()` | `XiangqiNeuralAiService.EncodeCnn()` |
| `combine_factored_logits()` | `XiangqiNeuralAiService.RunInference()` 中 factored 输出合并 |
| `flip_*()` | `train_cnn_policy_value.py` 中水平翻转增强 |

