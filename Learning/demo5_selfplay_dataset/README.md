# Demo 5: 自博弈 JSONL 到训练样本

这个 demo 对应项目里的数据生成与训练链路：

- `ChineseChess.SelfPlay/Program.cs`
- `train_cnn_policy_value.py`
- `train_chinese_chess_policy_value.py`

它演示一行自博弈数据如何进入训练集：

1. `BoardEncoding` 必须是 `float[1260]`
2. `LegalMoves` 和 `SelectedMove` 都是 `MoveEncoder` 的 action id
3. `SelectedMove` 必须在 `LegalMoves` 里
4. factored policy 的标签来自 `selected_move // 90` 与 `selected_move % 90`
5. V5 value target 使用当前行棋方视角：`result * side_to_move`
6. `ValueWeight`、`PolicyWeight` 控制这一行样本是否用于对应任务

## 运行

```powershell
python Learning\demo5_selfplay_dataset\selfplay_dataset_demo.py
```

脚本只使用 Python 标准库，不依赖训练环境。它不会真正训练模型，只把训练脚本读取一行 JSONL 后会做的关键转换打印出来。

## 和项目代码的对应关系

| Demo 概念 | 项目代码 |
|---|---|
| `build_selfplay_row()` | `ChineseChess.SelfPlay` 写出的 `SelfPlayRow` |
| `validate_row()` | `Program.cs` 中的 `ValidateRow()` |
| `policy_targets()` | `train_cnn_policy_value.py` 中的 `from_target` / `to_target` |
| `current_player_value_target()` | V5 的当前行棋方 value target |
| `build_legal_mask()` | 训练脚本中的 `legal_mask` |

