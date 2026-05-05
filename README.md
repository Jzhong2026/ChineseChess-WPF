# ChineseChess-WPF

## 训练脚本

保留通用 JSONL 训练脚本 `train_pytorch_jsonl.py`，并新增针对 `ChineseChess.SelfPlay` 输出格式的专用脚本 `train_chinese_chess_policy_value.py`。

示例：

```bash
python train_chinese_chess_policy_value.py \
  --input data/selfplay/games-random.jsonl \
  --output artifacts/chinese_chess_policy_value.pt \
  --epochs 20 \
  --batch-size 256 \
  --hidden-dim 512 \
  --num-layers 3
```
