# ChineseChess-WPF

## 项目介绍文档

- [面试项目介绍与技术复盘](docs/INTERVIEW_PROJECT_GUIDE.md)：适合面试时介绍项目开发过程、技术细节、问题排查和项目亮点。
- [开发日志](docs/DEV_LOG.md)：记录 AI 训练、ONNX 集成、MCTS 优化等迭代过程。

## Self-play 数据生成

`ChineseChess.SelfPlay` 会输出 JSONL，每一行是一个可直接供 `train_chinese_chess_policy_value.py` 使用的局面样本。默认参数已调成更适合训练数据采样的配置：

- 前 12 ply 使用更宽的开局随机候选，降低重复局谱比例。
- 全局从近似最优候选中采样，避免完全确定性自博弈反复走同一盘。
- 达到长局或长期无吃子时按子力优势裁定，减少“未完成/平局”标签占比。

推荐先生成一小批探测数据，观察胜负/平局比例：

```bash
dotnet run --project ChineseChess/ChineseChess.SelfPlay -- \
  --games 20 \
  --out data/selfplay/probe.jsonl \
  --level 3 \
  --timeMs 120 \
  --maxMoves 220 \
  --topK 2 \
  --nearBestWindow 80 \
  --randomOpeningPlies 12 \
  --openingTopK 6 \
  --openingNearBestWindow 180 \
  --adjudicateAfterMoves 120 \
  --adjudicateNoCapturePlies 60 \
  --adjudicateMaterialMargin 450 \
  --adjudicateAtMaxMoves true
```

如果探测数据中仍然有较多平局，可以生成训练集时跳过平局局谱：

```bash
dotnet run --project ChineseChess/ChineseChess.SelfPlay -- \
  --games 500 \
  --out data/selfplay/train-nondraw.jsonl \
  --level 3 \
  --timeMs 120 \
  --skipDraws true
```

更高质量但更慢的训练数据可以使用 `--level 4 --timeMs 300`。如果重复局谱仍然较多，可以提高 `--openingTopK` 或 `--openingNearBestWindow`；如果裁定过于激进，可以提高 `--adjudicateMaterialMargin`。

## 训练脚本

保留通用 JSONL 训练脚本 `train_pytorch_jsonl.py`，并新增针对 `ChineseChess.SelfPlay` 输出格式的专用脚本 `train_chinese_chess_policy_value.py`。

示例：

```bash
python train_chinese_chess_policy_value.py \
  --input data/selfplay/train-nondraw.jsonl \
  --output artifacts/chinese_chess_policy_value.pt \
  --epochs 20 \
  --batch-size 256 \
  --hidden-dim 512 \
  --num-layers 3
```
