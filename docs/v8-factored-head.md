# V8 Factored Head 训练 + 测试 完整记录

> 日期: 2026-06-27 | 架构: cnn_factored_v8 | 基于 V7 backbone + V6 factored head

## 1. 技术选型

### 问题诊断
V7 (cnn_full_v7) 经过 683K 自对弈数据训练后，棋力提升微弱：
- Classic Level 4 对比：V7 batch2 0-16-4，旧版 V7 0-20-0
- 训练指标：Top-10 始终卡在 34%，数据量翻倍无改善

**根因：参数瓶颈。** V7 的 `Linear(720 → 8100)` 层占 5.83M 参数（总参数 6.32M 的 92%），对 8100 维策略空间形成严重过拟合。

### 方案选择
回退到 V6 的 **factored from/to head**，结合 V7 的**优化 backbone**：
- `from_head[90]`：哪个子移动（90 个格子）
- `to_head[90]`：移到哪个格子（90 个格子）
- 合并：`policy[from*90+to] = from_logit[from] + to_logit[to]`

### 参数量对比

| 版本 | 总参数 | 策略头参数 | ONNX 大小 | 每 epoch 时间 |
|------|--------|----------|----------|-------------|
| V7 (full 8100) | 6.32M | 5.83M (92%) | 24.1 MB | ~7 min |
| **V8 (factored)** | **0.61M** | **0.13M (21%)** | **2.3 MB** | **~1 min** |

参数降低 10 倍，ONNX 缩小 10 倍，训练加速 ~7 倍。

## 2. 架构设计

```
输入: [batch, 16, 10, 9] (14 piece + 2 side planes)
  │
  ├─ stem: Conv(16→64) + BN + ReLU
  ├─ tower: 6× ResBlock(channels=64)
  │
  ├─ Policy Head (factored)
  │   ├─ policy_conv: Conv2d(64→8, 1×1)
  │   ├─ BN + ReLU → flatten → Dropout(0.2)
  │   ├─ from_fc: Linear(720→90)  → from_logits[batch, 90]
  │   └─ to_fc:   Linear(720→90)  → to_logits[batch, 90]
  │
  └─ Value Head
      ├─ value_conv: Conv2d(64→1, 1×1)
      ├─ BN + ReLU → flatten → Dropout(0.2)
      └─ fc(90→256)→ReLU→Dropout→fc(256→1)→tanh
          → value_pred[batch]
```

## 3. 训练

### 训练参数
```
数据: train_batch2_60k.jsonl (54K train / 6K val)
Checkpoint: cnn_policy_value_v7_cont2.best.pt (迁移 backbone + value head weights)
Epochs: 30 | Batch: 256 | LR: 1e-4 (Cosine Annealing)
Weight Decay: 1e-2 | Workers: 0
```

### 训练曲线

| Epoch | Val Loss | Top-5 | Top-10 | From Acc | To Acc |
|-------|---------|-------|--------|----------|--------|
| 1 | 5.76 | 18.3% | 32.7% | 9.2% | 5.0% |
| 3 | 5.13 | 19.3% | 33.0% | 23.6% | 6.0% |
| 5 | 4.74 | 20.0% | 33.8% | 26.7% | 7.1% |
| 7 | 4.55 | 20.8% | 35.0% | 27.7% | 7.9% |
| **9** | **4.36** | **21.3%** | **35.3%** | **27.7%** | **8.8%** |
| 10 | 4.39 | 20.3% | 34.2% | 26.5% | 7.8% |
| 15 | 4.49 | 19.3% | 34.3% | 26.3% | 7.8% |
| 20 | 4.47 | 18.7% | 34.0% | 26.4% | 7.9% |
| 25 | 4.48 | 18.6% | 34.0% | 26.5% | 8.1% |
| 30 | 4.49 | 18.9% | 33.9% | 26.5% | 8.1% |

### 训练分析
- **Best epoch: 9** — val_loss=4.36, Top-10=35.3%, from_acc=27.7%
- **过去 10 倍 VS V7：** Top-10 从 34.1% → 35.3%（+1.2pp）
- **from_acc 快速收敛到 27.7%**：从 90 个格子中选「Which piece」的正确率，远超随机 1.1%
- **to_acc 仅 8.8%**：「Move where」更难，但 factored 表示让两阶段互相独立学
- **epoch 9 后 val 开始退化**：Cosine LR 降到 5e-5 以下，过拟合抬头
- **拓扑 Loss 不可对比 V7：** V7 直接用 CE(8100)，V8 用 CE(90)+CE(90) 求和，数值无直接对比意义

### 与 V7 的关键差异
- V7 训练：train_loss 3.38→2.17（降 36%），val_loss 最低 3.21
- V8 训练：train_loss 6.15→3.73（降 39%），val_loss 最低 4.36
- V8 from_acc 从 9.2% 跃升至 27.7%，展示了 factored head 对「选子」的强学习能力
- V8 收敛更快：epoch 9 即过拟合（V7 是 epoch 5），说明参数少了更容易进入过拟合区

## 4. ONNX 导出

```
python export_cnn_onnx.py --input cnn_policy_value_v8.best.pt --output cnn_policy_value.onnx
```

- 架构: cnn_factored_v5, 607,175 参数
- ONNX: **2.3 MB** (V7 是 24.1 MB，缩小 10 倍)
- 输入: 16 planes
- 输出: `from_logits[90]`, `to_logits[90]`, `value_pred[1]`
- C# 推理层**零改动**：自动检测输出名适配 factored 模型

## 5. 棋力测试

### 配置
```
MCTS: 3000 sims (V7 是 600, 提升 5×)
Classic: Level 4 (8-ply Alpha-Beta + QSearch), 5s/步
对局: 20 局（轮流执红/黑）
```

### 结果对比

| 版本 | 参数 | MCTS | ONNX 胜 | Classic 胜 | 和棋 | 平均步数 |
|------|------|------|---------|-----------|------|---------|
| V7 batch2 | 6.32M | 600 | 0 | 16 | **4** | 42.0 |
| **V8 factored** | **0.61M** | **3000** | **0** | **18** | **2** | **41.5** |

### 逐局记录

V8 (3000 sims):
- 0 胜 / 18 负 / 2 和
- 和棋: Game 12 (59步), Game 19 (86步) — 三次重复局面
- 13 局 ≤ 50 步结束，最短 13 步被将杀

V7 (600 sims):
- 0 胜 / 16 负 / 4 和
- 和棋: Game 8 (83步), Game 13 (57步), Game 15 (60步), Game 18 (75步)

### 输棋模式分析

V8 的短局（13、17、18 步）表明开局阶段存在严重漏洞，Classic AI 能在极短时间内找到将杀路径。这与 V7 类似（Game 4 和 Game 16 也都是 15 步），说明 factored head 并未解决 CNN 在开局感知上的根本弱点。

### 结论

| 维度 | 结论 |
|------|------|
| 训练效率 | ✅ 参数少 10 倍，训练快 10 倍，ONNX 小 10 倍 |
| 训练指标 | ✅ Top-10 35.3% vs V7 34.1% (+1.2pp) |
| 实际棋力 | ❌ 和棋率 10% vs V7 20%，输棋更多 |
| MCTS 提升 | ❌ 3000 sims 未转化为棋力优势 |

**核心教训：训练指标（Top-10）与实战棋力不相关。** V8 有更好的 Top-10 和 from_acc，但在 Classic L4 面前表现更差。CNN 本身的表征能力（6 层 ResBlock 在 10×9 棋盘上）可能是真正的瓶颈，而非策略头的形式。

### 下一步建议

1. **放弃纯 CNN 路线**：CNN 对 10×9 棋盘的感知能力不足，无论 flat 还是 factored 策略头都打不过 8 层 Alpha-Beta
2. **尝试 Transformer 编码器**：Self-attention 对大棋盘、长距离依赖（车炮远攻）天然更优
3. **MCTS 超参数调优**：当前 3000 sims 也可能不在最优区间，需要系统性扫参数


## 6. 踩坑

| 问题 | 原因 | 解决 |
|------|------|------|
| `num_workers=2` 导致训练崩溃 | Windows 多进程 pickle 序列化问题 | 设 `num_workers=0` |
| export_cnn_onnx.py 不支持 policy_channels | 旧 factored 模型固定 policy_channels=2 | 新增 `policy_channels` 参数支持 |
| C# 推理输出 isV5Model 检测 | Auto-detect 判断 `from_logits` 输出名即可 | 无需改动，一直支持 |
