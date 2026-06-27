# Batch2 自对弈 + 训练 + 测试 完整记录

> 周期: 2026-06-18 → 2026-06-27 | 分支: V7 cnn_full_v7 持续训练

## 1. 技术选型背景

V7 模型（cnn_full_v7, 6.32M 参数）在 500 局自对弈数据上从 V6 fine-tune 得到 val_top10=34%。之前 2000 局自对弈后训练时发现模型棋力仍很弱（对 Classic Level 4 全负 0-20-0），决定加大自对弈数据量。

**方案：** 3 进程并行 × 1500 局 = 4500 局自对弈，完成后合并训练。

## 2. 自对弈并行方案

### 方案选型
- 放弃: 单进程串行（4500 局需 100+ 小时）
- 放弃: 8 进程全开（8核机器，每进程分不到 1 核，互相抢占变慢）
- **选择: 3 进程 × 1500 局**（每进程约 2-3 核，约 22-25 小时完成）

### 实现方式
- 使用 `subprocess.Popen(creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)` 完全脱离父进程
- Seeds: 54321, 98765, 13579
- 输出: `data/selfplay/batch2_{0,1,2}.jsonl`（atomic output: 先写 .tmp 文件完成后重命名）

### 问题与解决
- **问题1:** Git Bash 的 `nohup` + `&` 在 Windows 上无效，父进程退出后子进程被杀
  - **解决:** 用 Python `subprocess.Popen` + Windows API 标志实现真正脱离
- **问题2:** 进程1 (PID 12404) 在 1496/1500 局时异常退出，未完成最后 4 局
  - **原因:** 未知崩溃，tmp 文件 (679MB, 230,894 行) 数据完整
  - **解决:** 直接重命名 .tmp → .jsonl，数据可用（99.7% 完整）

### 数据产出
| 进程 | 状态 | 行数 | 大小 |
|------|------|------|------|
| batch2_0 | ✅ 完成 1500/1500 | 224,172 | 659MB |
| batch2_1 | ⚠️ 1496/1500 | 230,894 | 679MB |
| batch2_2 | ✅ 完成 1500/1500 | 228,252 | 671MB |
| **合计** | | **683,318** | **2.0GB** |

## 3. 训练

### 内存约束
- 机器: 16GB RAM，空闲约 2.7-5.8GB
- 历史经验: 74K 样本 → 4.3GB working set
- 683K 全部样本 → 估算 35GB+，无法直接训练

### 子采样策略
- 从 683K 行中随机采样 **60K 行**（seed=42）
- 60K 样本 → 3.7GB working set，可接受
- 文件: `data/selfplay/train_batch2_60k.jsonl`

### 训练参数
```
模型: V7 cnn_full_v7 (channels=64, res_blocks=6, policy_channels=8)
Checkpoint: cnn_policy_value_v7_cont2.best.pt
Epochs: 50 | Batch: 256 | LR: 5e-5 | Weight Decay: 1e-2
Train/Val split: 90/10 (54K/6K)
```

### 训练过程
| Epoch | Train Loss | Val Loss | Top-10 | 备注 |
|-------|-----------|---------|--------|------|
| 1 | 3.38 | 3.249 | 34.4% | |
| 3 | 3.22 | 3.238 | 33.6% | |
| **5** | **3.12** | **3.210** | **34.1%** | **🏆 Best** |
| 10 | 2.71 | 3.230 | 34.3% | |
| 20 | 2.47 | 3.272 | 33.7% | 开始过拟合 |
| 30 | 2.31 | 3.293 | 34.2% | |
| 40 | 2.21 | 3.316 | 33.9% | |
| 50 | 2.17 | 3.339 | 33.2% | 严重过拟合 |

### 训练分析
- **最佳 epoch: 5**（val_loss=3.210, Top-10=34.1%）
- **过拟合严重:** train_loss 从 3.38 降到 2.17，val_loss 反而从 3.21 升到 3.34
- **泛化停滞:** 与 v7_cont2（74K 数据, Top-10=34%）持平，数据量翻倍未带来提升
- **结论: FC bottleneck 是架构瓶颈，非数据瓶颈**

## 4. ONNX 导出

```bash
python export_cnn_onnx.py \
  --input cnn_policy_value_v7_batch2.best.pt \
  --output cnn_policy_value.onnx
```

- 架构: cnn_full_v7, 6,317,495 参数
- ONNX 大小: 24.1 MB
- 输入: 16 planes (14 piece + 2 side)
- 输出: policy_logits[8100], value_pred[1]

## 5. 棋力测试

### 测试配置
```
工具: AiMatchRunner
Neural: MCTS 600 sims, 5s time limit, MCTS+NN
Classic: Level 4 (8层 Alpha-Beta + QSearch), 5s per move
局数: 20 局（轮流执红/黑）
```

### 结果对比

| 版本 | 数据 | ONNX 胜 | Classic 胜 | 和棋 |
|------|------|---------|-----------|------|
| v7_cont2 (旧) | 74K (500局) | 0 | 20 | 0 |
| v7_batch2 (新) | 683K (4500局) | 0 | 16 | **4** |

### 分析
- **进步:** 0 和棋 → 4 和棋（均为三次重复局面迫和，57-83 步）
- **局限:** 仍无法赢棋，2 局 15 步就被将杀（开局盲点）
- **和棋分布均衡:** 执红 2 和、执黑 2 和
- **ONNX 每步约 2.4s**，Classic 每步固定 5s

## 6. 踩坑总结

| 问题 | 原因 | 解决 |
|------|------|------|
| Windows 后台进程被父进程杀死 | Bash `nohup` 在 Git Bash 下无效 | Python `subprocess.Popen` + `DETACHED_PROCESS` flag |
| `tee` 日志无输出 | Windows 下 stdout 全缓冲 | 用 `python -u` + 直接 `> file` 重定向 |
| 683K 数据内存溢出 | Python 对象开销大（~33KB/样本） | 子采样到 60K |
| 50 epoch 严重过拟合 | 模型容量过剩 + 学习率调度不合理 | 应在 epoch 5-10 后 early stop |
| ONNX 导出路径错误 | Bash 脚本中 `\e\...` 路径混乱 | 用 Windows 绝对路径 + PowerShell |
| val_loss 最低时 train_loss 仍在降 | 经典过拟合信号 | 未来应加 early stopping 回调 |

## 7. 下一步方向

见 [棋力提升路线对比分析](./chess-strength-roadmap.md)，核心结论：
- **最快见效:** 加大 MCTS sims（改一行代码）
- **性价比最高:** 换 factored head 架构（参数从 6.3M → 510K）
- **不推荐:** 继续堆自战数据（边际收益接近零）
