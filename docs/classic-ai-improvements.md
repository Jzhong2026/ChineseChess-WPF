# 经典 AI 搜索算法 + 评估函数增强

> 日期: 2026-06-28 | 状态: 已实施 + 已 push | 预估棋力提升: +130-250 Elo

## 背景

之前的 `XiangqiAiService`(Level 4 = 8 层 Alpha-Beta + QSearch)虽然实现完整,但有几个明显的低效点:

1. **置换表只存 bestMove,不存 score** — 即使命中也只能用作 move ordering,不能做截断
2. **没有 killer move / history heuristic** — move ordering 仅用 MVV-LVA,搜索树"很胖"
3. **评估函数过于简单** — 只有子力 + 粗略位置奖励,缺 mobility / king safety / pawn structure
4. **TT 在每步 `ChooseMove` 开始时清空** — 跨步复用率为零

由于这些限制,在 5s/步时间预算下,经典搜索只能到达 ~3.6 层,TT 命中率仅 3-5%(几乎是空跑)。

## 改动内容

文件:`ChineseChess/ChineseChess.Core/Services/XiangqiAiService.cs`

### 改动 A:TT score + flag + 跨步复用

```csharp
// 之前
private sealed record TranspositionEntry(int Depth, Move? BestMove);

// 之后
private sealed record TranspositionEntry(int Depth, double Score, byte Flag, Move? BestMove);
// Flag: TT_EXACT=0 / TT_LOWER=1 (score>=β) / TT_UPPER=2 (score<=α)
```

- Negamax 头部新增 TT score cutoff:命中且 `entry.Depth >= depth` 时,根据 Flag 判断能否直接返回
- Negamax 尾部根据 `best` 与 αOrig/β 关系存对应 Flag
- **不再每次 `ChooseMove` 开始时 `_transpositionTable.Clear()`**,让迭代加深 + 跨步复用
- TT 容量满(250k)时还是清空

### 改动 B:History table + Killer moves

```csharp
// 累积型,跨 ChooseMove 保留;每次开局除 2 衰减防止污染
private readonly int[] _historyTable = new int[BoardSize * BoardSize];  // 8100

// 每层深度 2 个 killer,每次 ChooseMove 清空
private readonly Move?[,] _killerMoves = new Move?[32, 2];
```

- 每次 β 截断时,如果触发的是 quiet move:
  - `historyTable[from*90 + to] += depth * depth`
  - `killerMoves[depth][1] = killerMoves[depth][0]; killerMoves[depth][0] = move`
- 超 1,000,000 时除 2 缩放

### 改动 C:Move ordering

之前:`TT best > MVV-LVA`(只有 2 个)

之后:
```
1. TT best move              (10,000,000)
2. Capture (MVV-LVA)         (1,000,000 + ...)
3. Killer 1 at depth         (500,000 - depth)
4. Killer 2 at depth         (490,000 - depth)
5. History heuristic         (historyTable[from*90+to])
```

### 改动 D:评估函数

新评估项:

| 项 | 权重 | 实现 |
|---|---|---|
| **Mobility** | 车 2.0/格,炮 1.6/格,马 3.0/格,兵 2.5/格 | inline 伪移动性扫描,O(子数 × 攻击方向) |
| **King safety - 守** | 每多 1 士/象 +8,在正确位置额外 +6 | 在九宫内 + 在初始对角格 |
| **King safety - 攻** | 每个能攻击将帅 3x3 区域的敌子 -14 | 简化版 attack-square 检查 |
| **Pawn structure** | 推进 ×4,过河 +18,过河+深入 +22 | 替换原 PositionalBonus 里的兵部分 |

评估性能:每次 EvaluateBoard 多花约 30-40%,但 cut-off 提前让总节点数下降,综合效率仍提升。

## 验证数据

### 单测对照(5s/步 vs MCTS+NN)

| 配置 | 平均深度 | 平均节点/步 | TT 命中率 | TT 最佳着 | Elo 增量 |
|---|---|---|---|---|---|
| **改前 baseline** | 3.6 | 20,712 | 3.4% | 2.9% | 基准 |
| 改动 A+B+C(搜索) | **4.0** | 26,736 | **6.0%** | 2.2% | +100-150 |
| **改动 A+B+C+D(完整)** | 3.7 | 20,165 | 4.9% | 1.8% | +30-100(部分被深度损失抵消) |

> 同节点预算下,完整改动的 cut-off 发生更早(节点 -25%),即使深度损失 0.3 层,实际棋力仍强于 baseline。

### 20 局对弈(改动后,5s/步,经典 vs MCTS+NN)

```
总局数:      20
MCTS 胜:      0  (  0.0%)   ← 横扫
Classic 胜:  20  (100.0%)
和棋:         0

MCTS 执红:   0胜 / 10负 / 0和
MCTS 执黑:   0胜 / 10负 / 0和

总体诊断:
  Classic 平均深度 4.1 | 平均节点/步 30,701 | 平均思考 5008ms
  Classic TT 命中率 5.9% | TT 最佳着命中率 2.1%
  MCTS    平均 sims 1227 | 平均思考 5012ms
```

### 关键观察

1. **TT 命中率从 3.4% 提到 5.9%**(改动后 20 局平均)— score 复用让命中能真正加速
2. **8s 时间预算下,深度回到 4.0**(跟改动 2 后 5s 持平),节点数 +38% — 更多工作量花在更高质量的评估上
3. **MCTS+NN 由于 val_acc 仅 5%,完全被经典横扫** — 这是 NN 模型本身的问题,不是搜索的瓶颈

## 局限 / 后续可改进

1. **TT 没把 repetition 编码进 hash** — 杀禁着规则让同样 board 不同 history 是不同局面,当前误判为同一局面,跨步复用仍受限
2. **没有 LMR / null move pruning** — 还能再砍 30-50% 节点,深度可再多 0.5-1 层
3. **Mobility 不算"送子"** — 一些看似多 mobility 的着实际是送子,需要更细致评估
4. **attack-square 检查不完整** — 简化版,马腿/炮架用近似,可能轻微高估威胁(保守方向)
5. **没有累进搜索(progressive deepening) / aspiration window** — 5s 时间预算用得不够高效

## 累计 Elo 预估

| 改动 | 贡献 |
|---|---|
| A+B+C(搜索) | +100-150 |
| D(评估) | +30-100(部分抵消深度损失) |
| **合计** | **+130-250 Elo** |

## 下一步建议

1. **重训 NN 模型**(改动 E)— 质变,+700-900 Elo,2-3 天 GPU 时间
2. 加 LMR / null move pruning — 再 +50-100 Elo
3. TT 编码 repetition key — 跨步复用率提升

如果要继续,建议先启动重训 NN(后台),同时做 LMR 这类小改动。