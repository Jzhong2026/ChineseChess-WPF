# 中国象棋 AI 项目教程

## 从零构建一个 AlphaZero 风格的象棋 AI

<div align="center">
  <p><strong>技术栈：</strong>C# (.NET 8) + Python (PyTorch) + ONNX Runtime</p>
  <p><strong>适用对象：</strong>有一定编程基础、想了解 AI + 传统搜索结合的技术爱好者</p>
</div>

---

## 目录

### 第一篇：基础篇 — 传统象棋 AI

1. [象棋引擎核心](#1-象棋引擎核心-xiangqiengine)
2. [局面评估函数](#2-局面评估函数)
3. [Minimax + Alpha-Beta 剪枝](#3-minimax--alpha-beta-剪枝)
4. [Quiescence Search（静态搜索）](#4-quiescence-search静态搜索)
5. [Transposition Table（置换表）](#5-transposition-table置换表)
6. [走法排序与迭代加深](#6-走法排序与迭代加深)
7. [完整 AI 服务与难度分级](#7-完整-ai-服务与难度分级)

### 第二篇：进阶篇 — 神经网络象棋 AI

8. [棋盘编码（Board Encoding）](#8-棋盘编码board-encoding)
9. [CNN 策略-价值网络](#9-cnn-策略-价值网络)
10. [自对弈数据生成（Self-Play）](#10-自对弈数据生成self-play)
11. [训练脚本详解](#11-训练脚本详解)
12. [ONNX 模型导出](#12-onnx-模型导出)
13. [MCTS + 神经网络（AlphaZero 核心）](#13-mcts--神经网络alphazero核心)

### 第三篇：实战篇 — 工程落地

14. [C# ONNX Runtime 集成](#14-c-onnx-runtime-集成)
15. [WPF 界面集成](#15-wpf-界面集成)
16. [AI 对弈测试](#16-ai-对弈测试)
17. [并行自对弈加速](#17-并行自对弈加速)
18. [内存优化与生产部署](#18-内存优化与生产部署)

---

# 第一篇：基础篇 — 传统象棋 AI

---

## 1. 象棋引擎核心 (XiangqiEngine)

### 原理

象棋引擎是整个项目的基石。它负责：

- 棋盘状态的表示与初始化
- 所有棋子的走法生成
- 将军/将杀/困毙的判定
- 走法的合法性校验
- 重复局面的检测

### 依赖

```xml
<!-- 纯 .NET 8，无外部依赖 -->
<TargetFramework>net8.0</TargetFramework>
```

### 核心数据结构

```csharp
// 棋盘: Piece?[10, 9] 二维数组，null 表示空格
public sealed record Piece(string Id, Side Side, PieceType Type);

// 坐标: (Row, Col)，Row 0-9，Col 0-8
public readonly record struct Position(int Row, int Col);

// 走法: 从哪里到哪里，吃了什么子
public sealed record Move(Position From, Position To, Piece Piece, Piece? Captured);

// 游戏状态: 棋盘 + 轮次 + 状态 + 历史
public sealed record GameState(Piece?[,] Board, Side Turn, GameStatus Status, IReadOnlyList<Move> History);
```

### 重要方法

| 方法 | 用途 | 核心逻辑 |
|------|------|---------|
| `CreateInitialState()` | 创建初始棋局 | 红方先行，标准布局 |
| `MakeMove(GameState, Move)` | 执行一步走法 | 更新棋盘、切换轮次、检测将军/将杀 |
| `IsLegalMove(board, move, side)` | 校验走法合法性 | 检查目标是否在棋子走法范围内、是否造成己方被将 |
| `IsInCheck(board, side)` | 判断某方是否被将军 | 检查对方所有棋子是否能吃掉己方将/帅 |
| `GetLegalMoves(board, side, history)` | 获取某方所有合法走法 | 生成所有候选走法 → 过滤非法走法 |
| `GetGameStatus(board, turn)` | 获取游戏状态 | Playing / RedCheck / BlackCheck / RedWins / BlackWins / Draw |
| `ApplyMove(board, move)` | 应用走法（只返回棋盘，不返回状态） | Clone 棋盘 → 移动棋子 → 返回新棋盘 |

### 代码位置

- `ChineseChess/ChineseChess.Core/Services/XiangqiEngine.cs`
- `ChineseChess/ChineseChess.Core/Models/XiangqiModels.cs`（数据模型）

---

## 2. 局面评估函数

### 原理

在没有神经网络的情况下，我们需要一个**启发式评分函数**来评估一个局面的好坏。

```
score = Σ(己方子力价值 + 位置奖励) - Σ(对方子力价值 + 位置奖励)
```

### 子力价值表

| 棋子 | 价值 |
|------|------|
| 将/帅 (General) | 10,000 |
| 车 (Rook) | 900 |
| 炮 (Cannon) | 450 |
| 马 (Horse) | 400 |
| 象/相 (Elephant) | 200 |
| 士/仕 (Advisor) | 200 |
| 兵/卒 (Soldier) | 120 |

### 位置奖励

位置奖励给予棋子更好的位置。例如：

- **兵/卒**：过河后位置价值递增（`forwardProgress * 14`），中路更好
- **马**：靠近中心位置更优（`25 - centerDistance * 4`）
- **炮**：中路位置更优

### 代码位置

- `XiangqiAiService.cs` → `EvaluateBoard()` 方法和 `PositionalBonus()` 方法

---

## 3. Minimax + Alpha-Beta 剪枝

### 原理

**Minimax** 是双人博弈的经典搜索算法：

- 我方（MAX）选择评分最高的走法
- 对方（MIN）选择评分最低的走法（对我方而言）
- 递归搜索到固定深度后，用评估函数打分

**Alpha-Beta 剪枝** 是 Minimax 的优化：

- **Alpha**：我方能保证的最低分数（下界）
- **Beta**：对方能保证的最高分数（上界）
- 当 `alpha >= beta` 时，剪枝（该分支无需继续搜索）

```
function Negamax(board, turn, depth, alpha, beta):
    if depth == 0: return Evaluate(board)
    
    for each move in legal_moves:
        score = -Negamax(next_board, opposite(turn), depth-1, -beta, -alpha)
        if score > alpha: alpha = score
        if alpha >= beta: break  // 剪枝
    return alpha
```

### 级别 3（4 层搜索）vs 级别 4（8 层搜索）

| 级别 | 搜索深度 | 特点 |
|------|---------|------|
| Level 1 | 0 | 随机走法，用于测试 |
| Level 2 | 1 | 基本评估，吃最大子 |
| Level 3 | 4 | 4 层搜索 + QSearch，适合快速对战 |
| Level 4 | 8 | 8 层搜索 + QSearch，最强但最慢 |

### 代码位置

- `XiangqiAiService.cs` → `Negamax()` 方法

---

## 4. Quiescence Search（静态搜索）

### 原理

**水平线效应（Horizon Effect）**：固定深度的搜索可能在一个"看上去很好"的局面截断，但实际上对方下一步就能吃回一个大子。

**Quiescence Search** 解决了这个问题：

- 到达搜索树末端时，不直接用评估函数打分
- 而是继续搜索所有"吃子走法"（战术性走法）
- 直到局面"安静"（没有有意义的吃子）为止

```
function Quiescence(board, turn, alpha, beta, depth):
    stand_pat = Evaluate(board)          // 当前局面的静态度量
    if stand_pat >= beta: return beta    // 对方不愿意
    if alpha < stand_pat: alpha = stand_pat
    
    for each capture_move in legal_captures:
        score = -Quiescence(next_board, opposite(turn), -beta, -alpha, depth+1)
        if score >= beta: return beta
        if score > alpha: alpha = score
    return alpha
```

### 关键参数

- 最大 QSearch 深度：6 层（防止无限递归）
- 只搜索吃子走法（`move.Captured is not null`）

### 代码位置

- `XiangqiAiService.cs` → `Quiescence()` 方法

---

## 5. Transposition Table（置换表）

### 原理

象棋中，同样的局面可能通过不同的走法顺序到达（例如：车 1 平 2 再车 2 平 1，等同于交换两步棋）。重复搜索相同局面浪费算力。

**Zobrist Hashing** 为每个局面生成一个 64 位哈希值：

1. 为每个 (位置, 棋子颜色, 棋子类型) 组合分配一个随机 64 位数
2. 哈希值 = 所有棋子对应随机数的 XOR + 当前轮次的随机数
3. XOR 运算可以增量更新（移动棋子时只需 XOR 新旧位置的随机数）

**Transposition Table** 存储已搜索过的局面：

```
hash = ComputeZobristHash(board, turn)
if hash in transposition_table:
    return transposition_table[hash].best_move  // 缓存命中
```

### 关键参数

- 最大条目数：250,000
- 替换策略：深度优先（保留搜索更深的条目）

### Demo 学习

```bash
# 基础概念 Demo - 跑一下看看 TT 能加速多少
python Learning/demo1_basic_concepts/transposition_table.py
```

### 代码位置

- `XiangqiAiService.cs` → `ComputeZobristHash()`, `StoreTransposition()` 方法

---

## 6. 走法排序与迭代加深

### 走法排序（Move Ordering）

Alpha-Beta 剪枝的效率高度依赖于走法搜索的顺序。好的走法排前面，剪枝越多。

排序优先级：

```
1. TT 最佳走法  → 1,000,000 分（最高的优先级）
2. 吃子走法     → 被吃棋子价值 - 己方棋子价值/20（MVV-LVA）
3. 其他走法     → 0 分
```

### 迭代加深（Iterative Deepening）

从深度 1 开始逐层加深搜索，每层利用上一层的 TT 最佳走法来排序：

```
for depth = 1 to max_depth:
    search(depth, alpha, beta)
    // 保存当前层的最佳走法到 TT
    // 下一层优先搜索 TT 最佳走法
```

优势：
- 时间可控（随时可以截断）
- TT 为下一层提供走法排序线索

### 代码位置

- `XiangqiAiService.cs` → `OrderMoves()` 方法, `ChooseMove()` 的迭代加深循环

---

## 7. 完整 AI 服务与难度分级

### 架构

```
XiangqiAiService
├── Level 1: RandomMove          — 纯随机
├── Level 2: PickBasicMove       — 贪心（吃最大子）
├── Level 3: Negamax(depth=4)    — 4层搜索 + QSearch + TT
└── Level 4: Negamax(depth=8)    — 8层搜索 + QSearch + TT
```

### 输入参数

| 参数 | 用途 | 默认值 |
|------|------|--------|
| `board` | 当前棋盘 | - |
| `side` | 当前轮次 | - |
| `level` | AI 强度 (1-4) | 2 |
| `history` | 历史走法（重复局面检测） | - |
| `timeLimitMs` | 思考时间上限 | 1200ms |
| `selectionOptions` | 随机选择参数（自对弈用） | Deterministic |

### 输出

```csharp
public sealed record AiSearchResult(Move? Move, SearchStats Stats);

public sealed record SearchStats(
    int DepthReached,      // 实际达到的搜索深度
    int Nodes,             // 搜索节点数
    double TimeMs,         // 耗时
    double BestScore,      // 最佳走法评分
    int TtHits,            // TT 命中次数
    int TtStores,          // TT 存储次数
    int TtBestMoveHits,    // TT 最佳走法命中
    int TtScoreHits);      // TT 评分命中
```

### 代码位置

- `ChineseChess/ChineseChess.Core/Services/XiangqiAiService.cs`

---

# 第二篇：进阶篇 — 神经网络象棋 AI

---

## 8. 棋盘编码（Board Encoding）

### 原理

神经网络不能直接理解"马在 (2,1)"这样的符号信息，需要将棋盘编码为数值张量。

### 14 平面编码（V5 之前）

每个棋子类型 + 颜色组合占据一个独立的 10×9 平面（channel）：

```
平面 0:  红方帅 (Red General)
平面 1:  红方仕 (Red Advisor)
平面 2:  红方相 (Red Elephant)
平面 3:  红方马 (Red Horse)
平面 4:  红方车 (Red Rook)
平面 5:  红方炮 (Red Cannon)
平面 6:  红方兵 (Red Soldier)
平面 7:  黑方将 (Black General)
...以此类推到平面 13
```

输出为 `float[1260]`，布局为 `index = (row * 9 + col) * 14 + plane`（square-major）。

### 16 平面编码（V5+）

增加 2 个 side-to-move 平面：

```
平面 14:  红方行棋指示（全 1 如果红方走，否则全 0）
平面 15:  黑方行棋指示（全 1 如果黑方走，否则全 0）
```

这告诉神经网络当前轮到谁走棋，对价值判断至关重要。

### 输入形状

CNN 需要 `[batch, planes, rows, cols]` = `[1, 16, 10, 9]` 的 4D 张量。

编码过程：

```
flat[1260]  ← BoardEncoder.Encode(board)     // square-major [sq*14+plane]
     ↓ reshape + transpose
tensor[1, 16, 10, 9]                         // plane-major [N, C, H, W]
```

### 代码位置

- `ChineseChess/ChineseChess.Core/Encoding/BoardEncoder.cs`（C# 编码）
- `train_cnn_policy_value.py` → `BoardEncoder` 类（Python 编码，与 C# 保持兼容）

---

## 9. CNN 策略-价值网络

### 架构演进

#### V5 — Factored 架构（分体式策略头）

```
输入 [1, 16, 10, 9]
  │
  ┌─ Stem: Conv2d(3×3) → BN → ReLU
  │
  ├─ Tower: 6× ResidualBlock (Conv2d → BN → ReLU × 2)
  │
  ├─ From Head → from_logits [90]          ← 从哪个位置走
  ├─ To Head   → to_logits   [90]          ← 走到哪个位置（独立预测）
  └─ Value Head → value_pred  [1]          ← 局面评分 [-1, 1]
```

问题：`from_logits[90] + to_logits[90]` 假设 From 和 To 是独立的。

实际上马在 (2,1) 只能去 8 个位置，车在 (0,0) 能去 17 个位置。独立预测浪费模型容量。

#### V7 — Full 8100 架构（全连接策略头）

```
输入 [1, 16, 10, 9]
  │
  ├─ Stem → Tower (6 ResidualBlocks)
  │
  ├─ Policy Head → policy_logits [8100]    ← from*90+to 的联合概率
  └─ Value Head → value_pred  [1]
```

优势：策略头直接输出 `P(from, to)` 的联合分布，不需要独立假设。

#### V7 Full vs Factored 对比

| 特性 | V5 Factored | V7 Full 8100 |
|------|-------------|--------------|
| 策略输出 | from[90] + to[90] | policy[8100] |
| 参数 | ~6.3M | ~6.3M |
| From Acc | ~25% | ~32% |
| To Acc | ~8% | ~10% |
| Top-10 合法 | ~33% | ~34% |
| C# 使用 | from+to 组合 | 直接取 policy[actionId] |
| ONNX 输出 | 3 个输出 | 2 个输出 |

### 关键方法

```python
class CNNFullPolicyValueNet(nn.Module):
    def __init__(self, in_channels=16, channels=64, res_blocks=6, ...)
        # Stem: Conv2d 3×3 → BN → ReLU
        # Tower: 6× ResidualBlock
        # PolicyHead: Conv2d → BN → ReLU → Flatten → Linear(2*90, 8100)
        # ValueHead: Conv2d → BN → ReLU → Flatten → Linear(90, 256) → Linear(256, 1) → Tanh
    
    def forward(self, x):
        # x: [batch, 16, 10, 9]
        # returns: (policy_logits [batch, 8100], value_pred [batch])
```

### 损失函数

```
Loss = (value_target - value_pred)^2           # MSE 价值损失
     + CrossEntropy(policy_logits, move_target) # 策略交叉熵
     + L2_weight * Σ(weight^2)                  # L2 正则化
```

### 代码位置

- `train_cnn_policy_value.py` → `CNNFullPolicyValueNet` 类

---

## 10. 自对弈数据生成（Self-Play）

### 原理

训练神经网络需要大量高质量的对局数据。自对弈是用 AI 自己跟自己下棋来生成数据。

自对弈使用**经典 AI**（XiangqiAiService）进行对弈，因为：

- 经典 AI 速度快（主要用 C++ 级别的搜索）
- 无需加载神经网络
- 可以随机化走法选择，增加数据多样性

### 数据格式

每行 JSONL 包含一个局面样本：

```json
{
  "gameId": 1,
  "moveIndex": 5,
  "side": "Red",
  "sideToMove": 1,
  "boardEncoding": [0.0, 0.0, ...],     // float[1260]
  "legalMoves": [123, 456, ...],         // 合法走法的 actionId
  "selectedMove": 456,                   // AI 实际选择的走法
  "result": 1,                           // 终局结果: 1=红胜, -1=黑胜, 0=和棋
  "searchScoreSidePerspective": 120.5,   // 搜索评分
  "searchScoreRedPerspective": 120.5,    // 红方视角评分
  "depthReached": 8,                     // 搜索深度
  "valueWeight": 1,                      // 价值权重（裁定局降低）
  "policyWeight": 1,                     // 策略权重
  "endReason": "RedWins"
}
```

### 随机化策略

为避免自对弈反复走同一盘棋：

| 参数 | 值 | 效果 |
|------|------|------|
| `topK=2` | 从 Top-2 走法中采样 | 增加多样性 |
| `nearBestWindow=80` | 80 分以内的走法都算"近似最优" | 允许合理但不完美的走法 |
| `randomOpeningPlies=12` | 前 12 步有额外随机性 | 开局多样化 |
| `openingTopK=6` | 开局 Top-6 采样 | 开局广泛探索 |

### 裁定规则

长局或长期无吃子时，按子力优势裁定胜负：

```
无吃子超过 60 步 + 子力优势 > 450 分 → 优势方胜（权重 0.75）
超过 220 步 → 按子力裁定（权重 0.5）
```

### 代码位置

- `ChineseChess/ChineseChess.SelfPlay/Program.cs`
- `SelfPlayOptions` 类（参数配置）

---

## 11. 训练脚本详解

### 训练流程

```
train_500.jsonl (74K 行)
    +
train_2000.jsonl (188K 行)    ← 修复了崩溃导致的截断
    ||
train_combined.jsonl (262K 行) ← 合并数据集
    ||
train_v7_combined.py           ← 分块加载 + 50 epoch
    ||
cnn_policy_value_v7_cont2.best.pt  ← 最佳 checkpoint (epoch 10)
    ||
export_cnn_onnx.py             ← ONNX 导出
    ||
cnn_policy_value.onnx (24.1MB) ← 部署到 C#
```

### 脚本列表

| 脚本 | 用途 | 关键参数 |
|------|------|---------|
| `train_chinese_chess_policy_value.py` | MLP 策略-价值网络训练 | `--hidden-dim 512`, `--num-layers 3` |
| `train_cnn_policy_value.py` | CNN 策略-价值网络训练 | `--channels 64`, `--res-blocks 6` |
| `train_v7_continue.py` | V7 继续训练（全量加载） | `--checkpoint`, `--epochs 50` |
| `train_v7_combined.py` | V7 分块训练（内存友好） | Chunk size=5000, StreamingDataset |
| `train_v7_full_policy.py` | V7 Full 8100 从头训练 | 从随机初始化开始 |

### 训练参数

| 参数 | 值 | 说明 |
|------|------|------|
| `epochs` | 50 | 训练轮数 |
| `batch-size` | 256 | 每批样本数 |
| `lr` | 5e-5 | 学习率 |
| `lr-scheduler` | CosineAnnealingLR | 余弦退火学习率衰减 |
| `val-split` | 0.05 | 5% 作为验证集 |

### 内存优化

`train_v7_combined.py` 解决了之前的内存爆炸问题：

```
旧方案: 一次性加载 262K 样本 → 14GB 内存 ❌
新方案: 预处理为 5000 样本的 chunk → 670MB 内存 ✅
        训练时逐 chunk 加载，用完释放
```

### 代码位置

- `train_v7_combined.py` → `ChunkedDataset` 类
- `train_cnn_policy_value.py` → 训练循环 + 数据增强（水平翻转）

---

## 12. ONNX 模型导出

### 原理

PyTorch 模型不能直接在 C# 中加载，需要导出为 **ONNX**（Open Neural Network Exchange）格式——一种跨框架的模型交换格式。

### 导出过程

```python
torch.onnx.export(
    model,
    dummy_input,        # 形状 [1, 16, 10, 9] 的虚拟输入
    output_path,        # artifacts/cnn_policy_value.onnx
    export_params=True,
    opset_version=17,
    input_names=["board_input"],
    output_names=["policy_logits", "value_pred"],
    dynamic_axes={"board_input": {0: "batch_size"}, ...})
```

### V7 Full 模型输出

```
board_input [1, 16, 10, 9]          ← 输入棋盘
    │
    ├─ policy_logits [1, 8100]      ← 策略输出（所有 from*90+to 组合）
    └─ value_pred    [1]            ← 价值输出 [-1, 1]
```

### C# 集成注意事项

```
- V7 Full 8100 模型有 2 个输出: policy_logits[1,8100], value_pred[1]
- 输入平面: 16 (14 piece + 2 side-to-move)
- Value 是从当前走棋方视角输出 (+1 = 我优势)
- Side-to-move 编码: plane[14]=1 if Red, plane[15]=1 if Black
```

### 自动化导出

`auto_export_onnx.py` 监控训练日志，训练完成后自动执行 ONNX 导出：

```bash
python auto_export_onnx.py  # 后台运行，训练完成自动导出
```

### 代码位置

- `export_cnn_onnx.py`（导出脚本）
- `auto_export_onnx.py`（自动导出监控）

---

## 13. MCTS + 神经网络（AlphaZero 核心）

### 原理

**Monte Carlo Tree Search (MCTS)** 是一种结合搜索和神经网络的算法，AlphaGo/AlphaZero 的核心技术。

### 四步循环

```
1. SELECT（选择）
   ┌─────────────────────────────────────────────┐
   │  从根节点开始，按 PUCT 公式选择子节点，     │
   │  直到到达一个叶节点                          │
   │                                             │
   │  PUCT: Q(s,a) + c_puct * P(s,a) * √N(p)    │
   │                    ─────────────────────     │
   │                        1 + N(s,a)           │
   └─────────────────────────────────────────────┘
   
2. EXPAND（扩展）
   ┌─────────────────────────────────────────────┐
   │  叶节点生成所有合法走法作为子节点，         │
   │  用 NN 的 Policy Head 给每个子节点赋先验概率 │
   └─────────────────────────────────────────────┘

3. EVALUATE（评估）
   ┌─────────────────────────────────────────────┐
   │  用 NN 的 Value Head 评估该局面             │
   │  如果是终局，直接返回 ±1                    │
   └─────────────────────────────────────────────┘

4. BACKUP（回传）
   ┌─────────────────────────────────────────────┐
   │  将评估值沿 SELECT 路径向上回传，           │
   │  翻转视角符号（我优 → 敌劣）               │
   │  更新每个节点的 N (访问次数) 和 W (总价值)  │
   └─────────────────────────────────────────────┘
```

### 节点数据结构

```csharp
class MctsNode {
    MctsNode? Parent;          // 父节点
    float Prior;               // 先验概率 P(s,a) 来自 NN
    Piece?[,] Board;           // 棋盘状态
    Side Side;                 // 当前走棋方
    int VisitCount;            // 访问次数 N
    MctsNode?[] Children;      // 子节点（按 actionId 索引）
    float[] ChildP;            // 子节点先验概率
    int[] ChildN;              // 子节点访问次数
    float[] ChildW;            // 子节点总价值
    float CachedValue;         // 缓存的价值
}
```

### 蒙特卡洛树搜索参数

| 参数 | 值 | 说明 |
|------|------|------|
| `CPuct` | 1.5 | 探索常数（越大越鼓励探索） |
| `DirichletAlpha` | 0.3 | Dirichlet 噪声参数 |
| `DirichletEpsilon` | 0.25 | 根节点噪声混合比例 |
| `simulations` | 400-800 | 每次搜索的模拟次数 |
| `temperature` | 0.0 (对弈) / 1.0 (训练) | 走法选择温度 |

### MCTS 输出

```csharp
public sealed record MctsResult(
    Move? BestMove,                                  // 最佳走法
    IReadOnlyList<(Move, int Visits, float Q)> Stats, // 所有走法统计
    int SimulationsRun,                               // 实际模拟次数
    float BestQ);                                     // 最佳走法价值
```

### 性能数据

```
纯 NN Policy:    ~5ms/步   (只做一次推理，取最高概率)
MCTS 400 sims:   ~1.5s/步  (CPU 推理，每次推理 ~3-4ms)
MCTS 600 sims:   ~2.3s/步
```

### 代码位置

- `ChineseChess/ChineseChess.Core/Services/MctsAiService.cs`
- `ChineseChess/ChineseChess.Core/Services/XiangqiNeuralAiService.cs`

---

# 第三篇：实战篇 — 工程落地

---

## 14. C# ONNX Runtime 集成

### 依赖

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.19.2" />
```

### 核心类

```csharp
public sealed class XiangqiNeuralAiService : IDisposable
{
    private readonly InferenceSession _session;  // ONNX Runtime 会话
    private readonly XiangqiEngine _engine;
    private readonly bool _isCnnModel;
    private readonly bool _isFactoredModel;
    private readonly bool _isV5Model;
    private readonly int _inputPlanes;
    
    // 构造: 加载 ONNX 模型，自动检测架构类型
    public XiangqiNeuralAiService(string onnxModelPath, XiangqiEngine engine, bool isCnnModel)
    
    // 纯策略推理（无搜索）：选择合法走法中 policy logit 最高的
    public Move? ChooseMoveByPolicy(Piece?[,] board, Side side, IReadOnlyList<Move> history)
    
    // 完整推理：返回 policy_logits[1,8100] + value_pred[1]
    public (DenseTensor<float> PolicyLogits, DenseTensor<float> ValuePred) RunInference(Piece?[,] board, Side side)
    
    // 局面评估：返回 [-1, 1]
    public float EvaluatePosition(Piece?[,] board, Side side)
    
    // 策略分布：所有合法走法的 softmax 概率
    public IReadOnlyList<(Move Move, float Probability)> GetPolicyDistribution(Piece?[,] board, Side side, IReadOnlyList<Move> history)
}
```

### 模型自动检测

构造函数自动检测模型类型：

```
检查输出名称:
  "from_logits" 存在 → Factored 模型
  否则 → Flat 模型

检查输入形状:
  第 2 维 >= 16 → V5 模型（16 平面）
  否则 → Legacy 模型（14 平面）
```

### 编码方式

```csharp
// BoardEncoder.Encode() 输出 float[1260], layout: square-major
// 转换为 plane-major [1, 16, 10, 9]
DenseTensor<float> EncodeCnn(Piece?[,] board, Side side)
```

### 代码位置

- `ChineseChess/ChineseChess.Core/Services/XiangqiNeuralAiService.cs`

---

## 15. WPF 界面集成

### 架构

```
Bootstrapper (Caliburn.Micro DI 容器)
├── XiangqiEngine          (单例 - 象棋引擎)
├── XiangqiAiService       (单例 - 经典 AI)
├── SoundService           (单例 - 音效)
├── ShellViewModel         (主 ViewModel)
│   ├── XiangqiNeuralAiService  (神经网络 AI，懒加载)
│   ├── MctsAiService           (MCTS，懒加载)
│   ├── BoardViewModel          (棋盘)
│   └── SidePanelViewModel      (侧面板)
└── Views → XAML 界面
```

### AI 模式选择

```csharp
public enum AiEngineMode { Classic, Neural, NeuralMcts }
```

| 模式 | AI 引擎 | 特点 |
|------|---------|------|
| Classic | XiangqiAiService | 8 层 Alpha-Beta + QSearch |
| Neural | XiangqiNeuralAiService | 纯 NN Policy（~5ms） |
| NeuralMcts | MctsAiService | MCTS + NN（~2s/步） |

### 消息驱动

使用 Caliburn.Micro 的 EventAggregator 实现 UI 和逻辑解耦：

```
SquareSelectedMessage  → 选棋/走棋
SideChangedMessage      → 切换执红/黑
AiLevelChangedMessage   → AI 难度调整
AiEngineModeChangedMessage → AI 模式切换
RestartRequestedMessage → 重新开局
UndoRequestedMessage    → 悔棋
```

### 代码位置

- `ChineseChess/ChineseChess/ViewModels/ShellViewModel.cs`
- `ChineseChess/ChineseChess/ViewModels/SidePanelViewModel.cs`
- `ChineseChess/ChineseChess/Messages/GameMessages.cs`

---

## 16. AI 对弈测试

### AiMatchRunner

专门测试经典 AI vs ONNX 模型的命令行工具：

```bash
cd ChineseChess/AiMatchRunner
dotnet run
```

### 测试配置

```csharp
var gamesToPlay = 20;          // 20 局
var hardAiLevel = 4;           // 经典 AI 最大难度
var hardAiTimeMs = 5000;       // 每步 5 秒
var useMcts = true;            // ONNX 使用 MCTS
var mctsSimulations = 600;     // MCTS 模拟次数
```

### 测试结果示例

```
第  1 局 — ONNX(红) vs Classic(黑) ... 黑方将杀 | 8步 | 30s
第  2 局 — ONNX(黑) vs Classic(红) ... 红方将杀 | 15步 | 55s
...

  总局数:     20
  ONNX 胜:     0  (  0.0%)
  Classic 胜: 20  (100.0%)
  和棋:        0  (  0.0%)
```

### 代码位置

- `ChineseChess/AiMatchRunner/Program.cs`

---

## 17. 并行自对弈加速

### 问题

单进程自对弈 2000 局需要约 21 小时。

### 方案

5 个进程并行，每个进程 400 局，不同随机种子：

```bash
for i in 0 1 2 3 4; do
  dotnet exec ChineseChess.SelfPlay.dll \
    --games 400 --out "parallel_$i.jsonl" \
    --level 4 --timeMs 300 --seed $((12345 + i * 10000)) &
done
```

### 数据合并

```bash
cat parallel_0.jsonl parallel_1.jsonl ... > train_parallel_2000.jsonl
cat train_combined.jsonl train_parallel_2000.jsonl > train_all.jsonl
```

### 注意事项

- CPU 4 核 8 线程，5 进程争抢 CPU，加速比约 1.5x（不是 5x）
- 每个进程写独立文件，无文件冲突
- 使用 atomicOutput（先写 temp 文件，完成后再 rename）

---

## 18. 内存优化与生产部署

### 训练内存优化

| 问题 | 方案 | 效果 |
|------|------|------|
| 全量加载 262K 样本 → 14GB | 分块加载 (chunk=5000) | **670MB** (↓95%) |
| Python 对象开销大 | 预处理为 PyTorch Tensor 文件 | 每 chunk ~30MB |
| 数据增强重复计算 | `__getitem__` 中实时做随机翻转 | 无额外存储 |

### ChunkedDataset 实现

```python
class ChunkedDataset(Dataset):
    def __init__(self, chunk_dir):
        self.chunks = sorted(glob(f"{chunk_dir}/chunk_*.pt"))
        self.current_idx = -1
        self._features = None  # 懒加载
    
    def load_chunk(self, idx):
        # 加载新 chunk，释放旧 chunk
        data = torch.load(self.chunks[idx])
        self._features = data['features']
        ...
    
    def __getitem__(self, idx):
        # 实时做数据增强
        if random.random() < 0.5:
            features = features.flip(-1)  # 水平翻转
            # ... 重新映射走法
```

### 部署 checklist

```
□  training_v7_cont2.log 检查训练指标
□  artifacts/cnn_policy_value.onnx 确认导出成功
□  复制 ONNX 到 WPF 输出目录
□  启动 WPF → 选择 NeuralMcts 模式
□  AiMatchRunner 测试棋力
□  git commit + push
```

---

## 快速开始

### 环境要求

```
- .NET 8 SDK
- Python 3.10+ (推荐 3.14)
- PyTorch (CPU 版即可)
- Microsoft.ML.OnnxRuntime (NuGet)
```

### 运行经典 AI 对弈

```bash
cd ChineseChess
dotnet run --project ChineseChess
```

### 生成自对弈数据

```bash
dotnet run --project ChineseChess.SelfPlay -- \
  --games 100 --out data/selfplay/games.jsonl --level 3 --timeMs 120
```

### 训练神经网络

```bash
python train_v7_combined.py \
  --input data/selfplay/train_combined.jsonl \
  --checkpoint artifacts/cnn_policy_value_v7_cont.best.pt \
  --output artifacts/cnn_policy_value_v7_cont2.pt \
  --epochs 50 --batch-size 256 --lr 5e-5
```

### 导出并测试

```bash
python export_cnn_onnx.py \
  --input artifacts/cnn_policy_value_v7_cont2.best.pt \
  --output artifacts/cnn_policy_value.onnx

# 运行 AI 对弈测试
dotnet run --project ChineseChess/AiMatchRunner
```

---

## 学习路径推荐

如果你是新人，建议按这个顺序学习：

1. **先跑 Demo**：`Learning/demo1~demo3` 理解 QSearch + TT + Minimax
2. **理解棋盘编码**：`Learning/demo4_neural_encoding/README.md`
3. **理解自对弈数据**：`Learning/demo5_selfplay_dataset/README.md`
4. **跑教程 Demo 4-8**：从 MNIST 到文本分类，建立 PyTorch 基础
5. **阅读本项目代码**：从 XiangqiEngine → XiangqiAiService → 训练脚本
6. **实验改进**：调参 → 重新训练 → 测试 → 迭代

---

> **文档版本**: v2.0 | **最后更新**: 2026-06-18
>
> 项目地址: `E:\Projects\WorkBuddy\Chess\ChineseChess-WPF`
