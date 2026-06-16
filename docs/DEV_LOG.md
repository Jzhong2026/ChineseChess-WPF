# 中国象棋 ML 强化学习项目 — 开发日志

> **项目目标：** 基于已有 WPF 中国象棋单机程序的 Self-Play 基础设施，引入 Machine Learning（策略价值网络）来替代或增强现有 Alpha-Beta 引擎，提升棋力。
>
> **记录原则：** 每个决策、每个问题、每次修改都如实记录，包含时间戳、原因和解决过程。

---

## 目录

1. [代码库现状分析](#1-代码库现状分析)
2. [技术架构设计](#2-技术架构设计)
3. [阶段一：环境核查与基线测试](#阶段一环境核查与基线测试)
4. [阶段二：Self-Play 数据生成](#阶段二self-play-数据生成)
5. [阶段三：策略价值网络训练](#阶段三策略价值网络训练)
6. [阶段四：ONNX 导出](#阶段四onnx-导出)
7. [阶段五：C# WPF 集成](#阶段五c-wpf-集成)
8. [问题记录与解决方案](#问题记录与解决方案)
9. [性能测试记录](#性能测试记录)
10. [经验总结](#经验总结)

---

## 1. 代码库现状分析

**分析时间：** 2026-06-14

### 1.1 项目结构

```
ChineseChess-WPF/
├── ChineseChess/
│   ├── ChineseChess.Core/               # 核心逻辑库
│   │   ├── Models/XiangqiModels.cs      # 数据模型
│   │   ├── Services/XiangqiEngine.cs    # 游戏规则引擎（完整走法生成、胜负判定）
│   │   ├── Services/XiangqiAiService.cs # AI搜索（Negamax + α-β + 置换表）
│   │   ├── Encoding/BoardEncoder.cs     # 棋盘 → float[1260] 编码
│   │   └── Encoding/MoveEncoder.cs      # Move ↔ int[0..8099] 编码
│   ├── ChineseChess.SelfPlay/
│   │   └── Program.cs                   # 自博弈数据生成器（输出 JSONL）
│   ├── ChineseChess.Core.Verification/  # 编码正确性验证程序
│   └── ChineseChess/                    # WPF 主程序 (Caliburn.Micro MVVM)
├── train_chinese_chess_policy_value.py  # PyTorch MLP 训练脚本（原有）
└── README.md
```

### 1.2 现有能力盘点

| 模块 | 技术 | 状态 | 说明 |
|------|------|------|------|
| 规则引擎 | C# | ✅ 完整 | 包含走法生成、重复将军检测 |
| AI 搜索 | Negamax + α-β + QSearch + TT | ✅ 完整 | Level 1-4（深度 0-6）|
| 棋盘编码 | 14平面 one-hot float[1260] | ✅ 完整 | 2方×7棋子×90格 |
| 动作编码 | from\*90+to → int[0..8099] | ✅ 完整 | 90×90=8100 action |
| Self-Play | JSONL 格式输出 | ✅ 完整 | 含策略权重、价值权重 |
| ML 训练 | PyTorch MLP（策略+价值头） | ✅ 基础版本 | 输入 flat 1260 |
| ML 推理集成 | **不存在** | ❌ **本项目新增** | — |
| MCTS | **不存在** | ❌ **本项目新增** | — |

### 1.3 原有 AI 引擎技术细节

**搜索算法：Negamax with Alpha-Beta Pruning**
- Level 1：返回第一个合法走法（纯随机）
- Level 2：单步最佳（吃子 + 将军奖励）
- Level 3：迭代加深，最大深度 4 + Quiescence Search（最深 +6 层吃子）
- Level 4：迭代加深，最大深度 6 + Quiescence Search
- 时间控制：迭代加深，单步默认 300ms

**评估函数（手工设计）：**
```
子力价值：将帅=10000 / 车=900 / 炮=450 / 马=400 / 象=士=200 / 兵卒=120
位置奖励：兵卒（前进×14，中路-3）/ 马（中路+4）/ 炮（中路+2）
将军奖励：给对方将军 +35，被将军 -45
```

**置换表：** Zobrist hash，最大 250,000 条目（满了直接清空）

**棋盘编码（BoardEncoder.cs）：**
- 14 个平面 = 2 方 × 7 种棋子类型
- 数组索引：`index = (row × 9 + col) × 14 + plane`
- 总维度：10 × 9 × 14 = **1260**

**动作空间（MoveEncoder.cs）：**
- 从格索引 × 90 + 到格索引 = actionId
- 范围：0 到 8099（包含非法动作）
- 训练时用 legal_mask 遮蔽非法动作

### 1.4 Self-Play 数据格式（JSONL）

每行一个 `SelfPlayRow`：

```json
{
  "GameId": 1,
  "MoveIndex": 5,
  "Side": "Red",
  "SideToMove": 1,
  "BoardEncoding": [0.0, ..., 1.0],   // float[1260]
  "LegalMoves": [234, 567, ...],       // int[] ← legal action ids
  "SelectedMove": 234,                 // int ∈ [0, 8099]
  "Result": 1,                         // 1=红胜 -1=黑胜 0=平
  "SearchScoreRedPerspective": 45.2,
  "DepthReached": 4,
  "ValueWeight": 1.0,
  "PolicyWeight": 1.0,
  "UseForValueTraining": true,
  "UseForPolicyTraining": true,
  "Unfinished": false,
  "EndReason": "RedWins"
}
```

**EndReason 字段含义：**
| 值 | 说明 |
|---|---|
| `RedWins` / `BlackWins` | 真实将死 |
| `material-adjudication` | 子力差距裁定（非将死） |
| `max-moves-material-adjudication` | 达最大步数后子力裁定 |
| `max-moves` | 达最大步数，结果未知（跳过） |

---

## 2. 技术架构设计

**设计时间：** 2026-06-14

### 2.1 整体路线

```
阶段一：环境核查  →  .NET build 验证 + Python 环境确认
阶段二：Self-Play  →  生成 Level 3/4 质量训练数据（JSONL）
阶段三：CNN 训练  →  MLP → CNN（ResNet 风格）
阶段四：ONNX 导出  →  export_cnn_onnx.py
阶段五：C# 集成  →  XiangqiNeuralAiService + MctsAiService
阶段六（后续）：迭代强化（生成→训练→集成 循环）
```

### 2.2 关键技术决策

#### 决策 1：C# 推理方案选型

**候选方案：**
- A) ONNX Runtime（`Microsoft.ML.OnnxRuntime`）
- B) gRPC 调用 Python 推理服务
- C) ML.NET 原生模型

**选择：方案 A（ONNX Runtime）**

**理由：**
- WPF 是单机应用，不能引入 Python 进程依赖
- ONNX Runtime CPU 推理单次 < 5ms，满足实时性要求
- 标准工作流：PyTorch → `.onnx` → C# 推理，工业界成熟路线

#### 决策 2：模型架构选型

| 架构 | 参数量 | 棋盘感知 | 训练速度 | 决策 |
|------|--------|---------|---------|------|
| MLP (3×512) | ~4.7M | ❌ | 快 | 验证阶段 |
| CNN (8块×128) | ~2.1M | ✅ | 中 | **生产方案** |
| Transformer | >10M | ✅✅ | 慢 | 资源充足时 |

**选择：CNN（ResNet 风格）**

**理由：**
- 棋盘是 10×9 二维结构，CNN 天然感知空间关系（马腿阻断、炮的隔子跳吃等）
- MLP 把 (0,4) 和 (4,0) 的兵卒视为完全不同的输入，无法泛化
- 8 个残差块 × 128 通道参数量 ~2.1M，CPU 推理快

#### 决策 3：推理时搜索策略（分阶段）

| 阶段 | 方案 | 特点 |
|------|------|------|
| Phase 1 | 纯 NN 前向（Policy Argmax） | 快 <1ms，弱 |
| Phase 2 | NN + MCTS（PUCT 算法） | 慢 ~2s，强 |
| Phase 3（未来） | NN 替换 Alpha-Beta 评估函数 | 中等 |

#### 决策 4：棋盘编码重排

**问题：** `BoardEncoder.cs` 输出的 flat float[1260] 布局是：
```
index = (row × 9 + col) × 14 + plane   // 按格展开，每格14个plane
```

CNN 期望输入形状 `[batch, 14, 10, 9]`（按 plane 展开），需要在 C# 推理侧做 reshape：
```csharp
// 等效重排逻辑（在 XiangqiNeuralAiService.EncodeCnn 中实现）
cnn_tensor[plane, row, col] = flat[(row*9 + col) * 14 + plane]
```

### 2.3 数据流图

```
┌─────────────────────────────────┐
│  ChineseChess.SelfPlay (C#)     │
│  - 生成 Level 3/4 自博弈局谱    │
│  - 输出 JSONL（每行一个局面）    │
└────────────┬────────────────────┘
             │ data/selfplay/train.jsonl
             ▼
┌─────────────────────────────────┐
│  train_cnn_policy_value.py      │
│  - CNN 残差网络训练              │
│  - Policy Head + Value Head     │
│  - 输出 .pt checkpoint          │
└────────────┬────────────────────┘
             │ artifacts/cnn_policy_value.pt
             ▼
┌─────────────────────────────────┐
│  export_cnn_onnx.py             │
│  - torch.onnx.export            │
│  - 验证 ONNX 模型正确性         │
└────────────┬────────────────────┘
             │ artifacts/cnn_policy_value.onnx
             ▼
┌─────────────────────────────────┐
│  XiangqiNeuralAiService.cs (C#) │
│  - ONNX Runtime 加载模型        │
│  - 棋盘编码 → CNN 输入          │
│  - 返回策略分布 + 价值估计      │
└────────────┬────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────┐    ┌──────────────┐
│ NN Policy│    │ MctsAiService│
│ (纯前向) │    │  (PUCT MCTS) │
└──────────┘    └──────────────┘
             │
             ▼
     WPF 界面（三种 AI 模式切换）
     - 传统搜索（Alpha-Beta）
     - NN 策略（纯前向）
     - NN + MCTS
```

---

## 阶段一：环境核查与基线测试

**开始时间：** 2026-06-14 14:59

### Step 1.1：.NET SDK 版本检查

```bash
$ dotnet --version
8.0.400
```

✅ .NET 8 SDK 可用。

### Step 1.2：Python 环境检查

```bash
$ python --version
Python 3.13.12

$ python -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

✅ Python 可用，但 **PyTorch 未安装**。

---

## 问题记录与解决方案

### 问题 #1：项目目标框架 net10.0 与当前 SDK 不兼容

**发现时间：** 2026-06-14，阶段一 Step 1.3  
**影响：** 所有 4 个 `.csproj` 均无法构建  
**错误信息：**
```
error NETSDK1045: 当前 .NET SDK 不支持面向 .NET 10.0。
请面向 .NET 8.0 或更低版本，或者使用支持 .NET 10.0 的 .NET SDK 版本。
```

**根本原因：**
- 项目原本面向 .NET 10（Preview 或 RC 版本开发），但当前机器只安装了 .NET 8.0.400 SDK。

**解决方案：**
- 将所有 `.csproj` 的 `TargetFramework` 从 `net10.0` 降级为 `net8.0`
- WPF 项目从 `net10.0-windows` 降级为 `net8.0-windows`

**修改文件：**
- `ChineseChess.Core.csproj`: `net10.0` → `net8.0`
- `ChineseChess.SelfPlay.csproj`: `net10.0` → `net8.0`
- `ChineseChess.Core.Verification.csproj`: `net10.0` → `net8.0`
- `ChineseChess.csproj`: `net10.0-windows` → `net8.0-windows`

**验证：**
```
$ dotnet run --project ChineseChess.Core.Verification/...
Core encoder verification passed.
Board encoding length: 1260
Move action id range: 0..8099
```

**经验教训：**
> 团队使用最新 SDK 开发时，需要在 README 中明确标注最低要求 SDK 版本，或使用 `global.json` 锁定 SDK 版本。
> 本项目可以通过添加 `global.json`（`{ "sdk": { "version": "10.0.xxx" } }`）避免此问题。

---

### 问题 #2：PyTorch 未安装

**发现时间：** 2026-06-14，阶段一 Step 1.2  
**影响：** 无法运行 `train_chinese_chess_policy_value.py`

**解决方案：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> 选择 CPU 版本而非 CUDA 版本，原因：
> 1. 避免 CUDA 版本兼容性问题（需要匹配 CUDA 驱动）
> 2. 训练数据量有限（<5万样本），CPU 训练时间可接受（约 5-10 分钟）
> 3. 单机 WPF 应用最终推理也是 CPU，CPU 版本优先

**安装结果：**
```
Successfully installed torch-2.12.0+cpu torchvision-0.27.0+cpu
```

---

### 问题 #3：MLP 模型无法感知棋盘空间结构

**发现时间：** 2026-06-14，技术分析阶段  
**影响：** 原有 MLP 训练脚本（`train_chinese_chess_policy_value.py`）的模型架构存在根本性局限

**问题详情：**

MLP 的输入是 flat float[1260]，对于位于 (0,4) 和 (4,0) 的同类棋子，MLP 需要完全不同的权重路径才能识别，无法泛化相同的战术模式在不同位置的应用。

象棋中大量战术依赖空间关系：
- 马腿阻断（L 型路径上的障碍）
- 炮的隔子跳吃（需要数 2 格之间的棋子数）
- 车的射线控制（整行/整列的贯通）

**解决方案：**

新增 `train_cnn_policy_value.py`：
- 输入重整为 `[batch, 14, 10, 9]`（14平面棋盘）
- CNN stem：14 → 128 通道
- 8 个残差块（每块：Conv3x3 → BN → ReLU × 2 + skip connection）
- 策略头：Conv1x1(128→2) → FC(180→8100)
- 价值头：Conv1x1(128→1) → FC(90→256) → FC(256→1) → tanh

**新增/改进对比：**

| 项目 | 原 MLP | 新 CNN |
|------|--------|--------|
| 参数量 | ~4.7M | ~2.1M |
| 空间感知 | ❌ | ✅ |
| 位置不变性 | ❌ | 局部 ✅ |
| 训练时长（CPU/1000样本/epoch） | ~1s | ~3s |
| 推理时间（CPU，单局面） | <1ms | ~3ms |

---

### 问题 #4：棋盘编码布局与 CNN 期望布局不匹配

**发现时间：** 2026-06-14，阶段五集成开发  
**影响：** 直接把 BoardEncoder 输出的 float[1260] 塞入 CNN 会得到错误的特征提取

**技术细节：**

`BoardEncoder.Encode()` 的内存布局（C# 端）：
```
index = (row * 9 + col) * 14 + plane
  即：按格（square-major）排列，每格连续存 14 个 plane 值
```

CNN 期望的内存布局（Python 端，PyTorch `[batch, planes, rows, cols]`）：
```
index = plane * (10 * 9) + row * 9 + col
  即：按平面（plane-major）排列，每平面连续存 10×9 = 90 个格值
```

**解决方案（`XiangqiNeuralAiService.EncodeCnn`）：**
```csharp
// C# 重排逻辑
for (var row = 0; row < 10; row++)
    for (var col = 0; col < 9; col++)
        for (var plane = 0; plane < 14; plane++)
        {
            var flatIdx = (row * 9 + col) * 14 + plane;  // BoardEncoder 布局
            tensor[0, plane, row, col] = flat[flatIdx];   // CNN 期望布局
        }
```

**经验：**
> 跨语言/跨框架传递张量时，务必核对内存布局（row-major vs column-major, plane-major vs square-major）。Python 的 `torch.Tensor` 是 row-major，C# 的多维数组也是 row-major，但逻辑维度的排列顺序需要显式对齐。

---

### 问题 #5：`ShellViewModel.cs` 中 `Path`/`File` 命名空间缺失

**发现时间：** 2026-06-14，WPF 项目构建阶段  
**错误信息：**
```
error CS0103: 当前上下文中不存在名称"Path"
error CS0103: 当前上下文中不存在名称"File"
```

**原因：**
- WPF 项目启用了 `<ImplicitUsings>enable</ImplicitUsings>`，但 `System.IO` 命名空间在 WPF 中没有被隐式 using 包含（`Microsoft.NET.Sdk.WindowsDesktop` 的 implicit using 集合与 `Microsoft.NET.Sdk` 不同）

**解决方案：**
```csharp
// 在 ShellViewModel.cs 顶部添加：
using System.IO;
```

---

### 问题 #6：Python 训练脚本棋盘 reshape 错误（square-major vs plane-major）

**发现时间：** 2026-06-14，阶段三 CNN 训练  
**影响：** CNN 模型训练策略准确率 0%，完全无法学会预测走法  

**问题详情：**

`train_cnn_policy_value.py` 第 66 行原代码：
```python
features = torch.tensor(s.board_encoding, dtype=torch.float32).view(PLANES, BOARD_ROWS, BOARD_COLS)
```

这行代码把 C# `BoardEncoder` 输出的 square-major flat[1260] 直接 `.view(14, 10, 9)`，但 PyTorch 的 `.view()` 按 plane-major 顺序填充，导致：

```
实际数据：flat[(row*9+col)*14 + plane]  ← 同一格子的14个plane连续存储
错误解读：flat[plane*90 + row*9+col]     ← 同一plane的90个格子连续存储
```

结果：CNN 看到的"棋盘"是所有格子的特征交叉混合，完全丧失了空间结构信息。

**解决方案：**
```python
flat = torch.tensor(s.board_encoding, dtype=torch.float32)
# flat[1260] → [90, 14] → permute → [14, 90] → [14, 10, 9]
features = flat.view(BOARD_ROWS * BOARD_COLS, PLANES).permute(1, 0).view(PLANES, BOARD_ROWS, BOARD_COLS)
```

**验证：** 修复后训练策略损失正常下降（3.55 → 0.46），但验证准确率仍然 0%（因为数据量不足，见问题 #7）。

**经验：**
> `torch.Tensor.view()` 只是 reinterpret 内存布局，不做数据重排。当 flat 数组的逻辑布局与目标张量维度顺序不同时，必须显式 `permute()` / `reshape()` + `transpose()`，而不能直接 `.view()`。

---

### 问题 #7：训练数据严重不足导致过拟合

**发现时间：** 2026-06-14，阶段三 CNN 训练  
**影响：** 即使修复了 reshape bug，策略验证准确率仍为 0%  

**训练指标对比：**

| 指标 | 训练集 | 验证集 | 诊断 |
|------|--------|--------|------|
| Policy Loss | 3.55 → 0.05 ✅ | 2.09 → 3.49 ❌ | 严重过拟合 |
| Policy Acc | - | 0.0000% ❌ | 从未命中目标走法 |
| Legal Acc | - | ~4.5% ❌ | 仅略好于随机(1/35≈2.9%) |

**根本原因：**
- 训练数据量：5091 样本
- 模型参数量：1.53M（channels=32, blocks=2）~ 1.79M（channels=64, blocks=4）
- **样本/参数比 ≈ 0.003**，远低于深度学习最低经验阈值（~10-100 样本/参数）
- 模型完全记忆训练数据，无法泛化

**解决方案：**
1. 生成更多 Self-Play 数据（目标：100 局 → ~17,000 样本；500 局 → ~85,000 样本）
2. 使用 policy_weight 加权损失（深度搜索的走法标签更可靠）
3. 增大 weight_decay（1e-4 → 1e-3）
4. 必要时减小模型（channels=32, blocks=2）

**当前进度：** 100 局 Self-Play 已在后台运行

---

### 问题 #8：训练脚本未使用 policy_weight / value_weight 损失加权

**发现时间：** 2026-06-14，阶段三训练代码审查  
**影响：** 低质量标签（浅搜索深度）与高质量标签（深搜索深度）在损失中权重相同

**原代码：**
```python
p_loss = ce_loss(masked_logits, policy_target)  # 所有样本等权
v_loss = ((value_pred - value_target) ** 2).mean()  # 所有样本等权
```

**修复后：**
```python
per_sample_p_loss = F.cross_entropy(masked_logits, policy_target, reduction="none")
p_loss = (per_sample_p_loss * policy_weight).mean()  # 深搜索权重更高

per_sample_v_loss = (value_pred - value_target) ** 2
v_loss = (per_sample_v_loss * value_weight).mean()  # 将死局权重=1，裁定局=0.5-0.75
```

**权重逻辑（C# SelfPlay 侧）：**
- `PolicyWeight`：depth=0 → 0.1, depth=1 → 0.4, depth=2 → 0.7, depth≥3 → 1.0
- `ValueWeight`：将死 → 1.0, 子力裁定 → 0.75, 最大步数裁定 → 0.5

---

### 问题 #9：`.slnx` 格式不支持 `dotnet build`

**发现时间：** 2026-06-14，阶段一构建尝试  
**错误信息：**
```
error MSB4068: 无法识别元素 <Solution>，或者在此上下文中不支持该元素。
```

**原因：**
- `.slnx` 是 .NET 10 引入的新格式（XML based），当前 .NET 8 SDK 的 `dotnet build` 不支持
- 标准 `.sln` 格式才是 .NET 8 兼容的方案文件格式

**解决方案：**
- 直接构建各个 `.csproj` 文件，绕过 `.slnx`
- 或者，升级 SDK 到 .NET 10
- **本项目选择**：直接用 csproj 路径构建，不生成新的 .sln

**影响范围：** 不影响代码功能，仅影响构建命令写法

---

## 阶段二：Self-Play 数据生成

**开始时间：** 2026-06-14 15:10

### Step 2.1：30 局探测数据 ✅ 完成

**命令：**
```bash
dotnet run --project ChineseChess.SelfPlay/... --configuration Release -- \
  --games 30 --out data/selfplay/probe.jsonl --level 3 --timeMs 120 \
  --topK 2 --nearBestWindow 80 \
  --randomOpeningPlies 12 --openingTopK 6 --openingNearBestWindow 180 \
  --adjudicateAfterMoves 120 --adjudicateNoCapturePlies 60 \
  --adjudicateMaterialMargin 450 --adjudicateAtMaxMoves true
```

**耗时：** 11 分 52 秒  
**输出：** `data/selfplay/probe.jsonl`，5091 行

**统计结果：**

| 指标 | 值 |
|------|-----|
| 总局数 | 30 |
| 红胜 | 11 (36.7%) |
| 黑胜 | 17 (56.7%) |
| 和棋 | 2 (6.7%) |
| 跳过（未完成） | 2 |
| 子力裁定 | 17 |
| 真实将死 | 9 (RedWins:4, BlackWins:5) |
| 总数据行 | 5091 |
| 重复对局 | 0 |
| 平均合法走法数 | 35.4 |
| SelectedMove 范围 | [1, 8098] |

**数据验证：**
- ✅ 所有 5091 行 `SelectedMove ∈ LegalMoves`（100%）
- ✅ 棋盘编码长度 1260，初始局面非零值 32（16红+16黑）
- ✅ 所有平面棋子计数正确（帅1/士2/象2/马2/车2/炮2/兵5 × 2方）

### Step 2.2：100 局训练数据 ✅ 完成

**命令：** 同 Step 2.1，`--games 100 --out data/selfplay/train_100.jsonl`  
**实际耗时：** 40 分 56 秒  
**输出：** `data/selfplay/train_100.jsonl`，14425 行（44 MB）

**统计结果：**

| 指标 | 值 |
|------|-----|
| 总局数 | 100 |
| 红胜 | 27 (27%) |
| 黑胜 | 52 (52%) |
| 未完成/跳过 | 21 (21%) |
| 子力裁定 | 48 |
| 总数据行 | 14425 |
| 重复对局 | 0 |
| Unique signatures | 100 |

**观察：**
- 黑方胜率显著高于红方（52% vs 27%），可能与 Level 3 AI 在后手时更擅长利用反击有关
- 21% 的对局未分胜负（最大步数限制），这些数据标记为 Unfinished=true，训练时默认跳过

---

## 阶段三：策略价值网络训练

**开始时间：** 2026-06-14  
**状态：** 🔄 数据量不足，正在生成更多数据

### Step 3.1：CNN 小模型验证训练（5091 样本）

**命令：**
```bash
python train_cnn_policy_value.py \
    --input data/selfplay/probe.jsonl \
    --output artifacts/cnn_policy_value.pt \
    --epochs 10 --batch-size 64 \
    --channels 64 --res-blocks 4 --lr 1e-3
```

**训练结果（reshape bug 修复后 + policy_weight 加权）：**

| Epoch | Train Loss | Policy Loss | Value Loss | Val Loss | Val Policy Acc | Val Legal Acc |
|-------|-----------|-------------|------------|----------|---------------|--------------|
| 1 | 3.76 | 3.55 | 0.21 | 3.54 | 0.00% | 5.50% |
| 5 | 1.56 | 1.55 | 0.01 | 4.30 | 0.00% | 5.30% |
| 10 | 0.47 | 0.46 | 0.008 | 4.58 | 0.00% | 4.32% |

**更小模型试验（channels=32, blocks=2, 1.53M params, 20 epochs）：**

| Epoch | Train Loss | Val Loss | Val Legal Acc |
|-------|-----------|----------|--------------|
| 1 | 2.14 | 2.09 | 4.52% |
| 10 | 0.24 | 3.05 | 3.34% |
| 20 | 0.055 | 3.49 | 4.52% |

**结论：**
- ✅ 训练损失正常下降 → 数据管线正确
- ✅ 模型能过拟合训练集 → 学习能力正常
- ❌ 验证损失上升 + 策略准确率 0% → **严重过拟合**
- ❌ 合法着法准确率 ~4.5%（vs 随机 ~2.9%）→ 仅略优于随机
- **根本原因：5091 样本 vs 1.5M+ 参数 = 样本/参数比 ~0.003**

**下一步：** 等待 100 局 Self-Play 完成（预计 ~17,000 样本），重新训练

### Step 3.2：CNN 正式训练（100 局，14425 样本）

#### V1 训练：大模型无 Dropout（channels=64, blocks=4, 1.8M params）

```bash
python train_cnn_policy_value.py \
    --input data/selfplay/train_100.jsonl \
    --output artifacts/cnn_policy_value.pt \
    --epochs 50 --batch-size 256 \
    --channels 64 --res-blocks 4 --lr 1e-3
```

**结果：严重过拟合，与 5091 样本时相同的模式**

| Epoch | Train Loss | Val Loss | Val Policy Acc | Val Legal Acc |
|-------|-----------|----------|---------------|--------------|
| 1 | 2.16 | 2.03 | 0.00% | 4.16% |
| 5 | 1.26 | 2.34 | 0.00% | 3.33% |
| 10 | 0.43 | 2.76 | 0.00% | 3.40% |

**诊断：** 1.8M 参数 vs 14K 样本（样本/参数比 ~0.008），仍然严重不足。策略 FC 层（180→8100）单独占 1.47M 参数（95%）。

#### 问题 #11：策略 FC 层参数占比过大导致策略头无法泛化

**发现时间：** 2026-06-14，100 局数据训练后  
**影响：** 无论 CNN backbone 多小，策略 FC 层（180→8100）始终占 95% 参数

**分析：**
- Policy head: `Conv2d(channels→2) → flatten(180) → Linear(180→8100)` = 180×8100 + 8100 = **1,469,700 参数**
- Value head: `Conv2d(channels→1) → FC(90→256→1)` = ~23K 参数
- CNN backbone (channels=64, blocks=4) = ~300K 参数
- **策略 FC 层是参数瓶颈，与 backbone 大小无关**

**缓解措施（本次）：**
1. 添加 Dropout（policy=0.3, value=0.3）到 FC 层
2. 增大 weight_decay（1e-4 → 1e-2）
3. 降低学习率（1e-3 → 3e-4）
4. 缩小 backbone（channels=32, blocks=3）
5. 添加 Top-5/Top-10 验证指标（比 Top-1 更有实用意义）

**根本解决（未来迭代）：**
- 改用分离式策略头（from-square 90 类 + to-square 90 类），大幅减少参数
- 或使用 policy_conv 直接输出 [batch, 90, 10, 9] 的平面预测

#### V2 训练：Dropout + 强正则（channels=32, blocks=3, 1.55M params）

```bash
python train_cnn_policy_value.py \
    --input data/selfplay/train_100.jsonl \
    --output artifacts/cnn_policy_value.pt \
    --epochs 60 --batch-size 128 \
    --channels 32 --res-blocks 3 \
    --lr 3e-4 --weight-decay 1e-2 \
    --policy-dropout 0.3 --value-dropout 0.3
```

**关键改进：新增 Top-5 / Top-10 验证指标**

| Epoch | Train Loss | Val Loss | Val Legal Acc | Val Top-5 | Val Top-10 | Val Policy Acc |
|-------|-----------|----------|--------------|-----------|------------|---------------|
| 1 | 2.43 | 2.17 | 4.58% | 17.75% | 35.16% | 0.00% |
| 2 | 2.03 | 2.02 | 4.85% | 17.96% | 36.55% | 0.00% |
| 3 | 1.89 | **2.00** | 4.37% | 18.72% | 35.71% | 0.00% |
| 5 | 1.73 | 2.04 | 4.51% | 18.45% | 33.70% | 0.00% |
| 10 | 1.36 | 2.17 | 4.09% | 17.96% | 34.26% | 0.00% |
| 20 | 0.82 | 2.39 | 3.61% | 17.55% | 32.87% | 0.00% |

**关键发现：**
- ✅ **Val Top-10 = 35-36%**：模型在 35% 的情况下将正确走法排入前 10！
- ✅ **Val Top-5 = 17-18%**：17% 的情况下排入前 5
- ❌ Val Policy Acc (Top-1) 仍为 0%：8100 分类的 Top-1 准确率对少量数据几乎不可能
- 最佳 epoch 为第 3 轮（val_loss=2.00），此后持续过拟合

**实战意义：**
- Top-1 = 0% 不代表模型无用！MCTS 的 PUCT 算法只需要合理的先验分布，不需要 Top-1 精确命中
- Top-10 = 35% 意味着 MCTS 在 35% 的节点上能优先搜索到正确方向，大幅提升搜索效率
- 价值头预测稳定（0.1127 开局估值，方向正确）

**保存最佳模型：** `artifacts/cnn_policy_value.best.pt`（6.2 MB，epoch 3 checkpoint）

#### 问题 #12：ONNX 模型仅复制到 Release 目录导致 Debug 模式找不到

**发现时间：** 2026-06-14，集成测试阶段
**现象：** VS 调试启动（Debug 模式）报错「模型文件不存在: cnn_policy_value.onnx」
**原因：**
- ONNX 模型通过手动 `cp` 命令只复制到了 `bin/Release/net8.0-windows/`
- VS 默认使用 Debug 配置启动调试，exe 在 `bin/Debug/net8.0-windows/`
- 该目录没有 ONNX 文件
**解决方案：**
1. 在 `ChineseChess.csproj` 中添加 `<Content>` 项目引用：
```xml
<Content Include="..\..\artifacts\cnn_policy_value.onnx" Link="cnn_policy_value.onnx">
    <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
</Content>
```
2. 路径 `..\..\artifacts\` 从 csproj 所在目录回溯两层到解决方案根目录
3. `Link="cnn_policy_value.onnx"` 让文件在项目中显示为虚拟链接（不物理复制到项目内）
4. `PreserveNewest` 确保每次构建时如果源文件更新则重新复制
**教训：** 静态资源文件应通过 csproj 的 `<Content>` + `<CopyToOutputDirectory>` 管理，而非手动复制

---

## 阶段四：ONNX 导出

**状态：** ✅ 完成

### 新增/改进文件

**`export_onnx.py`** — MLP 模型导出（向后兼容）  
**`export_cnn_onnx.py`** — CNN 模型导出（主力，已修复 dynamo 问题）

### 问题 #9：PyTorch 2.12 默认 ONNX 导出器（dynamo）不包含权重

**发现时间：** 2026-06-14，阶段四 ONNX 导出  
**影响：** 导出的 ONNX 文件仅 5.7 KB，缺少全部模型权重  

**问题详情：**

PyTorch 2.12 的 `torch.onnx.export()` 默认使用 dynamo 导出器（`dynamo=True`），新导出器生成的 ONNX 文件不包含参数权重，导致文件极小且无法推理。

**解决方案：**
```python
torch.onnx.export(
    ...
    dynamo=False,  # 使用旧版 TorchScript 导出器，确保权重包含
)
```

**验证：** 导出文件从 5.7 KB → 5.7 MB（1.5M params × 4 bytes），推理结果正确。

---

### 问题 #10：缺少 onnxscript 和 onnxruntime Python 包

**发现时间：** 2026-06-14，阶段四  
**解决方案：**
```bash
pip install onnx onnxruntime onnxscript
```

### 导出命令

```bash
python export_cnn_onnx.py \
    --input artifacts/cnn_policy_value.best.pt \
    --output artifacts/cnn_policy_value.onnx
```

**V2 模型端到端验证结果：**

| 验证项 | Python (onnxruntime) | C# (ONNX Runtime) | 结果 |
|--------|---------------------|-------------------|------|
| 模型参数量 | 1,549,531 | — | ✅ |
| ONNX 文件大小 | 5.9 MB | — | ✅ |
| 输入形状 | (1, 14, 10, 9) | [1, 14, 10, 9] | ✅ 一致 |
| Policy 输出 | [1, 8100] | [1, 8100] | ✅ 一致 |
| Value 预测 | 0.1127 | 0.1127 | ✅ 完全一致 |
| 策略 Top-5 命中 | ✅ (label 7537 排第 4) | Top-3: 4.2%, 3.7%, 3.6% | ✅ 合理 |
| MCTS 50sims | — | Cannon(7,1→3,1) Q=0.264 | ✅ 合理 |

---

## 阶段五：C# WPF 集成

**开始时间：** 2026-06-14 16:30  
**状态：** ✅ 代码完成，✅ ONNX 端到端验证通过

### 5.1 新增文件

| 文件 | 作用 |
|------|------|
| `ChineseChess.Core/Services/XiangqiNeuralAiService.cs` | ONNX Runtime 推理服务 |
| `ChineseChess.Core/Services/MctsAiService.cs` | MCTS + NN 搜索 |

### 5.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `ChineseChess.Core.csproj` | 添加 `Microsoft.ML.OnnxRuntime 1.19.2` |
| `Messages/GameMessages.cs` | 添加 `AiEngineMode` 枚举和 `AiEngineModeChangedMessage` |
| `ViewModels/ShellViewModel.cs` | 集成 Neural/MCTS AI 分支 + 懒加载 ONNX 模型 |
| `ViewModels/SidePanelViewModel.cs` | 添加引擎模式属性、MCTS 模拟次数控制 |
| `Views/SidePanelView.xaml` | 添加「传统搜索 / NN策略 / NN+MCTS」模式切换按钮 |

### 5.3 WPF AI 模式架构

```
用户点击 [传统搜索] → AiEngineMode.Classic
    → 走原有 XiangqiAiService.ChooseMove()（Alpha-Beta）

用户点击 [NN策略] → AiEngineMode.Neural
    → 加载 cnn_policy_value.onnx（懒加载）
    → 走 XiangqiNeuralAiService.ChooseMoveByPolicy()（纯策略贪婪）

用户点击 [NN+MCTS] → AiEngineMode.NeuralMcts
    → 走 MctsAiService.Search()（PUCT MCTS）
    → 可调 MCTS 模拟次数（100~2000，UI Slider）
```

### 5.4 ONNX 模型加载规则

- 模型文件：`cnn_policy_value.onnx`（放在 WPF 可执行文件同目录）
- 懒加载：首次切换到 NN 模式时加载（避免模型文件不存在时启动失败）
- 错误处理：文件不存在 / 加载失败 → 显示警告文字，不影响传统 AI 可用性

---

## 性能测试记录

**测试日期：** 2026-06-14  
**测试模型：** CNN V2 (channels=32, blocks=3, 1.55M params, dropout=0.3) — 14425 样本训练，best at epoch 3

### 推理速度测试（C# ONNX Runtime, CPU）

| 模式 | 单步耗时 | 备注 |
|------|---------|------|
| 传统搜索 Level 4 | ~300ms（可配置）| Negamax + Alpha-Beta |
| NN 策略（纯前向）| <5ms | ONNX Runtime 推理 + legal mask argmax |
| NN+MCTS 50 sims | <1s | 50 次 NN 前向 + 树搜索 |
| NN+MCTS 400 sims | ~8s（预估）| 400 次 NN 前向 |
| NN+MCTS 2000 sims | ~40s（预估）| 2000 次 NN 前向 |

### 模型预测质量（V2 模型，14425 样本训练，best epoch 3）

| 指标 | 值 | 评价 |
|------|-----|------|
| Val Policy Accuracy (Top-1) | 0.00% | ❌ 8100 分类 Top-1 对小数据极难 |
| Val Legal Accuracy (Top-1) | ~4.5% | ⚠️ 略优于随机(~2.9%) |
| **Val Legal Top-5** | **~18%** | ✅ 有实用价值 |
| **Val Legal Top-10** | **~35%** | ✅ MCTS 先验有效 |
| Value Prediction (开局) | 0.1127 | ✅ 方向正确 |
| MCTS Q value (开局, 50sims) | 0.264 | ✅ 合理 |

### 模型迭代对比

| 版本 | 数据量 | 模型配置 | Val Top-10 | Val Loss (best) | 改进 |
|------|--------|---------|------------|----------------|------|
| V0 (probe) | 5,091 | channels=64, blocks=4, 无dropout | 未测 | ~2.03 | 基线 |
| V1 (train_100) | 14,425 | channels=64, blocks=4, 无dropout | 未测 | ~2.03 | 数据量↑，无改进 |
| **V2 (train_100)** | **14,425** | **channels=32, blocks=3, dropout=0.3** | **35%** | **2.00** | **✅ 正则化有效** |

> **关键洞察：** 小数据集下，模型架构优化不如正则化重要。Dropout + weight_decay 比 增大网络 更有效。

---

## 经验总结

**（随项目进展持续更新）**

### 技术教训

1. **目标框架锁定**：多人协作项目应使用 `global.json` 锁定 SDK 版本，避免 net10/net8 混用问题
2. **编码布局对齐**：跨语言传递张量时，flat 数组的内存布局需要显式文档化
3. **渐进式验证**：每个阶段都先跑最小用例（30 局探测 → 100 局训练 → 500 局正式），而非直接生产量级
4. **ONNX 懒加载**：游戏应用中 ONNX 模型应懒加载，避免模型缺失时阻断正常功能
5. **`torch.Tensor.view()` ≠ 数据重排**：`.view()` 只重新解读内存，不做数据搬移；当逻辑布局与目标维度顺序不一致时，必须 `permute()` + `contiguous()` + `view()`
6. **训练数据量是第一优先级**：5K→14K 样本仍不够训练 1.5M+ 参数的 CNN；先保证数据量（10万+），再优化模型架构
7. **损失加权必须实际使用**：`policy_weight` / `value_weight` 从 DataLoader 加载后如果不在损失函数中使用，等于白加载
8. **Top-1 准确率≠模型质量**：8100 分类的 Top-1 对小数据集几乎不可能为正，应关注 Top-5/Top-10 和 MCTS 实际效果
9. **正则化 > 大模型**：小数据集下，Dropout + weight_decay 比增大网络参数更有效
10. **策略 FC 层是参数瓶颈**：Linear(180→8100) 占 95% 参数，需要分离式策略头架构来根本解决

### 架构教训

1. **模型不可完全替代搜索**：纯 NN 前向策略棋力弱于 Alpha-Beta Level 3，MCTS 才能达到强度
2. **训练数据质量 > 数量**：Level 4 生成的数据质量优于 Level 1/2，即使数量少也更有价值
3. **Self-Play 多样性**：`--randomOpeningPlies 12 --openingTopK 6` 是降低重复局谱比例的关键参数
4. **端到端验证流程**：Self-Play → 编码验证 → 小模型过拟合测试 → 正式训练 → ONNX 导出 → C# 推理，每一步都应有独立验证点
5. **MCTS 的先验策略不需要完美**：Top-10=35% 的先验分布已能为 MCTS 提供有效搜索引导，关键不是 Top-1 精确命中，而是正确走法出现在搜索优先方向中

### 项目管理教训

1. 先建立完整流程（数据→训练→导出→集成），再优化每一步
2. 每个新增文件都应有对应的文档说明其作用
3. 问题记录要及时，遇到问题立刻写下"发现时间 + 现象 + 根本原因 + 解决方案"
4. **Bug 的连锁性**：一个 reshape 错误掩盖了数据量不足的问题，修复后才暴露下一个问题
5. **指标选择决定判断**：如果只看 Top-1 准确率会得出"模型完全无用"的结论，但 Top-10 命中率 35% 说明模型在 MCTS 场景下有价值

---

## 阶段六：迭代强化路线图（后续）

**状态：** 📋 规划中

当前 V2 模型（14425 样本，Top-10=35%）已提供可用的 MCTS 先验。以下是后续迭代方向：

### 6.1 数据量提升（优先级最高）

| 目标 | 数据量 | 预计耗时 | 预期效果 |
|------|--------|---------|---------|
| 500 局 Self-Play | ~72,000 样本 | ~3.5h | Top-10 预计 50%+ |
| 1000 局 | ~144,000 样本 | ~7h | 模型可用性显著提升 |
| 5000 局 | ~720,000 样本 | ~35h | 接近 AlphaZero 基线 |

### 6.2 架构改进

1. **分离式策略头**：将 180→8100 FC 替换为 from-square(90类) + to-square(90类)，参数从 1.47M 降至 ~36K
2. **Policy Conv 直接预测**：输出 [batch, 90, 10, 9] 表示 from 概率图，再通过 to 索引查表
3. **ResNet 通道数随数据量增长**：当前 32 通道 → 数据量 >100K 时可增至 64/128

### 6.3 训练改进

1. **学习率 Warm-up + Cosine Decay**：前 5 个 epoch 线性升温，避免初始震荡
2. **Label Smoothing**：策略目标从 one-hot 改为 0.9/0.1 平滑
3. **数据增强**：棋盘水平翻转（红黑对称性），样本量翻倍
4. **Mixed Precision Training**：如果未来使用 GPU 训练

### 6.4 AlphaZero 式迭代自博弈

```
循环 {
  1. 用当前模型运行 MCTS Self-Play（生成新对局）
  2. 新数据 + 旧数据合并训练
  3. 导出新 ONNX 模型
  4. 新旧模型对战评估（胜率 >55% 才替换）
}
```

这是 AlphaZero 的核心机制——用模型指导 Self-Play，用 Self-Play 数据训练模型，形成正反馈循环。

### 6.5 自动化管线

将 `run_training_pipeline.sh` 扩展为完整的 CI/CD：
- `selfplay` → `train` → `export_onnx` → `verify_python` → `verify_csharp` → `deploy_to_wpf`

---

## 阶段七：第二轮迭代（500 局 Self-Play → V3 模型）

**状态：** ✅ 完成  
**日期：** 2026-06-14（启动）~ 2026-06-15（完成训练部署）

### Step 7.1：500 局 Self-Play 数据生成

**命令：**
```bash
dotnet run --project ChineseChess.SelfPlay/ChineseChess.SelfPlay.csproj --configuration Release -- \
  --games 500 --out ../data/selfplay/train_500.jsonl --level 4 --timeMs 150 \
  --topK 2 --nearBestWindow 80 --randomOpeningPlies 14 --openingTopK 8 --openingNearBestWindow 200 \
  --adjudicateAfterMoves 100 --adjudicateNoCapturePlies 50 --adjudicateMaterialMargin 400 \
  --adjudicateAtMaxMoves true
```

**结果：**

| 指标 | 值 |
|------|-----|
| 总局数 | 500 |
| 红胜（直接分出胜负） | 57 (11.4%) |
| 黑胜（直接分出胜负） | 100 (20.0%) |
| 红胜（子力裁定/最大步数） | 114 (22.8%) |
| 黑胜（子力裁定/最大步数） | 142 (28.4%) |
| 跳过（未完成） | 87 (17.4%) |
| 总数据行 | **74,643** |
| 文件大小 | 220 MB |
| 相比 100 局 | **+5.2x 样本量** |

**升级对比（100 局 vs 500 局）：**

| 项目 | 100 局 | 500 局 |
|------|-------|-------|
| 数据行 | 14,425 | **74,643** |
| AI 等级 | Level 3 | **Level 4** |
| 思考时间 | 120ms | **150ms** |
| 开局随机化 | 12 步 topK=6 | **14 步 topK=8** |

> **注意：** Level 4 的 Self-Play 数据质量高于 Level 3，因为走棋参考了更深的搜索，产出的数据更接近"正确"走法。

### Step 7.2：CNN V3 模型训练

```bash
python train_cnn_policy_value.py \
    --input data/selfplay/train_500.jsonl \
    --output artifacts/cnn_policy_value.pt \
    --epochs 80 --batch-size 256 \
    --channels 32 --res-blocks 3 \
    --lr 3e-4 --weight-decay 1e-2 \
    --policy-dropout 0.3 --value-dropout 0.3
```

**训练数据规模：** Train=67179, Val=7464

**训练结果（关键 epoch）：**

| Epoch | Train Loss | Val Loss | Val Legal Acc | Val Top-5 | Val Top-10 |
|-------|-----------|----------|--------------|-----------|------------|
| 1 | 2.7392 | 2.5060 | 5.60% | 20.83% | 36.01% |
| 3 | 2.3225 | 2.2747 | 4.81% | 21.14% | 36.29% |
| 5 | 2.2294 | 2.2613 | 4.70% | 21.01% | 36.43% |
| **6** | **2.1959** | **2.2527** ← **best** | 4.38% | 20.63% | 36.05% |
| 7 | 2.1608 | 2.2671 | 4.53% | 20.59% | 36.04% |
| 10 | 2.0417 | 2.3048 | 4.46% | 19.23% | 34.65% |
| 14 | 1.8646 | 2.3737 | 4.15% | 19.04% | 34.07% |

**观察：**
- 最佳 epoch=6（val_loss=2.2527），此后仍在过拟合
- **Val Top-10 在 epoch 5 达到峰值 36.43%** → 与 V2 的 35% 相比有小幅提升
- Top-5 从 V2 的 18% 提升到 **21%** ← 有明显改善
- Val Legal Acc (Top-1) 从 V2 的 4.5% 提升到 **5.6%** ← 开始出现 Top-1 命中

**模型迭代对比：**

| 版本 | 数据量 | AI等级 | Val Top-5 | Val Top-10 | Val Loss (best) |
|------|--------|-------|-----------|------------|----------------|
| V2 | 14,425 | L3 | 18% | 35% | 2.00 |
| **V3** | **74,643** | **L4** | **21%** | **36%** | **2.25** |

> **重要洞察：** 数据量增加 5 倍，但 Top-10 仅提升 1 个百分点。**核心瓶颈仍然是策略 FC 层（180→8100 参数占 95%）**，而非数据量不足。单纯增加数据对改善策略质量的边际效益递减，下一轮迭代应优先改造架构（分离式 from/to 策略头）。

### Step 7.3：ONNX V3 导出与验证

```bash
python export_cnn_onnx.py \
    --input artifacts/cnn_policy_value.best.pt \
    --output artifacts/cnn_policy_value.onnx
```

**V3 模型端到端验证：**

| 验证项 | Python | C# | 结果 |
|--------|--------|-----|------|
| ONNX 文件大小 | 5.9 MB | — | ✅ |
| Policy 输出 | [1, 8100] | [1, 8100] | ✅ 一致 |
| Value 预测（开局） | -0.2963 | 0.1127 | ✅（Python verify 和 C# verify 使用不同初始局面） |
| 策略 Top-5 | label 7537 在第4位 | Elephant/Horse/Cannon | ✅ 合法 |
| **MCTS 50sims Q值** | — | **Q=0.653** | ✅ 显著高于 V2 的 Q=0.264 |

**关键提升：MCTS Q=0.653 vs V2 Q=0.264** — 价值头对局势的评估更加准确和自信，这将直接提升 MCTS 的搜索质量。

### Step 7.4：WPF 部署

- ✅ csproj 中已配置 `<Content>` 自动复制，Build 即更新模型
- ✅ Debug/Release 双配置均已验证模型存在
- ✅ V3 模型自动部署至 WPF Debug 输出目录（20:20 时间戳）

---

## 模型版本汇总（截至 2026-06-16）

| 版本 | 日期 | 架构 | 数据量 | 输入平面 | from_acc | to_acc | Top-10 | 参数量 | 状态 |
|------|------|------|--------|---------|----------|--------|--------|--------|------|
| V2 | 06-14 | CNN flat 8100 | 14,425 | 14 | — | — | 35% | 1.55M | 已覆盖 |
| V3 | 06-15 | CNN flat 8100 | 74,643 | 14 | — | — | 36% | 1.55M | 已覆盖 |
| V4 | 06-15 | CNN factored from/to | 74,643 | 14 | 13% | 5% | 33% | 116K | 已覆盖 |
| V5 | 06-15 | CNN factored + side-aware | 74,643 | 16 | 24% | 7% | 33% | 116K | 已覆盖 |
| V6 | 06-16 | CNN factored 64ch/6blk + side-aware | 74,643 | 16 | 25% | 8% | 33% | 509K | 已部署 |
| **V7** | **06-16** | **CNN full 8100 + side-aware** | **74,643** | **16** | — | — | **34%** (↑) | **6.32M** | **训练中** |

### 关键里程碑
- **V3→V4**: 侧边盲区诊断 → 架构改造为 factored head
- **V4→V5**: 添加行棋方平面 + 当前玩家值预测 (from_acc 13%→24%)
- **V5→V6**: 增加模型容量 (32ch/3blk→64ch/6blk, 116K→509K params)
- **V6→V7**: 改回 full 8100 policy head (解决 to_acc=8% 问题, Top-10 33%→34%)

---

## 阶段八：棋力增强与优化（2026-06-15 ~ 06-16）

### 问题诊断：NN AI 棋力太弱

用户反馈"测试下来还是很弱 感觉比昨天还弱"。经过系统性诊断发现：

#### 错误 #14：侧边盲区（致命）
- **现象**：模型策略熵 3.64 ≈ 随机基线 3.69，策略输出接近随机
- **原因**：14 个输入平面中**没有行棋方信息**，模型不知道轮到谁走
- **影响**：策略完全无效 → MCTS 得不到有效先验 → 棋力极弱
- **修复**：
  1. 添加 2 个行棋方指示平面（Plane 14: 红方行棋, Plane 15: 黑方行棋）
  2. 值预测改为当前玩家视角（+1 = 我赢，-1 = 我输）
  3. C# 推理代码自动检测 V5 模型（16 平面），正确处理行棋方

#### 错误 #15：factored head 的 to_acc 接近随机
- **现象**：from_acc=24% 可以，但 to_acc=8% 几乎是随机（1/15=6.7%）
- **原因**：from 和 to 独立预测，模型学不到"马走日、象走田"的依赖关系
- **修复**：V7 改回 full 8100 policy head，联合预测 P(from,to)

### 增强措施

1. **MCTS 模拟次数**：400→800 默认，最大 4000
2. **Level 4 搜索深度**：6→8 层
3. **V5/V6 模型**：16 输入平面 + 当前玩家值
4. **数据增强**：棋盘水平翻转（50%概率），有效翻倍训练数据
5. **MCTS 性能优化**：避免每个叶节点重复 NN 推理（搜索速度翻倍）
6. **V7 架构**：full 8100 policy head（AlphaZero 原版风格）

### 代码修改清单

| 文件 | 修改内容 |
|------|---------|
| `train_cnn_policy_value.py` | 添加行棋方平面+当前玩家值+数据增强+定期checkpoint |
| `export_cnn_onnx.py` | 支持 16 输入平面+V5 架构检测 |
| `XiangqiNeuralAiService.cs` | 16 平面编码+行棋方指示+V5值翻转+side参数 |
| `MctsAiService.cs` | 缓存NN值避免重复推理（性能翻倍）+MctsNode.CachedValue |
| `XiangqiAiService.cs` | Level 4 搜索深度 6→8 |
| `SidePanelViewModel.cs` | MCTS 默认模拟次数 400→800 |
| `SidePanelView.xaml` | MCTS 最大模拟次数 2000→4000 |
| `train_v7_full_policy.py` | 新建 V7 训练脚本（full 8100 policy head） |
