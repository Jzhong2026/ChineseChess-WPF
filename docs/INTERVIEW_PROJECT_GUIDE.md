# 中国象棋 WPF + AI 项目面试介绍稿

> 适用场景：简历亮点项目、技术面试项目深挖、AI/客户端/工程化综合项目介绍。
>
> 推荐定位：这是一个从传统棋类程序逐步演进到“搜索引擎 + 自博弈数据 + 深度学习模型 + ONNX 本地推理 + MCTS”的端到端 AI 应用项目。

---

## 1. 项目一句话介绍

这是一个基于 WPF 的中国象棋单机应用。我在原有规则引擎和传统 Alpha-Beta 搜索 AI 的基础上，补齐了自博弈数据生成、PyTorch 策略价值网络训练、ONNX 导出、C# 本地推理，以及 NN + MCTS 搜索增强，形成了一条从“数据生成 → 模型训练 → 模型部署 → 游戏内推理”的完整闭环。

面试时可以这样开场：

> 这个项目不是只做一个象棋界面，而是把棋类 AI 的几个核心环节都串起来了：规则引擎负责合法走法，传统搜索提供基线和自博弈标签，PyTorch 训练策略价值网络，最后用 ONNX Runtime 集成回 WPF 客户端，实现本地可运行的神经网络 AI 和 MCTS AI。

---

## 2. 项目背景与目标

### 2.1 为什么做这个项目

传统棋类 AI 有两类典型方案：

- 基于人工评估函数 + Alpha-Beta 剪枝搜索
- 基于自博弈数据 + 神经网络 + MCTS 的 AlphaZero 式路线

这个项目的目标是把二者结合起来：

- 保留传统搜索作为稳定基线
- 用传统搜索自博弈生成训练数据
- 训练神经网络学习局面价值和走法倾向
- 将模型部署到 WPF 应用中，让普通用户无需 Python 环境也能使用 AI

### 2.2 项目最终能力

当前项目具备以下能力：

| 能力 | 说明 |
|---|---|
| WPF 象棋客户端 | 支持棋盘显示、走棋、悔棋、音效、AI 对弈 |
| 规则引擎 | 完整生成中国象棋合法走法，并处理将军、胜负、历史局面 |
| 传统 AI | Negamax + Alpha-Beta + Quiescence Search + Transposition Table |
| 自博弈数据生成 | C# 控制台程序输出 JSONL 训练样本 |
| 棋盘/走法编码 | `float[1260]` 棋盘编码和 `0..8099` action 编码 |
| 神经网络训练 | PyTorch CNN 策略价值网络 |
| ONNX 部署 | PyTorch checkpoint 导出为 ONNX，C# 使用 ONNX Runtime 推理 |
| MCTS 增强 | 使用神经网络 policy/value 作为先验与评估，进行 PUCT 搜索 |

---

## 3. 技术栈

### 3.1 客户端与工程

- C# / .NET
- WPF
- Caliburn.Micro MVVM
- 多项目结构：
  - `ChineseChess.Core`：规则、模型、AI、编码
  - `ChineseChess`：WPF UI
  - `ChineseChess.SelfPlay`：自博弈数据生成
  - `ChineseChess.Core.Verification`：编码和核心逻辑验证

### 3.2 AI 与机器学习

- 传统搜索：
  - Negamax
  - Alpha-Beta Pruning
  - Iterative Deepening
  - Quiescence Search
  - Transposition Table
  - Zobrist Hash
- 神经网络：
  - PyTorch
  - CNN / ResNet 风格残差块
  - Policy Head
  - Value Head
  - 16 平面输入：14 个棋子平面 + 2 个行棋方平面
- 部署：
  - ONNX
  - Microsoft.ML.OnnxRuntime
- 搜索增强：
  - MCTS
  - PUCT
  - legal mask

### 3.3 数据与训练

- JSONL 自博弈数据
- 棋盘编码：`10 * 9 * 14 = 1260`
- 动作空间：`90 * 90 = 8100`
- 训练目标：
  - policy：预测搜索 AI 选择的走法
  - value：预测最终胜负结果
- 数据增强：
  - 左右水平翻转
- 样本加权：
  - `PolicyWeight`：深度越高，走法标签越可信
  - `ValueWeight`：真实将死局权重大于裁定局

---

## 4. 整体架构

### 4.1 核心架构图

```text
WPF UI
  |
  v
ShellViewModel / BoardViewModel
  |
  +--> XiangqiEngine
  |      - 合法走法生成
  |      - 胜负判断
  |      - MakeMove
  |
  +--> XiangqiAiService
  |      - Negamax
  |      - Alpha-Beta
  |      - QSearch
  |      - Transposition Table
  |
  +--> XiangqiNeuralAiService
  |      - BoardEncoder
  |      - ONNX Runtime
  |      - policy/value 推理
  |
  +--> MctsAiService
         - PUCT
         - NN policy 作为先验
         - NN value 作为叶节点估值
```

### 4.2 训练部署链路

```text
1. ChineseChess.SelfPlay
   传统 AI 自博弈，输出 JSONL

2. train_cnn_policy_value.py
   读取 JSONL，训练 CNN 策略价值网络

3. export_cnn_onnx.py
   PyTorch checkpoint 导出 ONNX

4. XiangqiNeuralAiService.cs
   WPF 运行时加载 ONNX，完成本地推理

5. MctsAiService.cs
   将神经网络 policy/value 融入 MCTS 搜索
```

---

## 5. 开发过程

### 阶段一：先把规则引擎和传统 AI 跑稳

项目最基础的是规则正确性。如果合法走法、将军判断、胜负判断有问题，后面的训练数据和模型都会被污染。

这一阶段的重点：

- 梳理 `XiangqiEngine` 的职责
- 确认棋盘坐标系统为 `10 x 9`
- 确认 `PieceType`、`Side`、`Position`、`Move` 等核心模型稳定
- 保留传统搜索 AI 作为基线

面试表达：

> 我没有一开始就上神经网络，而是先保证规则引擎和传统搜索 AI 稳定。因为棋类 AI 的训练数据来自规则引擎，如果规则层有 bug，后面模型学到的就是错误标签。

### 阶段二：实现可训练的数据编码

为了让模型学习棋盘，需要将对象化的 C# 棋盘转成数值特征。

棋盘编码：

```text
10 行 * 9 列 * 14 平面 = 1260
14 平面 = 2 方 * 7 种棋子
index = (row * 9 + col) * 14 + plane
```

走法编码：

```text
fromIndex = from.Row * 9 + from.Col
toIndex   = to.Row * 9 + to.Col
actionId  = fromIndex * 90 + toIndex
范围：0..8099
```

这里的关键点是：动作空间包含大量非法动作，所以训练和推理时必须用 legal mask，只在当前合法走法中计算损失或选择最大概率。

### 阶段三：用传统 AI 生成自博弈数据

`ChineseChess.SelfPlay` 会让传统 AI 自己下棋，并在每一步记录：

- 当前棋盘编码
- 当前行棋方
- 合法走法列表
- 搜索 AI 选中的走法
- 搜索分数
- 最终胜负结果
- policy/value 权重

每行 JSONL 是一个训练样本。

这样做的好处：

- 不依赖人工棋谱
- 可以快速生成大量带标签数据
- 搜索深度越高，标签质量越好
- 可以通过随机开局和 Top-K 采样增加局谱多样性

### 阶段四：从 MLP 迭代到 CNN

早期可以用 MLP 验证训练流程，但 MLP 不适合棋盘类任务，因为它不天然理解空间结构。

后来改为 CNN / ResNet 风格网络：

- 输入 `[batch, 16, 10, 9]`
- 14 个棋子平面
- 2 个行棋方平面
- CNN backbone 提取空间关系
- policy head 输出走法倾向
- value head 输出当前玩家胜率估计

面试表达：

> 我最开始用 MLP 快速打通流程，但发现它对棋盘空间结构泛化很弱。象棋里的马腿、炮架、车的直线控制都依赖二维关系，所以我改成 CNN，并通过残差块增强表达能力。

### 阶段五：导出 ONNX 并接入 WPF

训练完成后，不能让 WPF 程序依赖 Python 环境，所以使用 ONNX Runtime。

流程：

```text
PyTorch .pt checkpoint
  -> torch.onnx.export
  -> cnn_policy_value.onnx
  -> C# InferenceSession
  -> 本地推理
```

接入时做了几个工程化处理：

- ONNX 模型懒加载，避免模型缺失导致应用启动失败
- 自动识别模型输出是 flat policy 还是 factored from/to policy
- 自动识别输入是 14 平面还是 16 平面
- 推理结果只在合法走法中选择

### 阶段六：用 MCTS 增强纯神经网络策略

纯神经网络直接选最大概率走法很快，但棋力不稳定。于是增加 MCTS：

- policy 输出作为先验概率
- value 输出作为叶节点估值
- PUCT 公式平衡探索与利用
- 多次模拟后根据访问次数选择最佳走法

面试表达：

> 纯神经网络只做一次前向传播，速度快但容易短视；MCTS 可以用神经网络给出的先验缩小搜索空间，再用价值网络评估叶节点，相当于把模型判断和搜索结合起来。

---

## 6. 关键技术细节

### 6.1 Alpha-Beta + QSearch

Alpha-Beta 用来减少搜索树分支，QSearch 用来解决水平线效应。

普通固定深度搜索的问题是：到深度边界时可能刚好停在一个激烈吃子局面，评估会失真。QSearch 在深度用完后继续只搜索吃子走法，直到局面相对稳定。

核心思想：

```text
if depth == 0:
    return Quiescence(alpha, beta)

for move in orderedMoves:
    make(move)
    score = -Search(-beta, -alpha, depth - 1)
    undo(move)
```

### 6.2 Transposition Table

同一局面可能通过不同走法顺序到达。置换表用 Zobrist Hash 缓存已经搜索过的局面：

- `EXACT`：精确分数
- `LOWERBOUND`：Beta 截断下界
- `UPPERBOUND`：Alpha 截断上界
- `BestMove`：用于下一次优先排序

价值：

- 避免重复搜索
- 提升剪枝效率
- 配合迭代加深效果更明显

### 6.3 训练样本设计

一行自博弈样本包含：

```json
{
  "BoardEncoding": [0.0, 1.0, "..."],
  "SideToMove": 1,
  "LegalMoves": [7362, 7445],
  "SelectedMove": 7362,
  "Result": 1,
  "ValueWeight": 1.0,
  "PolicyWeight": 1.0
}
```

policy 目标：

```text
selected_move = from * 90 + to
from_target = selected_move // 90
to_target   = selected_move % 90
```

value 目标：

```text
value_target = result * side_to_move
```

这样 `+1` 永远表示“当前行棋方最终赢”，`-1` 表示“当前行棋方最终输”。

### 6.4 CNN 输入重排

C# `BoardEncoder` 输出是 square-major：

```text
flat[(row * 9 + col) * 14 + plane]
```

PyTorch / ONNX CNN 需要 plane-major：

```text
tensor[batch, plane, row, col]
```

所以 C# 推理端必须重排：

```csharp
for (var row = 0; row < 10; row++)
for (var col = 0; col < 9; col++)
for (var plane = 0; plane < 14; plane++)
{
    var flatIdx = (row * 9 + col) * 14 + plane;
    tensor[0, plane, row, col] = flat[flatIdx];
}
```

这是项目中非常适合面试展开的细节，因为它体现了跨语言张量布局排查能力。

### 6.5 legal mask

模型输出覆盖 8100 个 from-to 组合，其中大部分在当前局面非法。

所以推理时不能直接全局 argmax，而要：

```text
legalMoves = engine.GetLegalMoves(...)
best = legalMoves.MaxBy(move => policy[MoveEncoder.Encode(move)])
```

训练评估时也要关注 legal top-k，而不只看 8100 分类的全局 top-1。

---

## 7. 遇到的问题与解决方案

### 问题一：.NET SDK 版本不兼容

现象：

- 项目原本目标框架包含 `net10.0`
- 当前环境只有 .NET 8 SDK
- 构建报错 `NETSDK1045`

解决：

- 将项目目标框架调整为 `net8.0` / `net8.0-windows`
- 直接构建 `.csproj`，避免 `.slnx` 在旧 SDK 下不兼容

收获：

> 工程项目最好用 `global.json` 锁定 SDK，避免不同机器构建环境不一致。

### 问题二：CNN 训练时棋盘 reshape 错误

现象：

- 训练 loss 看起来下降
- 但验证策略准确率接近 0
- 模型几乎学不到有效走法

根因：

- C# 输出的 `BoardEncoding` 是 square-major
- Python 训练脚本一开始直接 `.view(14, 10, 9)`
- 这会把棋盘特征错误解释成 plane-major，导致空间结构全乱

解决：

```python
flat = torch.tensor(board_encoding, dtype=torch.float32)
features = flat.view(90, 14).permute(1, 0).view(14, 10, 9)
```

收获：

> `.view()` 不是数据重排，只是重新解释连续内存。跨 C# 和 PyTorch 传张量时，必须明确每个维度的语义和内存布局。

### 问题三：模型不知道轮到谁走

现象：

- 14 平面只表示棋子位置
- 同一个棋盘在红方走和黑方走时，输入完全一样
- 策略熵接近随机，value 预测不稳定

根因：

- 输入缺失 side-to-move 信息
- 模型无法区分“这个局面轮到红走”还是“轮到黑走”

解决：

- 增加 2 个行棋方平面：
  - plane 14：红方行棋时全 1
  - plane 15：黑方行棋时全 1
- value 改为当前玩家视角：
  - `+1 = 当前玩家赢`
  - `-1 = 当前玩家输`

收获：

> 神经网络不是自动知道游戏状态的，输入特征必须包含决策所需的全部信息。

### 问题四：训练数据不足与过拟合

现象：

- 训练集 loss 降得很快
- 验证集指标提升有限
- Top-1 准确率很低

根因：

- 象棋动作空间大，8100 分类很稀疏
- 早期样本只有几千到一万多行，不足以支撑百万级参数模型
- 自博弈局谱重复度较高

解决：

- 增加自博弈局数
- 使用随机开局、Top-K 采样提高多样性
- 使用 dropout、weight decay
- 引入左右翻转数据增强
- 评估时关注 legal top-5 / top-10，而不仅是 top-1

收获：

> 棋类模型的 Top-1 命中不是唯一指标。如果正确走法能进入 Top-10，就已经能为 MCTS 提供有价值的搜索先验。

### 问题五：factored policy head 的局限

尝试过将策略拆成：

```text
from_logits[90] + to_logits[90]
```

优点：

- 参数量大幅下降
- from-square 学得比较快

问题：

- from 和 to 独立预测，较难表达“某个棋子从某个位置能走到哪里”的强依赖
- 对马、象、炮这类走法依赖上下文的棋子不够友好

改进：

- 后续重新尝试 full 8100 policy head
- 保留模型类型自动检测逻辑，兼容两种输出格式

收获：

> 降参数不一定等于效果更好。棋类动作是 from-to 的联合决策，过度拆分会损失动作结构信息。

### 问题六：ONNX 集成需要兼容多模型版本

现象：

- 早期模型是 14 平面
- 新模型是 16 平面
- 有的模型输出 `policy_logits[8100]`
- 有的模型输出 `from_logits[90]` 和 `to_logits[90]`

解决：

- C# 推理服务读取 ONNX metadata
- 自动判断输入平面数量
- 自动判断输出 head 类型
- 对 factored 输出做合并：

```text
policy[from * 90 + to] = from_logits[from] + to_logits[to]
```

收获：

> 模型会迭代，推理服务需要设计成可兼容，而不是只写死某一个版本。

---

## 8. 项目亮点

### 亮点一：端到端 AI 工程闭环

这个项目不是只训练一个模型，也不是只写一个界面，而是完整串起：

```text
规则引擎 -> 自博弈数据 -> 训练 -> ONNX 导出 -> C# 推理 -> WPF AI 对弈
```

这体现了端到端工程能力。

### 亮点二：传统搜索与深度学习结合

传统搜索不是被丢弃，而是承担三个角色：

- 作为可用基线 AI
- 作为自博弈数据标注器
- 作为评估神经网络棋力的对照组

### 亮点三：跨语言模型部署

训练在 Python / PyTorch，推理在 C# / WPF：

- 需要处理张量布局
- 需要处理 ONNX 输入输出 metadata
- 需要处理模型文件缺失、懒加载、异常降级

### 亮点四：真实问题驱动的模型迭代

项目中不是一次写完，而是根据指标不断修正：

- MLP → CNN
- 14 平面 → 16 平面
- Red perspective value → current-player value
- factored head → full 8100 head
- 普通准确率 → legal top-k 指标

### 亮点五：有工程化验证意识

项目中增加了学习 demo、验证程序和开发日志：

- `ChineseChess.Core.Verification`
- `Learning/demo4_neural_encoding`
- `Learning/demo5_selfplay_dataset`
- `docs/DEV_LOG.md`

这些内容能说明项目不是“能跑就行”，而是有可解释、可验证、可复盘的开发过程。

---

## 9. 面试讲述版本

### 9.1 1 分钟版本

> 我做的是一个 WPF 中国象棋 AI 项目。它一开始有完整的规则引擎和传统 Alpha-Beta 搜索 AI，我在这个基础上扩展了自博弈数据生成、PyTorch 策略价值网络训练、ONNX 导出和 C# 本地推理。项目里棋盘用 14 个棋子平面编码，后续又加了 2 个行棋方平面，动作空间是 90×90 的 8100 个 from-to 组合。模型训练完成后通过 ONNX Runtime 集成到 WPF，支持纯神经网络策略和 NN + MCTS 两种 AI 模式。过程中我解决了张量布局不一致、模型缺失行棋方信息、数据过拟合、legal mask 等问题，是一个比较完整的端到端 AI 工程项目。

### 9.2 3 分钟版本

> 这个项目是一个中国象棋 WPF 应用，我把它从传统棋类程序扩展成了带机器学习能力的 AI 系统。整体分成几层：最底层是 `XiangqiEngine`，负责合法走法、胜负判断和走棋；传统 AI 使用 Negamax、Alpha-Beta 剪枝、QSearch 和置换表；然后我写了 SelfPlay 程序，让传统 AI 自博弈并输出 JSONL 训练数据；Python 侧用 PyTorch 训练 CNN 策略价值网络；最后导出 ONNX，在 C# 侧用 ONNX Runtime 加载模型，集成回 WPF。
>
> 数据编码方面，棋盘是 10×9，共 90 个格子，每个格子有 14 个棋子平面，所以是 `float[1260]`。走法编码是 `fromIndex * 90 + toIndex`，范围 0 到 8099。因为 8100 个动作里大部分是非法走法，所以训练和推理都必须结合 legal mask，只在合法走法中计算或选择。
>
> 项目中遇到的一个典型问题是 C# 的棋盘编码是 square-major，而 PyTorch CNN 需要 `[plane,row,col]` 的 plane-major。如果直接 `.view(14,10,9)`，模型看到的棋盘就是乱的。我通过 `view(90,14).permute(1,0).view(14,10,9)` 修复了训练端，在 C# 推理端也做了同样的重排。
>
> 后来还发现 14 平面只描述棋子位置，没有告诉模型轮到谁走，所以同一个盘面红走和黑走输入完全一样。这个问题会导致策略接近随机。我加了两个 side-to-move 平面，并把 value target 改为当前玩家视角，模型才具备正确判断局面的输入信息。
>
> 最后，为了提升纯神经网络的短视问题，我又接了 MCTS，用 policy 作为先验，用 value 评估叶节点。这让项目从规则、搜索、训练、部署到搜索增强形成了完整闭环。

### 9.3 5 分钟版本结构

如果面试官让你详细讲，可以按这个顺序：

1. 项目目标：WPF 中国象棋 + AI 升级
2. 原有基础：规则引擎 + Alpha-Beta 搜索
3. 数据闭环：SelfPlay 输出 JSONL
4. 编码设计：1260 棋盘特征 + 8100 动作空间
5. 模型训练：CNN policy/value 网络
6. 部署方案：PyTorch → ONNX → C# ONNX Runtime
7. 搜索增强：NN + MCTS
8. 典型问题：
   - SDK 版本问题
   - square-major vs plane-major
   - side-to-move 缺失
   - 数据不足和过拟合
   - legal mask
9. 项目亮点：
   - 端到端闭环
   - 跨语言部署
   - 搜索和深度学习结合
   - 真实问题驱动的迭代

---

## 10. 面试高频问答

### Q1：为什么动作空间是 8100？

棋盘有 90 个格子，一个动作可以抽象成“从某格到某格”，所以是 `90 * 90 = 8100`。虽然其中大部分动作在具体局面下非法，但这种编码统一、简单，便于模型输出固定维度。非法动作通过 legal mask 过滤。

### Q2：为什么不用模型直接输出合法走法？

每个局面的合法走法数量不固定，大约几十个，而神经网络通常需要固定维度输出。用 8100 维固定输出再结合 legal mask，是棋类 AI 常见做法。

### Q3：为什么要加行棋方平面？

只看棋子位置无法判断轮到谁走。同一个棋盘，红方走和黑方走对应的最佳动作和胜率可能完全不同。行棋方是局面状态的一部分，必须进入模型输入。

### Q4：为什么 value target 要乘 `side_to_move`？

原始结果通常是红方视角：红胜为 `+1`，黑胜为 `-1`。但模型推理时更自然的是当前玩家视角：当前玩家赢就是 `+1`。所以：

```text
value_target = result_red_perspective * side_to_move
```

当黑方行棋且红方最终赢时，目标就是 `1 * -1 = -1`，表示当前玩家黑方会输。

### Q5：为什么要用 ONNX？

训练在 Python/PyTorch 侧完成，但 WPF 客户端是 C# 应用。ONNX 是跨框架模型格式，ONNX Runtime 可以在 C# 中直接本地推理，不需要用户安装 Python 环境，也不需要启动额外服务。

### Q6：MCTS 相比纯神经网络有什么优势？

纯神经网络只做一次判断，速度快但容易短视。MCTS 会基于 policy 先验做多次模拟，并用 value 评估叶节点，相当于把模型的直觉和局部搜索结合起来，棋力更稳定。

### Q7：这个项目最难的问题是什么？

最难的不是写模型，而是让整个链路的数据语义一致。比如棋盘编码在 C# 是 square-major，在 PyTorch 是 plane-major；value 既可以是红方视角，也可以是当前玩家视角；policy 输出覆盖 8100 个动作，但实际只能从合法走法里选。这些如果任何一个地方没对齐，模型指标就会异常。

### Q8：如果继续优化，你会怎么做？

优先方向：

1. 生成更高质量、更大规模的自博弈数据
2. 改成流式 Dataset，降低大 JSONL 全量加载内存压力
3. 用模型指导自博弈，走 AlphaZero 式迭代
4. 做新旧模型自动对战评估，胜率超过阈值才替换
5. 优化 MCTS 批量推理，减少重复 ONNX 调用

---

## 11. 简历写法参考

可以放在简历项目经历里：

```text
中国象棋 WPF AI 对弈系统
- 基于 C# WPF 和 Caliburn.Micro 实现中国象棋客户端，拆分 Core、SelfPlay、Verification 等模块，完成规则引擎、棋盘交互和 AI 对弈功能。
- 实现 Negamax + Alpha-Beta 剪枝 + Quiescence Search + Transposition Table 的传统搜索 AI，作为游戏基线和自博弈数据标注器。
- 设计棋盘与走法编码：10x9x14 棋盘特征、90x90=8100 动作空间，并通过 legal mask 处理非法动作。
- 构建 Self-Play 数据生成管线，输出 JSONL 样本，包含棋盘编码、合法走法、搜索选招、胜负结果、policy/value 权重。
- 使用 PyTorch 训练 CNN 策略价值网络，引入 side-to-move 输入平面、当前玩家视角 value target、水平翻转增强和加权损失。
- 通过 ONNX Runtime 将模型部署到 C# WPF 客户端，并实现 NN 策略模式与 NN + MCTS 模式。
- 排查并解决 C#/PyTorch 张量布局不一致、行棋方信息缺失、训练过拟合、模型版本兼容等问题。
```

---

## 12. 面试时最值得主动强调的点

1. **我不是只做 UI，而是做了完整 AI 工程闭环。**
2. **我理解传统棋类搜索，不只是调模型。**
3. **我处理过跨语言张量布局、ONNX 部署、legal mask 这些真实工程问题。**
4. **我能根据指标诊断问题，而不是盲目调参。**
5. **我有开发日志和学习 demo，说明项目过程可复盘。**

---

## 13. 推荐现场讲解顺序

如果面试时对方让你打开代码，可以按下面顺序展示：

1. `ChineseChess.Core/Services/XiangqiEngine.cs`
   - 说明规则引擎和合法走法生成
2. `ChineseChess.Core/Encoding/BoardEncoder.cs`
   - 说明棋盘编码
3. `ChineseChess.Core/Encoding/MoveEncoder.cs`
   - 说明 8100 动作空间
4. `ChineseChess.SelfPlay/Program.cs`
   - 说明自博弈数据生成
5. `train_cnn_policy_value.py`
   - 说明 CNN 输入、side-to-move、policy/value loss
6. `export_cnn_onnx.py`
   - 说明模型导出
7. `ChineseChess.Core/Services/XiangqiNeuralAiService.cs`
   - 说明 C# ONNX 推理和张量重排
8. `ChineseChess.Core/Services/MctsAiService.cs`
   - 说明 NN + MCTS 搜索增强

---

## 14. 项目总结

这个项目最大的价值在于它不是一个单点 demo，而是一个完整的 AI 应用系统：

- 有可交互的客户端
- 有确定性的规则引擎
- 有传统搜索基线
- 有自博弈数据生成
- 有深度学习训练
- 有 ONNX 本地部署
- 有 MCTS 搜索增强
- 有问题记录、验证脚本和学习文档

面试中可以把它定位为：

> 一个结合 C# 客户端工程、传统搜索算法、机器学习训练和模型部署的综合型项目。

