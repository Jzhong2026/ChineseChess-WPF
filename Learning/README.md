# Quiescence Search + Transposition Table 学习笔记

> 这是中国象棋 ML 项目中 AI 搜索的核心技术拆解。
> 现有代码位置：`ChineseChess.Core/Services/XiangqiAiService.cs`

---

## 一、Quiescence Search（静态搜索/静止搜索）

### 是什么

Alpha-Beta 剪枝在固定深度截断时，会遇到 **水平线效应（Horizon Effect）**：
> 电脑评估一个局面"很好"，但走完这步才发现对方下一步就把自己将死了——这个"将死"刚好在搜索深度之外。

**QSearch 解决思路：** 到达固定深度后，不直接返回评估值，而是**只搜索吃子走法**（capture moves），直到没有吃子可走（达到「静止」状态）。

### 核心逻辑

```
Quiescence(alpha, beta):
    stand_pat = Evaluate()        ← 当前局面的静态估值
    if stand_pat >= beta: return beta     ← 对手过激，剪枝
    alpha = max(alpha, stand_pat)

    for each capture move:
        make_move(capture)
        score = -Quiescence(-beta, -alpha)
        undo_move()
        if score >= beta: return beta
        alpha = max(alpha, score)
    return alpha
```

### 为什么有效

- 吃子走法数量远少于全部走法（象棋中 ~5-15 vs ~35-40），搜索树可控
- 能「看到」固定深度之外的战术组合（弃子攻杀、连续将军）
- 跟 Alpha-Beta 天然互补——Alpha-Beta 做广度，QSearch 做深度

---

## 二、Transposition Table（置换表/转置表）

### 是什么

同一个盘面可以通过不同的走法顺序到达（例如「炮二平五、马8进7」和「马8进7、炮二平五」结果相同）。
每次搜索都重复计算相同局面，浪费算力。

**TT 解决思路：** 用哈希表缓存已经搜索过的局面及其搜索结果。

### 核心哈希算法：Zobrist Hashing

```python
# 初始化：为每个(棋子, 位置)生成随机数
zobrist_table[piece_type][row][col] = random.randint(0, 2^64)

# 更新：走棋时 XOR 变化的部分
hash ^= zobrist_table[piece][from_row][from_col]   # 移除旧位置
hash ^= zobrist_table[piece][to_row][to_col]       # 放到新位置
if capture:
    hash ^= zobrist_table[captured_piece][to_row][to_col]  # 移除被吃子
```

### TT 存储内容

| 字段 | 含义 |
|------|------|
| `Key` | Zobrist hash（高 64 位作为索引，低 64 位作为验证）|
| `Depth` | 搜索深度（更深的搜索结果可以替代较浅的）|
| `Score` | 评估值（需要根据搜索层级转换）|
| `Flag` | EXACT（精确值）/ LOWERBOUND（Beta 截断）/ UPPERBOUND（Alpha 截断）|
| `BestMove` | 最佳走法（启发式排序，提升剪枝效率）|

### 为什么有效

- **加速：** 避免重复搜索相同局面（象棋中 30-50% 的节点是重复的）
- **启发式排序：** TT 中缓存的最佳走法优先搜索，剪枝率大幅提升
- **迭代加深配合：** 浅层搜索的结果填充 TT，指导深层搜索的走法排序

---

## 三、两者结合

```
AlphaBeta(alpha, beta, depth, ...):
    # 1. TT 查表
    tt_entry = ProbeTT(hash)
    if tt_entry.Depth >= depth:
        if tt_entry.Flag == EXACT: return tt_entry.Score
        if tt_entry.Flag == LOWERBOUND: alpha = max(alpha, tt_entry.Score)
        if tt_entry.Flag == UPPERBOUND: beta = min(beta, tt_entry.Score)
        if alpha >= beta: return tt_entry.Score

    # 2. QSearch 截断
    if depth == 0:
        return Quiescence(alpha, beta)

    # 3. 走法生成 + TT BestMove 优先
    moves = GenerateMoves()
    sort moves: TT BestMove first, then captures, then others

    # 4. 搜索
    for move in moves:
        make_move(move)
        score = -AlphaBeta(-beta, -alpha, depth - 1)
        undo_move()
        if score >= beta:
            SaveTT(hash, score, depth, LOWERBOUND, move)
            return score
        alpha = max(alpha, score)

    # 5. 存入 TT
    SaveTT(hash, alpha, depth, EXACT, bestMove)
    return alpha
```

---

## 四、在本项目中的应用

- `XiangqiAiService.cs` 中 `EvaluateWithQSearch()` 负责静态搜索
- `TranspositionTable` 类使用 Zobrist 哈希缓存搜索结果
- Level 3/4 的差异：搜索深度不同 + QSearch 的 Quiescence 深度不同
- 置换表大小：250,000 条目（满了直接清空——简单但有效）

---

## 五、演示代码

本目录包含一组渐进式 Demo：

| Demo | 语言 | 内容 |
|------|------|------|
| 1 | Python | QSearch + TT 基础概念，纯函数式，< 100 行 |
| 2 | Python + C# | 完整 Minimax + QSearch + TT 的井字棋 |
| 3 | Python + C# | 在简化象棋上集成 QSearch + TT 到现有引擎 |
| 4 | Python + C# | 神经网络棋盘编码、走法编码、CNN 输入重排、factored policy 解码 |
| 5 | Python | 自博弈 JSONL 样本校验、policy/value 训练标签、legal mask |
| 6 | Python | PyTorch MNIST 图像分类 |
| 7 | Python | PyTorch 表格回归 |
| 8 | Python | PyTorch 文本分类 |
| 9 | Python | 本地知识库 RAG、placeholder LLM、大模型配置预留 |
