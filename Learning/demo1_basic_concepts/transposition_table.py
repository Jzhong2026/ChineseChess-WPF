"""
Demo 1b: Transposition Table + Zobrist Hashing 基础
===================================================
纯 Python，展示：
1. Zobrist 哈希的生成和增量更新
2. 置换表的数据结构和存取逻辑
3. 用井字棋展示 TT 的加速效果
"""

import random

# ─── Zobrist 哈希 ───

class ZobristHasher:
    """
    Zobrist 哈希：通过 XOR 增量更新棋盘哈希值。
    核心特性：
    - 走棋时只需 XOR 变化的部分，O(1) 更新
    - 不同走法顺序到达同一盘面 → 相同哈希值
    - 64 位哈希碰撞概率极低（约 1/2^64）
    """

    def __init__(self, board_rows, board_cols, piece_types):
        # 为每个(棋子类型, 行, 列)生成 64 位随机数
        self.table = {}
        for piece in range(1, piece_types + 1):
            for r in range(board_rows):
                for c in range(board_cols):
                    self.table[(piece, r, c)] = random.getrandbits(64)

        # 行棋方标记（轮到谁走也影响哈希值）
        self.side_to_move = random.getrandbits(64)

    def initial_hash(self, board, rows, cols):
        """从棋盘状态计算初始哈希值"""
        h = 0
        for r in range(rows):
            for c in range(cols):
                piece = board[r][c]
                if piece != 0:
                    h ^= self.table[(abs(piece), r, c)]
        return h

    def update_hash(self, old_hash, piece, from_r, from_c, to_r, to_c, captured=0):
        """走棋后增量更新哈希值"""
        h = old_hash
        # 移除旧位置的棋子
        h ^= self.table[(abs(piece), from_r, from_c)]
        # 放到新位置
        h ^= self.table[(abs(piece), to_r, to_c)]
        # 移除被吃子
        if captured:
            h ^= self.table[(abs(captured), to_r, to_c)]
        return h


# ─── 置换表 ───

TT_EXACT, TT_LOWERBOUND, TT_UPPERBOUND = 0, 1, 2


class TTEntry:
    """置换表条目"""

    def __init__(self, key, depth, score, flag, best_move=None):
        self.key = key          # 完整 Zobrist hash（验证用）
        self.depth = depth      # 搜索深度
        self.score = score      # 评估值
        self.flag = flag        # EXACT / LOWERBOUND / UPPERBOUND
        self.best_move = best_move  # 最佳走法（启发式排序用）


class TranspositionTable:
    """
    置换表：用哈希表缓存搜索结果。
    容量满时直接清空（简单策略，本项目采用的方式）。
    """

    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.table = {}
        self.hits = 0
        self.misses = 0

    def probe(self, zobrist_key, depth, alpha, beta):
        """
        查询置换表。
        返回 (found, score, best_move)
        - found=True 时可以直接使用 score
        - 需要根据 flag 和 depth 判断可用性
        """
        entry = self.table.get(zobrist_key)
        if entry is None:
            self.misses += 1
            return False, None, None

        # 验证 key 完整匹配
        if entry.key != zobrist_key:
            self.misses += 1
            return False, None, None

        self.hits += 1

        # 只有当缓存的搜索深度 >= 当前需要的深度时才可用
        if entry.depth >= depth:
            if entry.flag == TT_EXACT:
                return True, entry.score, entry.best_move
            if entry.flag == TT_LOWERBOUND and entry.score >= beta:
                return True, entry.score, entry.best_move
            if entry.flag == TT_UPPERBOUND and entry.score <= alpha:
                return True, entry.score, entry.best_move

        # 缓存深度不够，但 best_move 仍然可用（用于走法排序）
        return False, None, entry.best_move

    def save(self, zobrist_key, depth, score, flag, best_move=None):
        """存入置换表"""
        if len(self.table) >= self.max_size:
            # 简单策略：满了就清空
            self.table.clear()
        self.table[zobrist_key] = TTEntry(
            zobrist_key, depth, score, flag, best_move
        )

    def stats(self):
        return f"TT: {self.hits} hits / {self.hits + self.misses} probes = {self.hits / max(1, self.hits + self.misses) * 100:.1f}%"


# ─── 演示：TT 加速效果 ───

def demo_tt_acceleration():
    """
    用井字棋来演示 TT 的加速效果。
    井字棋虽然简单，但不同走法顺序（如 左上→中上 vs 中上→左上）会到达相同局面，
    TT 能避免重复计算。
    """

    print("=" * 60)
    print("Demo: 置换表加速效果 (Tic-Tac-Toe)")
    print("=" * 60)

    # 使用更复杂的棋盘（4x4 井字棋，增大搜索空间来体现差异）
    import time

    class TicTacToeDemo:
        def __init__(self, size=3):
            self.size = size
            self.board = [0] * (size * size)
            self.zobrist = ZobristHasher(size, size, 1)  # 1 种棋子类型
            self._build_zobrist()

        def _build_zobrist(self):
            self.ztable = {}
            for piece in (1, 2):  # player 1 and 2
                for i in range(self.size * self.size):
                    self.ztable[(piece, i)] = random.getrandbits(64)
            self.side_hash = random.getrandbits(64)

        def hash_board(self, board, side):
            h = 0
            for i, p in enumerate(board):
                if p != 0:
                    h ^= self.ztable[(p, i)]
            if side == 2:  # 黑方行棋时标记
                h ^= self.side_hash
            return h

        def is_winner(self, board, player):
            n = self.size
            lines = []
            # 行
            for r in range(n):
                lines.append([r * n + c for c in range(n)])
            # 列
            for c in range(n):
                lines.append([r * n + c for r in range(n)])
            # 对角线
            lines.append([i * n + i for i in range(n)])
            lines.append([i * n + (n - 1 - i) for i in range(n)])

            for line in lines:
                if all(board[pos] == player for pos in line):
                    return True
            return False

        def is_full(self, board):
            return all(p != 0 for p in board)

        def generate_moves(self, board):
            return [i for i, p in enumerate(board) if p == 0]

        def negamax(self, board, player, tt, depth=0):
            """Negamax with TT (简化版，无 QSearch)"""
            h = self.hash_board(board, player)

            # TT probe
            found, score, _ = tt.probe(h, depth, -999999, 999999)
            if found:
                return score

            # 终局判断
            opponent = 1 if player == 2 else 2
            if self.is_winner(board, opponent):
                return -100 + depth  # 越早赢越好
            if self.is_full(board):
                return 0

            best = -999999
            for move in self.generate_moves(board):
                board[move] = player
                score = -self.negamax(board, opponent, tt, depth + 1)
                board[move] = 0
                if score > best:
                    best = score

            # TT save
            tt.save(h, depth, best, TT_EXACT)
            return best

        def negamax_no_tt(self, board, player, depth=0):
            """Negamax without TT"""
            opponent = 1 if player == 2 else 2
            if self.is_winner(board, opponent):
                return -100 + depth
            if self.is_full(board):
                return 0

            best = -999999
            for move in self.generate_moves(board):
                board[move] = player
                score = -self.negamax_no_tt(board, opponent, depth + 1)
                board[move] = 0
                if score > best:
                    best = score
            return best

    # 测试
    game = TicTacToeDemo(3)

    # 一个接近填满的盘面，有很多路径会重合
    board = [0] * 9
    board[0] = 1  # X
    board[4] = 2  # O
    board[8] = 1  # X

    tt_with = TranspositionTable(50000)
    tt_without = TranspositionTable(1)

    # 带 TT
    t0 = time.time()
    for _ in range(50):
        b = board.copy()
        game.negamax(b, 2, tt_with)
    t_with = time.time() - t0

    # 不带 TT（每次使用新 TT，相当于无缓存）
    t1 = time.time()
    for _ in range(50):
        b = board.copy()
        game.negamax_no_tt(b, 2)
    t_no_tt = time.time() - t1

    print(f"\n测试条件：50 次重复搜索相同初始局面")
    print(f"带 TT:   {t_with:.3f}s ({tt_with.stats()})")
    print(f"无 TT:   {t_no_tt:.3f}s")
    print(f"加速比:  {t_no_tt / t_with:.1f}x")
    print()

    # 展示置换表的走法顺序无关性
    print("-" * 40)
    print("关键洞察：置换表的走法无关性")
    print("-" * 40)
    print("走法 A→B→C 和 走法 B→A→C 到达相同盘面，")
    print("TT 让第二次搜索直接返回结果，无需重新计算。")
    print("象棋中约 30-50% 的节点是重复的，TT 收益显著。")
    print()


def demo_hash_collision():
    """演示 Zobrist 哈希的增量更新特性"""
    print("=" * 60)
    print("Demo: Zobrist 哈希增量更新")
    print("=" * 60)

    hasher = ZobritHasher(3, 3, 3)

    # 创建一个 3x3 棋盘
    board = [[0] * 3 for _ in range(3)]
    board[0][0] = 1  # 红方某子
    board[1][1] = -2  # 黑方某子

    h1 = hasher.initial_hash(board, 3, 3)
    print(f"初始哈希值: {h1:#018x}")

    # 模拟走棋：红子从 (0,0) 走到 (0,1)，吃掉黑子
    h2 = hasher.update_hash(h1, 1, 0, 0, 0, 1, captured=0)
    print(f"走棋后哈希: {h2:#018x}")

    # 再走回来
    h3 = hasher.update_hash(h2, 1, 0, 1, 0, 0)
    print(f"走回原位的哈希: {h3:#018x}")
    print(f"与初始相同: {'✅' if h1 == h3 else '❌'}")
    print()

    # 展示交换走法得到相同哈希
    print("-" * 40)
    print("走法顺序无关性演示：")
    print("-" * 40)
    # A走法：先走子A，再走子B
    h_a = hasher.update_hash(h1, 1, 0, 0, 0, 1)
    h_a = hasher.update_hash(h_a, -2, 1, 1, 1, 2)
    # B走法：先走子B，再走子A（到达相同盘面）
    h_b = hasher.update_hash(h1, -2, 1, 1, 1, 2)
    h_b = hasher.update_hash(h_b, 1, 0, 0, 0, 1)
    print(f"走法 A 哈希: {h_a:#018x}")
    print(f"走法 B 哈希: {h_b:#018x}")
    print(f"相同盘面: {'✅' if h_a == h_b else '❌'}")
    print()


if __name__ == "__main__":
    demo_hash_collision()
    demo_tt_acceleration()
