"""
Demo 2a: 带 QSearch + TT 的完整井字棋 AI (Python)
=================================================
从简单到完整的渐进实现。
阶段1：纯 Minimax
阶段2：+ Alpha-Beta
阶段3：+ TT
阶段4：+ QSearch (井字棋没有吃子，但展示概念)
"""

import time
import random

random.seed(42)

# ─── 阶段 0：棋盘与规则 ───

N = 3
EMPTY, X, O = 0, 1, 2

class Board:
    """井字棋棋盘"""

    def __init__(self):
        self.cells = [EMPTY] * (N * N)
        self.side = X  # X 先手

    def clone(self):
        b = Board()
        b.cells = self.cells.copy()
        b.side = self.side
        return b

    def moves(self):
        return [i for i, c in enumerate(self.cells) if c == EMPTY]

    def apply(self, move):
        self.cells[move] = self.side
        self.side = O if self.side == X else X

    def undo(self, move):
        self.cells[move] = EMPTY
        self.side = O if self.side == X else X

    def is_win(self, player):
        c = self.cells
        return any(
            all(c[i] == player for i in line)
            for line in [
                [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
                [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
                [0, 4, 8], [2, 4, 6],              # diag
            ]
        )

    def is_draw(self):
        return not self.moves() and not self.is_win(X) and not self.is_win(O)

    def game_over(self):
        return self.is_win(X) or self.is_win(O) or self.is_draw()

    def evaluate(self):
        """简单评估：从 X 视角"""
        if self.is_win(X): return 100
        if self.is_win(O): return -100
        return 0

    def print(self):
        chars = {EMPTY: '.', X: 'X', O: 'O'}
        for r in range(N):
            print(' '.join(chars[self.cells[r * N + c]] for c in range(N)))


# ─── 阶段 1：纯 Minimax ───

def minimax(board):
    """纯递归 Minimax"""
    if board.game_over():
        return board.evaluate()

    best = -999 if board.side == X else 999
    for m in board.moves():
        board.apply(m)
        score = minimax(board)
        board.undo(m)
        if board.side == X:
            best = max(best, score)
        else:
            best = min(best, score)
    return best


# ─── 阶段 2：Minimax + Alpha-Beta ───

def minimax_ab(board, alpha, beta):
    """Alpha-Beta 剪枝版"""
    if board.game_over():
        return board.evaluate()

    if board.side == X:
        best = -999
        for m in board.moves():
            board.apply(m)
            score = minimax_ab(board, alpha, beta)
            board.undo(m)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best
    else:
        best = 999
        for m in board.moves():
            board.apply(m)
            score = minimax_ab(board, alpha, beta)
            board.undo(m)
            best = min(best, score)
            beta = min(beta, score)
            if alpha >= beta:
                break
        return best


# ─── 阶段 3：Minimax + Alpha-Beta + TT ───

class TTEntry:
    __slots__ = ('key', 'depth', 'score', 'flag', 'best_move')

    def __init__(self, key, depth, score, flag, best_move=None):
        self.key = key
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

class TranspositionTable:
    def __init__(self, max_size=50000):
        self.table = {}
        self.max_size = max_size
        self.hits = 0

    def hash_board(self, board):
        """简单的棋盘哈希（非 Zobrist，用于演示）"""
        h = 0
        for i, c in enumerate(board.cells):
            h ^= (c + 1) << (i * 2)
        h ^= board.side << 18  # 行棋方
        return h

    def probe(self, key, depth, alpha, beta):
        entry = self.table.get(key)
        if not entry or entry.key != key:
            return False, None, None
        self.hits += 1
        if entry.depth >= depth:
            if entry.flag == TT_EXACT:
                return True, entry.score, entry.best_move
            if entry.flag == TT_LOWER and entry.score >= beta:
                return True, entry.score, entry.best_move
            if entry.flag == TT_UPPER and entry.score <= alpha:
                return True, entry.score, entry.best_move
        return False, None, entry.best_move

    def save(self, key, depth, score, flag, best_move=None):
        if len(self.table) >= self.max_size:
            self.table.clear()
        self.table[key] = TTEntry(key, depth, score, flag, best_move)


def minimax_ab_tt(board, alpha, beta, tt, depth=0):
    """Alpha-Beta + TT"""
    key = tt.hash_board(board)

    found, score, best_move = tt.probe(key, depth, alpha, beta)
    if found:
        return score

    if board.game_over():
        s = board.evaluate()
        tt.save(key, depth, s, TT_EXACT)
        return s

    moves = board.moves()
    # TT best move 优先
    if best_move and best_move in moves:
        moves.remove(best_move)
        moves.insert(0, best_move)

    if board.side == X:
        best = -999
        for m in moves:
            board.apply(m)
            score = minimax_ab_tt(board, alpha, beta, tt, depth + 1)
            board.undo(m)
            if score > best:
                best = score
                best_move = m
            alpha = max(alpha, score)
            if alpha >= beta:
                tt.save(key, depth, best, TT_LOWER, best_move)
                return best
        tt.save(key, depth, best, TT_EXACT, best_move)
        return best
    else:
        best = 999
        for m in moves:
            board.apply(m)
            score = minimax_ab_tt(board, alpha, beta, tt, depth + 1)
            board.undo(m)
            if score < best:
                best = score
                best_move = m
            beta = min(beta, score)
            if alpha >= beta:
                tt.save(key, depth, best, TT_UPPER, best_move)
                return best
        tt.save(key, depth, best, TT_EXACT, best_move)
        return best


# ─── 阶段 4：完整带 QSearch 的 Negamax ───
# 井字棋没有吃子，QSearch 在这只是概念演示

def quiescence_demo(board, alpha, beta):
    """
    井字棋的 QSearch 演示。
    实际井字棋不需要 QSearch（没有吃子），
    但这里展示概念：评估"静止"局面。
    """
    stand_pat = board.evaluate()

    # 如果有立即获胜的走法，能超过 stand_pat
    for m in board.moves():
        board.apply(m)
        if board.is_win(board.side == O and X or O):  # 检查对手是否获胜
            board.undo(m)
            continue
        # 检查自己是否可以通过这一步获胜
        if board.is_win(board.side == X and X or O):
            score = 200
            board.undo(m)
            return score if board.side == X else -score
        board.undo(m)

    # 标准 QSearch 逻辑
    if board.side == X:
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
        best = stand_pat
        for m in board.moves():
            board.apply(m)
            if board.is_win(X):
                score = 200
            else:
                score = -quiescence_demo(board, -beta, -alpha)
            board.undo(m)
            if score > best:
                best = score
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best
    else:
        if stand_pat <= alpha: return alpha
        beta = min(beta, stand_pat)
        best = stand_pat
        for m in board.moves():
            board.apply(m)
            if board.is_win(O):
                score = -200
            else:
                score = -quiescence_demo(board, -beta, -alpha)
            board.undo(m)
            if score < best:
                best = score
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best


# ─── 性能对比 ───

def benchmark():
    """对比各阶段性能"""
    print("=" * 60)
    print("Demo 2a: 井字棋 AI 各阶段性能对比")
    print("=" * 60)

    # 在初始局面下搜索，重复多次
    trials = 100

    # 纯 Minimax
    t0 = time.time()
    for _ in range(trials):
        b = Board()
        b.side = X
        minimax(b)
    t1 = time.time()

    # Alpha-Beta
    for _ in range(trials):
        b = Board()
        b.side = X
        minimax_ab(b, -999, 999)
    t2 = time.time()

    # Alpha-Beta + TT
    tt = TranspositionTable()
    for _ in range(trials):
        b = Board()
        b.side = X
        minimax_ab_tt(b, -999, 999, tt)
    t3 = time.time()

    print(f"\n{'算法':<25} {'耗时':>10} {'加速比':>10}")
    print("-" * 47)
    print(f"{'纯 Minimax':<25} {t1 - t0:>10.3f}s {'1.0x':>10}")
    print(f"{'Alpha-Beta':<25} {t2 - t1:>10.3f}s {(t1-t0)/(max(0.001,t2-t1)):>9.1f}x")
    print(f"{'Alpha-Beta + TT':<25} {t3 - t2:>10.3f}s {(t1-t0)/(max(0.001,t3-t2)):>9.1f}x")
    print(f"\n{'TT 命中率':<25} {tt.hits / (tt.hits + len(tt.table)) * 100 if (tt.hits + len(tt.table)) > 0 else 0:>9.1f}%")
    print()


def interactive_demo():
    """人机对战演示"""
    print("=" * 60)
    print("人机对战（AI 使用 Alpha-Beta + TT）")
    print("=" * 60)
    print("你走 O，AI 走 X")
    print("输入位置编号：")
    print("0 1 2")
    print("3 4 5")
    print("6 7 8")
    print()

    board = Board()
    tt = TranspositionTable()

    while not board.game_over():
        board.print()
        print()

        if board.side == X:
            # AI 走
            best_move = None
            best_score = -999
            for m in board.moves():
                board.apply(m)
                score = minimax_ab_tt(board, -999, 999, tt)
                board.undo(m)
                if score > best_score:
                    best_score = score
                    best_move = m
            board.apply(best_move)
            print(f"AI 走: {best_move}")
        else:
            # 人类走
            move = int(input("你的走法: "))
            if move in board.moves():
                board.apply(move)
            else:
                print("无效走法！")
                continue

    board.print()
    if board.is_win(X):
        print("AI 胜！")
    elif board.is_win(O):
        print("你胜！")
    else:
        print("平局！")


if __name__ == "__main__":
    benchmark()
    # interactive_demo()  # 取消注释可玩
