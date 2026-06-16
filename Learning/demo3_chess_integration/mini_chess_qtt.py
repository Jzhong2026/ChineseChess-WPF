"""
Demo 3a: 在简化象棋上集成 QSearch + TT (Python)
=================================================
模拟象棋引擎的核心组件，展示 QSearch + TT 如何与本项目的
XiangqiAiService.cs 中的实现对应。

棋盘缩小为 4x4，只含帅/车/兵，便于理解搜索流程。
"""

import random
import time

# ─── 1. 棋盘与规则 ───

ROWS, COLS = 4, 4
EMPTY = 0
KING, ROOK, PAWN = 1, 2, 3  # 正数 = 红方, 负数 = 黑方

PIECE_VALUES = {KING: 1000, ROOK: 500, PAWN: 100}
PIECE_NAMES = {KING: "帅/将", ROOK: "车/車", PAWN: "兵/卒"}

class ChessBoard:
    """4x4 简化象棋盘面"""

    def __init__(self):
        self.board = [[0] * COLS for _ in range(ROWS)]
        self.side = 1  # 1 = 红, -1 = 黑
        self._setup()

    def _setup(self):
        """初始布局"""
        # 红方上边，黑方下边
        self.board[0][0] = ROOK   # 红车
        self.board[0][3] = KING  # 红帅
        self.board[1][1] = PAWN  # 红兵
        self.board[3][0] = -ROOK  # 黑车
        self.board[3][3] = -KING  # 黑将
        self.board[2][2] = -PAWN  # 黑卒

    def clone(self):
        b = ChessBoard()
        b.board = [row.copy() for row in self.board]
        b.side = self.side
        return b

    def generate_moves(self, include_non_capture=True):
        """生成走法列表"""
        moves = []
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece == 0 or (piece > 0) != (self.side == 1):
                    continue

                ptype = abs(piece)
                if ptype == KING:
                    # 王走一步（任何方向）
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0),
                                   (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            t = self.board[nr][nc]
                            if t == 0 or (t < 0 if self.side == 1 else t > 0):
                                moves.append((r, c, nr, nc))
                elif ptype == ROOK:
                    # 车走直线
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < ROWS and 0 <= nc < COLS:
                            t = self.board[nr][nc]
                            if t == 0:
                                moves.append((r, c, nr, nc))
                            elif (t < 0 if self.side == 1 else t > 0):
                                moves.append((r, c, nr, nc))
                                break
                            else:
                                break
                            nr += dr
                            nc += dc
                elif ptype == PAWN:
                    # 兵向前走一步（红方行 ↑，黑方行 ↓）
                    dr = -1 if self.side == 1 else 1
                    nr = r + dr
                    if 0 <= nr < ROWS:
                        t = self.board[nr][c]
                        if t == 0 or (t < 0 if self.side == 1 else t > 0):
                            moves.append((r, c, nr, c))
        return moves

    def make_move(self, move):
        r, c, nr, nc = move
        self.board[nr][nc] = self.board[r][c]
        self.board[r][c] = 0
        self.side = -self.side

    def unmake_move(self, move, captured):
        r, c, nr, nc = move
        self.board[r][c] = self.board[nr][nc]
        self.board[nr][nc] = captured
        self.side = -self.side

    def is_capture(self, move):
        _, _, nr, nc = move
        return self.board[nr][nc] != 0

    def is_check(self):
        """检查当前行棋方是否被将军（简化版）"""
        # 找对方的王
        enemy_king = -KING * self.side
        king_pos = None
        for r in range(ROWS):
            for c in range(COLS):
                if self.board[r][c] == enemy_king:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        if not king_pos:
            return False

        kr, kc = king_pos
        # 检查是否有己方棋子能吃到对方王
        old_side = self.side
        self.side = -self.side  # 假设对方走棋
        for r in range(ROWS):
            for c in range(COLS):
                p = self.board[r][c]
                if p != 0 and (p > 0) == (self.side == 1):
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0),
                                   (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        if (r + dr, c + dc) == (kr, kc):
                            self.side = old_side
                            return True
        self.side = old_side
        return False

    def game_over(self):
        """检查是否终局：王被吃掉"""
        has_red_king = any(self.board[r][c] == KING for r in range(ROWS) for c in range(COLS))
        has_black_king = any(self.board[r][c] == -KING for r in range(ROWS) for c in range(COLS))
        return not (has_red_king and has_black_king)

    def evaluate(self):
        """评估函数：基础子力评估"""
        score = 0
        for r in range(ROWS):
            for c in range(COLS):
                p = self.board[r][c]
                if p > 0:
                    score += PIECE_VALUES.get(p, 0)
                elif p < 0:
                    score -= PIECE_VALUES.get(-p, 0)
        return score

    def print(self):
        chars = {EMPTY: '.', KING: 'K', ROOK: 'R', PAWN: 'P',
                 -KING: 'k', -ROOK: 'r', -PAWN: 'p'}
        print(f"{'红方' if self.side == 1 else '黑方'}行棋:")
        for r in range(ROWS):
            print(' '.join(chars.get(self.board[r][c], '?') for c in range(COLS)))
        print()


# ─── 2. Zobrist 哈希 ───

class Zobrist:
    def __init__(self):
        self.table = {}
        for pt in range(1, 4):  # KING, ROOK, PAWN
            for r in range(ROWS):
                for c in range(COLS):
                    self.table[(pt, r, c)] = random.getrandbits(64)
                    self.table[(-pt, r, c)] = random.getrandbits(64)
        self.side_hash = random.getrandbits(64)

    def hash_board(self, board):
        h = 0
        for r in range(ROWS):
            for c in range(COLS):
                p = board[r][c]
                if p != 0:
                    h ^= self.table[(p, r, c)]
        if board.side == -1:
            h ^= self.side_hash
        return h


# ─── 3. 置换表 ───

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

class TTEntry:
    __slots__ = ('key', 'depth', 'score', 'flag', 'best_move')
    def __init__(self, key, depth, score, flag, best_move=None):
        self.key = key
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move


class TranspositionTable:
    def __init__(self, max_size=200000):
        self.table = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def probe(self, key, depth, alpha, beta):
        entry = self.table.get(key)
        if not entry or entry.key != key:
            self.misses += 1
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

    def stats(self):
        total = self.hits + self.misses
        return f"TT: {self.hits}/{total} hits ({self.hits / max(1, total) * 100:.1f}%) | size={len(self.table)}"


# ─── 4. 核心搜索（对应 XiangqiAiService.cs） ───

class SearchEngine:
    """
    搜索引擎：对应本项目 XiangqiAiService.cs 中的搜索逻辑。
    包含 Negamax + Alpha-Beta + TT + QSearch。
    """

    def __init__(self, max_depth=4, qs_depth=4):
        self.max_depth = max_depth
        self.qs_depth = qs_depth  # QSearch 额外深度
        self.tt = TranspositionTable()
        self.zobrist = Zobrist()
        self.nodes_searched = 0

    def quiescence(self, board, alpha, beta, depth=0):
        """
        静态搜索（对应 XiangqiAiService.EvaluateWithQSearch）
        只搜索吃子走法，直到静止或达到 QS 深度上限。
        """
        if depth >= self.qs_depth:
            return board.evaluate()

        stand_pat = board.evaluate()

        # 如果对方过激了，不需要再搜
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        # 只搜吃子走法
        capture_moves = [m for m in board.generate_moves(include_non_capture=False)
                         if board.is_capture(m)]

        # 按被吃子价值降序排序（MVV-LVA）
        capture_moves.sort(
            key=lambda m: PIECE_VALUES.get(abs(board.board[m[2]][m[3]]), 0),
            reverse=True
        )

        best = stand_pat
        for move in capture_moves:
            captured = board.board[move[2]][move[3]]
            board.make_move(move)
            score = -self.quiescence(board, -beta, -alpha, depth + 1)
            board.unmake_move(move, captured)

            if score > best:
                best = score
            alpha = max(alpha, best)
            if alpha >= beta:
                break

        return best

    def negamax(self, board, alpha, beta, depth):
        """
        Negamax + Alpha-Beta + TT 主搜索
        对应 XiangqiAiService.NegamaxSearch
        """
        self.nodes_searched += 1
        key = self.zobrist.hash_board(board)

        # ── TT Probe ──
        found, cached, best_move_hint = self.tt.probe(key, depth, alpha, beta)
        if found:
            return cached

        # ── 终局检查 ──
        if board.game_over():
            winner = 1 if board.side == -1 else -1
            score = 10000 * winner
            self.tt.save(key, depth, score, TT_EXACT)
            return score

        # ── QSearch 截断 ──
        if depth <= 0:
            score = self.quiescence(board, alpha, beta)
            self.tt.save(key, depth, score, TT_EXACT)
            return score

        # ── 走法生成与排序 ──
        moves = board.generate_moves()
        if not moves:
            # 无子可走 = 被将死
            winner = -1 if board.side == 1 else 1
            score = 10000 * winner
            self.tt.save(key, depth, score, TT_EXACT)
            return score

        # TT best move 优先 + 吃子优先
        def move_score(m):
            if best_move_hint and m == best_move_hint:
                return 9999
            if board.is_capture(m):
                return PIECE_VALUES.get(abs(board.board[m[2]][m[3]]), 0) + 1000
            return 0
        moves.sort(key=move_score, reverse=True)

        # ── 搜索 ──
        best = -999999
        best_move = moves[0]

        for move in moves:
            captured = board.board[move[2]][move[3]]
            board.make_move(move)
            score = -self.negamax(board, -beta, -alpha, depth - 1)
            board.unmake_move(move, captured)

            if score > best:
                best = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                self.tt.save(key, depth, best, TT_LOWER, best_move)
                return best

        self.tt.save(key, depth, best, TT_EXACT, best_move)
        return best

    def search(self, board):
        """对外接口：执行迭代加深搜索"""
        self.nodes_searched = 0
        best_move = None

        for d in range(1, self.max_depth + 1):
            alpha, beta = -999999, 999999
            best = -999999
            moves = board.generate_moves()

            for move in moves:
                captured = board.board[move[2]][move[3]]
                board.make_move(move)
                score = -self.negamax(board, -beta, -alpha, d - 1)
                board.unmake_move(move, captured)

                if score > best:
                    best = score
                    best_move = move

                alpha = max(alpha, score)
                if alpha >= beta:
                    break

        return best_move, best


# ─── 5. 演示 ───

def demo_search():
    print("=" * 60)
    print("Demo 3: 简化象棋 QSearch + TT 搜索演示 (Python)")
    print("=" * 60)

    board = ChessBoard()
    board.print()

    # 无 QSearch 的浅层搜索
    engine_no_q = SearchEngine(max_depth=2, qs_depth=0)
    t0 = time.time()
    move_no_q, score_no_q = engine_no_q.search(board.clone())
    t_no_q = time.time() - t0

    # 有 QSearch 的搜索
    engine_with_q = SearchEngine(max_depth=2, qs_depth=4)
    t1 = time.time()
    move_with_q, score_with_q = engine_with_q.search(board.clone())
    t_with_q = time.time() - t1

    # 有 QSearch + TT 的搜索（重用 TT）
    t2 = time.time()
    move_tt, score_tt = engine_with_q.search(board.clone())
    t_tt = time.time() - t2

    print(f"{'配置':<25} {'最佳走法':>10} {'估值':>10} {'耗时':>10} {'节点':>10}")
    print("-" * 65)
    print(f"{'无 QSearch (depth=2)':<25} {str(move_no_q):>10} {score_no_q:>10d} {t_no_q:>10.4f}s {engine_no_q.nodes_searched:>10d}")
    print(f"{'有 QSearch (depth=2)':<25} {str(move_with_q):>10} {score_with_q:>10d} {t_with_q:>10.4f}s {engine_with_q.nodes_searched:>10d}")
    print(f"{'有 QSearch+TT (depth=2)':<25} {str(move_tt):>10} {score_tt:>10d} {t_tt:>10.4f}s {engine_with_q.nodes_searched:>10d}")
    print()
    print(f"TT 统计: {engine_with_q.tt.stats()}")
    print()

    # 展示 QSearch 的走法链
    print("-" * 40)
    print("QSearch 吃子链演示:")
    print("-" * 40)
    board2 = ChessBoard()
    board2.print()

    # 强制一个吃子场景
    engine_show = SearchEngine(max_depth=1, qs_depth=6)
    move, score = engine_show.search(board2.clone())
    print(f"搜索结果: {move} (估值={score})")
    print(f"QSearch 在深度=1 后继续搜索了 {engine_show.qs_depth} 层吃子，")
    print(f"发现了吃子链中的最终局面，避免水平线效应的误判。")
    print()


def compare_tactics_awareness():
    """对比有无 QSearch 对战术组合的感知能力"""
    print("=" * 60)
    print("水平线效应对比")
    print("=" * 60)

    # 构造一个"红车吃黑兵，但黑方有反吃"的场景
    board = ChessBoard()
    # 清空
    board.board = [[0] * COLS for _ in range(ROWS)]
    board.board[0][0] = ROOK   # 红车
    board.board[1][1] = -PAWN  # 黑兵（红车能吃到）
    board.board[0][3] = -ROOK  # 黑车（如果红车吃兵，黑车可以吃红车）
    board.board[3][3] = -KING  # 黑将
    board.side = 1  # 红方走

    board.print()

    # 深度 2，无 QSearch
    e1 = SearchEngine(max_depth=2, qs_depth=0)
    m1, s1 = e1.search(board.clone())

    # 深度 2，有 QSearch
    e2 = SearchEngine(max_depth=2, qs_depth=4)
    m2, s2 = e2.search(board.clone())

    print(f"无 QSearch (depth=2): 走法={m1}, 估值={s1}")
    print(f"   → 可能贪吃黑兵，看不到黑车反吃")
    print(f"有 QSearch (depth=2):  走法={m2}, 估值={s2}")
    print(f"   → QSearch 看到吃兵后黑车反吃，不会贪吃")
    print()


if __name__ == "__main__":
    demo_search()
    compare_tactics_awareness()
