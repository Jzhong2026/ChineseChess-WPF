"""
Demo 1a: Quiescence Search 基础概念
===================================
纯 Python 实现，< 100 行，展示 QSearch 如何解决水平线效应。
使用简化棋盘（5x5 只含兵/车/马/王），只搜索吃子走法。
"""

import random

# 棋子类型
EMPTY, PAWN, ROOK, KNIGHT, KING = 0, 1, 2, 3, 4

# 简化棋盘 5x5
ROWS, COLS = 5, 5

# 棋子价值
PIECE_VALUE = {
    PAWN: 100,
    ROOK: 500,
    KNIGHT: 300,
    KING: 10000,
}

def evaluate(board):
    """简单评估函数：红方总和 - 黑方总和"""
    score = 0
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece > 0:  # 红方
                score += PIECE_VALUE.get(piece, 0)
            elif piece < 0:  # 黑方
                score -= PIECE_VALUE.get(-piece, 0)
    return score


def generate_captures(board, is_red):
    """生成所有吃子走法（简化版：只能吃相邻格子的棋子）"""
    captures = []
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if (is_red and piece > 0) or (not is_red and piece < 0):
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0),
                               (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        target = board[nr][nc]
                        if target != 0:
                            enemy = target < 0 if is_red else target > 0
                            if enemy:
                                captures.append((r, c, nr, nc, abs(target)))
    return captures


# ─── 没有 QSearch 的 Alpha-Beta ───
def alpha_beta_no_q(board, depth, alpha, beta, is_red):
    """固定深度搜索——会受水平线效应影响"""
    if depth == 0:
        return evaluate(board)

    captures = generate_captures(board, is_red)
    if not captures:
        return evaluate(board)

    if is_red:
        best = -999999
        for r, c, nr, nc, _ in captures:
            captured = board[nr][nc]
            board[nr][nc] = board[r][c]
            board[r][c] = 0
            score = alpha_beta_no_q(board, depth - 1, alpha, beta, False)
            board[r][c] = board[nr][nc]
            board[nr][nc] = captured
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best
    else:
        best = 999999
        for r, c, nr, nc, _ in captures:
            captured = board[nr][nc]
            board[nr][nc] = board[r][c]
            board[r][c] = 0
            score = alpha_beta_no_q(board, depth - 1, alpha, beta, True)
            board[r][c] = board[nr][nc]
            board[nr][nc] = captured
            best = min(best, score)
            beta = min(beta, score)
            if alpha >= beta:
                break
        return best


# ─── 带 QSearch 的 Alpha-Beta ───
def quiescence(board, alpha, beta, is_red):
    """
    静态搜索：只搜吃子走法，直到没有吃子可走为止。
    解决水平线效应：即使搜索到底层，也能继续"看到"吃子序列的最终结果。
    """
    stand_pat = evaluate(board)

    if is_red:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= beta:
            return beta  # 注意：黑方视角，beta 是下界
        # 修正：黑方视角
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)

    captures = generate_captures(board, is_red)
    if not captures:
        return stand_pat

    if is_red:
        best = stand_pat
        for r, c, nr, nc, _ in captures:
            captured = board[nr][nc]
            board[nr][nc] = board[r][c]
            board[r][c] = 0
            score = quiescence(board, alpha, beta, False)
            board[r][c] = board[nr][nc]
            board[nr][nc] = captured
            if score > best:
                best = score
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best
    else:
        best = stand_pat
        for r, c, nr, nc, _ in captures:
            captured = board[nr][nc]
            board[nr][nc] = board[r][c]
            board[r][c] = 0
            score = quiescence(board, alpha, beta, True)
            board[r][c] = board[nr][nc]
            board[nr][nc] = captured
            if score < best:
                best = score
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best


def alpha_beta_with_q(board, depth, alpha, beta, is_red):
    """带 QSearch 截断的 Alpha-Beta"""
    if depth == 0:
        # ★ 关键：到深度后进入 QSearch，继续搜吃子
        return quiescence(board, alpha, beta, is_red)

    captures = generate_captures(board, is_red)
    if not captures:
        return evaluate(board)

    if is_red:
        best = -999999
        for r, c, nr, nc, _ in captures:
            captured = board[nr][nc]
            board[nr][nc] = board[r][c]
            board[r][c] = 0
            score = alpha_beta_with_q(board, depth - 1, alpha, beta, False)
            board[r][c] = board[nr][nc]
            board[nr][nc] = captured
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best
    else:
        best = 999999
        for r, c, nr, nc, _ in captures:
            captured = board[nr][nc]
            board[nr][nc] = board[r][c]
            board[r][c] = 0
            score = alpha_beta_with_q(board, depth - 1, alpha, beta, True)
            board[r][c] = board[nr][nc]
            board[nr][nc] = captured
            best = min(best, score)
            beta = min(beta, score)
            if alpha >= beta:
                break
        return best


# ─── 演示：水平线效应 ───
def demo_horizon_effect():
    """
    场景：红方一个车正在攻击黑方的马。
    黑方有一个兵可以吃掉红方的车。
    如果搜索深度不够，AI 认为"局面很好"（多一个车），
    但实际走完黑方会吃掉车——这个吃子刚好在搜索深度之外。
    """
    board = [[0] * COLS for _ in range(ROWS)]
    # 红方车在 (0, 0)
    board[0][0] = ROOK
    # 黑方马在 (1, 1)，被红车攻击
    board[1][1] = -KNIGHT
    # 黑方兵在 (2, 2)，能走到 (1, 1) 吗？简化场景
    # 再放一个黑方兵在 (1, 0)，可以吃红车
    board[1][0] = -PAWN

    print("=" * 60)
    print("Demo: 水平线效应 (Horizon Effect)")
    print("=" * 60)
    print("盘面：红车@(0,0), 黑马@(1,1), 黑兵@(1,0)")
    print("黑方可以牺牲马？不，黑兵可以直接吃红车！")
    print()

    depth = 2
    score_no_q = alpha_beta_no_q(board, depth, -999999, 999999, True)
    score_with_q = alpha_beta_with_q(board, depth, -999999, 999999, True)

    print(f"搜索深度: {depth}")
    print(f"无 QSearch 估值: {score_no_q}")
    print(f"   → AI 认为红方优势（车值 500），看不到黑兵吃车")
    print()
    print(f"有 QSearch 估值: {score_with_q}")
    print(f"   → QSearch 继续搜吃子，发现黑兵吃车，估值更低（更准确）")
    print()

    # 验证差异
    delta = score_no_q - score_with_q
    print(f"估值差异: {delta} 分")
    print(f"{'✅ QSearch 有效修正了水平线效应!' if delta > 0 else '⚠️ 未见明显差异'}")
    print()


def demo_forced_capture_chain():
    """
    场景：连续吃子的战术组合（弃子攻杀）。
    红方连续弃两个兵，最终吃掉黑方一个车。
    """
    board = [[0] * COLS for _ in range(ROWS)]
    board[0][0] = PAWN  # 红兵
    board[1][1] = -PAWN  # 黑兵
    board[2][2] = PAWN  # 红兵
    board[3][3] = -ROOK  # 黑车

    print("=" * 60)
    print("Demo: 连续吃子链 (Forced Capture Chain)")
    print("=" * 60)
    print("盘面：红兵@(0,0) > 黑兵@(1,1) > 红兵@(2,2) > 黑车@(3,3)")
    print()

    depth = 1  # 很浅的搜索，但 QSearch 能继续深入
    score_no_q = alpha_beta_no_q(board, depth, -999999, 999999, True)
    score_with_q = alpha_beta_with_q(board, depth, -999999, 999999, True)

    print(f"搜索深度: {depth}")
    print(f"无 QSearch: {score_no_q} (红方只能看到一步)")
    print(f"有 QSearch: {score_with_q} (红方看到连续吃子链)")
    print()

    delta = score_with_q - score_no_q
    verdict = (
        "✅ QSearch 发现了更深的吃子组合，评分更高！"
        if delta > 0
        else "⚠️ 本场景未见差异"
    )
    print(f"估值差异: {delta}")
    print(verdict)
    print()


if __name__ == "__main__":
    demo_horizon_effect()
    demo_forced_capture_chain()
