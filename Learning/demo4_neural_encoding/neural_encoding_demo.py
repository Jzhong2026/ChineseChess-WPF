#!/usr/bin/env python3
"""
Demo 4: Chinese Chess neural encoding pipeline.

This mirrors the current project pipeline:
  - BoardEncoder.Encode(): float[1260], square-major layout
  - XiangqiNeuralAiService.EncodeCnn(): [1, 16, 10, 9], plane-major layout
  - MoveEncoder.Encode(): action id in [0, 8099]
  - Factored policy head: combined[from * 90 + to] = from_logit + to_logit

Run:
    python Learning/demo4_neural_encoding/neural_encoding_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


ROWS = 10
COLS = 9
BOARD_SIZE = ROWS * COLS

RED = 0
BLACK = 1

GENERAL = 0
ADVISOR = 1
ELEPHANT = 2
HORSE = 3
ROOK = 4
CANNON = 5
SOLDIER = 6

PIECE_PLANES = 14
SIDE_PLANES = 2
CNN_PLANES = PIECE_PLANES + SIDE_PLANES
ENCODED_LENGTH = BOARD_SIZE * PIECE_PLANES
ACTION_COUNT = BOARD_SIZE * BOARD_SIZE

SIDE_NAMES = {RED: "Red", BLACK: "Black"}
PIECE_NAMES = {
    GENERAL: "General",
    ADVISOR: "Advisor",
    ELEPHANT: "Elephant",
    HORSE: "Horse",
    ROOK: "Rook",
    CANNON: "Cannon",
    SOLDIER: "Soldier",
}


@dataclass(frozen=True)
class Piece:
    side: int
    kind: int


@dataclass(frozen=True)
class Move:
    from_row: int
    from_col: int
    to_row: int
    to_col: int


Board = list[list[Piece | None]]


def square_index(row: int, col: int) -> int:
    validate_square(row, col)
    return row * COLS + col


def position(square: int) -> tuple[int, int]:
    if not 0 <= square < BOARD_SIZE:
        raise ValueError(f"square must be in [0, {BOARD_SIZE - 1}]")
    return square // COLS, square % COLS


def validate_square(row: int, col: int) -> None:
    if not 0 <= row < ROWS:
        raise ValueError(f"row must be in [0, {ROWS - 1}]")
    if not 0 <= col < COLS:
        raise ValueError(f"col must be in [0, {COLS - 1}]")


def board_plane(side: int, kind: int) -> int:
    # Matches C#: plane = (int)side * 7 + (int)type
    return side * 7 + kind


def get_board_index(row: int, col: int, side: int, kind: int) -> int:
    # Matches C#: index = (row * 9 + col) * 14 + plane
    return square_index(row, col) * PIECE_PLANES + board_plane(side, kind)


def encode_board_square_major(board: Board) -> list[float]:
    encoded = [0.0] * ENCODED_LENGTH
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece is None:
                continue
            encoded[get_board_index(row, col, piece.side, piece.kind)] = 1.0
    return encoded


def pack_cnn_input(flat: list[float], side_to_move: int) -> list[list[list[float]]]:
    """
    Convert square-major [90 * 14] into plane-major [16][10][9].

    Planes 0..13: pieces
    Plane 14: all ones if Red to move
    Plane 15: all ones if Black to move
    """
    if len(flat) != ENCODED_LENGTH:
        raise ValueError(f"flat board must have length {ENCODED_LENGTH}")

    tensor = [[[0.0 for _ in range(COLS)] for _ in range(ROWS)] for _ in range(CNN_PLANES)]

    for row in range(ROWS):
        for col in range(COLS):
            for plane in range(PIECE_PLANES):
                flat_index = (row * COLS + col) * PIECE_PLANES + plane
                tensor[plane][row][col] = flat[flat_index]

    red_value = 1.0 if side_to_move == RED else 0.0
    black_value = 1.0 if side_to_move == BLACK else 0.0
    for row in range(ROWS):
        for col in range(COLS):
            tensor[14][row][col] = red_value
            tensor[15][row][col] = black_value

    return tensor


def encode_move(move: Move) -> int:
    from_index = square_index(move.from_row, move.from_col)
    to_index = square_index(move.to_row, move.to_col)
    return from_index * BOARD_SIZE + to_index


def decode_move(action_id: int) -> Move:
    if not 0 <= action_id < ACTION_COUNT:
        raise ValueError(f"action_id must be in [0, {ACTION_COUNT - 1}]")
    from_index = action_id // BOARD_SIZE
    to_index = action_id % BOARD_SIZE
    from_row, from_col = position(from_index)
    to_row, to_col = position(to_index)
    return Move(from_row, from_col, to_row, to_col)


def combine_factored_logits(from_logits: list[float], to_logits: list[float]) -> list[float]:
    """
    Matches XiangqiNeuralAiService.RunInference() for factored models.
    """
    if len(from_logits) != BOARD_SIZE or len(to_logits) != BOARD_SIZE:
        raise ValueError("from_logits and to_logits must both have length 90")

    combined = [0.0] * ACTION_COUNT
    for from_index in range(BOARD_SIZE):
        for to_index in range(BOARD_SIZE):
            combined[from_index * BOARD_SIZE + to_index] = from_logits[from_index] + to_logits[to_index]
    return combined


def choose_best_legal_move(policy_logits: list[float], legal_moves: Iterable[Move]) -> tuple[Move, float]:
    best_move: Move | None = None
    best_logit = float("-inf")

    for move in legal_moves:
        action_id = encode_move(move)
        logit = policy_logits[action_id]
        if logit > best_logit:
            best_move = move
            best_logit = logit

    if best_move is None:
        raise ValueError("legal_moves must not be empty")

    return best_move, best_logit


def flip_col(col: int) -> int:
    return COLS - 1 - col


def flip_move_horizontal(move: Move) -> Move:
    return Move(move.from_row, flip_col(move.from_col), move.to_row, flip_col(move.to_col))


def flip_action_horizontal(action_id: int) -> int:
    return encode_move(flip_move_horizontal(decode_move(action_id)))


def flip_board_horizontal(board: Board) -> Board:
    return [[board[row][COLS - 1 - col] for col in range(COLS)] for row in range(ROWS)]


def initial_board() -> Board:
    board: Board = [[None for _ in range(COLS)] for _ in range(ROWS)]

    # A small but real-looking initial position. It is enough to verify indices.
    board[0][0] = Piece(BLACK, ROOK)
    board[0][1] = Piece(BLACK, HORSE)
    board[0][4] = Piece(BLACK, GENERAL)
    board[2][1] = Piece(BLACK, CANNON)
    board[3][0] = Piece(BLACK, SOLDIER)

    board[9][0] = Piece(RED, ROOK)
    board[9][1] = Piece(RED, HORSE)
    board[9][4] = Piece(RED, GENERAL)
    board[7][1] = Piece(RED, CANNON)
    board[6][0] = Piece(RED, SOLDIER)

    return board


def nonzero_piece_features(flat: list[float]) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for index, value in enumerate(flat):
        if value == 0.0:
            continue
        square = index // PIECE_PLANES
        plane = index % PIECE_PLANES
        row, col = position(square)
        result.append((row, col, plane))
    return result


def format_move(move: Move) -> str:
    return f"({move.from_row},{move.from_col}) -> ({move.to_row},{move.to_col})"


def assert_demo_invariants() -> None:
    board = initial_board()
    flat = encode_board_square_major(board)
    cnn = pack_cnn_input(flat, RED)

    assert len(flat) == 1260
    assert len(cnn) == 16
    assert len(cnn[0]) == 10
    assert len(cnn[0][0]) == 9

    red_general_index = get_board_index(9, 4, RED, GENERAL)
    black_rook_index = get_board_index(0, 0, BLACK, ROOK)
    assert flat[red_general_index] == 1.0
    assert flat[black_rook_index] == 1.0
    assert cnn[board_plane(RED, GENERAL)][9][4] == 1.0
    assert cnn[14][0][0] == 1.0
    assert cnn[15][0][0] == 0.0

    move = Move(9, 0, 8, 0)
    assert decode_move(encode_move(move)) == move
    assert flip_action_horizontal(encode_move(move)) == encode_move(Move(9, 8, 8, 8))


def demo_board_encoding() -> None:
    print("=" * 72)
    print("1) BoardEncoder: square-major float[1260]")
    print("=" * 72)

    board = initial_board()
    flat = encode_board_square_major(board)
    features = nonzero_piece_features(flat)

    print(f"Encoded length: {len(flat)}")
    print(f"Non-zero piece features: {len(features)}")
    print("First few encoded pieces:")
    for row, col, plane in features[:6]:
        side = RED if plane < 7 else BLACK
        kind = plane % 7
        print(f"  row={row}, col={col}, plane={plane:2d} -> {SIDE_NAMES[side]} {PIECE_NAMES[kind]}")

    red_general_index = get_board_index(9, 4, RED, GENERAL)
    print(f"Red General at (9,4) flat index: {red_general_index}")
    print()


def demo_cnn_packing() -> None:
    print("=" * 72)
    print("2) CNN/ONNX input: [1, 16, 10, 9]")
    print("=" * 72)

    flat = encode_board_square_major(initial_board())
    cnn_red = pack_cnn_input(flat, RED)
    cnn_black = pack_cnn_input(flat, BLACK)

    red_general_plane = board_plane(RED, GENERAL)
    black_rook_plane = board_plane(BLACK, ROOK)
    print(f"Plane {red_general_plane} at [9][4] = {cnn_red[red_general_plane][9][4]}")
    print(f"Plane {black_rook_plane} at [0][0] = {cnn_red[black_rook_plane][0][0]}")
    print(f"Red-to-move planes:   plane14={cnn_red[14][0][0]}, plane15={cnn_red[15][0][0]}")
    print(f"Black-to-move planes: plane14={cnn_black[14][0][0]}, plane15={cnn_black[15][0][0]}")
    print()


def demo_move_encoding_and_policy() -> None:
    print("=" * 72)
    print("3) MoveEncoder + factored policy head")
    print("=" * 72)

    legal_moves = [
        Move(9, 0, 8, 0),
        Move(9, 1, 7, 2),
        Move(7, 1, 0, 1),
    ]

    for move in legal_moves:
        print(f"{format_move(move):18s} -> action_id={encode_move(move)}")

    # Fake model outputs. The legal move with the best from+to score should win.
    from_logits = [0.0] * BOARD_SIZE
    to_logits = [0.0] * BOARD_SIZE
    from_logits[square_index(9, 0)] = 1.0
    to_logits[square_index(8, 0)] = 0.5
    from_logits[square_index(9, 1)] = 0.2
    to_logits[square_index(7, 2)] = 0.2
    from_logits[square_index(7, 1)] = 1.3
    to_logits[square_index(0, 1)] = 1.4

    policy_logits = combine_factored_logits(from_logits, to_logits)
    best_move, best_logit = choose_best_legal_move(policy_logits, legal_moves)

    print(f"Best legal move: {format_move(best_move)} with logit {best_logit:.2f}")
    print("Important: choose from legal moves only; never argmax over all 8100 actions directly.")
    print()


def demo_horizontal_flip() -> None:
    print("=" * 72)
    print("4) Horizontal flip data augmentation")
    print("=" * 72)

    move = Move(9, 0, 8, 0)
    action = encode_move(move)
    flipped = flip_move_horizontal(move)
    flipped_action = flip_action_horizontal(action)

    print(f"Original move: {format_move(move)} -> {action}")
    print(f"Flipped move:  {format_move(flipped)} -> {flipped_action}")

    board = initial_board()
    flipped_board = flip_board_horizontal(board)
    assert flipped_board[9][8] == board[9][0]
    print("Board piece at (9,0) moves to (9,8) after flip.")
    print()


def main() -> None:
    assert_demo_invariants()
    demo_board_encoding()
    demo_cnn_packing()
    demo_move_encoding_and_policy()
    demo_horizontal_flip()
    print("All neural encoding checks passed.")


if __name__ == "__main__":
    main()

