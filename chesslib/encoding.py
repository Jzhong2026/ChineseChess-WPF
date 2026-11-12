"""Board/action encoding shared by all trainers.

Contract truth source (C#): ChineseChess.Core/Encoding/BoardEncoder.cs,
MoveEncoder.cs and Services/XiangqiNeuralAiService.cs (EncodeCnn).
Deliberately torch-free: numpy in, numpy out, callers convert to tensors.
"""
from __future__ import annotations

import numpy as np

BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
PIECE_PLANES = 14
SIDE_PLANES = 2
INPUT_PLANES = PIECE_PLANES + SIDE_PLANES
FLAT_DIM = BOARD_SIZE * PIECE_PLANES
ACTION_DIM = BOARD_SIZE * BOARD_SIZE
RED_TO_MOVE_PLANE = PIECE_PLANES
BLACK_TO_MOVE_PLANE = PIECE_PLANES + 1


def encode_board(flat, side_to_move) -> np.ndarray:
    """square-major flat[1260] -> plane-major float32 [16, 10, 9].

    flat layout is BoardEncoder.GetIndex: (row*9+col)*14 + side*7 + piece_type,
    with Side.Red=0 / Side.Black=1. side_to_move follows the JSONL convention
    (1=Red, -1=Black); anything else is treated as Black, as the trainers did.
    """
    values = np.asarray(flat, dtype=np.float32)
    if values.size != FLAT_DIM:
        raise ValueError(f"BoardEncoding length {values.size} != {FLAT_DIM}")

    features = np.zeros((INPUT_PLANES, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    features[:PIECE_PLANES] = values.reshape(BOARD_SIZE, PIECE_PLANES).T.reshape(
        PIECE_PLANES, BOARD_ROWS, BOARD_COLS)
    features[RED_TO_MOVE_PLANE if side_to_move == 1 else BLACK_TO_MOVE_PLANE] = 1.0
    return features


def flip_square(square: int) -> int:
    """Mirror one square index horizontally (col -> 8-col), row unchanged."""
    row, col = divmod(int(square), BOARD_COLS)
    return row * BOARD_COLS + (BOARD_COLS - 1 - col)


def remap_move(action: int) -> int:
    """Remap a from*90+to action id for the horizontal flip augmentation."""
    from_square, to_square = divmod(int(action), BOARD_SIZE)
    return flip_square(from_square) * BOARD_SIZE + flip_square(to_square)


def remap_legal_moves(legal_moves) -> list[int]:
    return [remap_move(move) for move in legal_moves]


def flip_features(features):
    """Mirror the column axis of [..., 10, 9] features.

    Accepts ndarray/list, or any tensor-like exposing .flip(dim) (torch), so the
    augmentation stays a single shared code path without importing torch here.
    """
    flip = getattr(features, "flip", None)
    if flip is not None:
        return flip(-1)
    return np.flip(np.asarray(features), axis=-1)
