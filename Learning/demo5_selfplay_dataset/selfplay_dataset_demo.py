#!/usr/bin/env python3
"""
Demo 5: Self-play JSONL row -> neural training sample.

This mirrors:
  - ChineseChess.SelfPlay/Program.cs
  - train_cnn_policy_value.py

Run:
    python Learning/demo5_selfplay_dataset/selfplay_dataset_demo.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any


ROWS = 10
COLS = 9
BOARD_SIZE = ROWS * COLS
BOARD_PLANES = 14
INPUT_DIM = ROWS * COLS * BOARD_PLANES
ACTION_DIM = BOARD_SIZE * BOARD_SIZE

RED_TO_MOVE = 1
BLACK_TO_MOVE = -1


@dataclass(frozen=True)
class SelfPlayRow:
    game_id: int
    move_index: int
    side: str
    side_to_move: int
    board_encoding: list[float]
    legal_moves: list[int]
    selected_move: int
    result: int
    search_score_side_perspective: float
    search_score_red_perspective: float
    depth_reached: int
    nodes: int
    time_ms: float
    value_weight: float
    policy_weight: float
    use_for_value_training: bool
    use_for_policy_training: bool
    unfinished: bool
    end_reason: str


def square(row: int, col: int) -> int:
    if not 0 <= row < ROWS:
        raise ValueError("row out of range")
    if not 0 <= col < COLS:
        raise ValueError("col out of range")
    return row * COLS + col


def encode_move(from_row: int, from_col: int, to_row: int, to_col: int) -> int:
    return square(from_row, from_col) * BOARD_SIZE + square(to_row, to_col)


def decode_move(action_id: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if not 0 <= action_id < ACTION_DIM:
        raise ValueError("action id out of range")

    from_index = action_id // BOARD_SIZE
    to_index = action_id % BOARD_SIZE
    return (
        (from_index // COLS, from_index % COLS),
        (to_index // COLS, to_index % COLS),
    )


def sample_board_encoding() -> list[float]:
    """
    Minimal board encoding with a few pieces.

    Layout matches BoardEncoder:
        flat[(row * 9 + col) * 14 + plane] = 1
    """
    flat = [0.0] * INPUT_DIM

    # Planes: Red 0..6, Black 7..13.
    red_general_plane = 0
    red_rook_plane = 4
    black_general_plane = 7
    black_rook_plane = 11

    flat[(9 * COLS + 4) * BOARD_PLANES + red_general_plane] = 1.0
    flat[(9 * COLS + 0) * BOARD_PLANES + red_rook_plane] = 1.0
    flat[(0 * COLS + 4) * BOARD_PLANES + black_general_plane] = 1.0
    flat[(0 * COLS + 0) * BOARD_PLANES + black_rook_plane] = 1.0
    return flat


def get_value_weight(result: int, unfinished: bool, end_reason: str) -> float:
    if unfinished:
        return 0.0

    if end_reason in ("RedWins", "BlackWins", "Draw"):
        return 1.0
    if end_reason == "material-adjudication":
        return 0.75
    if end_reason == "max-moves-material-adjudication":
        return 0.5
    return 0.25 if result == 0 else 0.5


def get_policy_weight(depth_reached: int) -> float:
    if depth_reached <= 0:
        return 0.1
    if depth_reached == 1:
        return 0.4
    if depth_reached == 2:
        return 0.7
    return 1.0


def to_json_keys(row: SelfPlayRow) -> dict[str, Any]:
    """
    Convert snake_case Python fields to the PascalCase JSON names written by C#.
    """
    return {
        "GameId": row.game_id,
        "MoveIndex": row.move_index,
        "Side": row.side,
        "SideToMove": row.side_to_move,
        "BoardEncoding": row.board_encoding,
        "LegalMoves": row.legal_moves,
        "SelectedMove": row.selected_move,
        "Result": row.result,
        "SearchScoreSidePerspective": row.search_score_side_perspective,
        "SearchScoreRedPerspective": row.search_score_red_perspective,
        "DepthReached": row.depth_reached,
        "Nodes": row.nodes,
        "TimeMs": row.time_ms,
        "ValueWeight": row.value_weight,
        "PolicyWeight": row.policy_weight,
        "UseForValueTraining": row.use_for_value_training,
        "UseForPolicyTraining": row.use_for_policy_training,
        "Unfinished": row.unfinished,
        "EndReason": row.end_reason,
    }


def from_json_keys(obj: dict[str, Any]) -> SelfPlayRow:
    return SelfPlayRow(
        game_id=int(obj["GameId"]),
        move_index=int(obj["MoveIndex"]),
        side=str(obj["Side"]),
        side_to_move=int(obj["SideToMove"]),
        board_encoding=[float(v) for v in obj["BoardEncoding"]],
        legal_moves=[int(v) for v in obj["LegalMoves"]],
        selected_move=int(obj["SelectedMove"]),
        result=int(obj["Result"]),
        search_score_side_perspective=float(obj["SearchScoreSidePerspective"]),
        search_score_red_perspective=float(obj["SearchScoreRedPerspective"]),
        depth_reached=int(obj["DepthReached"]),
        nodes=int(obj["Nodes"]),
        time_ms=float(obj["TimeMs"]),
        value_weight=float(obj["ValueWeight"]),
        policy_weight=float(obj["PolicyWeight"]),
        use_for_value_training=bool(obj["UseForValueTraining"]),
        use_for_policy_training=bool(obj["UseForPolicyTraining"]),
        unfinished=bool(obj["Unfinished"]),
        end_reason=str(obj["EndReason"]),
    )


def build_selfplay_row() -> SelfPlayRow:
    legal_moves = [
        encode_move(9, 0, 8, 0),
        encode_move(9, 0, 7, 0),
        encode_move(9, 4, 8, 4),
    ]
    selected_move = legal_moves[1]
    result = 1
    side_to_move = RED_TO_MOVE
    depth_reached = 3
    unfinished = False
    end_reason = "RedWins"
    value_weight = get_value_weight(result, unfinished, end_reason)
    policy_weight = get_policy_weight(depth_reached)

    return SelfPlayRow(
        game_id=1,
        move_index=0,
        side="Red",
        side_to_move=side_to_move,
        board_encoding=sample_board_encoding(),
        legal_moves=legal_moves,
        selected_move=selected_move,
        result=result,
        search_score_side_perspective=420.0,
        search_score_red_perspective=420.0,
        depth_reached=depth_reached,
        nodes=12345,
        time_ms=87.5,
        value_weight=value_weight,
        policy_weight=policy_weight,
        use_for_value_training=value_weight > 0,
        use_for_policy_training=policy_weight > 0,
        unfinished=unfinished,
        end_reason=end_reason,
    )


def validate_row(row: SelfPlayRow) -> None:
    if len(row.board_encoding) != INPUT_DIM:
        raise ValueError(f"BoardEncoding must be float[{INPUT_DIM}]")
    if row.side_to_move not in (RED_TO_MOVE, BLACK_TO_MOVE):
        raise ValueError("SideToMove must be 1 or -1")
    if not row.legal_moves:
        raise ValueError("LegalMoves must not be empty")
    if row.selected_move not in row.legal_moves:
        raise ValueError("SelectedMove must be included in LegalMoves")
    if not 0 <= row.selected_move < ACTION_DIM:
        raise ValueError("SelectedMove out of range")
    if any(move < 0 or move >= ACTION_DIM for move in row.legal_moves):
        raise ValueError("LegalMoves values out of range")
    if not 0 <= row.value_weight <= 1:
        raise ValueError("ValueWeight must be in [0, 1]")
    if not 0 <= row.policy_weight <= 1:
        raise ValueError("PolicyWeight must be in [0, 1]")


def policy_targets(selected_move: int) -> tuple[int, int]:
    from_target = selected_move // BOARD_SIZE
    to_target = selected_move % BOARD_SIZE
    return from_target, to_target


def current_player_value_target(result_red_perspective: int, side_to_move: int) -> float:
    # V5 training uses current-player perspective:
    # Red row:   Red win = +1
    # Black row: Red win = -1
    return float(result_red_perspective * side_to_move)


def build_legal_mask(legal_moves: list[int]) -> list[bool]:
    mask = [False] * ACTION_DIM
    for move in legal_moves:
        mask[move] = True
    return mask


def summarize_row(row: SelfPlayRow) -> None:
    validate_row(row)
    from_target, to_target = policy_targets(row.selected_move)
    value_target = current_player_value_target(row.result, row.side_to_move)
    legal_mask = build_legal_mask(row.legal_moves)
    selected_from, selected_to = decode_move(row.selected_move)

    print("=" * 72)
    print("1) One JSONL row from self-play")
    print("=" * 72)
    json_line = json.dumps(to_json_keys(row), ensure_ascii=False)
    print(json_line[:240] + " ...")
    print()

    print("=" * 72)
    print("2) Validation")
    print("=" * 72)
    print(f"BoardEncoding length: {len(row.board_encoding)}")
    print(f"LegalMoves count: {len(row.legal_moves)}")
    print(f"SelectedMove in LegalMoves: {row.selected_move in row.legal_moves}")
    print(f"UseForValueTraining: {row.use_for_value_training}, weight={row.value_weight}")
    print(f"UseForPolicyTraining: {row.use_for_policy_training}, weight={row.policy_weight}")
    print()

    print("=" * 72)
    print("3) Training targets")
    print("=" * 72)
    print(f"Selected move id: {row.selected_move}")
    print(f"Decoded move: {selected_from} -> {selected_to}")
    print(f"Factored policy targets: from={from_target}, to={to_target}")
    print(f"Value target: result({row.result}) * side_to_move({row.side_to_move}) = {value_target:+.1f}")
    print(f"Legal mask true count: {sum(legal_mask)} / {len(legal_mask)}")
    print()


def demo_black_row_value_flip() -> None:
    red_win_from_black_turn = current_player_value_target(1, BLACK_TO_MOVE)
    black_win_from_black_turn = current_player_value_target(-1, BLACK_TO_MOVE)

    print("=" * 72)
    print("4) Why result * side_to_move matters")
    print("=" * 72)
    print(f"Black to move, Red eventually wins:  1 * -1 = {red_win_from_black_turn:+.1f}")
    print(f"Black to move, Black eventually wins: -1 * -1 = {black_win_from_black_turn:+.1f}")
    print("So +1 always means: the player to move in this row is winning.")
    print()


def demo_round_trip_json() -> None:
    row = build_selfplay_row()
    text = json.dumps(to_json_keys(row), ensure_ascii=False)
    loaded = from_json_keys(json.loads(text))
    assert asdict(row) == asdict(loaded)
    validate_row(loaded)
    print("JSON round-trip check passed.")


def main() -> None:
    row = build_selfplay_row()
    summarize_row(row)
    demo_black_row_value_flip()
    demo_round_trip_json()


if __name__ == "__main__":
    main()

