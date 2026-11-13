"""Leakage and color/perspective contracts for offline teacher distillation."""
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from train_teacher import TeacherNet, augment, read_split


def game_id(valid):
    for i in range(100):
        key = f"fixture-{i}"
        if (int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 5 == 0) == valid:
            return key
    raise AssertionError("No matching fixture group")


def test_split_removes_shared_positions_without_splitting_games(tmp_path):
    train, valid = game_id(False), game_id(True)
    rows = [dict(GameId=game, Fen=fen, SelectedMove=1, LegalMoves=[1, 2], SideToMove=1, TeacherValue=0.2)
            for game, fen in [(train, "shared"), (train, "train-only"),
                              (valid, "shared"), (valid, "valid-only")]]
    path = tmp_path / "teacher.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    training, validation, info = read_split([path])
    assert [r["Fen"] for r in training] == ["train-only"]
    assert {r["Fen"] for r in validation} == {"shared", "valid-only"}
    assert info["overlapping_positions_removed"] == 1
    assert {r["GameId"] for r in training}.isdisjoint(r["GameId"] for r in validation)


def test_material_skip_and_actions_preserve_side_to_move_under_symmetry(monkeypatch):
    board = torch.zeros((1, 16, 10, 9))
    board[0, 4, 9, 0] = 1  # Red rook.
    board[0, 14] = 1       # Red to move.
    actions = torch.tensor([[81 * 90 + 72]])  # (9,0)->(8,0)
    monkeypatch.setattr("train_teacher.random.random", lambda: 0.0)
    transformed, transformed_actions = augment(board, actions)
    assert transformed[0, 11, 0, 0] == 1  # Black rook after both symmetries.
    assert transformed[0, 15].sum() == 90
    assert transformed_actions.item() == 9  # (0,0)->(1,0)
    model = TeacherNet(pair_rank=0, material_skip=True).eval()
    torch.nn.init.zeros_(model.value_fc2.weight)
    torch.nn.init.zeros_(model.value_fc2.bias)
    with torch.no_grad():
        original = model(board)[1]
        rotated = model(transformed)[1]
    assert original.item() == pytest.approx(torch.tanh(torch.tensor(1.5)).item())
    assert original.item() == pytest.approx(rotated.item())


def test_illegal_teacher_label_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(dict(GameId="bad", SelectedMove=3, LegalMoves=[1, 2])), encoding="utf-8")
    with pytest.raises(ValueError, match="illegal"):
        read_split([path])
