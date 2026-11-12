"""B3: legal_mask must be remapped together with the horizontal flip augmentation."""
from __future__ import annotations

import ast
import types

import numpy as np
import pytest

from tests._helpers import (
    BOARD_COLS,
    BOARD_SIZE,
    extract_method,
    extract_symbols,
    find_class,
    find_function,
    make_torch_shim,
    ShimTensor,
    tree_of,
)

ACTION_DIM = BOARD_SIZE * BOARD_SIZE
COMBINED = "train_v7_combined.py"

# Hand-computed on a 10x9 board (col -> 8 - col):
#   (9,1)=82 -> (7,2)=65 : 82*90+65 = 7445   flip -> 88*90+69 = 7989
#   (9,1)=82 -> (7,0)=63 : 82*90+63 = 7443   flip -> 88*90+71 = 7991
#   (0,4)=4  -> (2,4)=22 :  4*90+22 =  382   flip -> unchanged (centre column)
#   (5,8)=53 -> (5,0)=45 : 53*90+45 = 4815   flip -> 45*90+53 = 4103
LEGAL = [7445, 7443, 382, 4815]
LEGAL_FLIPPED = [7989, 7991, 382, 4103]
SELECTED = 7445
SELECTED_FLIPPED = 7989


def test_remap_legal_moves_matches_hand_computed_values():
    ns = extract_symbols(COMBINED, ["_remap_legal_moves"])
    remap = ns["_remap_legal_moves"]
    assert sorted(remap(LEGAL, flip_columns=True)) == sorted(LEGAL_FLIPPED)
    assert remap(LEGAL, flip_columns=False) == LEGAL
    assert remap(LEGAL) == LEGAL


def test_remap_is_an_involution():
    ns = extract_symbols(COMBINED, ["_remap_legal_moves"])
    remap = ns["_remap_legal_moves"]
    assert sorted(remap(remap(LEGAL, flip_columns=True), flip_columns=True)) == sorted(LEGAL)


def test_remap_is_actually_called_in_getitem():
    tree = tree_of(COMBINED)
    getitem = None
    for sub in find_class(tree, "ChunkedDataset").body:
        if isinstance(sub, ast.FunctionDef) and sub.name == "__getitem__":
            getitem = sub
    assert getitem is not None
    calls = [n for n in ast.walk(getitem)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_remap_legal_moves"]
    assert calls, "_remap_legal_moves is defined but never called in ChunkedDataset.__getitem__"

    # the remap must live inside the flip branch, not after it
    flip_branches = [n for n in ast.walk(getitem)
                     if isinstance(n, ast.If) and ast.dump(n.test).find("random") >= 0]
    assert flip_branches, "no augmentation branch found"
    assert any(any(c in ast.walk(br) for c in calls) for br in flip_branches), \
        "legal remap must happen inside the flip branch"


def _run_getitem(flip: bool):
    shim = make_torch_shim()
    ns = extract_symbols(COMBINED, ["_remap_legal_moves"], extra_ns={"torch": shim})
    ns = extract_method(
        COMBINED, "ChunkedDataset", "__getitem__",
        extra_ns={
            "torch": shim,
            "random": types.SimpleNamespace(random=lambda: 0.0 if flip else 1.0),
            "_remap_legal_moves": ns["_remap_legal_moves"],
        },
    )
    features = np.arange(16 * 10 * 9, dtype=np.float32).reshape(1, 16, 10, 9)
    holder = types.SimpleNamespace(
        _features=ShimTensor(features),
        _moves=np.array([SELECTED], dtype=np.int64),
        _values=np.array([1.0], dtype=np.float32),
        _legal=[ShimTensor(np.array(LEGAL, dtype=np.int64))],
        _vw=np.array([1.0], dtype=np.float32),
        _pw=np.array([1.0], dtype=np.float32),
    )
    return ns["__getitem__"](holder, 0), features[0]


def test_getitem_flipped_mask_contains_flipped_target():
    (feat, move, _value, mask, _vw, _pw), original = _run_getitem(flip=True)
    assert int(move.item()) == SELECTED_FLIPPED
    indices = set(np.flatnonzero(mask.array).tolist())
    assert indices == set(LEGAL_FLIPPED), f"legal_mask not remapped: {sorted(indices)}"
    assert bool(mask.array[SELECTED_FLIPPED]) is True
    assert int(mask.array.sum()) == len(LEGAL)
    np.testing.assert_array_equal(feat.array, np.flip(original, axis=-1))


def test_getitem_unflipped_mask_is_unchanged():
    (_feat, move, _value, mask, _vw, _pw), original = _run_getitem(flip=False)
    assert int(move.item()) == SELECTED
    assert set(np.flatnonzero(mask.array).tolist()) == set(LEGAL)
    np.testing.assert_array_equal(_feat.array, original)


# ── the other trainers must keep remapping (v8 was already fixed; guard the rest) ──

@pytest.mark.parametrize("rel_path", [
    "train_v7_continue.py",
    "train_v7_full_policy.py",
])
def test_v7_inmemory_trainers_remap_legal(rel_path):
    tree = tree_of(rel_path)
    cls = find_class(tree, "ChineseChessDataset")
    getitem = next(s for s in cls.body if isinstance(s, ast.FunctionDef) and s.name == "__getitem__")
    calls = [n for n in ast.walk(getitem)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_remap_legal_moves"]
    assert calls, f"{rel_path}: legal moves are not remapped on flip"


@pytest.mark.parametrize("rel_path", [
    "train_v8_factored.py",
    "train_cnn_policy_value.py",
])
def test_factored_trainers_rebuild_mask_from_remapped_legal(rel_path):
    tree = tree_of(rel_path)
    cls = find_class(tree, "ChineseChessDataset")
    getitem = next(s for s in cls.body if isinstance(s, ast.FunctionDef) and s.name == "__getitem__")
    branches = [n for n in ast.walk(getitem)
                if isinstance(n, ast.If) and "random" in ast.dump(n.test)]
    assert branches
    src = ast.dump(ast.Module(body=branches[0].body, type_ignores=[]))
    assert "legal_mask" in src, f"{rel_path}: flip branch does not rebuild legal_mask"


def test_mlp_trainer_has_no_flip_augmentation():
    # train_chinese_chess_policy_value.py has no flip at all, so B3 cannot apply.
    tree = tree_of("train_chinese_chess_policy_value.py")
    cls = find_class(tree, "ChineseChessDataset")
    getitem = next(s for s in cls.body if isinstance(s, ast.FunctionDef) and s.name == "__getitem__")
    assert "flip" not in ast.dump(getitem)