"""Cross-language encoding contract + DRY convergence guards.

Truth source: ChineseChess.Core/Encoding/BoardEncoder.cs, MoveEncoder.cs and
Services/XiangqiNeuralAiService.cs (EncodeCnn). Every expected value below is
hand-derived from those sources or from an independent literal transcription of
them -- never computed with chesslib itself.
"""
from __future__ import annotations

import ast
import types

import numpy as np
import pytest

from tests._helpers import (
    REPO_ROOT,
    ShimTensor,
    extract_method,
    extract_symbols,
    initial_board_encoding,
    make_torch_shim,
    source_of,
    tree_of,
)

# Hand-derived from C#: XiangqiEngine.BoardRows/BoardCols, BoardEncoder.Planes,
# MoveEncoder.ActionCount, XiangqiNeuralAiService._inputPlanes (V5 = 16).
ROWS = 10
COLS = 9
SQUARES = 90
PIECE_PLANES = 14
INPUT_PLANES = 16
FLAT_DIM = 1260
ACTION_DIM = 8100

TRAINERS = [
    "train_chinese_chess_policy_value.py",
    "train_cnn_policy_value.py",
    "train_v7_combined.py",
    "train_v7_continue.py",
    "train_v7_full_policy.py",
    "train_v8_factored.py",
]

CNN_TRAINERS = [
    "train_cnn_policy_value.py",
    "train_v7_combined.py",
    "train_v7_continue.py",
    "train_v7_full_policy.py",
    "train_v8_factored.py",
]


@pytest.fixture(scope="module")
def enc():
    from chesslib import encoding
    return encoding


# ── independent reference: literal transcription of the C# code ──────────────

def reference_encode_cnn(flat, side_to_move):
    """BoardEncoder.Encode (square-major) + EncodeCnn (plane-major + side planes)."""
    tensor = np.zeros((INPUT_PLANES, ROWS, COLS), dtype=np.float32)
    for row in range(ROWS):
        for col in range(COLS):
            for plane in range(PIECE_PLANES):
                flat_idx = (row * COLS + col) * PIECE_PLANES + plane
                tensor[plane, row, col] = flat[flat_idx]
    # Side.Red == 0 -> JSONL SideToMove 1; Side.Black == 1 -> JSONL SideToMove -1
    red_plane = 1.0 if side_to_move == 1 else 0.0
    black_plane = 1.0 if side_to_move == -1 else 0.0
    for row in range(ROWS):
        for col in range(COLS):
            tensor[14, row, col] = red_plane
            tensor[15, row, col] = black_plane
    return tensor


def reference_remap(action):
    """MoveEncoder.Decode -> mirror col (8-col) -> MoveEncoder.Encode."""
    from_idx, to_idx = divmod(action, SQUARES)
    from_row, from_col = divmod(from_idx, COLS)
    to_row, to_col = divmod(to_idx, COLS)
    return ((from_row * COLS + (COLS - 1 - from_col)) * SQUARES
            + (to_row * COLS + (COLS - 1 - to_col)))


def reference_mirror_flat(flat):
    """Mirror a square-major BoardEncoder payload column-wise."""
    out = [0.0] * FLAT_DIM
    for row in range(ROWS):
        for col in range(COLS):
            src = (row * COLS + col) * PIECE_PLANES
            dst = (row * COLS + (COLS - 1 - col)) * PIECE_PLANES
            out[dst:dst + PIECE_PLANES] = list(flat[src:src + PIECE_PLANES])
    return out


# (flat index, plane, row, col) hand-computed via BoardEncoder.GetIndex:
#   square = row*9+col, plane = side*7+type, flat = square*14+plane
#   Side: Red=0, Black=1. PieceType: General0 Advisor1 Elephant2 Horse3 Rook4 Cannon5 Soldier6
SPARSE_PIECES = [
    (1190, 0, 9, 4),    # red general   (9,4): 85*14+0
    (1177, 1, 9, 3),    # red advisor   (9,3): 84*14+1
    (1138, 4, 9, 0),    # red rook      (9,0): 81*14+4
    (901, 5, 7, 1),     # red cannon    (7,1): 64*14+5
    (63, 7, 0, 4),      # black general (0,4):  4*14+7
    (317, 9, 2, 4),     # black elephant(2,4): 22*14+9
    (24, 10, 0, 1),     # black horse   (0,1):  1*14+10
    (447, 13, 3, 4),    # black soldier (3,4): 31*14+13
]

# Hand-computed on a 10x9 board, col -> 8-col:
#   (9,1)=82 -> (7,2)=65 : 7445  ->  (9,7)=88 -> (7,6)=69 : 7989
#   (9,1)=82 -> (7,0)=63 : 7443  ->  (9,7)=88 -> (7,8)=71 : 7991
#   (0,4)=4  -> (2,4)=22 :  382  ->  centre column, unchanged
#   (5,8)=53 -> (5,0)=45 : 4815  ->  (5,0)=45 -> (5,8)=53 : 4103
#   (0,0)=0  -> (0,0)=0  :    0  ->  (0,8)=8  -> (0,8)=8  :  728
#   (9,8)=89 -> (9,8)=89 : 8099  ->  (9,0)=81 -> (9,0)=81 : 7371
REMAP_CASES = [(7445, 7989), (7443, 7991), (382, 382), (4815, 4103),
               (0, 728), (8099, 7371)]


# ── A. constants and C# truth source ─────────────────────────────────────────

def test_constants_match_csharp_contract(enc):
    assert enc.BOARD_ROWS == ROWS
    assert enc.BOARD_COLS == COLS
    assert enc.BOARD_SIZE == SQUARES
    assert enc.PIECE_PLANES == PIECE_PLANES
    assert enc.SIDE_PLANES == 2
    assert enc.ACTION_DIM == ACTION_DIM
    assert enc.BOARD_SIZE * enc.PIECE_PLANES == FLAT_DIM
    assert enc.PIECE_PLANES + enc.SIDE_PLANES == INPUT_PLANES


def _csharp_source(filename):
    hits = sorted((REPO_ROOT / "ChineseChess").rglob(filename))
    if not hits:
        pytest.skip(f"{filename} not found (C# Core refactor in progress)")
    return " ".join(hits[0].read_text(encoding="utf-8-sig").split())


def test_csharp_board_encoder_still_declares_the_contract():
    src = _csharp_source("BoardEncoder.cs")
    assert "Planes = 14" in src
    assert "square * Planes + plane" in src
    assert "(int)side * 7 + (int)type" in src


# C# CNN layout/output parity is exercised by NeuralInferenceContractTests
# against a separate raw ONNX session for both sides and every action. Avoid
# prescribing a particular tensor indexer spelling: contiguous buffer access
# implements the same contract without the former per-element index overhead.


def test_csharp_move_encoder_still_declares_the_contract():
    src = _csharp_source("MoveEncoder.cs")
    assert "fromIndex * BoardSquareCount + toIndex" in src


# ── B. encode_board ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("side", [1, -1])
def test_sparse_board_matches_hand_computed_planes(enc, side):
    flat = [0.0] * FLAT_DIM
    expected = np.zeros((INPUT_PLANES, ROWS, COLS), dtype=np.float32)
    for flat_idx, plane, row, col in SPARSE_PIECES:
        flat[flat_idx] = 1.0
        expected[plane, row, col] = 1.0
    expected[14 if side == 1 else 15] = 1.0

    got = enc.encode_board(flat, side)
    assert got.shape == (INPUT_PLANES, ROWS, COLS)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)
    assert got[:PIECE_PLANES].sum() == len(SPARSE_PIECES)


def test_every_plane_lands_on_its_own_square(enc):
    # plane p written at square p: flat[p*14+p] -> features[p, p//9, p%9]
    flat = [0.0] * FLAT_DIM
    expected = np.zeros((INPUT_PLANES, ROWS, COLS), dtype=np.float32)
    for plane in range(PIECE_PLANES):
        flat[plane * PIECE_PLANES + plane] = 1.0
        expected[plane, plane // COLS, plane % COLS] = 1.0
    expected[14] = 1.0
    np.testing.assert_array_equal(enc.encode_board(flat, 1), expected)


@pytest.mark.parametrize("side", [1, -1])
def test_plane_major_layout_matches_csharp_encode_cnn(enc, side):
    flat = np.asarray([(i % 13) - 6 for i in range(FLAT_DIM)], dtype=np.float32)
    np.testing.assert_array_equal(enc.encode_board(flat, side),
                                  reference_encode_cnn(flat, side))


@pytest.mark.parametrize("side", [1, -1])
def test_opening_position_matches_reference(enc, side):
    flat = initial_board_encoding()
    got = enc.encode_board(flat, side)
    np.testing.assert_array_equal(got, reference_encode_cnn(flat, side))
    assert got[:PIECE_PLANES].sum() == 32.0
    assert got[3, 9, 1] == 1.0    # red horse  (9,1) -> plane 0*7+3
    assert got[7, 0, 4] == 1.0    # black general (0,4) -> plane 1*7+0
    assert got[13, 3, 0] == 1.0   # black soldier (3,0) -> plane 1*7+6


@pytest.mark.parametrize("side,red_sum,black_sum", [(1, 90.0, 0.0), (-1, 0.0, 90.0)])
def test_side_to_move_planes_are_mutually_exclusive(enc, side, red_sum, black_sum):
    features = enc.encode_board(initial_board_encoding(), side)
    assert features[14].sum() == red_sum
    assert features[15].sum() == black_sum
    assert set(np.unique(features[14]).tolist()) <= {0.0, 1.0}
    assert set(np.unique(features[15]).tolist()) <= {0.0, 1.0}


def test_piece_planes_do_not_depend_on_side_to_move(enc):
    flat = initial_board_encoding()
    np.testing.assert_array_equal(enc.encode_board(flat, 1)[:PIECE_PLANES],
                                  enc.encode_board(flat, -1)[:PIECE_PLANES])


@pytest.mark.parametrize("bad", [[], [0.0] * 10, [0.0] * 1259, [0.0] * 1261,
                                 [0.0] * (SQUARES * INPUT_PLANES)])
def test_encode_board_rejects_wrong_length(enc, bad):
    with pytest.raises(ValueError):
        enc.encode_board(bad, 1)


def test_encode_board_accepts_list_and_ndarray(enc):
    flat = initial_board_encoding()
    from_list = enc.encode_board(flat, 1)
    from_array = enc.encode_board(np.asarray(flat, dtype=np.float64), 1)
    np.testing.assert_array_equal(from_list, from_array)
    assert from_list.dtype == np.float32


# ── C. remap_move / remap_legal_moves ────────────────────────────────────────

@pytest.mark.parametrize("action,expected", REMAP_CASES)
def test_remap_move_matches_hand_computed_values(enc, action, expected):
    assert enc.remap_move(action) == expected


def test_remap_move_matches_reference_for_every_action(enc):
    for action in range(ACTION_DIM):
        assert enc.remap_move(action) == reference_remap(action), action


def test_remap_move_is_an_involution_and_preserves_rows(enc):
    for action in range(ACTION_DIM):
        flipped = enc.remap_move(action)
        assert enc.remap_move(flipped) == action
        assert flipped // SQUARES // COLS == action // SQUARES // COLS
        assert flipped % SQUARES // COLS == action % SQUARES // COLS


def test_remap_legal_moves_preserves_order_and_duplicates(enc):
    legal = [7445, 382, 7443, 382, 4815]
    assert enc.remap_legal_moves(legal) == [reference_remap(m) for m in legal]
    assert enc.remap_legal_moves([]) == []
    assert legal == [7445, 382, 7443, 382, 4815]


def test_remap_legal_moves_accepts_any_iterable(enc):
    assert enc.remap_legal_moves(iter([7445, 382])) == [7989, 382]
    assert enc.remap_legal_moves(np.asarray([7445, 382], dtype=np.int64)) == [7989, 382]


# ── D. flip_features ─────────────────────────────────────────────────────────

def test_flip_features_mirrors_columns_only(enc):
    features = np.arange(INPUT_PLANES * ROWS * COLS, dtype=np.float32).reshape(
        INPUT_PLANES, ROWS, COLS)
    flipped = enc.flip_features(features)
    for col in range(COLS):
        np.testing.assert_array_equal(flipped[:, :, col], features[:, :, COLS - 1 - col])
    assert flipped.shape == features.shape


def test_flip_features_handles_batch_and_list(enc):
    batch = np.arange(2 * INPUT_PLANES * ROWS * COLS, dtype=np.float32).reshape(
        2, INPUT_PLANES, ROWS, COLS)
    np.testing.assert_array_equal(enc.flip_features(batch), np.flip(batch, axis=-1))
    nested = batch.tolist()
    np.testing.assert_array_equal(enc.flip_features(nested), np.flip(batch, axis=-1))


def test_flip_features_delegates_to_tensor_like_objects(enc):
    array = np.arange(INPUT_PLANES * ROWS * COLS, dtype=np.float32).reshape(
        INPUT_PLANES, ROWS, COLS)
    flipped = enc.flip_features(ShimTensor(array))
    assert isinstance(flipped, ShimTensor)
    np.testing.assert_array_equal(flipped.array, np.flip(array, axis=-1))


def test_flip_features_is_an_involution(enc):
    features = enc.encode_board(initial_board_encoding(), -1)
    np.testing.assert_array_equal(enc.flip_features(enc.flip_features(features)), features)


@pytest.mark.parametrize("side", [1, -1])
def test_flipping_equals_encoding_the_mirrored_board(enc, side):
    flat = initial_board_encoding()
    mirrored = reference_mirror_flat(flat)
    np.testing.assert_array_equal(enc.flip_features(enc.encode_board(flat, side)),
                                  enc.encode_board(mirrored, side))
    np.testing.assert_array_equal(enc.encode_board(mirrored, side),
                                  reference_encode_cnn(mirrored, side))


def test_chesslib_encoding_does_not_import_torch(enc):
    tree = ast.parse(source_of("chesslib/encoding.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not a.name.split(".")[0] == "torch" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "torch"


# ── E. DRY / convergence structure guards ────────────────────────────────────

def _chesslib_imports(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "chesslib":
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name for alias in node.names
                         if alias.name.split(".")[0] == "chesslib")
    return names


@pytest.mark.parametrize("rel_path", TRAINERS)
def test_trainers_import_chesslib(rel_path):
    assert _chesslib_imports(tree_of(rel_path)), f"{rel_path}: does not import chesslib"


# Signatures of the copy-pasted square-major -> plane-major encoder and flip remap.
DUPLICATION_SIGNATURES = [
    ".permute(1, 0)",
    "side_planes",
    "BOARD_COLS - 1 -",
    ".flip(-1)",
    "* BOARD_PLANES +",
]


@pytest.mark.parametrize("rel_path", CNN_TRAINERS)
def test_trainers_have_no_duplicated_encoding_logic(rel_path):
    src = source_of(rel_path)
    for signature in DUPLICATION_SIGNATURES:
        assert signature not in src, f"{rel_path}: still contains duplicated logic {signature!r}"


@pytest.mark.parametrize("rel_path", TRAINERS)
def test_trainers_do_not_redefine_contract_constants(rel_path):
    guarded = {"BOARD_ROWS", "BOARD_COLS", "BOARD_SIZE", "BOARD_PLANES", "PIECE_PLANES",
               "SIDE_PLANES", "INPUT_PLANES", "ACTION_DIM", "INPUT_DIM", "FLAT_DIM"}
    tree = tree_of(rel_path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in guarded:
                assert not isinstance(node.value, ast.Constant), \
                    f"{rel_path}:{node.lineno} redefines {target.id} with a literal"
                assert not isinstance(node.value, ast.BinOp), \
                    f"{rel_path}:{node.lineno} re-derives {target.id}; import it from chesslib"


@pytest.mark.parametrize("rel_path", TRAINERS)
def test_legacy_private_helpers_are_pure_delegations(rel_path):
    tree = tree_of(rel_path)
    shared = _chesslib_imports(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in {"_encode_features", "_remap_legal_moves"}:
            continue
        params = {a.arg for a in list(node.args.args) + list(node.args.kwonlyargs)}
        allowed = params | shared
        assert len(node.body) == 1 and isinstance(node.body[0], ast.Return), \
            f"{rel_path}:{node.lineno} {node.name} must be a one-line delegation"
        for stmt in node.body:
            for sub in ast.walk(stmt):
                assert not isinstance(sub, (ast.BinOp, ast.For, ast.While,
                                            ast.ListComp, ast.Subscript)), \
                    f"{rel_path}:{node.lineno} {node.name} still computes indices itself"
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    assert sub.id in allowed, \
                        f"{rel_path}:{node.lineno} {node.name} uses non-shared {sub.id!r}"


@pytest.mark.parametrize("rel_path,owner", [
    ("train_cnn_policy_value.py", "ChineseChessDataset"),
    ("train_v7_continue.py", "ChineseChessDataset"),
    ("train_v7_full_policy.py", "ChineseChessDataset"),
    ("train_v8_factored.py", "ChineseChessDataset"),
    ("train_v7_combined.py", None),
])
def test_call_sites_use_the_shared_encoder(rel_path, owner):
    tree = tree_of(rel_path)
    if owner is None:
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "preprocess_jsonl")
    else:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
        fn = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == "__getitem__")
    calls = {n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "encode_board" in calls, f"{rel_path}: does not call chesslib.encode_board"
    assert "flip_features" in calls or owner is None, \
        f"{rel_path}: flip augmentation does not use chesslib.flip_features"


# ── F. behaviour characterisation (torch-free shim, same before/after) ───────

LEGAL = [7445, 7443, 7989, 5827, 5269]
SELECTED = 7445


def _sample(side=1, result=1.0):
    return types.SimpleNamespace(
        board_encoding=initial_board_encoding(),
        side_to_move=side,
        legal_moves=list(LEGAL),
        selected_move=SELECTED,
        result=result,
        value_weight=0.5,
        policy_weight=0.25,
    )


def _run_dataset(rel_path, sample, flip, class_name="ChineseChessDataset", extra=None):
    shim = make_torch_shim()
    ns = {"torch": shim,
          "random": types.SimpleNamespace(random=lambda: 0.0 if flip else 1.0)}
    if extra:
        ns.update(extra)
    if rel_path in ("train_v7_continue.py", "train_v7_full_policy.py"):
        ns["SelfPlaySample"] = extract_symbols(rel_path, ["SelfPlaySample"])["SelfPlaySample"]
    if any(isinstance(n, ast.FunctionDef) and n.name == "_remap_legal_moves"
           for n in tree_of(rel_path).body):
        ns["_remap_legal_moves"] = extract_symbols(
            rel_path, ["_remap_legal_moves"])["_remap_legal_moves"]
    method = extract_method(rel_path, class_name, "__getitem__", extra_ns=ns)
    holder = types.SimpleNamespace(samples=[sample], aug_prob=0.5)
    return method["__getitem__"](holder, 0)


def _mask_indices(mask):
    return set(np.flatnonzero(np.asarray(mask)).tolist())


@pytest.mark.parametrize("rel_path", ["train_cnn_policy_value.py", "train_v8_factored.py"])
@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("flip", [False, True])
def test_factored_datasets_match_the_csharp_contract(rel_path, side, flip):
    sample = _sample(side=side, result=-1.0)
    features, from_t, to_t, value, mask, vw, pw = _run_dataset(rel_path, sample, flip)

    expected = reference_encode_cnn(sample.board_encoding, side)
    if flip:
        expected = np.flip(expected, axis=-1)
    np.testing.assert_array_equal(np.asarray(features), expected)

    action = reference_remap(SELECTED) if flip else SELECTED
    assert int(from_t.item()) == action // SQUARES
    assert int(to_t.item()) == action % SQUARES
    assert float(value.item()) == pytest.approx(sample.result * side)
    assert _mask_indices(mask) == {reference_remap(m) if flip else m for m in LEGAL}
    assert float(vw.item()) == 0.5
    assert float(pw.item()) == 0.25


@pytest.mark.parametrize("rel_path", ["train_v7_continue.py", "train_v7_full_policy.py"])
@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("flip", [False, True])
def test_v7_inmemory_datasets_match_the_csharp_contract(rel_path, side, flip):
    sample = _sample(side=side, result=1.0)
    features, move, value, mask, vw, pw = _run_dataset(rel_path, sample, flip)

    expected = reference_encode_cnn(sample.board_encoding, side)
    if flip:
        expected = np.flip(expected, axis=-1)
    np.testing.assert_array_equal(np.asarray(features), expected)
    assert int(move.item()) == (reference_remap(SELECTED) if flip else SELECTED)
    assert float(value.item()) == pytest.approx(sample.result * side)
    assert _mask_indices(mask) == {reference_remap(m) if flip else m for m in LEGAL}
    assert float(vw.item()) == 0.5
    assert float(pw.item()) == 0.25


@pytest.mark.parametrize("side", [1, -1])
def test_mlp_dataset_keeps_the_flat_1260_contract(side):
    sample = _sample(side=side, result=1.0)
    features, move, value, mask = _run_dataset(
        "train_chinese_chess_policy_value.py", sample, flip=False)
    np.testing.assert_array_equal(np.asarray(features),
                                  np.asarray(sample.board_encoding, dtype=np.float32))
    assert np.asarray(features).shape == (FLAT_DIM,)
    assert int(move.item()) == SELECTED
    assert float(value.item()) == 1.0
    assert _mask_indices(mask) == set(LEGAL)
