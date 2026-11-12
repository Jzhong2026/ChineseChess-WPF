"""B5: trainers must persist full architecture metadata; export must not guess it."""
from __future__ import annotations

import pytest

from tests._helpers import extract_symbols, torch_save_dicts, tree_of

REQUIRED = {"architecture", "planes", "channels", "res_blocks", "policy_channels"}

TRAINERS = [
    "train_v7_combined.py",
    "train_v7_continue.py",
    "train_v7_full_policy.py",
    "train_v8_factored.py",
    "train_cnn_policy_value.py",
]


@pytest.mark.parametrize("rel_path", TRAINERS)
def test_every_state_dict_checkpoint_carries_architecture_metadata(rel_path):
    saves = torch_save_dicts(tree_of(rel_path))
    assert saves, f"{rel_path}: no torch.save of a state dict found"
    for lineno, keys in saves:
        missing = REQUIRED - keys
        assert not missing, f"{rel_path}:{lineno} checkpoint missing {sorted(missing)}"


@pytest.fixture(scope="module")
def resolve():
    ns = extract_symbols("export_cnn_onnx.py", ["resolve_architecture"])
    return ns["resolve_architecture"]


FULL_V7 = {
    "architecture": "cnn_full_v7", "planes": 16, "channels": 64,
    "res_blocks": 6, "policy_channels": 8,
}
FACTORED = {
    "architecture": "cnn_factored_v5", "planes": 16, "channels": 128,
    "res_blocks": 8, "policy_channels": 2,
}


def test_resolve_accepts_complete_metadata(resolve):
    assert resolve(FULL_V7)["planes"] == 16
    assert resolve(FACTORED)["policy_channels"] == 2


@pytest.mark.parametrize("key", ["planes", "channels", "res_blocks", "policy_channels", "architecture"])
def test_resolve_rejects_missing_key(resolve, key):
    ckpt = dict(FULL_V7)
    ckpt.pop(key)
    with pytest.raises(ValueError) as exc:
        resolve(ckpt)
    assert key in str(exc.value)


def test_resolve_rejects_unknown_architecture(resolve):
    with pytest.raises(ValueError):
        resolve({"architecture": "transformer", "planes": 16, "channels": 64,
                 "res_blocks": 6, "policy_channels": 8})


def test_legacy_cnn_does_not_require_policy_channels(resolve):
    ckpt = {"architecture": "cnn", "planes": 14, "channels": 128, "res_blocks": 8}
    assert resolve(ckpt)["planes"] == 14


def test_export_no_longer_defaults_planes_to_14():
    from tests._helpers import source_of
    src = source_of("export_cnn_onnx.py")
    assert 'checkpoint.get("planes", BOARD_PLANES)' not in src
    assert "resolve_architecture" in src