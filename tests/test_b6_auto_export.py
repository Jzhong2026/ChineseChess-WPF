"""B6: auto_export_onnx.py completion markers, timeout, liveness, honest checkpoint choice."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import time

import pytest

from tests._helpers import REPO_ROOT, source_of, tree_of

TRAINER_FINAL_PRINTS = {
    "train_v7_combined.py": "Saved final: ",
    "train_v7_continue.py": "Saved final: ",
    "train_v7_full_policy.py": "Saved final checkpoint: ",
    "train_cnn_policy_value.py": "Saved final checkpoint: ",
    "train_v8_factored.py": "Final model: ",
}


@pytest.fixture(scope="module")
def auto():
    import auto_export_onnx
    return auto_export_onnx


def _final_print_prefixes(rel_path):
    prefixes = []
    for node in ast.walk(tree_of(rel_path)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print" and node.args):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.JoinedStr) and arg.values and isinstance(arg.values[0], ast.Constant):
            text = arg.values[0].value
        elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            text = arg.value
        else:
            continue
        if "final" in text.lower():
            prefixes.append(text)
    return prefixes


@pytest.mark.parametrize("rel_path,expected", sorted(TRAINER_FINAL_PRINTS.items()))
def test_trainer_final_marker_is_recognised(auto, rel_path, expected):
    prefixes = _final_print_prefixes(rel_path)
    assert any(expected in p for p in prefixes), f"{rel_path}: expected print {expected!r}, got {prefixes}"
    for prefix in prefixes:
        assert auto.is_completed(prefix + "artifacts/model.pt"), \
            f"{rel_path} final marker {prefix!r} is not recognised by auto_export_onnx"


def test_epoch_line_alone_is_not_completion(auto):
    assert not auto.is_completed("epoch=050 lr=1.00e-05 train_loss=1.2345 val_loss=2.0000\n")
    assert not auto.is_completed("  Periodic checkpoint: model.epoch50.pt\n")


def test_wait_times_out(auto, tmp_path):
    log = tmp_path / "training.log"
    log.write_text("epoch=001 lr=1e-4\n", encoding="utf-8")
    started = time.monotonic()
    assert auto.wait_for_completion(str(log), timeout_s=0.3, poll_s=0.05) is False
    assert time.monotonic() - started < 10


def test_wait_returns_true_on_real_marker(auto, tmp_path):
    log = tmp_path / "training.log"
    log.write_text("epoch=002 ...\n\nSaved final: artifacts/x.pt\n", encoding="utf-8")
    assert auto.wait_for_completion(str(log), timeout_s=1.0, poll_s=0.05) is True


def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    proc.wait(timeout=60)
    return proc.pid


def test_wait_detects_dead_trainer(auto, tmp_path):
    log = tmp_path / "training.log"
    log.write_text("epoch=001 ...\n", encoding="utf-8")
    started = time.monotonic()
    assert auto.wait_for_completion(str(log), timeout_s=30, poll_s=0.05, pid=_dead_pid()) is False
    assert time.monotonic() - started < 10


def test_process_alive(auto):
    assert auto._process_alive(os.getpid()) is True
    assert auto._process_alive(_dead_pid()) is False


def test_select_checkpoint_prefers_best(auto, tmp_path):
    (tmp_path / "m.best.pt").write_bytes(b"x")
    (tmp_path / "m.epoch5.pt").write_bytes(b"x")
    (tmp_path / "m.epoch40.pt").write_bytes(b"x")
    chosen = auto.select_checkpoint(str(tmp_path), "m", str(tmp_path / "m.best.pt"))
    assert chosen == str(tmp_path / "m.best.pt")


def test_select_checkpoint_falls_back_to_highest_epoch(auto, tmp_path):
    (tmp_path / "m.epoch5.pt").write_bytes(b"x")
    (tmp_path / "m.epoch40.pt").write_bytes(b"x")
    (tmp_path / "m.epoch10.pt").write_bytes(b"x")
    chosen = auto.select_checkpoint(str(tmp_path), "m", str(tmp_path / "m.best.pt"))
    assert chosen == str(tmp_path / "m.epoch40.pt")


def test_select_checkpoint_returns_none_when_empty(auto, tmp_path):
    assert auto.select_checkpoint(str(tmp_path), "m", str(tmp_path / "m.best.pt")) is None


def test_no_hardcoded_fake_conclusion():
    src = source_of("auto_export_onnx.py")
    assert "Best model (epoch 10)" not in src
    assert "epoch40" not in src