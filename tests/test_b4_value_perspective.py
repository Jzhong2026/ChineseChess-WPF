"""B4: diagnose_nn.py must not flip the value a second time for 16-plane models."""
from __future__ import annotations

import pytest

from tests._helpers import ONNX_MODEL, opening_sample, source_of, write_jsonl

pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def diagnose_module():
    import diagnose_nn
    return diagnose_nn


def test_module_is_importable_without_side_effects(diagnose_module):
    assert hasattr(diagnose_module, "main")


@pytest.mark.parametrize("side", [1, -1])
def test_16_plane_value_is_already_current_player(diagnose_module, side):
    # C# contract: XiangqiNeuralAiService.EvaluatePosition -> _isV5Model (planes >= 16) => no flip
    assert diagnose_module.value_to_player_perspective(0.75, side, 16) == pytest.approx(0.75)
    assert diagnose_module.value_to_player_perspective(-0.25, side, 16) == pytest.approx(-0.25)


@pytest.mark.parametrize("side,expected", [(1, 0.75), (-1, -0.75)])
def test_legacy_14_plane_value_is_red_perspective(diagnose_module, side, expected):
    assert diagnose_module.value_to_player_perspective(0.75, side, 14) == pytest.approx(expected)


def test_check3_no_longer_multiplies_by_side_unconditionally(diagnose_module):
    src = source_of("diagnose_nn.py")
    assert "val_player = val * side_to_move" not in src
    assert "value_to_player_perspective" in src


def test_encode_cnn_honours_plane_count(diagnose_module):
    sample = opening_sample(side_to_move=-1)
    tensor = diagnose_module.encode_cnn(sample["BoardEncoding"], 16, -1)
    assert tensor.shape == (1, 16, 10, 9)
    assert tensor[0, 15].sum() == 90.0
    assert tensor[0, 14].sum() == 0.0


def test_main_runs_against_real_model(tmp_path, diagnose_module, capsys):
    samples = [opening_sample(1, 1.0), opening_sample(-1, -1.0),
               opening_sample(1, -1.0), opening_sample(-1, 1.0)]
    data = write_jsonl(tmp_path / "samples.jsonl", samples)
    rc = diagnose_module.main(["--model", str(ONNX_MODEL), "--data", str(data), "--max-samples", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CHECK 3" in out
    assert "current player" in out.lower()


def test_main_fails_on_missing_data(tmp_path, diagnose_module, capsys):
    rc = diagnose_module.main(["--model", str(ONNX_MODEL), "--data", str(tmp_path / "nope.jsonl")])
    assert rc != 0
    assert "nope.jsonl" in capsys.readouterr().out


def test_no_hardcoded_14_plane_tensor(diagnose_module):
    src = source_of("diagnose_nn.py")
    assert "(1, PLANES, BOARD_ROWS, BOARD_COLS)" not in src