"""B2: verify_onnx.py must read the real input spec and actually assert things."""
from __future__ import annotations

import numpy as np
import pytest

from tests._helpers import BOARD_SIZE, ONNX_MODEL, opening_sample, write_jsonl

ort = pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def verify_module():
    import verify_onnx
    return verify_onnx


@pytest.fixture(scope="module")
def real_session():
    if not ONNX_MODEL.is_file():
        pytest.skip(f"model missing: {ONNX_MODEL}")
    return ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])


def test_module_is_importable_without_side_effects(verify_module):
    assert hasattr(verify_module, "main")


def test_input_spec_comes_from_model_metadata(verify_module, real_session):
    name, planes, rows, cols = verify_module.resolve_input_spec(real_session)
    assert name == "board_input"
    assert (planes, rows, cols) == (16, 10, 9)


def test_encode_board_uses_16_planes_and_side_plane(verify_module):
    sample = opening_sample(side_to_move=-1)
    tensor = verify_module.encode_board(sample["BoardEncoding"], 16, -1)
    assert tensor.shape == (1, 16, 10, 9)
    assert tensor.dtype == np.float32
    # square (9,1) holds a red horse -> plane 3
    assert tensor[0, 3, 9, 1] == 1.0
    # square (0,4) holds the black general -> plane 7
    assert tensor[0, 7, 0, 4] == 1.0
    assert tensor[0, 14].sum() == 0.0
    assert tensor[0, 15].sum() == 90.0
    assert tensor[0, :14].sum() == 32.0


def test_encode_board_legacy_14_planes(verify_module):
    sample = opening_sample()
    tensor = verify_module.encode_board(sample["BoardEncoding"], 14, 1)
    assert tensor.shape == (1, 14, 10, 9)
    assert tensor[0, 3, 9, 1] == 1.0


def test_encode_board_rejects_wrong_encoding_length(verify_module):
    with pytest.raises(verify_module.VerificationError):
        verify_module.encode_board([0.0] * 10, 16, 1)


def test_mask_illegal_argmax_is_legal(verify_module):
    logits = np.zeros(8100, dtype=np.float32)
    logits[7] = 100.0          # illegal but highest
    logits[42] = 1.0
    masked = verify_module.mask_illegal(logits, [42, 43])
    assert int(np.argmax(masked)) == 42
    assert masked[7] == pytest.approx(-1e9)


def test_output_contract_rejects_out_of_range_value(verify_module):
    names = ["from_logits", "to_logits", "value_pred"]
    outputs = [np.zeros((1, BOARD_SIZE)), np.zeros((1, BOARD_SIZE)), np.array([1.7])]
    with pytest.raises(verify_module.VerificationError):
        verify_module.check_output_contract(names, outputs, 1)


def test_output_contract_rejects_nan_logits(verify_module):
    names = ["from_logits", "to_logits", "value_pred"]
    bad = np.zeros((1, BOARD_SIZE))
    bad[0, 0] = np.nan
    outputs = [bad, np.zeros((1, BOARD_SIZE)), np.array([0.0])]
    with pytest.raises(verify_module.VerificationError):
        verify_module.check_output_contract(names, outputs, 1)


def test_output_contract_rejects_wrong_shape(verify_module):
    names = ["from_logits", "to_logits", "value_pred"]
    outputs = [np.zeros((1, 8100)), np.zeros((1, BOARD_SIZE)), np.array([0.0])]
    with pytest.raises(verify_module.VerificationError):
        verify_module.check_output_contract(names, outputs, 1)


def test_output_contract_rejects_unexpected_names(verify_module):
    names = ["policy_logits", "value_pred"]
    outputs = [np.zeros((1, 8100)), np.array([0.0])]
    with pytest.raises(verify_module.VerificationError):
        verify_module.check_output_contract(["policy_logits", "wrong"], outputs, 1)


def test_output_contract_accepts_real_model(verify_module, real_session):
    sample = opening_sample()
    tensor = verify_module.encode_board(sample["BoardEncoding"], 16, 1)
    names = [o.name for o in real_session.get_outputs()]
    outputs = real_session.run(None, {"board_input": tensor})
    assert verify_module.check_output_contract(names, outputs, 1) is True


def test_main_passes_on_real_model(tmp_path, verify_module):
    data = write_jsonl(tmp_path / "samples.jsonl",
                       [opening_sample(1, 1.0), opening_sample(-1, -1.0)])
    rc = verify_module.main(["--model", str(ONNX_MODEL), "--data", str(data)])
    assert rc == 0


def test_main_fails_on_missing_data(tmp_path, verify_module, capsys):
    rc = verify_module.main(["--model", str(ONNX_MODEL), "--data", str(tmp_path / "nope.jsonl")])
    assert rc != 0
    assert "nope.jsonl" in capsys.readouterr().out


def test_main_fails_on_missing_model(tmp_path, verify_module):
    data = write_jsonl(tmp_path / "samples.jsonl", [opening_sample()])
    rc = verify_module.main(["--model", str(tmp_path / "nope.onnx"), "--data", str(data)])
    assert rc != 0


def test_main_fails_when_label_is_not_legal(tmp_path, verify_module):
    sample = opening_sample()
    sample["SelectedMove"] = 1234  # not in LegalMoves
    data = write_jsonl(tmp_path / "bad.jsonl", [sample])
    rc = verify_module.main(["--model", str(ONNX_MODEL), "--data", str(data)])
    assert rc != 0


def test_main_fails_on_nondeterministic_session(tmp_path, verify_module, monkeypatch):
    data = write_jsonl(tmp_path / "samples.jsonl", [opening_sample()])
    real = ort.InferenceSession

    class Flaky:
        def __init__(self, *a, **k):
            self._inner = real(*a, **k)
            self._calls = 0

        def get_inputs(self):
            return self._inner.get_inputs()

        def get_outputs(self):
            return self._inner.get_outputs()

        def run(self, *a, **k):
            out = self._inner.run(*a, **k)
            self._calls += 1
            if self._calls > 1:
                out = [o + 0.5 for o in out]
            return out

    monkeypatch.setattr(verify_module.ort, "InferenceSession", Flaky)
    rc = verify_module.main(["--model", str(ONNX_MODEL), "--data", str(data)])
    assert rc != 0


def test_no_hardcoded_14_plane_input(verify_module):
    from tests._helpers import source_of
    src = source_of("verify_onnx.py")
    assert "(1, PLANES, BOARD_ROWS, BOARD_COLS)" not in src
    assert "1, 14, 10, 9" not in src