#!/usr/bin/env python3
"""
Verify the deployed ONNX model with onnxruntime only (no PyTorch required).

Asserts: input spec read from the model, output names/shapes, finite logits,
value in [-1, 1], run-to-run determinism, and that the legal-masked argmax is a
legal move. Any failure exits non-zero.

Usage:
    python verify_onnx.py --model artifacts/cnn_policy_value.onnx \
        --data data/selfplay/train_500.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS  # 90
ACTION_DIM = BOARD_SIZE * BOARD_SIZE  # 8100
SIDE_PLANES = 2
INVALID_LOGIT = -1e9

DEFAULT_MODEL = "artifacts/cnn_policy_value.onnx"
DEFAULT_DATA = "data/selfplay/train_500.jsonl"


class VerificationError(Exception):
    """A verification assertion failed."""


def resolve_input_spec(session):
    """Read (name, planes, rows, cols) from the model instead of hardcoding 14 planes."""
    meta = session.get_inputs()[0]
    shape = list(meta.shape)
    if len(shape) != 4:
        raise VerificationError(f"expected a 4D board input, got {shape}")
    dims = shape[1:]
    if not all(isinstance(d, int) for d in dims):
        raise VerificationError(f"board input needs static plane/row/col dims, got {shape}")
    return meta.name, int(dims[0]), int(dims[1]), int(dims[2])


def encode_board(board_encoding, planes, side_to_move, rows=BOARD_ROWS, cols=BOARD_COLS):
    # >=16 planes means 14 piece planes + 2 side-to-move planes (same contract as C# _isV5Model)
    board_planes = planes - SIDE_PLANES if planes >= 16 else planes
    flat = np.asarray(board_encoding, dtype=np.float32)
    expected = rows * cols * board_planes
    if flat.size != expected:
        raise VerificationError(
            f"BoardEncoding length {flat.size} != {expected} for a {planes}-plane input")
    tensor = np.zeros((1, planes, rows, cols), dtype=np.float32)
    tensor[0, :board_planes] = flat.reshape(rows * cols, board_planes).T.reshape(board_planes, rows, cols)
    if planes >= 16:
        tensor[0, board_planes + (0 if side_to_move == 1 else 1)] = 1.0
    return tensor


def load_samples(path, max_samples):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= max_samples:
                break
    return samples


def combine_factored(from_logits, to_logits):
    return (np.asarray(from_logits).reshape(BOARD_SIZE, 1)
            + np.asarray(to_logits).reshape(1, BOARD_SIZE)).reshape(ACTION_DIM)


def policy_vector(outputs, output_names, index, factored):
    if factored:
        return combine_factored(outputs[output_names.index("from_logits")][index],
                                outputs[output_names.index("to_logits")][index])
    return np.asarray(outputs[output_names.index("policy_logits")][index])


def mask_illegal(logits, legal_moves):
    masked = np.array(logits, dtype=np.float64, copy=True)
    illegal = np.ones(ACTION_DIM, dtype=bool)
    illegal[list(legal_moves)] = False
    masked[illegal] = INVALID_LOGIT
    return masked


def check_output_contract(output_names, outputs, batch):
    """Return whether the model is factored; raise on any contract violation."""
    factored = "from_logits" in output_names
    if factored:
        expected = {"from_logits": (batch, BOARD_SIZE),
                    "to_logits": (batch, BOARD_SIZE),
                    "value_pred": (batch,)}
    else:
        expected = {"policy_logits": (batch, ACTION_DIM), "value_pred": (batch,)}
    if set(output_names) != set(expected):
        raise VerificationError(f"unexpected output names {output_names}, expected {sorted(expected)}")
    for name, arr in zip(output_names, outputs):
        if tuple(arr.shape) != expected[name]:
            raise VerificationError(f"{name}: shape {arr.shape} != {expected[name]}")
        if not np.all(np.isfinite(arr)):
            raise VerificationError(f"{name}: contains NaN/Inf")
    value = np.asarray(outputs[output_names.index("value_pred")], dtype=np.float64)
    if np.any(value < -1.0 - 1e-6) or np.any(value > 1.0 + 1e-6):
        raise VerificationError(f"value_pred outside [-1, 1]: min={value.min():.4f} max={value.max():.4f}")
    return factored


def check_determinism(output_names, first, second):
    for name, a, b in zip(output_names, first, second):
        if not np.array_equal(a, b):
            delta = float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
            raise VerificationError(f"{name}: repeated inference on identical input differs (max |delta|={delta:.3e})")


def build_parser():
    p = argparse.ArgumentParser(description="Verify the deployed ONNX model with onnxruntime")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default=DEFAULT_DATA, help="Self-play JSONL providing board/legal-move samples")
    p.add_argument("--max-samples", type=int, default=32)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"ERROR: ONNX model not found: {model_path}")
        return 2
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"ERROR: sample data not found: {data_path} (pass --data PATH to an existing self-play JSONL)")
        return 2

    try:
        samples = load_samples(data_path, args.max_samples)
        if not samples:
            raise VerificationError(f"no samples in {data_path}")

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name, planes, rows, cols = resolve_input_spec(session)
        output_names = [o.name for o in session.get_outputs()]

        print("=== ONNX Inference Verification ===")
        print(f"Model: {model_path}")
        print(f"Input: {input_name} [batch, {planes}, {rows}, {cols}]")
        print(f"Output names: {output_names}")

        batch = np.concatenate([
            encode_board(s["BoardEncoding"], planes, s.get("SideToMove", 1), rows, cols)
            for s in samples
        ], axis=0)
        first = session.run(None, {input_name: batch})
        second = session.run(None, {input_name: batch})

        factored = check_output_contract(output_names, first, batch.shape[0])
        check_determinism(output_names, first, second)
        print(f"Checked {batch.shape[0]} samples: shapes/finiteness/value-range/determinism OK")

        top1 = top5 = top10 = 0
        for i, sample in enumerate(samples):
            legal = sample["LegalMoves"]
            selected = sample["SelectedMove"]
            if not legal:
                raise VerificationError(f"sample {i}: empty LegalMoves")
            if selected not in legal:
                raise VerificationError(f"sample {i}: label {selected} is not in LegalMoves")
            masked = mask_illegal(policy_vector(first, output_names, i, factored), legal)
            best = int(np.argmax(masked))
            if best not in legal:
                raise VerificationError(f"sample {i}: masked argmax {best} is not a legal move")
            order = np.argsort(masked)[::-1]
            top1 += int(order[0] == selected)
            top5 += int(selected in order[:5])
            top10 += int(selected in order[:10])

        n = len(samples)
        print(f"Legal-masked argmax is legal for all {n} samples")
        print(f"Label hit rate: top1={top1/n:.3f} top5={top5/n:.3f} top10={top10/n:.3f}")
    except VerificationError as exc:
        print(f"\nVERIFICATION FAILED: {exc}")
        return 1
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"\nVERIFICATION FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
