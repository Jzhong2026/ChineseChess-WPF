#!/usr/bin/env python3
"""
Comprehensive diagnostic: Check every layer of the NN pipeline to find why the AI is weak.

Checks:
1. Side-to-move blindness (is the model confused about sides?)
2. Policy quality: top-K accuracy
3. Value head quality: calibration, correlation with game results
4. Red vs Black policy quality
5. Policy entropy

Usage:
    python diagnose_nn.py --model artifacts/cnn_policy_value.onnx \
        --data data/selfplay/train_500.jsonl
"""
import argparse
import json
import os
import numpy as np
import sys

BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS  # 90
ACTION_DIM = BOARD_SIZE * BOARD_SIZE  # 8100
SIDE_PLANES = 2

DEFAULT_MODEL = "artifacts/cnn_policy_value.onnx"
DEFAULT_DATA = "data/selfplay/train_500.jsonl"


def load_samples(path, max_samples=500):
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


def resolve_input_spec(session):
    """Read the real input shape from the model instead of hardcoding 14 planes."""
    meta = session.get_inputs()[0]
    shape = list(meta.shape)
    if len(shape) != 4 or not all(isinstance(d, int) for d in shape[1:]):
        raise ValueError(f"board input needs static [batch, planes, rows, cols] dims, got {shape}")
    return meta.name, int(shape[1]), int(shape[2]), int(shape[3])


def encode_cnn(board_encoding, planes, side_to_move=1, rows=BOARD_ROWS, cols=BOARD_COLS):
    """Convert square-major flat[90*board_planes] to plane-major [1, planes, rows, cols]."""
    board_planes = planes - SIDE_PLANES if planes >= 16 else planes
    flat = np.array(board_encoding, dtype=np.float32)
    expected = rows * cols * board_planes
    if flat.size != expected:
        raise ValueError(f"BoardEncoding length {flat.size} != {expected} for {planes} input planes")
    tensor = np.zeros((1, planes, rows, cols), dtype=np.float32)
    tensor[0, :board_planes] = flat.reshape(rows * cols, board_planes).T.reshape(board_planes, rows, cols)
    if planes >= 16:
        tensor[0, board_planes + (0 if side_to_move == 1 else 1)] = 1.0
    return tensor


def value_to_player_perspective(value, side_to_move, planes):
    # 16-plane models already emit current-player value (C# XiangqiNeuralAiService does not flip)
    return value if planes >= 16 else value * side_to_move


def value_perspective_label(planes):
    return "current player perspective" if planes >= 16 else "Red's perspective"


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def build_parser():
    p = argparse.ArgumentParser(description="Diagnose the deployed Chinese Chess NN")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default=DEFAULT_DATA, help="Self-play JSONL providing board/legal-move samples")
    p.add_argument("--max-samples", type=int, default=500)
    return p


def main(argv=None):
    import onnxruntime as ort

    args = build_parser().parse_args(argv)
    onnx_path = args.model
    if not os.path.isfile(onnx_path):
        print(f"ERROR: ONNX model not found: {onnx_path}")
        return 2
    if not os.path.isfile(args.data):
        print(f"ERROR: sample data not found: {args.data} (pass --data PATH to an existing self-play JSONL)")
        return 2

    print("=" * 70)
    print("CHINESE CHESS NN AI DIAGNOSTIC REPORT")
    print("=" * 70)

    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name, planes, rows, cols = resolve_input_spec(session)
    output_names = [o.name for o in session.get_outputs()]
    is_factored = "from_logits" in output_names
    print(f"\nONNX model: {onnx_path}")
    print(f"Model type: {'Factored (from/to heads)' if is_factored else 'Flat (8100 policy)'}")
    print(f"Input: {input_name} [batch, {planes}, {rows}, {cols}]")
    print(f"Output names: {output_names}")

    # Load samples
    samples = load_samples(args.data, max_samples=args.max_samples)
    if not samples:
        print(f"ERROR: no samples in {args.data}")
        return 2
    print(f"Loaded {len(samples)} samples for analysis")

    def predict(sample):
        tensor = encode_cnn(sample["BoardEncoding"], planes, sample["SideToMove"], rows, cols)
        return session.run(None, {input_name: tensor})

    def value_of(outputs):
        return outputs[2][0] if is_factored else outputs[1][0]

    def combined_of(outputs):
        if is_factored:
            from_logits = outputs[0][0]
            to_logits = outputs[1][0]
            return (from_logits.reshape(BOARD_SIZE, 1) + to_logits.reshape(1, BOARD_SIZE)).reshape(ACTION_DIM)
        return outputs[0][0]

    def masked_of(outputs, legal):
        masked = np.array(combined_of(outputs), dtype=np.float64, copy=True)
        illegal_mask = np.ones(ACTION_DIM, dtype=bool)
        illegal_mask[legal] = False
        masked[illegal_mask] = -1e9
        return masked

    # ─── Check 1: Side-to-move analysis ──────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 1: SIDE-TO-MOVE BLINDNESS")
    print("=" * 70)

    red_samples = [s for s in samples if s["SideToMove"] == 1]
    black_samples = [s for s in samples if s["SideToMove"] == -1]
    print(f"Red-to-move samples: {len(red_samples)}")
    print(f"Black-to-move samples: {len(black_samples)}")

    # Check if the model can distinguish Red vs Black to-move positions
    # Run inference on a Red and Black sample and compare value predictions
    if red_samples and black_samples:
        red_val_preds = [value_of(predict(s)) for s in red_samples[:50]]
        black_val_preds = [value_of(predict(s)) for s in black_samples[:50]]

        print(f"\nValue predictions ({value_perspective_label(planes)}):")
        print(f"  Red-to-move:   mean={np.mean(red_val_preds):.4f}, std={np.std(red_val_preds):.4f}")
        print(f"  Black-to-move: mean={np.mean(black_val_preds):.4f}, std={np.std(black_val_preds):.4f}")

        # Value targets analysis
        red_val_targets = [s["Result"] for s in red_samples[:50]]
        black_val_targets = [s["Result"] for s in black_samples[:50]]
        print(f"\nValue targets (Red's perspective):")
        print(f"  Red-to-move:   mean={np.mean(red_val_targets):.4f}")
        print(f"  Black-to-move: mean={np.mean(black_val_targets):.4f}")

        # For 16-plane models the value is the mover's own score, so a Black win is +1.
        black_wins_positive = planes >= 16
        black_wins = [s for s in black_samples[:50] if s["Result"] == -1]
        red_wins = [s for s in black_samples[:50] if s["Result"] == 1]
        if black_wins:
            bw_vals = [value_of(predict(s)) for s in black_wins[:20]]
            expect = "> 0" if black_wins_positive else "< 0"
            print(f"\n  Black-to-move, Black won: val_pred mean={np.mean(bw_vals):.4f} (should be {expect})")

        if red_wins:
            rw_vals = [value_of(predict(s)) for s in red_wins[:20]]
            expect = "< 0" if black_wins_positive else "> 0"
            print(f"  Black-to-move, Red won:   val_pred mean={np.mean(rw_vals):.4f} (should be {expect})")

    # ─── Check 2: Policy quality ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 2: POLICY QUALITY")
    print("=" * 70)

    top1_hit = top5_hit = top10_hit = 0
    from_hit = to_hit = 0
    n = min(200, len(samples))

    for s in samples[:n]:
        outputs = predict(s)

        selected = s["SelectedMove"]
        from_target = selected // BOARD_SIZE
        to_target = selected % BOARD_SIZE

        if is_factored:
            from_hit += (np.argmax(outputs[0][0]) == from_target)
            to_hit += (np.argmax(outputs[1][0]) == to_target)

        masked = masked_of(outputs, s["LegalMoves"])

        top1 = np.argmax(masked)
        top5 = np.argsort(masked)[-5:][::-1]
        top10 = np.argsort(masked)[-10:][::-1]

        top1_hit += (top1 == selected)
        top5_hit += (selected in top5)
        top10_hit += (selected in top10)

    print(f"Evaluated on {n} samples:")
    print(f"  Top-1 accuracy (legal):  {top1_hit/n*100:.1f}%")
    print(f"  Top-5 accuracy (legal):  {top5_hit/n*100:.1f}%")
    print(f"  Top-10 accuracy (legal): {top10_hit/n*100:.1f}%")
    if is_factored:
        print(f"  From head accuracy:      {from_hit/n*100:.1f}%")
        print(f"  To head accuracy:        {to_hit/n*100:.1f}%")

    # ─── Check 3: Value calibration ──────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 3: VALUE CALIBRATION")
    print("=" * 70)

    val_preds = []
    val_targets = []
    for s in samples[:200]:
        outputs = predict(s)
        side_to_move = s["SideToMove"]
        val_preds.append(value_to_player_perspective(value_of(outputs), side_to_move, planes))
        val_targets.append(s["Result"] * side_to_move)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # MSE
    mse = np.mean((val_preds - val_targets) ** 2)
    # Correlation
    corr = np.corrcoef(val_preds, val_targets)[0, 1] if len(val_preds) > 1 else 0

    print(f"Value predictions (current player perspective):")
    print(f"  MSE: {mse:.4f}")
    print(f"  Correlation: {corr:.4f}")
    print(f"  Mean pred: {np.mean(val_preds):.4f}, Mean target: {np.mean(val_targets):.4f}")

    # By game result
    for label, mask in [("Win", val_targets > 0), ("Loss", val_targets < 0), ("Draw", val_targets == 0)]:
        if mask.any():
            print(f"  {label}: pred_mean={val_preds[mask].mean():.4f} (target={val_targets[mask].mean():.1f})")

    # ─── Check 4: Red vs Black policy quality ────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 4: RED VS BLACK POLICY QUALITY")
    print("=" * 70)

    for side_label, side_val in [("Red", 1), ("Black", -1)]:
        side_samples = [s for s in samples[:200] if s["SideToMove"] == side_val]
        if not side_samples:
            print(f"  {side_label}: no samples")
            continue

        t1 = t5 = t10 = 0
        for s in side_samples:
            masked = masked_of(predict(s), s["LegalMoves"])
            selected = s["SelectedMove"]
            t1 += (np.argmax(masked) == selected)
            t5 += (selected in np.argsort(masked)[-5:][::-1])
            t10 += (selected in np.argsort(masked)[-10:][::-1])

        n_side = len(side_samples)
        print(f"  {side_label} (n={n_side}): Top1={t1/n_side*100:.1f}% Top5={t5/n_side*100:.1f}% Top10={t10/n_side*100:.1f}%")

    # ─── Check 5: Entropy of policy distribution ─────────────────────
    print("\n" + "=" * 70)
    print("CHECK 5: POLICY ENTROPY (how focused/confident is the model?)")
    print("=" * 70)

    entropies = []
    for s in samples[:100]:
        masked = masked_of(predict(s), s["LegalMoves"])
        probs = softmax(masked)
        # Entropy of legal move distribution
        legal_probs = probs[s["LegalMoves"]]
        entropy = -np.sum(legal_probs * np.log(legal_probs + 1e-10))
        entropies.append(entropy)

    print(f"  Mean entropy: {np.mean(entropies):.2f}")
    print(f"  Min entropy:  {np.min(entropies):.2f}")
    print(f"  Max entropy:  {np.max(entropies):.2f}")
    print(f"  Random baseline entropy: ~{np.log(40):.2f} (for ~40 legal moves)")
    print(f"  Lower entropy = more confident/focused policy")

    # ─── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print("""
Key issues to check:
1. If Red vs Black quality differs significantly → model is side-confused
2. If value calibration is poor → MCTS won't work well
3. If policy entropy is too high → model is essentially random
4. If Top-1 is very low but Top-10 is ok → model has some signal but needs search
5. If the model doesn't know whose turn it is → add side-to-move planes
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
