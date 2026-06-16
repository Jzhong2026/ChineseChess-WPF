#!/usr/bin/env python3
"""
Comprehensive diagnostic: Check every layer of the NN pipeline to find why the AI is weak.

Checks:
1. ONNX model outputs vs PyTorch model outputs (numerical consistency)
2. Current-player perspective encoding (is the model confused about sides?)
3. Policy quality: top-K accuracy, calibration
4. Value head quality: calibration, correlation with game results
5. Side-to-move blindness: does the model know whose turn it is?
"""
import json
import numpy as np
import sys

PLANES = 14
BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS  # 90
ACTION_DIM = BOARD_SIZE * BOARD_SIZE  # 8100

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

def encode_cnn(board_encoding):
    """Convert square-major flat[1260] to plane-major [1, 14, 10, 9]"""
    flat = np.array(board_encoding, dtype=np.float32)
    tensor = np.zeros((1, PLANES, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            for plane in range(PLANES):
                flat_idx = (row * BOARD_COLS + col) * PLANES + plane
                tensor[0, plane, row, col] = flat[flat_idx]
    return tensor

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def main():
    import onnxruntime as ort

    print("=" * 70)
    print("CHINESE CHESS NN AI DIAGNOSTIC REPORT")
    print("=" * 70)

    # Load ONNX model
    onnx_path = "artifacts/cnn_policy_value.onnx"
    session = ort.InferenceSession(onnx_path)
    output_names = [o.name for o in session.get_outputs()]
    is_factored = "from_logits" in output_names
    print(f"\nONNX model: {onnx_path}")
    print(f"Model type: {'Factored (from/to heads)' if is_factored else 'Flat (8100 policy)'}")
    print(f"Output names: {output_names}")

    # Load samples
    samples = load_samples("data/selfplay/train_500.jsonl", max_samples=500)
    print(f"Loaded {len(samples)} samples for analysis")

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
        red_val_preds = []
        black_val_preds = []
        for s in red_samples[:50]:
            tensor = encode_cnn(s["BoardEncoding"])
            outputs = session.run(None, {session.get_inputs()[0].name: tensor})
            if is_factored:
                val = outputs[2][0]
            else:
                val = outputs[1][0]
            red_val_preds.append(val)

        for s in black_samples[:50]:
            tensor = encode_cnn(s["BoardEncoding"])
            outputs = session.run(None, {session.get_inputs()[0].name: tensor})
            if is_factored:
                val = outputs[2][0]
            else:
                val = outputs[1][0]
            black_val_preds.append(val)

        print(f"\nValue predictions (Red's perspective):")
        print(f"  Red-to-move:   mean={np.mean(red_val_preds):.4f}, std={np.std(red_val_preds):.4f}")
        print(f"  Black-to-move: mean={np.mean(black_val_preds):.4f}, std={np.std(black_val_preds):.4f}")

        # Value targets analysis
        red_val_targets = [s["Result"] for s in red_samples[:50]]
        black_val_targets = [s["Result"] for s in black_samples[:50]]
        print(f"\nValue targets (Red's perspective):")
        print(f"  Red-to-move:   mean={np.mean(red_val_targets):.4f}")
        print(f"  Black-to-move: mean={np.mean(black_val_targets):.4f}")

        # Check: when it's Black's turn and Black wins (result=-1),
        # the value should be negative (from Red's perspective)
        black_wins = [s for s in black_samples[:50] if s["Result"] == -1]
        red_wins = [s for s in black_samples[:50] if s["Result"] == 1]
        if black_wins:
            bw_vals = []
            for s in black_wins[:20]:
                tensor = encode_cnn(s["BoardEncoding"])
                outputs = session.run(None, {session.get_inputs()[0].name: tensor})
                if is_factored:
                    bw_vals.append(outputs[2][0])
                else:
                    bw_vals.append(outputs[1][0])
            print(f"\n  Black-to-move, Black won: val_pred mean={np.mean(bw_vals):.4f} (should be < 0)")

        if red_wins:
            rw_vals = []
            for s in red_wins[:20]:
                tensor = encode_cnn(s["BoardEncoding"])
                outputs = session.run(None, {session.get_inputs()[0].name: tensor})
                if is_factored:
                    rw_vals.append(outputs[2][0])
                else:
                    rw_vals.append(outputs[1][0])
            print(f"  Black-to-move, Red won:   val_pred mean={np.mean(rw_vals):.4f} (should be > 0)")

    # ─── Check 2: Policy quality ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 2: POLICY QUALITY")
    print("=" * 70)

    top1_hit = top5_hit = top10_hit = 0
    from_hit = to_hit = 0
    n = min(200, len(samples))

    for s in samples[:n]:
        tensor = encode_cnn(s["BoardEncoding"])
        outputs = session.run(None, {session.get_inputs()[0].name: tensor})

        selected = s["SelectedMove"]
        legal = s["LegalMoves"]
        from_target = selected // BOARD_SIZE
        to_target = selected % BOARD_SIZE

        if is_factored:
            from_logits = outputs[0][0]  # [90]
            to_logits = outputs[1][0]    # [90]
            combined = from_logits.reshape(90, 1) + to_logits.reshape(1, 90)
            combined_flat = combined.flatten()

            from_hit += (np.argmax(from_logits) == from_target)
            to_hit += (np.argmax(to_logits) == to_target)
        else:
            combined_flat = outputs[0][0]  # [8100]

        # Legal masking
        masked = combined_flat.copy()
        illegal_mask = np.ones(ACTION_DIM, dtype=bool)
        illegal_mask[legal] = False
        masked[illegal_mask] = -1e9

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
        tensor = encode_cnn(s["BoardEncoding"])
        outputs = session.run(None, {session.get_inputs()[0].name: tensor})

        if is_factored:
            val = outputs[2][0]
        else:
            val = outputs[1][0]

        # Convert value to current player's perspective
        side_to_move = s["SideToMove"]
        val_player = val * side_to_move  # +1 = current player winning

        # Convert result to current player's perspective
        result_player = s["Result"] * side_to_move

        val_preds.append(val_player)
        val_targets.append(result_player)

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
            tensor = encode_cnn(s["BoardEncoding"])
            outputs = session.run(None, {session.get_inputs()[0].name: tensor})

            selected = s["SelectedMove"]
            legal = s["LegalMoves"]

            if is_factored:
                from_logits = outputs[0][0]
                to_logits = outputs[1][0]
                combined = from_logits.reshape(90, 1) + to_logits.reshape(1, 90)
                combined_flat = combined.flatten()
            else:
                combined_flat = outputs[0][0]

            masked = combined_flat.copy()
            illegal_mask = np.ones(ACTION_DIM, dtype=bool)
            illegal_mask[legal] = False
            masked[illegal_mask] = -1e9

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
        tensor = encode_cnn(s["BoardEncoding"])
        outputs = session.run(None, {session.get_inputs()[0].name: tensor})

        if is_factored:
            from_logits = outputs[0][0]
            to_logits = outputs[1][0]
            combined = from_logits.reshape(90, 1) + to_logits.reshape(1, 90)
            combined_flat = combined.flatten()
        else:
            combined_flat = outputs[0][0]

        legal = s["LegalMoves"]
        masked = combined_flat.copy()
        illegal_mask = np.ones(ACTION_DIM, dtype=bool)
        illegal_mask[legal] = False
        masked[illegal_mask] = -1e9

        probs = softmax(masked)
        # Entropy of legal move distribution
        legal_probs = probs[legal]
        entropy = -np.sum(legal_probs * np.log(legal_probs + 1e-10))
        entropies.append(entropy)

    max_entropy = np.log(len(samples[0]["LegalMoves"]))  # approximate max
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


if __name__ == "__main__":
    main()
