#!/usr/bin/env python3
"""
Quick verification: Run the ONNX model in Python to check it works correctly.
Auto-detects factored (from/to) vs flat (8100) policy architecture.
"""
import json
import numpy as np
import onnxruntime as ort

PLANES = 14
BOARD_ROWS = 10
BOARD_COLS = 9
ACTION_DIM = 8100
BOARD_SIZE = BOARD_ROWS * BOARD_COLS  # 90

# Load a sample from the training data
with open("data/selfplay/train_500.jsonl", "r", encoding="utf-8") as f:
    sample = json.loads(f.readline().strip())

# Convert square-major flat[1260] to plane-major [1, 14, 10, 9]
flat = np.array(sample["BoardEncoding"], dtype=np.float32)
board_tensor = np.zeros((1, PLANES, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
for row in range(BOARD_ROWS):
    for col in range(BOARD_COLS):
        for plane in range(PLANES):
            flat_idx = (row * BOARD_COLS + col) * PLANES + plane
            board_tensor[0, plane, row, col] = flat[flat_idx]

# Run ONNX inference
session = ort.InferenceSession("artifacts/cnn_policy_value.onnx")
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
outputs = session.run(None, {input_name: board_tensor})

print("=== ONNX Inference Verification ===")
print(f"Input shape: {board_tensor.shape}")
print(f"Output names: {output_names}")

selected_move = sample["SelectedMove"]
legal_moves = sample["LegalMoves"]
from_target = selected_move // BOARD_SIZE
to_target = selected_move % BOARD_SIZE

is_factored = "from_logits" in output_names

if is_factored:
    from_logits = outputs[0]  # [1, 90]
    to_logits = outputs[1]    # [1, 90]
    value_pred = outputs[2]   # [1]

    print(f"From logits shape: {from_logits.shape}")
    print(f"To logits shape: {to_logits.shape}")
    print(f"Value pred shape: {value_pred.shape}")
    print(f"Value prediction: {value_pred[0]:.4f}")

    # Combine: policy[from*90+to] = from_logits[from] + to_logits[to]
    combined = from_logits[0].reshape(90, 1) + to_logits[0].reshape(1, 90)  # [90, 90]
    combined_flat = combined.flatten()  # [8100]

    # Legal masking
    masked_logits = combined_flat.copy()
    illegal_mask = np.ones(ACTION_DIM, dtype=bool)
    illegal_mask[legal_moves] = False
    masked_logits[illegal_mask] = -1e9

    predicted_from = np.argmax(from_logits[0])
    predicted_to = np.argmax(to_logits[0])
    predicted_legal_move = np.argmax(masked_logits)

    print(f"\nLabel: from={from_target}, to={to_target}, action={selected_move}")
    print(f"Predicted from: {predicted_from} {'✅' if predicted_from == from_target else '❌'}")
    print(f"Predicted to:   {predicted_to} {'✅' if predicted_to == to_target else '❌'}")
    print(f"Predicted legal move: {predicted_legal_move} {'✅' if predicted_legal_move == selected_move else '❌'}")

    # Top-5 legal moves
    top_indices = np.argsort(masked_logits)[-5:][::-1]
    print(f"\nTop-5 legal moves: {top_indices.tolist()}")
    print(f"Top-5 logits: {[f'{masked_logits[i]:.2f}' for i in top_indices]}")

    # Check if label is in top-K
    for k in [1, 5, 10]:
        top_k = np.argsort(masked_logits)[-k:][::-1]
        hit = selected_move in top_k
        print(f"Label in Top-{k}: {'✅' if hit else '❌'}")

    # From/to head softmax probabilities
    from_probs = np.exp(from_logits[0] - from_logits[0].max())
    from_probs /= from_probs.sum()
    to_probs = np.exp(to_logits[0] - to_logits[0].max())
    to_probs /= to_probs.sum()
    print(f"\nFrom head: P(label from)={from_probs[from_target]:.4f}")
    print(f"To head:   P(label to)={to_probs[to_target]:.4f}")

else:
    # Legacy flat model
    policy_logits = outputs[0]  # [1, 8100]
    value_pred = outputs[1]     # [1]

    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value pred shape: {value_pred.shape}")
    print(f"Value prediction: {value_pred[0]:.4f}")

    masked_logits = policy_logits[0].copy()
    illegal_mask = np.ones(ACTION_DIM, dtype=bool)
    illegal_mask[legal_moves] = False
    masked_logits[illegal_mask] = -1e9

    predicted_move = np.argmax(policy_logits[0])
    predicted_legal_move = np.argmax(masked_logits)

    print(f"\nSelected move (label): {selected_move}")
    print(f"Predicted move (unmasked): {predicted_move}")
    print(f"Predicted move (legal only): {predicted_legal_move}")
    print(f"Predicted == Label: {'✅' if predicted_legal_move == selected_move else '❌'}")

    top_indices = np.argsort(masked_logits)[-5:][::-1]
    print(f"\nTop-5 legal moves: {top_indices.tolist()}")
    print(f"Top-5 logits: {[f'{masked_logits[i]:.2f}' for i in top_indices]}")

print("\n✅ ONNX inference pipeline verified!")
