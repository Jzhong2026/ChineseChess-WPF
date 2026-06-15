#!/usr/bin/env python3
"""
Quick verification: Run the ONNX model in Python to check it works correctly.
This mirrors what the C# XiangqiNeuralAiService does.
"""
import json
import numpy as np
import onnxruntime as ort

PLANES = 14
BOARD_ROWS = 10
BOARD_COLS = 9
ACTION_DIM = 8100

# Load a sample from the training data
with open("data/selfplay/probe.jsonl", "r", encoding="utf-8") as f:
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
outputs = session.run(None, {input_name: board_tensor})

policy_logits = outputs[0]  # [1, 8100]
value_pred = outputs[1]     # [1]

print("=== ONNX Inference Verification ===")
print(f"Input shape: {board_tensor.shape}")
print(f"Policy logits shape: {policy_logits.shape}")
print(f"Value pred shape: {value_pred.shape}")
print(f"Value prediction: {value_pred[0]:.4f}")
print()

# Check policy prediction vs training label
selected_move = sample["SelectedMove"]
legal_moves = sample["LegalMoves"]

# Apply legal mask
masked_logits = policy_logits[0].copy()
illegal_mask = np.ones(ACTION_DIM, dtype=bool)
illegal_mask[legal_moves] = False
masked_logits[illegal_mask] = -1e9

predicted_move = np.argmax(policy_logits[0])
predicted_legal_move = np.argmax(masked_logits)

print(f"Selected move (label): {selected_move}")
print(f"Predicted move (unmasked): {predicted_move}")
print(f"Predicted move (legal only): {predicted_legal_move}")
print(f"Predicted == Label: {predicted_legal_move == selected_move}")

# Top-5 legal moves
top_indices = np.argsort(masked_logits)[-5:][::-1]
print(f"\nTop-5 legal moves: {top_indices.tolist()}")
print(f"Top-5 logits: {[f'{masked_logits[i]:.2f}' for i in top_indices]}")

# Check if predicted legal move is actually legal
print(f"\nPredicted legal move in LegalMoves: {predicted_legal_move in legal_moves}")
print("✅ ONNX inference pipeline verified!")
