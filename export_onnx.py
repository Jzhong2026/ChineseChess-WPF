#!/usr/bin/env python3
"""
Export a trained Chinese Chess policy-value network to ONNX format.

Usage:
    python export_onnx.py --input artifacts/chinese_chess_policy_value.pt \
                          --output artifacts/chinese_chess_policy_value.onnx

The exported ONNX model has:
    Input:  board_input  [batch, 1260]  (float32)
    Output: policy_logits [batch, 8100] (float32)
            value_pred    [batch]       (float32, tanh range -1..1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


# ─── Model definition (must match train_chinese_chess_policy_value.py) ────────

INPUT_DIM = 1260
ACTION_DIM = 8100


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, action_dim: int) -> None:
        super().__init__()
        from typing import List
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(dim, action_dim)
        self.value_head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        policy_logits = self.policy_head(h)
        value_pred = torch.tanh(self.value_head(h)).squeeze(1)
        return policy_logits, value_pred


# ─── Export ───────────────────────────────────────────────────────────────────

def export(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    hidden_dim: int = checkpoint.get("hidden_dim", 512)
    num_layers: int = checkpoint.get("num_layers", 3)
    input_dim: int = checkpoint.get("input_dim", INPUT_DIM)
    action_dim: int = checkpoint.get("action_dim", ACTION_DIM)

    print(f"  Architecture: input={input_dim}, hidden={hidden_dim}, layers={num_layers}, actions={action_dim}")

    model = PolicyValueNet(input_dim, hidden_dim, num_layers, action_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["board_input"],
        output_names=["policy_logits", "value_pred"],
        dynamic_axes={
            "board_input": {0: "batch_size"},
            "policy_logits": {0: "batch_size"},
            "value_pred": {0: "batch_size"},
        },
    )
    print("ONNX export complete.")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")
    except ImportError:
        print("onnx package not installed; skipping model check (pip install onnx to verify).")

    print(f"\nModel outputs:")
    print(f"  policy_logits: float32[batch, {action_dim}]")
    print(f"  value_pred:    float32[batch]  (tanh, range -1..1)")
    print(f"\nC# usage example:")
    print(f"""  var session = new InferenceSession("{output_path.name}");
  var input = new DenseTensor<float>(boardEncoding, new[] {{ 1, {input_dim} }});
  var result = session.Run(new[] {{ NamedOnnxValue.CreateFromTensor("board_input", input) }});
  var policyLogits = result[0].AsTensor<float>();
  var valuePred = result[1].AsTensor<float>()[0];""")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export PyTorch checkpoint to ONNX")
    p.add_argument("--input", required=True, help="Input .pt checkpoint path")
    p.add_argument("--output", required=True, help="Output .onnx file path")
    return p


if __name__ == "__main__":
    export(build_parser().parse_args())
