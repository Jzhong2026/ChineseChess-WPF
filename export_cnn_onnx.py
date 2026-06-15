#!/usr/bin/env python3
"""
Export a trained Chinese Chess CNN policy-value network to ONNX format.

Usage:
    python export_cnn_onnx.py \
        --input artifacts/cnn_policy_value.pt \
        --output artifacts/cnn_policy_value.onnx

ONNX model:
    Input:  board_input   [batch, 14, 10, 9]  float32
    Output: policy_logits [batch, 8100]        float32
            value_pred    [batch]              float32 (tanh -1..1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

PLANES = 14
BOARD_ROWS = 10
BOARD_COLS = 9
INPUT_DIM = PLANES * BOARD_ROWS * BOARD_COLS
ACTION_DIM = BOARD_ROWS * BOARD_COLS * BOARD_ROWS * BOARD_COLS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class CNNPolicyValueNet(nn.Module):
    def __init__(self, in_channels: int = PLANES, channels: int = 128,
                 res_blocks: int = 8, policy_dropout: float = 0.3,
                 value_dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(res_blocks)])
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_dropout = nn.Dropout(policy_dropout)
        self.policy_fc = nn.Linear(2 * BOARD_ROWS * BOARD_COLS, ACTION_DIM)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_dropout1 = nn.Dropout(value_dropout)
        self.value_fc1 = nn.Linear(BOARD_ROWS * BOARD_COLS, 256)
        self.value_dropout2 = nn.Dropout(value_dropout)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.tower(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        p = self.policy_dropout(p)
        policy_logits = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = self.value_dropout1(v)
        v = F.relu(self.value_fc1(v))
        v = self.value_dropout2(v)
        value_pred = torch.tanh(self.value_fc2(v)).squeeze(1)
        return policy_logits, value_pred


def export(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    channels = checkpoint.get("channels", 128)
    res_blocks = checkpoint.get("res_blocks", 8)
    policy_dropout = checkpoint.get("policy_dropout", 0.3)
    value_dropout = checkpoint.get("value_dropout", 0.3)
    architecture = checkpoint.get("architecture", "cnn")

    print(f"  Architecture: {architecture}, channels={channels}, res_blocks={res_blocks}")

    model = CNNPolicyValueNet(in_channels=PLANES, channels=channels, res_blocks=res_blocks,
                              policy_dropout=policy_dropout, value_dropout=value_dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # CNN input shape: [batch, planes, rows, cols]
    dummy_input = torch.zeros(1, PLANES, BOARD_ROWS, BOARD_COLS, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")
    # Use legacy export API (dynamo=False) for reliable weight inclusion
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
        dynamo=False,
    )
    print("ONNX export complete.")

    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model check: passed")
        
        # Print model size
        import os
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"ONNX file size: {size_mb:.1f} MB")
    except ImportError:
        print("onnx not installed; skipping check.")

    print(f"\nC# BoardEncoder reshaping note:")
    print(f"  Current BoardEncoder.Encode() outputs float[1260] in layout: [square * 14 + plane]")
    print(f"  CNN needs [14, 10, 9] — reshape in C# before feeding to ONNX session.")
    print(f"  See XiangqiNeuralAiService.cs for the reshape logic.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export CNN checkpoint to ONNX")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    export(build_parser().parse_args())
