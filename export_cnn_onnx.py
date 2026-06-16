#!/usr/bin/env python3
"""
Export a trained Chinese Chess CNN policy-value network to ONNX format.

Supports three architectures:
  - cnn_factored_v5: 16 input planes (14 piece + 2 side-to-move), factored from/to heads
  - cnn_factored:    14 input planes, factored from/to heads (legacy)
  - cnn:             14 input planes, flat 8100 policy head (legacy)

Usage:
    python export_cnn_onnx.py \
        --input artifacts/cnn_policy_value.best.pt \
        --output artifacts/cnn_policy_value.onnx

ONNX model (cnn_factored_v5):
    Input:  board_input   [batch, 16, 10, 9]  float32
    Output: from_logits   [batch, 90]          float32
            to_logits     [batch, 90]          float32
            value_pred    [batch]              float32 (tanh -1..1, current player perspective)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

BOARD_PLANES = 14
SIDE_PLANES = 2
INPUT_PLANES = BOARD_PLANES + SIDE_PLANES  # 16
BOARD_ROWS = 10
BOARD_COLS = 9
INPUT_DIM = BOARD_PLANES * BOARD_ROWS * BOARD_COLS  # 1260
ACTION_DIM = BOARD_ROWS * BOARD_COLS * BOARD_ROWS * BOARD_COLS
BOARD_SIZE = BOARD_ROWS * BOARD_COLS  # 90


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


class CNNFactoredPolicyValueNet(nn.Module):
    """V5: 16 input planes (14 piece + 2 side-to-move), factored from/to heads."""
    def __init__(self, in_channels: int = INPUT_PLANES, channels: int = 128,
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
        self.from_fc = nn.Linear(2 * BOARD_SIZE, BOARD_SIZE)
        self.to_fc = nn.Linear(2 * BOARD_SIZE, BOARD_SIZE)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_dropout1 = nn.Dropout(value_dropout)
        self.value_fc1 = nn.Linear(BOARD_SIZE, 256)
        self.value_dropout2 = nn.Dropout(value_dropout)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.tower(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        p = self.policy_dropout(p)
        from_logits = self.from_fc(p)
        to_logits = self.to_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = self.value_dropout1(v)
        v = F.relu(self.value_fc1(v))
        v = self.value_dropout2(v)
        value_pred = torch.tanh(self.value_fc2(v)).squeeze(1)
        return from_logits, to_logits, value_pred


# Legacy flat model (for backward compatibility)
class CNNFullPolicyValueNet(nn.Module):
    """V7: 16 input planes, full 8100 policy head (AlphaZero-style)."""
    def __init__(self, in_channels: int = INPUT_PLANES, channels: int = 64,
                 res_blocks: int = 6, policy_dropout: float = 0.2,
                 value_dropout: float = 0.2, policy_channels: int = 8) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(res_blocks)])
        self.policy_conv = nn.Conv2d(channels, policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_dropout = nn.Dropout(policy_dropout)
        self.policy_fc = nn.Linear(policy_channels * BOARD_SIZE, ACTION_DIM)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_dropout1 = nn.Dropout(value_dropout)
        self.value_fc1 = nn.Linear(BOARD_SIZE, 256)
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


class CNNPolicyValueNet(nn.Module):
    def __init__(self, in_channels: int = BOARD_PLANES, channels: int = 128,
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
        self.policy_fc = nn.Linear(2 * BOARD_SIZE, ACTION_DIM)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_dropout1 = nn.Dropout(value_dropout)
        self.value_fc1 = nn.Linear(BOARD_SIZE, 256)
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

    # V5 models have 16 input planes; older models have 14
    model_planes = checkpoint.get("planes", BOARD_PLANES)
    is_factored = architecture in ("cnn_factored", "cnn_factored_v5")
    is_full_v7 = architecture in ("cnn_full_v7",)

    print(f"  Architecture: {architecture}, channels={channels}, res_blocks={res_blocks}, input_planes={model_planes}")

    if is_full_v7:
        policy_channels = checkpoint.get("policy_channels", 8)
        model = CNNFullPolicyValueNet(
            in_channels=model_planes, channels=channels, res_blocks=res_blocks,
            policy_dropout=policy_dropout, value_dropout=value_dropout,
            policy_channels=policy_channels)
    elif is_factored:
        model = CNNFactoredPolicyValueNet(
            in_channels=model_planes, channels=channels, res_blocks=res_blocks,
            policy_dropout=policy_dropout, value_dropout=value_dropout)
    else:
        model = CNNPolicyValueNet(
            in_channels=model_planes, channels=channels, res_blocks=res_blocks,
            policy_dropout=policy_dropout, value_dropout=value_dropout)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # CNN input shape: [batch, planes, rows, cols]
    dummy_input = torch.zeros(1, model_planes, BOARD_ROWS, BOARD_COLS, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_factored:
        output_names = ["from_logits", "to_logits", "value_pred"]
        dynamic_axes = {
            "board_input": {0: "batch_size"},
            "from_logits": {0: "batch_size"},
            "to_logits": {0: "batch_size"},
            "value_pred": {0: "batch_size"},
        }
    elif is_full_v7:
        output_names = ["policy_logits", "value_pred"]
        dynamic_axes = {
            "board_input": {0: "batch_size"},
            "policy_logits": {0: "batch_size"},
            "value_pred": {0: "batch_size"},
        }
    else:
        output_names = ["policy_logits", "value_pred"]
        dynamic_axes = {
            "board_input": {0: "batch_size"},
            "policy_logits": {0: "batch_size"},
            "value_pred": {0: "batch_size"},
        }

    print(f"Exporting to ONNX: {output_path}")
    print(f"  Output names: {output_names}")

    # Use legacy export API (dynamo=False) for reliable weight inclusion
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["board_input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print("ONNX export complete.")

    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model check: passed")

        import os
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"ONNX file size: {size_mb:.1f} MB")
    except ImportError:
        print("onnx not installed; skipping check.")

    if is_full_v7:
        print(f"\nC# integration note:")
        print(f"  V7 full 8100 model has 2 outputs: policy_logits[1,8100], value_pred[1]")
        print(f"  Input planes: {model_planes} (14 piece + 2 side-to-move)")
        print(f"  Value is from CURRENT PLAYER's perspective (+1 = I'm winning)")
        print(f"  Side-to-move encoding: plane[14]=1 if Red, plane[15]=1 if Black")
        print(f"  policy_channels={policy_channels}")
    elif is_factored:
        print(f"\nC# integration note:")
        print(f"  Factored model has 3 outputs: from_logits[1,90], to_logits[1,90], value_pred[1]")
        print(f"  Input planes: {model_planes} (14 piece + 2 side-to-move)")
        print(f"  Combine: policy_logit[from*90+to] = from_logit[from] + to_logit[to]")
        if architecture == "cnn_factored_v5":
            print(f"  Value is from CURRENT PLAYER's perspective (+1 = I'm winning)")
            print(f"  C# inference: NO need to flip value based on side!")
        else:
            print(f"  Value is from Red's perspective; flip for Black")
        print(f"  Side-to-move encoding: plane[14]=1 if Red, plane[15]=1 if Black")
    else:
        print(f"\nC# BoardEncoder reshaping note:")
        print(f"  Current BoardEncoder.Encode() outputs float[1260] in layout: [square * 14 + plane]")
        print(f"  CNN needs [14, 10, 9] — reshape in C# before feeding to ONNX session.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export CNN checkpoint to ONNX")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    export(build_parser().parse_args())
