#!/usr/bin/env python3
"""
Train a Chinese Chess CNN policy-value network from self-play JSONL data.

Architecture upgrade over train_chinese_chess_policy_value.py:
  - Input is reshaped to [batch, 14, 10, 9] (planes × rows × cols)
  - CNN backbone with residual blocks captures spatial board patterns
  - Policy head: Global average pool → FC → 8100 logits
  - Value head:  Global average pool → FC → tanh

Usage:
    python train_cnn_policy_value.py \
        --input data/selfplay/train.jsonl \
        --output artifacts/cnn_policy_value.pt \
        --epochs 30 \
        --batch-size 256 \
        --channels 128 \
        --res-blocks 8
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ─── Constants ────────────────────────────────────────────────────────────────
PLANES = 14       # 2 sides × 7 piece types
BOARD_ROWS = 10
BOARD_COLS = 9
INPUT_DIM = PLANES * BOARD_ROWS * BOARD_COLS   # 1260
ACTION_DIM = BOARD_ROWS * BOARD_COLS * BOARD_ROWS * BOARD_COLS  # 8100
INVALID_LOGIT = -1e9


# ─── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class SelfPlaySample:
    board_encoding: List[float]   # flat float[1260]
    legal_moves: List[int]
    selected_move: int
    result: float
    value_weight: float
    policy_weight: float


class ChineseChessDataset(Dataset):
    def __init__(self, samples: Sequence[SelfPlaySample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        # C# BoardEncoder outputs square-major: flat[(row*9+col)*14 + plane]
        # CNN expects plane-major:  [plane, row, col]
        flat = torch.tensor(s.board_encoding, dtype=torch.float32)
        # flat: [1260] → [90, 14] → permute → [14, 90] → [14, 10, 9]
        features = flat.view(BOARD_ROWS * BOARD_COLS, PLANES).permute(1, 0).view(PLANES, BOARD_ROWS, BOARD_COLS)
        policy_target = torch.tensor(s.selected_move, dtype=torch.long)
        value_target = torch.tensor(s.result, dtype=torch.float32)
        legal_mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
        legal_mask[s.legal_moves] = True
        value_weight = torch.tensor(s.value_weight, dtype=torch.float32)
        policy_weight = torch.tensor(s.policy_weight, dtype=torch.float32)
        return features, policy_target, value_target, legal_mask, value_weight, policy_weight


# ─── Model ────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Standard ResNet basic block with batch normalisation."""

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
    """
    AlphaZero-style CNN for Chinese Chess.

    Board representation: [batch, 14, 10, 9]
    Policy output:        [batch, 8100] raw logits (apply legal mask before softmax)
    Value output:         [batch]       tanh ∈ (-1, 1)
    """

    def __init__(self, in_channels: int = PLANES, channels: int = 128,
                 res_blocks: int = 8, policy_dropout: float = 0.3,
                 value_dropout: float = 0.3) -> None:
        super().__init__()

        # Stem: project input planes to `channels` feature maps
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Tower of residual blocks
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_dropout = nn.Dropout(policy_dropout)
        self.policy_fc = nn.Linear(2 * BOARD_ROWS * BOARD_COLS, ACTION_DIM)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_dropout1 = nn.Dropout(value_dropout)
        self.value_fc1 = nn.Linear(BOARD_ROWS * BOARD_COLS, 256)
        self.value_dropout2 = nn.Dropout(value_dropout)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, 14, 10, 9]
        x = self.stem(x)
        x = self.tower(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))  # [batch, 2, 10, 9]
        p = p.flatten(1)                                   # [batch, 180]
        p = self.policy_dropout(p)
        policy_logits = self.policy_fc(p)                  # [batch, 8100]

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))     # [batch, 1, 10, 9]
        v = v.flatten(1)                                   # [batch, 90]
        v = self.value_dropout1(v)
        v = F.relu(self.value_fc1(v))                     # [batch, 256]
        v = self.value_dropout2(v)
        value_pred = torch.tanh(self.value_fc2(v)).squeeze(1)  # [batch]

        return policy_logits, value_pred


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_samples(path: Path, include_unfinished: bool) -> List[SelfPlaySample]:
    samples: List[SelfPlaySample] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            unfinished = bool(obj.get("Unfinished", False))
            if unfinished and not include_unfinished:
                skipped += 1
                continue

            use_value = bool(obj.get("UseForValueTraining", True))
            use_policy = bool(obj.get("UseForPolicyTraining", True))
            if not use_value and not use_policy:
                skipped += 1
                continue

            board_encoding = obj.get("BoardEncoding")
            legal_moves = obj.get("LegalMoves")
            selected_move = obj.get("SelectedMove")
            result = obj.get("Result")

            if not isinstance(board_encoding, list) or len(board_encoding) != INPUT_DIM:
                raise ValueError(f"Line {line_no}: BoardEncoding must be float[{INPUT_DIM}]")
            if not isinstance(legal_moves, list) or len(legal_moves) == 0:
                raise ValueError(f"Line {line_no}: LegalMoves must be a non-empty int[]")
            if not isinstance(selected_move, int) or not (0 <= selected_move < ACTION_DIM):
                raise ValueError(f"Line {line_no}: SelectedMove must be in [0, {ACTION_DIM - 1}]")

            legal_moves_int = [int(m) for m in legal_moves]
            if selected_move not in legal_moves_int:
                raise ValueError(f"Line {line_no}: SelectedMove not in LegalMoves")

            result_value = float(result)
            if result_value not in (-1.0, 0.0, 1.0):
                raise ValueError(f"Line {line_no}: Result must be -1/0/1, got {result_value}")

            samples.append(SelfPlaySample(
                board_encoding=[float(v) for v in board_encoding],
                legal_moves=legal_moves_int,
                selected_move=selected_move,
                result=result_value,
                value_weight=float(obj.get("ValueWeight", 1.0)),
                policy_weight=float(obj.get("PolicyWeight", 1.0)),
            ))

    if not samples:
        raise ValueError("No valid samples loaded from input JSONL")
    print(f"Loaded {len(samples)} samples ({skipped} skipped).")
    return samples


def split_dataset(
    samples: List[SelfPlaySample], val_ratio: float, seed: int
) -> Tuple[List[SelfPlaySample], List[SelfPlaySample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]
    if not train_samples:
        train_samples, val_samples = val_samples, []
    return train_samples, val_samples


# ─── Training ─────────────────────────────────────────────────────────────────

def masked_policy_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~legal_mask, INVALID_LOGIT)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, value_loss_weight: float) -> dict:
    model.eval()
    total_loss = total_count = correct = legal_correct = 0.0
    legal_top5_correct = legal_top10_correct = value_mse_sum = 0.0

    for x, policy_target, value_target, legal_mask, value_weight, policy_weight in loader:
        x, policy_target, value_target, legal_mask = (
            x.to(device), policy_target.to(device),
            value_target.to(device), legal_mask.to(device),
        )
        value_weight = value_weight.to(device)
        policy_weight = policy_weight.to(device)

        policy_logits, value_pred = model(x)
        masked_logits = masked_policy_logits(policy_logits, legal_mask)

        # Use weighted loss for consistency with training
        per_sample_p_loss = F.cross_entropy(masked_logits, policy_target, reduction="none")
        p_loss = (per_sample_p_loss * policy_weight).mean()
        per_sample_v_loss = (value_pred - value_target) ** 2
        v_loss = (per_sample_v_loss * value_weight).mean()
        loss = p_loss + value_loss_weight * v_loss

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        correct += (policy_logits.argmax(1) == policy_target).sum().item()
        legal_correct += (masked_logits.argmax(1) == policy_target).sum().item()
        value_mse_sum += v_loss.item() * bs

        # Top-5 and Top-10 accuracy on masked logits
        _, top5_idx = masked_logits.topk(5, dim=1)
        _, top10_idx = masked_logits.topk(10, dim=1)
        legal_top5_correct += (top5_idx == policy_target.unsqueeze(1)).any(dim=1).sum().item()
        legal_top10_correct += (top10_idx == policy_target.unsqueeze(1)).any(dim=1).sum().item()

    n = max(1, int(total_count))
    return {
        "val_loss": total_loss / n,
        "val_policy_acc": correct / n,
        "val_value_mse": value_mse_sum / n,
        "val_legal_acc": legal_correct / n,
        "val_legal_top5": legal_top5_correct / n,
        "val_legal_top10": legal_top10_correct / n,
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    all_samples = load_samples(Path(args.input), include_unfinished=args.include_unfinished)
    train_samples, val_samples = split_dataset(all_samples, args.val_ratio, args.seed)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_loader = DataLoader(
        ChineseChessDataset(train_samples), batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = (
        DataLoader(ChineseChessDataset(val_samples), batch_size=args.batch_size, shuffle=False, num_workers=0)
        if val_samples else None
    )

    model = CNNPolicyValueNet(
        in_channels=PLANES,
        channels=args.channels,
        res_blocks=args.res_blocks,
        policy_dropout=args.policy_dropout,
        value_dropout=args.value_dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: CNN channels={args.channels}, res_blocks={args.res_blocks}, params={total_params:,}")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss()
    epoch_metrics: List[dict] = []

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_policy_loss = total_value_loss = total_count = 0.0

        for x, policy_target, value_target, legal_mask, value_weight, policy_weight in train_loader:
            x = x.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            legal_mask = legal_mask.to(device)
            value_weight = value_weight.to(device)
            policy_weight = policy_weight.to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = model(x)
            masked_logits = masked_policy_logits(policy_logits, legal_mask)

            # Weighted policy loss: give more weight to moves from deeper search
            per_sample_p_loss = F.cross_entropy(masked_logits, policy_target, reduction="none")
            p_loss = (per_sample_p_loss * policy_weight).mean()

            # Weighted value loss
            per_sample_v_loss = (value_pred - value_target) ** 2
            v_loss = (per_sample_v_loss * value_weight).mean()

            loss = p_loss + args.value_loss_weight * v_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = x.size(0)
            total_count += bs
            total_loss += loss.item() * bs
            total_policy_loss += p_loss.item() * bs
            total_value_loss += v_loss.item() * bs

        scheduler.step()
        n = max(1, int(total_count))
        metrics = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": total_loss / n,
            "train_policy_loss": total_policy_loss / n,
            "train_value_loss": total_value_loss / n,
        }

        if val_loader is not None:
            metrics.update(evaluate(model, val_loader, device, args.value_loss_weight))
        else:
            metrics.update({"val_loss": float("nan"), "val_policy_acc": float("nan"),
                            "val_value_mse": float("nan"), "val_legal_policy_acc": float("nan")})

        epoch_metrics.append(metrics)
        print(
            f"epoch={epoch:03d} lr={metrics['lr']:.2e} "
            f"train_loss={metrics['train_loss']:.4f} "
            f"policy_loss={metrics['train_policy_loss']:.4f} "
            f"value_loss={metrics['train_value_loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_legal_acc={metrics.get('val_legal_acc', 0):.4f} "
            f"val_top5={metrics.get('val_legal_top5', 0):.4f} "
            f"val_top10={metrics.get('val_legal_top10', 0):.4f} "
            f"val_policy_acc={metrics['val_policy_acc']:.4f}"
        )

        # Save best model
        val_loss = metrics.get("val_loss", float("inf"))
        if not (val_loss != val_loss) and val_loss < best_val_loss:  # not NaN
            best_val_loss = val_loss
            best_path = Path(args.output).with_suffix(".best.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "architecture": "cnn",
                        "channels": args.channels,
                        "res_blocks": args.res_blocks,
                        "policy_dropout": args.policy_dropout,
                        "value_dropout": args.value_dropout,
                        "planes": PLANES,
                        "board_rows": BOARD_ROWS,
                        "board_cols": BOARD_COLS,
                        "input_dim": INPUT_DIM,
                        "action_dim": ACTION_DIM}, best_path)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "cnn",
        "channels": args.channels,
        "res_blocks": args.res_blocks,
        "policy_dropout": args.policy_dropout,
        "value_dropout": args.value_dropout,
        "planes": PLANES,
        "board_rows": BOARD_ROWS,
        "board_cols": BOARD_COLS,
        "input_dim": INPUT_DIM,
        "action_dim": ACTION_DIM,
        "args": vars(args),
        "epoch_metrics": epoch_metrics,
    }, output)
    print(f"\nSaved final checkpoint: {output}")
    if best_val_loss < float("inf"):
        print(f"Best val_loss: {best_val_loss:.4f} → {output.with_suffix('.best.pt')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Chinese Chess CNN policy-value network")
    p.add_argument("--input", required=True, help="Path to ChineseChess.SelfPlay JSONL")
    p.add_argument("--output", required=True, help="Output checkpoint path (.pt)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--channels", type=int, default=128, help="CNN feature channels")
    p.add_argument("--res-blocks", type=int, default=8, help="Number of residual blocks")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU training")
    p.add_argument("--include-unfinished", action="store_true")
    p.add_argument("--value-loss-weight", type=float, default=1.0)
    p.add_argument("--policy-dropout", type=float, default=0.3, help="Dropout rate for policy FC layer")
    p.add_argument("--value-dropout", type=float, default=0.3, help="Dropout rate for value FC layers")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
