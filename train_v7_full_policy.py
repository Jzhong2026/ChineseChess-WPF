#!/usr/bin/env python3
"""
V7: Full 8100 policy head (AlphaZero-style), fine-tuned from V6.

Key changes from V5/V6 factored head:
  - Policy head: linear(policy_features, 8100) instead of from_logits[90] + to_logits[90]
  - This allows the model to learn full P(from,to) jointly, not P(from)*P(to)
  - to_accuracy should improve dramatically because the model can learn
    that a Horse at (2,1) can only go to 8 specific squares.

Fine-tuning strategy:
  - Load V6 checkpoint (CNN backbone + value head pre-trained)
  - Replace policy head with full 8100 head
  - Freeze CNN backbone and value head (optional: partial freeze)
  - Train only the policy head for 20-40 epochs (fast convergence)
  - Then unfreeze all and fine-tune for 10-20 epochs

Usage:
    python train_v7_full_policy.py \
        --input data/selfplay/train_500.jsonl \
        --checkpoint artifacts/cnn_policy_value_v6.pt \
        --output artifacts/cnn_policy_value_v7.pt \
        --epochs 40 --batch-size 256 \
        --lr 1e-4 --weight-decay 1e-2
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

# ─── Constants (same as V5/V6) ───────────────────────────────────────
BOARD_PLANES = 14
SIDE_PLANES = 2
INPUT_PLANES = BOARD_PLANES + SIDE_PLANES  # 16
BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS  # 90
INPUT_DIM = BOARD_PLANES * BOARD_ROWS * BOARD_COLS  # 1260
ACTION_DIM = BOARD_SIZE * BOARD_SIZE  # 8100
INVALID_LOGIT = -1e9


@dataclass
class SelfPlaySample:
    board_encoding: List[float]
    side_to_move: int
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        s = self.samples[idx]
        flat = torch.tensor(s.board_encoding, dtype=torch.float32)
        board_features = flat.view(BOARD_SIZE, BOARD_PLANES).permute(1, 0).view(BOARD_PLANES, BOARD_ROWS, BOARD_COLS)

        side_planes = torch.zeros(SIDE_PLANES, BOARD_ROWS, BOARD_COLS, dtype=torch.float32)
        if s.side_to_move == 1:
            side_planes[0] = 1.0
        else:
            side_planes[1] = 1.0

        features = torch.cat([board_features, side_planes], dim=0)  # [16, 10, 9]

        # Data augmentation: horizontal flip
        if random.random() < 0.5:
            features = features.flip(-1)
            from_idx = s.selected_move // BOARD_SIZE
            to_idx = s.selected_move % BOARD_SIZE
            f_r, f_c = from_idx // BOARD_COLS, from_idx % BOARD_COLS
            t_r, t_c = to_idx // BOARD_COLS, to_idx % BOARD_COLS
            f_c = BOARD_COLS - 1 - f_c
            t_c = BOARD_COLS - 1 - t_c
            s = SelfPlaySample(
                board_encoding=s.board_encoding,
                side_to_move=s.side_to_move,
                legal_moves=_remap_legal_moves(s.legal_moves),
                selected_move=(f_r * BOARD_COLS + f_c) * BOARD_SIZE + (t_r * BOARD_COLS + t_c),
                result=s.result,
                value_weight=s.value_weight,
                policy_weight=s.policy_weight,
            )

        move_target = torch.tensor(s.selected_move, dtype=torch.long)
        value_target = torch.tensor(s.result * s.side_to_move, dtype=torch.float32)
        legal_mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
        legal_mask[s.legal_moves] = True
        value_weight = torch.tensor(s.value_weight, dtype=torch.float32)
        policy_weight = torch.tensor(s.policy_weight, dtype=torch.float32)
        return features, move_target, value_target, legal_mask, value_weight, policy_weight


def _remap_legal_moves(legal_moves: List[int]) -> List[int]:
    new_indices = []
    for move_idx in legal_moves:
        f_idx = move_idx // BOARD_SIZE
        t_idx = move_idx % BOARD_SIZE
        f_r, f_c = f_idx // BOARD_COLS, f_idx % BOARD_COLS
        t_r, t_c = t_idx // BOARD_COLS, t_idx % BOARD_COLS
        f_c = BOARD_COLS - 1 - f_c
        t_c = BOARD_COLS - 1 - t_c
        new_indices.append((f_r * BOARD_COLS + f_c) * BOARD_SIZE + (t_r * BOARD_COLS + t_c))
    return new_indices


# ─── Model ─────────────────────────────────────────────────────────────────
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


class CNNFullPolicyValueNet(nn.Module):
    """
    V7: Full 8100 policy head (AlphaZero-style).

    Input:  [batch, 16, 10, 9]
    Outputs:
      - policy_logits: [batch, 8100]
      - value_pred:    [batch]  (tanh, current player perspective)
    """

    def __init__(self, in_channels: int = INPUT_PLANES, channels: int = 64,
                 res_blocks: int = 6, policy_dropout: float = 0.2,
                 value_dropout: float = 0.2,
                 policy_channels: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.policy_channels = policy_channels

        # CNN backbone (same as V6)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(res_blocks)])

        # Policy head: conv(policy_channels) → flatten → linear(→8100)
        self.policy_conv = nn.Conv2d(channels, policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_dropout = nn.Dropout(policy_dropout)
        self.policy_fc = nn.Linear(policy_channels * BOARD_SIZE, ACTION_DIM)

        # Value head (same as V5/V6)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_dropout1 = nn.Dropout(value_dropout)
        self.value_fc1 = nn.Linear(BOARD_SIZE, 256)
        self.value_dropout2 = nn.Dropout(value_dropout)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, 16, 10, 9]
        x = self.stem(x)
        x = self.tower(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))  # [batch, policy_channels, 10, 9]
        p = self.policy_dropout(p)
        p = p.view(p.size(0), -1)  # [batch, policy_channels * 90]
        policy_logits = self.policy_fc(p)  # [batch, 8100]

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))  # [batch, 1, 10, 9]
        v = self.value_dropout1(v)
        v = v.view(v.size(0), -1)  # [batch, 90]
        v = F.relu(self.value_fc1(v))
        v = self.value_dropout2(v)
        value_pred = torch.tanh(self.value_fc2(v)).squeeze(1)  # [batch]

        return policy_logits, value_pred


# ─── Data loading (same as V5/V6) ───────────────────────────────────────
def load_samples(path: Path, include_unfinished: bool = False) -> List[SelfPlaySample]:
    samples = []
    line_no = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_no += 1
            obj = json.loads(line)
            board_encoding = obj.get("BoardEncoding")
            legal_moves = obj.get("LegalMoves")
            selected_move = obj.get("SelectedMove")
            result = obj.get("Result")
            side_to_move = obj.get("SideToMove", 1)

            if not isinstance(board_encoding, list) or len(board_encoding) != INPUT_DIM:
                raise ValueError(f"Line {line_no}: BoardEncoding must be float[{INPUT_DIM}]")
            if not isinstance(legal_moves, list) or len(legal_moves) == 0:
                raise ValueError(f"Line {line_no}: LegalMoves must be a non-empty int[]")
            if not isinstance(selected_move, int) or not (0 <= selected_move < ACTION_DIM):
                raise ValueError(f"Line {line_no}: SelectedMove must be in [0, {ACTION_DIM - 1}]")
            if side_to_move not in (1, -1):
                raise ValueError(f"Line {line_no}: SideToMove must be 1 or -1, got {side_to_move}")

            result_value = float(result)
            legal_moves_int = [int(m) for m in legal_moves]

            samples.append(SelfPlaySample(
                board_encoding=[float(v) for v in board_encoding],
                side_to_move=int(side_to_move),
                legal_moves=legal_moves_int,
                selected_move=selected_move,
                result=result_value,
                value_weight=float(obj.get("ValueWeight", 1.0)),
                policy_weight=float(obj.get("PolicyWeight", 1.0)),
            ))
    print(f"Loaded {len(samples)} samples ({line_no - len(samples)} skipped).")
    return samples


# ─── Training loop (adapted for full 8100 policy) ─────────────────────
def train():
    parser = build_parser()
    args = parser.parse_args()

    samples = load_samples(Path(args.input))
    train_size = int(len(samples) * 0.9)
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = ChineseChessDataset(train_samples)
    val_ds = ChineseChessDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = CNNFullPolicyValueNet(
        in_channels=INPUT_PLANES,
        channels=args.channels,
        res_blocks=args.res_blocks,
        policy_dropout=args.policy_dropout,
        value_dropout=args.value_dropout,
        policy_channels=args.policy_channels,
    )

    # Load V6 checkpoint (fine-tune from V6)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        # Only load matching parameters (skip policy head since it changed)
        model_dict = model.state_dict()
        ckpt_dict = ckpt['model_state_dict']
        loaded_keys = []
        skipped_keys = []
        for k, v in ckpt_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)
        model.load_state_dict(model_dict)
        print(f"Loaded V6 checkpoint: {args.checkpoint}")
        print(f"  Loaded {len(loaded_keys)} params, skipped {len(skipped_keys)} (policy head)")
        if skipped_keys:
            print(f"  Skipped keys: {skipped_keys[:5]}...")

    device = torch.device('cpu')
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: CNN Full Policy channels={args.channels}, res_blocks={args.res_blocks}, "
          f"policy_channels={args.policy_channels}, total_params={total:,} ({total/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        for features, move_target, value_target, legal_mask, value_weight, policy_weight in train_loader:
            features = features.to(device)
            move_target = move_target.to(device)
            value_target = value_target.to(device)
            legal_mask = legal_mask.to(device)
            value_weight = value_weight.to(device)
            policy_weight = policy_weight.to(device)

            policy_logits, value_pred = model(features)

            # Policy loss: cross-entropy on full 8100 actions
            per_sample_p_loss = F.cross_entropy(policy_logits, move_target, reduction='none')
            # Mask illegal moves (set to -inf so they don't affect softmax)
            masked_logits = policy_logits.masked_fill(~legal_mask, INVALID_LOGIT)
            # Policy accuracy (Top-1 on masked logits)
            policy_pred = masked_logits.argmax(1)
            correct = (policy_pred == move_target).sum().item()

            p_loss = (per_sample_p_loss * policy_weight).mean()
            v_loss = ((value_pred - value_target) ** 2 * value_weight).mean()
            loss = p_loss + args.value_weight * v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_correct += correct
            train_count += features.size(0)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_count = 0
        val_top5 = 0
        val_top10 = 0
        with torch.no_grad():
            for features, move_target, value_target, legal_mask, value_weight, policy_weight in val_loader:
                features = features.to(device)
                move_target = move_target.to(device)
                legal_mask = legal_mask.to(device)

                policy_logits, value_pred = model(features)
                masked_logits = policy_logits.masked_fill(~legal_mask, INVALID_LOGIT)

                per_sample_p_loss = F.cross_entropy(policy_logits, move_target, reduction='none')
                p_loss = (per_sample_p_loss * policy_weight).mean()
                v_loss = ((value_pred - value_target) ** 2 * value_weight).mean()
                loss = p_loss + args.value_weight * v_loss

                val_loss += loss.item() * features.size(0)
                policy_pred = masked_logits.argmax(1)
                val_correct += (policy_pred == move_target).sum().item()
                val_count += features.size(0)

                # Top-5 and Top-10
                _, top5_idx = masked_logits.topk(5, dim=1)
                _, top10_idx = masked_logits.topk(10, dim=1)
                val_top5 += (top5_idx == move_target.unsqueeze(1)).any(dim=1).sum().item()
                val_top10 += (top10_idx == move_target.unsqueeze(1)).any(dim=1).sum().item()

        train_loss /= train_count
        val_loss /= val_count
        train_acc = train_correct / train_count
        val_acc = val_correct / val_count
        val_top5_acc = val_top5 / val_count
        val_top10_acc = val_top10 / val_count

        scheduler.step()

        print(f"epoch={epoch:03d} lr={optimizer.param_groups[0]['lr']:.2e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_top5={val_top5_acc:.4f} val_top10={val_top10_acc:.4f}")

        # Save best
        if not (val_loss != val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(args.output).with_suffix(".best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "cnn_full_v7",
                "channels": args.channels,
                "res_blocks": args.res_blocks,
                "policy_dropout": args.policy_dropout,
                "value_dropout": args.value_dropout,
                "policy_channels": args.policy_channels,
                "planes": INPUT_PLANES,
                "board_planes": BOARD_PLANES,
                "board_rows": BOARD_ROWS,
                "board_cols": BOARD_COLS,
                "input_dim": INPUT_DIM,
                "action_dim": ACTION_DIM,
            }, best_path)
            print(f"  ✅ Best model saved: {best_path.name}")

        # Periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            periodic_path = Path(args.output).with_suffix(f".epoch{epoch}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "cnn_full_v7",
                "channels": args.channels,
                "res_blocks": args.res_blocks,
                "epoch": epoch,
            }, periodic_path)
            print(f"  💾 Periodic checkpoint saved: {periodic_path.name}")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "cnn_full_v7",
        "channels": args.channels,
        "res_blocks": args.res_blocks,
        "policy_dropout": args.policy_dropout,
        "value_dropout": args.value_dropout,
        "policy_channels": args.policy_channels,
        "planes": INPUT_PLANES,
        "board_planes": BOARD_PLANES,
        "board_rows": BOARD_ROWS,
        "board_cols": BOARD_COLS,
        "input_dim": INPUT_DIM,
        "action_dim": ACTION_DIM,
        "args": vars(args),
    }, args.output)
    print(f"\nSaved final checkpoint: {args.output}")
    if best_val_loss < float("inf"):
        print(f"Best val_loss: {best_val_loss:.4f} → {Path(args.output).with_suffix('.best.pt').name}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train V7 full 8100 policy head (fine-tune from V6)")
    p.add_argument("--input", required=True, help="Input JSONL path")
    p.add_argument("--checkpoint", default=None, help="V6 checkpoint to fine-tune from")
    p.add_argument("--output", required=True, help="Output checkpoint path (.pt)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--res-blocks", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--policy-dropout", type=float, default=0.2)
    p.add_argument("--value-dropout", type=float, default=0.2)
    p.add_argument("--policy-channels", type=int, default=8, help="Channels in policy conv head")
    p.add_argument("--value-weight", type=float, default=1.0)
    return p


if __name__ == "__main__":
    train()
