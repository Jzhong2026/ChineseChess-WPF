#!/usr/bin/env python3
"""
V8: Factored policy head (from/to) with V7 backbone improvements.

Key difference from V7:
  - V7: policy_conv → Linear(policy_channels*90, 8100) — 5.83M params in FC
  - V8: policy_conv → from_fc(policy_channels*90 → 90) + to_fc(policy_channels*90 → 90)
  - This reduces policy head params from 5.83M to ~130K

Fine-tuning from V7:
  - Load V7 checkpoint → transfer CNN backbone + value head weights
  - Policy head is randomly initialized (shape differs)
  - Train full model end-to-end

Usage:
    python train_v8_factored.py \
        --input data/selfplay/train_batch2_60k.jsonl \
        --checkpoint artifacts/cnn_policy_value_v7_cont2.best.pt \
        --output artifacts/cnn_policy_value_v8.pt \
        --channels 64 --res-blocks 6 --policy-channels 8 \
        --epochs 50 --batch-size 256 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ─── Constants ────────────────────────────────────────────────────
BOARD_PLANES = 14
SIDE_PLANES = 2
INPUT_PLANES = BOARD_PLANES + SIDE_PLANES  # 16
BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS  # 90
ACTION_DIM = BOARD_SIZE * BOARD_SIZE  # 8100
INVALID_LOGIT = -1e9


# ─── Data ─────────────────────────────────────────────────────────

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
    def __init__(self, samples, aug_prob=0.5):
        self.samples = list(samples)
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        flat = torch.tensor(s.board_encoding, dtype=torch.float32)
        board_features = flat.view(BOARD_SIZE, BOARD_PLANES).permute(1, 0).view(
            BOARD_PLANES, BOARD_ROWS, BOARD_COLS)

        side_planes = torch.zeros(SIDE_PLANES, BOARD_ROWS, BOARD_COLS, dtype=torch.float32)
        if s.side_to_move == 1:
            side_planes[0] = 1.0
        else:
            side_planes[1] = 1.0
        features = torch.cat([board_features, side_planes], dim=0)  # [16, 10, 9]

        from_idx = s.selected_move // BOARD_SIZE
        to_idx = s.selected_move % BOARD_SIZE
        from_target = torch.tensor(from_idx, dtype=torch.long)
        to_target = torch.tensor(to_idx, dtype=torch.long)

        value_target = torch.tensor(s.result * s.side_to_move, dtype=torch.float32)
        legal_mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
        legal_mask[s.legal_moves] = True
        value_weight = torch.tensor(s.value_weight, dtype=torch.float32)
        policy_weight = torch.tensor(s.policy_weight, dtype=torch.float32)

        # Data augmentation: horizontal flip
        if random.random() < self.aug_prob:
            features = features.flip(-1)
            f_r, f_c = from_idx // BOARD_COLS, from_idx % BOARD_COLS
            t_r, t_c = to_idx // BOARD_COLS, to_idx % BOARD_COLS
            f_c = BOARD_COLS - 1 - f_c
            t_c = BOARD_COLS - 1 - t_c
            from_target = torch.tensor(f_r * BOARD_COLS + f_c, dtype=torch.long)
            to_target = torch.tensor(t_r * BOARD_COLS + t_c, dtype=torch.long)
            # Remap legal_mask
            new_legal = []
            for move_idx in s.legal_moves:
                mf = move_idx // BOARD_SIZE
                mt = move_idx % BOARD_SIZE
                mfr, mfc = mf // BOARD_COLS, mf % BOARD_COLS
                mtr, mtc = mt // BOARD_COLS, mt % BOARD_COLS
                mfc = BOARD_COLS - 1 - mfc
                mtc = BOARD_COLS - 1 - mtc
                new_legal.append((mfr * BOARD_COLS + mfc) * BOARD_SIZE + (mtr * BOARD_COLS + mtc))
            legal_mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
            legal_mask[new_legal] = True

        return features, from_target, to_target, value_target, legal_mask, value_weight, policy_weight


def load_samples(path: Path) -> List[SelfPlaySample]:
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            board_encoding = obj.get("BoardEncoding")
            legal_moves = obj.get("LegalMoves")
            selected_move = obj.get("SelectedMove")
            result = obj.get("Result")
            side_to_move = obj.get("SideToMove", 1)

            if not isinstance(board_encoding, list) or len(board_encoding) != BOARD_SIZE * BOARD_PLANES:
                continue
            if not isinstance(legal_moves, list) or len(legal_moves) == 0:
                continue
            if not isinstance(selected_move, int) or not (0 <= selected_move < ACTION_DIM):
                continue

            samples.append(SelfPlaySample(
                board_encoding=[float(v) for v in board_encoding],
                side_to_move=int(side_to_move),
                legal_moves=[int(m) for m in legal_moves],
                selected_move=selected_move,
                result=float(result),
                value_weight=float(obj.get("ValueWeight", 1.0)),
                policy_weight=float(obj.get("PolicyWeight", 1.0)),
            ))
    print(f"Loaded {len(samples)} samples.")
    return samples


# ─── Model ─────────────────────────────────────────────────────────

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


class CNNV8FactoredNet(nn.Module):
    """
    V8: V7 backbone + factored from/to policy heads.
    
    Input:  [batch, 16, 10, 9]
    Outputs:
      - from_logits: [batch, 90]
      - to_logits:   [batch, 90]
      - value_pred:  [batch]  (tanh, current player perspective)
    
    Combined policy: logit_combined[from*90+to] = from_logits[from] + to_logits[to]
    """

    def __init__(self, in_channels: int = INPUT_PLANES, channels: int = 64,
                 res_blocks: int = 6, policy_channels: int = 8,
                 policy_dropout: float = 0.2, value_dropout: float = 0.2) -> None:
        super().__init__()
        self.channels = channels
        self.policy_channels = policy_channels

        # CNN backbone (same as V7)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(res_blocks)])

        # Policy head — factored
        self.policy_conv = nn.Conv2d(channels, policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_dropout = nn.Dropout(policy_dropout)
        policy_flat = policy_channels * BOARD_SIZE
        self.from_fc = nn.Linear(policy_flat, BOARD_SIZE)
        self.to_fc = nn.Linear(policy_flat, BOARD_SIZE)

        # Value head (same as V7)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_dropout1 = nn.Dropout(value_dropout)
        self.value_fc1 = nn.Linear(BOARD_SIZE, 256)
        self.value_dropout2 = nn.Dropout(value_dropout)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.tower(x)

        # Policy (factored)
        p = F.relu(self.policy_bn(self.policy_conv(x)))  # [B, pc, 10, 9]
        p = p.flatten(1)                                   # [B, pc*90]
        p = self.policy_dropout(p)
        from_logits = self.from_fc(p)                      # [B, 90]
        to_logits = self.to_fc(p)                          # [B, 90]

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))     # [B, 1, 10, 9]
        v = v.flatten(1)                                   # [B, 90]
        v = self.value_dropout1(v)
        v = F.relu(self.value_fc1(v))                     # [B, 256]
        v = self.value_dropout2(v)
        value_pred = torch.tanh(self.value_fc2(v)).squeeze(1)  # [B]

        return from_logits, to_logits, value_pred


# ─── Training ──────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Train V8 factored policy CNN")
    p.add_argument("--input", required=True, help="Training JSONL file")
    p.add_argument("--output", required=True, help="Output checkpoint path")
    p.add_argument("--checkpoint", default=None, help="V7 checkpoint to init backbone from")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--res-blocks", type=int, default=6)
    p.add_argument("--policy-channels", type=int, default=8)
    p.add_argument("--policy-dropout", type=float, default=0.2)
    p.add_argument("--value-dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--value-loss-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return p


@torch.no_grad()
def evaluate(model, loader, device, value_loss_weight):
    model.eval()
    total_loss = total_count = top1 = top5 = top10 = 0.0
    from_acc = to_acc = value_mse = 0.0

    for features, from_t, to_t, value_t, legal_mask, vw, pw in loader:
        features = features.to(device)
        from_t = from_t.to(device)
        to_t = to_t.to(device)
        value_t = value_t.to(device)
        legal_mask = legal_mask.to(device)
        vw = vw.to(device)
        pw = pw.to(device)

        from_logits, to_logits, value_pred = model(features)

        # Factored policy loss
        p_from = (F.cross_entropy(from_logits, from_t, reduction='none') * pw).mean()
        p_to = (F.cross_entropy(to_logits, to_t, reduction='none') * pw).mean()
        v_loss = (((value_pred - value_t) ** 2) * vw).mean()
        loss = p_from + p_to + value_loss_weight * v_loss

        bs = features.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        value_mse += v_loss.item() * bs

        # Combined logits for Top-K
        combined = from_logits.unsqueeze(2) + to_logits.unsqueeze(1)  # [B, 90, 90]
        combined_flat = combined.view(bs, ACTION_DIM)                  # [B, 8100]
        masked = combined_flat.masked_fill(~legal_mask, INVALID_LOGIT)
        target = from_t * BOARD_SIZE + to_t

        top1 += (masked.argmax(1) == target).sum().item()
        _, t5 = masked.topk(5, dim=1)
        _, t10 = masked.topk(10, dim=1)
        top5 += (t5 == target.unsqueeze(1)).any(dim=1).sum().item()
        top10 += (t10 == target.unsqueeze(1)).any(dim=1).sum().item()
        from_acc += (from_logits.argmax(1) == from_t).sum().item()
        to_acc += (to_logits.argmax(1) == to_t).sum().item()

    n = max(1, int(total_count))
    return {
        "loss": total_loss / n, "top1": top1 / n, "top5": top5 / n,
        "top10": top10 / n, "from_acc": from_acc / n, "to_acc": to_acc / n,
        "value_mse": value_mse / n,
    }


def train():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    samples = load_samples(Path(args.input))
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train_samples, val_samples = samples[:split], samples[split:]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_loader = DataLoader(
        ChineseChessDataset(train_samples), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        ChineseChessDataset(val_samples, aug_prob=0.0), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    model = CNNV8FactoredNet(
        in_channels=INPUT_PLANES, channels=args.channels,
        res_blocks=args.res_blocks, policy_channels=args.policy_channels,
        policy_dropout=args.policy_dropout, value_dropout=args.value_dropout)

    # Load V7 checkpoint (transfer backbone + value head)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model_dict = model.state_dict()
        ckpt_dict = ckpt['model_state_dict']
        loaded = []
        skipped = []
        for k, v in ckpt_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(loaded)} params from V7 checkpoint, skipped {len(skipped)}")
        if skipped:
            print(f"  Skipped (policy head): {skipped[:5]}...")

    device = torch.device('cpu')
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model V8 Factored: channels={args.channels}, res_blocks={args.res_blocks}, "
          f"policy_channels={args.policy_channels}, params={total:,} ({total/1e6:.2f}M)")
    print(f"  Policy head: {args.policy_channels * BOARD_SIZE}*90*2 = "
          f"{args.policy_channels * BOARD_SIZE * 90 * 2:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = train_top1 = train_count = 0.0
        for features, from_t, to_t, value_t, legal_mask, vw, pw in train_loader:
            features = features.to(device)
            from_t = from_t.to(device)
            to_t = to_t.to(device)
            value_t = value_t.to(device)
            legal_mask = legal_mask.to(device)
            vw = vw.to(device)
            pw = pw.to(device)

            from_logits, to_logits, value_pred = model(features)

            p_from = (F.cross_entropy(from_logits, from_t, reduction='none') * pw).mean()
            p_to = (F.cross_entropy(to_logits, to_t, reduction='none') * pw).mean()
            v_loss = (((value_pred - value_t) ** 2) * vw).mean()
            loss = p_from + p_to + args.value_loss_weight * v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = features.size(0)
            train_loss += loss.item() * bs
            train_count += bs

            # Track combined top-1
            combined = from_logits.unsqueeze(2) + to_logits.unsqueeze(1)
            combined_flat = combined.view(bs, ACTION_DIM)
            masked = combined_flat.masked_fill(~legal_mask, INVALID_LOGIT)
            target = from_t * BOARD_SIZE + to_t
            train_top1 += (masked.argmax(1) == target).sum().item()

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # Validation
        val = evaluate(model, val_loader, device, args.value_loss_weight)

        print(f"epoch={epoch:03d} lr={lr_now:.2e} "
              f"train_loss={train_loss/train_count:.4f} train_acc={train_top1/train_count:.4f} "
              f"val_loss={val['loss']:.4f} val_top1={val['top1']:.4f} "
              f"val_top5={val['top5']:.4f} val_top10={val['top10']:.4f} "
              f"from_acc={val['from_acc']:.4f} to_acc={val['to_acc']:.4f}")

        # Save best
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_path = Path(args.output).with_suffix(".best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "cnn_factored_v5",  # compatible with export_cnn_onnx.py
                "channels": args.channels,
                "res_blocks": args.res_blocks,
                "policy_dropout": args.policy_dropout,
                "value_dropout": args.value_dropout,
                "policy_channels": args.policy_channels,
                "planes": INPUT_PLANES,
                "epoch": epoch,
                "val_loss": val["loss"],
                "val_top10": val["top10"],
            }, best_path)
            print(f"  ✅ Best saved: {best_path.name}")

        # Periodic checkpoint
        if epoch % 5 == 0:
            path = Path(args.output).with_suffix(f".epoch{epoch}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "cnn_factored_v5",
                "channels": args.channels,
                "res_blocks": args.res_blocks,
                "policy_channels": args.policy_channels,
                "planes": INPUT_PLANES,
            }, path)
            print(f"  💾 Checkpoint: {path.name}")

    # Final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "cnn_factored_v5",
        "channels": args.channels,
        "res_blocks": args.res_blocks,
        "policy_dropout": args.policy_dropout,
        "value_dropout": args.value_dropout,
        "policy_channels": args.policy_channels,
        "planes": INPUT_PLANES,
        "epoch": args.epochs,
        "val_loss": val["loss"],
        "val_top10": val["top10"],
    }, args.output)
    print(f"\nFinal model: {args.output}")


if __name__ == "__main__":
    train()
