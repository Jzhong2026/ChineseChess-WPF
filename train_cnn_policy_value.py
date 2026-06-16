#!/usr/bin/env python3
"""
Train a Chinese Chess CNN policy-value network from self-play JSONL data.

V5 Architecture (key fix: side-to-move planes + current-player value):
  - Input: [batch, 16, 10, 9] = 14 piece planes + 2 side-to-move planes
  - CNN backbone with residual blocks captures spatial board patterns
  - Factored policy head: from_logits[90] + to_logits[90]
  - Value head: tanh output from current player's perspective (+1 = I'm winning)
  - Side-to-move planes added on-the-fly from JSONL SideToMove field

Previous versions (V1-V4) were "side-blind" — the model couldn't tell whose
turn it was, causing policy entropy ≈ random baseline and near-uniform value
predictions. V5 fixes this by adding 2 binary indicator planes.

Usage:
    python train_cnn_policy_value.py \
        --input data/selfplay/train.jsonl \
        --output artifacts/cnn_policy_value.pt \
        --epochs 80 \
        --batch-size 256 \
        --channels 32 \
        --res-blocks 3
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
BOARD_PLANES = 14       # 2 sides × 7 piece types (stored in JSONL)
SIDE_PLANES = 2         # Red-to-move + Black-to-move indicator planes
INPUT_PLANES = BOARD_PLANES + SIDE_PLANES  # 16 total input planes
BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS  # 90
INPUT_DIM = BOARD_PLANES * BOARD_ROWS * BOARD_COLS   # 1260 (JSONL encoding size)
ACTION_DIM = BOARD_SIZE * BOARD_SIZE  # 8100
INVALID_LOGIT = -1e9


# ─── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class SelfPlaySample:
    board_encoding: List[float]   # flat float[1260] (14 planes, square-major)
    side_to_move: int             # 1=Red, -1=Black
    legal_moves: List[int]
    selected_move: int
    result: float                 # from Red's perspective: 1=Red win, -1=Black win, 0=draw
    value_weight: float
    policy_weight: float


class ChineseChessDataset(Dataset):
    def __init__(self, samples: Sequence[SelfPlaySample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        # C# BoardEncoder outputs square-major: flat[(row*9+col)*14 + plane]
        # CNN expects plane-major:  [plane, row, col]
        flat = torch.tensor(s.board_encoding, dtype=torch.float32)
        # flat: [1260] → [90, 14] → permute → [14, 90] → [14, 10, 9]
        board_features = flat.view(BOARD_SIZE, BOARD_PLANES).permute(1, 0).view(BOARD_PLANES, BOARD_ROWS, BOARD_COLS)

        # Add side-to-move indicator planes (V5 fix: critical for policy quality)
        # Plane 14: all 1s if Red to move, all 0s otherwise
        # Plane 15: all 1s if Black to move, all 0s otherwise
        side_planes = torch.zeros(SIDE_PLANES, BOARD_ROWS, BOARD_COLS, dtype=torch.float32)
        if s.side_to_move == 1:    # Red
            side_planes[0] = 1.0
        else:                       # Black
            side_planes[1] = 1.0

        features = torch.cat([board_features, side_planes], dim=0)  # [16, 10, 9]

        # Factored targets: decompose selected_move = from_idx * 90 + to_idx
        from_target = torch.tensor(s.selected_move // BOARD_SIZE, dtype=torch.long)
        to_target = torch.tensor(s.selected_move % BOARD_SIZE, dtype=torch.long)

        # V5 fix: Value target from CURRENT PLAYER's perspective
        # +1 = current player wins, -1 = current player loses, 0 = draw
        value_target = torch.tensor(s.result * s.side_to_move, dtype=torch.float32)

        legal_mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
        legal_mask[s.legal_moves] = True
        value_weight = torch.tensor(s.value_weight, dtype=torch.float32)
        policy_weight = torch.tensor(s.policy_weight, dtype=torch.float32)

        # ── Data augmentation: horizontal flip (50% chance) ────────────
        # Chinese chess board is left-right symmetric, so flipping is safe.
        # Flip: col → 8-col. This effectively doubles the training data.
        if random.random() < 0.5:
            # Flip features along column dimension: [16, 10, 9] → [16, 10, 9]
            features = features.flip(-1)

            # Remap from_target and to_target (row unchanged, col → 8-col)
            from_row, from_col = from_target.item() // BOARD_COLS, from_target.item() % BOARD_COLS
            to_row, to_col = to_target.item() // BOARD_COLS, to_target.item() % BOARD_COLS
            from_col = BOARD_COLS - 1 - from_col
            to_col = BOARD_COLS - 1 - to_col
            from_target = torch.tensor(from_row * BOARD_COLS + from_col, dtype=torch.long)
            to_target = torch.tensor(to_row * BOARD_COLS + to_col, dtype=torch.long)

            # Remap legal_mask: for each legal move idx, remap from/to cols
            new_legal_indices = []
            for move_idx in s.legal_moves:
                f_idx = move_idx // BOARD_SIZE
                t_idx = move_idx % BOARD_SIZE
                f_r, f_c = f_idx // BOARD_COLS, f_idx % BOARD_COLS
                t_r, t_c = t_idx // BOARD_COLS, t_idx % BOARD_COLS
                f_c = BOARD_COLS - 1 - f_c
                t_c = BOARD_COLS - 1 - t_c
                new_legal_indices.append((f_r * BOARD_COLS + f_c) * BOARD_SIZE + (t_r * BOARD_COLS + t_c))
            legal_mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
            legal_mask[new_legal_indices] = True

        return features, from_target, to_target, value_target, legal_mask, value_weight, policy_weight


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


class CNNFactoredPolicyValueNet(nn.Module):
    """
    AlphaZero-style CNN for Chinese Chess with FACTORED policy head (V5).

    Input: [batch, 16, 10, 9] = 14 piece planes + 2 side-to-move planes
      - Planes 0-6:   Red pieces (General, Advisor, Elephant, Horse, Rook, Cannon, Soldier)
      - Planes 7-13:  Black pieces
      - Plane 14:     Red-to-move indicator (all 1s if Red's turn, all 0s)
      - Plane 15:     Black-to-move indicator (all 1s if Black's turn, all 0s)

    Outputs:
      - from_logits: [batch, 90]  which square the piece moves FROM
      - to_logits:   [batch, 90]  which square the piece moves TO
      - value_pred:  [batch]      tanh ∈ (-1, 1), from CURRENT PLAYER's perspective

    Combined policy:
      logit_combined[from*90+to] = from_logits[from] + to_logits[to]
    """

    def __init__(self, in_channels: int = INPUT_PLANES, channels: int = 128,
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

        BOARD_SIZE = BOARD_ROWS * BOARD_COLS  # 90

        # Policy head — factored into from + to
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_dropout = nn.Dropout(policy_dropout)
        self.from_fc = nn.Linear(2 * BOARD_SIZE, BOARD_SIZE)   # 180 → 90
        self.to_fc = nn.Linear(2 * BOARD_SIZE, BOARD_SIZE)     # 180 → 90

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_dropout1 = nn.Dropout(value_dropout)
        self.value_fc1 = nn.Linear(BOARD_SIZE, 256)
        self.value_dropout2 = nn.Dropout(value_dropout)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [batch, 16, 10, 9]
        x = self.stem(x)
        x = self.tower(x)

        # Policy (factored)
        p = F.relu(self.policy_bn(self.policy_conv(x)))  # [batch, 2, 10, 9]
        p = p.flatten(1)                                   # [batch, 180]
        p = self.policy_dropout(p)
        from_logits = self.from_fc(p)                      # [batch, 90]
        to_logits = self.to_fc(p)                          # [batch, 90]

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))     # [batch, 1, 10, 9]
        v = v.flatten(1)                                   # [batch, 90]
        v = self.value_dropout1(v)
        v = F.relu(self.value_fc1(v))                     # [batch, 256]
        v = self.value_dropout2(v)
        value_pred = torch.tanh(self.value_fc2(v)).squeeze(1)  # [batch]

        return from_logits, to_logits, value_pred


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
            side_to_move = obj.get("SideToMove", 1)

            if not isinstance(board_encoding, list) or len(board_encoding) != INPUT_DIM:
                raise ValueError(f"Line {line_no}: BoardEncoding must be float[{INPUT_DIM}]")
            if not isinstance(legal_moves, list) or len(legal_moves) == 0:
                raise ValueError(f"Line {line_no}: LegalMoves must be a non-empty int[]")
            if not isinstance(selected_move, int) or not (0 <= selected_move < ACTION_DIM):
                raise ValueError(f"Line {line_no}: SelectedMove must be in [0, {ACTION_DIM - 1}]")
            if side_to_move not in (1, -1):
                raise ValueError(f"Line {line_no}: SideToMove must be 1 or -1, got {side_to_move}")

            legal_moves_int = [int(m) for m in legal_moves]
            if selected_move not in legal_moves_int:
                raise ValueError(f"Line {line_no}: SelectedMove not in LegalMoves")

            result_value = float(result)
            if result_value not in (-1.0, 0.0, 1.0):
                raise ValueError(f"Line {line_no}: Result must be -1/0/1, got {result_value}")

            samples.append(SelfPlaySample(
                board_encoding=[float(v) for v in board_encoding],
                side_to_move=int(side_to_move),
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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, value_loss_weight: float) -> dict:
    model.eval()
    total_loss = total_count = correct = legal_correct = 0.0
    legal_top5_correct = legal_top10_correct = value_mse_sum = 0.0
    from_correct = to_correct = 0.0

    for x, from_target, to_target, value_target, legal_mask, value_weight, policy_weight in loader:
        x, from_target, to_target, value_target, legal_mask = (
            x.to(device), from_target.to(device), to_target.to(device),
            value_target.to(device), legal_mask.to(device),
        )
        value_weight = value_weight.to(device)
        policy_weight = policy_weight.to(device)

        from_logits, to_logits, value_pred = model(x)

        # Factored policy losses with weighting
        per_sample_from_loss = F.cross_entropy(from_logits, from_target, reduction="none")
        per_sample_to_loss = F.cross_entropy(to_logits, to_target, reduction="none")
        from_loss = (per_sample_from_loss * policy_weight).mean()
        to_loss = (per_sample_to_loss * policy_weight).mean()

        per_sample_v_loss = (value_pred - value_target) ** 2
        v_loss = (per_sample_v_loss * value_weight).mean()
        loss = from_loss + to_loss + value_loss_weight * v_loss

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        value_mse_sum += v_loss.item() * bs

        # Combined logits for Top-K evaluation
        # combined[b, from*90+to] = from_logits[b, from] + to_logits[b, to]
        combined = from_logits.unsqueeze(2) + to_logits.unsqueeze(1)  # [batch, 90, 90]
        combined_flat = combined.view(bs, ACTION_DIM)                 # [batch, 8100]
        masked_combined = combined_flat.masked_fill(~legal_mask, INVALID_LOGIT)

        policy_target = from_target * (BOARD_ROWS * BOARD_COLS) + to_target

        # Top-1 accuracy
        correct += (combined_flat.argmax(1) == policy_target).sum().item()
        legal_correct += (masked_combined.argmax(1) == policy_target).sum().item()

        # Top-5 and Top-10 accuracy on masked combined logits
        _, top5_idx = masked_combined.topk(5, dim=1)
        _, top10_idx = masked_combined.topk(10, dim=1)
        legal_top5_correct += (top5_idx == policy_target.unsqueeze(1)).any(dim=1).sum().item()
        legal_top10_correct += (top10_idx == policy_target.unsqueeze(1)).any(dim=1).sum().item()

        # Individual head accuracy
        from_correct += (from_logits.argmax(1) == from_target).sum().item()
        to_correct += (to_logits.argmax(1) == to_target).sum().item()

    n = max(1, int(total_count))
    return {
        "val_loss": total_loss / n,
        "val_policy_acc": correct / n,
        "val_value_mse": value_mse_sum / n,
        "val_legal_acc": legal_correct / n,
        "val_legal_top5": legal_top5_correct / n,
        "val_legal_top10": legal_top10_correct / n,
        "val_from_acc": from_correct / n,
        "val_to_acc": to_correct / n,
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

    model = CNNFactoredPolicyValueNet(
        in_channels=INPUT_PLANES,
        channels=args.channels,
        res_blocks=args.res_blocks,
        policy_dropout=args.policy_dropout,
        value_dropout=args.value_dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    # Estimate policy head params separately
    policy_params = sum(p.numel() for p in model.from_fc.parameters()) + \
                    sum(p.numel() for p in model.to_fc.parameters())
    print(f"Model: CNN Factored channels={args.channels}, res_blocks={args.res_blocks}, "
          f"total_params={total_params:,}, policy_head_params={policy_params:,}")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    epoch_metrics: List[dict] = []

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Save periodic checkpoint every 5 epochs (V5 enhancement: allows early deployment)
        if epoch % 5 == 0:
            periodic_path = Path(args.output).with_suffix(f".epoch{epoch}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "cnn_factored_v5",
                "channels": args.channels,
                "res_blocks": args.res_blocks,
                "policy_dropout": args.policy_dropout,
                "value_dropout": args.value_dropout,
                "planes": INPUT_PLANES,
                "board_planes": BOARD_PLANES,
                "board_rows": BOARD_ROWS,
                "board_cols": BOARD_COLS,
                "input_dim": INPUT_DIM,
                "action_dim": ACTION_DIM,
                "epoch": epoch,
            }, periodic_path)
            print(f"  💾 Periodic checkpoint saved: {periodic_path.name}")
        model.train()
        total_loss = total_policy_loss = total_value_loss = total_count = 0.0

        for x, from_target, to_target, value_target, legal_mask, value_weight, policy_weight in train_loader:
            x = x.to(device)
            from_target = from_target.to(device)
            to_target = to_target.to(device)
            value_target = value_target.to(device)
            legal_mask = legal_mask.to(device)
            value_weight = value_weight.to(device)
            policy_weight = policy_weight.to(device)

            optimizer.zero_grad()
            from_logits, to_logits, value_pred = model(x)

            # Factored policy loss: from + to
            per_sample_from_loss = F.cross_entropy(from_logits, from_target, reduction="none")
            per_sample_to_loss = F.cross_entropy(to_logits, to_target, reduction="none")
            from_loss = (per_sample_from_loss * policy_weight).mean()
            to_loss = (per_sample_to_loss * policy_weight).mean()
            p_loss = from_loss + to_loss

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
            f"val_from_acc={metrics.get('val_from_acc', 0):.4f} "
            f"val_to_acc={metrics.get('val_to_acc', 0):.4f} "
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
                        "architecture": "cnn_factored_v5",
                        "channels": args.channels,
                        "res_blocks": args.res_blocks,
                        "policy_dropout": args.policy_dropout,
                        "value_dropout": args.value_dropout,
                        "planes": INPUT_PLANES,
                        "board_planes": BOARD_PLANES,
                        "board_rows": BOARD_ROWS,
                        "board_cols": BOARD_COLS,
                        "input_dim": INPUT_DIM,
                        "action_dim": ACTION_DIM}, best_path)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "cnn_factored_v5",
        "channels": args.channels,
        "res_blocks": args.res_blocks,
        "policy_dropout": args.policy_dropout,
        "value_dropout": args.value_dropout,
        "planes": INPUT_PLANES,
        "board_planes": BOARD_PLANES,
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
