#!/usr/bin/env python3
"""Train a Chinese Chess policy-value network from self-play JSONL data."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

INPUT_DIM = 1260
ACTION_DIM = 8100
INVALID_LOGIT = -1e9


@dataclass
class SelfPlaySample:
    board_encoding: List[float]
    legal_moves: List[int]
    selected_move: int
    result: float


class ChineseChessDataset(Dataset):
    def __init__(self, samples: Sequence[SelfPlaySample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        features = torch.tensor(sample.board_encoding, dtype=torch.float32)
        policy_target = torch.tensor(sample.selected_move, dtype=torch.long)
        value_target = torch.tensor(sample.result, dtype=torch.float32)
        legal_mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
        legal_mask[sample.legal_moves] = True
        return features, policy_target, value_target, legal_mask


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, action_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(dim, action_dim)
        self.value_head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        policy_logits = self.policy_head(h)
        value_pred = torch.tanh(self.value_head(h)).squeeze(1)
        return policy_logits, value_pred


def load_samples(path: Path, include_unfinished: bool) -> List[SelfPlaySample]:
    samples: List[SelfPlaySample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            unfinished = bool(obj.get("Unfinished", False))
            if unfinished and not include_unfinished:
                continue

            board_encoding = obj.get("BoardEncoding")
            legal_moves = obj.get("LegalMoves")
            selected_move = obj.get("SelectedMove")
            result = obj.get("Result")

            if not isinstance(board_encoding, list) or len(board_encoding) != INPUT_DIM:
                raise ValueError(f"Invalid line {line_no}: BoardEncoding must be float[{INPUT_DIM}]")
            if not isinstance(legal_moves, list) or len(legal_moves) == 0:
                raise ValueError(f"Invalid line {line_no}: LegalMoves must be a non-empty int[]")
            if not isinstance(selected_move, int) or not (0 <= selected_move < ACTION_DIM):
                raise ValueError(f"Invalid line {line_no}: SelectedMove must be in [0, {ACTION_DIM - 1}]")

            legal_moves_int = [int(m) for m in legal_moves]
            if any(m < 0 or m >= ACTION_DIM for m in legal_moves_int):
                raise ValueError(f"Invalid line {line_no}: LegalMoves values must be in [0, {ACTION_DIM - 1}]")
            if selected_move not in legal_moves_int:
                raise ValueError(f"Invalid line {line_no}: SelectedMove must be included in LegalMoves")

            result_value = float(result)
            if result_value not in (-1.0, 0.0, 1.0):
                raise ValueError("Invalid line {}: Result must be one of -1/0/1".format(line_no))

            samples.append(
                SelfPlaySample(
                    board_encoding=[float(v) for v in board_encoding],
                    legal_moves=legal_moves_int,
                    selected_move=selected_move,
                    result=result_value,
                )
            )

    if not samples:
        raise ValueError("No valid samples loaded from input JSONL")
    return samples


def split_dataset(samples: List[SelfPlaySample], val_ratio: float, seed: int) -> Tuple[List[SelfPlaySample], List[SelfPlaySample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]
    if not train_samples:
        train_samples, val_samples = val_samples, []
    return train_samples, val_samples


def masked_policy_logits(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return policy_logits.masked_fill(~legal_mask, INVALID_LOGIT)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, value_loss_weight: float) -> dict:
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    model.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0
    legal_correct = 0
    value_mse_sum = 0.0

    with torch.no_grad():
        for x, policy_target, value_target, legal_mask in loader:
            x = x.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            legal_mask = legal_mask.to(device)

            policy_logits, value_pred = model(x)
            masked_logits = masked_policy_logits(policy_logits, legal_mask)

            p_loss = ce_loss(masked_logits, policy_target)
            v_loss = mse_loss(value_pred, value_target)
            loss = p_loss + value_loss_weight * v_loss

            preds_unmasked = policy_logits.argmax(dim=1)
            preds_masked = masked_logits.argmax(dim=1)
            batch_size = x.size(0)

            total_loss += loss.item() * batch_size
            total_count += batch_size
            correct += (preds_unmasked == policy_target).sum().item()
            legal_correct += (preds_masked == policy_target).sum().item()
            value_mse_sum += v_loss.item() * batch_size

    return {
        "val_loss": total_loss / max(1, total_count),
        "val_policy_acc": correct / max(1, total_count),
        "val_value_mse": value_mse_sum / max(1, total_count),
        "val_legal_policy_acc": legal_correct / max(1, total_count),
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    all_samples = load_samples(Path(args.input), include_unfinished=args.include_unfinished)
    train_samples, val_samples = split_dataset(all_samples, args.val_ratio, args.seed)

    train_loader = DataLoader(ChineseChessDataset(train_samples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ChineseChessDataset(val_samples), batch_size=args.batch_size, shuffle=False) if val_samples else None

    model = PolicyValueNet(INPUT_DIM, args.hidden_dim, args.num_layers, ACTION_DIM)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    epoch_metrics: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_count = 0

        for x, policy_target, value_target, legal_mask in train_loader:
            x = x.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            legal_mask = legal_mask.to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = model(x)
            masked_logits = masked_policy_logits(policy_logits, legal_mask)

            p_loss = ce_loss(masked_logits, policy_target)
            v_loss = mse_loss(value_pred, value_target)
            loss = p_loss + args.value_loss_weight * v_loss

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_count += batch_size
            total_loss += loss.item() * batch_size
            total_policy_loss += p_loss.item() * batch_size
            total_value_loss += v_loss.item() * batch_size

        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_count),
            "train_policy_loss": total_policy_loss / max(1, total_count),
            "train_value_loss": total_value_loss / max(1, total_count),
        }

        if val_loader is not None:
            metrics.update(evaluate(model, val_loader, device, args.value_loss_weight))
        else:
            metrics.update({
                "val_loss": float("nan"),
                "val_policy_acc": float("nan"),
                "val_value_mse": float("nan"),
                "val_legal_policy_acc": float("nan"),
            })

        epoch_metrics.append(metrics)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={metrics['train_loss']:.6f} "
            f"train_policy_loss={metrics['train_policy_loss']:.6f} "
            f"train_value_loss={metrics['train_value_loss']:.6f} "
            f"val_loss={metrics['val_loss']:.6f} "
            f"val_policy_acc={metrics['val_policy_acc']:.4f} "
            f"val_value_mse={metrics['val_value_mse']:.6f} "
            f"val_legal_policy_acc={metrics['val_legal_policy_acc']:.4f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "value_loss_weight": args.value_loss_weight,
            "args": vars(args),
            "epoch_metrics": epoch_metrics,
        },
        output,
    )
    print(f"Saved checkpoint to: {output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Chinese Chess policy-value network from self-play JSONL")
    p.add_argument("--input", required=True, help="Path to ChineseChess.SelfPlay JSONL")
    p.add_argument("--output", required=True, help="Output checkpoint path")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU training")
    p.add_argument("--include-unfinished", action="store_true", help="Include unfinished samples")
    p.add_argument("--value-loss-weight", type=float, default=1.0)
    return p


if __name__ == "__main__":
    parser = build_parser()
    train(parser.parse_args())
