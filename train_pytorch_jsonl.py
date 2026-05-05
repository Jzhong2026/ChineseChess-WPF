#!/usr/bin/env python3
"""Train a simple PyTorch MLP model from JSONL data.

Expected JSONL format (one sample per line):
{"features": [0.1, 0.2, ...], "label": 3}

- features: list[float]
- label: int (for classification) or float (for regression)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Sample:
    features: List[float]
    label: float


class JsonlDataset(Dataset):
    def __init__(self, samples: List[Sample], task: str) -> None:
        self.samples = samples
        self.task = task

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        x = torch.tensor(sample.features, dtype=torch.float32)
        if self.task == "classification":
            y = torch.tensor(int(sample.label), dtype=torch.long)
        else:
            y = torch.tensor(float(sample.label), dtype=torch.float32)
        return x, y


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_jsonl(path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "features" not in obj or "label" not in obj:
                raise ValueError(f"Invalid line {line_no}: missing 'features' or 'label'")
            features = obj["features"]
            label = obj["label"]
            if not isinstance(features, list) or len(features) == 0:
                raise ValueError(f"Invalid line {line_no}: 'features' must be a non-empty list")
            samples.append(Sample(features=[float(v) for v in features], label=float(label)))

    if not samples:
        raise ValueError("No samples loaded from JSONL")

    feature_dim = len(samples[0].features)
    for i, s in enumerate(samples, 1):
        if len(s.features) != feature_dim:
            raise ValueError(f"Inconsistent feature dimensions at sample {i}")
    return samples


def split_dataset(samples: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    random.Random(seed).shuffle(samples)
    val_size = max(1, int(len(samples) * val_ratio)) if len(samples) > 1 else 0
    val = samples[:val_size]
    train = samples[val_size:]
    if not train:
        train, val = val, []
    return train, val


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float | None]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if task == "classification":
                loss = criterion(logits, y)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
            else:
                loss = criterion(logits.squeeze(1), y)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

    avg_loss = total_loss / max(1, total)
    if task == "classification":
        acc = correct / max(1, total)
        return avg_loss, acc
    return avg_loss, None


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    all_samples = load_jsonl(Path(args.input))
    train_samples, val_samples = split_dataset(all_samples, args.val_ratio, args.seed)

    if args.task == "classification":
        labels = [int(s.label) for s in all_samples]
        num_classes = max(labels) + 1
        if min(labels) < 0:
            raise ValueError("Classification labels must be non-negative integers")
        output_dim = num_classes
        criterion: nn.Module = nn.CrossEntropyLoss()
    else:
        output_dim = 1
        criterion = nn.MSELoss()

    input_dim = len(all_samples[0].features)
    model = MLP(input_dim, args.hidden_dim, args.num_layers, output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    train_ds = JsonlDataset(train_samples, args.task)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if val_samples:
        val_ds = JsonlDataset(val_samples, args.task)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if args.task == "classification":
                loss = criterion(logits, y)
            else:
                loss = criterion(logits.squeeze(1), y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

        train_loss = running_loss / max(1, seen)

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, args.task, criterion, device)
            if val_acc is not None:
                print(
                    f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f} val_acc={val_acc:.4f}"
                )
            else:
                print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        else:
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "task": args.task,
            "output_dim": output_dim,
        },
        output,
    )
    print(f"Saved model to: {output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a PyTorch model from JSONL data")
    p.add_argument("--input", required=True, help="Path to input JSONL")
    p.add_argument("--output", default="artifacts/model.pt", help="Path to output model file")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU training")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
