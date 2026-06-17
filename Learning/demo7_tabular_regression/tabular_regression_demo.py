#!/usr/bin/env python3
"""
Demo 7: Tabular regression with PyTorch.

Run:
    python Learning/demo7_tabular_regression/tabular_regression_demo.py train
    python Learning/demo7_tabular_regression/tabular_regression_demo.py eval --model artifacts/tabular_house_price.pt
    python Learning/demo7_tabular_regression/tabular_regression_demo.py predict --area 96 --rooms 3 --age 8 --distance 6 --school-score 7
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "tabular_house_price.pt"

FEATURE_NAMES = ["area", "rooms", "age", "distance", "school_score"]


@dataclass(frozen=True)
class Normalizer:
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = self.x_mean.to(x.device)
        x_std = self.x_std.to(x.device)
        return (x - x_mean) / x_std.clamp_min(1e-6)

    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        y_mean = self.y_mean.to(y.device)
        y_std = self.y_std.to(y.device)
        return (y - y_mean) / y_std.clamp_min(1e-6)

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        y_mean = self.y_mean.to(y.device)
        y_std = self.y_std.to(y.device)
        return y * y_std.clamp_min(1e-6) + y_mean


class HousePriceMlp(nn.Module):
    def __init__(self, input_dim: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_synthetic_housing_data(samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    area = torch.empty(samples).uniform_(45, 180, generator=generator)
    rooms = torch.randint(1, 6, (samples,), generator=generator).float()
    age = torch.empty(samples).uniform_(0, 30, generator=generator)
    distance = torch.empty(samples).uniform_(1, 25, generator=generator)
    school_score = torch.empty(samples).uniform_(1, 10, generator=generator)

    noise = torch.randn(samples, generator=generator) * 8.0
    price = (
        32.0
        + area * 1.75
        + rooms * 16.0
        - age * 2.2
        - distance * 4.0
        + school_score * 13.0
        + area * school_score * 0.035
        + noise
    )

    x = torch.stack([area, rooms, age, distance, school_score], dim=1)
    y = price.float()
    return x.float(), y


def split_train_test(
    x: torch.Tensor,
    y: torch.Tensor,
    test_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(x), generator=generator)
    test_size = int(len(x) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]


def build_normalizer(x_train: torch.Tensor, y_train: torch.Tensor) -> Normalizer:
    return Normalizer(
        x_mean=x_train.mean(dim=0),
        x_std=x_train.std(dim=0),
        y_mean=y_train.mean(),
        y_std=y_train.std(),
    )


def build_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_count += features.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    normalizer: Normalizer,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    predictions_norm = model(x.to(device)).cpu()
    predictions = normalizer.denormalize_y(predictions_norm)
    mae = torch.mean(torch.abs(predictions - y)).item()
    rmse = torch.sqrt(torch.mean((predictions - y) ** 2)).item()
    return mae, rmse


def save_checkpoint(model: nn.Module, normalizer: Normalizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "x_mean": normalizer.x_mean,
            "x_std": normalizer.x_std,
            "y_mean": normalizer.y_mean,
            "y_std": normalizer.y_std,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[HousePriceMlp, Normalizer]:
    if not path.exists():
        raise FileNotFoundError(f"model checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = HousePriceMlp(input_dim=len(checkpoint["feature_names"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    normalizer = Normalizer(
        x_mean=checkpoint["x_mean"],
        x_std=checkpoint["x_std"],
        y_mean=checkpoint["y_mean"],
        y_std=checkpoint["y_std"],
    )
    return model, normalizer


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    x, y = make_synthetic_housing_data(args.samples, args.seed)
    x_train, y_train, x_test, y_test = split_train_test(x, y, args.test_ratio, args.seed)
    normalizer = build_normalizer(x_train, y_train)

    x_train_norm = normalizer.normalize_x(x_train)
    y_train_norm = normalizer.normalize_y(y_train)
    x_test_norm = normalizer.normalize_x(x_test)

    train_loader = build_loader(x_train_norm, y_train_norm, args.batch_size, shuffle=True)
    model = HousePriceMlp().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device}")
    print(f"Train samples: {len(x_train)}, test samples: {len(x_test)}")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        if epoch == 1 or epoch % args.report_every == 0 or epoch == args.epochs:
            mae, rmse = evaluate(model, x_test_norm, y_test, normalizer, device)
            print(f"epoch={epoch:03d} loss={loss:.4f} mae={mae:.2f} rmse={rmse:.2f}")

    save_checkpoint(model, normalizer, args.model)
    print(f"Saved model: {args.model}")
    print("Unit: price is in 10k CNY.")


def eval_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model, normalizer = load_checkpoint(args.model, device)
    x, y = make_synthetic_housing_data(args.samples, args.seed)
    _, _, x_test, y_test = split_train_test(x, y, args.test_ratio, args.seed)
    x_test_norm = normalizer.normalize_x(x_test)
    mae, rmse = evaluate(model, x_test_norm, y_test, normalizer, device)
    print(f"Device: {device}")
    print(f"mae={mae:.2f} rmse={rmse:.2f} unit=10k CNY")


@torch.no_grad()
def predict_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model, normalizer = load_checkpoint(args.model, device)
    model.eval()

    features = torch.tensor(
        [[args.area, args.rooms, args.age, args.distance, args.school_score]],
        dtype=torch.float32,
    )
    normalized = normalizer.normalize_x(features).to(device)
    prediction_norm = model(normalized).cpu()
    prediction = normalizer.denormalize_y(prediction_norm).item()

    print("Input:")
    for name, value in zip(FEATURE_NAMES, features.squeeze(0).tolist()):
        print(f"  {name}: {value:g}")
    print(f"Predicted price: {prediction:.2f} (10k CNY)")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch tabular regression demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a house-price regressor.")
    add_common_args(train_parser)
    train_parser.add_argument("--samples", type=int, default=5000)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--test-ratio", type=float, default=0.2)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--report-every", type=int, default=10)
    train_parser.set_defaults(func=train_command)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved regressor.")
    add_common_args(eval_parser)
    eval_parser.add_argument("--samples", type=int, default=5000)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--test-ratio", type=float, default=0.2)
    eval_parser.set_defaults(func=eval_command)

    predict_parser = subparsers.add_parser("predict", help="Predict one house price.")
    add_common_args(predict_parser)
    predict_parser.add_argument("--area", type=float, required=True)
    predict_parser.add_argument("--rooms", type=float, required=True)
    predict_parser.add_argument("--age", type=float, required=True)
    predict_parser.add_argument("--distance", type=float, required=True)
    predict_parser.add_argument("--school-score", type=float, required=True)
    predict_parser.set_defaults(func=predict_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
