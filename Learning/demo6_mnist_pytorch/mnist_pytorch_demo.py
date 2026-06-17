#!/usr/bin/env python3
"""
Demo 6: Handwritten digit recognition with PyTorch and MNIST.

Run:
    python Learning/demo6_mnist_pytorch/mnist_pytorch_demo.py train
    python Learning/demo6_mnist_pytorch/mnist_pytorch_demo.py eval --model artifacts/mnist_cnn.pt
    python Learning/demo6_mnist_pytorch/mnist_pytorch_demo.py predict --model artifacts/mnist_cnn.pt --image path/to/digit.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "mnist"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "mnist_cnn.pt"


class SmallMnistCnn(nn.Module):
    """A compact CNN that is intentionally easy to read for a first demo."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def maybe_limit(dataset: datasets.MNIST, limit: int) -> Iterable:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def build_loaders(
    data_dir: Path,
    batch_size: int,
    train_limit: int,
    test_limit: int,
) -> tuple[DataLoader, DataLoader]:
    transform = build_transform()
    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        maybe_limit(train_dataset, train_limit),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        maybe_limit(test_dataset, test_limit),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        if log_interval > 0 and batch_index % log_interval == 0:
            print(f"  batch {batch_index:4d}/{len(loader)} loss={loss.item():.4f}")

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total_count += batch_size

    accuracy = correct / max(total_count, 1)
    average_loss = total_loss / max(total_count, 1)
    return average_loss, accuracy


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)


def load_checkpoint(path: Path, device: torch.device) -> SmallMnistCnn:
    if not path.exists():
        raise FileNotFoundError(f"model checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model = SmallMnistCnn().to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def train_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    train_loader, test_loader = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
    model = SmallMnistCnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            log_interval=args.log_interval,
        )
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        print(
            f"  train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_accuracy={test_accuracy:.2%}"
        )

    save_checkpoint(model, args.model)
    print(f"Saved model: {args.model}")


def eval_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    _, test_loader = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_limit=1,
        test_limit=args.test_limit,
    )
    model = load_checkpoint(args.model, device)
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    print(f"Device: {device}")
    print(f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.2%}")


def preprocess_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((28, 28))
    transform = build_transform()
    return transform(image).unsqueeze(0)


@torch.no_grad()
def predict_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model = load_checkpoint(args.model, device)
    model.eval()

    image = preprocess_image(args.image).to(device)
    logits = model(image)
    probabilities = F.softmax(logits, dim=1).squeeze(0)
    digit = int(probabilities.argmax().item())
    confidence = float(probabilities[digit].item())

    ranked = torch.argsort(probabilities, descending=True)[:3]
    top3 = ", ".join(f"{int(i)}={float(probabilities[i]):.2%}" for i in ranked)
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Top 3: {top3}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch MNIST handwritten digit demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the CNN on MNIST.")
    add_common_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--train-limit", type=int, default=12000)
    train_parser.add_argument("--test-limit", type=int, default=2000)
    train_parser.add_argument("--log-interval", type=int, default=25)
    train_parser.set_defaults(func=train_command)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved model.")
    add_common_args(eval_parser)
    eval_parser.add_argument("--test-limit", type=int, default=2000)
    eval_parser.set_defaults(func=eval_command)

    predict_parser = subparsers.add_parser("predict", help="Predict one digit image.")
    add_common_args(predict_parser)
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.set_defaults(func=predict_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
