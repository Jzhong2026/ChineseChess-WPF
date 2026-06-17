#!/usr/bin/env python3
"""
Demo 8: Text classification with PyTorch.

Run:
    python Learning/demo8_text_classification/text_classification_demo.py train
    python Learning/demo8_text_classification/text_classification_demo.py eval --model artifacts/text_sentiment_gru.pt
    python Learning/demo8_text_classification/text_classification_demo.py predict --text "这个项目体验很好"
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "text_sentiment_gru.pt"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


RAW_SAMPLES: list[tuple[str, int]] = [
    ("这个功能很好用", 1),
    ("界面清爽响应很快", 1),
    ("训练结果比预期更好", 1),
    ("这次更新非常稳定", 1),
    ("模型预测很准确", 1),
    ("我喜欢这个项目", 1),
    ("操作简单效果不错", 1),
    ("这个 demo 很适合学习", 1),
    ("速度快而且结果清楚", 1),
    ("体验顺畅没有卡顿", 1),
    ("文档解释很明白", 1),
    ("代码结构清晰", 1),
    ("这个功能不好用", 0),
    ("界面混乱响应很慢", 0),
    ("训练结果让人失望", 0),
    ("这次更新很不稳定", 0),
    ("模型预测经常错误", 0),
    ("我不喜欢这个项目", 0),
    ("操作复杂效果很差", 0),
    ("这个 demo 不适合学习", 0),
    ("速度慢而且结果模糊", 0),
    ("体验卡顿问题很多", 0),
    ("文档解释不清楚", 0),
    ("代码结构混乱", 0),
]


@dataclass(frozen=True)
class EncodedSample:
    token_ids: list[int]
    label: int


class TextDataset(Dataset):
    def __init__(self, samples: list[EncodedSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EncodedSample:
        return self.samples[index]


class SentimentGru(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 48, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(features)


def tokenize(text: str) -> list[str]:
    return [ch for ch in text.strip() if not ch.isspace()]


def build_vocab(samples: list[tuple[str, int]]) -> dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for text, _ in samples:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int]) -> list[int]:
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)]
    return token_ids or [vocab[UNK_TOKEN]]


def encode_samples(samples: list[tuple[str, int]], vocab: dict[str, int]) -> list[EncodedSample]:
    return [EncodedSample(encode_text(text, vocab), label) for text, label in samples]


def split_samples(seed: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    samples = RAW_SAMPLES.copy()
    random.Random(seed).shuffle(samples)
    test_size = max(4, len(samples) // 4)
    return samples[test_size:], samples[:test_size]


def collate_batch(batch: list[EncodedSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(sample.token_ids) for sample in batch], dtype=torch.long)
    max_length = int(lengths.max().item())
    token_ids = torch.zeros((len(batch), max_length), dtype=torch.long)
    labels = torch.tensor([sample.label for sample in batch], dtype=torch.long)

    for row, sample in enumerate(batch):
        token_ids[row, : len(sample.token_ids)] = torch.tensor(sample.token_ids, dtype=torch.long)

    return token_ids, lengths, labels


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(
    seed: int,
    batch_size: int,
) -> tuple[dict[str, int], DataLoader, DataLoader]:
    train_raw, test_raw = split_samples(seed)
    vocab = build_vocab(train_raw)
    train_dataset = TextDataset(encode_samples(train_raw, vocab))
    test_dataset = TextDataset(encode_samples(test_raw, vocab))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    return vocab, train_loader, test_loader


def build_test_loader(seed: int, batch_size: int, vocab: dict[str, int]) -> DataLoader:
    _, test_raw = split_samples(seed)
    test_dataset = TextDataset(encode_samples(test_raw, vocab))
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for token_ids, lengths, labels in loader:
        token_ids = token_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(token_ids, lengths)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_count = 0

    for token_ids, lengths, labels in loader:
        token_ids = token_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        logits = model(token_ids, lengths)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / max(total_count, 1), correct / max(total_count, 1)


def save_checkpoint(model: nn.Module, vocab: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "vocab": vocab}, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[SentimentGru, dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"model checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]
    model = SentimentGru(vocab_size=len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, vocab


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    vocab, train_loader, test_loader = build_loaders(args.seed, args.batch_size)
    model = SentimentGru(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Train samples: {len(train_loader.dataset)}, test samples: {len(test_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        if epoch == 1 or epoch % args.report_every == 0 or epoch == args.epochs:
            test_loss, test_accuracy = evaluate(model, test_loader, device)
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_accuracy={test_accuracy:.2%}"
            )

    save_checkpoint(model, vocab, args.model)
    print(f"Saved model: {args.model}")


def eval_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model, vocab = load_checkpoint(args.model, device)
    test_loader = build_test_loader(args.seed, args.batch_size, vocab)
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    print(f"Device: {device}")
    print(f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.2%}")


@torch.no_grad()
def predict_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model, vocab = load_checkpoint(args.model, device)
    model.eval()

    token_ids = torch.tensor([encode_text(args.text, vocab)], dtype=torch.long)
    lengths = torch.tensor([token_ids.size(1)], dtype=torch.long)
    logits = model(token_ids.to(device), lengths.to(device))
    probabilities = F.softmax(logits, dim=1).squeeze(0).cpu()

    negative = float(probabilities[0].item())
    positive = float(probabilities[1].item())
    label = "positive" if positive >= negative else "negative"
    print(f"Text: {args.text}")
    print(f"Prediction: {label}")
    print(f"Scores: negative={negative:.2%}, positive={positive:.2%}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch text classification demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a GRU sentiment classifier.")
    add_common_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--learning-rate", type=float, default=2e-3)
    train_parser.add_argument("--report-every", type=int, default=10)
    train_parser.set_defaults(func=train_command)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved classifier.")
    add_common_args(eval_parser)
    eval_parser.set_defaults(func=eval_command)

    predict_parser = subparsers.add_parser("predict", help="Predict one text sample.")
    add_common_args(predict_parser)
    predict_parser.add_argument("--text", required=True)
    predict_parser.set_defaults(func=predict_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
