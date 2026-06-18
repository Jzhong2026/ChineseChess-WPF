#!/usr/bin/env python3
"""
Memory-efficient V7 continue training with chunked data loading.

Usage:
    python train_v7_combined.py \
        --input data/selfplay/train_combined.jsonl \
        --checkpoint artifacts/cnn_policy_value_v7_cont.best.pt \
        --output artifacts/cnn_policy_value_v7_cont2.pt \
        --epochs 50 --batch-size 256 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ─── Constants ─────────────────────────────────────────────────────────
BOARD_PLANES = 14
SIDE_PLANES = 2
INPUT_PLANES = BOARD_PLANES + SIDE_PLANES  # 16
BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS  # 90
INPUT_DIM = BOARD_PLANES * BOARD_ROWS * BOARD_COLS  # 1260
ACTION_DIM = BOARD_SIZE * BOARD_SIZE  # 8100
INVALID_LOGIT = -1e9
CHUNK_SIZE = 5000


# ─── Model (unchanged) ────────────────────────────────────────────────

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
    def __init__(self, in_channels: int = INPUT_PLANES, channels: int = 64,
                 res_blocks: int = 6, policy_dropout: float = 0.2,
                 value_dropout: float = 0.2, policy_channels: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.policy_channels = policy_channels

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
        p = self.policy_dropout(p)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = self.value_dropout1(v)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_dropout2(v)
        value_pred = torch.tanh(self.value_fc2(v)).squeeze(1)

        return policy_logits, value_pred


# ─── Chunked Dataset ──────────────────────────────────────────────────

def _encode_features(board_encoding, side_to_move):
    """Convert raw board encoding + side to [16, 10, 9] feature tensor."""
    flat = torch.tensor(board_encoding, dtype=torch.float32)
    board_features = flat.view(BOARD_SIZE, BOARD_PLANES).permute(1, 0).view(BOARD_PLANES, BOARD_ROWS, BOARD_COLS)
    side_planes = torch.zeros(SIDE_PLANES, BOARD_ROWS, BOARD_COLS, dtype=torch.float32)
    if side_to_move == 1:
        side_planes[0] = 1.0
    else:
        side_planes[1] = 1.0
    return torch.cat([board_features, side_planes], dim=0)


def _remap_legal_moves(legal_moves, flip_columns=False):
    """Remap legal moves for horizontal flip augmentation."""
    if not flip_columns:
        return legal_moves
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


def preprocess_jsonl(jsonl_path: str, chunk_dir: str):
    """
    Read JSONL line by line, group into chunks, save each chunk as a .pt file.
    Each chunk is a dict with stacked tensors.
    """
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_idx = 0
    buf_features = []
    buf_moves = []
    buf_values = []
    buf_legal = []
    buf_vw = []
    buf_pw = []
    total_samples = 0
    line_no = 0
    skipped = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
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

            # Validate
            if not isinstance(board_encoding, list) or len(board_encoding) != INPUT_DIM:
                skipped += 1
                continue
            if not isinstance(legal_moves, list) or len(legal_moves) == 0:
                skipped += 1
                continue
            if not isinstance(selected_move, int) or not (0 <= selected_move < ACTION_DIM):
                skipped += 1
                continue
            if side_to_move not in (1, -1):
                skipped += 1
                continue

            # Encode features
            features = _encode_features(board_encoding, side_to_move)
            move_target = torch.tensor(selected_move, dtype=torch.long)
            value_target = torch.tensor(result * side_to_move, dtype=torch.float32)
            legal_tensor = torch.tensor(legal_moves, dtype=torch.long)
            vw = torch.tensor(obj.get("ValueWeight", 1.0), dtype=torch.float32)
            pw = torch.tensor(obj.get("PolicyWeight", 1.0), dtype=torch.float32)

            buf_features.append(features)
            buf_moves.append(move_target)
            buf_values.append(value_target)
            buf_legal.append(legal_tensor)
            buf_vw.append(vw)
            buf_pw.append(pw)

            if len(buf_features) >= CHUNK_SIZE:
                _save_chunk(chunk_dir, chunk_idx, buf_features, buf_moves, buf_values, buf_legal, buf_vw, buf_pw)
                total_samples += len(buf_features)
                buf_features, buf_moves, buf_values, buf_legal, buf_vw, buf_pw = [], [], [], [], [], []
                chunk_idx += 1
                print(f"  Chunk {chunk_idx} saved ({total_samples} total so far)", flush=True)

    # Flush remaining
    if buf_features:
        _save_chunk(chunk_dir, chunk_idx, buf_features, buf_moves, buf_values, buf_legal, buf_vw, buf_pw)
        total_samples += len(buf_features)
        chunk_idx += 1

    print(f"Preprocessing complete: {total_samples} samples in {chunk_idx} chunks "
          f"({skipped} skipped, {line_no - total_samples - skipped} filtered).")

    # Save metadata
    meta = {"num_chunks": chunk_idx, "total_samples": total_samples, "chunk_size": CHUNK_SIZE}
    torch.save(meta, os.path.join(chunk_dir, "meta.pt"))
    return chunk_idx, total_samples


def _save_chunk(chunk_dir, idx, features, moves, values, legal, vw, pw):
    """Save one chunk as a .pt file."""
    path = os.path.join(chunk_dir, f"chunk_{idx:04d}.pt")
    # We store legal moves as a list of tensors (variable length)
    # Everything else is stacked
    torch.save({
        "features": torch.stack(features),
        "moves": torch.stack(moves),
        "values": torch.stack(values),
        "legal": legal,  # list of tensors
        "vw": torch.stack(vw),
        "pw": torch.stack(pw),
    }, path)


class ChunkedDataset(Dataset):
    """
    Dataset that loads one chunk at a time from disk.
    The outer loop (trainer) iterates over chunks and calls load_chunk().
    """

    def __init__(self, chunk_dir: str):
        self.chunk_dir = chunk_dir
        meta = torch.load(os.path.join(chunk_dir, "meta.pt"), weights_only=True)
        self.num_chunks = meta["num_chunks"]
        self.total_samples = meta["total_samples"]

        self._features: torch.Tensor = None
        self._moves: torch.Tensor = None
        self._values: torch.Tensor = None
        self._legal: list = None
        self._vw: torch.Tensor = None
        self._pw: torch.Tensor = None

    def load_chunk(self, chunk_idx: int):
        """Load a specific chunk into memory."""
        path = os.path.join(self.chunk_dir, f"chunk_{chunk_idx:04d}.pt")
        data = torch.load(path, weights_only=True)
        self._features = data["features"]
        self._moves = data["moves"]
        self._values = data["values"]
        self._legal = data["legal"]
        self._vw = data["vw"]
        self._pw = data["pw"]

    def __len__(self):
        if self._features is None:
            return 0
        return self._features.size(0)

    def __getitem__(self, idx: int):
        # Data augmentation: horizontal flip with 50% chance
        features = self._features[idx]
        move = torch.as_tensor(self._moves[idx], dtype=torch.long)
        value = torch.as_tensor(self._values[idx], dtype=torch.float32)
        legal = self._legal[idx]
        vw = torch.as_tensor(self._vw[idx], dtype=torch.float32)
        pw = torch.as_tensor(self._pw[idx], dtype=torch.float32)

        if random.random() < 0.5:
            features = features.flip(-1)
            from_idx = move.item() // BOARD_SIZE
            to_idx = move.item() % BOARD_SIZE
            f_r, f_c = from_idx // BOARD_COLS, from_idx % BOARD_COLS
            t_r, t_c = to_idx // BOARD_COLS, to_idx % BOARD_COLS
            f_c = BOARD_COLS - 1 - f_c
            t_c = BOARD_COLS - 1 - t_c
            move = torch.tensor((f_r * BOARD_COLS + f_c) * BOARD_SIZE + (t_r * BOARD_COLS + t_c), dtype=torch.long)

        legal_mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
        legal_mask[legal] = True
        return features, move, value, legal_mask, vw, pw


# ─── Training ─────────────────────────────────────────────────────────

def train():
    parser = argparse.ArgumentParser(description="Memory-efficient V7 continue training")
    parser.add_argument("--input", required=True, help="JSONL input file")
    parser.add_argument("--checkpoint", required=True, help="V7 .best.pt checkpoint to continue from")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-dir", default="data/chunks_combined",
                        help="Temp directory for preprocessed chunks")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only preprocess, don't train")
    args = parser.parse_args()

    chunk_dir = args.chunk_dir

    # Step 1: Preprocess JSONL → chunked .pt files (if not already done)
    if not os.path.exists(os.path.join(chunk_dir, "meta.pt")):
        print("Preprocessing JSONL into chunks...", flush=True)
        num_chunks, total_samples = preprocess_jsonl(args.input, chunk_dir)
    else:
        meta = torch.load(os.path.join(chunk_dir, "meta.pt"), weights_only=True)
        num_chunks = meta["num_chunks"]
        total_samples = meta["total_samples"]
        print(f"Using existing chunks: {num_chunks} chunks, {total_samples} samples", flush=True)

    if args.preprocess_only:
        print("Preprocessing done. Exiting (--preprocess-only).")
        return

    # Step 2: Set aside last chunk for validation
    train_chunks = list(range(num_chunks - 1)) if num_chunks > 1 else [0]
    val_chunk_idx = num_chunks - 1 if num_chunks > 1 else 0
    print(f"Train chunks: {len(train_chunks)}, Val chunk: {val_chunk_idx}", flush=True)

    # Step 3: Build model and load checkpoint
    device = torch.device('cpu')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = CNNFullPolicyValueNet(
        in_channels=INPUT_PLANES,
        channels=ckpt.get("channels", 64),
        res_blocks=ckpt.get("res_blocks", 6),
        policy_dropout=ckpt.get("policy_dropout", 0.2),
        value_dropout=ckpt.get("value_dropout", 0.2),
        policy_channels=ckpt.get("policy_channels", 8),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model: {total:,} params, architecture={ckpt.get('architecture', 'unknown')}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Step 4: Training loop
    dataset = ChunkedDataset(chunk_dir)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        random.shuffle(train_chunks)

        for ci in train_chunks:
            dataset.load_chunk(ci)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, pin_memory=False)

            for features, move_target, value_target, legal_mask, value_weight, policy_weight in loader:
                features = features.to(device)
                move_target = move_target.to(device)
                value_target = value_target.to(device)
                legal_mask = legal_mask.to(device)
                value_weight = value_weight.to(device)
                policy_weight = policy_weight.to(device)

                policy_logits, value_pred = model(features)

                per_sample_p_loss = F.cross_entropy(policy_logits, move_target, reduction='none')
                masked_logits = policy_logits.masked_fill(~legal_mask, INVALID_LOGIT)
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

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_count = 0
        val_top5 = 0
        val_top10 = 0

        dataset.load_chunk(val_chunk_idx)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=False)

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

                _, top5_idx = masked_logits.topk(5, dim=1)
                _, top10_idx = masked_logits.topk(10, dim=1)
                val_top5 += (top5_idx == move_target.unsqueeze(1)).any(dim=1).sum().item()
                val_top10 += (top10_idx == move_target.unsqueeze(1)).any(dim=1).sum().item()

        train_loss /= train_count if train_count > 0 else 1
        val_loss /= val_count if val_count > 0 else 1
        train_acc = train_correct / train_count if train_count > 0 else 0
        val_acc = val_correct / val_count if val_count > 0 else 0
        val_top5_acc = val_top5 / val_count if val_count > 0 else 0
        val_top10_acc = val_top10 / val_count if val_count > 0 else 0

        scheduler.step()

        print(f"epoch={epoch:03d} lr={optimizer.param_groups[0]['lr']:.2e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_top5={val_top5_acc:.4f} val_top10={val_top10_acc:.4f}", flush=True)

        # Save best
        if not (val_loss != val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(args.output).with_suffix(".best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "cnn_full_v7",
                "channels": ckpt.get("channels", 64),
                "res_blocks": ckpt.get("res_blocks", 6),
                "policy_channels": ckpt.get("policy_channels", 8),
                "planes": INPUT_PLANES,
                "board_planes": BOARD_PLANES,
                "board_rows": BOARD_ROWS,
                "board_cols": BOARD_COLS,
                "input_dim": INPUT_DIM,
                "action_dim": ACTION_DIM,
                "continued_from": str(args.checkpoint),
                "training_samples": total_samples,
            }, best_path)
            print(f"  Best model saved: {best_path.name}", flush=True)

        # Periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            periodic_path = Path(args.output).with_suffix(f".epoch{epoch}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "cnn_full_v7",
                "epoch": epoch,
            }, periodic_path)
            print(f"  Periodic checkpoint: {periodic_path.name}", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "cnn_full_v7",
        "channels": ckpt.get("channels", 64),
        "res_blocks": ckpt.get("res_blocks", 6),
        "policy_channels": ckpt.get("policy_channels", 8),
        "planes": INPUT_PLANES,
        "board_planes": BOARD_PLANES,
        "board_rows": BOARD_ROWS,
        "board_cols": BOARD_COLS,
        "input_dim": INPUT_DIM,
        "action_dim": ACTION_DIM,
        "continued_from": str(args.checkpoint),
        "training_samples": total_samples,
    }, args.output)
    print(f"\nSaved final: {args.output}")
    if best_val_loss < float("inf"):
        print(f"Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
