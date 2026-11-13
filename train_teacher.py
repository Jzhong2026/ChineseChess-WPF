"""CUDA distillation from offline teacher JSONL, with game-disjoint validation.

TeacherValue is a search estimate from the side-to-move perspective, NOT a
game result. Outputs are candidate artifacts; validation loss is not strength.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

from chesslib.encoding import encode_board
from train_v8_factored import CNNV8FactoredNet


class TeacherNet(CNNV8FactoredNet):
    """V8 backbone with an optional learned from/to interaction term."""

    def __init__(self, pair_rank=16, material_skip=False):
        super().__init__(policy_dropout=0.1, value_dropout=0.1)
        self.pair_rank = pair_rank
        self.material_skip = material_skip
        self.register_buffer("material_weights", torch.tensor([0, 200, 200, 400, 900, 450, 120,
                                                              0, -200, -200, -400, -900, -450, -120], dtype=torch.float32))
        if pair_rank:
            self.pair_from = nn.Conv2d(64, pair_rank, 1)
            self.pair_to = nn.Conv2d(64, pair_rank, 1)
            nn.init.zeros_(self.pair_to.weight)
            nn.init.zeros_(self.pair_to.bias)

    def forward(self, board):
        x = self.tower(self.stem(board))
        p = self.policy_dropout(F.relu(self.policy_bn(self.policy_conv(x))).flatten(1))
        policy = self.from_fc(p).unsqueeze(2) + self.to_fc(p).unsqueeze(1)
        if self.pair_rank:
            policy = policy + torch.bmm(self.pair_from(x).flatten(2).transpose(1, 2),
                                       self.pair_to(x).flatten(2)) / math.sqrt(self.pair_rank)
        v = self.value_dropout1(F.relu(self.value_bn(self.value_conv(x))).flatten(1))
        v = self.value_dropout2(F.relu(self.value_fc1(v)))
        value_logit = self.value_fc2(v).squeeze(1)
        if self.material_skip:
            # Exact piece counts generalize to unseen exchanges; the learned
            # residual still models position, king danger and tactical prospects.
            material = (board[:, :14].sum((2, 3)) * self.material_weights).sum(1)
            value_logit = value_logit + material * (board[:, 14, 0, 0] - board[:, 15, 0, 0]) / 600
        return policy.flatten(1), torch.tanh(value_logit)


def restore_fused_v8(model, path):
    """Recover folded Conv+BN as Conv(weight), BN(identity scale, folded bias)."""
    graph = onnx.load(path).graph
    arrays = {v.name: onnx.numpy_helper.to_array(v).copy() for v in graph.initializer}
    convs = [n for n in graph.node if n.op_type == "Conv"]
    pairs = [(model.stem[0], model.stem[1])]
    for block in model.tower:
        pairs.extend([(block.conv1, block.bn1), (block.conv2, block.bn2)])
    pairs.extend([(model.policy_conv, model.policy_bn), (model.value_conv, model.value_bn)])
    if len(convs) != len(pairs):
        raise ValueError("Warm start requires the deployed 64-channel / 6-block V8 graph")
    with torch.no_grad():
        for node, (conv, bn) in zip(convs, pairs):
            conv.weight.copy_(torch.from_numpy(arrays[node.input[1]]))
            bn.weight.fill_(math.sqrt(1 + bn.eps))
            bn.bias.copy_(torch.from_numpy(arrays[node.input[2]]))
            bn.running_mean.zero_()
            bn.running_var.fill_(1)
        for name in ("from_fc", "to_fc", "value_fc1", "value_fc2"):
            layer = getattr(model, name)
            layer.weight.copy_(torch.from_numpy(arrays[name + ".weight"]))
            layer.bias.copy_(torch.from_numpy(arrays[name + ".bias"]))
    model.eval()
    rng = np.random.default_rng(123)
    sample = rng.normal(size=(1, 16, 10, 9)).astype(np.float32)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = dict(zip([v.name for v in session.get_outputs()], session.run(None, {"board_input": sample})))
    actual_p, actual_v = model(torch.from_numpy(sample))
    expected_p = (expected["from_logits"][:, :, None] + expected["to_logits"][:, None, :]).reshape(1, 8100)
    errors = [float(np.max(np.abs(actual_p.detach().numpy() - expected_p))),
              float(np.max(np.abs(actual_v.detach().numpy() - expected["value_pred"])))]
    if max(errors) > 1e-3:
        raise ValueError(f"Warm start parity failed: {errors}")
    return errors


def read_split(paths):
    train, valid = {}, {}
    games = [set(), set()]
    raw = 0
    for path in paths:
        with open(path, encoding="utf-8-sig") as stream:
            for line in stream:
                row = json.loads(line)
                raw += 1
                if row["SelectedMove"] not in row["LegalMoves"]:
                    raise ValueError("Teacher selected an illegal move")
                if row["SideToMove"] not in (-1, 1) or not math.isfinite(row["TeacherValue"]):
                    raise ValueError("Invalid side/value")
                is_valid = int(hashlib.sha256(row["GameId"].encode()).hexdigest()[:8], 16) % 5 == 0
                games[int(is_valid)].add(row["GameId"])
                (valid if is_valid else train).setdefault(row["Fen"], row)
    overlaps = train.keys() & valid.keys()
    for key in overlaps:
        del train[key]
    assert not games[0] & games[1]
    if not train or not valid:
        raise ValueError("Both game-disjoint sets must be nonempty")
    return list(train.values()), list(valid.values()), dict(raw=raw, train=len(train), valid=len(valid),
        overlapping_positions_removed=len(overlaps), train_games=len(games[0]), valid_games=len(games[1]))


def tensors(rows, device, soft_policy=False):
    width = max(len(r["LegalMoves"]) for r in rows)
    legal = np.zeros((len(rows), width), dtype=np.int64)
    mask = np.zeros_like(legal, dtype=bool)
    target = []
    probabilities = np.zeros_like(legal, dtype=np.float32)
    for i, r in enumerate(rows):
        moves = r["LegalMoves"]
        legal[i, :len(moves)] = moves
        mask[i, :len(moves)] = True
        target.append(moves.index(r["SelectedMove"]))
        probabilities[i, target[-1]] = 1
        if soft_policy and r.get("Candidates"):
            candidates = [c for c in r["Candidates"] if c["Action"] in moves and c["Depth"] >= r["Depth"] - 1]
            if candidates:
                scores = np.asarray([c["Score"] for c in candidates], dtype=np.float32)
                weights = np.exp(np.clip((scores - scores.max()) / 80, -50, 0))
                weights /= weights.sum()
                probabilities[i] *= 0.5
                for candidate, weight in zip(candidates, weights):
                    probabilities[i, moves.index(candidate["Action"])] += 0.5 * weight
    return tuple(torch.as_tensor(v, device=device) for v in (
        np.stack([encode_board(r["BoardEncoding"], r["SideToMove"]) for r in rows]),
        legal, mask, np.asarray(target, dtype=np.int64),
        np.asarray([r["TeacherValue"] for r in rows], dtype=np.float32), probabilities))


def augment(x, legal):
    # One symmetry per batch is cheaper than per-sample indexing and remains unbiased.
    if random.random() < 0.5:
        x = x.flip(-1)
        frm, to = legal // 90, legal % 90
        legal = (frm // 9 * 9 + 8 - frm % 9) * 90 + (to // 9 * 9 + 8 - to % 9)
    if random.random() < 0.5:
        x = x[:, list(range(7, 14)) + list(range(7)) + [15, 14]].flip((-2, -1))
        legal = (89 - legal // 90) * 90 + (89 - legal % 90)
    return x, legal


def epoch(model, data, batch, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    count = len(data[0])
    indices = torch.randperm(count, device=data[0].device) if training else torch.arange(count, device=data[0].device)
    sums = torch.zeros(4, device=data[0].device)
    for ids in indices.split(batch):
        x, legal, mask, target, value, probabilities = [v[ids] for v in data]
        if training:
            x, legal = augment(x, legal)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
            p, v = model(x)
            logits = p.float().gather(1, legal).masked_fill(~mask, -1e9)
            policy_loss = F.cross_entropy(logits, target)
            policy_objective = -(F.log_softmax(logits, 1) * probabilities).sum(1).mean() if training else policy_loss
            value_loss = F.mse_loss(v.float(), value)
            loss = policy_objective + 2 * value_loss
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
        sums += torch.stack((loss.detach(), policy_loss.detach(), value_loss.detach(),
                            (logits.argmax(1) == target).float().mean())) * len(ids)
    return dict(zip(("loss", "policy_ce", "value_mse", "top1"), (sums / count).tolist()))


def export(model, output, sample):
    model = model.cpu().eval()
    sample = sample[:1].cpu()
    torch.onnx.export(model, sample, str(output), input_names=["board_input"],
                      output_names=["policy_logits", "value_pred"], opset_version=18,
                      dynamo=False, dynamic_axes={"board_input": {0: "batch"},
                                                "policy_logits": {0: "batch"}, "value_pred": {0: "batch"}})
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    expected = [v.detach().numpy() for v in model(sample)]
    actual = session.run(None, {"board_input": sample.numpy()})
    errors = [float(np.max(np.abs(a-b))) for a, b in zip(actual, expected)]
    if max(errors) > 1e-3:
        raise ValueError(f"ONNX parity failed: {errors}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pair-rank", type=int, default=16)
    parser.add_argument("--material-skip", action="store_true")
    parser.add_argument("--soft-policy", action="store_true", help="Mix bestmove CE with near-best MultiPV targets during training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "metadata.json").exists():
        raise FileExistsError("Use a new output directory to preserve earlier experiment evidence")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("This trainer requires CUDA; CPU fallback would hide a setup failure")
    sources = [dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.input]
    train_rows, valid_rows, split = read_split(args.input)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != source["sha256"] for p, source in zip(args.input, sources)):
        raise RuntimeError("Training input changed while loading; use a frozen snapshot")
    model = TeacherNet(args.pair_rank)
    parity = restore_fused_v8(model, args.warm_start) if args.warm_start else None
    if args.material_skip:
        model.material_skip = True
        # The restored value head predicts a complete value, not a residual.
        nn.init.zeros_(model.value_fc2.weight)
        nn.init.zeros_(model.value_fc2.bias)
    metadata = dict(config={k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v for k,v in vars(args).items()},
        split=split, warm_start_max_abs_error=parity, torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()),
        sources=sources, trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output_dir / "train_teacher.snapshot.py").write_bytes(Path(__file__).read_bytes())
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    train = tensors(train_rows, "cuda", args.soft_policy); valid = tensors(valid_rows, "cuda")
    model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda")
    best = math.inf
    with open(args.output_dir / "metrics.jsonl", "w", encoding="utf-8") as log:
        for index in range(args.epochs):
            start = time.perf_counter()
            training = epoch(model, train, args.batch_size, opt, scaler)
            validation = epoch(model, valid, args.batch_size)
            row = dict(epoch=index+1, train=training, valid=validation, seconds=time.perf_counter()-start,
                       peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20, lr=opt.param_groups[0]["lr"])
            print(json.dumps(row), flush=True); log.write(json.dumps(row)+"\n"); log.flush()
            if validation["loss"] < best:
                best = validation["loss"]
                torch.save(dict(model=model.state_dict(), pair_rank=args.pair_rank, material_skip=args.material_skip, epoch=index+1,
                                validation=validation), args.output_dir / "best.pt")
            schedule.step()
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    errors = export(model, args.output_dir / "candidate.onnx", valid[0])
    (args.output_dir / "export.json").write_text(json.dumps(dict(epoch=checkpoint["epoch"], validation=checkpoint["validation"],
        parity_max_abs_error=errors, sha256=hashlib.sha256((args.output_dir / "candidate.onnx").read_bytes()).hexdigest()), indent=2))
    print("COMPLETE " + json.dumps(dict(best_epoch=checkpoint["epoch"], parity=errors)), flush=True)


if __name__ == "__main__":
    main()
