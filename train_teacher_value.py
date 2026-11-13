"""Train a small canonical ONNX centipawn evaluator for neural alpha-beta."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

from train_teacher import read_split


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1260, 256), nn.ReLU(), nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        self.register_buffer('piece_values', torch.tensor([0, 200, 200, 400, 900, 450, 120,
                                                         0, -200, -200, -400, -900, -450, -120], dtype=torch.float32))

    def forward(self, board_input):
        material = (board_input.reshape(-1, 14, 90).sum(2) * self.piece_values).sum(1)
        return material + 600 * self.layers(board_input).squeeze(1)


def canonical_features(row):
    board = np.asarray(row['BoardEncoding'], dtype=np.float32).reshape(90, 14).T.reshape(14, 10, 9)
    if row['SideToMove'] == -1:
        board = np.concatenate((board[7:], board[:7]))[:, ::-1, ::-1]
    return board.reshape(1260).copy()


def tensors(rows):
    return (torch.as_tensor(np.stack([canonical_features(r) for r in rows]), device='cuda'),
            torch.tensor([max(-2400, min(2400, r['TeacherCp'])) / 600 for r in rows], device='cuda'),
            torch.tensor([r['TeacherValue'] for r in rows], device='cuda'))


def epoch(model, data, batch_size, optimizer=None, scaler=None):
    model.train(optimizer is not None)
    indices = torch.randperm(len(data[0]), device='cuda') if optimizer else torch.arange(len(data[0]), device='cuda')
    sums = torch.zeros(3, device='cuda')
    for ids in indices.split(batch_size):
        x, cp, value = [item[ids] for item in data]
        if optimizer:
            if random.random() < 0.5:
                x = x.reshape(-1, 14, 10, 9).flip(-1).reshape(-1, 1260)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None), torch.autocast('cuda', dtype=torch.float16):
            pred = model(x).float() / 600
            huber = F.huber_loss(pred, cp)
            mse = F.mse_loss(pred.tanh(), value)
            loss = huber + mse
        if optimizer:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
        sums += torch.stack([loss.detach(), huber.detach(), mse.detach()]) * len(ids)
    return dict(zip(['loss', 'cp_huber', 'value_mse'], (sums / len(data[0])).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'metadata.json').exists():
        raise FileExistsError('Use a new experiment output directory')
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    sources = [dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.input]
    training, validation, split = read_split(args.input)
    if any(s['sha256'] != hashlib.sha256(p.read_bytes()).hexdigest() for s, p in zip(sources, args.input)):
        raise RuntimeError('Input changed during load')
    model = ValueNet().cuda()
    metadata = dict(split=split, sources=sources, encoding='canonical-piece-planes-14-v1',
                    score='current-player-centipawns', seed=42, epochs=args.epochs, batch_size=args.batch_size,
                    torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata, indent=2))
    (args.output_dir/'trainer.snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(metadata), flush=True)
    train, valid = tensors(training), tensors(validation)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=5e-5)
    scaler = torch.amp.GradScaler('cuda')
    best = float('inf')
    with (args.output_dir/'metrics.jsonl').open('w') as log:
        for i in range(args.epochs):
            start = time.perf_counter()
            tr = epoch(model, train, args.batch_size, opt, scaler)
            va = epoch(model, valid, args.batch_size)
            row = dict(epoch=i+1, train=tr, valid=va, seconds=time.perf_counter()-start,
                       peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
            print(json.dumps(row), flush=True); log.write(json.dumps(row)+'\n'); log.flush()
            if va['value_mse'] < best:
                best = va['value_mse']
                torch.save(dict(model=model.state_dict(), epoch=i+1, validation=va), args.output_dir/'best.pt')
            scheduler.step()
    checkpoint = torch.load(args.output_dir/'best.pt', weights_only=True, map_location='cpu')
    model.cpu().eval().load_state_dict(checkpoint['model'])
    sample = valid[0][:8].cpu()
    path = args.output_dir/'value.onnx'
    torch.onnx.export(model, sample[:1], str(path), input_names=['board_input'], output_names=['value_cp'],
                      dynamic_axes={'board_input': {0: 'batch'}, 'value_cp': {0: 'batch'}}, opset_version=18, dynamo=False)
    graph = onnx.load(path)
    onnx.helper.set_model_props(graph, {'encoding': metadata['encoding'], 'score': metadata['score']})
    onnx.save(graph, path); onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    with torch.no_grad(): expected = model(sample).numpy()
    actual = session.run(None, {'board_input': sample.numpy()})[0]
    error = float(np.max(np.abs(actual-expected)))
    if error > 0.01:
        raise RuntimeError(f'ONNX parity error in centipawns: {error}')
    report = dict(epoch=checkpoint['epoch'], validation=checkpoint['validation'], parity_max_cp_error=error,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (args.output_dir/'export.json').write_text(json.dumps(report, indent=2))
    print('COMPLETE '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
