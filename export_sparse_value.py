"""Export the same value MLP as active-feature sums; no retraining or score change."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

from train_teacher_value import ValueNet, canonical_features


class SparseValueNet(nn.Module):
    def __init__(self, dense):
        super().__init__()
        self.register_buffer('features', torch.cat((dense.layers[0].weight.detach().T, torch.zeros(1, 256))))
        self.register_buffer('first_bias', dense.layers[0].bias.detach().clone())
        self.register_buffer('material', torch.cat((dense.piece_values.repeat_interleave(90), torch.zeros(1))))
        self.hidden = dense.layers[2]
        self.head = dense.layers[4]

    def forward(self, board_indices):
        x = F.embedding(board_indices, self.features).sum(1) + self.first_bias
        residual = self.head(F.relu(self.hidden(F.relu(x)))).squeeze(1)
        return self.material[board_indices].sum(1) + 600 * residual


def active_features(flat):
    indices = np.flatnonzero(flat).astype(np.int64)
    result = np.full(90, 1260, dtype=np.int64)
    result[:len(indices)] = indices
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'value.onnx'
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    dense = ValueNet().eval(); dense.load_state_dict(checkpoint['model'])
    model = SparseValueNet(dense).eval()
    with args.samples.open(encoding='utf-8-sig') as stream:
        rows = [json.loads(next(stream)) for _ in range(256)]
    flat = np.stack([canonical_features(r) for r in rows])
    active = np.stack([active_features(x) for x in flat])
    torch.onnx.export(model, torch.from_numpy(active[:1]), str(output), input_names=['board_indices'],
                      output_names=['value_cp'], dynamic_axes={'board_indices': {0: 'batch'}, 'value_cp': {0: 'batch'}},
                      opset_version=18, dynamo=False)
    graph = onnx.load(output)
    onnx.helper.set_model_props(graph, {'encoding': 'canonical-active-indices-90-v1', 'score': 'current-player-centipawns'})
    onnx.save(graph, output); onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'board_indices': active})[0]
    with torch.no_grad(): expected = dense(torch.from_numpy(flat)).numpy()
    error = float(np.max(np.abs(actual - expected)))
    if error > 0.01: raise RuntimeError(f'Sparse equivalence failed: {error}cp')
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  positions=len(rows), dense_sparse_max_cp_error=error,
                  model_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), encoding='canonical-active-indices-90-v1')
    (args.output_dir/'export.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
