import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

from train_teacher_value import ValueNet, canonical_features
from export_sparse_value import SparseValueNet, active_features


def test_color_rotation_has_identical_canonical_features():
    board = np.zeros((14, 10, 9), dtype=np.float32)
    board[3, 8, 2] = 1
    board[12, 2, 7] = 1
    rotated = np.concatenate((board[7:], board[:7]))[:, ::-1, ::-1]
    def row(planes, side):
        return dict(BoardEncoding=planes.reshape(14, 90).T.reshape(-1).tolist(), SideToMove=side)
    red = canonical_features(row(board, 1))
    black = canonical_features(row(rotated, -1))
    np.testing.assert_array_equal(red, black)
    np.testing.assert_array_equal(active_features(red), active_features(black))


def test_sparse_feature_sum_preserves_a_nonzero_dense_network():
    torch.manual_seed(123)
    dense = ValueNet().eval()
    # Exercise learned residuals; the default zero last layer would test only material.
    torch.nn.init.normal_(dense.layers[-1].weight, std=0.05)
    torch.nn.init.normal_(dense.layers[-1].bias, std=0.05)
    sparse = SparseValueNet(dense).eval()
    rng = np.random.default_rng(123)
    features = np.zeros((32, 1260), dtype=np.float32)
    for row in features:
        squares = rng.choice(90, 32, replace=False)
        planes = rng.integers(0, 14, 32)
        row[planes * 90 + squares] = 1
    active = np.stack([active_features(row) for row in features])
    with torch.no_grad():
        expected = dense(torch.from_numpy(features)).numpy()
        actual = sparse(torch.from_numpy(active)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0.01)
