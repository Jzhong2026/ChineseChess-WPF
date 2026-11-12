from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ONNX_MODEL = REPO_ROOT / "artifacts" / "cnn_policy_value.onnx"

BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
BOARD_PLANES = 14


def source_of(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def tree_of(rel_path: str) -> ast.Module:
    return ast.parse(source_of(rel_path), filename=rel_path)


def _const_assignments(tree: ast.Module, ns: dict) -> list:
    """Collect module-level constant assignments, resolving simple arithmetic (10 * 9)."""
    body = []
    consts = {}
    allowed = (ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
               ast.Mult, ast.FloorDiv, ast.Div, ast.Mod, ast.USub, ast.UAdd, ast.Load,
               ast.Dict, ast.Tuple, ast.List, ast.Set)
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        if not all(isinstance(n, allowed) for n in ast.walk(node.value)):
            continue
        try:
            value = eval(compile(ast.Expression(node.value), "<const>", "eval"), {}, consts)
        except Exception:
            continue
        consts[node.targets[0].id] = value
        body.append(node)
    ns.update(consts)
    return body


def _import_nodes(tree: ast.Module, ns: dict) -> list:
    """Bind stdlib imports into ns; skip unavailable ones (e.g. torch).

    The import statements are deliberately not re-executed so extra_ns can override them.
    """
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        probe = {}
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<import>", "exec"), probe)
        except Exception:
            continue
        ns.update(probe)
    return []


def extract_symbols(rel_path: str, names, extra_ns=None) -> dict:
    """exec only the requested top-level functions/classes plus literals and stdlib imports.

    Lets us test torch-importing modules without torch installed.
    """
    tree = tree_of(rel_path)
    wanted = set(names)
    ns = {"__name__": f"extracted_{Path(rel_path).stem}"}
    body = _import_nodes(tree, ns) + _const_assignments(tree, ns)
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            body.append(node)
            found.add(node.name)
    missing = wanted - found
    if missing:
        raise KeyError(f"{rel_path}: symbols not defined: {sorted(missing)}")
    if extra_ns:
        ns.update(extra_ns)
    exec(compile(ast.Module(body=body, type_ignores=[]), rel_path, "exec"), ns)
    return ns


def extract_method(rel_path: str, class_name: str, method_name: str, extra_ns=None) -> dict:
    tree = tree_of(rel_path)
    ns = {"__name__": f"extracted_{Path(rel_path).stem}"}
    body = _import_nodes(tree, ns) + _const_assignments(tree, ns)
    target = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                    target = sub
    if target is None:
        raise KeyError(f"{rel_path}: {class_name}.{method_name} not found")
    body.append(target)
    if extra_ns:
        ns.update(extra_ns)
    exec(compile(ast.Module(body=body, type_ignores=[]), rel_path, "exec"), ns)
    return ns


def find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise KeyError(name)


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def torch_save_dicts(tree: ast.Module):
    """Yield (lineno, literal_key_set) for every torch.save({...}) with a state dict."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "save"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "torch"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Dict):
            continue
        keys = {k.value for k in node.args[0].keys if isinstance(k, ast.Constant)}
        if "model_state_dict" in keys:
            out.append((node.lineno, keys))
    return out


# ── numpy-backed torch shim (only what ChunkedDataset.__getitem__ touches) ──

class ShimTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def flip(self, dim):
        return ShimTensor(np.flip(self.array, axis=dim))

    def item(self):
        return self.array.item()

    def tolist(self):
        return self.array.tolist()

    def __getitem__(self, key):
        return ShimTensor(self.array[key])

    def __setitem__(self, key, value):
        self.array[key] = value

    def __array__(self, dtype=None, copy=None):
        return self.array if dtype is None else self.array.astype(dtype)

    def __index__(self):
        return int(self.array.item())


def make_torch_shim():
    return types.SimpleNamespace(
        long=np.int64,
        float32=np.float32,
        bool=np.bool_,
        tensor=lambda x, dtype=None: ShimTensor(np.asarray(x, dtype=dtype)),
        as_tensor=lambda x, dtype=None: ShimTensor(np.asarray(x, dtype=dtype)),
        zeros=lambda n, dtype=None: ShimTensor(np.zeros(n, dtype=dtype)),
    )


# ── real-ish self-play sample data ──

def initial_board_encoding() -> list[float]:
    """Standard opening position, layout [square * 14 + plane], plane = side*7 + type."""
    flat = [0.0] * (BOARD_SIZE * BOARD_PLANES)
    back_rank = [4, 3, 2, 1, 0, 1, 2, 3, 4]  # Rook Horse Elephant Advisor General ...

    def put(row, col, side, piece_type):
        flat[(row * BOARD_COLS + col) * BOARD_PLANES + side * 7 + piece_type] = 1.0

    for col, piece_type in enumerate(back_rank):
        put(0, col, 1, piece_type)
        put(9, col, 0, piece_type)
    for col in (1, 7):
        put(2, col, 1, 5)
        put(7, col, 0, 5)
    for col in (0, 2, 4, 6, 8):
        put(3, col, 1, 6)
        put(6, col, 0, 6)
    return flat


def opening_sample(side_to_move: int = 1, result: float = 1.0) -> dict:
    # Hand-derived legal red opening moves: (from_row,from_col)->(to_row,to_col)
    pairs = [((9, 1), (7, 2)), ((9, 1), (7, 0)), ((9, 7), (7, 6)),
             ((9, 7), (7, 8)), ((7, 1), (7, 4)), ((6, 4), (5, 4)), ((9, 0), (8, 0))]
    legal = [(fr * BOARD_COLS + fc) * BOARD_SIZE + (tr * BOARD_COLS + tc)
             for (fr, fc), (tr, tc) in pairs]
    return {
        "BoardEncoding": initial_board_encoding(),
        "SideToMove": side_to_move,
        "LegalMoves": legal,
        "SelectedMove": legal[0],
        "Result": result,
        "ValueWeight": 1.0,
        "PolicyWeight": 1.0,
    }


def write_jsonl(path: Path, samples) -> Path:
    path.write_text("".join(json.dumps(s) + "\n" for s in samples), encoding="utf-8")
    return path