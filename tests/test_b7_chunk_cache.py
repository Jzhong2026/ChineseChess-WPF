"""B7: chunk cache must be keyed by an input fingerprint, not by meta.pt existence."""
from __future__ import annotations

import ast
import os
import time

import pytest

from tests._helpers import extract_symbols, find_function, tree_of

COMBINED = "train_v7_combined.py"


@pytest.fixture(scope="module")
def cache_fns():
    ns = extract_symbols(COMBINED, ["_input_fingerprint", "_cache_matches"])
    return ns["_input_fingerprint"], ns["_cache_matches"]


def test_fingerprint_tracks_path_size_and_mtime(cache_fns, tmp_path):
    fingerprint, _ = cache_fns
    data = tmp_path / "a.jsonl"
    data.write_bytes(b"one\n")
    fp1 = fingerprint(str(data))
    assert fp1["path"] == str(data.resolve())
    assert fp1["size"] == 4
    assert "mtime" in fp1
    assert fingerprint(str(data)) == fp1


def test_fingerprint_changes_when_input_changes(cache_fns, tmp_path):
    fingerprint, matches = cache_fns
    data = tmp_path / "a.jsonl"
    data.write_text("one\n", encoding="utf-8")
    fp1 = fingerprint(str(data))
    time.sleep(0.01)
    data.write_text("one\ntwo\n", encoding="utf-8")
    os.utime(data, (time.time() + 5, time.time() + 5))
    fp2 = fingerprint(str(data))
    assert fp2 != fp1
    assert matches({"input_fingerprint": fp1}, fp2) is False
    assert matches({"input_fingerprint": fp2}, fp2) is True


def test_different_input_path_invalidates_cache(cache_fns, tmp_path):
    fingerprint, matches = cache_fns
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    for p in (a, b):
        p.write_text("same\n", encoding="utf-8")
    os.utime(b, (os.stat(a).st_atime, os.stat(a).st_mtime))
    assert matches({"input_fingerprint": fingerprint(str(a))}, fingerprint(str(b))) is False


def test_legacy_meta_without_fingerprint_is_rejected(cache_fns, tmp_path):
    fingerprint, matches = cache_fns
    data = tmp_path / "a.jsonl"
    data.write_text("x\n", encoding="utf-8")
    legacy_meta = {"num_chunks": 3, "total_samples": 12000, "chunk_size": 5000}
    assert matches(legacy_meta, fingerprint(str(data))) is False


def test_preprocess_stores_fingerprint_in_meta():
    fn = find_function(tree_of(COMBINED), "preprocess_jsonl")
    stored = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "meta" for t in node.targets):
            if isinstance(node.value, ast.Dict):
                stored.append({k.value for k in node.value.keys if isinstance(k, ast.Constant)})
    assert stored, "preprocess_jsonl no longer builds a meta dict"
    assert any("input_fingerprint" in keys for keys in stored)


def test_train_reuses_cache_only_after_fingerprint_check():
    fn = find_function(tree_of(COMBINED), "train")
    calls = [n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "_cache_matches" in calls, "train() does not validate the cached chunks against --input"
    assert "_input_fingerprint" in calls