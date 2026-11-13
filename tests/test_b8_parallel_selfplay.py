"""B8: run_parallel_selfplay.sh must derive paths from itself and reach the failure counter."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._helpers import REPO_ROOT, source_of

SCRIPT = "run_parallel_selfplay.sh"


def _bash_executable():
    # Windows' bare bash command can resolve to WSL, which cannot consume a
    # Windows argv/path or PATH containing our native test stubs. Prefer Git Bash.
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            candidate = Path(git).resolve().parent.parent / "bin" / "bash.exe"
            if candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("bash")


BASH = _bash_executable()

DOTNET_STUB = """#!/bin/bash
out=""
seed=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out="$2"; shift 2;;
    --seed) seed="$2"; shift 2;;
    *) shift;;
  esac
done
printf 'row1\\nrow2\\nrow3\\n' > "$out"
case "$seed" in
  11111|33333) exit 3;;
esac
exit 0
"""

DOTNET_STUB_OK = DOTNET_STUB.replace("  11111|33333) exit 3;;\n", "")


def _make_tree(tmp_path, stub):
    # A Windows checkout may use CRLF; Bash fixtures must use LF.
    (tmp_path / SCRIPT).write_text(source_of(SCRIPT), encoding="utf-8", newline="\n")
    bindir = tmp_path / "ChineseChess" / "ChineseChess.SelfPlay" / "bin" / "Release" / "net8.0"
    bindir.mkdir(parents=True)
    (bindir / "ChineseChess.SelfPlay.dll").write_bytes(b"")
    (tmp_path / "data" / "selfplay").mkdir(parents=True)
    stubdir = tmp_path / "stubbin"
    stubdir.mkdir()
    dotnet = stubdir / "dotnet"
    dotnet.write_text(stub, encoding="utf-8", newline="\n")
    dotnet.chmod(0o755)
    return tmp_path


def _run(tree):
    env = dict(os.environ)
    env["PATH"] = str(tree / "stubbin") + os.pathsep + env["PATH"]
    return subprocess.run([BASH, (tree / SCRIPT).as_posix()], cwd=str(tree), env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=180)


def test_no_hardcoded_foreign_drive_paths():
    src = source_of(SCRIPT)
    assert "/e/Projects" not in src
    assert "WorkBuddy" not in src


def test_paths_derived_from_script_location():
    src = source_of(SCRIPT)
    assert "BASH_SOURCE" in src


@pytest.mark.skipif(BASH is None, reason="native bash not available")
def test_script_parses():
    proc = subprocess.run([BASH, "-n"], input=source_of(SCRIPT),
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(BASH is None, reason="native bash not available")
def test_failure_counting_and_merge_skip_are_reachable(tmp_path):
    tree = _make_tree(tmp_path, DOTNET_STUB)
    proc = _run(tree)
    assert "失败: 2" in proc.stdout, f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert "合并跳过" in proc.stdout
    assert not (tree / "data" / "selfplay_parallel" / "train_parallel_2000.jsonl").exists()


@pytest.mark.skipif(BASH is None, reason="native bash not available")
def test_all_success_merges_data(tmp_path):
    tree = _make_tree(tmp_path, DOTNET_STUB_OK)
    proc = _run(tree)
    assert "失败: 0" in proc.stdout, f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    merged = tree / "data" / "selfplay_parallel" / "train_parallel_2000.jsonl"
    assert merged.is_file(), proc.stdout + proc.stderr
    assert len(merged.read_text(encoding="utf-8").splitlines()) == 15
    assert "总行数: 15" in proc.stdout
