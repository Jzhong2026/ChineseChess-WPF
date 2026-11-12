"""
Monitor training completion, then export ONNX automatically.
Usage: python auto_export_onnx.py [--log training_v7_cont2.log] [--timeout 21600] [--pid 1234]
"""
import argparse
import time
import os
import subprocess
import sys

LOG_FILE = "training_v7_cont2.log"
EXPORT_SCRIPT = "export_cnn_onnx.py"
CHECKPOINT_SRC = "artifacts/cnn_policy_value_v7_cont2.best.pt"
CHECKPOINT_PREFIX = "cnn_policy_value_v7_cont2"
ARTIFACTS_DIR = "artifacts"
ONNX_OUT = "artifacts/cnn_policy_value.onnx"

# Must match what the trainers actually print on their final save.
COMPLETION_MARKERS = ("Saved final:", "Saved final checkpoint:", "Final model:")

DEFAULT_TIMEOUT_S = 6 * 3600
DEFAULT_POLL_S = 120


def is_completed(content):
    return any(marker in content for marker in COMPLETION_MARKERS)


def _process_alive(pid):
    if pid is None:
        return True
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_completion(log_path, timeout_s=DEFAULT_TIMEOUT_S, poll_s=DEFAULT_POLL_S, pid=None):
    """Poll the log until a trainer completion marker appears, the trainer dies, or we time out."""
    deadline = time.monotonic() + timeout_s
    last_size = -1
    while True:
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            if is_completed(content):
                print(f"[{time.strftime('%H:%M:%S')}] Training completed!")
                return True

            if len(content) != last_size:
                last_size = len(content)
                tail = content.strip().split('\n')[-1] if content.strip() else ""
                print(f"[{time.strftime('%H:%M:%S')}] Still running... last: {tail[:80]}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Log file not found, waiting: {log_path}")

        if not _process_alive(pid):
            print(f"[{time.strftime('%H:%M:%S')}] Trainer pid={pid} is gone without a completion marker.")
            return False

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"[{time.strftime('%H:%M:%S')}] Timed out after {timeout_s}s waiting for completion.")
            return False
        time.sleep(min(poll_s, remaining))


def select_checkpoint(artifacts_dir, prefix, best_path):
    """best.pt wins (lowest val_loss); otherwise the highest periodic epoch actually on disk."""
    if os.path.exists(best_path):
        return best_path
    latest_epoch = -1
    latest_ckpt = None
    if os.path.isdir(artifacts_dir):
        for name in os.listdir(artifacts_dir):
            if not (name.startswith(prefix + ".epoch") and name.endswith(".pt")):
                continue
            try:
                epoch_num = int(name[len(prefix) + len(".epoch"):-len(".pt")])
            except ValueError:
                continue
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_ckpt = os.path.join(artifacts_dir, name)
    return latest_ckpt


def run_export():
    """Run the ONNX export script."""
    print(f"[{time.strftime('%H:%M:%S')}] Starting ONNX export...")

    checkpoint = select_checkpoint(ARTIFACTS_DIR, CHECKPOINT_PREFIX, CHECKPOINT_SRC)
    if checkpoint is None:
        print(f"  ERROR: No checkpoint found in {ARTIFACTS_DIR} for prefix {CHECKPOINT_PREFIX}!")
        return False

    reason = "lowest val_loss" if checkpoint == CHECKPOINT_SRC else "latest periodic epoch on disk"
    print(f"  Using checkpoint: {checkpoint} ({reason})")

    cmd = [
        sys.executable, EXPORT_SCRIPT,
        "--input", checkpoint,
        "--output", ONNX_OUT,
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  stdout: {result.stdout}")
    if result.stderr:
        print(f"  stderr: {result.stderr}")
    print(f"  return code: {result.returncode}")

    if result.returncode == 0 and os.path.exists(ONNX_OUT):
        size_mb = os.path.getsize(ONNX_OUT) / 1024 / 1024
        print(f"  ✅ ONNX exported successfully: {ONNX_OUT} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ❌ Export failed!")
        return False


def build_parser():
    p = argparse.ArgumentParser(description="Wait for training to finish, then export ONNX")
    p.add_argument("--log", default=LOG_FILE)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="Seconds to wait for completion")
    p.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_S)
    p.add_argument("--pid", type=int, default=None, help="Trainer pid; abort if it dies before completing")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"[{time.strftime('%H:%M:%S')}] Auto-export monitor started")
    print(f"  Working dir: {os.getcwd()}")

    if wait_for_completion(args.log, timeout_s=args.timeout, poll_s=args.poll_interval, pid=args.pid):
        success = run_export()
        if success:
            print(f"\n[{time.strftime('%H:%M:%S')}] All done!")
            # Notify via a marker file
            with open(".export_done", 'w') as f:
                f.write(f"ONNX export completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            sys.exit(0)
        else:
            print(f"\n[{time.strftime('%H:%M:%S')}] Export failed, check logs.")
            sys.exit(1)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Monitor exited without completion.")
        sys.exit(2)
