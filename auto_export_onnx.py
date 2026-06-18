"""
Monitor training completion, then export ONNX automatically.
Usage: python auto_export_onnx.py
"""
import time
import os
import subprocess
import sys

LOG_FILE = "training_v7_cont2.log"
EXPORT_SCRIPT = "export_cnn_onnx.py"
CHECKPOINT_SRC = "artifacts/cnn_policy_value_v7_cont2.best.pt"
CHECKPOINT_FALLBACK = "artifacts/cnn_policy_value_v7_cont2.epoch40.pt"
ONNX_OUT = "artifacts/cnn_policy_value.onnx"

def wait_for_completion(log_path):
    """Poll the log file until training finishes."""
    last_pos = 0
    while True:
        if not os.path.exists(log_path):
            print(f"[{time.strftime('%H:%M:%S')}] Log file not found, waiting...")
            time.sleep(60)
            continue

        with open(log_path, 'r') as f:
            content = f.read()

        # Check for completion markers
        if "Training completed" in content or "Saved final model" in content or "Export ONNX" in content:
            print(f"[{time.strftime('%H:%M:%S')}] Training completed!")
            return True

        # Also check for "epoch=050" as fallback
        if "epoch=050" in content:
            # Check if epoch 50 is done (next line should appear)
            lines = content.strip().split('\n')
            if len(lines) > 3:
                last_epoch_line = [l for l in lines if "epoch=050" in l]
                if last_epoch_line:
                    # Wait a bit more to ensure the final save
                    time.sleep(120)
                    print(f"[{time.strftime('%H:%M:%S')}] Epoch 50 confirmed, proceeding...")
                    return True

        # Check if process is still alive
        if len(content) > 10:
            tail = content.strip().split('\n')[-1] if content.strip() else ""
            if len(content) != last_pos:
                last_pos = len(content)
                print(f"[{time.strftime('%H:%M:%S')}] Still running... last: {tail[:80]}")

        time.sleep(120)  # Check every 2 minutes


def run_export():
    """Run the ONNX export script."""
    print(f"[{time.strftime('%H:%M:%S')}] Starting ONNX export...")

    # Determine which checkpoint to use
    checkpoint = CHECKPOINT_SRC
    if not os.path.exists(checkpoint):
        print(f"  Best checkpoint not found, using epoch40 instead.")
        checkpoint = CHECKPOINT_FALLBACK
        if not os.path.exists(checkpoint):
            print(f"  ERROR: No checkpoint found!")
            return False

    print(f"  Using checkpoint: {checkpoint}")

    # Find the latest periodic checkpoint (might be better than best.pt if val hasn't improved)
    latest_epoch = 0
    latest_ckpt = checkpoint
    for f in os.listdir("artifacts"):
        if f.startswith("cnn_policy_value_v7_cont2.epoch") and f.endswith(".pt"):
            try:
                epoch_num = int(f.split(".epoch")[1].split(".")[0])
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_ckpt = os.path.join("artifacts", f)
            except (ValueError, IndexError):
                pass

    if latest_ckpt != checkpoint:
        print(f"  Found newer checkpoint: {latest_ckpt}")
        best_size = os.path.getsize(checkpoint)
        latest_size = os.path.getsize(latest_ckpt)
        print(f"  best.pt: {best_size} bytes, latest: {latest_size} bytes")
        print(f"  Best model (epoch 10) had lowest val_loss, keeping that for export.")

    # Run export (auto-detects V7 from checkpoint metadata)
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


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"[{time.strftime('%H:%M:%S')}] Auto-export monitor started")
    print(f"  Working dir: {os.getcwd()}")

    if wait_for_completion(LOG_FILE):
        success = run_export()
        if success:
            print(f"\n[{time.strftime('%H:%M:%S')}] All done!")
            # Notify via a marker file
            with open(".export_done", 'w') as f:
                f.write(f"ONNX export completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            print(f"\n[{time.strftime('%H:%M:%S')}] Export failed, check logs.")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Monitor exited without completion.")
