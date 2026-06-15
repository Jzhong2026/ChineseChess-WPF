#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Chinese Chess ML Training Pipeline
# 
# Complete pipeline from Self-Play data generation to ONNX model export.
# Run this script from the repository root:
#   bash run_training_pipeline.sh
#
# Prerequisites:
#   - .NET 8 SDK
#   - Python 3.x with torch, onnx installed
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configuration ─────────────────────────────────────────────────────────────
GAMES="${GAMES:-500}"
LEVEL="${LEVEL:-3}"
TIME_MS="${TIME_MS:-200}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CNN_CHANNELS="${CNN_CHANNELS:-128}"
CNN_BLOCKS="${CNN_BLOCKS:-8}"
DATA_DIR="data/selfplay"
ARTIFACTS_DIR="artifacts"

echo "═══════════════════════════════════════════════════"
echo "  Chinese Chess ML Training Pipeline"
echo "  Games: $GAMES | Level: $LEVEL | Epochs: $EPOCHS"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Build Self-Play generator ─────────────────────────────────────────
echo ""
echo "▶ Step 1: Building ChineseChess.SelfPlay..."
dotnet build ChineseChess/ChineseChess.SelfPlay/ChineseChess.SelfPlay.csproj \
    --configuration Release --nologo -q

# ── Step 2: Generate Self-Play data ──────────────────────────────────────────
echo ""
echo "▶ Step 2: Generating $GAMES self-play games (level=$LEVEL, timeMs=$TIME_MS)..."
mkdir -p "$DATA_DIR"

dotnet run --project ChineseChess/ChineseChess.SelfPlay/ChineseChess.SelfPlay.csproj \
    --configuration Release -- \
    --games "$GAMES" \
    --out "$DATA_DIR/train.jsonl" \
    --level "$LEVEL" \
    --timeMs "$TIME_MS" \
    --maxMoves 220 \
    --topK 2 \
    --nearBestWindow 80 \
    --randomOpeningPlies 12 \
    --openingTopK 6 \
    --openingNearBestWindow 180 \
    --adjudicateAfterMoves 120 \
    --adjudicateNoCapturePlies 60 \
    --adjudicateMaterialMargin 450 \
    --adjudicateAtMaxMoves true \
    --skipUnfinished true

echo "  Data saved to: $DATA_DIR/train.jsonl"
LINE_COUNT=$(wc -l < "$DATA_DIR/train.jsonl")
echo "  Total samples: $LINE_COUNT"

# ── Step 3: Train CNN policy-value network ────────────────────────────────────
echo ""
echo "▶ Step 3: Training CNN (channels=$CNN_CHANNELS, blocks=$CNN_BLOCKS, epochs=$EPOCHS)..."
mkdir -p "$ARTIFACTS_DIR"

python train_cnn_policy_value.py \
    --input "$DATA_DIR/train.jsonl" \
    --output "$ARTIFACTS_DIR/cnn_policy_value.pt" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --channels "$CNN_CHANNELS" \
    --res-blocks "$CNN_BLOCKS" \
    --val-ratio 0.1 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --value-loss-weight 1.0

echo "  Checkpoint saved to: $ARTIFACTS_DIR/cnn_policy_value.pt"

# ── Step 4: Export to ONNX ────────────────────────────────────────────────────
echo ""
echo "▶ Step 4: Exporting best checkpoint to ONNX..."

BEST_PT="$ARTIFACTS_DIR/cnn_policy_value.best.pt"
FINAL_PT="$ARTIFACTS_DIR/cnn_policy_value.pt"
INPUT_PT=$( [ -f "$BEST_PT" ] && echo "$BEST_PT" || echo "$FINAL_PT" )

python export_cnn_onnx.py \
    --input "$INPUT_PT" \
    --output "$ARTIFACTS_DIR/cnn_policy_value.onnx"

echo "  ONNX model saved to: $ARTIFACTS_DIR/cnn_policy_value.onnx"

# ── Step 5: Verification ──────────────────────────────────────────────────────
echo ""
echo "▶ Step 5: Running encoder verification..."
dotnet run --project ChineseChess/ChineseChess.Core.Verification/ChineseChess.Core.Verification.csproj \
    --configuration Release

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo "  ONNX model: $ARTIFACTS_DIR/cnn_policy_value.onnx"
echo ""
echo "  Next step: copy the .onnx file to the WPF app's"
echo "  output directory and enable Neural AI mode."
echo "═══════════════════════════════════════════════════"
