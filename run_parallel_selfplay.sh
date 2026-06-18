#!/bin/bash
# Parallel self-play: launch 5 processes, each generating 400 games
# Total: 2000 games, then merge

set -e

SELFPLAY_DIR="/e/Projects/WorkBuddy/Chess/ChineseChess-WPF/ChineseChess/ChineseChess.SelfPlay/bin/Release/net8.0"
DATA_DIR="/e/Projects/WorkBuddy/Chess/ChineseChess-WPF/data/selfplay"
MERGE_DIR="/e/Projects/WorkBuddy/Chess/ChineseChess-WPF/data/selfplay_parallel"

echo "========================================"
echo "  并行自对弈启动 — 5 进程 × 400 局"
echo "========================================"
echo ""
echo "CPU: $(nproc 2>/dev/null || echo unknown) 核心"
echo "内存: $(free -h 2>/dev/null | grep Mem | awk '{print $7}' || echo unknown) 可用"
echo ""

# Clean old data
rm -f "$DATA_DIR"/parallel_*.jsonl "$DATA_DIR"/parallel_*.tmp 2>/dev/null
mkdir -p "$MERGE_DIR"

PIDS=()
SEEDS=(12345 67890 11111 22222 33333)
GAMES_PER_PROC=400

for i in 0 1 2 3 4; do
    SEED=${SEEDS[$i]}
    OUTFILE="$DATA_DIR/parallel_$i.jsonl"
    LOGFILE="$DATA_DIR/parallel_$i.log"
    
    echo "[进程 $i] 启动: seed=$SEED, 输出=$OUTFILE, 局数=$GAMES_PER_PROC"
    
    dotnet "$SELFPLAY_DIR/ChineseChess.SelfPlay.dll" \
        --games $GAMES_PER_PROC \
        --out "$OUTFILE" \
        --level 4 \
        --timeMs 300 \
        --maxMoves 220 \
        --topK 2 \
        --nearBestWindow 80 \
        --randomOpeningPlies 12 \
        --openingTopK 6 \
        --openingNearBestWindow 180 \
        --seed $SEED \
        --skipDraws false \
        --skipUnfinished true \
        --atomicOutput true \
        > "$LOGFILE" 2>&1 &
    
    PIDS+=($!)
    echo "  PID: $!"
done

echo ""
echo "所有进程已启动，等待完成..."

# Wait for all processes
FAILED=0
for i in 0 1 2 3 4; do
    PID=${PIDS[$i]}
    wait $PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[进程 $i] PID=$PID 失败 (exit=$EXIT_CODE)"
        FAILED=$((FAILED + 1))
    else
        echo "[进程 $i] PID=$PID 完成 ✅"
    fi
done

echo ""
echo "========================================"
echo "  全部进程完成！失败: $FAILED"
echo "========================================"
echo ""

# Check output files
echo "各进程输出行数:"
TOTAL_ROWS=0
for i in 0 1 2 3 4; do
    OUTFILE="$DATA_DIR/parallel_$i.jsonl"
    if [ -f "$OUTFILE" ]; then
        ROWS=$(wc -l < "$OUTFILE")
        SIZE=$(ls -lh "$OUTFILE" | awk '{print $5}')
        echo "  进程 $i: ${ROWS} 行, ${SIZE}"
        TOTAL_ROWS=$((TOTAL_ROWS + ROWS))
    else
        echo "  进程 $i: 文件未找到 ❌"
    fi
done

echo ""
echo "总行数: $TOTAL_ROWS"
echo ""

# Merge all data
if [ $FAILED -eq 0 ] && [ $TOTAL_ROWS -gt 0 ]; then
    MERGED_FILE="$MERGE_DIR/train_parallel_2000.jsonl"
    > "$MERGED_FILE"
    for i in 0 1 2 3 4; do
        cat "$DATA_DIR/parallel_$i.jsonl" >> "$MERGED_FILE"
    done
    
    MERGED_ROWS=$(wc -l < "$MERGED_FILE")
    MERGED_SIZE=$(ls -lh "$MERGED_FILE" | awk '{print $5}')
    echo "已合并: $MERGED_FILE (${MERGED_ROWS} 行, ${MERGED_SIZE})"
    echo ""
    echo "如需与之前数据合并训练:"
    echo "  cat data/selfplay/train_combined.jsonl $MERGED_FILE > data/selfplay/train_all.jsonl"
else
    echo "合并跳过 — 存在失败进程或无数据"
fi

echo ""
echo "日志文件: $DATA_DIR/parallel_*.log"
