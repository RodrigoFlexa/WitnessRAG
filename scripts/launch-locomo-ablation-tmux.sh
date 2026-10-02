#!/usr/bin/env bash
# Use from the frozen checkout, with the persistent OUTPUT_DIR and CACHE_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUTPUT_DIR:?Set persistent OUTPUT_DIR}"
: "${CACHE_DIR:?Set persistent CACHE_DIR}"
: "${BENCH_PYTHON:?Set BENCH_PYTHON}"
SESSION=${SESSION:-locomo-ablation}
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists; inspect it rather than launching twice."
  exit 1
fi
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
"$BENCH_PYTHON" scripts/check-qwen-server.py --base-url http://127.0.0.1:8096/v1 \
  --model Qwen/Qwen2.5-14B-Instruct --check-cuda --output "$OUTPUT_DIR/server.json"
printf -v panel '%q ' "$BENCH_PYTHON" -u scripts/watch-locomo-ablation.py --output "$OUTPUT_DIR" --watch
tmux new-session -d -s "$SESSION" -n scoreboard -c "$PWD" "$panel"
printf -v queue '%q ' nohup "$BENCH_PYTHON" -u scripts/run-locomo-component-ablation.py \
  --output "$OUTPUT_DIR" --cache "$CACHE_DIR" --gpu 0 --port 8096 --concurrency 2
printf -v logfile '%q' "$OUTPUT_DIR/queue.log"
tmux new-window -d -t "$SESSION" -n executor -c "$PWD" "exec $queue >> $logfile 2>&1 < /dev/null"
printf -v detailed '%q ' tail -n 20 -F "$OUTPUT_DIR/full/benchmark.log"
tmux new-window -d -t "$SESSION" -n details -c "$PWD" "$detailed"
printf -v serving '%q ' tail -n 20 -F "$OUTPUT_DIR/vllm.log"
tmux new-window -d -t "$SESSION" -n server -c "$PWD" "$serving"
tmux select-window -t "$SESSION:scoreboard"
echo "Study launched with nohup. Attach: tmux attach -t $SESSION"
