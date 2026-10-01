#!/usr/bin/env bash
# Queue the remaining eight paper tasks after verified completion of FC-SH/FC-MH.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_BIN=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
SF_OUTPUT=${SF_OUTPUT_DIR:-"$PWD/runs/standard-memoryagentbench-sf-qwen14b/facts40"}
OUTPUT=${OUTPUT_DIR:-"$PWD/runs/standard-memoryagentbench-rest-qwen14b/facts40"}
CACHE=${CACHE_DIR:-"$PWD/runs/.cache/memoryagentbench-qwen14b"}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
export CUDA_VISIBLE_DEVICES=${EMBED_GPU:-0} CUDA_DEVICE_ORDER=PCI_BUS_ID
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "Select one embedding GPU" >&2; exit 2; }
export WRAG_EMBED_STRICT_DEVICE=1
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:8095/v1}
export OPENAI_API_KEY=${VLLM_API_KEY:-local-pilot}
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
unset WRAG_REFLECTION_STUDY_ROOT WRAG_CONTROLLED_ROOT WRAG_FROZEN_MEMORY_SOURCE
mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/queue.lock"
flock -n 9 || { echo "Queue already running: $OUTPUT" >&2; exit 2; }
export SF_OUTPUT_DIR="$SF_OUTPUT"
printf 'Waiting for completed selective forgetting: %s\n' "$SF_OUTPUT"
# Do not use a PID as the completion condition: it can exit on a failure or be reused.
# The child probe returns 0 only for a complete, matching 200-question result.
# Active SF launch locks tolerate an interruption followed by a verified resume.
while true; do
  if "$BENCH_BIN" - <<'PY'
import json
import os
from pathlib import Path
import sys

p = Path(os.environ["SF_OUTPUT_DIR"])
summary, manifest = p / "summary.json", p / "manifest.json"
if not summary.exists() or not manifest.exists():
    sys.exit(1)
s = json.loads(summary.read_text())
m = json.loads(manifest.read_text())["identity"]
expected = {"factconsolidation_sh_262k", "factconsolidation_mh_262k"}
if (m["model"] != "Qwen/Qwen2.5-14B-Instruct"
        or m["engine"]["witness"]["fact_budget"] != 40
        or set(s["sources"]) != expected
        or any(v["protocol"] != "paper" for v in m["settings"].values())):
    sys.exit(2)
if not s["generation_complete"] or not s["evaluation_complete"]:
    sys.exit(1)
rows = [json.loads(line) for line in (p / "results.jsonl").read_text().splitlines() if line.strip()]
if (len(rows) != 200 or len({r["qid"] for r in rows}) != 200
        or any(sum(r["source"] == source for r in rows) != 100 for source in expected)):
    sys.exit(2)
PY
  then
    break
  else
    result=$?
    if [[ "$result" != 1 ]]; then
      echo "SF validation failed; remaining tasks were not started." >&2
      exit "$result"
    fi
  fi
  sleep 30
done
echo "Selective forgetting complete. Starting the remaining paper tasks."
"$BENCH_BIN" scripts/check-qwen-server.py --base-url "$OPENAI_BASE_URL" \
  --model "$MODEL" --check-cuda --output "$OUTPUT/server.json"
"$BENCH_BIN" -m benchmarks.memoryagentbench prepare --cache "$CACHE"
"$BENCH_BIN" -u -m benchmarks.memoryagentbench run \
  --backend vllm --base-url "$OPENAI_BASE_URL" --model "$MODEL" \
  --embed-device cuda --embed-model BAAI/bge-m3 --fact-budget 40 \
  --fact-rerank cross-encoder/ms-marco-MiniLM-L6-v2 --conflict-recency-weight 0.65 \
  --protocol paper --suite paper \
  --splits Accurate_Retrieval Test_Time_Learning Long_Range_Understanding \
  --cache "$CACHE" --output "$OUTPUT" --resume
"$BENCH_BIN" -m benchmarks.memoryagentbench report --output "$OUTPUT"
echo "Remaining tasks generated. LongMemEval and summary official judges are pending."
