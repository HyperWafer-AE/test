#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python cfg.py
mkdir -p results
CFG="./cfg/config_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0.cfg"
rm -f ./results/summary.csv

for policy in central full_replication krd_selective; do
  python main.py \
    --cfg "$CFG" \
    --policy "$policy" \
    --num-workflows 2 \
    --agents-per-workflow 3 \
    --repo-blocks 8 \
    --issue-blocks 2 \
    --private-blocks 4 \
    --block-elems 65536 \
    --region-size 8 \
    --krd-mode workflow \
    --dijkstra-routing \
    --out "./results/${policy}.json"
done

python - <<'PY'
import glob
import json

for path in sorted(glob.glob("results/*.json")):
    with open(path, encoding="utf-8") as handle:
        metrics = json.load(handle)
    print(
        path,
        metrics["policy"],
        "cycles=", metrics["total_cycles"],
        "kv_hop_bytes=", metrics["kv_hop_bytes"],
        "replica_bytes=", metrics["replica_bytes"],
    )
PY

