#!/bin/bash
set -euo pipefail

cp /solution/solve.py /app/train.py

for SPANS in 1 4 8; do
    python3 /app/train.py --span_count "$SPANS" --out "/app/logs/span_${SPANS}.json"
done