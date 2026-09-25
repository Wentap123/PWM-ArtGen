#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" "$PROJECT_ROOT/infer.py" \
  --split test --views 1 --frames 1 --img_size 256 256 --num_types 2 \
  --batch_size 1 --sample_steps 10 --num_samples 0 \
  --out_dir "$PROJECT_ROOT/outputs/predictions" --vision_backbone dinov2 "$@"
