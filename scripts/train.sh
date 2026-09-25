#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-26880}"
launch_args=(--num_processes "$NUM_PROCESSES" --num_machines 1 --mixed_precision no --main_process_port "$MAIN_PROCESS_PORT")
if (( NUM_PROCESSES > 1 )); then
  launch_args+=(--multi_gpu)
fi

exec "$PYTHON_BIN" -m accelerate.commands.launch "${launch_args[@]}" "$PROJECT_ROOT/train.py" \
  --split train --views 1 --frames 1 --img_size 256 256 --num_types 2 \
  --batch_size 32 --lr 1e-4 --warmup_steps 50000 --num_steps 200000 \
  --log_every 100 --eval_every 2000 --eval_batches 0 --save_every 2000 \
  --vision_backbone dinov2 --ckpt "$PROJECT_ROOT/outputs/" \
  --repa --repa-teacher dinov2_vitb14_reg --repa-depth 8 \
  --repa-coeff 0.5 --repa-stop-steps 80000 "$@"
