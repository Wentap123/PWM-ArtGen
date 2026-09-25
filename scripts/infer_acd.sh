#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
exec bash "$SCRIPT_DIR/infer.sh" --dataset acd --out_dir "$PROJECT_ROOT/outputs/acd/predictions" "$@"
