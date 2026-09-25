#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${RETRIEVAL_PYTHON:-python}" "$SCRIPT_DIR/retrieve.py" "$@"
