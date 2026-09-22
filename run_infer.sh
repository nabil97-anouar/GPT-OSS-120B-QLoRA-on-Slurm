#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/00_env.sh"

exec python -u "${PROJECT_DIR}/infer_lora_local.py" "$@"
