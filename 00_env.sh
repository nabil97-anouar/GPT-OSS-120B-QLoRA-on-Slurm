#!/usr/bin/env bash
# Source this file before running the training or inference scripts.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR

VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
if [[ -z "${VIRTUAL_ENV:-}" && -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PROJECT_DIR}/.cache/triton}"
export MODEL_ID="${MODEL_ID:-unsloth/gpt-oss-120b-unsloth-bnb-4bit}"
export DATASET="${DATASET:-${PROJECT_DIR}/toy.jsonl}"
export OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/runs/gptoss120b-lora}"
export MERGED_OUT="${MERGED_OUT:-${PROJECT_DIR}/runs/gptoss120b-merged}"

mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${OUT_DIR}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"

echo "Environment ready"
echo "  project: ${PROJECT_DIR}"
echo "  model:   ${MODEL_ID}"
echo "  data:    ${DATASET}"
echo "  output:  ${OUT_DIR}"
