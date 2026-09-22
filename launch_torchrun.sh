#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${SLURM_NNODES:-${NNODES:-1}}"
NODE_RANK="${SLURM_NODEID:-${NODE_RANK:-0}}"

if [[ -n "${SLURM_NODELIST:-}" ]]; then
  MASTER_HOST="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"
else
  MASTER_HOST="${MASTER_ADDR:-127.0.0.1}"
fi

MASTER_ADDR="${MASTER_ADDR:-$(getent ahostsv4 "${MASTER_HOST}" 2>/dev/null | awk 'NR==1 {print $1; exit}')}"
MASTER_ADDR="${MASTER_ADDR:-${MASTER_HOST}}"
MASTER_PORT="${MASTER_PORT:-43001}"
RDZV_ID="${RDZV_ID:-${SLURM_JOB_ID:-local}}"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "[distributed] nodes=${NNODES} GPUs/node=${GPUS_PER_NODE} world_size=${WORLD_SIZE}"
echo "[distributed] node_rank=${NODE_RANK} rendezvous=${MASTER_ADDR}:${MASTER_PORT}"

exec torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${GPUS_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv-id="${RDZV_ID}" \
  train_lora_gptoss_unsloth.py
