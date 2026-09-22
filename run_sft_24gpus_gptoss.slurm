#!/usr/bin/env bash
#SBATCH --job-name=gptoss120b-qlora
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:8
#SBATCH --gres-flags=enforce-binding
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

# Submit this file from the repository root. Supply partition and QoS as sbatch
# flags because those names are cluster-specific.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/00_env.sh"

[[ "${SLURM_NNODES}" -eq 3 ]] || {
  echo "Expected 3 nodes, received ${SLURM_NNODES}" >&2
  exit 2
}

srun \
  --mpi=none \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  "${PROJECT_DIR}/launch_torchrun.sh"
