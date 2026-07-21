#!/bin/bash

#SBATCH --job-name=ssrf-traj-organize
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=physics/lineshape/rate_eqs_test/slurm_logs/ssrf_traj_organize_%j.out
#SBATCH --error=physics/lineshape/rate_eqs_test/slurm_logs/ssrf_traj_organize_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_DIR="${SHARD_DIR:-physics/lineshape/rate_eqs_test/ssrf_shards}"
OUTPUT_DIR="${OUTPUT_DIR:-physics/lineshape/rate_eqs_test/ssrf_train}"
NUM_BINS="${NUM_BINS:-500}"

cd "${REPO_ROOT}"

mkdir -p physics/lineshape/rate_eqs_test/slurm_logs
mkdir -p "${OUTPUT_DIR}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
export PYTHONUNBUFFERED=1

module purge
module load apptainer pytorch/2.9.0

if [[ -z "${CONTAINERDIR:-}" ]]; then
  echo "CONTAINERDIR is unset after loading pytorch module" >&2
  exit 1
fi

n_shards="$(find "${SHARD_DIR}" -maxdepth 1 -name 'ssrf_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
echo "Organizing ssRF traj shards on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}"
echo "Shard dir: ${SHARD_DIR} (${n_shards} shards found)"
echo "Output dir: ${OUTPUT_DIR} (${NUM_BINS} per-bin training files)"
echo "Container: ${CONTAINERDIR}/pytorch-2.9.0.sif"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  physics/lineshape/rate_eqs_test/ssrf_bin_traj.py \
  --organize \
  --strict \
  --shard-dir "${SHARD_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  "$@"
