#!/bin/bash
# Organize Dulya-fit v2 AFP shards into one training NPZ per spectral bin.
#
# Submit from THIS directory (dulya_fit_v2/):
#   sbatch afp_traj_combine.slurm

#SBATCH --job-name=dulya-v2-afp-organize
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/afp_traj_organize_%j.out
#SBATCH --error=slurm_logs/afp_traj_organize_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

PKG_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SHARD_DIR="${SHARD_DIR:-data/afp_shards}"
OUTPUT_DIR="${OUTPUT_DIR:-data/afp_train}"
NUM_BINS="${NUM_BINS:-500}"

cd "${PKG_DIR}"

mkdir -p slurm_logs
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

n_shards="$(find "${SHARD_DIR}" -maxdepth 1 -name 'afp_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
echo "Organizing Dulya v2 AFP shards (${n_shards} found) -> ${OUTPUT_DIR}"

apptainer run --bind "${PKG_DIR}:${PKG_DIR}" --pwd "${PKG_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  afp_bin_traj.py \
  --organize \
  --strict \
  --shard-dir "${SHARD_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  "$@"
