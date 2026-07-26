#!/bin/bash
# Organize Dulya-fit ssRF shards into one training NPZ per spectral bin.

#SBATCH --job-name=dulya-ssrf-organize
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=Data_Creation/dulya_fit/slurm_logs/ssrf_traj_organize_%j.out
#SBATCH --error=Data_Creation/dulya_fit/slurm_logs/ssrf_traj_organize_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_DIR="${SHARD_DIR:-Data_Creation/dulya_fit/data/ssrf_shards}"
OUTPUT_DIR="${OUTPUT_DIR:-Data_Creation/dulya_fit/data/ssrf_train}"
NUM_BINS="${NUM_BINS:-500}"

cd "${REPO_ROOT}"

mkdir -p Data_Creation/dulya_fit/slurm_logs
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
echo "Organizing Dulya ssRF shards (${n_shards} found) -> ${OUTPUT_DIR}"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  Data_Creation/dulya_fit/ssrf_bin_traj.py \
  --organize \
  --strict \
  --shard-dir "${SHARD_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  "$@"
