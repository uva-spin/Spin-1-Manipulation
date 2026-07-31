#!/bin/bash
# Flatten one ssRF spectrum shard per array task into row-oriented NPZs.
#
# Run after ssrf_spectrum_array.slurm (or ssrf_traj_array.slurm --spectrum-mode):
#   SSRF_ID=$(sbatch --parsable ssrf_spectrum_array.slurm)
#   sbatch --dependency=afterok:${SSRF_ID} flatten_ssrf_spectrum_rows.slurm
#
# Submit from THIS directory (dulya_fit_v2/):
#   sbatch flatten_ssrf_spectrum_rows.slurm

#SBATCH --job-name=dulya-v2-flatten-ssrf-rows
#SBATCH --array=125-374
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/flatten_ssrf_rows_%A_%a.out
#SBATCH --error=slurm_logs/flatten_ssrf_rows_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

PKG_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SHARD_DIR="${SHARD_DIR:-data/ssrf_shards}"
OUTPUT_DIR="${OUTPUT_DIR:-data/ssrf_spectrum_rows}"
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
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is unset; submit as an array job" >&2
  exit 1
fi

echo "Flatten ssRF spectrum rows bin ${SLURM_ARRAY_TASK_ID} shard=${SHARD_DIR} -> ${OUTPUT_DIR}"

apptainer run --bind "${PKG_DIR}:${PKG_DIR}" --pwd "${PKG_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  flatten_spectrum_rows.py \
  --source ssrf \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --shard-dir "${SHARD_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  --skip-if-exists \
  "$@"
