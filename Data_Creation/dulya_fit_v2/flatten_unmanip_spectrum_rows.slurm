#!/bin/bash
# Flatten unmanipulated per-bin equilibrium NPZs into one row-oriented NPZ.
#
# Run after unmanipulated_bin_array.slurm:
#   UNMANIP_ID=$(sbatch --parsable unmanipulated_bin_array.slurm)
#   sbatch --dependency=afterok:${UNMANIP_ID} flatten_unmanip_spectrum_rows.slurm
#
# Submit from THIS directory (dulya_fit_v2/):
#   sbatch flatten_unmanip_spectrum_rows.slurm

#SBATCH --job-name=dulya-v2-flatten-unmanip-rows
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/flatten_unmanip_rows_%j.out
#SBATCH --error=slurm_logs/flatten_unmanip_rows_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PKG_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
UNMANIP_DIR="${UNMANIP_DIR:-data/unmanip_train}"
OUTPUT_DIR="${OUTPUT_DIR:-data/unmanip_spectrum_rows}"
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

echo "Flatten unmanipulated rows unmanip=${UNMANIP_DIR} -> ${OUTPUT_DIR}"

apptainer run --bind "${PKG_DIR}:${PKG_DIR}" --pwd "${PKG_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  flatten_spectrum_rows.py \
  --source unmanip \
  --shard-dir "${UNMANIP_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  --skip-if-exists \
  --strict \
  "$@"
