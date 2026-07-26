#!/bin/bash
# Dulya-fit unmanipulated lineshape array — one spectral bin per task.
#
# Submit from repo root:
#   sbatch Data_Creation/dulya_fit/unmanipulated_bin_array.slurm

#SBATCH --job-name=dulya-unmanip-bin
#SBATCH --array=0-499
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=Data_Creation/dulya_fit/slurm_logs/unmanip_bin_%A_%a.out
#SBATCH --error=Data_Creation/dulya_fit/slurm_logs/unmanip_bin_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-Data_Creation/dulya_fit/data/unmanip_train}"
NUM_BINS="${NUM_BINS:-500}"
P_MIN="${P_MIN:--0.9}"
P_MAX="${P_MAX:-0.9}"
P_STEP="${P_STEP:-0.05}"

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
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is unset; submit as an array job" >&2
  exit 1
fi

echo "Dulya unmanipulated bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  Data_Creation/dulya_fit/unmanipulated_bin_lineshape.py \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  --p-min "${P_MIN}" \
  --p-max "${P_MAX}" \
  --p-step "${P_STEP}" \
  --skip-if-exists \
  "$@"
