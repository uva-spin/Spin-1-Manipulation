#!/bin/bash
# Generate unmanipulated per-bin lineshape NPZs (GenerateVectorLineshape).
#
# Submit from repo root:
#   sbatch physics/lineshape/rate_eqs_test/unmanipulated_bin_lineshape.slurm
#
# Writes unmanip_train/unmanip_bin_XXXX.npz for P in [-0.70, 0.70] step 0.005.
# Then merge into combined ssRF/AFP train files:
#   sbatch physics/lineshape/rate_eqs_test/merge_unmanip_into_combined.slurm

#SBATCH --job-name=unmanip-bins
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=physics/lineshape/rate_eqs_test/slurm_logs/unmanip_bins_%j.out
#SBATCH --error=physics/lineshape/rate_eqs_test/slurm_logs/unmanip_bins_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-physics/lineshape/rate_eqs_test/unmanip_train}"
NUM_BINS="${NUM_BINS:-500}"
P_MIN="${P_MIN:--0.70}"
P_MAX="${P_MAX:-0.70}"
P_STEP="${P_STEP:-0.005}"

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

echo "Generating unmanipulated per-bin lineshapes on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}"
echo "Output: ${OUTPUT_DIR}"
echo "P: [${P_MIN}, ${P_MAX}] step=${P_STEP}  bins=${NUM_BINS}"
echo "Container: ${CONTAINERDIR}/pytorch-2.9.0.sif"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  physics/lineshape/rate_eqs_test/unmanipulated_bin_lineshape.py \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  --p-min "${P_MIN}" \
  --p-max "${P_MAX}" \
  --p-step "${P_STEP}" \
  "$@"
