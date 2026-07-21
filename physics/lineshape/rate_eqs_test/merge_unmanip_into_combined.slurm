#!/bin/bash
# Merge unmanipulated per-bin NPZs into combined ssRF/AFP train_bin files.
#
# Typical workflow (from repo root):
#   UNMANIP_ID=$(sbatch --parsable physics/lineshape/rate_eqs_test/unmanipulated_bin_lineshape.slurm)
#   COMBINE_ID=$(sbatch --parsable physics/lineshape/rate_eqs_test/ssrf_afp_combine_bins.slurm)
#   sbatch --dependency=afterok:${UNMANIP_ID}:${COMBINE_ID} \
#       physics/lineshape/rate_eqs_test/merge_unmanip_into_combined.slurm
#
# Or standalone once both dirs exist:
#   sbatch physics/lineshape/rate_eqs_test/merge_unmanip_into_combined.slurm
#
# Writes combined_train_all/train_bin_XXXX.npz
#   source: 0=ssrf, 1=afp, 2=unmanipulated

#SBATCH --job-name=merge-unmanip
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=physics/lineshape/rate_eqs_test/slurm_logs/merge_unmanip_%j.out
#SBATCH --error=physics/lineshape/rate_eqs_test/slurm_logs/merge_unmanip_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMBINED_DIR="${COMBINED_DIR:-physics/lineshape/rate_eqs_test/combined_train}"
UNMANIP_DIR="${UNMANIP_DIR:-physics/lineshape/rate_eqs_test/unmanip_train}"
OUTPUT_DIR="${OUTPUT_DIR:-physics/lineshape/rate_eqs_test/combined_train_all}"
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

n_c="$(find "${COMBINED_DIR}" -maxdepth 1 -name 'train_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
n_u="$(find "${UNMANIP_DIR}" -maxdepth 1 -name 'unmanip_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"

echo "Merging unmanip into combined train bins on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}"
echo "Combined: ${COMBINED_DIR} (${n_c} files)"
echo "Unmanip:  ${UNMANIP_DIR} (${n_u} files)"
echo "Output:   ${OUTPUT_DIR}"
echo "Container: ${CONTAINERDIR}/pytorch-2.9.0.sif"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  physics/lineshape/rate_eqs_test/merge_unmanip_into_combined.py \
  --combined-dir "${COMBINED_DIR}" \
  --unmanip-dir "${UNMANIP_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  --strict \
  "$@"
