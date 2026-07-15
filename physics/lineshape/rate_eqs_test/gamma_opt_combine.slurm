#!/bin/bash
# Combine independent gamma-opt shards after the array job finishes.
#
# Submit with dependency on a completed array job (from repo root):
#   ARRAY_JOB_ID=$(sbatch --parsable physics/lineshape/rate_eqs_test/gamma_opt_array.slurm)
#   sbatch --dependency=afterok:${ARRAY_JOB_ID} \
#       physics/lineshape/rate_eqs_test/gamma_opt_combine.slurm
#
# Or submit standalone once all shards exist:
#   sbatch physics/lineshape/rate_eqs_test/gamma_opt_combine.slurm
#
# Override defaults with the same env vars used for the array job:
#   N_STEPS=300 SHARD_DIR=... OUTPUT_DIR=... \
#     sbatch physics/lineshape/rate_eqs_test/gamma_opt_combine.slurm

#SBATCH --job-name=gamma-opt-combine
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=physics/lineshape/rate_eqs_test/slurm_logs/gamma_opt_combine_%j.out
#SBATCH --error=physics/lineshape/rate_eqs_test/slurm_logs/gamma_opt_combine_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_DIR="${SHARD_DIR:-physics/lineshape/rate_eqs_test/gamma_opt_shards}"
OUTPUT_DIR="${OUTPUT_DIR:-physics/lineshape/rate_eqs_test/gamma_opt_combined}"
NUM_BINS="${NUM_BINS:-500}"
P="${P:-0.40}"
DT="${DT:-0.05}"
N_STEPS="${N_STEPS:-500}"
GAMMA_HI="${GAMMA_HI:-5.0}"

cd "${REPO_ROOT}"

mkdir -p physics/lineshape/rate_eqs_test/slurm_logs
mkdir -p "${OUTPUT_DIR}"

# Headless + writable caches on compute nodes
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

n_shards="$(find "${SHARD_DIR}" -maxdepth 1 -name 'gamma_opt_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
echo "Combining gamma-opt shards on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}"
echo "Repo root: ${REPO_ROOT}"
echo "Shard dir: ${SHARD_DIR} (${n_shards} shards found)"
echo "Output dir: ${OUTPUT_DIR}"
echo "Num bins: ${NUM_BINS}"
echo "P: ${P}"
echo "dt: ${DT}"
echo "n_steps: ${N_STEPS}"
echo "gamma_hi: ${GAMMA_HI}"
echo "Container: ${CONTAINERDIR}/pytorch-2.9.0.sif"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  physics/lineshape/rate_eqs_test/rate_eqs_test_ssrf_all_bins_gamma_opt_parallel.py \
  --combine \
  --strict \
  --shard-dir "${SHARD_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --polarization "${P}" \
  --num-bins "${NUM_BINS}" \
  --dt "${DT}" \
  --n-steps "${N_STEPS}" \
  --gamma-hi "${GAMMA_HI}" \
  "$@"
