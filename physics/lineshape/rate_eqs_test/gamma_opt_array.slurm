#!/bin/bash
# Independent per-bin gamma-opt array job (Rivanna / SLURM).
#
# Submit from the repo root:
#   sbatch physics/lineshape/rate_eqs_test/gamma_opt_array.slurm
#
# Then combine after the array finishes:
#   ARRAY_JOB_ID=$(sbatch --parsable physics/lineshape/rate_eqs_test/gamma_opt_array.slurm)
#   sbatch --dependency=afterok:${ARRAY_JOB_ID} \
#       physics/lineshape/rate_eqs_test/gamma_opt_combine.slurm
#
# Override defaults with env vars (must match combine job):
#   N_STEPS=300 DT=0.05 P=0.40 SHARD_DIR=physics/lineshape/rate_eqs_test/gamma_opt_shards \
#     sbatch physics/lineshape/rate_eqs_test/gamma_opt_array.slurm
#
# If NUM_BINS changes, update --array below to 0-(NUM_BINS-1).
# Throttle concurrent tasks with e.g. --array=0-499%64

#SBATCH --job-name=gamma-opt-bin
#SBATCH --array=0-499
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=physics/lineshape/rate_eqs_test/slurm_logs/gamma_opt_%A_%a.out
#SBATCH --error=physics/lineshape/rate_eqs_test/slurm_logs/gamma_opt_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_DIR="${SHARD_DIR:-physics/lineshape/rate_eqs_test/gamma_opt_shards}"
NUM_BINS="${NUM_BINS:-500}"
P="${P:-0.40}"
DT="${DT:-0.05}"
N_STEPS="${N_STEPS:-500}"
GAMMA_HI="${GAMMA_HI:-5.0}"
N_BISECT="${N_BISECT:-12}"

cd "${REPO_ROOT}"

mkdir -p physics/lineshape/rate_eqs_test/slurm_logs
mkdir -p "${SHARD_DIR}"

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
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is unset; submit as an array job" >&2
  exit 1
fi

echo "Running gamma-opt bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}  Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Repo root: ${REPO_ROOT}"
echo "Shard dir: ${SHARD_DIR}"
echo "Num bins: ${NUM_BINS}"
echo "P: ${P}"
echo "dt: ${DT}"
echo "n_steps: ${N_STEPS}"
echo "gamma_hi: ${GAMMA_HI}"
echo "n_bisect: ${N_BISECT}"
echo "Container: ${CONTAINERDIR}/pytorch-2.9.0.sif"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  physics/lineshape/rate_eqs_test/rate_eqs_test_ssrf_all_bins_gamma_opt_parallel.py \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --shard-dir "${SHARD_DIR}" \
  --polarization "${P}" \
  --num-bins "${NUM_BINS}" \
  --dt "${DT}" \
  --n-steps "${N_STEPS}" \
  --gamma-hi "${GAMMA_HI}" \
  --n-bisect "${N_BISECT}" \
  --skip-if-exists \
  "$@"
