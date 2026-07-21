#!/bin/bash
\
#SBATCH --job-name=ssrf-bin-traj
#SBATCH --array=0-499
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=physics/lineshape/rate_eqs_test/slurm_logs/ssrf_traj_%A_%a.out
#SBATCH --error=physics/lineshape/rate_eqs_test/slurm_logs/ssrf_traj_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_DIR="${SHARD_DIR:-physics/lineshape/rate_eqs_test/ssrf_shards}"
NUM_BINS="${NUM_BINS:-500}"
P_MIN="${P_MIN:--0.70}"
P_MAX="${P_MAX:-0.70}"
P_STEP="${P_STEP:-0.05}"
DT="${DT:-0.005}"
GAMMA_RF="${GAMMA_RF:-1.0}"
SIGMA_BINS="${SIGMA_BINS:-2.0}"
VOIGT_GAMMA_BINS="${VOIGT_GAMMA_BINS:-1.0}"
HALF_WIDTH="${HALF_WIDTH:-5}"
MAX_STEPS="${MAX_STEPS:-5000}"

cd "${REPO_ROOT}"

mkdir -p physics/lineshape/rate_eqs_test/slurm_logs
mkdir -p "${SHARD_DIR}"

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

echo "Running ssRF traj bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}  Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Repo root: ${REPO_ROOT}"
echo "Shard dir: ${SHARD_DIR}"
echo "P: [${P_MIN}, ${P_MAX}] step=${P_STEP}"
echo "dt=${DT}  gamma_rf=${GAMMA_RF}  voigt(sigma=${SIGMA_BINS}, gamma=${VOIGT_GAMMA_BINS})  +/-${HALF_WIDTH}"
echo "max_steps=${MAX_STEPS}  stop=mirror_turnover"
echo "Container: ${CONTAINERDIR}/pytorch-2.9.0.sif"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  physics/lineshape/rate_eqs_test/ssrf_bin_traj.py \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --shard-dir "${SHARD_DIR}" \
  --num-bins "${NUM_BINS}" \
  --p-min "${P_MIN}" \
  --p-max "${P_MAX}" \
  --p-step "${P_STEP}" \
  --dt "${DT}" \
  --gamma-rf "${GAMMA_RF}" \
  --sigma-bins "${SIGMA_BINS}" \
  --voigt-gamma-bins "${VOIGT_GAMMA_BINS}" \
  --half-width "${HALF_WIDTH}" \
  --max-steps "${MAX_STEPS}" \
  --skip-if-exists \
  "$@"
