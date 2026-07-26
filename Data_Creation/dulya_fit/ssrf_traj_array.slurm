#!/bin/bash
# Dulya-fit ssRF per-bin trajectory array (500 independent burn bins).
#
# Submit from repo root:
#   sbatch Data_Creation/dulya_fit/ssrf_traj_array.slurm
#
# Then organize shards into per-bin training NPZs:
#   ARRAY_JOB_ID=$(sbatch --parsable Data_Creation/dulya_fit/ssrf_traj_array.slurm)
#   sbatch --dependency=afterok:${ARRAY_JOB_ID} \
#       Data_Creation/dulya_fit/ssrf_traj_combine.slurm
#
# Throttle concurrency: --array=0-499%64

#SBATCH --job-name=dulya-ssrf-traj
#SBATCH --array=0-499
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=Data_Creation/dulya_fit/slurm_logs/ssrf_traj_%A_%a.out
#SBATCH --error=Data_Creation/dulya_fit/slurm_logs/ssrf_traj_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_DIR="${SHARD_DIR:-Data_Creation/dulya_fit/data/ssrf_shards}"
NUM_BINS="${NUM_BINS:-500}"
P_MIN="${P_MIN:--0.9}"
P_MAX="${P_MAX:-0.9}"
P_STEP="${P_STEP:-0.05}"
DT="${DT:-0.005}"
GAMMA_RF="${GAMMA_RF:-100.0}"
SIGMA_BINS="${SIGMA_BINS:-2.0}"
VOIGT_GAMMA_BINS="${VOIGT_GAMMA_BINS:-1.0}"
HALF_WIDTH="${HALF_WIDTH:-}"
MAX_STEPS="${MAX_STEPS:-5000}"

cd "${REPO_ROOT}"

mkdir -p Data_Creation/dulya_fit/slurm_logs
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

HW_ARGS=()
if [[ -n "${HALF_WIDTH}" ]]; then
  HW_ARGS=(--half-width "${HALF_WIDTH}")
fi

echo "Dulya ssRF traj bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Shard dir: ${SHARD_DIR}"
echo "P: [${P_MIN}, ${P_MAX}] step=${P_STEP}  gamma=${GAMMA_RF}  max_steps=${MAX_STEPS}"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  Data_Creation/dulya_fit/ssrf_bin_traj.py \
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
  --max-steps "${MAX_STEPS}" \
  --skip-if-exists \
  "${HW_ARGS[@]}" \
  "$@"
