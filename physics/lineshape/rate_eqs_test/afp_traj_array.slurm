#!/bin/bash
# Per-bin AFP + relaxation trajectory array job (Rivanna / SLURM).
#
# Submit from the repo root:
#   sbatch physics/lineshape/rate_eqs_test/afp_traj_array.slurm
#
# Then organize into per-bin training files after the array finishes:
#   ARRAY_JOB_ID=$(sbatch --parsable physics/lineshape/rate_eqs_test/afp_traj_array.slurm)
#   sbatch --dependency=afterok:${ARRAY_JOB_ID} \
#       physics/lineshape/rate_eqs_test/afp_traj_combine.slurm
#
# Override defaults with env vars:
#   P_MIN=-0.7 P_MAX=0.7 P_STEP=0.05 DT=0.005 N_RELAX=5000 \
#     SHARD_DIR=physics/lineshape/rate_eqs_test/afp_shards \
#     sbatch physics/lineshape/rate_eqs_test/afp_traj_array.slurm
#
# If NUM_BINS changes, update --array below to 0-(NUM_BINS-1).
# Throttle concurrent tasks with e.g. --array=0-499%64

#SBATCH --job-name=afp-bin-traj
#SBATCH --array=0-499
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=physics/lineshape/rate_eqs_test/slurm_logs/afp_traj_%A_%a.out
#SBATCH --error=physics/lineshape/rate_eqs_test/slurm_logs/afp_traj_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_DIR="${SHARD_DIR:-physics/lineshape/rate_eqs_test/afp_shards}"
NUM_BINS="${NUM_BINS:-500}"
P_MIN="${P_MIN:--0.70}"
P_MAX="${P_MAX:-0.70}"
P_STEP="${P_STEP:-0.05}"
DT="${DT:-0.005}"
N_RELAX="${N_RELAX:-5000}"
AFP_WINDOW="${AFP_WINDOW:-3}"
AFP_EFFICIENCY="${AFP_EFFICIENCY:-1.0}"

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

echo "Running AFP traj bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}  Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Repo root: ${REPO_ROOT}"
echo "Shard dir: ${SHARD_DIR}"
echo "P: [${P_MIN}, ${P_MAX}] step=${P_STEP}"
echo "dt=${DT}  n_relax=${N_RELAX}  afp_window=${AFP_WINDOW}  eff=${AFP_EFFICIENCY}"
echo "Container: ${CONTAINERDIR}/pytorch-2.9.0.sif"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  physics/lineshape/rate_eqs_test/afp_bin_traj.py \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --shard-dir "${SHARD_DIR}" \
  --num-bins "${NUM_BINS}" \
  --p-min "${P_MIN}" \
  --p-max "${P_MAX}" \
  --p-step "${P_STEP}" \
  --dt "${DT}" \
  --n-relax "${N_RELAX}" \
  --afp-window "${AFP_WINDOW}" \
  --afp-efficiency "${AFP_EFFICIENCY}" \
  --skip-if-exists \
  "$@"
