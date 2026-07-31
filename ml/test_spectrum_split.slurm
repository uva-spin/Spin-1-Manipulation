#!/bin/bash
# Evaluate SpectrumSplitNet on holdout spectra (and optional external pickle).
#
# Submit from repo root (after training has written the model checkpoint):
#   sbatch ml/test_spectrum_split.slurm
#
# Chain after training (replace JOBID):
#   sbatch --dependency=afterok:JOBID ml/test_spectrum_split.slurm
#
# Override:
#   MODEL=ml/models/spectrum_split_model.pth \
#   EVENTS_DIR=Data_Creation/sample_manipulation_events \
#   OUTPUT_DIR=ml/results/test_spectrum_split \
#     sbatch ml/test_spectrum_split.slurm
#
# Or evaluate NPZ holdout rows instead:
#   DATA=spectrum_train.npz sbatch ml/test_spectrum_split.slurm

#SBATCH --job-name=spectrum-split-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=TensorStudies/slurm_logs/spectrum_split_test_%j.out
#SBATCH --error=TensorStudies/slurm_logs/spectrum_split_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL="${MODEL:-ml/models/spectrum_split_model.pth}"
DATA="${DATA:-spectrum_train.npz}"
EVENTS_DIR="${EVENTS_DIR:-Data_Creation/sample_manipulation_events}"
OUTPUT_DIR="${OUTPUT_DIR:-ml/results/test_spectrum_split}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEST_PICKLE="${TEST_PICKLE:-}"

cd "${REPO_ROOT}"

mkdir -p TensorStudies/slurm_logs
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${MODEL}" ]]; then
  echo "Missing model checkpoint: ${MODEL}" >&2
  echo "Train first with ml/spectrum_split_model.slurm (or set MODEL=...)." >&2
  exit 1
fi

USE_EVENTS=0
if [[ -n "${EVENTS_DIR}" && -d "${EVENTS_DIR}" ]]; then
  USE_EVENTS=1
elif [[ ! -e "${DATA}" ]]; then
  echo "Missing evaluation data: ${DATA}" >&2
  echo "Run create_sample_manipulation_events.py or combine_spectrum_train.py first." >&2
  exit 1
fi

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

echo "Evaluating SpectrumSplitNet on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}"
echo "Repo root: ${REPO_ROOT}"
echo "Model: ${MODEL}"
if [[ "${USE_EVENTS}" == "1" ]]; then
  echo "Events dir: ${EVENTS_DIR}"
else
  echo "Data: ${DATA}"
fi
echo "Output dir: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
if [[ -n "${TEST_PICKLE}" ]]; then
  echo "External pickle: ${TEST_PICKLE}"
else
  echo "External pickle: (none)"
fi

TEST_ARGS=(
  --model "${MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
)
if [[ "${USE_EVENTS}" == "1" ]]; then
  TEST_ARGS+=(--events-dir "${EVENTS_DIR}")
else
  TEST_ARGS+=(--data "${DATA}")
fi
if [[ -n "${TEST_PICKLE}" ]]; then
  TEST_ARGS+=(--test-pickle "${TEST_PICKLE}")
fi

apptainer run --nv --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  test_spectrum_split.py \
  "${TEST_ARGS[@]}" \
  "$@"
