#!/bin/bash
# Train the global SpectrumSplitNet on spectrum_train.npz.
#
# Submit from repo root (after combine_spectrum_train.py has written the NPZ):
#   sbatch ml/spectrum_split_model.slurm
#
# Chain after a combine job that wrote spectrum_train.npz (replace JOBID):
#   sbatch --dependency=afterok:JOBID ml/spectrum_split_model.slurm
#
# Override:
#   DATA=Data_Creation/dulya_fit_v2/data/spectrum_train \
#   OUTPUT=ml/models/spectrum_split_model.pth \
#   EPOCHS=200 \
#   BATCH_SIZE=32 \
#   LR=1e-3 \
#   MASKED_ALPHA_WEIGHT=0.01 \
#   Q_LOSS_WEIGHT=0.01 \
#   ALPHA_MASK_THRESHOLD=1e-3 \
#     sbatch ml/spectrum_split_model.slurm
#
# DATA may be a single NPZ, a directory of spectrum_train_XXXX.npz shards,
# or spectrum_train_manifest.json from combine_spectrum_train.py --shard-size.

#SBATCH --job-name=spectrum-split-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=TensorStudies/slurm_logs/spectrum_split_train_%j.out
#SBATCH --error=TensorStudies/slurm_logs/spectrum_split_train_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATA="${DATA:-Data_Creation/dulya_fit_v2/data/spectrum_train}"
OUTPUT="${OUTPUT:-ml/models/spectrum_split_model.pth}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
NO_RESIDUAL="${NO_RESIDUAL:-0}"
MASKED_ALPHA_WEIGHT="${MASKED_ALPHA_WEIGHT:-0.01}"
Q_LOSS_WEIGHT="${Q_LOSS_WEIGHT:-0.01}"
ALPHA_MASK_THRESHOLD="${ALPHA_MASK_THRESHOLD:-1e-3}"

cd "${REPO_ROOT}"

mkdir -p TensorStudies/slurm_logs
mkdir -p "$(dirname "${OUTPUT}")"

if [[ ! -e "${DATA}" ]]; then
  echo "Missing training data: ${DATA}" >&2
  echo "Run combine_spectrum_train.py --unmanip-dir ... --shard-size 200000 first." >&2
  exit 1
fi
if [[ -d "${DATA}" ]]; then
  if [[ ! -f "${DATA}/spectrum_train_manifest.json" ]] && ! compgen -G "${DATA}/spectrum_train_*.npz" > /dev/null; then
    if [[ ! -f "${DATA}/spectrum_train.npz" ]]; then
      echo "No spectrum_train shards/manifest/NPZ under ${DATA}" >&2
      exit 1
    fi
  fi
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

echo "Training SpectrumSplitNet on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}"
echo "Repo root: ${REPO_ROOT}"
echo "Data: ${DATA}"
echo "Output: ${OUTPUT}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "Masked alpha loss weight: ${MASKED_ALPHA_WEIGHT}"
echo "Integrated Q loss weight: ${Q_LOSS_WEIGHT}"
echo "Alpha mask relative threshold: ${ALPHA_MASK_THRESHOLD}"
echo "Use residual channel: $([[ "${NO_RESIDUAL}" == "1" ]] && echo no || echo yes)"

TRAIN_ARGS=(
  train
  --data "${DATA}"
  --output "${OUTPUT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --masked-alpha-weight "${MASKED_ALPHA_WEIGHT}"
  --q-loss-weight "${Q_LOSS_WEIGHT}"
  --alpha-mask-threshold "${ALPHA_MASK_THRESHOLD}"
)
if [[ "${NO_RESIDUAL}" == "1" ]]; then
  TRAIN_ARGS+=(--no-residual)
fi

apptainer run --nv --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  ml/spectrum_split_model.py \
  "${TRAIN_ARGS[@]}" \
  "$@"
