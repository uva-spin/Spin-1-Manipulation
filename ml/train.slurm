#!/bin/bash
# Train one spectrum→PQ model. Submit from repo root:
#   MODEL=mlp sbatch ml/train.slurm
#   MODEL=cnn sbatch ml/train.slurm -- --epochs 30

#SBATCH --job-name=pq-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=ml/slurm_logs/train_%j.out
#SBATCH --error=ml/slurm_logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL="${MODEL:-mlp}"
SPECTRA="${SPECTRA:-ml/data/spectra_v6.npz}"

cd "${REPO_ROOT}"
mkdir -p ml/slurm_logs

module purge
module load apptainer pytorch/2.9.0

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

[[ "${1:-}" == "--" ]] && shift

apptainer run --nv \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  --pwd "${REPO_ROOT}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  "ml/${MODEL}.py" \
  --spectra "${SPECTRA}" \
  "$@"
