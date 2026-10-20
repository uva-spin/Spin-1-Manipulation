#!/bin/bash
# Train one spectrum→PQ model. Submit from ml/, the directory containing
# this script and the model scripts:
#   MODEL=mlp sbatch train.slurm
#   MODEL=cnn sbatch train.slurm -- --epochs 30
#   MODEL=lstm sbatch train.slurm
#   SPECTRA=data/spectra.npz MODEL=gru sbatch train.slurm

#SBATCH --job-name=pq-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(dirname "${WORKDIR}")"
MODEL="${MODEL:-mlp}"
SPECTRA="${SPECTRA:-data/spectra.npz}"

cd "${WORKDIR}"
mkdir -p slurm_logs

module purge
module load apptainer pytorch/2.9.0

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

[[ "${1:-}" == "--" ]] && shift

apptainer run --nv \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  --pwd "${WORKDIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  "${MODEL}.py" \
  --spectra "${SPECTRA}" \
  "$@"
