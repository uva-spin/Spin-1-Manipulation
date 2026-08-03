#!/bin/bash
# Flatten ssRF spectrum shards and build per-bin train NPZs for single_bin.py.
#
# Input:  data/ssrf_shards/ssrf_spectrum_bin_XXXX.npz
# Output: data/ssrf_spectrum_rows/ssrf_spectrum_rows_XXXX.npz
#         data/combined_train_all/train_bin_XXXX.npz  (ps, iplus, iminus, q, P, Q)
#
# Full pipeline (submit from Data_Creation/dulya_fit_v5/):
#   SPEC_JOB=$(sbatch --parsable ssrf_spectrum_array.slurm)
#   sbatch --dependency=afterok:${SPEC_JOB} combine_spectrum_train_bins.slurm
#
# Or standalone (re-flatten / re-combine):
#   sbatch combine_spectrum_train_bins.slurm

#SBATCH --job-name=dulya-v5-spec-combine
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/combine_spectrum_train_bins_%j.out
#SBATCH --error=slurm_logs/combine_spectrum_train_bins_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

PKG_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SSRF_SHARD_DIR="${SSRF_SHARD_DIR:-data/ssrf_shards}"
SSRF_ROWS_DIR="${SSRF_ROWS_DIR:-data/ssrf_spectrum_rows}"
OUTPUT_DIR="${OUTPUT_DIR:-data/combined_train_all}"
NUM_BINS="${NUM_BINS:-500}"

cd "${PKG_DIR}"

mkdir -p slurm_logs
mkdir -p "${SSRF_ROWS_DIR}"
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

n_spec="$(find "${SSRF_SHARD_DIR}" -maxdepth 1 -name 'ssrf_spectrum_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
echo "Dulya v5 ssRF spectrum -> train_bin on $(hostname)"
echo "Pkg dir: ${PKG_DIR}"
echo "Spectrum shards: ${SSRF_SHARD_DIR} (${n_spec} files)"
echo "Row NPZs:        ${SSRF_ROWS_DIR}"
echo "Train bins:      ${OUTPUT_DIR}  (includes calibrated P, Q)"

echo "--- flatten_spectrum_rows (ssrf --flatten-all) ---"
apptainer run --bind "${PKG_DIR}:${PKG_DIR}" --pwd "${PKG_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  flatten_spectrum_rows.py \
  --source ssrf \
  --flatten-all \
  --shard-dir "${SSRF_SHARD_DIR}" \
  --output-dir "${SSRF_ROWS_DIR}" \
  --num-bins "${NUM_BINS}" \
  --strict

echo "--- combine_spectrum_train_bins ---"
apptainer run --bind "${PKG_DIR}:${PKG_DIR}" --pwd "${PKG_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  combine_spectrum_train_bins.py \
  --ssrf-rows-dir "${SSRF_ROWS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  --strict \
  "$@"

n_train="$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'train_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
echo "Done. ${n_train} train_bin_*.npz under ${OUTPUT_DIR}"
