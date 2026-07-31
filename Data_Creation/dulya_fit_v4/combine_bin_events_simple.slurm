#!/bin/bash
# Reconstruct dense num_bins-wide spectrum rows from ssRF + AFP + unmanipulated
# *trajectory* shards (single-bin mode, NOT --spectrum-mode) into one
# data/spectrum_train/spectrum_train.npz.
#
# Every bin's shard shares the same (P, gamma_rf, burn_steps) combo grid, so
# for a given combo/timestep, column bin_idx = that bin's own shard value at
# that combo. Concatenation on bin_idx, nothing else: no equilibrium fill, no
# cross-bin matching beyond the shared combo index, no conservation checks.
#
# Pipeline:
#   sbatch ssrf_traj_array.slurm      # data/ssrf_shards/ssrf_bin_XXXX.npz
#   sbatch afp_traj_array.slurm       # data/afp_shards/afp_bin_XXXX.npz
#   sbatch unmanipulated_bin_array.slurm  # data/unmanip_train/unmanip_bin_XXXX.npz
#   sbatch combine_bin_events_simple.slurm   # <- this script
#
# Submit from the dulya_fit_v2 checkout (where data/ lives):
#   cd /path/to/dulya_fit_v2 && sbatch combine_bin_events_simple.slurm
#
# If data/ is on scratch instead, set DATA_DIR before submitting:
#   export DATA_DIR="${SCRATCH}/dulya_fit_v2/data"
#   sbatch combine_bin_events_simple.slurm

#SBATCH --job-name=dulya-v2-combine-bin-events
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/combine_bin_events_simple_%j.out
#SBATCH --error=slurm_logs/combine_bin_events_simple_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${0}}")" && pwd)"
# Under SLURM the batch script is copied into /var/spool/slurm/...; BASH_SOURCE
# then points at that spool copy, not your checkout. Always prefer SLURM_SUBMIT_DIR
# (the directory you ran sbatch from).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PKG_DIR="${PKG_DIR:-${SLURM_SUBMIT_DIR}}"
else
  PKG_DIR="${PKG_DIR:-${SCRIPT_DIR}}"
fi

if [[ "${PKG_DIR}" == /var/spool/slurm/* ]]; then
  echo "ERROR: PKG_DIR=${PKG_DIR} looks like a SLURM spool path." >&2
  echo "Submit from your dulya_fit_v2 checkout, e.g.:" >&2
  echo "  cd /path/to/dulya_fit_v2 && sbatch combine_bin_events_simple.slurm" >&2
  echo "Or set PKG_DIR explicitly to that checkout path." >&2
  exit 1
fi

DATA_DIR="${DATA_DIR:-${PKG_DIR}/data}"
SSRF_SHARD_DIR="${SSRF_SHARD_DIR:-${DATA_DIR}/ssrf_shards}"
AFP_SHARD_DIR="${AFP_SHARD_DIR:-${DATA_DIR}/afp_shards}"
UNMANIP_DIR="${UNMANIP_DIR:-${DATA_DIR}/unmanip_train}"
OUTPUT="${OUTPUT:-${DATA_DIR}/spectrum_train/spectrum_train.npz}"
NUM_BINS="${NUM_BINS:-500}"

cd "${PKG_DIR}"

ensure_writable_dir() {
  local dir="$1"
  if mkdir -p "${dir}" 2>/dev/null; then
    return 0
  fi
  echo "ERROR: cannot create writable directory '${dir}'." >&2
  echo "PKG_DIR=${PKG_DIR}  DATA_DIR=${DATA_DIR}" >&2
  echo "Submit from a writable dulya_fit_v2 checkout, or set DATA_DIR/OUTPUT to scratch:" >&2
  echo "  export DATA_DIR=\"\${SCRATCH}/dulya_fit_v2/data\"" >&2
  echo "  export OUTPUT=\"\${SCRATCH}/dulya_fit_v2/data/spectrum_train/spectrum_train.npz\"" >&2
  exit 1
}

ensure_writable_dir "$(dirname "${OUTPUT}")"

trap 'echo "FAILED at line ${LINENO} (exit $?)" >&2' ERR

echo "PKG_DIR=${PKG_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "SLURM job id: ${SLURM_JOB_ID:-local}"
echo "SLURM submit dir: ${SLURM_SUBMIT_DIR:-${PKG_DIR}}"
echo "Output NPZ: ${OUTPUT}"

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

n_ssrf="$(
  find "${SSRF_SHARD_DIR}" -maxdepth 1 -name 'ssrf_bin_*' 2>/dev/null \
    | sed -n 's|.*/ssrf_bin_\([0-9][0-9][0-9][0-9]\).*|\1|p' \
    | sort -u | wc -l | tr -d ' '
)"
n_afp="$(find "${AFP_SHARD_DIR}" -maxdepth 1 -name 'afp_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
n_unmanip="$(find "${UNMANIP_DIR}" -maxdepth 1 -name 'unmanip_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"

echo "Combining Dulya v2 ssRF + AFP + unmanip (simple bin concat) -> ${OUTPUT}"
echo "ssRF shards: ${SSRF_SHARD_DIR} (${n_ssrf})"
echo "AFP shards:  ${AFP_SHARD_DIR} (${n_afp})"
echo "Unmanip:     ${UNMANIP_DIR} (${n_unmanip})"

if [[ "${n_ssrf}" -eq 0 || "${n_afp}" -eq 0 || "${n_unmanip}" -eq 0 ]]; then
  echo "ERROR: expected ssrf/afp/unmanip shards under ${PKG_DIR}; found ssrf=${n_ssrf} afp=${n_afp} unmanip=${n_unmanip}" >&2
  echo "Run ssrf_traj_array.slurm, afp_traj_array.slurm, and unmanipulated_bin_array.slurm first." >&2
  exit 1
fi

if [[ "${n_ssrf}" -lt "${NUM_BINS}" || "${n_afp}" -lt "${NUM_BINS}" || "${n_unmanip}" -lt "${NUM_BINS}" ]]; then
  echo "WARNING: incomplete shard sets (need ${NUM_BINS} each); combine will fail with --strict" >&2
fi

APPTAINER_BINDS=(--bind "${PKG_DIR}:${PKG_DIR}")
if [[ "${DATA_DIR}" = /* ]]; then
  APPTAINER_BINDS+=(--bind "${DATA_DIR}:${DATA_DIR}")
fi
if [[ "${OUTPUT}" = /* ]]; then
  out_parent="$(dirname "${OUTPUT}")"
  if [[ "${out_parent}" != "${DATA_DIR}" && "${out_parent}" != "${DATA_DIR}/"* ]]; then
    APPTAINER_BINDS+=(--bind "${out_parent}:${out_parent}")
  fi
fi

COMBINE_ARGS=(
  --ssrf-shard-dir "${SSRF_SHARD_DIR}"
  --afp-shard-dir "${AFP_SHARD_DIR}"
  --unmanip-dir "${UNMANIP_DIR}"
  --output "${OUTPUT}"
  --num-bins "${NUM_BINS}"
  --strict
  --no-validate
)
# Only forward sbatch CLI extras when non-empty (bare "$@" can pass '' and break argparse).
if (( $# > 0 )); then
  COMBINE_ARGS+=("$@")
fi

apptainer run "${APPTAINER_BINDS[@]}" --pwd "${PKG_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  combine_bin_events_simple.py \
  "${COMBINE_ARGS[@]}"
