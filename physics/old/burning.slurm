#SBATCH --job-name=gamma-burn-sweep
#SBATCH --array=0-499
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --output=Data_Creation/slurm_logs/gamma_burn_sweep_%a.out
#SBATCH --error=Data_Creation/slurm_logs/gamma_burn_sweep_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-physics/burning_shards}"
NUM_BINS="${NUM_BINS:-500}"
MAX_STEPS="${MAX_STEPS:-10000}"
DTS="${DTS:-0.15,0.015,0.0015,0.00015}"
P0S="${P0S:-0.25,0.35,0.45,0.55}"
GAMMA_AS="${GAMMA_AS:-0.8,0.5,0.3,0.15}"
GAMMA_BS="${GAMMA_BS:-0.7,0.4,0.2,0.1}"
BIN_HEIGHTS="${BIN_HEIGHTS:-0.9}"

cd "${REPO_ROOT}"

mkdir -p Data_Creation/slurm_logs
mkdir -p "${OUTPUT_DIR}"

module purge
module load apptainer pytorch/2.9.0

echo "Running gamma-burn sweep bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Repo root: ${REPO_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Num bins: ${NUM_BINS}"
echo "Max steps: ${MAX_STEPS}"
echo "dts: ${DTS}"
echo "p0s: ${P0S}"
echo "gamma_as: ${GAMMA_AS}"
echo "gamma_bs: ${GAMMA_BS}"
echo "bin_heights: ${BIN_HEIGHTS}"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" physics/sweep_gamma_burn_diffs.py \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --n-bins "${NUM_BINS}" \
  --max-steps "${MAX_STEPS}" \
  --dts "${DTS}" \
  --p0s "${P0S}" \
  --gamma-as "${GAMMA_AS}" \
  --gamma-bs "${GAMMA_BS}" \
  --bin-heights "${BIN_HEIGHTS}" \
  --output "${OUTPUT_DIR}/sweep_bin_${SLURM_ARRAY_TASK_ID}.csv" \
  "$@"
