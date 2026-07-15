#!/bin/bash -l
#SBATCH --job-name=hvsc1_gen_data_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48gb
#SBATCH --time=10:00:00
#SBATCH --account=ag-toepfer
#SBATCH --output=/home/jkaatz/MA/MLDynamicMetabolicControl/results/slurm/%x-%A_%a.out
#SBATCH --array=0-7

# Parallel shard of hvsc1_generate_data.py. Each array task draws a
# distinct, deterministic Dirichlet seed (BASE_SEED + task id) and writes
# its own shard file; scripts/hvsc1_merge_shards.py concatenates them
# afterwards into the single data/hvsc1_training.npz the rest of the
# pipeline expects.
#
# N_SAMPLES_PER_SHARD, SHARD_SUBDIR, BASE_SEED, SAMPLING_MODE and
# MIXTURE_WEIGHTS are all overridable via `sbatch --export` so the same
# script serves the pilot, the original naive full run, and the
# corner-coverage topup without duplicating scripts or colliding
# seeds/outputs between phases.
#
# Usage:
#   Pilot (3 tasks, 200 samples each, ~15-20 min, naive mode):
#     sbatch --array=0-2 --export=N_SAMPLES_PER_SHARD=200,SHARD_SUBDIR=pilot \
#         scripts/hvsc1_generate_data_array.sh
#
#   Original full run (8 tasks, 3500 samples each = 28,000 total, naive mode):
#     sbatch --array=0-7 --export=N_SAMPLES_PER_SHARD=3500,SHARD_SUBDIR=full \
#         scripts/hvsc1_generate_data_array.sh
#
#   Corner-coverage topup pilot (distinct BASE_SEED so it never collides with
#   the full/pilot naive runs above, even though it lands in its own subdir):
#     sbatch --array=0-2 \
#         --export=N_SAMPLES_PER_SHARD=200,SHARD_SUBDIR=corner_pilot,BASE_SEED=2000,SAMPLING_MODE=mixture \
#         scripts/hvsc1_generate_data_array.sh
#
#   Corner-coverage topup full run (16 tasks x 4000 = 64,000 total):
#     sbatch --array=0-15 \
#         --export=N_SAMPLES_PER_SHARD=4000,SHARD_SUBDIR=corner_topup,BASE_SEED=2000,SAMPLING_MODE=mixture \
#         scripts/hvsc1_generate_data_array.sh
#
#   Resubmit only failed/missing task indices, e.g. 3 and 7 (repeat the same
#   --export values used for the original submission so seeds match exactly):
#     sbatch --array=3,7 --export=N_SAMPLES_PER_SHARD=4000,SHARD_SUBDIR=corner_topup,BASE_SEED=2000,SAMPLING_MODE=mixture \
#         scripts/hvsc1_generate_data_array.sh

module load lang/Python/3.10.8-GCCcore-12.2.0
module load lang/Java/11.0.20
module load math/CPLEX/22.1.1

export PYTHONPATH=$PYTHONPATH:$CPLEX_HOME/python/3.10/x86-64_linux

VENV=/home/jkaatz/MA/MLDynamicMetabolicControl/.venv
SCRIPT=/home/jkaatz/MA/MLDynamicMetabolicControl/scripts/hvsc1_generate_data.py
REPO_ROOT=/home/jkaatz/MA/MLDynamicMetabolicControl

BASE_SEED=${BASE_SEED:-1000}
N_SAMPLES_PER_SHARD=${N_SAMPLES_PER_SHARD:-3500}
SHARD_SUBDIR=${SHARD_SUBDIR:-full}
SAMPLING_MODE=${SAMPLING_MODE:-naive}
MIXTURE_WEIGHTS=${MIXTURE_WEIGHTS:-naive=0.10,low-alpha=0.20,one-dominant=0.45,co-dominant=0.25}

SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))
SHARD_DIR="$REPO_ROOT/data/hvsc1_shards/$SHARD_SUBDIR"
mkdir -p "$SHARD_DIR"
SHARD_ID=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
OUTPUT="$SHARD_DIR/hvsc1_training_shard${SHARD_ID}_seed${SEED}.npz"

echo "Started at: $(date)"
echo "Array task    : $SLURM_ARRAY_TASK_ID"
echo "Seed          : $SEED"
echo "N samples     : $N_SAMPLES_PER_SHARD"
echo "Sampling mode : $SAMPLING_MODE"
echo "Output        : $OUTPUT"

$VENV/bin/python $SCRIPT \
    --n-samples "$N_SAMPLES_PER_SHARD" \
    --fraction 0.5 \
    --model-path "$REPO_ROOT/model/hvsc1_comm.pickle" \
    --output "$OUTPUT" \
    --seed "$SEED" \
    --sampling-mode "$SAMPLING_MODE" \
    --mixture-weights "$MIXTURE_WEIGHTS"

echo "Finished at: $(date)"
