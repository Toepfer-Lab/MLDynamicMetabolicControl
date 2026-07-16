#!/bin/bash -l
#SBATCH --job-name=hvsc1_al_eval_candidates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48gb
#SBATCH --time=10:00:00
#SBATCH --account=ag-toepfer
#SBATCH --output=/home/jkaatz/MA/MLDynamicMetabolicControl/results/slurm/%x-%A_%a.out
#SBATCH --array=0-7

# Stage B of an active-learning round: LP-evaluates a FIXED, already-selected
# candidate set (from scripts/hvsc1_al_generate_pool.py) instead of drawing
# fresh samples. Mirrors scripts/hvsc1_generate_data_array.sh's sharding
# pattern almost exactly, but each array task deterministically recomputes
# its own slice of the same candidates file via --shard-index/--n-shards
# (np.array_split in hvsc1_generate_data.py), so no coordination file is
# needed between tasks.
#
# Output filenames use the SAME hvsc1_training_shard##_seed###.npz pattern
# scripts/hvsc1_merge_shards.py's regex already expects -- no changes to
# that script are needed to merge this batch in. BASE_SEED here is nominal
# (used only for filename bookkeeping, since the actual sampling already
# happened in Stage A); pick a fresh block per round so filenames never
# collide with full/corner_topup/reserved_corner_test or earlier AL rounds.
#
# Usage:
#   sbatch --array=0-7 \
#       --export=CANDIDATES_FILE=data/hvsc1_al_candidates/round01_candidates.npz,N_SHARDS=8,SHARD_SUBDIR=al_round01,BASE_SEED=5000 \
#       scripts/hvsc1_al_eval_candidates_array.sh
#
#   Resubmit only failed/missing task indices, e.g. 3 and 7 (repeat the same
#   --export values used for the original submission so seeds/shards match exactly):
#     sbatch --array=3,7 --export=CANDIDATES_FILE=...,N_SHARDS=8,SHARD_SUBDIR=al_round01,BASE_SEED=5000 \
#         scripts/hvsc1_al_eval_candidates_array.sh

module load lang/Python/3.10.8-GCCcore-12.2.0
module load lang/Java/11.0.20
module load math/CPLEX/22.1.1

export PYTHONPATH=$PYTHONPATH:$CPLEX_HOME/python/3.10/x86-64_linux

VENV=/home/jkaatz/MA/MLDynamicMetabolicControl/.venv
SCRIPT=/home/jkaatz/MA/MLDynamicMetabolicControl/scripts/hvsc1_generate_data.py
REPO_ROOT=/home/jkaatz/MA/MLDynamicMetabolicControl

BASE_SEED=${BASE_SEED:-5000}
N_SHARDS=${N_SHARDS:?N_SHARDS must be set (must equal the --array size)}
CANDIDATES_FILE=${CANDIDATES_FILE:?CANDIDATES_FILE must be set}
SHARD_SUBDIR=${SHARD_SUBDIR:-al_round01}

SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))
SHARD_DIR="$REPO_ROOT/data/hvsc1_shards/$SHARD_SUBDIR"
mkdir -p "$SHARD_DIR"
SHARD_ID=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
OUTPUT="$SHARD_DIR/hvsc1_training_shard${SHARD_ID}_seed${SEED}.npz"

echo "Started at: $(date)"
echo "Array task      : $SLURM_ARRAY_TASK_ID"
echo "Seed (nominal)  : $SEED"
echo "Candidates file : $CANDIDATES_FILE"
echo "N shards        : $N_SHARDS"
echo "Output          : $OUTPUT"

$VENV/bin/python $SCRIPT \
    --candidates-file "$CANDIDATES_FILE" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --n-shards "$N_SHARDS" \
    --fraction 0.5 \
    --model-path "$REPO_ROOT/model/hvsc1_comm.pickle" \
    --output "$OUTPUT" \
    --seed "$SEED"

echo "Finished at: $(date)"
