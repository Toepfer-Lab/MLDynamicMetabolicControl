#!/bin/bash -l
#SBATCH --job-name=hvsc1_convergence_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48gb
#SBATCH --time=06:00:00
#SBATCH --account=ag-toepfer
#SBATCH --output=/home/jkaatz/MA/MLDynamicMetabolicControl/results/slurm/%x-%A_%a.out
#SBATCH --array=0-19

# SLURM array wrapper for scripts/hvsc1_convergence_sweep.py -- browses a
# large pool of mixture-sampled 27D starting conditions (from
# hvsc1_convergence_sweep_generate_ics.py) via convergence-truncated LP +
# surrogate trajectories, exporting one comparison plot per starting config.
# Each array task deterministically recomputes its own slice of the fixed
# ICs file via --shard-index/--n-shards (np.array_split), matching the
# pattern already used by hvsc1_al_eval_candidates_array.sh -- no
# coordination file needed between tasks.
#
# Default --time=06:00:00 assumes ~8.7s/LP-step and typical convergence in
# 5-55 steps (see results/hvsc1_benchmark.npz); with N_SHARDS=20 over
# N_ICS=300 ICs (~15/task), worst case is 15*55*8.7s =~ 2h -- adjust --time
# if you raise N_ICS or the observed convergence steps run longer.
#
# Usage:
#   1) Generate the fixed IC pool once (no CPLEX needed, run directly):
#        python scripts/hvsc1_convergence_sweep_generate_ics.py --n 300
#
#   2) Submit the array job:
#        sbatch --array=0-19 \
#            --export=ICS_FILE=results/hvsc1_convergence_sweep_ics.npz,N_SHARDS=20,OUTPUT_SUBDIR=sweep01,PLOTS_SUBDIR=sweep01 \
#            scripts/hvsc1_convergence_sweep_array.sh
#
#   Resubmit only failed/missing task indices, e.g. 3 and 7 (repeat the same
#   --export values used for the original submission):
#     sbatch --array=3,7 --export=ICS_FILE=...,N_SHARDS=20,OUTPUT_SUBDIR=sweep01,PLOTS_SUBDIR=sweep01 \
#         scripts/hvsc1_convergence_sweep_array.sh

module load lang/Python/3.10.8-GCCcore-12.2.0
module load lang/Java/11.0.20
module load math/CPLEX/22.1.1

export PYTHONPATH=$PYTHONPATH:$CPLEX_HOME/python/3.10/x86-64_linux

VENV=/home/jkaatz/MA/MLDynamicMetabolicControl/.venv
SCRIPT=/home/jkaatz/MA/MLDynamicMetabolicControl/scripts/hvsc1_convergence_sweep.py
REPO_ROOT=/home/jkaatz/MA/MLDynamicMetabolicControl

ICS_FILE=${ICS_FILE:-$REPO_ROOT/results/hvsc1_convergence_sweep_ics.npz}
N_SHARDS=${N_SHARDS:?N_SHARDS must be set (must equal the --array size)}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-sweep01}
PLOTS_SUBDIR=${PLOTS_SUBDIR:-sweep01}
CHECKPOINT=${CHECKPOINT:-$REPO_ROOT/trained_models/hvsc1_community_input-27_output-27_hidden-128.pt}

SHARD_ID=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
OUTPUT_DIR="$REPO_ROOT/results/hvsc1_convergence_sweep/$OUTPUT_SUBDIR"
PLOTS_DIR="$REPO_ROOT/plots/hvsc1_convergence_sweep/$PLOTS_SUBDIR"
mkdir -p "$OUTPUT_DIR" "$PLOTS_DIR"
OUTPUT="$OUTPUT_DIR/shard${SHARD_ID}.npz"

echo "Started at: $(date)"
echo "Array task : $SLURM_ARRAY_TASK_ID"
echo "ICs file   : $ICS_FILE"
echo "N shards   : $N_SHARDS"
echo "Checkpoint : $CHECKPOINT"
echo "Output     : $OUTPUT"
echo "Plots dir  : $PLOTS_DIR"

$VENV/bin/python $SCRIPT \
    --ics-file "$ICS_FILE" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --n-shards "$N_SHARDS" \
    --model-path "$REPO_ROOT/model/hvsc1_comm.pickle" \
    --checkpoint "$CHECKPOINT" \
    --fraction 0.5 \
    --plots-dir "$PLOTS_DIR" \
    --output "$OUTPUT"

echo "Finished at: $(date)"
