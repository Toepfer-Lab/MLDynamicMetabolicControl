#!/bin/bash -l
#SBATCH --job-name=hvsc1_al_capacity_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --account=ag-toepfer
#SBATCH --output=/home/jkaatz/MA/MLDynamicMetabolicControl/results/slurm/%x-%j.out

# Stage D (part 1) of an active-learning round: capacity sweep
# (scripts/hvsc1_al_capacity_sweep.py). Pure torch/numpy/sklearn -- no
# micom/cobra/CPLEX needed (it only subprocess-calls train_surrogate.py and
# hvsc1_eval_reserved_corner.py, both CPLEX-free), so only the Python module
# is loaded here.
#
# Usage:
#   sbatch --export=ROUND_ID=round01,DATA_PATH=data/hvsc1_training.npz \
#       scripts/hvsc1_al_capacity_sweep.sh

module load lang/Python/3.10.8-GCCcore-12.2.0

VENV=/home/jkaatz/MA/MLDynamicMetabolicControl/.venv
SCRIPT=/home/jkaatz/MA/MLDynamicMetabolicControl/scripts/hvsc1_al_capacity_sweep.py
REPO_ROOT=/home/jkaatz/MA/MLDynamicMetabolicControl

ROUND_ID=${ROUND_ID:-round01}
DATA_PATH=${DATA_PATH:-$REPO_ROOT/data/hvsc1_training.npz}
HIDDEN_DIMS=${HIDDEN_DIMS:-128,256,384}
N_LAYERS_LIST=${N_LAYERS_LIST:-2,3}
EPOCHS=${EPOCHS:-3000}
PATIENCE=${PATIENCE:-100}

echo "Started at: $(date)"
echo "Round ID  : $ROUND_ID"
echo "Data path : $DATA_PATH"

$VENV/bin/python $SCRIPT \
    --data-path "$DATA_PATH" \
    --hidden-dims "$HIDDEN_DIMS" \
    --n-layers-list "$N_LAYERS_LIST" \
    --epochs "$EPOCHS" --patience "$PATIENCE" \
    --round-id "$ROUND_ID"

echo "Finished at: $(date)"
