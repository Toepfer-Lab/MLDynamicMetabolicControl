#!/bin/bash -l
#SBATCH --job-name=hvsc1_al_ensemble_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --account=ag-toepfer
#SBATCH --output=/home/jkaatz/MA/MLDynamicMetabolicControl/results/slurm/%x-%j.out

# Stage A of an active-learning round: trains the 5-member bootstrap
# ensemble (scripts/hvsc1_al_ensemble_train.py). Pure torch/numpy/sklearn --
# no micom/cobra/CPLEX needed, so only the Python module is loaded here
# (unlike the CPLEX-dependent data-generation/simulation jobs).
#
# Usage:
#   sbatch --export=ROUND_ID=round01,DATA_PATH=data/hvsc1_training.npz,BASE_SEED=7000 \
#       scripts/hvsc1_al_ensemble_train.sh

module load lang/Python/3.10.8-GCCcore-12.2.0

VENV=/home/jkaatz/MA/MLDynamicMetabolicControl/.venv
SCRIPT=/home/jkaatz/MA/MLDynamicMetabolicControl/scripts/hvsc1_al_ensemble_train.py
REPO_ROOT=/home/jkaatz/MA/MLDynamicMetabolicControl

ROUND_ID=${ROUND_ID:-round01}
DATA_PATH=${DATA_PATH:-$REPO_ROOT/data/hvsc1_training.npz}
BASE_SEED=${BASE_SEED:-7000}
N_MEMBERS=${N_MEMBERS:-5}
HIDDEN_DIM=${HIDDEN_DIM:-128}
N_LAYERS=${N_LAYERS:-2}
EPOCHS=${EPOCHS:-3000}
PATIENCE=${PATIENCE:-100}

echo "Started at: $(date)"
echo "Round ID  : $ROUND_ID"
echo "Data path : $DATA_PATH"

$VENV/bin/python $SCRIPT \
    --data-path "$DATA_PATH" \
    --n-members "$N_MEMBERS" \
    --hidden-dim "$HIDDEN_DIM" --n-layers "$N_LAYERS" \
    --epochs "$EPOCHS" --patience "$PATIENCE" \
    --base-seed "$BASE_SEED" \
    --round-id "$ROUND_ID"

echo "Finished at: $(date)"
