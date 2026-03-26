#!/bin/bash
#SBATCH --job-name=greedy_optim
#SBATCH --account=ag-toepfer
#SBATCH --partition=longsmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/jkaatz/MA/MLDynamicMetabolicControl/logs/slurm_%j_greedy_optim.out


# ---- environment ----
REPO_DIR="/home/jkaatz/MA/MLDynamicMetabolicControl"
cd "$REPO_DIR" || { echo "ERROR: could not cd to $REPO_DIR"; exit 1; }

source "$REPO_DIR/.venv/bin/activate" || { echo "ERROR: could not activate venv at $REPO_DIR/.venv"; exit 1; }

# ---- run ----
echo "[$(date)] Starting dvc repro plot_greedy"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Params:"
cat params.yaml

dvc repro plot_greedy

echo "[$(date)] Job finished (exit code $?)"
