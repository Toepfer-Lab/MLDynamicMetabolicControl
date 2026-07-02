#!/bin/bash -l
#SBATCH --job-name=mcsm_gen_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --time=05:00:00
#SBATCH --account=ag-toepfer
#SBATCH --output=/home/jkaatz/MA/MLDynamicMetabolicControl/results/slurm/%x-%j.out

module load lang/Python/3.10.8-GCCcore-12.2.0
module load lang/Java/11.0.20
module load math/CPLEX/22.1.1

export PYTHONPATH=$PYTHONPATH:$CPLEX_HOME/python/3.10/x86-64_linux

VENV=/home/jkaatz/MA/MLDynamicMetabolicControl/.venv
SCRIPT=/home/jkaatz/MA/MLDynamicMetabolicControl/scripts/mcsm_generate_data.py

echo "Started at: $(date)"
$VENV/bin/python $SCRIPT \
    --n-samples 5000 \
    --fraction 0.5 \
    --model-path /home/jkaatz/MA/MLDynamicMetabolicControl/model/dcom.pickle \
    --medium-path /home/jkaatz/MA/MLDynamicMetabolicControl/data/Completed_maize_leaf_medium.csv \
    --output /home/jkaatz/MA/MLDynamicMetabolicControl/data/mcsm_training.npz
echo "Finished at: $(date)"
