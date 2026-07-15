#!/bin/bash -l
#SBATCH --job-name=hvsc1_gen_data_smoketest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48gb
#SBATCH --time=00:20:00
#SBATCH --account=ag-toepfer
#SBATCH --output=/home/jkaatz/MA/MLDynamicMetabolicControl/results/slurm/%x-%j.out

module load lang/Python/3.10.8-GCCcore-12.2.0
module load lang/Java/11.0.20
module load math/CPLEX/22.1.1

export PYTHONPATH=$PYTHONPATH:$CPLEX_HOME/python/3.10/x86-64_linux

VENV=/home/jkaatz/MA/MLDynamicMetabolicControl/.venv
SCRIPT=/home/jkaatz/MA/MLDynamicMetabolicControl/scripts/hvsc1_generate_data.py

echo "Started at: $(date)"
$VENV/bin/python $SCRIPT \
    --n-samples 10 \
    --fraction 0.5 \
    --model-path /home/jkaatz/MA/MLDynamicMetabolicControl/model/hvsc1_comm.pickle \
    --output /home/jkaatz/MA/MLDynamicMetabolicControl/results/hvsc1_training_smoketest.npz
echo "Finished at: $(date)"
