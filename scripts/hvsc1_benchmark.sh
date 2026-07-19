#!/bin/bash -l
#SBATCH --job-name=hvsc1_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48gb
#SBATCH --time=02:00:00
#SBATCH --account=ag-toepfer
#SBATCH --output=/home/jkaatz/MA/MLDynamicMetabolicControl/results/slurm/%x-%j.out

module load lang/Python/3.10.8-GCCcore-12.2.0
module load lang/Java/11.0.20
module load math/CPLEX/22.1.1

export PYTHONPATH=$PYTHONPATH:$CPLEX_HOME/python/3.10/x86-64_linux

VENV=/home/jkaatz/MA/MLDynamicMetabolicControl/.venv
SCRIPT=/home/jkaatz/MA/MLDynamicMetabolicControl/scripts/hvsc1_benchmark.py
REPO_ROOT=/home/jkaatz/MA/MLDynamicMetabolicControl

# TAG/CHECKPOINT overridable via `sbatch --export=TAG=al_round01,CHECKPOINT=...`
# so a re-run against a new round's promoted checkpoint doesn't clobber an
# earlier round's results/hvsc1_benchmark*.npz.
TAG=${TAG:-}
CHECKPOINT=${CHECKPOINT:-$REPO_ROOT/trained_models/hvsc1_community_input-27_output-27_hidden-128.pt}

EXTRA_ARGS=""
if [ -n "$TAG" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --tag $TAG"
fi

echo "Started at: $(date)"
echo "Tag        : ${TAG:-<none>}"
echo "Checkpoint : $CHECKPOINT"
$VENV/bin/python $SCRIPT --checkpoint "$CHECKPOINT" $EXTRA_ARGS
echo "Finished at: $(date)"
