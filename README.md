# MLDynamicMetabolicControl

This repository will be used to attempt to recreate the paper "Linking intra- and extra-cellular metabolic domains via neural-network surrogates for dynamic metabolic control" by Sebastian Espinel-Rıos and Jose L. Avalos.

## Batch-friendly scripts
The notebook workflow was split into standalone scripts that can be launched locally or on RAMSES via `sbatch`. Run them from the repo root:

- Generate FBA data: `python scripts/generate_fba_data.py --vman PYK --condition anaerobic --n-samples 1000`
- Train surrogate NN: `python scripts/train_surrogate.py --vman PYK --condition anaerobic --hidden-dim 4 --epochs 5000`
- Simulate hybrid model: `python scripts/run_hybrid_simulation.py --checkpoint trained_models/<file>.pt --vman-value 5.0`
- Optimize vman: `python scripts/optimize_vman.py --checkpoint trained_models/<file>.pt --num-intervals 20 --log-trajectories`

Quick `sbatch` example (adjust modules/conda as needed):

```bash
sbatch --job-name fba_gen --wrap "python scripts/generate_fba_data.py --vman PYK --condition anaerobic"
```
