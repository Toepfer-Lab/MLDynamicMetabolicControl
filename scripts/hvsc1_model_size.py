"""
Print the authoritative reaction/metabolite count for the built hvsc1
(27-strain) MICOM community model.

No existing script reports this (unlike mcsm_explore.py for dcom.pickle),
so the hvsc1 community's true LP scale has never been documented anywhere.
Needed to place hvsc1 on the cross-scale complexity comparison as a
stub row (size only — the surrogate/benchmark columns stay "pending"
until the colleague's corrected barley model is ready).

The LP is not solved here — this is a pure size/structure read, no CPLEX
solve required — but micom/cobra still need CPLEX on the import path to
load the pickle, so this is run via sbatch like the other MICOM scripts.
"""

from pathlib import Path

from micom import load_pickle

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "model" / "hvsc1_comm.pickle"


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    section("Loading hvsc1 community model")
    comm = load_pickle(str(MODEL_PATH))

    taxa = list(comm.taxa)
    rxns = len(comm.reactions)
    mets = len(comm.metabolites)

    section("Model size")
    print(f"  Number of taxa    : {len(taxa)}")
    print(f"  Taxa              : {taxa}")
    print(f"  Total reactions   : {rxns:,}")
    print(f"  Total metabolites : {mets:,}")


if __name__ == "__main__":
    main()
