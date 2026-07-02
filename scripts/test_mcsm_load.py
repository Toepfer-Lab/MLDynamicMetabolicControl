import sys
import os

print("Python:", sys.version)
print("CPLEX_HOME:", os.environ.get('CPLEX_HOME', 'NOT SET'))
print("PYTHONPATH:", os.environ.get('PYTHONPATH', 'NOT SET'))
print()

import micom
from micom import load_pickle
import cobra

cobra.Configuration.solver = 'cplex'

MODEL_DIR = "/home/jkaatz/MA/MLDynamicMetabolicControl/model/dcom.pickle"
print(f"Loading model from {MODEL_DIR} ...")
comm = load_pickle(MODEL_DIR)
print("Success! Community model loaded.")
print(comm)
