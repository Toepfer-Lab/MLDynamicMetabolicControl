#!/bin/bash -l
# Creates a fresh Python 3.10 venv and registers a Jupyter kernel.
# Run AFTER: module load lang/Python/3.10.8-GCCcore-12.2.0
# Usage: bash scripts/setup_venv310.sh

set -e

PROJECT=/home/jkaatz/MA/MLDynamicMetabolicControl
VENV=$PROJECT/.venv

if ! python3 --version 2>&1 | grep -q "3\.10"; then
    echo "ERROR: Python 3.10 not active. Run: module load lang/Python/3.10.8-GCCcore-12.2.0"
    exit 1
fi

echo "Using: $(python3 --version)"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$PROJECT/requirements.txt"

"$VENV/bin/python" -m ipykernel install --user \
    --name=mlcontrol \
    --display-name "ML Control (.venv)"

echo ""
echo "Done. Kernel 'ML Control (.venv)' registered for $($VENV/bin/python --version)."
echo "Restart VS Code or reload the window to pick up the new kernel."
