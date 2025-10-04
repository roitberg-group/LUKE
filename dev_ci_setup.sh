#!/usr/bin/env bash
set -euo pipefail

# Local developer helper to mirror CI as closely as possible.
# Usage: bash ./dev_ci_setup.sh
# Ensures torch (CPU) + torchani editable + project extras installed.
# On macOS the '+cpu' wheel variant does not exist; we install plain torch.

PYTHON_BIN=${PYTHON_BIN:-python3.11}
TORCH_VERSION=${TORCH_VERSION:-2.3.1}
export TORCHANI_NO_WARN_EXTENSIONS=1

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# On Linux we can use the cpu-only index + '+cpu' tag.
UNAME_S=$(uname -s)
case "$UNAME_S" in
  Linux)
    export PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}
    echo "Installing CPU-only torch ${TORCH_VERSION}+cpu from $PYTORCH_INDEX_URL"
    # Some versions may not have +cpu tag; fallback gracefully to plain version.
    if ! pip install --index-url "$PYTORCH_INDEX_URL" "torch==${TORCH_VERSION}+cpu" 2>/dev/null; then
      echo "+cpu variant not found for ${TORCH_VERSION}; falling back to torch==${TORCH_VERSION}"
      pip install "torch==${TORCH_VERSION}"
    fi
    ;;
  Darwin)
    echo "macOS detected — installing torch==${TORCH_VERSION} (no +cpu suffix)"
    pip install "torch==${TORCH_VERSION}"
    ;;
  *)
    echo "Unknown platform ($UNAME_S); attempting plain torch pin"
    pip install "torch==${TORCH_VERSION}"
    ;;
esac

# Editable torchani (with deps) then project extras
pip install -e external/torchani
pip install -e .[chem,dev]

python -c "import torch, torchani; from torchani.tuples import SpeciesCoordinates; print('OK torch', torch.__version__, 'torchani', torchani.__version__);"

echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Run quality gates: ruff check luke tests && mypy luke && pytest --disable-warnings"