#!/usr/bin/env bash
# Recreate the environment described for the ZeRO Stage 3 + torch.func issue
# (LinearFunctionForZeroStage3 / functorch transforms).
#
# Reference (reporter setup):
#   OS:        Ubuntu 22.04
#   Python:    3.11
#   PyTorch:   2.8.0+cu128
#   CUDA:      12.8 (via PyTorch wheels; full toolkit optional for building DS ops)
#   GPU:       H100 (any CUDA GPU works for repro_zero3_functorch.py)
#   DeepSpeed: issue used wheel 0.16.4; this script installs THIS checkout in editable mode.
#
# Repro:  source .venv-repro-zero3-functorch/bin/activate && python repro_zero3_functorch.py
#
# Optional env overrides:
#   VENV=path/to/venv
#   PYTHON=python3.11
#   TORCH_VERSION=2.8.0
#   TORCH_CUDA=cu128          # PyTorch wheel channel (cu128, cu126, cpu, ...)
#   DS_BUILD_OPS=0            # skip compiling DeepSpeed CUDA extensions (faster setup; ZeRO-3 linear path is Python)
#   SKIP_EDITABLE_DS=1        # only install deps + torch, skip pip install -e .

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-${ROOT}/.venv-repro-zero3-functorch}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"

pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "$PYTHON"
    return
  fi
  if command -v python3.11 &>/dev/null; then
    echo "python3.11"
    return
  fi
  local py
  py="$(command -v python3)"
  local ver
  ver="$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$ver" != "3.11" ]]; then
    echo "Expected Python 3.11 (issue reporter). Found ${py} -> ${ver}. Set PYTHON=python3.11." >&2
    exit 1
  fi
  echo "$py"
}

PY="$(pick_python)"
if ! "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), "need 3.11"' 2>/dev/null; then
  echo "Python 3.11 required. Current: $("$PY" -V)" >&2
  exit 1
fi

echo "==> Using interpreter: $PY ($("$PY" -V))"
echo "==> venv: ${VENV}"
echo "==> PyTorch: ${TORCH_VERSION}+${TORCH_CUDA} (index ${TORCH_INDEX})"

if [[ ! -d "$VENV" ]]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if [[ "${TORCH_CUDA}" == "cpu" ]]; then
  echo "==> Installing PyTorch (CPU). repro_zero3_functorch.py uses device='cuda' — use TORCH_CUDA=cu128 on a GPU machine." >&2
fi
pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"

# Install DeepSpeed runtime deps; keep the cu128 torch already installed.
pip install -r "${ROOT}/requirements/requirements.txt"

if [[ "${SKIP_EDITABLE_DS:-0}" != "1" ]]; then
  # Optional: DS_BUILD_OPS=0 ./setup_repro_zero3_functorch_env.sh to skip compiling CUDA extensions
  pip install -e "${ROOT}"
else
  echo "==> SKIP_EDITABLE_DS=1: not running pip install -e . (activate venv and install manually if needed)"
fi

echo ""
echo "Environment ready. Activate with:"
echo "  source ${VENV}/bin/activate"
echo "Run repro:"
echo "  cd ${ROOT} && python repro_zero3_functorch.py"
echo "Optional sanity check (matches issue ds_report style):"
echo "  ds_report"

if [[ "${1:-}" == "--run" ]]; then
  echo ""
  echo "==> Running repro_zero3_functorch.py ..."
  cd "$ROOT"
  python repro_zero3_functorch.py
fi
