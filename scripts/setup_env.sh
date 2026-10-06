#!/usr/bin/env bash
# Create the project venv (Python 3.11) inside WSL2 and install pinned-by-resolution deps.
set -euo pipefail
export UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4
VENV=${VENV:-$HOME/ftmi-venv}
if [ ! -x "$HOME/uvtool/bin/uv" ]; then
  python3 -m venv "$HOME/uvtool"
  "$HOME/uvtool/bin/pip" install -q uv
fi
UV="$HOME/uvtool/bin/uv"
if [ ! -x "$VENV/bin/python" ]; then
  "$UV" venv --python 3.11 --seed "$VENV"
fi
export VIRTUAL_ENV="$VENV"
# vllm pins its own torch (CUDA 12.x build); install it first so the rest resolves against it.
"$UV" pip install --python "$VENV/bin/python" vllm --torch-backend=auto
"$UV" pip install --python "$VENV/bin/python" \
  transformers accelerate bitsandbytes datasets math-verify numpy scipy scikit-learn \
  statsmodels pandas matplotlib safetensors pyyaml tqdm pytest
"$VENV/bin/python" -c "import torch, vllm, transformers, bitsandbytes; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print('vllm', vllm.__version__); print('transformers', transformers.__version__); print('bnb', bitsandbytes.__version__)"
