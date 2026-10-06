#!/usr/bin/env bash
# Writes reports/env.txt and requirements.lock (§2).
set -euo pipefail
PY=${PY:-$HOME/ftmi-venv/bin/python}
cd "$(dirname "$0")/.."
mkdir -p reports
{
  echo "# format-tax-mi environment — $(date -Is)"
  echo "## host"; uname -a; grep PRETTY_NAME /etc/os-release; echo "WSL: $(uname -r)"
  echo "## python"; $PY --version
  echo "## key packages"
  $PY - <<'EOF'
import importlib
for m in ["torch", "vllm", "transformers", "bitsandbytes", "accelerate", "datasets", "math_verify",
          "numpy", "scipy", "sklearn", "statsmodels", "pandas", "matplotlib", "safetensors", "xgrammar"]:
    try:
        mod = importlib.import_module(m)
        import importlib.metadata as md
        v = getattr(mod, "__version__", None)
        if v is None:
            try: v = md.version({"sklearn": "scikit-learn", "math_verify": "math-verify"}.get(m, m))
            except Exception: v = "?"
        print(f"{m}=={v}")
    except Exception as e:
        print(f"{m}: NOT IMPORTABLE ({e!r})")
import torch
print("torch.version.cuda =", torch.version.cuda, "| cuda available =", torch.cuda.is_available(),
      "| device =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "| capability =", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
      "| bf16 =", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
EOF
  echo "## nvidia-smi"; nvidia-smi
  echo "## git"; git rev-parse HEAD 2>/dev/null || echo nogit
} > reports/env.txt 2>&1
"$HOME/uvtool/bin/uv" pip freeze --python "$PY" > requirements.lock
echo "wrote reports/env.txt and requirements.lock"
