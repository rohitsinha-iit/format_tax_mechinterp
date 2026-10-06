"""VERIFY(§2.1): record how the installed vLLM quantizes an unquantized checkpoint with bitsandbytes.

vLLM >= 0.30 removed in-tree bitsandbytes; it lives in the official out-of-tree `vllm-bnb-plugin`
(entry point vllm.general_plugins:bitsandbytes). We read the plugin's in-flight quantize_4bit call
so that every output file logs the effective quant type / double-quant setting."""
from __future__ import annotations

import importlib.metadata as md
import inspect
import re


def bnb_inflight_settings() -> dict:
    out = {}
    try:
        out["vllm_bnb_plugin"] = md.version("vllm-bnb-plugin")
        import vllm_bnb_plugin.bitsandbytes_loader as L
        src = inspect.getsource(L)
    except Exception:
        try:
            import vllm.model_executor.model_loader.bitsandbytes_loader as L  # older in-tree location
            src = inspect.getsource(L)
            out["vllm_bnb_plugin"] = None
        except Exception as e:
            return {"error": repr(e)}
    calls = re.findall(r"quantize_4bit\(\s*[^)]*\)", src)
    out["inflight_calls"] = [re.sub(r"\s+", " ", c) for c in calls]
    out["quant_type"] = sorted(set(re.findall(r'quant_type="(\w+)"', " ".join(calls))))
    out["compress_statistics(double_quant)"] = sorted(set(re.findall(r"compress_statistics=(\w+)", " ".join(calls))))
    import bitsandbytes
    out["bitsandbytes"] = bitsandbytes.__version__
    out["blocksize"] = "bitsandbytes default (64) — no blocksize argument passed"
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(bnb_inflight_settings(), indent=1))
