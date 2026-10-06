"""HF NF4 model loading for activation extraction / parity (§2)."""
from __future__ import annotations

import inspect

import torch

from src.common import load_cfg, model_info


def bnb_config():
    from transformers import BitsAndBytesConfig
    h = load_cfg("models")["hf"]
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=h["bnb_4bit_quant_type"],
                              bnb_4bit_use_double_quant=h["bnb_4bit_use_double_quant"],
                              bnb_4bit_compute_dtype=getattr(torch, h["bnb_4bit_compute_dtype"]))


def load_hf(model_key: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    info = model_info(model_key)
    h = load_cfg("models")["hf"]
    kw = dict(quantization_config=bnb_config(), attn_implementation=h["attn_implementation"],
              device_map={"": 0})
    # transformers >= 4.56 renamed torch_dtype -> dtype
    params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    kw["dtype" if "dtype" in params or _accepts_dtype() else "torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(info["hf_id"], **kw)
    model.eval()
    tok = AutoTokenizer.from_pretrained(info["hf_id"])
    return model, tok, info


def _accepts_dtype() -> bool:
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return (major, minor) >= (4, 56)


def describe(model) -> dict:
    cfg = model.config
    qc = getattr(cfg, "quantization_config", None)
    return {"num_hidden_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
            "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
            "quantization_config": qc.to_dict() if hasattr(qc, "to_dict") else str(qc),
            "attn_implementation": getattr(cfg, "_attn_implementation", None),
            "n_layers_module": len(model.model.layers)}
