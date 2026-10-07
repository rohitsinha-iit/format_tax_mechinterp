"""E3 steering: vectors at L* and residual-stream hooks for HF and vLLM (docs/E3_plan.md).

The steering vector is added to the residual stream after block L* (= output of model.model.layers[L*-1]),
at every token position (scope "all") or prompt tokens only (scope "prompt", HF only).

  python -m src.steer --model qwen3-4b --stage vectors                 # results/{tag}/e3_vectors.safetensors
  python -m src.steer --model qwen3-4b --stage manip                   # HF manipulation check on dev set
  python -m src.steer --model qwen3-4b --stage parity-vllm --gpu-mem 0.80
  python -m src.steer --model qwen3-4b --stage parity-hf               # -> reports/e3_steer_parity_{tag}.md
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from src.common import REPORTS, RESULTS, log, model_info, now

N_RANDOM = 5
RANDOM_SEED0 = 100
PARITY_SPECS = [("JS", "minus_b", 1.0), ("FF", "plus_b", 1.0)]


# ----------------------------------------------------------------------------- vectors

def frozen(tag: str) -> dict:
    return json.loads((RESULTS / tag / "e1_frozen.json").read_text())


def build_vectors(tag: str) -> dict:
    """Unit directions at L*: b̂ (frozen in E1), f̂, random r̂_s ⊥ {b̂, f̂}; plus k = f·b̂ and f itself."""
    fz = frozen(tag)
    Ls = fz["L_star"]
    b_hat = load_file(str(RESULTS / tag / "e1_b_hat.safetensors"))["b_hat"].double()
    f = load_file(str(RESULTS / tag / "e1_directions_last.safetensors"))["f"][Ls].double()
    k = float(f @ b_hat)
    f_hat = f / f.norm()
    out = {"b_hat": b_hat, "f": f}
    q1 = b_hat
    q2 = f_hat - (f_hat @ q1) * q1
    q2 = q2 / q2.norm()
    for s in range(N_RANDOM):
        g = torch.from_numpy(np.random.default_rng(RANDOM_SEED0 + s).standard_normal(b_hat.shape[0]))
        g = g - (g @ q1) * q1 - (g @ q2) * q2
        out[f"r{s}"] = g / g.norm()
    meta = {"L_star": Ls, "k": k, "norm_f": float(f.norm()), "random_seeds": [RANDOM_SEED0 + s for s in range(N_RANDOM)],
            "b_hat_sha256": fz["b_hat_sha256"], "created": now()}
    return out, meta


def save_vectors(tag: str):
    vecs, meta = build_vectors(tag)
    p = RESULTS / tag / "e3_vectors.safetensors"
    save_file({k: v.float().contiguous() for k, v in vecs.items()}, str(p), metadata={"meta": json.dumps(meta)})
    log(f"saved {p}: {meta}")
    return vecs, meta


def load_vectors(tag: str):
    from safetensors import safe_open
    p = RESULTS / tag / "e3_vectors.safetensors"
    if not p.exists():
        return save_vectors(tag)
    with safe_open(str(p), "pt") as fh:
        meta = json.loads(fh.metadata()["meta"])
    return {k: v.double() for k, v in load_file(str(p)).items()}, meta


def steer_vector(tag: str, name: str, alpha: float) -> torch.Tensor | None:
    """Named steering vectors (float32, on CPU). none | minus_b | plus_b | minus_r{s} | plus_r{s} | minus_f"""
    if name == "none" or alpha == 0:
        return None
    vecs, meta = load_vectors(tag)
    k = meta["k"]
    sign = -1.0 if name.startswith("minus_") else 1.0
    base = name.split("_", 1)[1]
    if base == "b":
        v = sign * alpha * k * vecs["b_hat"]
    elif base.startswith("r"):
        v = sign * alpha * k * vecs[base]          # same norm as the b̂ vector
    elif base == "f":
        v = sign * alpha * vecs["f"]
    else:
        raise ValueError(name)
    return v.float()


# ----------------------------------------------------------------------------- HF hook

class HFSteer:
    """Adds `vec` to the output of model.model.layers[L*-1]. Register BEFORE any capture hooks on that layer."""

    def __init__(self, model, layer_idx: int, scope: str = "all"):
        self.layer = model.model.layers[layer_idx - 1]
        self.vec, self.scope, self.calls = None, scope, 0
        self.h = self.layer.register_forward_hook(self._fn)

    def set(self, vec: torch.Tensor | None):
        self.vec = None if vec is None else vec.to(next(self.layer.parameters()).device)

    def _fn(self, mod, inp, out):
        if self.vec is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        if self.scope == "prompt" and h.shape[1] == 1:   # decode step under KV cache: leave untouched
            return None
        self.calls += 1
        h = h + self.vec.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def remove(self):
        self.h.remove()


# ----------------------------------------------------------------------------- vLLM hook
# vLLM's Qwen3DecoderLayer returns (mlp_out, residual); the true stream is mlp_out + residual (the add is fused
# into the next layer's RMSNorm). Adding the vector to `residual` therefore adds it to the stream. Hooks fire
# because the engine runs with enforce_eager=True. The engine must run in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0)
# so that apply_model can pass these closures without pickling.

def vllm_install(llm, layer_idx: int):
    def fn(model):
        layer = model.model.layers[layer_idx - 1]
        if getattr(layer, "_ftmi_hook", None) is None:
            def hook(mod, inp, out):
                v = getattr(mod, "_ftmi_vec", None)
                if v is None:
                    return None
                mod._ftmi_tokens += out[1].shape[0]
                return out[0], out[1] + v.to(out[1].dtype)
            layer._ftmi_vec, layer._ftmi_tokens = None, 0
            layer._ftmi_hook = layer.register_forward_hook(hook)
        return type(layer).__name__
    return llm.apply_model(fn)


def vllm_set(llm, layer_idx: int, vec: torch.Tensor | None):
    def fn(model):
        layer = model.model.layers[layer_idx - 1]
        dev = next(p.device for p in layer.parameters())
        layer._ftmi_vec = None if vec is None else vec.to(dev)
        layer._ftmi_tokens = 0
        return True
    return llm.apply_model(fn)


def vllm_tokens_steered(llm, layer_idx: int) -> int:
    return llm.apply_model(lambda m: m.model.layers[layer_idx - 1]._ftmi_tokens)[0]


# ----------------------------------------------------------------------------- manipulation check (HF, dev set)

def manip_check(model_key: str, n: int = 20):
    """Projection of r_l (last prompt token) on b̂_l = unit(b_l) for l = 0..L, under: FF, JS, JS−b, C1, FF+b.
    Expectation: at l = L* JS−b sits ≈ k below JS (≈ FF) by construction; the question is whether the gap persists
    downstream (l > L*), i.e. whether later layers re-impose the format-induced brevity component."""
    from src import prompting as P
    from src.acts_hf import ResidCapture
    from src.data import load_dev_set
    from src.hf_model import load_hf
    info = model_info(model_key)
    tag = info["tag"]
    Ls = frozen(tag)["L_star"]
    b_all = load_file(str(RESULTS / tag / "e1_directions_last.safetensors"))["b"]
    b_units = (b_all / b_all.norm(dim=-1, keepdim=True).clamp_min(1e-9)).float()  # [L+1, d]
    model, tok, _ = load_hf(model_key)
    steer = HFSteer(model, Ls)           # registered first so the capture hook on layer L* sees the steered output
    cap = ResidCapture(model)
    rows = load_dev_set()[:n]
    settings = [("FF", "none"), ("JS", "none"), ("JS", "minus_b"), ("C1", "none"), ("FF", "plus_b"),
                ("JS", "minus_r0")]
    res = {}
    for cond, vname in settings:
        steer.set(steer_vector(tag, vname, 1.0))
        proj = []
        for r in rows:
            ids = P.build_prompt(tok, cond, r["problem"])["prompt_ids"]
            acts = cap.run(ids, lambda h: h[-1])                 # [L+1, d]
            proj.append((acts * b_units).sum(-1))
        res[f"{cond}|{vname}"] = torch.stack(proj).mean(0)       # [L+1]
        log(f"manip {cond}|{vname}: proj at L*={float(res[f'{cond}|{vname}'][Ls]):.2f}, "
            f"L={float(res[f'{cond}|{vname}'][-1]):.2f}")
    steer.set(None)
    L = b_all.shape[0] - 1
    js, jsb, ff = res["JS|none"], res["JS|minus_b"], res["FF|none"]
    gap = js - ff
    recovered = [(float((js[l] - jsb[l]) / gap[l]) if abs(float(gap[l])) > 1e-6 else None) for l in range(L + 1)]
    out = {"tag": tag, "L_star": Ls, "n_dev_prompts": n, "alpha": 1.0, "k": load_vectors(tag)[1]["k"],
           "mean_projection_on_unit_b_l": {k: [round(float(x), 3) for x in v] for k, v in res.items()},
           "fraction_of_JS_FF_gap_removed_by_minus_b": recovered}
    (RESULTS / tag / "e3_manip_check.json").write_text(json.dumps(out, indent=1))
    for l in [Ls, Ls + 2, Ls + 5, Ls + 9, L]:
        if l <= L:
            log(f"  l={l:2d}  FF={float(ff[l]):7.2f}  JS={float(js[l]):7.2f}  JS-b={float(jsb[l]):7.2f}  "
                f"FF+b={float(res['FF|plus_b'][l]):7.2f}  C1={float(res['C1|none'][l]):7.2f}  "
                f"JS-r0={float(res['JS|minus_r0'][l]):7.2f}  gap removed={recovered[l]}")
    cap.remove(); steer.remove()
    return out


# ----------------------------------------------------------------------------- steered parity

def _parity_prompts(tok, n=20):
    from src import prompting as P
    from src.data import load_dev_set
    rows = load_dev_set()[:n]
    return [dict(uid=r["uid"], cond=c, vname=v, alpha=a, **P.build_prompt(tok, c, r["problem"]))
            for (c, v, a) in PARITY_SPECS for r in rows]


def parity_vllm(model_key: str, eargs):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    eargs.no_prefix_caching = True
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from src.gen_vllm import make_llm
    info = model_info(model_key)
    tag = info["tag"]
    Ls = frozen(tag)["L_star"]
    tok = AutoTokenizer.from_pretrained(info["hf_id"])
    items = _parity_prompts(tok)
    llm, ekw = make_llm(info, eargs)
    layer_type = vllm_install(llm, Ls)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=0)
    res, token_counts = [], {}
    for (c, v, a) in PARITY_SPECS + [("JS", "none", 0.0)]:
        group = [it for it in items if (it["cond"], it["vname"]) == (c, v)] or \
                [dict(it, vname="none") for it in items if it["cond"] == c and it["vname"] == "minus_b"]
        vllm_set(llm, Ls, steer_vector(tag, v, a))
        outs = llm.generate([{"prompt_token_ids": it["prompt_ids"]} for it in group], sp, use_tqdm=False)
        token_counts[f"{c}|{v}"] = {"steered_tokens": vllm_tokens_steered(llm, Ls),
                                    "prompt_tokens": sum(it["prompt_n_tokens"] for it in group)}
        for it, o in zip(group, outs):
            lp = o.outputs[0].logprobs[0]
            top = sorted(((tid, l.logprob) for tid, l in lp.items()), key=lambda x: -x[1])[:20]
            res.append({"uid": it["uid"], "cond": c, "vname": v, "top": top, "prompt_sha256": it["prompt_sha256"]})
    vllm_set(llm, Ls, None)
    out = {"engine": ekw, "layer_type": layer_type, "token_counts": token_counts, "items": res}
    (RESULTS / tag / "e3_parity_vllm.json").write_text(json.dumps(out))
    log(f"vLLM steered logprobs saved; hook token counts {token_counts}")


def parity_hf(model_key: str):
    from src.hf_model import load_hf
    info = model_info(model_key)
    tag = info["tag"]
    Ls = frozen(tag)["L_star"]
    v = json.loads((RESULTS / tag / "e3_parity_vllm.json").read_text())
    model, tok, _ = load_hf(model_key)
    steer = HFSteer(model, Ls)
    rows = []
    by_key = {}
    for r in v["items"]:
        by_key.setdefault((r["cond"], r["vname"]), []).append(r)
    from src import prompting as P
    from src.data import load_dev_set
    probs = {r["uid"]: r["problem"] for r in load_dev_set()[:20]}
    with torch.inference_mode():
        for (c, vname), group in by_key.items():
            steer.set(steer_vector(tag, vname, 1.0))
            for r in group:
                pr = P.build_prompt(tok, c, probs[r["uid"]])
                assert pr["prompt_sha256"] == r["prompt_sha256"]
                logits = model(input_ids=torch.tensor([pr["prompt_ids"]], device=model.device)).logits[0, -1].float()
                lp = torch.log_softmax(logits, -1)
                ids = [t for t, _ in r["top"]]
                pv = torch.softmax(torch.tensor([l for _, l in r["top"]], dtype=torch.float64), 0)
                ph = torch.softmax(lp[ids].double().cpu(), 0)
                rows.append({"cond": c, "vname": vname, "uid": r["uid"], "agree": int(lp.argmax()) == ids[0],
                             "kl": float((pv * (pv.log() - ph.log())).sum())})
    steer.set(None); steer.remove()
    summ = {}
    for key in by_key:
        rr = [x for x in rows if (x["cond"], x["vname"]) == key]
        summ["|".join(key)] = {"n": len(rr), "top1_agreement": sum(x["agree"] for x in rr) / len(rr),
                               "mean_kl": sum(x["kl"] for x in rr) / len(rr)}
    steered = [x for x in rows if x["vname"] != "none"]
    agree = sum(x["agree"] for x in steered) / len(steered)
    rep = {"tag": tag, "L_star": Ls, "by_setting": summ, "steered_top1_agreement": agree, "PASS": agree >= 0.90,
           "hook_token_counts_vllm": v["token_counts"], "rows": rows}
    (RESULTS / tag / "e3_parity.json").write_text(json.dumps(rep, indent=1))
    md = [f"# E3 steered parity (vLLM hook vs HF hook): {tag}", "",
          f"- L* = {Ls}; 20 dev prompts per setting; next-token distribution at the last prompt position", "",
          "| setting | n | top-1 agreement | mean KL (top-20, renormalised) |", "|---|---|---|---|"]
    md += [f"| {k} | {s['n']} | {s['top1_agreement']:.3f} | {s['mean_kl']:.5f} |" for k, s in summ.items()]
    md += ["", f"- steered top-1 agreement {agree:.3f} (threshold 0.90) → **{'PASS' if rep['PASS'] else 'FAIL'}**",
           f"- vLLM hook token counts (steered vs prompt tokens submitted): {v['token_counts']}"]
    (REPORTS / f"e3_steer_parity_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    from src.gen_vllm import add_engine_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    ap.add_argument("--stage", required=True, choices=["vectors", "manip", "parity-vllm", "parity-hf"])
    add_engine_args(ap)
    a = ap.parse_args()
    tag = model_info(a.model)["tag"]
    if a.stage == "vectors":
        save_vectors(tag)
    elif a.stage == "manip":
        manip_check(a.model)
    elif a.stage == "parity-vllm":
        parity_vllm(a.model, a)
    else:
        parity_hf(a.model)
