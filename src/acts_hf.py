"""Residual-stream activation dumps (§8.1–8.2) with HF NF4, batch size 1.

  python -m src.acts_hf --model qwen3-4b --set extract --what prompt   # FF,JS,NC,C1-3,D1-3 at P_last + span
  python -m src.acts_hf --model qwen3-4b --set extract --what gen16    # FF,JS teacher-forced, first 16 content tokens
  python -m src.acts_hf --model qwen3-4b --set eval                    # FF,JS at P_last, layer L* only (needs freeze)

r_l = output of decoder block l (forward hook on model.model.layers[l-1]); r_0 = token embeddings.
Saved tensors (float16) in acts/{tag}/{set}/:
  {COND}_span.safetensors  acts [n, L+1, S, d]  (S = post-instruction span; P_last = span[:, :, -1])
  {COND}_gen16.safetensors acts [n, L+1, d]
  eval: {COND}_Lstar_last.safetensors acts [n, d]
"""
from __future__ import annotations

import argparse
import json
import re
import time

import torch
from safetensors.torch import load_file, save_file

from src.common import ACTS, OUTPUTS, REPORTS, RESULTS, load_cfg, log, meta_record, model_info, read_jsonl
from src import prompting as P


class ResidCapture:
    """Hooks embed_tokens (r_0) and every decoder block (r_1..r_L); stores a slice of positions."""

    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self.L = len(self.layers)
        self.sel = None  # callable: [T, d] tensor -> captured tensor
        self.buf = [None] * (self.L + 1)
        self.out_type = None
        self.handles = [model.model.embed_tokens.register_forward_hook(self._hook(0))]
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._hook(i + 1)))

    def _hook(self, idx):
        def fn(mod, inp, out):
            if idx > 0 and self.out_type is None:
                self.out_type = type(out).__name__
            if self.sel is None:  # hooks idle (e.g. the reference output_hidden_states pass)
                return
            h = out[0] if isinstance(out, tuple) else out   # VERIFY: tuple (older) vs tensor (newer HF)
            assert h.dim() == 3 and h.shape[0] == 1, h.shape
            self.buf[idx] = self.sel(h[0]).detach().float().cpu()
        return fn

    @torch.inference_mode()
    def run(self, ids: list[int], sel):
        self.sel = sel
        x = torch.tensor([ids], device=self.model.device)
        self.model.model(input_ids=x, use_cache=False)  # base model only: no lm_head needed
        self.sel = None
        return torch.stack(self.buf, 0)  # [L+1, ...]

    def remove(self):
        for h in self.handles:
            h.remove()


def sanity_check(model, cap: ResidCapture, ids: list[int], logfile) -> dict:
    """Hooked r_l must equal output_hidden_states[l] for l < L, and differ at l = L (final norm)."""
    with torch.inference_mode():
        hs = model.model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True,
                         use_cache=False).hidden_states
    mine = cap.run(ids, lambda h: h[-1])
    diffs = [float((mine[l] - hs[l][0, -1].float().cpu()).abs().max()) for l in range(len(hs))]
    res = {"decoder_layer_output_type": cap.out_type, "n_hidden_states": len(hs), "L": cap.L,
           "max_abs_diff_l_lt_L": max(diffs[:-1]), "abs_diff_at_L_vs_hidden_states[-1]": diffs[-1]}
    log(f"hook sanity: {res}", logfile)
    assert res["max_abs_diff_l_lt_L"] == 0.0, res
    assert diffs[-1] > 0, "hidden_states[-1] should be post-norm and differ from block-L output"
    return res


def _save(path, acts: torch.Tensor, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = acts.half()
    # residual streams can carry very large outlier dims; fp16 must not overflow (max 65504)
    assert torch.isfinite(h).all(), f"fp16 overflow: max |act| = {float(acts.abs().max())}"
    meta = dict(meta, max_abs_act=float(acts.abs().max()))
    save_file({"acts": h.contiguous()}, str(path),
              metadata={"meta": json.dumps(meta_record(**meta)["_meta"], default=str)})


def load_acts(path) -> tuple[torch.Tensor, dict]:
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        meta = json.loads(f.metadata()["meta"])
    return load_file(str(path))["acts"], meta


def dump_prompt(model, tok, cap, info, rows, conds, outdir, logfile):
    for c in conds:
        path = outdir / f"{c}_span.safetensors"
        if path.exists():
            log(f"skip {path} (exists)", logfile)
            continue
        out, uids, shas, span_ref, t0 = [], [], [], None, time.time()
        for r in rows:
            pr = P.build_prompt(tok, c, r["problem"])
            start, span = P.post_instruction_span(tok, pr["prompt"])
            span_ref = span_ref or span
            assert span == span_ref and pr["prompt_ids"][start:] == span
            S = len(span)
            out.append(cap.run(pr["prompt_ids"], lambda h, S=S: h[-S:]))
            uids.append(r["uid"]); shas.append(pr["prompt_sha256"])
        acts = torch.stack(out, 0)  # [n, L+1, S, d]
        _save(path, acts, {"model": info["hf_id"], "condition": c, "uids": uids, "prompt_sha256": shas,
                           "span_ids": span_ref, "span_tokens": tok.convert_ids_to_tokens(span_ref),
                           "shape": list(acts.shape), "position": "post-instruction span; P_last = [:, :, -1]"})
        log(f"saved {path} {tuple(acts.shape)} in {time.time() - t0:.0f}s", logfile)


def _char_starts(tok, text: str, ids: list[int]) -> tuple[list[int], str]:
    """Character start offset of each generated token within `text`."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] == ids:
        return [a for a, _ in enc["offset_mapping"]], "offsets_mapping"
    # generated ids are not the canonical re-tokenisation: fall back to prefix decoding
    return [len(tok.decode(ids[:k])) for k in range(len(ids))], "prefix_decode"


def dump_gen16(model, tok, cap, info, rows, outdir, logfile):
    e1 = load_cfg("e1")["gen16"]
    n_ct = e1["n_content_tokens"]
    scaffold = re.compile(e1["js_scaffold_regex"])
    excl_report = {}
    for c in e1["conditions"]:
        path = outdir / f"{c}_gen16.safetensors"
        if path.exists():
            log(f"skip {path} (exists)", logfile)
            continue
        gens = {r["uid"]: r for r in read_jsonl(OUTPUTS / info["tag"] / f"e1_ext80_{c}.jsonl")}
        out, uids, excluded, offs_method = [], [], {"no_scaffold": [], "too_few_content_tokens": [], "missing": []}, {}
        for r in rows:
            g = gens.get(r["uid"])
            if g is None:
                excluded["missing"].append(r["uid"]); continue
            pr = P.build_prompt(tok, c, r["problem"])
            assert pr["prompt_sha256"] == g["prompt_sha256"]
            gid = g["output_token_ids"]
            # drop a trailing EOS so it is never counted as content
            while gid and gid[-1] in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id):
                gid = gid[:-1]
            if c == "FF":
                k0 = 0
            else:
                m = scaffold.match(g["output_text"])
                if m is None:
                    excluded["no_scaffold"].append(r["uid"]); continue
                starts, how = _char_starts(tok, g["output_text"], gid)
                offs_method[how] = offs_method.get(how, 0) + 1
                cand = [k for k, s in enumerate(starts) if s >= m.end()]
                if not cand:
                    excluded["too_few_content_tokens"].append(r["uid"]); continue
                k0 = cand[0]
            if len(gid) - k0 < n_ct:
                excluded["too_few_content_tokens"].append(r["uid"]); continue
            a = len(pr["prompt_ids"]) + k0
            out.append(cap.run(pr["prompt_ids"] + gid, lambda h, a=a: h[a:a + n_ct].mean(0)))
            uids.append(r["uid"])
        acts = torch.stack(out, 0)
        _save(path, acts, {"model": info["hf_id"], "condition": c, "uids": uids, "excluded": excluded,
                           "offsets_method_counts": offs_method, "n_content_tokens": n_ct,
                           "shape": list(acts.shape)})
        excl_report[c] = {k: len(v) for k, v in excluded.items()} | {"kept": len(uids), "offsets": offs_method}
        log(f"saved {path} {tuple(acts.shape)}; exclusions {excl_report[c]}", logfile)
    if excl_report:
        p = REPORTS / f"gen16_exclusions_{info['tag']}.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(excl_report)
        p.write_text(json.dumps(old, indent=1))


def dump_eval(model, tok, cap, info, outdir, logfile):
    from src.data import load_math500
    fz = RESULTS / info["tag"] / "e1_frozen.json"
    if not fz.exists():
        raise SystemExit(f"{fz} missing: freeze L* on extract_set before touching eval activations (§8.4.5)")
    Ls = json.loads(fz.read_text())["L_star"]
    rows = load_math500()
    for c in ("FF", "JS"):
        path = outdir / f"{c}_Lstar_last.safetensors"
        if path.exists():
            log(f"skip {path}", logfile); continue
        out, uids, shas = [], [], []
        for r in rows:
            pr = P.build_prompt(tok, c, r["problem"])
            out.append(cap.run(pr["prompt_ids"], lambda h: h[-1])[Ls])
            uids.append(r["uid"]); shas.append(pr["prompt_sha256"])
        acts = torch.stack(out, 0)
        _save(path, acts, {"model": info["hf_id"], "condition": c, "uids": uids, "prompt_sha256": shas,
                           "layer": Ls, "position": "P_last", "frozen_file": str(fz)})
        log(f"saved {path} {tuple(acts.shape)} (layer {Ls})", logfile)


def main(a):
    from src.data import load_extract_set
    from src.hf_model import describe, load_hf
    info = model_info(a.model)
    logfile = REPORTS / "logs" / f"acts_{info['tag']}.log"
    model, tok, info = load_hf(a.model)
    d = describe(model)
    log(f"HF model: {d}", logfile)
    cap = ResidCapture(model)
    sanity = sanity_check(model, cap, P.build_prompt(tok, "FF", "What is $1+1$?")["prompt_ids"], logfile)
    (RESULTS / info["tag"]).mkdir(parents=True, exist_ok=True)
    (RESULTS / info["tag"] / "hf_model_info.json").write_text(json.dumps({"model": d, "hook_sanity": sanity}, indent=1, default=str))
    outdir = ACTS / info["tag"] / a.set
    if a.set == "eval":
        dump_eval(model, tok, cap, info, outdir, logfile)
    elif a.what == "prompt":
        conds = a.conds.split(",") if a.conds else load_cfg("e1")["extraction_conditions"]
        dump_prompt(model, tok, cap, info, load_extract_set(), conds, outdir, logfile)
    else:
        dump_gen16(model, tok, cap, info, load_extract_set(), outdir, logfile)
    cap.remove()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    ap.add_argument("--set", required=True, choices=["extract", "eval"])
    ap.add_argument("--what", default="prompt", choices=["prompt", "gen16"])
    ap.add_argument("--conds", default=None)
    main(ap.parse_args())
