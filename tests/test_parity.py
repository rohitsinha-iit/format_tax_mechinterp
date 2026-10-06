"""§2 parity check: next-token distribution at the last prompt position, vLLM vs HF (both NF4).

Two separate processes (never co-load two models):
  python -m tests.test_parity --model qwen3-4b --stage vllm   -> results/{tag}/parity_vllm.json
  python -m tests.test_parity --model qwen3-4b --stage hf     -> results/{tag}/parity.json + reports/parity_{tag}.md
  pytest tests/test_parity.py   (asserts top-1 agreement >= 0.90 for every model that has a parity.json)
"""
from __future__ import annotations

import argparse
import json
import math

import pytest

from src.common import REPORTS, RESULTS, MODEL_KEYS, model_info

N_PROMPTS = 20
CONDS = ("FF", "JS")
TOPK = 20


def _prompts(tok):
    from src import prompting as P
    from src.data import load_math500
    rows = load_math500()[:N_PROMPTS]
    return [dict(uid=r["uid"], cond=c, **P.build_prompt(tok, c, r["problem"])) for r in rows for c in CONDS]


def stage_vllm(model: str):
    from vllm import SamplingParams
    from transformers import AutoTokenizer
    from src.gen_vllm import make_llm
    info = model_info(model)
    tok = AutoTokenizer.from_pretrained(info["hf_id"])
    items = _prompts(tok)
    llm, ekw = make_llm(info)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=TOPK, seed=0)
    outs = llm.generate([{"prompt_token_ids": it["prompt_ids"]} for it in items], sp)
    res = []
    for it, o in zip(items, outs):
        lp = o.outputs[0].logprobs[0]  # dict token_id -> Logprob
        top = sorted(((tid, l.logprob) for tid, l in lp.items()), key=lambda x: -x[1])[:TOPK]
        res.append({"uid": it["uid"], "cond": it["cond"], "prompt_sha256": it["prompt_sha256"],
                    "top": top, "sampled": o.outputs[0].token_ids[0]})
    out = RESULTS / info["tag"] / "parity_vllm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"engine": ekw, "items": res}, indent=0))
    print("wrote", out)


def stage_hf(model: str):
    import torch
    from src.hf_model import describe, load_hf
    m, tok, info = load_hf(model)
    items = _prompts(tok)
    v = json.loads((RESULTS / info["tag"] / "parity_vllm.json").read_text())["items"]
    vmap = {(r["uid"], r["cond"]): r for r in v}
    rows = []
    with torch.inference_mode():
        for it in items:
            vr = vmap[(it["uid"], it["cond"])]
            assert vr["prompt_sha256"] == it["prompt_sha256"], "prompt mismatch between stages"
            ids = torch.tensor([it["prompt_ids"]], device="cuda")
            logits = m(input_ids=ids).logits[0, -1].float()
            lp = torch.log_softmax(logits, -1)
            hf_top1 = int(lp.argmax())
            v_ids = [t for t, _ in vr["top"]]
            v_lp = torch.tensor([l for _, l in vr["top"]], dtype=torch.float64)
            h_lp = lp[v_ids].double().cpu()
            # KL(vLLM || HF) over vLLM's top-20 support, both renormalised on that support
            pv = torch.softmax(v_lp, 0)
            ph = torch.softmax(h_lp, 0)
            kl = float((pv * (pv.log() - ph.log())).sum())
            rows.append({"uid": it["uid"], "cond": it["cond"], "vllm_top1": v_ids[0], "hf_top1": hf_top1,
                         "agree": v_ids[0] == hf_top1, "kl_top20": kl,
                         "vllm_top1_lp": float(v_lp[0]), "hf_lp_of_vllm_top1": float(h_lp[0]),
                         "abs_lp_diff_top1": abs(float(v_lp[0]) - float(h_lp[0]))})
    agree = sum(r["agree"] for r in rows) / len(rows)
    mean_kl = sum(r["kl_top20"] for r in rows) / len(rows)
    rep = {"tag": info["tag"], "n": len(rows), "top1_agreement": agree, "mean_kl_top20": mean_kl,
           "max_kl_top20": max(r["kl_top20"] for r in rows),
           "by_cond": {c: sum(r["agree"] for r in rows if r["cond"] == c) / sum(r["cond"] == c for r in rows) for c in CONDS},
           "hf_model": describe(m), "PASS": agree >= 0.90, "rows": rows}
    (RESULTS / info["tag"] / "parity.json").write_text(json.dumps(rep, indent=1, default=str))
    md = [f"# vLLM vs HF parity: {info['tag']}", "",
          f"- prompts: {N_PROMPTS} MATH-500 problems x {CONDS} = {len(rows)}",
          f"- top-1 agreement: {agree:.3f} (threshold 0.90) -> {'PASS' if rep['PASS'] else 'FAIL (STOP)'}",
          f"- mean KL(vLLM||HF) over vLLM top-{TOPK} (renormalised): {mean_kl:.5f}; max {rep['max_kl_top20']:.5f}",
          f"- by condition: {rep['by_cond']}"]
    (REPORTS / f"parity_{info['tag']}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_parity(model):
    p = RESULTS / model_info(model)["tag"] / "parity.json"
    if not p.exists():
        pytest.skip(f"no parity run for {model}")
    rep = json.loads(p.read_text())
    assert rep["top1_agreement"] >= 0.90, rep["top1_agreement"]
    assert math.isfinite(rep["mean_kl_top20"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_KEYS))
    ap.add_argument("--stage", required=True, choices=["vllm", "hf"])
    a = ap.parse_args()
    stage_vllm(a.model) if a.stage == "vllm" else stage_hf(a.model)
