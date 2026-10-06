"""Qwen3-8B NF4 thinking-mode judge (§6.4) and self-consistency reruns (§6.5.1).

  python -m src.judge --model qwen3-4b --conds FF,JS,JC                 # judged items, seed 0
  python -m src.judge --model qwen3-4b --conds FF,JS,JC --selfcons      # 100 items x seeds 1,2
The judge model is always configs/models.yaml:judge_model; --model names the GENERATOR outputs.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict

import numpy as np

from src.common import OUTPUTS, REPORTS, ROOT, JsonlWriter, done_uids, load_cfg, log, model_info, read_jsonl, sha256
from src.score import JSON_CONDS, parse_json_output, repair_latex_escapes

JUDGE_TEMPLATE = (ROOT / "prompts" / "judge_prompt.txt").read_text(encoding="utf-8").rstrip("\n")
JUDGE_TEMPLATE_SHA = sha256(JUDGE_TEMPLATE)
_VERDICT = re.compile(r"VERDICT:\s*(EQUIVALENT|NOT_EQUIVALENT)")
_FIELD_KEYS = re.compile(r'"reasoning"\s*:|"answer"\s*:')


def fill(problem, gold, candidate_str, tail) -> str:
    vals = {"problem": problem, "gold": gold, "candidate_str": candidate_str, "tail": tail}
    # single pass, so braces/placeholders inside the values are never re-substituted
    return re.sub(r"\{(problem|gold|candidate_str|tail)\}", lambda m: vals[m.group(1)], JUDGE_TEMPLATE)


def make_tail(r: dict, n: int) -> str:
    text = r["output_text"] or ""
    if r["condition"] not in JSON_CONDS:
        return text[-n:]
    p = parse_json_output(text)
    if p["lenient_valid"]:
        reasoning, _ = repair_latex_escapes(p["obj"]["reasoning"])
        return reasoning[-n:]
    raw = _FIELD_KEYS.sub("", text)
    raw = re.sub(r'[{}"]', "", raw)
    raw = re.sub(r"\\n(?![A-Za-z])", "\n", raw)  # unescape \n (but keep LaTeX such as \neq, \nu)
    return raw[-n:]


def parse_verdict(text: str, finish_reason: str) -> dict:
    if finish_reason == "length":
        return {"judge_verdict": None, "judge_failed": True, "fail_reason": "length"}
    if "</think>" not in text:
        return {"judge_verdict": None, "judge_failed": True, "fail_reason": "no_think_close"}
    after = text.split("</think>", 1)[1]
    found = _VERDICT.findall(after)
    if not found:
        return {"judge_verdict": None, "judge_failed": True, "fail_reason": "no_verdict"}
    if len(set(found)) > 1:
        return {"judge_verdict": None, "judge_failed": True, "fail_reason": "conflicting_verdicts"}
    return {"judge_verdict": found[0], "judge_failed": False, "fail_reason": None}


def dataset_order():
    from src.data import load_math500
    m = load_math500()
    return {r["uid"]: i for i, r in enumerate(m)}, {r["uid"]: r for r in m}


def select_items(tag: str, cond: str, order: dict) -> list[dict]:
    """(a) every mv-incorrect item; (b) 10% of mv-correct items, fresh default_rng(1) per model x cond."""
    jc = load_cfg("e0")["judge"]
    rows = sorted(read_jsonl(OUTPUTS / tag / f"e0_{cond}_mv.jsonl"), key=lambda r: order[r["uid"]])
    wrong = [dict(r, role="mv_wrong") for r in rows if not r["mv_correct"]]
    right = [r for r in rows if r["mv_correct"]]
    k = int(round(jc["calib_frac"] * len(right)))
    rng = np.random.default_rng(jc["calib_seed"])
    pick = sorted(rng.choice(len(right), size=k, replace=False).tolist()) if k else []
    calib = [dict(right[i], role="calib") for i in pick]
    return wrong + calib


def build_request(tok, r: dict, problems: dict, max_model_len: int, max_tokens: int) -> dict:
    from src import prompting as P
    jc = load_cfg("e0")["judge"]
    cand = r["candidate"] if r["candidate"] is not None else "NONE"
    for n in (jc["tail_chars"], jc["tail_chars_fallback"]):
        tail = make_tail(r, n)
        text = fill(problems[r["uid"]]["problem"], r["gold"], cand, tail)
        rendered = P.render(tok, [{"role": "user", "content": text}], enable_thinking=True)
        ids = P.encode(tok, rendered)
        if len(ids) + max_tokens <= max_model_len:
            break
    return {"prompt_ids": ids, "prompt_n_tokens": len(ids), "tail_chars_used": n, "candidate_str": cand,
            "tail": tail, "judge_input_sha256": sha256(rendered),
            "fits": len(ids) + max_tokens <= max_model_len}


def run(args):
    from transformers import AutoTokenizer
    from src.gen_vllm import make_llm, sampling_params, stream_generate

    gen_info = model_info(args.model)
    tag = gen_info["tag"]
    judge_info = model_info(load_cfg("models")["judge_model"])
    jc = load_cfg("e0")["judge"]
    ecfg = load_cfg("models")["engine"]
    max_model_len = args.max_model_len or ecfg["max_model_len"]
    logfile = REPORTS / "logs" / f"judge_{tag}.log"
    tok = AutoTokenizer.from_pretrained(judge_info["hf_id"])
    order, problems = dataset_order()
    conds = args.conds.split(",")
    jdir = OUTPUTS / tag / "judge"

    # ---- work list
    jobs = []  # (seed, path, items)
    if not args.selfcons:
        for c in conds:
            path = jdir / f"e0_{c}_judge_seed{jc['seed']}.jsonl"
            items = select_items(tag, c, order)
            have = done_uids(path)
            log(f"judge {tag} {c}: {len(items)} items "
                f"({sum(i['role'] == 'mv_wrong' for i in items)} mv_wrong, "
                f"{sum(i['role'] == 'calib' for i in items)} calib); {len(have)} done", logfile)
            jobs.append((jc["seed"], path, [i for i in items if i["uid"] not in have]))
    else:
        pool = []
        for c in conds:
            pool += [dict(r, condition=c) for r in read_jsonl(jdir / f"e0_{c}_judge_seed0.jsonl")]
        sample = selfcons_sample(pool, jc["selfcons_n"], jc["selfcons_sample_seed"])
        mv = {c: {r["uid"]: r for r in read_jsonl(OUTPUTS / tag / f"e0_{c}_mv.jsonl")} for c in conds}
        items = [dict(mv[s["condition"]][s["uid"]], role=s["role"]) for s in sample]
        for s in jc["selfcons_rerun_seeds"]:
            path = jdir / f"selfcons_seed{s}.jsonl"
            have = {(r["condition"], r["uid"]) for r in read_jsonl(path)}
            jobs.append((s, path, [i for i in items if (i["condition"], i["uid"]) not in have]))
    if not any(items for _, _, items in jobs):
        log("nothing to judge", logfile)
        return

    class A:  # engine overrides, same knobs as generation
        max_num_seqs = args.max_num_seqs or judge_info["max_num_seqs"]
        gpu_mem = args.gpu_mem
    A.max_model_len = max_model_len
    llm, ekw = make_llm(judge_info, A, logfile)
    for seed, path, items in jobs:
        if not items:
            continue
        g = dict(temperature=jc["temperature"], top_p=jc["top_p"], top_k=jc["top_k"],
                 max_tokens=jc["max_tokens"], seed=seed)
        sp = sampling_params(g)
        reqs = []
        w = JsonlWriter(path, {"judge_model": judge_info["hf_id"], "generator_tag": tag, "seed": seed,
                               "judge_template_sha256": JUDGE_TEMPLATE_SHA, "engine": ekw,
                               "sampling_params": repr(sp), "selfcons": bool(args.selfcons)})
        for r in items:
            b = build_request(tok, r, problems, ekw["max_model_len"], jc["max_tokens"])
            base = {"uid": r["uid"], "condition": r["condition"], "generator_tag": tag, "role": r["role"],
                    "seed": seed, "mv_correct": r["mv_correct"], "candidate_str": b["candidate_str"],
                    "tail": b["tail"], "tail_chars_used": b["tail_chars_used"],
                    "prompt_n_tokens": b["prompt_n_tokens"], "judge_input_sha256": b["judge_input_sha256"]}
            if not b["fits"]:  # counted as judge_failed, never silently dropped
                w.write(dict(base, judge_verdict=None, judge_failed=True, fail_reason="prompt_too_long",
                             judge_output=None, judge_raw_tail=None, finish_reason=None, judge_output_tokens=0))
                continue
            reqs.append(dict(base, rid=f"{r['condition']}|{r['uid']}|{seed}", prompt_ids=b["prompt_ids"]))

        def on_done(it, out, wall, w=w):
            o = out.outputs[0]
            v = parse_verdict(o.text, o.finish_reason)
            rec = {k: v_ for k, v_ in it.items() if k not in ("rid", "prompt_ids")}
            w.write(dict(rec, **v, judge_output=o.text, judge_raw_tail=o.text[-200:],
                         finish_reason=o.finish_reason, judge_output_tokens=len(o.token_ids),
                         gen_wall_s=round(wall, 3)))

        log(f"judging {len(reqs)} items seed={seed} -> {path}", logfile)
        st = stream_generate(llm, reqs, lambda it, sp=sp: sp, on_done, max_inflight=2 * ekw["max_num_seqs"],
                             max_model_len=ekw["max_model_len"], logfile=logfile)
        w.close()
        log(f"done seed={seed}: {st}", logfile)
    if args.selfcons:
        selfcons_report(tag, conds)


def selfcons_sample(pool: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified by condition (proportional, largest remainder), default_rng(seed)."""
    from src.data import _largest_remainder
    pool = sorted(pool, key=lambda r: (r["condition"], r["uid"]))
    by = OrderedDict()
    for r in pool:
        by.setdefault(r["condition"], []).append(r)
    n = min(n, len(pool))
    q = _largest_remainder(OrderedDict((k, len(v)) for k, v in by.items()), n)
    rng = np.random.default_rng(seed)
    out = []
    for k, rows in by.items():
        idx = sorted(rng.choice(len(rows), size=q[k], replace=False).tolist())
        out += [rows[i] for i in idx]
    return out


def selfcons_report(tag: str, conds):
    jdir = OUTPUTS / tag / "judge"
    s0 = {}
    for c in conds:
        s0.update({(c, r["uid"]): r for r in read_jsonl(jdir / f"e0_{c}_judge_seed0.jsonl")})
    runs = [{(r["condition"], r["uid"]): r for r in read_jsonl(jdir / f"selfcons_seed{s}.jsonl")} for s in (1, 2)]
    keys = sorted(set(runs[0]) & set(runs[1]))
    lab = lambda r: r["judge_verdict"] if not r["judge_failed"] else "FAILED"
    triples = [(lab(s0[k]), lab(runs[0][k]), lab(runs[1][k])) for k in keys]
    all3 = sum(len(set(t)) == 1 for t in triples)
    pair = lambda i, j: sum(t[i] == t[j] for t in triples) / max(len(triples), 1)
    rep = {"generator_tag": tag, "n": len(triples), "three_way_agreement": all3 / max(len(triples), 1),
           "pairwise": {"s0_s1": pair(0, 1), "s0_s2": pair(0, 2), "s1_s2": pair(1, 2)},
           "by_condition": dict(Counter(k[0] for k in keys)),
           "label_counts_seed0": dict(Counter(t[0] for t in triples)),
           "n_any_failed": sum("FAILED" in t for t in triples)}
    (REPORTS / f"judge_selfcons_{tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"], help="generator whose outputs are judged")
    ap.add_argument("--conds", default="FF,JS,JC")
    ap.add_argument("--selfcons", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    a = ap.parse_args()
    if a.report_only:
        selfcons_report(model_info(a.model)["tag"], a.conds.split(","))
    else:
        run(a)
