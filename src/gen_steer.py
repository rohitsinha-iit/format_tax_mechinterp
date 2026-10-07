"""E3 steered generation with vLLM (docs/E3_plan.md). Resumable, one steering setting at a time.

A setting is COND:VECTOR:ALPHA, e.g. JS:minus_b:1.0 (vector names in src/steer.py).
  python -m src.gen_steer --model qwen3-4b --set dev  --settings JS:none:0,JS:minus_b:1.0 --gpu-mem 0.80
  python -m src.gen_steer --model qwen3-4b --set eval --settings JS:minus_b:1.0,FF:plus_b:1.0 --gpu-mem 0.80
Outputs: outputs/{tag}/e3/{set}_{COND}_{VECTOR}_a{ALPHA}.jsonl (same record schema as E0, plus steering fields).
Engine runs in-process with prefix caching OFF (cached KV from another steering setting must never be reused).
"""
from __future__ import annotations

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json

from src.common import OUTPUTS, REPORTS, JsonlWriter, done_uids, load_cfg, log, model_info
from src import prompting as P


def setting_name(set_: str, cond: str, vname: str, alpha: float) -> str:
    return f"{set_}_{cond}_{vname}_a{alpha:g}"


def parse_settings(s: str) -> list[tuple[str, str, float]]:
    out = []
    for part in s.split(","):
        c, v, a = part.split(":")
        out.append((c, v, float(a)))
    return out


def items_for(set_: str, tok, cond: str) -> list[dict]:
    if set_ == "eval":
        from src.data import load_math500
        rows = load_math500()
    else:
        from src.data import load_dev_set
        from src.score import last_boxed
        rows = [dict(r, gold=last_boxed(r["solution"])) for r in load_dev_set()]
    return [{"rid": r["uid"], "uid": r["uid"], "subject": r["subject"], "level": r["level"], "gold": r["gold"],
             **P.build_prompt(tok, cond, r["problem"])} for r in rows]


def main(a):
    import vllm
    from transformers import AutoTokenizer
    from src.gen_vllm import make_llm, quant_config_str, sampling_params, stream_generate
    from src.steer import frozen, load_vectors, steer_vector, vllm_install, vllm_set, vllm_tokens_steered

    info = model_info(a.model)
    tag = info["tag"]
    Ls = frozen(tag)["L_star"]
    vec_meta = load_vectors(tag)[1]
    logfile = REPORTS / "logs" / f"gen_steer_{tag}.log"
    tok = AutoTokenizer.from_pretrained(info["hf_id"])
    gen = dict(load_cfg("e0")["generation"])
    outdir = OUTPUTS / tag / "e3"
    jobs = []
    for c, v, al in parse_settings(a.settings):
        name = setting_name(a.set, c, v, al)
        path = outdir / f"{name}.jsonl"
        items = items_for(a.set, tok, c)
        have = done_uids(path)
        rest = [it for it in items if it["uid"] not in have]
        log(f"{name}: {len(items)} items, {len(have)} done, {len(rest)} to run", logfile)
        if rest:
            jobs.append((c, v, al, name, path, rest))
    if not jobs:
        return
    a.no_prefix_caching = True
    llm, ekw = make_llm(info, a, logfile)
    assert ekw.get("enable_prefix_caching") is False
    layer_type = vllm_install(llm, Ls)
    qcfg = quant_config_str(llm)
    sp = sampling_params(gen)
    checks_path = REPORTS / f"e3_hook_checks_{tag}.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    for c, v, al, name, path, items in jobs:
        vec = steer_vector(tag, v, al)
        vllm_set(llm, Ls, vec)
        meta = {"model": info["hf_id"], "tag": tag, "task": "e3", "setting": name, "condition": c,
                "steer": {"vector": v, "alpha": al, "layer": Ls, "scope": "all", "k": vec_meta["k"],
                          "norm": None if vec is None else float(vec.norm())},
                "engine": ekw, "sampling_params": repr(sp), "quant_config": qcfg, "layer_type": layer_type,
                "vllm_version": vllm.__version__}
        w = JsonlWriter(path, meta, flush_every=gen["flush_every"])
        expected = {"prompt": 0, "decode": 0}

        def on_done(it, out, wall, w=w, c=c, v=v, al=al, name=name, expected=expected):
            o = out.outputs[0]
            expected["prompt"] += it["prompt_n_tokens"]
            expected["decode"] += max(len(o.token_ids) - 1, 0)   # the last sampled token is never fed back
            w.write({"uid": it["uid"], "subject": it["subject"], "level": it["level"], "condition": c,
                     "setting": name, "steer_vector": v, "steer_alpha": al, "steer_layer": Ls,
                     "model": info["hf_id"], "gold": it["gold"], "prompt_sha256": it["prompt_sha256"],
                     "prompt_n_tokens": it["prompt_n_tokens"], "output_text": o.text,
                     "output_token_ids_len": len(o.token_ids), "finish_reason": o.finish_reason,
                     "stop_reason": o.stop_reason, "gen_wall_s": round(wall, 3),
                     "vllm_version": vllm.__version__, "quant_config": qcfg, "sampling_params": repr(sp)})

        log(f"generating {name} ({len(items)} items)", logfile)
        st = stream_generate(llm, items, lambda it: sp, on_done, max_inflight=2 * ekw["max_num_seqs"],
                             max_model_len=ekw["max_model_len"], logfile=logfile)
        w.close()
        got = vllm_tokens_steered(llm, Ls)
        need = expected["prompt"] + expected["decode"]
        # preemption/recompute can only ADD hook calls; fewer means some tokens were not steered
        ok = (vec is None and got == 0) or (vec is not None and got >= need)
        checks[name] = {"hook_tokens": got, "expected_min": need, "prompt_tokens": expected["prompt"],
                        "decode_tokens": expected["decode"], "OK": ok, **st}
        checks_path.write_text(json.dumps(checks, indent=1))
        log(f"finished {name}: {st}; hook tokens {got} vs expected ≥{need} -> {'OK' if ok else 'FAIL'}", logfile)
        if not ok:
            raise SystemExit(f"hook check failed for {name}")
    vllm_set(llm, Ls, None)


if __name__ == "__main__":
    from src.gen_vllm import add_engine_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    ap.add_argument("--set", required=True, choices=["dev", "eval"])
    ap.add_argument("--settings", required=True)
    add_engine_args(ap)
    main(ap.parse_args())
