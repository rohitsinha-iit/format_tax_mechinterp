"""vLLM generation (E0 conditions, extraction-set generations, brevity check). Resumable.

Usage (inside WSL venv, from repo root):
  python -m src.gen_vllm --model qwen3-4b --task e0 --conds FF,JS,JC
  python -m src.gen_vllm --model qwen3-4b --task e0 --conds FF,JS,JC --limit 10 --subdir smoke
  python -m src.gen_vllm --model qwen3-4b --task e0 --conds 2T
  python -m src.gen_vllm --model qwen3-4b --task ext80
  python -m src.gen_vllm --model qwen3-4b --task brevity
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.common import (OUTPUTS, REPORTS, JsonlWriter, done_uids, load_cfg, log, model_info,
                        read_jsonl)
from src import prompting as P


# ----------------------------------------------------------------------------- engine

def engine_kwargs(info: dict, args=None) -> dict:
    eng = load_cfg("models")["engine"]
    kw = dict(model=info["hf_id"], quantization="bitsandbytes", dtype=eng["dtype"],
              max_model_len=eng["max_model_len"], gpu_memory_utilization=eng["gpu_memory_utilization"],
              enforce_eager=eng["enforce_eager"], max_num_seqs=info["max_num_seqs"], seed=eng["seed"])
    if args is not None:
        if args.max_num_seqs: kw["max_num_seqs"] = args.max_num_seqs
        if args.gpu_mem: kw["gpu_memory_utilization"] = args.gpu_mem
        if args.max_model_len: kw["max_model_len"] = args.max_model_len
        # steering (E3): cached KV computed under another steering vector must never be reused
        if getattr(args, "no_prefix_caching", False): kw["enable_prefix_caching"] = False
    # VERIFY(§2.1): older vLLM needs load_format="bitsandbytes" alongside quantization.
    import inspect
    import vllm
    from vllm.engine.arg_utils import EngineArgs
    v = tuple(int(x) for x in vllm.__version__.split(".")[:2] if x.isdigit())
    if v < (0, 8):
        kw["load_format"] = "bitsandbytes"
    assert "quantization" in inspect.signature(EngineArgs).parameters
    return kw


def make_llm(info: dict, args=None, logfile=None):
    from vllm import LLM
    kw = engine_kwargs(info, args)
    log(f"vLLM engine kwargs: {kw}", logfile)
    llm = LLM(**kw)
    return llm, kw


def quant_config_str(llm) -> str:
    """Effective bnb quant config as seen by the engine (logged into every record)."""
    try:
        mc = llm.llm_engine.model_config
        qc = getattr(mc, "hf_config", None)
        qc = getattr(qc, "quantization_config", None)
        from src.vllm_introspect import bnb_inflight_settings
        return json.dumps({"quantization": mc.quantization, "checkpoint_quant_config": qc,
                           "inflight": bnb_inflight_settings()}, default=str)
    except Exception as e:  # never fail a run on logging
        return f"unavailable: {e!r}"


def sampling_params(gen: dict, structured_json_schema: dict | None = None, logprobs: int | None = None):
    from vllm import SamplingParams
    kw = dict(temperature=gen["temperature"], top_p=gen["top_p"], max_tokens=gen["max_tokens"],
              seed=gen["seed"])
    if "top_k" in gen:
        kw["top_k"] = gen["top_k"]
    if logprobs is not None:
        kw["logprobs"] = logprobs
    if structured_json_schema is not None:
        # VERIFY(§2.2): newer vLLM uses structured_outputs=StructuredOutputsParams, older guided_decoding.
        try:
            from vllm.sampling_params import StructuredOutputsParams
            kw["structured_outputs"] = StructuredOutputsParams(json=structured_json_schema)
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            kw["guided_decoding"] = GuidedDecodingParams(json=structured_json_schema)
    return SamplingParams(**kw)


def stream_generate(llm, items, sp_for, on_done, max_inflight: int, max_model_len: int, logfile=None):
    """Feed requests so that ~max_inflight are always queued, and hand each finished
    RequestOutput to on_done(item, out, wall_s) immediately (so files are written as we go).

    items: dicts with 'rid' and 'prompt_ids'. sp_for(item) -> SamplingParams.
    Prompts whose length leaves no room for generation are still submitted; vLLM caps
    generation at max_model_len and we record the effective cap."""
    eng = llm.llm_engine
    queue = deque(items)
    pending = {}
    n_done, tok_out, t_start = 0, 0, time.time()

    def submit():
        it = queue.popleft()
        sp = sp_for(it)
        eng.add_request(it["rid"], {"prompt_token_ids": it["prompt_ids"]}, sp)
        pending[it["rid"]] = (it, time.time())

    while queue and len(pending) < max_inflight:
        submit()
    while pending:
        for out in eng.step():
            if not out.finished:
                continue
            rid = out.request_id
            if rid not in pending:  # some versions decorate ids; map back by prefix
                rid = next(k for k in pending if rid.startswith(k) or k.startswith(rid))
            it, t0 = pending.pop(rid)
            on_done(it, out, time.time() - t0)
            n_done += 1
            tok_out += len(out.outputs[0].token_ids)
            if queue:
                submit()
            if n_done % 25 == 0:
                el = time.time() - t_start
                log(f"  {n_done} done, {tok_out} out tokens, {tok_out / el:.1f} tok/s, {el / 60:.1f} min", logfile)
        while queue and len(pending) < max_inflight:
            submit()
    el = time.time() - t_start
    return {"n": n_done, "out_tokens": tok_out, "wall_s": el, "tok_per_s": tok_out / max(el, 1e-9)}


# ----------------------------------------------------------------------------- tasks

def e0_items(tok, cond: str, info: dict, limit: int | None, subdir: str | None):
    from src.data import load_math500
    rows = load_math500()[: limit or None]
    ff = {}
    if cond == "2T":
        ffp = OUTPUTS / info["tag"] / "e0_FF.jsonl"
        ff = {r["uid"]: r for r in read_jsonl(ffp)}
        missing = [r["uid"] for r in rows if r["uid"] not in ff]
        assert not missing, f"2T needs complete FF outputs; missing {len(missing)}"
    items = []
    for r in rows:
        ff_out = ff[r["uid"]]["output_text"] if cond == "2T" else None
        pr = P.build_prompt(tok, cond, r["problem"], ff_output=ff_out)
        if cond == "2T":
            assert ("<|im_start|>assistant\n" + ff_out + "<|im_end|>\n") in pr["prompt"], r["uid"]
        items.append({"rid": r["uid"], "uid": r["uid"], "subject": r["subject"], "level": r["level"],
                      "gold": r["gold"], **pr})
    return items


def ext_items(tok, cond: str, n: int | None = None):
    from src.data import load_extract_set
    rows = load_extract_set()[: n or None]
    return [{"rid": r["uid"], "uid": r["uid"], "subject": r["subject"], "level": r["level"], "gold": None,
             **P.build_prompt(tok, cond, r["problem"])} for r in rows]


def run(args):
    import vllm
    from transformers import AutoTokenizer
    from prompts.templates import SCHEMA

    info = model_info(args.model)
    tag = info["tag"]
    e0 = load_cfg("e0")
    e1 = load_cfg("e1")
    outdir = OUTPUTS / tag / (args.subdir or "")
    logfile = REPORTS / "logs" / f"gen_{tag}.log"
    tok = AutoTokenizer.from_pretrained(info["hf_id"])

    if args.task == "e0":
        conds = args.conds.split(",")
        gen = dict(e0["generation"])
        jobs = [(c, outdir / f"e0_{c}.jsonl", e0_items(tok, c, info, args.limit, args.subdir), gen) for c in conds]
    elif args.task == "ext80":
        gen = dict(e0["generation"], max_tokens=e1["gen16"]["max_tokens"])
        jobs = [(c, outdir / f"e1_ext80_{c}.jsonl", ext_items(tok, c), gen) for c in e1["gen16"]["conditions"]]
    elif args.task == "brevity":
        gen = dict(e0["generation"], max_tokens=e1["brevity_check"]["max_tokens"])
        n = e1["brevity_check"]["n_problems"]
        jobs = [(c, outdir / f"e1_brevity_{c}.jsonl", ext_items(tok, c, n), gen) for c in ("C1", "D1")]
    else:
        raise ValueError(args.task)

    # skip finished uids up front so we do not even load the engine when nothing is left
    todo = []
    for c, path, items, g in jobs:
        have = done_uids(path)
        rest = [it for it in items if it["uid"] not in have]
        log(f"{tag} {args.task} {c}: {len(items)} items, {len(have)} done, {len(rest)} to run -> {path}", logfile)
        if rest:
            todo.append((c, path, rest, g))
    if not todo:
        return

    llm, ekw = make_llm(info, args, logfile)
    qcfg = quant_config_str(llm)
    log(f"effective quant config: {qcfg}", logfile)
    max_model_len = ekw["max_model_len"]
    stats = {}
    for c, path, items, g in todo:
        sp = sampling_params(g, structured_json_schema=SCHEMA if c == "JC" else None)
        sp_repr = repr(sp)
        meta = {"model": info["hf_id"], "tag": tag, "task": args.task, "condition": c,
                "engine": ekw, "sampling_params": sp_repr, "quant_config": qcfg,
                "vllm_version": vllm.__version__, "seeds": {"global": 0, "sampling": g["seed"]}}
        w = JsonlWriter(path, meta, flush_every=e0["generation"]["flush_every"])

        def on_done(it, out, wall, c=c, w=w, g=g):
            o = out.outputs[0]
            extra = {"output_token_ids": list(o.token_ids)} if args.task == "ext80" else {}
            w.write({**extra, "uid": it["uid"], "subject": it["subject"], "level": it["level"], "condition": c,
                     "model": info["hf_id"], "gold": it["gold"], "prompt_sha256": it["prompt_sha256"],
                     "prompt_n_tokens": it["prompt_n_tokens"], "output_text": o.text,
                     "output_token_ids_len": len(o.token_ids), "finish_reason": o.finish_reason,
                     "stop_reason": o.stop_reason, "gen_wall_s": round(wall, 3),
                     "effective_max_tokens": min(g["max_tokens"], max_model_len - it["prompt_n_tokens"]),
                     "vllm_version": vllm.__version__, "quant_config": qcfg, "sampling_params": sp_repr})

        log(f"generating {tag} {args.task} {c}: {len(items)} items", logfile)
        st = stream_generate(llm, items, lambda it, sp=sp: sp, on_done, max_inflight=2 * ekw["max_num_seqs"],
                             max_model_len=max_model_len, logfile=logfile)
        w.close()
        stats[c] = st
        log(f"finished {tag} {args.task} {c}: {st}", logfile)
    tp = REPORTS / f"throughput_{tag}_{args.task}{'_' + args.subdir if args.subdir else ''}.json"
    old = json.loads(tp.read_text()) if tp.exists() else {}
    old.update({k: dict(v, engine=ekw) for k, v in stats.items()})
    tp.write_text(json.dumps(old, indent=1))


def add_engine_args(ap):
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--no-prefix-caching", action="store_true")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    ap.add_argument("--task", required=True, choices=["e0", "ext80", "brevity"])
    ap.add_argument("--conds", default="FF,JS,JC")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subdir", default=None)
    add_engine_args(ap)
    run(ap.parse_args())
