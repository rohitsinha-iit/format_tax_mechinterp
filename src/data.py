"""Dataset loading (MATH-500 eval) and the stratified extract/dev split from MATH train (§3)."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict

import numpy as np

from src.common import DATA, REPORTS, load_cfg, log, meta_record

_WS = re.compile(r"\s+")


def norm_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def load_math500() -> list[dict]:
    from datasets import load_dataset
    cfg = load_cfg("e0")["eval_dataset"]
    ds = load_dataset(cfg["path"], split=cfg["split"])
    rows = []
    for r in ds:  # keep dataset order
        rows.append({"uid": r["unique_id"], "problem": r["problem"], "solution": r["solution"],
                     "gold": r["answer"], "subject": r["subject"], "level": int(r["level"])})
    assert len(rows) == 500, len(rows)
    return rows


def load_extract_set() -> list[dict]:
    with open(DATA / "extract_set.jsonl", encoding="utf-8") as f:
        return [r for r in map(json.loads, f) if "_meta" not in r]


def _largest_remainder(sizes: "OrderedDict[str, int]", n: int) -> dict:
    total = sum(sizes.values())
    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(np.floor(x)) for k, x in raw.items()}
    short = n - sum(alloc.values())
    # largest fractional part first; ties broken by stratum key order (deterministic)
    order = sorted(sizes.keys(), key=lambda k: (-(raw[k] - alloc[k]), list(sizes).index(k)))
    for k in order[:short]:
        alloc[k] += 1
    return alloc


def build_splits():
    from datasets import load_dataset
    cfg = load_cfg("e1")["data"]
    logf = REPORTS / "data_splits.log"
    m500 = load_math500()
    m500_norm = {norm_ws(r["problem"]) for r in m500}

    pool = []
    for sub in cfg["train_configs"]:
        ds = load_dataset(cfg["train_source"], sub, split="train")
        log(f"loaded {cfg['train_source']}/{sub} train: {len(ds)} rows; fields={ds.column_names}", logf)
        for i, r in enumerate(ds):
            pool.append({"uid": f"{sub}/{i}", "subject": sub, "level": r["level"],
                         "type": r.get("type"), "problem": r["problem"], "solution": r["solution"]})
    n_all = len(pool)
    pool = [r for r in pool if norm_ws(r["problem"]) not in m500_norm]
    log(f"train pool: {n_all} problems; dropped {n_all - len(pool)} overlapping MATH-500 (whitespace-normalised)", logf)

    strata = OrderedDict()
    for r in sorted(pool, key=lambda r: (r["subject"], r["level"])):
        strata.setdefault((r["subject"], r["level"]), []).append(r)
    sizes = OrderedDict((f"{s}|{l}", len(v)) for (s, l), v in strata.items())
    keyed = {f"{s}|{l}": v for (s, l), v in strata.items()}
    # keep original within-stratum order (by uid index) before sampling
    for k in keyed:
        keyed[k] = sorted(keyed[k], key=lambda r: (r["subject"], int(r["uid"].split("/")[1])))

    rng = np.random.default_rng(cfg["sample_seed"])
    q_ext = _largest_remainder(sizes, cfg["n_extract"])
    extract, remaining = [], {}
    for k, rows in keyed.items():
        idx = rng.permutation(len(rows))
        take = set(idx[:q_ext[k]].tolist())
        extract += [rows[i] for i in sorted(take)]
        remaining[k] = [rows[i] for i in range(len(rows)) if i not in take]
    # dev: same population proportions, drawn from what is left
    q_dev = _largest_remainder(sizes, cfg["n_dev"])
    dev = []
    for k, rows in remaining.items():
        idx = rng.permutation(len(rows))
        dev += [rows[i] for i in sorted(idx[:q_dev[k]].tolist())]
    assert len(extract) == cfg["n_extract"] and len(dev) == cfg["n_dev"]
    assert not ({r["uid"] for r in extract} & {r["uid"] for r in dev})

    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "extract_set.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(meta_record(seed=cfg["sample_seed"], n=len(extract),
                                       strata="subject x level, proportional, largest remainder")) + "\n")
        for r in extract:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DATA / "dev_set_ids.json", "w") as f:
        json.dump({"_meta": meta_record(seed=cfg["sample_seed"])["_meta"],
                   "uids": [r["uid"] for r in dev]}, f, indent=1)
    log(f"extract_set n={len(extract)} by subject={dict(Counter(r['subject'] for r in extract))}", logf)
    log(f"extract_set by level={dict(sorted(Counter(r['level'] for r in extract).items()))}", logf)
    log(f"dev_set n={len(dev)} (ids only saved; reserved for E3)", logf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["splits"])
    a = ap.parse_args()
    if a.cmd == "splits":
        build_splits()
