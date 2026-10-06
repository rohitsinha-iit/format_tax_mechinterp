"""Human audit of the judge (§6.5.2–3).

  python -m src.audit --model qwen3-4b --stage export   -> reports/audit_{tag}.csv + reports/audit_key_{tag}.csv
  (user fills human_label with EQUIVALENT / NOT_EQUIVALENT)
  python -m src.audit --model qwen3-4b --stage agree    -> reports/audit_agreement_{tag}.{json,md}
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common import OUTPUTS, REPORTS, load_cfg, model_info, read_jsonl


def _judged(tag: str, conds) -> list[dict]:
    rows = []
    for c in conds:
        rows += read_jsonl(OUTPUTS / tag / "judge" / f"e0_{c}_judge_seed0.jsonl")
    return sorted(rows, key=lambda r: (r["condition"], r["uid"]))


def export(model: str, conds):
    from src.judge import dataset_order
    tag = model_info(model)["tag"]
    ac = load_cfg("e0")["audit"]
    _, problems = dataset_order()
    rows = _judged(tag, conds)
    cats = {
        "flip": [r for r in rows if r["role"] == "mv_wrong" and r["judge_verdict"] == "EQUIVALENT"],
        "not_equiv": [r for r in rows if r["role"] == "mv_wrong" and r["judge_verdict"] == "NOT_EQUIVALENT"],
        "calib": [r for r in rows if r["role"] == "calib"],
    }
    want = {"flip": ac["n_flip"], "not_equiv": ac["n_not_equiv"], "calib": ac["n_calib"]}
    rng = np.random.default_rng(ac["sample_seed"])
    picked, shortfall = [], {}
    for cat, pool in cats.items():
        k = min(want[cat], len(pool))
        if k < want[cat]:
            shortfall[cat] = {"wanted": want[cat], "available": len(pool)}
        for i in sorted(rng.choice(len(pool), size=k, replace=False).tolist()):
            picked.append(dict(pool[i], category=cat))
    order = rng.permutation(len(picked))
    audit, key = [], []
    for n, i in enumerate(order):
        r = picked[i]
        aid = f"{tag}-{n + 1:03d}"
        audit.append({"audit_id": aid, "problem": problems[r["uid"]]["problem"], "gold": problems[r["uid"]]["gold"],
                      "candidate_str": r["candidate_str"], "tail": r["tail"], "judge_verdict": r["judge_verdict"],
                      "human_label": ""})
        key.append({"audit_id": aid, "uid": r["uid"], "condition": r["condition"], "category": r["category"],
                    "role": r["role"], "mv_correct": r["mv_correct"]})
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(REPORTS / f"audit_{tag}.csv", index=False, encoding="utf-8")
    pd.DataFrame(key).to_csv(REPORTS / f"audit_key_{tag}.csv", index=False, encoding="utf-8")
    info = {"tag": tag, "n_rows": len(audit), "by_category": {c: sum(k["category"] == c for k in key) for c in cats},
            "pool_sizes": {c: len(v) for c, v in cats.items()}, "shortfall": shortfall,
            "n_judge_failed_mv_wrong": sum(r["role"] == "mv_wrong" and r["judge_failed"] for r in rows)}
    (REPORTS / f"audit_export_{tag}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def _norm(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper().replace(" ", "_")
    if s in ("EQUIVALENT", "EQ", "E", "1", "Y", "YES", "TRUE", "CORRECT"):
        return "EQUIVALENT"
    if s in ("NOT_EQUIVALENT", "NOT", "NE", "N", "0", "NO", "FALSE", "INCORRECT", "WRONG"):
        return "NOT_EQUIVALENT"
    raise ValueError(f"unrecognised human_label {x!r}")


def agree(model: str):
    from sklearn.metrics import cohen_kappa_score
    tag = model_info(model)["tag"]
    ac = load_cfg("e0")["audit"]
    a = pd.read_csv(REPORTS / f"audit_{tag}.csv", dtype=str, keep_default_na=False)
    k = pd.read_csv(REPORTS / f"audit_key_{tag}.csv", dtype=str)
    df = a.merge(k, on="audit_id")
    df["human"] = df["human_label"].map(lambda x: _norm(x) if x != "" else None)
    unl = int(df["human"].isna().sum())
    lab = df[df["human"].notna()].copy()
    lab["agree"] = lab["human"] == lab["judge_verdict"]
    by_cat = lab.groupby("category")["agree"].agg(["mean", "count"]).to_dict("index")
    rep = {"tag": tag, "n_labeled": int(len(lab)), "n_unlabeled": unl,
           "overall_agreement": float(lab["agree"].mean()) if len(lab) else None,
           "flip_agreement": float(lab.loc[lab.category == "flip", "agree"].mean()) if (lab.category == "flip").any() else None,
           "cohen_kappa": float(cohen_kappa_score(lab["human"], lab["judge_verdict"])) if len(lab) else None,
           "by_category": by_cat,
           "by_condition": lab.groupby("condition")["agree"].agg(["mean", "count"]).to_dict("index")}
    rep["accept_overall"] = rep["overall_agreement"] is not None and rep["overall_agreement"] >= ac["accept_overall"]
    rep["accept_flip"] = rep["flip_agreement"] is not None and rep["flip_agreement"] >= ac["accept_flip"]
    rep["ACCEPTED"] = bool(rep["accept_overall"] and rep["accept_flip"] and unl == 0)
    dis = lab[~lab["agree"]][["audit_id", "uid", "condition", "category", "gold", "candidate_str",
                              "judge_verdict", "human"]]
    rep["disagreements"] = dis.to_dict("records")
    (REPORTS / f"audit_agreement_{tag}.json").write_text(json.dumps(rep, indent=1, default=str))
    lines = [f"# Judge–human audit: {tag}", "",
             f"- labeled: {rep['n_labeled']} (unlabeled: {unl})",
             f"- overall agreement: {rep['overall_agreement']:.3f} (accept ≥ {ac['accept_overall']})" if len(lab) else "",
             f"- flip-item agreement: {rep['flip_agreement']:.3f} (accept ≥ {ac['accept_flip']})" if rep['flip_agreement'] is not None else "",
             f"- Cohen's κ: {rep['cohen_kappa']:.3f}" if len(lab) else "",
             f"- **ACCEPTED: {rep['ACCEPTED']}**", "", "## By category", "",
             "| category | agreement | n |", "|---|---|---|"]
    lines += [f"| {c} | {v['mean']:.3f} | {v['count']} |" for c, v in by_cat.items()]
    lines += ["", "## Disagreements", "", "| audit_id | condition | category | gold | candidate | judge | human |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {d['audit_id']} | {d['condition']} | {d['category']} | `{d['gold']}` | `{d['candidate_str']}` | "
              f"{d['judge_verdict']} | {d['human']} |" for d in rep["disagreements"]]
    (REPORTS / f"audit_agreement_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    ap.add_argument("--stage", required=True, choices=["export", "agree"])
    ap.add_argument("--conds", default="FF,JS,JC")
    a = ap.parse_args()
    export(a.model, a.conds.split(",")) if a.stage == "export" else agree(a.model)
