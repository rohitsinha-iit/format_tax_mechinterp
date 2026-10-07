"""E3 analysis (docs/E3_plan.md).

  python -m src.analysis_e3 --model qwen3-4b --stage dev      # calibration table + α=1 safety rule -> reports/e3_dev_{tag}.md
  python -m src.analysis_e3 --model qwen3-4b --stage freeze   # results/{tag}/e3_frozen.json (refuses after eval outputs exist)
  python -m src.analysis_e3 --model qwen3-4b --stage eval     # S1–S2 (sufficiency arm) -> reports/e3_{tag}.md

Safety rule for the primary dose α = 1 (fixed 2026-10-07, before any steered output was inspected):
  for both JS−b and FF+b on the dev set: degenerate rate (truncated + repetitive/blank tail) increases by ≤ 5 pp over
  the unsteered baseline of the same prompt, and for JS−b lenient JSON validity drops by ≤ 5 pp. Otherwise STOP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from src.common import FIGS, OUTPUTS, REPORTS, RESULTS, git_hash, model_info, now, read_jsonl

ALL_SETTINGS = {"E3-N": ("JS", "minus_b", 1.0), "E3-S": ("FF", "plus_b", 1.0),
                "E3-Nr": ("JS", "minus_r0", 1.0), "E3-Sr": ("FF", "plus_r0", 1.0)}
# User decision 2026-10-07: the necessity arm (E3-N) failed the pre-specified dev safety rule at α = 0.5 and 1
# (b̂ also carries "terse/unformatted" style; subtracting it breaks strict JSON). Only the sufficiency arm runs on eval.
EVAL_SETTINGS = {k: ALL_SETTINGS[k] for k in ("E3-S", "E3-Sr")}
JUDGED = ("E3-S",)
SAFETY_PP = 0.05


def name(set_, c, v, a):
    return f"{set_}_{c}_{v}_a{a:g}"


def load(tag, n, suffix="_mv"):
    rows = read_jsonl(OUTPUTS / tag / "e3" / f"{n}{suffix}.jsonl")
    return pd.DataFrame(rows).set_index("uid") if rows else None


# ----------------------------------------------------------------------------- dev calibration

def dev(tag):
    d = OUTPUTS / tag / "e3"
    names = sorted(p.stem[:-3] for p in d.glob("dev_*_mv.jsonl"))
    checks = json.loads((REPORTS / f"e3_hook_checks_{tag}.json").read_text())
    rows = []
    base = {c: load(tag, name("dev", c, "none", 0)) for c in ("JS", "FF")}
    for n in names:
        df = load(tag, n)
        _, c, *rest = n.split("_")
        b = base[c]
        common = df.index.intersection(b.index)
        r = {"setting": n, "cond": c, "n": len(df), "median_out_tok": float(df["output_token_ids_len"].median()),
             "median_paired_dtok_vs_unsteered": float((df.loc[common, "output_token_ids_len"] -
                                                       b.loc[common, "output_token_ids_len"]).median()),
             "mv_acc": float(df["mv_correct"].mean()), "trunc": float((df["finish_reason"] == "length").mean()),
             "degenerate": float(df["degenerate"].mean()), "hook_ok": checks.get(n, {}).get("OK")}
        if c == "JS":
            r["median_reasoning_span_tok"] = float(df["reasoning_span_tokens"].median())
            r["lenient_valid"] = float(df["lenient_valid"].mean())
        rows.append(r)
    t = pd.DataFrame(rows).set_index("setting")
    safety = {}
    for c, v in (("JS", "minus_b"), ("FF", "plus_b")):
        n1, n0 = name("dev", c, v, 1.0), name("dev", c, "none", 0)
        if n1 in t.index and n0 in t.index:
            s = {"degenerate_increase": t.loc[n1, "degenerate"] - t.loc[n0, "degenerate"]}
            if c == "JS":
                s["validity_drop"] = t.loc[n0, "lenient_valid"] - t.loc[n1, "lenient_valid"]
            s["PASS"] = bool(s["degenerate_increase"] <= SAFETY_PP and s.get("validity_drop", 0) <= SAFETY_PP)
            safety[f"{c}|{v}"] = s
    out = {"tag": tag, "time": now(), "settings": t.reset_index().to_dict("records"), "safety_alpha1": safety,
           "SAFE": bool(safety) and all(s["PASS"] for s in safety.values()) and len(safety) == 2}
    (RESULTS / tag / "e3_dev_summary.json").write_text(json.dumps(out, indent=1, default=float))
    L = [f"# E3 dev calibration: {tag}", "", f"_{now()} · git {git_hash()} · 100 reserved dev problems, greedy, max 3072._", "",
         "| setting | median out tok | median Δtok vs unsteered (paired) | median reasoning tok | mv acc | lenient valid | trunc | degenerate | hook |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, r in t.iterrows():
        L.append(f"| {n} | {r['median_out_tok']:.0f} | {r['median_paired_dtok_vs_unsteered']:+.0f} | "
                 f"{r.get('median_reasoning_span_tok', float('nan')):.0f} | {r['mv_acc'] * 100:.0f}% | "
                 f"{'' if pd.isna(r.get('lenient_valid')) else f'{r.lenient_valid * 100:.0f}%'} | {r['trunc'] * 100:.0f}% | "
                 f"{r['degenerate'] * 100:.0f}% | {'OK' if r['hook_ok'] else r['hook_ok']} |")
    L += ["", "## Safety rule for α = 1 (pre-specified)", ""]
    for k, s in safety.items():
        L.append(f"- {k}: degenerate +{s['degenerate_increase'] * 100:.1f} pp"
                 + (f", validity drop {s['validity_drop'] * 100:.1f} pp" if "validity_drop" in s else "")
                 + f" → {'PASS' if s['PASS'] else 'FAIL'}")
    for k, s in safety.items():
        arm = "necessity (JS−b)" if k.startswith("JS") else "sufficiency (FF+b)"
        L.append(f"\n**{arm}: α = 1 {'is safe' if s['PASS'] else 'is NOT safe → arm not run on eval'}**")
    L.append("\nUser decision (2026-10-07): only the sufficiency arm runs on MATH-500. JS−b also failed the rule at "
             "α = 0.5 (validity drop 17 pp). The JS−b α = 1.5 run was stopped at 81/100 and is excluded "
             "(`dev_JS_minus_b_a1.5.partial_stopped`); JS α ∈ {1.5, 2} and the JS random controls r̂₁/r̂₂ were dropped "
             "as moot.")
    (REPORTS / f"e3_dev_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    dose_fig(tag, t)
    print("\n".join(L))
    return out


def dose_fig(tag, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (c, v, col) in zip(axes, (("JS", "minus_b", "#1f5fbf"), ("FF", "plus_b", "#d0782a"))):
        al = [0.0, 0.5, 1.0, 1.5, 2.0]
        ns = [name("dev", c, "none", 0)] + [name("dev", c, v, a) for a in al[1:]]
        ok = [(a, n) for a, n in zip(al, ns) if n in t.index]
        ax.plot([a for a, _ in ok], [t.loc[n, "median_out_tok"] for _, n in ok], "-o", color=col, label=f"{c} {v}")
        rs = [n for n in t.index if n.startswith(f"dev_{c}_{'minus' if v.startswith('minus') else 'plus'}_r")]
        ax.scatter([1.0] * len(rs), [t.loc[n, "median_out_tok"] for n in rs], marker="x", color="k", label="random r̂ (α=1)")
        ax2 = ax.twinx()
        ax2.plot([a for a, _ in ok], [t.loc[n, "mv_acc"] * 100 for _, n in ok], "--s", color="0.5")
        ax2.set_ylabel("mv accuracy (%)", color="0.4")
        ax.set_xlabel("α (multiples of k)"); ax.set_ylabel("median output tokens"); ax.set_title(f"dev: {c} steered by {v}")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGS / f"e3_dev_dose_{tag}.png", dpi=150); plt.close(fig)


def freeze(tag):
    fz = RESULTS / tag / "e3_frozen.json"
    if any((OUTPUTS / tag / "e3").glob("eval_*.jsonl")):
        sys.exit("eval outputs already exist: refusing to (re)freeze E3")
    s = json.loads((RESULTS / tag / "e3_dev_summary.json").read_text())
    if not s["safety_alpha1"].get("FF|plus_b", {}).get("PASS"):
        sys.exit("α = 1 failed the dev safety rule for the sufficiency arm: STOP (see reports/e3_dev_*.md)")
    vec = (RESULTS / tag / "e3_vectors.safetensors").read_bytes()
    out = {"tag": tag, "frozen_at": now(), "git": git_hash(), "alpha": 1.0, "layer": json.loads(
        (RESULTS / tag / "e1_frozen.json").read_text())["L_star"], "scope": "all positions (prompt + generated)",
           "eval_settings": {k: list(v) for k, v in EVAL_SETTINGS.items()}, "judged": list(JUDGED),
           "random_control_vector": "r0 (seed 100), ⊥ b̂ and f̂, norm k",
           "vectors_sha256": hashlib.sha256(vec).hexdigest(), "prefix_caching": False, "backend": "vLLM + hook",
           "criteria": {"S1": "output tokens FF+b < FF, paired one-sided Wilcoxon p<0.05, and FF+b < FF+r0 (paired, p<0.05)",
                        "S2": "acc(FF+b) < acc(FF): paired bootstrap 95% CI excludes 0 (+ exact McNemar)"},
           "labels": "E3-S: math-verify + judge (§6.4 rule); control E3-Sr: math-verify only (user decision)",
           "dev_safety": s["safety_alpha1"],
           "necessity_arm": "NOT RUN on eval: JS−b failed the pre-specified dev safety rule at α=0.5 and α=1 "
                            "(lenient-validity drop 17 and 26 pp > 5 pp; markdown style reinstated). User decision 2026-10-07."}
    fz.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- eval

def boot_diff(a, b, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = d[idx].mean(1) * 100
    return float(d.mean() * 100), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def mcn(a, b):
    from statsmodels.stats.contingency_tables import mcnemar
    a, b = a.astype(bool), b.astype(bool)
    t = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    return {"table": t, "p": float(mcnemar(t, exact=True).pvalue)}


def wil(x, alternative):
    from scipy.stats import wilcoxon
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return {"n": int(len(x)), "median_diff": float(np.median(x)), "mean_diff": float(np.mean(x)),
            "p": float(wilcoxon(x, alternative=alternative, zero_method="wilcox").pvalue)}


def span_tokens(tok, texts):
    from src.score import reasoning_span
    out = []
    for t in texts:
        s, _ = reasoning_span(t)
        out.append(np.nan if s is None else len(tok(s, add_special_tokens=False)["input_ids"]))
    return np.array(out, float)


def evaluate(tag):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_info("qwen3-4b" if "4b" in tag else "qwen3-8b")["hf_id"])
    base = {c: pd.DataFrame(read_jsonl(OUTPUTS / tag / f"e0_{c}_scored.jsonl")).set_index("uid") for c in ("FF", "JS")}
    uids = base["FF"].index
    S = {}
    for k, (c, v, a) in EVAL_SETTINGS.items():
        df = load(tag, name("eval", c, v, a), "_scored" if k in JUDGED else "_mv")
        assert df is not None and set(df.index) == set(uids), f"{k}: missing/incomplete outputs"
        S[k] = df.loc[uids]
    JS, FF = base["JS"].loc[uids], base["FF"].loc[uids]
    tokn = lambda df: df["output_token_ids_len"].values.astype(float)
    R = {"tag": tag, "time": now(), "git": git_hash(), "n": len(uids), "arms": sorted(EVAL_SETTINGS)}
    # S1
    R["S1"] = {"vs_FF": wil(tokn(S["E3-S"]) - tokn(FF), "less"), "vs_random": wil(tokn(S["E3-S"]) - tokn(S["E3-Sr"]), "less"),
               "random_vs_FF": wil(tokn(S["E3-Sr"]) - tokn(FF), "less"),
               "median_out_tokens": {"FF": float(np.median(tokn(FF))), "E3-S": float(np.median(tokn(S["E3-S"]))),
                                     "E3-Sr": float(np.median(tokn(S["E3-Sr"]))), "JS": float(np.median(tokn(JS)))}}
    R["S1"]["share_of_FF_JS_compression_reproduced_mean_out_tokens"] = float(
        (tokn(FF).mean() - tokn(S["E3-S"]).mean()) / (tokn(FF).mean() - tokn(JS).mean()))
    R["S1"]["MET"] = bool(R["S1"]["vs_FF"]["p"] < .05 and R["S1"]["vs_random"]["p"] < .05)
    # S2
    d, ci = boot_diff(FF["final_correct"].values, S["E3-S"]["final_correct"].values)
    R["S2"] = {"acc_loss_pp": d, "ci95": ci, "mcnemar": mcn(FF["final_correct"].values, S["E3-S"]["final_correct"].values),
               "JS_tax_pp_for_reference": 4.8}
    R["S2"]["MET"] = bool(ci[0] > 0)
    # descriptive table (mv-only accuracy for every condition, so controls are comparable)
    desc = []
    for k, df in [("FF", FF), ("JS", JS)] + list(S.items()):
        desc.append({"cond": k, "acc_final": float(df["final_correct"].mean()) if "final_correct" in df else None,
                     "acc_mv": float(df["mv_correct"].mean()), "median_out_tok": float(df["output_token_ids_len"].median()),
                     "trunc": float((df["finish_reason"] == "length").mean()),
                     "lenient_valid": float(df["lenient_valid"].mean()) if df["lenient_valid"].notna().any() else None})
    R["table"] = desc
    for k in ("E3-S", "E3-Sr"):
        d, ci = boot_diff(S[k]["mv_correct"].values, FF["mv_correct"].values)
        R.setdefault("mv_only_vs_baseline", {})[k] = {"diff_pp": d, "ci95": ci}
    (RESULTS / tag / "e3_eval_summary.json").write_text(json.dumps(R, indent=1, default=float))
    report(tag, R)
    return R


def report(tag, R):
    fz = json.loads((RESULTS / tag / "e3_frozen.json").read_text())
    s1, s2 = R["S1"], R["S2"]
    f = lambda w: f"median Δ {w['median_diff']:+.0f}, mean Δ {w['mean_diff']:+.0f}, n={w['n']}, p={w['p']:.3g}"
    pct = lambda x: "—" if x is None else f"{x * 100:.1f}%"
    L = [f"# E3 — steering with the brevity direction: {tag}", "",
         f"_{now()} · git {git_hash()} · frozen {fz['frozen_at']} · L* = {fz['layer']}, α = 1 (k = 15.53), all positions, "
         "vLLM + hook (prefix caching off)._", "",
         "## Necessity arm (JS − k·b̂): not run on MATH-500", "", fz["necessity_arm"] + " See `reports/e3_dev_" + tag + ".md`.", "",
         "## Sufficiency arm (FF + k·b̂) on MATH-500, n = " + str(R["n"]), "",
         "| cond | acc (final) | acc (mv only) | median out tok | truncation |", "|---|---|---|---|---|"]
    for r in R["table"]:
        if r["cond"] in ("FF", "JS", "E3-S", "E3-Sr"):
            L.append(f"| {r['cond']} | {pct(r['acc_final'])} | {pct(r['acc_mv'])} | {r['median_out_tok']:.0f} | {pct(r['trunc'])} |")
    L += ["", "E3-S = FF + k·b̂ · E3-Sr = FF + k·r̂₀ (random control, math-verify labels only) · FF/JS = unsteered E0 outputs.", "",
          "## Pre-registered criteria (approved 2026-10-07)", "", "| criterion | result | met |", "|---|---|---|",
          f"| S1 output tokens FF+b < FF | {f(s1['vs_FF'])} | |",
          f"| S1 … and FF+b < FF+random | {f(s1['vs_random'])} | **{s1['MET']}** |",
          f"| S2 acc(FF) − acc(FF+b) > 0 | {s2['acc_loss_pp']:+.2f} pp [{s2['ci95'][0]:+.2f}, {s2['ci95'][1]:+.2f}], "
          f"McNemar p = {s2['mcnemar']['p']:.3g} | **{s2['MET']}** |",
          "", "## Supporting numbers", "",
          f"- share of the FF→JS compression reproduced by FF + k·b̂ (mean output tokens): "
          f"{s1['share_of_FF_JS_compression_reproduced_mean_out_tokens'] * 100:.0f}%",
          f"- random control vs FF (tokens): {f(s1['random_vs_FF'])}",
          f"- JS tax for reference: +{s2['JS_tax_pp_for_reference']:.1f} pp (E0)",
          "- math-verify-only accuracy differences vs FF: " +
          "; ".join(f"{k} {v['diff_pp']:+.1f} pp [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]" for k, v in R["mv_only_vs_baseline"].items()),
          f"- McNemar table FF vs FF+b (both ✓, FF✓ only, FF+b✓ only, both ✗): {s2['mcnemar']['table']}"]
    (REPORTS / f"e3_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    ap.add_argument("--stage", required=True, choices=["dev", "freeze", "eval"])
    a = ap.parse_args()
    tag = model_info(a.model)["tag"]
    {"dev": dev, "freeze": freeze, "eval": evaluate}[a.stage](tag)
