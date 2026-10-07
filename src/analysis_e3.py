"""E3 analysis (docs/E3_plan.md).

  python -m src.analysis_e3 --model qwen3-4b --stage dev      # calibration table + α=1 safety rule -> reports/e3_dev_{tag}.md
  python -m src.analysis_e3 --model qwen3-4b --stage freeze   # results/{tag}/e3_frozen.json (refuses after eval outputs exist)
  python -m src.analysis_e3 --model qwen3-4b --stage eval     # N1–N3, S1–S2 -> reports/e3_{tag}.md

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

EVAL_SETTINGS = {"E3-N": ("JS", "minus_b", 1.0), "E3-S": ("FF", "plus_b", 1.0),
                 "E3-Nr": ("JS", "minus_r0", 1.0), "E3-Sr": ("FF", "plus_r0", 1.0)}
JUDGED = ("E3-N", "E3-S")
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
    L.append(f"\n**α = 1 {'is safe → freeze and run eval' if out['SAFE'] else 'is NOT safe → STOP and report'}**")
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
    if not s["SAFE"]:
        sys.exit("α = 1 failed the dev safety rule: STOP (see reports/e3_dev_*.md)")
    vec = (RESULTS / tag / "e3_vectors.safetensors").read_bytes()
    out = {"tag": tag, "frozen_at": now(), "git": git_hash(), "alpha": 1.0, "layer": json.loads(
        (RESULTS / tag / "e1_frozen.json").read_text())["L_star"], "scope": "all positions (prompt + generated)",
           "eval_settings": {k: list(v) for k, v in EVAL_SETTINGS.items()}, "judged": list(JUDGED),
           "random_control_vector": "r0 (seed 100), ⊥ b̂ and f̂, norm k",
           "vectors_sha256": hashlib.sha256(vec).hexdigest(), "prefix_caching": False, "backend": "vLLM + hook",
           "criteria": {"N1": "reasoning-field tokens (tolerant span) JS−b > JS, paired one-sided Wilcoxon p<0.05, and JS−b > JS−r0 (paired, p<0.05)",
                        "N2": "acc(JS−b) − acc(JS) > 0: paired bootstrap 95% CI excludes 0 (+ exact McNemar)",
                        "N3": "lenient validity(JS−b) ≥ lenient validity(JS) − 5 pp",
                        "S1": "output tokens FF+b < FF, paired one-sided Wilcoxon p<0.05, and FF+b < FF+r0 (paired, p<0.05)",
                        "S2": "acc(FF+b) < acc(FF): paired bootstrap 95% CI excludes 0 (+ exact McNemar)"},
           "labels": "E3-N/E3-S: math-verify + judge (§6.4 rule); controls E3-Nr/E3-Sr: math-verify only (user decision)",
           "dev_safety": s["safety_alpha1"]}
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
    rs = {k: span_tokens(tok, S[k]["output_text"]) for k in ("E3-N", "E3-Nr")}
    rs["JS"] = span_tokens(tok, JS["output_text"])
    tokn = lambda df: df["output_token_ids_len"].values.astype(float)
    R = {"tag": tag, "time": now(), "git": git_hash(), "n": len(uids)}
    # N1
    strict = (JS["lenient_valid"].astype(bool).values & S["E3-N"]["lenient_valid"].astype(bool).values)
    R["N1"] = {"vs_JS": wil(rs["E3-N"] - rs["JS"], "greater"), "vs_random": wil(rs["E3-N"] - rs["E3-Nr"], "greater"),
               "random_vs_JS": wil(rs["E3-Nr"] - rs["JS"], "greater"),
               "strict_valid_subset_vs_JS": wil((rs["E3-N"] - rs["JS"])[strict], "greater"),
               "share_of_FF_JS_compression_recovered_mean_out_tokens":
                   float((tokn(S["E3-N"]).mean() - tokn(JS).mean()) / (tokn(FF).mean() - tokn(JS).mean())),
               "median_reasoning_tokens": {"JS": float(np.nanmedian(rs["JS"])), "E3-N": float(np.nanmedian(rs["E3-N"])),
                                           "E3-Nr": float(np.nanmedian(rs["E3-Nr"]))}}
    R["N1"]["MET"] = bool(R["N1"]["vs_JS"]["p"] < .05 and R["N1"]["vs_random"]["p"] < .05)
    # N2
    d, ci = boot_diff(S["E3-N"]["final_correct"].values, JS["final_correct"].values)
    R["N2"] = {"acc_gain_pp": d, "ci95": ci, "mcnemar": mcn(S["E3-N"]["final_correct"].values, JS["final_correct"].values)}
    R["N2"]["MET"] = bool(ci[0] > 0)
    # N3
    vN, vJ = float(S["E3-N"]["lenient_valid"].mean()), float(JS["lenient_valid"].mean())
    R["N3"] = {"lenient_valid_E3N": vN, "lenient_valid_JS": vJ, "MET": bool(vN >= vJ - 0.05)}
    # S1
    R["S1"] = {"vs_FF": wil(tokn(S["E3-S"]) - tokn(FF), "less"), "vs_random": wil(tokn(S["E3-S"]) - tokn(S["E3-Sr"]), "less"),
               "random_vs_FF": wil(tokn(S["E3-Sr"]) - tokn(FF), "less"),
               "median_out_tokens": {"FF": float(np.median(tokn(FF))), "E3-S": float(np.median(tokn(S["E3-S"]))),
                                     "E3-Sr": float(np.median(tokn(S["E3-Sr"]))), "JS": float(np.median(tokn(JS)))}}
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
    for k in ("E3-N", "E3-Nr"):
        d, ci = boot_diff(S[k]["mv_correct"].values, JS["mv_correct"].values)
        R.setdefault("mv_only_vs_baseline", {})[k] = {"diff_pp": d, "ci95": ci}
    for k in ("E3-S", "E3-Sr"):
        d, ci = boot_diff(S[k]["mv_correct"].values, FF["mv_correct"].values)
        R.setdefault("mv_only_vs_baseline", {})[k] = {"diff_pp": d, "ci95": ci}
    (RESULTS / tag / "e3_eval_summary.json").write_text(json.dumps(R, indent=1, default=float))
    report(tag, R)
    return R


def report(tag, R):
    fz = json.loads((RESULTS / tag / "e3_frozen.json").read_text())
    n1, n2, n3, s1, s2 = R["N1"], R["N2"], R["N3"], R["S1"], R["S2"]
    f = lambda w: f"median Δ {w['median_diff']:+.0f}, mean Δ {w['mean_diff']:+.0f}, n={w['n']}, p={w['p']:.3g}"
    L = [f"# E3 — steering with the brevity direction: {tag}", "",
         f"_{now()} · git {git_hash()} · frozen {fz['frozen_at']} · L* = {fz['layer']}, α = 1 (k = 15.53), all positions, "
         "vLLM + hook (prefix caching off)._", "",
         "## Conditions", "", "| cond | acc (final) | acc (mv only) | median out tok | truncation | lenient JSON valid |",
         "|---|---|---|---|---|---|"]
    pct = lambda x: "—" if x is None else f"{x * 100:.1f}%"
    for r in R["table"]:
        L.append(f"| {r['cond']} | {pct(r['acc_final'])} | {pct(r['acc_mv'])} | {r['median_out_tok']:.0f} | "
                 f"{pct(r['trunc'])} | {pct(r['lenient_valid'])} |")
    L += ["", "E3-N = JS − k·b̂ · E3-S = FF + k·b̂ · E3-Nr = JS − k·r̂ · E3-Sr = FF + k·r̂ (controls: math-verify labels only).", "",
          "## Pre-registered criteria (approved 2026-10-07)", "", "| criterion | result | met |", "|---|---|---|",
          f"| N1 reasoning tokens JS−b > JS | {f(n1['vs_JS'])} | |",
          f"| N1 … and > JS−random | {f(n1['vs_random'])} | **{n1['MET']}** |",
          f"| N2 acc(JS−b) − acc(JS) > 0 | {n2['acc_gain_pp']:+.2f} pp [{n2['ci95'][0]:+.2f}, {n2['ci95'][1]:+.2f}], McNemar p={n2['mcnemar']['p']:.3g} | **{n2['MET']}** |",
          f"| N3 validity within 5 pp | {n3['lenient_valid_E3N'] * 100:.1f}% vs {n3['lenient_valid_JS'] * 100:.1f}% | **{n3['MET']}** |",
          f"| S1 tokens FF+b < FF | {f(s1['vs_FF'])} | |",
          f"| S1 … and < FF+random | {f(s1['vs_random'])} | **{s1['MET']}** |",
          f"| S2 acc(FF) − acc(FF+b) > 0 | {s2['acc_loss_pp']:+.2f} pp [{s2['ci95'][0]:+.2f}, {s2['ci95'][1]:+.2f}], McNemar p={s2['mcnemar']['p']:.3g} | **{s2['MET']}** |",
          "", "## Supporting numbers", "",
          f"- N1 on the strictly-valid subset: {f(n1['strict_valid_subset_vs_JS'])}",
          f"- random control vs JS (reasoning tokens): {f(n1['random_vs_JS'])}; vs FF (S): {f(s1['random_vs_FF'])}",
          f"- share of FF→JS compression recovered (mean output tokens): {n1['share_of_FF_JS_compression_recovered_mean_out_tokens'] * 100:.0f}%",
          f"- mv-only accuracy differences vs unsteered baseline: " +
          "; ".join(f"{k} {v['diff_pp']:+.1f} pp [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]" for k, v in R["mv_only_vs_baseline"].items()),
          "", "N1 measures the reasoning field with tolerant span extraction (JSON parse if possible, else the raw "
          "`\"reasoning\": \"…\"` span), because only ~29% of JS outputs are strictly valid JSON (LaTeX escapes)."]
    (REPORTS / f"e3_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    ap.add_argument("--stage", required=True, choices=["dev", "freeze", "eval"])
    a = ap.parse_args()
    tag = model_info(a.model)["tag"]
    {"dev": dev, "freeze": freeze, "eval": evaluate}[a.stage](tag)
