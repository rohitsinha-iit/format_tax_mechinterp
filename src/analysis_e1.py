"""E1 analysis.

  python -m src.analysis_e1 --model qwen3-4b --stage brevity    # §8.3 behavioural check (exit 2 on fail)
  python -m src.analysis_e1 --model qwen3-4b --stage perq       # §8.5 per-question test (gated models only)
  python -m src.analysis_e1 --model qwen3-4b --stage report     # reports/e1_{tag}.md
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from src.common import ACTS, FIGS, OUTPUTS, REPORTS, RESULTS, git_hash, load_cfg, model_info, now, read_jsonl


# ----------------------------------------------------------------------------- brevity check

def brevity(tag: str) -> dict:
    bc = load_cfg("e1")["brevity_check"]
    C = pd.DataFrame(read_jsonl(OUTPUTS / tag / "e1_brevity_C1.jsonl"))
    D = pd.DataFrame(read_jsonl(OUTPUTS / tag / "e1_brevity_D1.jsonl"))
    assert len(C) == len(D) == bc["n_problems"], (len(C), len(D))
    mc, md = float(C["output_token_ids_len"].median()), float(D["output_token_ids_len"].median())
    rep = {"tag": tag, "n": len(C), "median_tokens_C1": mc, "median_tokens_D1": md, "ratio": mc / md,
           "max_ratio": bc["max_ratio"], "PASS": mc / md <= bc["max_ratio"],
           "truncation_C1": float((C["finish_reason"] == "length").mean()),
           "truncation_D1": float((D["finish_reason"] == "length").mean()),
           "iqr_C1": [float(C["output_token_ids_len"].quantile(q)) for q in (.25, .75)],
           "iqr_D1": [float(D["output_token_ids_len"].quantile(q)) for q in (.25, .75)]}
    (RESULTS / tag).mkdir(parents=True, exist_ok=True)
    (RESULTS / tag / "e1_brevity_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


# ----------------------------------------------------------------------------- per-question test

def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def spearman_boot(x, y, B, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    rx = pd.Series(x)
    vals = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        vals[b] = spearman(rx.values[i], np.asarray(y)[i])
    return [float(np.nanquantile(vals, .025)), float(np.nanquantile(vals, .975))]


def logit_fit(s, y, X_ctrl):
    """y ~ s + controls (controls already include intercept + level dummies + log tokens_FF)."""
    import statsmodels.api as sm
    X = np.column_stack([s, X_ctrl])
    m = sm.Logit(y, X).fit(disp=0, maxiter=200)
    b, se = m.params[0], m.bse[0]
    ci = m.conf_int()[0]
    return {"coef": float(b), "se": float(se), "OR_per_SD": float(np.exp(b)),
            "OR_ci95": [float(np.exp(ci[0])), float(np.exp(ci[1]))], "p": float(m.pvalues[0]),
            "converged": bool(m.mle_retvals.get("converged", True))}


def perq(tag: str) -> dict:
    pq = load_cfg("e1")["per_question"]
    fz = json.loads((RESULTS / tag / "e1_frozen.json").read_text())
    Ls = fz["L_star"]
    b_hat = load_file(str(RESULTS / tag / "e1_b_hat.safetensors"))["b_hat"].double()
    from src.acts_hf import load_acts
    aFF, mFF = load_acts(ACTS / tag / "eval" / "FF_Lstar_last.safetensors")
    aJS, mJS = load_acts(ACTS / tag / "eval" / "JS_Lstar_last.safetensors")
    assert mFF["uids"] == mJS["uids"] and mFF["layer"] == mJS["layer"] == Ls
    uids = mFF["uids"]
    diff = aJS.double() - aFF.double()
    s_raw = (diff @ b_hat).numpy()
    s = (s_raw - s_raw.mean()) / s_raw.std(ddof=0)

    ff = pd.DataFrame(read_jsonl(OUTPUTS / tag / "e0_FF_scored.jsonl")).set_index("uid").loc[uids]
    js = pd.DataFrame(read_jsonl(OUTPUTS / tag / "e0_JS_scored.jsonl")).set_index("uid").loc[uids]
    df = pd.DataFrame({"uid": uids, "s": s, "s_raw": s_raw, "level": ff["level"].values,
                       "tok_ff": ff["output_token_ids_len"].values, "tok_js": js["output_token_ids_len"].values,
                       "rtok_js": js["reasoning_tokens"].values.astype(float),
                       "ff_correct": ff["final_correct"].values.astype(bool),
                       "js_correct": js["final_correct"].values.astype(bool)})
    df["dtok"] = df["tok_js"] - df["tok_ff"]
    df["dtok_reasoning"] = df["rtok_js"] - df["tok_ff"]
    B, bseed = pq["bootstrap_resamples"], pq["bootstrap_seed"]

    out = {"tag": tag, "L_star": Ls, "n": len(df)}
    out["spearman_s_dtok"] = {"rho": spearman(df["s"], df["dtok"]), "ci95": spearman_boot(df["s"].values, df["dtok"].values, B, bseed)}
    v = df.dropna(subset=["dtok_reasoning"])
    out["spearman_s_dtok_reasoning"] = {"rho": spearman(v["s"], v["dtok_reasoning"]), "n": len(v),
                                        "n_excluded_invalid_json": int(len(df) - len(v)),
                                        "ci95": spearman_boot(v["s"].values, v["dtok_reasoning"].values, B, bseed)}
    fc = df[df["ff_correct"]].copy()
    fc["js_incorrect"] = (~fc["js_correct"]).astype(int)
    lv = pd.get_dummies(fc["level"].astype(int), prefix="L", drop_first=True).astype(float)
    X_ctrl = np.column_stack([np.ones(len(fc)), lv.values, np.log(fc["tok_ff"].clip(lower=1).values)])
    out["logit_flip"] = dict(logit_fit(fc["s"].values, fc["js_incorrect"].values, X_ctrl),
                             n=len(fc), events=int(fc["js_incorrect"].sum()),
                             formula="JS_incorrect ~ s + C(level) + log(tokens_FF), FF-correct items")

    # ---- negative control: random unit directions in place of b̂
    rng = np.random.default_rng(pq["random_dir_seed"])
    d = diff.shape[1]
    R = rng.standard_normal((pq["random_directions"], d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = diff.numpy() @ R.T                                         # [n, K]
    S = (S - S.mean(0)) / S.std(0)
    rho_r = np.array([spearman(S[:, k], df["dtok"].values) for k in range(S.shape[1])])
    fcmask = df["ff_correct"].values
    lor_r = []
    for k in range(S.shape[1]):
        try:
            lor_r.append(logit_fit(S[fcmask, k], fc["js_incorrect"].values, X_ctrl)["coef"])
        except Exception:
            lor_r.append(np.nan)
    lor_r = np.array(lor_r)
    rho, lor = out["spearman_s_dtok"]["rho"], out["logit_flip"]["coef"]
    out["random_dir_control"] = {
        "K": int(S.shape[1]), "seed": pq["random_dir_seed"],
        # predicted direction: rho < 0, log-OR > 0
        "spearman_pct_signed": float((-rho_r < -rho).mean() * 100), "spearman_rand_p95_of_neg_rho": float(np.quantile(-rho_r, .95)),
        "spearman_pct_abs": float((np.abs(rho_r) < abs(rho)).mean() * 100),
        "logOR_pct_signed": float((lor_r[~np.isnan(lor_r)] < lor).mean() * 100),
        "logOR_rand_p95": float(np.nanquantile(lor_r, .95)),
        "logOR_pct_abs": float((np.abs(lor_r[~np.isnan(lor_r)]) < abs(lor)).mean() * 100),
        "logOR_failed_fits": int(np.isnan(lor_r).sum())}
    np.save(RESULTS / tag / "e1_random_dir_rho.npy", rho_r)
    np.save(RESULTS / tag / "e1_random_dir_logOR.npy", lor_r)

    # ---- H1 criteria (pre-registered, reported separately)
    sp, lg, rc = out["spearman_s_dtok"], out["logit_flip"], out["random_dir_control"]
    c1 = fz["cos_fperpc_b_at_Lstar"] > fz["null_p99"]
    c2a = sp["rho"] < 0 and sp["ci95"][1] < 0
    c2b = lg["OR_per_SD"] > 1 and lg["p"] < 0.05
    c3a = rc["spearman_pct_signed"] > 95
    c3b = rc["logOR_pct_signed"] > 95
    out["H1"] = {
        "i_cos_fperpc_b_gt_null_p99": {"value": fz["cos_fperpc_b_at_Lstar"], "null_p99": fz["null_p99"], "met": bool(c1)},
        "ii_a_spearman_neg_ci_excl_0": {"rho": sp["rho"], "ci95": sp["ci95"], "met": bool(c2a)},
        "ii_b_flip_OR_gt1_p_lt_05": {"OR": lg["OR_per_SD"], "p": lg["p"], "met": bool(c2b)},
        "ii_met": bool(c2a or c2b),
        "iii_a_spearman_beats_random_p95": {"percentile_signed": rc["spearman_pct_signed"], "percentile_abs": rc["spearman_pct_abs"], "met": bool(c3a)},
        "iii_b_logOR_beats_random_p95": {"percentile_signed": rc["logOR_pct_signed"], "percentile_abs": rc["logOR_pct_abs"], "met": bool(c3b)},
        "iii_met_for_a_statistic_meeting_ii": bool((c2a and c3a) or (c2b and c3b)),
    }
    df.to_csv(RESULTS / tag / "e1_perq_frame.csv", index=False)
    (RESULTS / tag / "e1_perq.json").write_text(json.dumps(out, indent=1))
    figures_perq(tag, df)
    print(json.dumps(out, indent=1))
    return out


def figures_perq(tag, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.scatter(df["s"], df["dtok"], s=8, alpha=.5, color="#1f5fbf")
    ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel("s (standardised projection of JS−FF on b̂ at L*)"); ax.set_ylabel("Δtok = tokens(JS) − tokens(FF)")
    ax.set_title(f"{tag}: s vs Δtok (ρ={df[['s', 'dtok']].corr('spearman').iloc[0, 1]:.3f})")
    fig.tight_layout(); fig.savefig(FIGS / f"e1_s_vs_dtok_{tag}.png", dpi=150); plt.close(fig)
    fc = df[df["ff_correct"]].copy()
    fc["q"] = pd.qcut(fc["s"].rank(method="first"), 5, labels=False) + 1
    g = fc.groupby("q").agg(n=("s", "size"), flip=("js_correct", lambda x: (~x).mean()))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.bar(g.index, g["flip"] * 100, color="#d0782a")
    for i, (n, f) in g.iterrows():
        ax.text(i, f * 100 + .3, f"n={n}", ha="center", fontsize=7)
    ax.set_xlabel("s quintile (FF-correct items)"); ax.set_ylabel("P(JS wrong | FF correct) (%)")
    ax.set_title(f"{tag}: flip rate by s quintile")
    fig.tight_layout(); fig.savefig(FIGS / f"e1_flip_by_s_quintile_{tag}.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- report

def report(tag: str):
    R = RESULTS / tag
    fz = json.loads((R / "e1_frozen.json").read_text())
    bc = json.loads((R / "e1_brevity_check.json").read_text()) if (R / "e1_brevity_check.json").exists() else None
    st = {p: json.loads((R / f"e1_stats_{p}.json").read_text()) for p in ("last", "tmpl", "gen16") if (R / f"e1_stats_{p}.json").exists()}
    e0 = json.loads((R / "e0_summary.json").read_text()) if (R / "e0_summary.json").exists() else None
    pq = json.loads((R / "e1_perq.json").read_text()) if (R / "e1_perq.json").exists() else None
    excl = REPORTS / f"gen16_exclusions_{tag}.json"
    L = [f"# E1 — directions and H1: {tag}", "", f"_Generated {now()} · git {git_hash()}_", ""]
    if bc:
        L += ["## Brevity behavioural check (§8.3)", "",
              f"- first {bc['n']} extract_set problems: median tokens C_1 = {bc['median_tokens_C1']:.0f}, D_1 = {bc['median_tokens_D1']:.0f}; "
              f"ratio = **{bc['ratio']:.3f}** (require ≤ {bc['max_ratio']}) → **{'PASS' if bc['PASS'] else 'FAIL'}**; "
              f"truncation C_1 {bc['truncation_C1'] * 100:.0f}%, D_1 {bc['truncation_D1'] * 100:.0f}%", ""]
    d = st["last"]["d"]
    L += ["## Direction geometry (extract_set, n=300)", "",
          f"d = {d}, analytic random-direction scale 1/√d = {1 / np.sqrt(d):.4f}. Null = sign-flip of JS−FF paired differences "
          f"(1,000 draws, default_rng(3)), statistic cos(f_null⊥c, b).", ""]
    for pos in ("last", "tmpl"):
        rows = st[pos]["layers"]
        L += [f"### P_{pos}", "",
              "| ℓ | cos(f,b) | cos(f⊥c,b) | null p95 | null p99 | cos(c,b) | cos(f,c) | cos(b1,b2) | cos(b1,b3) | cos(b2,b3) | split-half f | split-half b |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            mark = " **(L*)**" if pos == "last" and r["layer"] == fz["L_star"] else ""
            L.append(f"| {r['layer']}{mark} | {r['cos_f_b']:.3f} | {r['cos_fperpc_b']:.3f} | {r['null_p95']:.3f} | {r['null_p99']:.3f} | "
                     f"{r['cos_c_b']:.3f} | {r['cos_f_c']:.3f} | {r['cos_b1-2']:.3f} | {r['cos_b1-3']:.3f} | {r['cos_b2-3']:.3f} | "
                     f"{r['splithalf_f_mean']:.3f} | {r['splithalf_b_mean']:.3f} |")
        L.append("")
    if "gen16" in st:
        rows = st["gen16"]["layers"]
        L += ["### P_gen16 (FF/JS only; b is taken from P_last because C/D/NC have no gen16 activations)", "",
              "Caveat: content tokens differ across conditions (FF content vs JSON-string content).", "",
              "| ℓ | cos(f_gen16, b_last) | null p95 | null p99 | split-half f |", "|---|---|---|---|---|"]
        L += [f"| {r['layer']} | {r['obs_vs_null']:.3f} | {r['null_p95']:.3f} | {r['null_p99']:.3f} | {r['splithalf_f_mean']:.3f} |" for r in rows]
        if excl.exists():
            L += ["", f"gen16 exclusions: `{json.loads(excl.read_text())}`"]
        L.append("")
    L += ["## Frozen layer", "", f"- L* = **{fz['L_star']}** (window ℓ ∈ [{fz['layer_window'][0]}, {fz['layer_window'][1]}], P_last, extract_set only), "
          f"cos(f⊥c, b) = {fz['cos_fperpc_b_at_Lstar']:.3f}; null p95 {fz['null_p95']:.3f}, p99 {fz['null_p99']:.3f}; frozen at {fz['frozen_at']}", "",
          f"![cos by layer](../figs/e1_cos_by_layer_{tag}.png)", ""]
    if e0:
        L += [f"## Gate (from E0): {'PASS' if e0['gate']['PASS'] else 'FAIL'} "
              f"(tax {e0['gate']['tax_ff_js_pp']:+.2f} pp, p = {e0['gate']['mcnemar_p']:.3g})", ""]
    if pq:
        h = pq["H1"]
        sp, lg, rc = pq["spearman_s_dtok"], pq["logit_flip"], pq["random_dir_control"]
        L += ["## Per-question test (§8.5, MATH-500, n=500)", "",
              f"- Spearman(s, Δtok) = {sp['rho']:.3f}, 95% bootstrap CI [{sp['ci95'][0]:.3f}, {sp['ci95'][1]:.3f}]",
              f"- Spearman(s, Δtok_reasoning) = {pq['spearman_s_dtok_reasoning']['rho']:.3f}, CI "
              f"[{pq['spearman_s_dtok_reasoning']['ci95'][0]:.3f}, {pq['spearman_s_dtok_reasoning']['ci95'][1]:.3f}] "
              f"(n = {pq['spearman_s_dtok_reasoning']['n']}; {pq['spearman_s_dtok_reasoning']['n_excluded_invalid_json']} invalid-JSON items excluded)",
              f"- Logit {lg['formula']}: OR per SD of s = {lg['OR_per_SD']:.3f} [{lg['OR_ci95'][0]:.3f}, {lg['OR_ci95'][1]:.3f}], "
              f"p = {lg['p']:.3g} (n = {lg['n']}, events = {lg['events']})",
              f"- Random-direction control (K = {rc['K']}): Spearman beats {rc['spearman_pct_signed']:.1f}% of random directions in the "
              f"predicted direction ({rc['spearman_pct_abs']:.1f}% by |ρ|); log-OR beats {rc['logOR_pct_signed']:.1f}% "
              f"({rc['logOR_pct_abs']:.1f}% by |log-OR|)", "",
              "## H1 criteria (pre-registered; each reported separately)", "",
              "| criterion | value | met |", "|---|---|---|",
              f"| (i) cos(f⊥c, b) at L* > null p99 | {h['i_cos_fperpc_b_gt_null_p99']['value']:.3f} vs {h['i_cos_fperpc_b_gt_null_p99']['null_p99']:.3f} | {h['i_cos_fperpc_b_gt_null_p99']['met']} |",
              f"| (ii-a) Spearman(s, Δtok) < 0, CI excludes 0 | {sp['rho']:.3f} [{sp['ci95'][0]:.3f}, {sp['ci95'][1]:.3f}] | {h['ii_a_spearman_neg_ci_excl_0']['met']} |",
              f"| (ii-b) flip OR > 1, p < 0.05 | {lg['OR_per_SD']:.3f}, p = {lg['p']:.3g} | {h['ii_b_flip_OR_gt1_p_lt_05']['met']} |",
              f"| (ii) either | | {h['ii_met']} |",
              f"| (iii-a) Spearman > random p95 (signed) | {rc['spearman_pct_signed']:.1f} pct | {h['iii_a_spearman_beats_random_p95']['met']} |",
              f"| (iii-b) log-OR > random p95 (signed) | {rc['logOR_pct_signed']:.1f} pct | {h['iii_b_logOR_beats_random_p95']['met']} |",
              f"| (iii) for a statistic that met (ii) | | {h['iii_met_for_a_statistic_meeting_ii']} |", "",
              f"![s vs dtok](../figs/e1_s_vs_dtok_{tag}.png) ![flip by s](../figs/e1_flip_by_s_quintile_{tag}.png)", ""]
    else:
        L += ["## Per-question test (§8.5)", "", "Not run (model did not pass the E0 gate, or eval activations not extracted yet).", ""]
    (REPORTS / f"e1_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    ap.add_argument("--stage", required=True, choices=["brevity", "perq", "report"])
    a = ap.parse_args()
    tag = model_info(a.model)["tag"]
    if a.stage == "brevity":
        sys.exit(0 if brevity(tag)["PASS"] else 2)
    elif a.stage == "perq":
        e0 = json.loads((RESULTS / tag / "e0_summary.json").read_text())
        if not e0["gate"]["PASS"]:
            sys.exit(f"{tag} did not pass the E0 gate; §8.5 runs only for gated models")
        perq(tag)
    else:
        report(tag)
