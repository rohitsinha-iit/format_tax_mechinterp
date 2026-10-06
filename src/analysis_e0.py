"""E0 analysis (§7): tables, paired-bootstrap tax CIs, exact McNemar, flip analysis, gate.

  python -m src.analysis_e0 --model qwen3-4b      -> reports/e0_{tag}.md, results/{tag}/e0_summary.json, figs
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common import FIGS, OUTPUTS, REPORTS, RESULTS, git_hash, load_cfg, model_info, now, read_jsonl

CONDS = ["FF", "JS", "JC", "2T"]
JSON_CONDS = ["JS", "JC", "2T"]


def load(tag: str) -> dict[str, pd.DataFrame]:
    out = {}
    for c in CONDS:
        rows = read_jsonl(OUTPUTS / tag / f"e0_{c}_scored.jsonl")
        if rows:
            df = pd.DataFrame(rows).set_index("uid")
            assert df.index.is_unique
            out[c] = df
    return out


def q(x, p):
    x = pd.Series(x).dropna()
    return float(np.quantile(x, p)) if len(x) else float("nan")


def cond_table(dfs: dict) -> list[dict]:
    rows = []
    for c, df in dfs.items():
        mvw = ~df["mv_correct"].astype(bool)
        r = {"condition": c, "n": len(df), "acc": df["final_correct"].mean(), "acc_mv": df["mv_correct"].mean(),
             "strict_valid": df["strict_valid"].mean() if c in JSON_CONDS else None,
             "lenient_valid": df["lenient_valid"].mean() if c in JSON_CONDS else None,
             "tok_median": q(df["output_token_ids_len"], .5), "tok_q25": q(df["output_token_ids_len"], .25),
             "tok_q75": q(df["output_token_ids_len"], .75),
             "reasoning_tok_median": q(df["reasoning_tokens"], .5) if c in JSON_CONDS else None,
             "reasoning_tok_q25": q(df["reasoning_tokens"], .25) if c in JSON_CONDS else None,
             "reasoning_tok_q75": q(df["reasoning_tokens"], .75) if c in JSON_CONDS else None,
             "truncation_rate": (df["finish_reason"] == "length").mean(),
             "judge_failed": int((mvw & df["judge_failed"].astype(bool)).sum()),
             "n_judged_mv_wrong": int(mvw.sum()),
             "judge_flipped_to_correct": int((mvw & df["final_correct"].astype(bool)).sum()),
             "extract_none": int((df["extract_method"] == "none").sum()),
             "latex_escape_repaired": int(df.get("latex_escape_repaired", pd.Series(False)).astype(bool).sum())}
        rows.append(r)
    return rows


def paired_tax(a: np.ndarray, b: np.ndarray, B: int, seed: int) -> dict:
    """tax = acc(a) - acc(b) in pp with paired bootstrap 95% CI over uids."""
    rng = np.random.default_rng(seed)
    n = len(a)
    d = a.astype(float) - b.astype(float)
    idx = rng.integers(0, n, size=(B, n))
    boots = d[idx].mean(1) * 100
    return {"tax_pp": float(d.mean() * 100), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    from statsmodels.stats.contingency_tables import mcnemar
    a, b = a.astype(bool), b.astype(bool)
    t = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    r = mcnemar(t, exact=True)
    return {"table": t, "a_only": t[0][1], "b_only": t[1][0], "statistic": float(r.statistic), "p": float(r.pvalue)}


def flip_analysis(ff: pd.DataFrame, js: pd.DataFrame) -> dict:
    j = ff[["final_correct", "output_token_ids_len", "level"]].join(
        js[["final_correct", "output_token_ids_len", "reasoning_tokens"]], lsuffix="_ff", rsuffix="_js")
    j["flip"] = j["final_correct_ff"].astype(bool) & ~j["final_correct_js"].astype(bool)
    j["rescue"] = ~j["final_correct_ff"].astype(bool) & j["final_correct_js"].astype(bool)
    j["dtok"] = j["output_token_ids_len_js"] - j["output_token_ids_len_ff"]
    j["decile"] = pd.qcut(j["dtok"].rank(method="first"), 10, labels=False) + 1
    dec = j.groupby("decile").agg(n=("flip", "size"), flips=("flip", "sum"), flip_rate=("flip", "mean"),
                                  dtok_min=("dtok", "min"), dtok_max=("dtok", "max"), dtok_median=("dtok", "median"))
    ffc = j[j["final_correct_ff"].astype(bool)]
    dec_cond = ffc.groupby("decile").agg(n_ff_correct=("flip", "size"), flip_rate_given_ff_correct=("flip", "mean"))
    lvl = j.groupby("level").agg(n=("flip", "size"), flips=("flip", "sum"), flip_rate=("flip", "mean"),
                                 rescues=("rescue", "sum"))
    lvl_c = ffc.groupby("level").agg(n_ff_correct=("flip", "size"), flip_rate_given_ff_correct=("flip", "mean"))
    return {"n_flip": int(j["flip"].sum()), "n_rescue": int(j["rescue"].sum()),
            "dtok_median": float(j["dtok"].median()),
            "by_decile": dec.join(dec_cond).reset_index().to_dict("records"),
            "by_level": lvl.join(lvl_c).reset_index().to_dict("records"), "_frame": j}


def figures(tag, dfs, table, fa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    cs = [r["condition"] for r in table]
    accs = [r["acc"] * 100 for r in table]
    mv = [r["acc_mv"] * 100 for r in table]
    x = np.arange(len(cs))
    ax.bar(x - 0.18, accs, 0.36, label="final_correct (mv + judge)", color="#1f5fbf")
    ax.bar(x + 0.18, mv, 0.36, label="math-verify only", color="#9db8e6")
    for i, v in enumerate(accs):
        ax.text(i - 0.18, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, cs); ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title(f"{tag}: MATH-500 accuracy by condition (n=500)"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / f"e0_acc_by_condition_{tag}.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    d = pd.DataFrame(fa["by_decile"])
    axes[0].plot(d["decile"], d["flip_rate"] * 100, "-o", label="P(flip)")
    axes[0].plot(d["decile"], d["flip_rate_given_ff_correct"] * 100, "--s", label="P(flip | FF correct)")
    axes[0].set_xticks(d["decile"], [f"{int(a)}..{int(b)}" for a, b in zip(d["dtok_min"], d["dtok_max"])],
                       rotation=45, fontsize=7)
    axes[0].set_xlabel("Δtok = tokens(JS) − tokens(FF), decile range"); axes[0].set_ylabel("flip rate (%)")
    axes[0].set_title("FF→JS correct→wrong flips by Δtok decile"); axes[0].legend(fontsize=8)
    lv = pd.DataFrame(fa["by_level"])
    axes[1].bar(lv["level"], lv["flip_rate_given_ff_correct"] * 100, color="#d0782a")
    axes[1].set_xlabel("MATH level"); axes[1].set_ylabel("P(flip | FF correct) (%)"); axes[1].set_title("flips by level")
    fig.tight_layout(); fig.savefig(FIGS / f"e0_flip_by_dtok_decile_{tag}.png", dpi=150); plt.close(fig)


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main(a):
    info = model_info(a.model)
    tag = info["tag"]
    ac = load_cfg("e0")["analysis"]
    dfs = load(tag)
    assert "FF" in dfs and "JS" in dfs, "need scored FF and JS"
    uids = dfs["FF"].index
    for c, df in dfs.items():
        assert set(df.index) == set(uids), f"{c}: uid set differs from FF"
    table = cond_table(dfs)
    B, seed = ac["bootstrap_resamples"], ac["bootstrap_seed"]
    taxes, taxes_mv = {}, {}
    for x in [c for c in ("JS", "JC", "2T") if c in dfs]:
        taxes[x] = paired_tax(dfs["FF"].loc[uids, "final_correct"].values, dfs[x].loc[uids, "final_correct"].values, B, seed)
        taxes_mv[x] = paired_tax(dfs["FF"].loc[uids, "mv_correct"].values, dfs[x].loc[uids, "mv_correct"].values, B, seed)
    pairs = [("FF", "JS"), ("FF", "JC"), ("JS", "JC"), ("FF", "2T")]
    mc = {f"{p}_vs_{r}": mcnemar_exact(dfs[p].loc[uids, "final_correct"].values, dfs[r].loc[uids, "final_correct"].values)
          for p, r in pairs if p in dfs and r in dfs}
    fa = flip_analysis(dfs["FF"].loc[uids], dfs["JS"].loc[uids])
    figures(tag, dfs, table, fa)
    gate = {"tax_ff_js_pp": taxes["JS"]["tax_pp"], "mcnemar_p": mc["FF_vs_JS"]["p"],
            "threshold_pp": ac["gate_tax_pp"], "alpha": ac["gate_alpha"]}
    gate["PASS"] = bool(gate["tax_ff_js_pp"] >= ac["gate_tax_pp"] and gate["mcnemar_p"] < ac["gate_alpha"])
    trunc_flags = {r["condition"]: r["truncation_rate"] for r in table if r["truncation_rate"] > ac["truncation_flag"]}
    summ = {"tag": tag, "time": now(), "git": git_hash(), "table": table, "tax": taxes, "tax_mv_only": taxes_mv,
            "mcnemar": mc, "flip": {k: v for k, v in fa.items() if k != "_frame"}, "gate": gate,
            "truncation_flags": trunc_flags}
    (RESULTS / tag).mkdir(parents=True, exist_ok=True)
    (RESULTS / tag / "e0_summary.json").write_text(json.dumps(summ, indent=1, default=float))

    L = [f"# E0 — format tax replication: {tag}", "",
         f"_Generated {now()} · git {git_hash()} · bootstrap B={B}, default_rng({seed}) · paired over n={len(uids)} MATH-500 uids._", "",
         "## Per-condition table", "",
         "| cond | n | acc (final) | acc (mv only) | strict_valid | lenient_valid | out tok median [IQR] | reasoning tok median [IQR] | trunc. rate | mv-wrong judged | judge→correct | judge_failed | extract none |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        rt = (f"{fmt(r['reasoning_tok_median'], 0)} [{fmt(r['reasoning_tok_q25'], 0)}–{fmt(r['reasoning_tok_q75'], 0)}]"
              if r["reasoning_tok_median"] is not None else "—")
        L.append(f"| {r['condition']} | {r['n']} | {r['acc'] * 100:.1f}% | {r['acc_mv'] * 100:.1f}% | "
                 f"{fmt(r['strict_valid'])} | {fmt(r['lenient_valid'])} | {r['tok_median']:.0f} [{r['tok_q25']:.0f}–{r['tok_q75']:.0f}] | "
                 f"{rt} | {r['truncation_rate'] * 100:.1f}% | {r['n_judged_mv_wrong']} | {r['judge_flipped_to_correct']} | "
                 f"{r['judge_failed']} | {r['extract_none']} |")
    L += ["", "## Tax = acc(FF) − acc(X) (pp), paired bootstrap 95% CI", "",
          "| X | tax (final) | 95% CI | tax (mv only) | 95% CI |", "|---|---|---|---|---|"]
    for x, t in taxes.items():
        tm = taxes_mv[x]
        L.append(f"| {x} | {t['tax_pp']:+.2f} | [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}] | {tm['tax_pp']:+.2f} | "
                 f"[{tm['ci95'][0]:+.2f}, {tm['ci95'][1]:+.2f}] |")
    L += ["", "## Exact McNemar (final_correct)", "", "| pair | both ✓ | A✓ B✗ | A✗ B✓ | both ✗ | p |", "|---|---|---|---|---|---|"]
    for k, m in mc.items():
        t = m["table"]
        L.append(f"| {k.replace('_vs_', ' vs ')} | {t[0][0]} | {t[0][1]} | {t[1][0]} | {t[1][1]} | {m['p']:.3g} |")
    L += ["", f"## Flip analysis FF → JS (flip = FF ✓ & JS ✗): {fa['n_flip']} flips, {fa['n_rescue']} rescues; "
              f"median Δtok = {fa['dtok_median']:.0f}", "",
          "| Δtok decile | Δtok range | n | flips | P(flip) | n FF✓ | P(flip | FF✓) |", "|---|---|---|---|---|---|---|"]
    for d in fa["by_decile"]:
        L.append(f"| {d['decile']} | {d['dtok_min']:.0f}..{d['dtok_max']:.0f} | {d['n']} | {d['flips']} | "
                 f"{d['flip_rate'] * 100:.1f}% | {d['n_ff_correct']} | {d['flip_rate_given_ff_correct'] * 100:.1f}% |")
    L += ["", "| level | n | flips | rescues | P(flip) | n FF✓ | P(flip | FF✓) |", "|---|---|---|---|---|---|---|"]
    for d in fa["by_level"]:
        L.append(f"| {d['level']} | {d['n']} | {d['flips']} | {d['rescues']} | {d['flip_rate'] * 100:.1f}% | "
                 f"{d['n_ff_correct']} | {d['flip_rate_given_ff_correct'] * 100:.1f}% |")
    L += ["", "## Gate (§7.3)", "",
          f"- tax(FF−JS) = {gate['tax_ff_js_pp']:+.2f} pp (threshold ≥ {gate['threshold_pp']}), "
          f"exact McNemar p = {gate['mcnemar_p']:.3g} (threshold < {gate['alpha']})",
          f"- **gate: {'PASS' if gate['PASS'] else 'FAIL'}**", "",
          "## Truncation", "",
          (f"- ⚠️ conditions with truncation > {ac['truncation_flag'] * 100:.0f}%: " +
           ", ".join(f"{k} ({v * 100:.1f}%)" for k, v in trunc_flags.items()) + " — truncation biases the tax.")
          if trunc_flags else f"- no condition exceeds {ac['truncation_flag'] * 100:.0f}% truncation.", "",
          "## Notes", "",
          "- Final label = math-verify if correct, else the Qwen3-8B thinking judge verdict (pre-registered rule §6.4); "
          "judge failures count as incorrect.",
          "- Known limitation: for Qwen3-8B outputs the judge is the same model. Mitigations: math-verify is primary, the judge "
          "task is narrow equivalence, inputs are format-blind, and the human audit is a gate.",
          "- JSON-decoded control characters that came from LaTeX commands (e.g. `\\f`rac → form-feed) are repaired in "
          "the extracted candidate (`latex_escape_repaired`; counts in the table summary JSON).",
          f"- Figures: `figs/e0_acc_by_condition_{tag}.png`, `figs/e0_flip_by_dtok_decile_{tag}.png`."]
    (REPORTS / f"e0_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    main(ap.parse_args())
