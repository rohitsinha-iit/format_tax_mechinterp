"""E1 directions and statistics on extract_set (§8.4). Freezes L* before any eval-set activation exists.

  python -m src.directions --model qwen3-4b            # all positions, stats, figure, freeze
Outputs (results/{tag}/):
  e1_directions_{pos}.safetensors  f, c, b, b1..b3, f_perp_c   [L+1, d] float32
  e1_null_{pos}.npy                sign-flip null cos(f_null⊥c, b)  [draws, L+1]
  e1_stats_{pos}.json              per-layer statistics
  e1_frozen.json + e1_b_hat.safetensors
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations

import numpy as np
import torch
from safetensors.torch import save_file

from src.common import ACTS, FIGS, RESULTS, git_hash, load_cfg, log, now, model_info, REPORTS
from src.acts_hf import load_acts


def unit(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def cos(a, b, dim=-1):
    return (unit(a, dim) * unit(b, dim)).sum(dim)


def perp(f, c):
    """f - (f·ĉ)ĉ, layer-wise. f: [..., L+1, d], c: [L+1, d]."""
    ch = unit(c)
    return f - (f * ch).sum(-1, keepdim=True) * ch


def load_position(tag: str, pos: str) -> dict:
    """Returns {cond: [n, L+1, d] float32} with uid order checked across conditions."""
    e1 = load_cfg("e1")
    d = ACTS / tag / "extract"
    out, uids = {}, None
    if pos in ("last", "tmpl"):
        for c in e1["extraction_conditions"]:
            a, m = load_acts(d / f"{c}_span.safetensors")
            a = a.float()
            out[c] = a[:, :, -1] if pos == "last" else a.mean(2)
            assert uids is None or m["uids"] == uids, f"uid order mismatch in {c}"
            uids = m["uids"]
    elif pos == "gen16":
        ms = {}
        for c in e1["gen16"]["conditions"]:
            a, m = load_acts(d / f"{c}_gen16.safetensors")
            ms[c] = (a.float(), m["uids"])
        common = sorted(set(ms["FF"][1]) & set(ms["JS"][1]))  # paired: keep uids valid in both
        for c, (a, u) in ms.items():
            idx = [u.index(x) for x in common]
            out[c] = a[idx]
        uids = common
    out["_uids"] = uids
    return out


def compute(A: dict, null_draws: int, null_seed: int, sh_draws: int, sh_seed: int, has_controls: bool) -> dict:
    D_f = A["JS"] - A["FF"]                                     # [n, L+1, d]
    n, Lp1, d = D_f.shape
    f = D_f.mean(0)
    res = {"n": n, "L": Lp1 - 1, "d": d, "f": f}
    if has_controls:
        c = (A["NC"] - A["FF"]).mean(0)
        bk = [(A[f"C{k}"] - A[f"D{k}"]).mean(0) for k in (1, 2, 3)]
        b = torch.stack(bk).mean(0)
        fpc = perp(f, c)
        res.update(c=c, b=b, bk=bk, f_perp_c=fpc)
        res["cos_f_b"] = cos(f, b); res["cos_fperpc_b"] = cos(fpc, b)
        res["cos_c_b"] = cos(c, b); res["cos_f_c"] = cos(f, c)
        res["cos_bj_bk"] = {f"{j + 1}-{k + 1}": cos(bk[j], bk[k]) for j, k in combinations(range(3), 2)}
    # ---- sign-flip null: random ±1 per paired difference
    rng = np.random.default_rng(null_seed)
    S = torch.from_numpy(rng.choice([-1.0, 1.0], size=(null_draws, n)).astype(np.float32))
    Df2 = D_f.reshape(n, -1)
    nulls = []
    for i in range(0, null_draws, 100):
        fn = (S[i:i + 100] @ Df2 / n).reshape(-1, Lp1, d)
        if has_controls:
            nulls.append(cos(perp(fn, res["c"]), res["b"]))
        else:
            nulls.append(fn)  # gen16: compared with b of another position by the caller
    res["null"] = torch.cat(nulls, 0)
    # ---- split-half reliability
    rng = np.random.default_rng(sh_seed)
    shf, shb = [], []
    for _ in range(sh_draws):
        p = rng.permutation(n); ia, ib = p[: n // 2], p[n // 2: 2 * (n // 2)]
        shf.append(cos(D_f[ia].mean(0), D_f[ib].mean(0)))
        if has_controls:
            bA = torch.stack([(A[f"C{k}"][ia] - A[f"D{k}"][ia]).mean(0) for k in (1, 2, 3)]).mean(0)
            bB = torch.stack([(A[f"C{k}"][ib] - A[f"D{k}"][ib]).mean(0) for k in (1, 2, 3)]).mean(0)
            shb.append(cos(bA, bB))
    res["splithalf_f"] = torch.stack(shf)
    if has_controls:
        res["splithalf_b"] = torch.stack(shb)
    return res


def layer_table(res: dict, null: torch.Tensor, obs: torch.Tensor) -> list[dict]:
    rows = []
    p95, p99 = null.quantile(0.95, 0), null.quantile(0.99, 0)
    for l in range(res["L"] + 1):
        r = {"layer": l, "obs_vs_null": float(obs[l]), "null_p95": float(p95[l]), "null_p99": float(p99[l]),
             "null_p_value": float(((null[:, l] >= obs[l]).sum() + 1) / (null.shape[0] + 1)),
             "splithalf_f_mean": float(res["splithalf_f"][:, l].mean()),
             "splithalf_f_p05": float(res["splithalf_f"][:, l].quantile(0.05))}
        for k in ("cos_f_b", "cos_fperpc_b", "cos_c_b", "cos_f_c"):
            if k in res:
                r[k] = float(res[k][l])
        if "cos_bj_bk" in res:
            for k, v in res["cos_bj_bk"].items():
                r[f"cos_b{k}"] = float(v[l])
        if "splithalf_b" in res:
            r["splithalf_b_mean"] = float(res["splithalf_b"][:, l].mean())
            r["splithalf_b_p05"] = float(res["splithalf_b"][:, l].quantile(0.05))
        r["norm_f"] = float(res["f"][l].norm())
        if "b" in res:
            r["norm_b"] = float(res["b"][l].norm()); r["norm_c"] = float(res["c"][l].norm())
        rows.append(r)
    return rows


def plot(tag, tables, d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(tables), figsize=(6.2 * len(tables), 4.4), squeeze=False)
    for ax, (pos, rows) in zip(axes[0], tables.items()):
        L = [r["layer"] for r in rows]
        if "cos_fperpc_b" in rows[0]:
            ax.fill_between(L, [-r["null_p99"] for r in rows], [r["null_p99"] for r in rows], color="0.88", label="null 99% (±)")
            ax.fill_between(L, [-r["null_p95"] for r in rows], [r["null_p95"] for r in rows], color="0.75", label="null 95% (±)")
            ax.plot(L, [r["cos_fperpc_b"] for r in rows], "-o", ms=3, lw=2, color="#1f5fbf", label="cos(f⊥c, b)")
            ax.plot(L, [r["cos_f_b"] for r in rows], "-", lw=1.5, color="#7fa6e0", label="cos(f, b)")
            ax.plot(L, [r["cos_c_b"] for r in rows], "-", lw=1.5, color="#d0782a", label="cos(c, b)")
        else:
            ax.fill_between(L, [-r["null_p99"] for r in rows], [r["null_p99"] for r in rows], color="0.88", label="null 99% (±)")
            ax.fill_between(L, [-r["null_p95"] for r in rows], [r["null_p95"] for r in rows], color="0.75", label="null 95% (±)")
            ax.plot(L, [r["obs_vs_null"] for r in rows], "-o", ms=3, lw=2, color="#1f5fbf", label="cos(f_gen16, b_last)")
        ax.axhline(1 / np.sqrt(d), ls=":", color="k", lw=1, label="1/√d")
        ax.axhline(-1 / np.sqrt(d), ls=":", color="k", lw=1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{tag} — P_{pos}"); ax.set_xlabel("layer ℓ (0 = embeddings)"); ax.set_ylabel("cosine")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def freeze(tag: str, rows_last: list[dict], res_last: dict, force: bool):
    fz = RESULTS / tag / "e1_frozen.json"
    if (ACTS / tag / "eval").exists() and not force:
        raise SystemExit("eval-set activations already exist: refusing to (re)freeze L*")
    if fz.exists() and not force:
        log(f"{fz} exists; not re-freezing (use --force before any eval acts exist)")
        return json.loads(fz.read_text())
    L = res_last["L"]
    w = load_cfg("e1")["directions"]["layer_window"]
    lo, hi = round(w[0] * L), round(w[1] * L)
    cand = {r["layer"]: r["cos_fperpc_b"] for r in rows_last if lo <= r["layer"] <= hi}
    Ls = max(cand, key=cand.get)
    b_hat = unit(res_last["b"][Ls]).contiguous()
    bp = RESULTS / tag / "e1_b_hat.safetensors"
    save_file({"b_hat": b_hat}, str(bp), metadata={"L_star": str(Ls), "position": "P_last"})
    r = next(x for x in rows_last if x["layer"] == Ls)
    out = {"tag": tag, "L_star": Ls, "layer_window": [lo, hi], "position": "P_last", "L": L, "d": res_last["d"],
           "cos_fperpc_b_at_Lstar": r["cos_fperpc_b"], "null_p95": r["null_p95"], "null_p99": r["null_p99"],
           "null_p_value": r["null_p_value"], "b_hat_file": bp.name,
           "b_hat_sha256": hashlib.sha256(b_hat.numpy().tobytes()).hexdigest(),
           "n_extract": res_last["n"], "frozen_at": now(), "git": git_hash(),
           "note": "computed on extract_set only, before any MATH-500 activation was extracted"}
    fz.write_text(json.dumps(out, indent=1))
    log(f"FROZE L*={Ls} for {tag}: {out}")
    return out


def main(a):
    info = model_info(a.model)
    tag = info["tag"]
    dc = load_cfg("e1")["directions"]
    outd = RESULTS / tag
    outd.mkdir(parents=True, exist_ok=True)
    tables, res_by = {}, {}
    for pos in ("last", "tmpl"):
        A = load_position(tag, pos)
        res = compute(A, dc["null_draws"], dc["null_seed"], dc["split_half_draws"], dc["split_half_seed"], True)
        res_by[pos] = res
        tables[pos] = layer_table(res, res["null"], res["cos_fperpc_b"])
        save_file({"f": res["f"].contiguous(), "c": res["c"].contiguous(), "b": res["b"].contiguous(),
                   "b1": res["bk"][0].contiguous(), "b2": res["bk"][1].contiguous(), "b3": res["bk"][2].contiguous(),
                   "f_perp_c": res["f_perp_c"].contiguous()}, str(outd / f"e1_directions_{pos}.safetensors"),
                  metadata={"n": str(res["n"]), "uids_sha256": hashlib.sha256(json.dumps(A["_uids"]).encode()).hexdigest()})
        np.save(outd / f"e1_null_{pos}.npy", res["null"].numpy())
        log(f"{tag} P_{pos}: n={res['n']} max cos(f⊥c,b)={float(res['cos_fperpc_b'].max()):.3f}")
    # gen16 (FF/JS only): f at gen16 vs b at P_last (b is not defined at gen16)
    if (ACTS / tag / "extract" / "FF_gen16.safetensors").exists():
        A = load_position(tag, "gen16")
        res = compute(A, dc["null_draws"], dc["null_seed"], dc["split_half_draws"], dc["split_half_seed"], False)
        b_last = res_by["last"]["b"]
        obs = cos(res["f"], b_last)
        null = cos(res["null"], b_last)
        res["cos_fgen16_blast"] = obs
        tables["gen16"] = layer_table(res, null, obs)
        save_file({"f": res["f"].contiguous()}, str(outd / "e1_directions_gen16.safetensors"),
                  metadata={"n": str(res["n"])})
        np.save(outd / "e1_null_gen16.npy", null.numpy())
    for pos, rows in tables.items():
        (outd / f"e1_stats_{pos}.json").write_text(json.dumps({"tag": tag, "position": pos, "d": res_by["last"]["d"],
                                                               "inv_sqrt_d": 1 / np.sqrt(res_by["last"]["d"]),
                                                               "layers": rows}, indent=1))
    FIGS.mkdir(exist_ok=True)
    plot(tag, tables, res_by["last"]["d"], FIGS / f"e1_cos_by_layer_{tag}.png")
    freeze(tag, tables["last"], res_by["last"], a.force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
