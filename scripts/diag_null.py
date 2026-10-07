"""EXPLORATORY (not pre-registered): diagnose the width of the §8.4.3 sign-flip null.

Hypothesis: with a strong mean shift f (split-half cos ≈ 0.999), f_null = mean_i(s_i d_i) =
s̄·f + mean_i(s_i (d_i − f)). |s̄| ~ 1/√n is tiny but still dominates the *direction* of f_null, so
cos(f_null⊥c, b) ≈ ±cos(f⊥c, b) and the null p99 tracks the observed statistic at every layer.
Check: (1) correlation of cos_null with s̄; (2) a balanced flip null (exactly n/2 of each sign,
s̄ = 0) and a centred null (flip the centred differences d_i − f), which remove that term.

  python scripts/diag_null.py --model qwen3-4b   -> results/{tag}/e1_null_diagnostic.json
"""
import argparse
import json

import numpy as np
import torch

from src.common import RESULTS, load_cfg, model_info
from src.directions import cos, load_position, perp


def main(model):
    tag = model_info(model)["tag"]
    dc = load_cfg("e1")["directions"]
    A = load_position(tag, "last")
    D = A["JS"] - A["FF"]
    n = D.shape[0]
    f = D.mean(0)
    c = (A["NC"] - A["FF"]).mean(0)
    b = torch.stack([(A[f"C{k}"] - A[f"D{k}"]).mean(0) for k in (1, 2, 3)]).mean(0)
    obs = cos(perp(f, c), b)
    K = dc["null_draws"]
    rng = np.random.default_rng(dc["null_seed"])
    S = torch.from_numpy(rng.choice([-1.0, 1.0], size=(K, n)).astype(np.float32))
    rng_b = np.random.default_rng(dc["null_seed"] + 100)
    Sb = torch.from_numpy(np.stack([rng_b.permutation(np.r_[np.ones(n // 2), -np.ones(n - n // 2)])
                                    for _ in range(K)]).astype(np.float32))
    Dc = D - f

    def null_of(Smat, X):
        out = []
        for i in range(0, K, 100):
            fn = torch.einsum("kn,nld->kld", Smat[i:i + 100], X) / n
            out.append(cos(perp(fn, c), b))
        return torch.cat(out)

    spec, bal, cen = null_of(S, D), null_of(Sb, D), null_of(S, Dc)
    sbar = S.mean(1)
    rows = []
    for l in range(D.shape[1]):
        r = np.corrcoef(sbar.numpy(), spec[:, l].numpy())[0, 1] if l > 0 else float("nan")
        rows.append({"layer": l, "obs_cos_fperpc_b": float(obs[l]),
                     "spec_null_p99": float(spec[:, l].quantile(.99)),
                     "corr_spec_null_with_sbar": float(r),
                     "balanced_null_p99": float(bal[:, l].quantile(.99)),
                     "centred_null_p99": float(cen[:, l].quantile(.99)),
                     "balanced_p_value": float(((bal[:, l] >= obs[l]).sum() + 1) / (K + 1)),
                     "centred_p_value": float(((cen[:, l] >= obs[l]).sum() + 1) / (K + 1))})
    fz = json.loads((RESULTS / tag / "e1_frozen.json").read_text())
    out = {"tag": tag, "EXPLORATORY": True, "L_star": fz["L_star"], "n": n, "draws": K,
           "Lstar_row": rows[fz["L_star"]], "layers": rows}
    (RESULTS / tag / "e1_null_diagnostic.json").write_text(json.dumps(out, indent=1))
    for r in rows[::3] + [rows[fz["L_star"]]]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    main(ap.parse_args().model)
