# E0 — format tax replication: qwen3-4b-nf4

_Generated 2026-10-07T03:18:39+00:00 · git 9993bd94147da552a99c039fce891c794eacd121-dirty · bootstrap B=10000, default_rng(0) · paired over n=500 MATH-500 uids._

## Per-condition table

| cond | n | acc (final) | acc (mv only) | strict_valid | lenient_valid | out tok median [IQR] | reasoning tok median [IQR] | trunc. rate | mv-wrong judged | judge→correct | judge_failed | extract none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FF | 500 | 81.4% | 79.4% | — | — | 558 [365–1034] | — | 8.2% | 103 | 10 | 20 | 37 |
| JS | 500 | 76.6% | 72.2% | 0.292 | 0.294 | 380 [214–796] | 176 [128–252] | 5.2% | 139 | 22 | 24 | 27 |
| JC | 500 | 62.2% | 59.0% | 0.834 | 0.834 | 364 [208–736] | 293 [178–459] | 16.6% | 205 | 16 | 16 | 83 |

## Tax = acc(FF) − acc(X) (pp), paired bootstrap 95% CI

| X | tax (final) | 95% CI | tax (mv only) | 95% CI |
|---|---|---|---|---|
| JS | +4.80 | [+1.40, +8.20] | +7.20 | [+3.40, +11.00] |
| JC | +19.20 | [+15.00, +23.40] | +20.40 | [+16.00, +24.80] |

## Exact McNemar (final_correct)

| pair | both ✓ | A✓ B✗ | A✗ B✓ | both ✗ | p |
|---|---|---|---|---|---|
| FF vs JS | 355 | 52 | 28 | 65 | 0.00968 |
| FF vs JC | 293 | 114 | 18 | 75 | 2.98e-18 |
| JS vs JC | 295 | 88 | 16 | 101 | 3.18e-13 |

## Flip analysis FF → JS (flip = FF ✓ & JS ✗): 52 flips, 28 rescues; median Δtok = -140

| Δtok decile | Δtok range | n | flips | P(flip) | n FF✓ | P(flip | FF✓) |
|---|---|---|---|---|---|---|
| 1 | -2791..-729 | 50 | 5 | 10.0% | 12 | 41.7% |
| 2 | -683..-329 | 50 | 10 | 20.0% | 43 | 23.3% |
| 3 | -329..-240 | 50 | 7 | 14.0% | 39 | 17.9% |
| 4 | -239..-179 | 50 | 5 | 10.0% | 48 | 10.4% |
| 5 | -178..-140 | 50 | 6 | 12.0% | 45 | 13.3% |
| 6 | -140..-110 | 50 | 4 | 8.0% | 49 | 8.2% |
| 7 | -109..-71 | 50 | 3 | 6.0% | 49 | 6.1% |
| 8 | -71..-25 | 50 | 2 | 4.0% | 49 | 4.1% |
| 9 | -24..69 | 50 | 2 | 4.0% | 35 | 5.7% |
| 10 | 75..2664 | 50 | 8 | 16.0% | 38 | 21.1% |

| level | n | flips | rescues | P(flip) | n FF✓ | P(flip | FF✓) |
|---|---|---|---|---|---|---|
| 1 | 43 | 0 | 1 | 0.0% | 40 | 0.0% |
| 2 | 90 | 4 | 5 | 4.4% | 82 | 4.9% |
| 3 | 105 | 10 | 3 | 9.5% | 95 | 10.5% |
| 4 | 128 | 15 | 9 | 11.7% | 107 | 14.0% |
| 5 | 134 | 23 | 10 | 17.2% | 83 | 27.7% |

## Gate (§7.3)

- tax(FF−JS) = +4.80 pp (threshold ≥ 3.0), exact McNemar p = 0.00968 (threshold < 0.05)
- **gate: PASS**

## Truncation

- ⚠️ conditions with truncation > 5%: FF (8.2%), JS (5.2%), JC (16.6%) — truncation biases the tax.

## Notes

- Final label = math-verify if correct, else the Qwen3-8B thinking judge verdict (pre-registered rule §6.4); judge failures count as incorrect.
- Known limitation: for Qwen3-8B outputs the judge is the same model. Mitigations: math-verify is primary, the judge task is narrow equivalence, inputs are format-blind, and the human audit is a gate.
- JSON-decoded control characters that came from LaTeX commands (e.g. `\f`rac → form-feed) are repaired in the extracted candidate (`latex_escape_repaired`; counts in the table summary JSON).
- Figures: `figs/e0_acc_by_condition_qwen3-4b-nf4.png`, `figs/e0_flip_by_dtok_decile_qwen3-4b-nf4.png`.
