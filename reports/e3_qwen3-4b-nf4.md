# E3 — steering with the brevity direction: qwen3-4b-nf4

_2026-10-07T20:49:39+00:00 · git ae7f2405b78f0e11df63a1f0d6b028abc7e5b2b2-dirty · frozen 2026-10-07T16:03:04+00:00 · L* = 22, α = 1 (k = 15.53), all positions, vLLM + hook (prefix caching off)._

## Necessity arm (JS − k·b̂): not run on MATH-500

NOT RUN on eval: JS−b failed the pre-specified dev safety rule at α=0.5 and α=1 (lenient-validity drop 17 and 26 pp > 5 pp; markdown style reinstated). User decision 2026-10-07. See `reports/e3_dev_qwen3-4b-nf4.md`.

## Sufficiency arm (FF + k·b̂) on MATH-500, n = 500

| cond | acc (final) | acc (mv only) | median out tok | truncation |
|---|---|---|---|---|
| FF | 81.4% | 79.4% | 558 | 8.2% |
| JS | 76.6% | 72.2% | 380 | 5.2% |
| E3-S | 75.6% | 72.4% | 405 | 7.4% |
| E3-Sr | — | 76.0% | 552 | 8.8% |

E3-S = FF + k·b̂ · E3-Sr = FF + k·r̂₀ (random control, math-verify labels only) · FF/JS = unsteered E0 outputs.

## Pre-registered criteria (approved 2026-10-07)

| criterion | result | met |
|---|---|---|
| S1 output tokens FF+b < FF | median Δ -112, mean Δ -198, n=500, p=1.3e-44 | |
| S1 … and FF+b < FF+random | median Δ -102, mean Δ -174, n=500, p=6.24e-45 | **True** |
| S2 acc(FF) − acc(FF+b) > 0 | +5.80 pp [+2.20, +9.40], McNemar p = 0.00219 | **True** |

## Supporting numbers

- share of the FF→JS compression reproduced by FF + k·b̂ (mean output tokens): 82%
- random control vs FF (tokens): median Δ -5, mean Δ -24, n=500, p=0.033
- JS tax for reference: +4.8 pp (E0)
- math-verify-only accuracy differences vs FF: E3-S -7.0 pp [-10.8, -3.2]; E3-Sr -3.4 pp [-6.6, -0.4]
- McNemar table FF vs FF+b (both ✓, FF✓ only, FF+b✓ only, both ✗): [[350, 57], [28, 65]]
