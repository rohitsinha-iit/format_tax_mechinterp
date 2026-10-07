# E3 dev calibration: qwen3-4b-nf4

_2026-10-07T16:02:57+00:00 · git f14ddae4cff48a8da3dfb22cc651646b40fa72dc-dirty · 100 reserved dev problems, greedy, max 3072._

| setting | median out tok | median Δtok vs unsteered (paired) | median reasoning tok | mv acc | lenient valid | trunc | degenerate | hook |
|---|---|---|---|---|---|---|---|---|
| dev_FF_none_a0 | 614 | +0 | nan | 78% |  | 9% | 0% | OK |
| dev_FF_plus_b_a0.5 | 510 | -94 | nan | 80% |  | 3% | 0% | OK |
| dev_FF_plus_b_a1 | 438 | -155 | nan | 77% |  | 4% | 0% | OK |
| dev_FF_plus_b_a1.5 | 373 | -192 | nan | 60% |  | 5% | 0% | OK |
| dev_FF_plus_b_a2 | 332 | -244 | nan | 41% |  | 4% | 0% | OK |
| dev_FF_plus_r0_a1 | 600 | -22 | nan | 76% |  | 7% | 0% | OK |
| dev_FF_plus_r1_a1 | 548 | -46 | nan | 67% |  | 8% | 0% | OK |
| dev_FF_plus_r2_a1 | 618 | +0 | nan | 70% |  | 10% | 0% | OK |
| dev_JS_minus_b_a0.5 | 612 | +124 | 400 | 67% | 11% | 9% | 0% | OK |
| dev_JS_minus_b_a1 | 1266 | +371 | 700 | 63% | 2% | 19% | 1% | OK |
| dev_JS_minus_f_a1 | 3072 | +2653 | nan | 0% | 0% | 100% | 2% | OK |
| dev_JS_minus_r0_a1 | 472 | +15 | 202 | 74% | 22% | 2% | 0% | OK |
| dev_JS_none_a0 | 419 | +0 | 250 | 73% | 28% | 3% | 0% | OK |

## Safety rule for α = 1 (pre-specified)

- JS|minus_b: degenerate +1.0 pp, validity drop 26.0 pp → FAIL
- FF|plus_b: degenerate +0.0 pp → PASS

**necessity (JS−b): α = 1 is NOT safe → arm not run on eval**

**sufficiency (FF+b): α = 1 is safe**

User decision (2026-10-07): only the sufficiency arm runs on MATH-500. JS−b also failed the rule at α = 0.5 (validity drop 17 pp). The JS−b α = 1.5 run was stopped at 81/100 and is excluded (`dev_JS_minus_b_a1.5.partial_stopped`); JS α ∈ {1.5, 2} and the JS random controls r̂₁/r̂₂ were dropped as moot.
