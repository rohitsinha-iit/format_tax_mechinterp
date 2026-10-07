# Format-tax MechInterp — E0 + E1 summary

Scope: **Qwen3-4B (NF4, non-thinking) only.** The user dropped Qwen3-8B as a generator because it does not fit in vLLM on the 8 GB GPU (D3). Qwen3-8B NF4 serves only as the judge, run in HF transformers (D4). All deviations are listed in [`DEVIATIONS.md`](DEVIATIONS.md).

## E0 — the format tax replicates (gate PASS)

MATH-500, n = 500, paired by uid. Label = math-verify, else the 8B thinking judge (§6.4).

| cond | accuracy | math-verify only | tax vs FF (pp) [95% paired-bootstrap CI] | exact McNemar vs FF | median output tok | truncation |
|---|---|---|---|---|---|---|
| FF | 81.4% | 79.4% | — | — | 558 | 8.2% ⚠️ |
| JS | 76.6% | 72.2% | **+4.80 [+1.40, +8.20]** | p = 0.0097 | 380 | 5.2% ⚠️ |
| JC | 62.2% | 59.0% | +19.20 [+15.0, +23.4] | p = 3e-18 | 364 | 16.6% ⚠️ |
| 2T | 82.4% | 80.2% | −1.00 [−2.20, 0.00] | p = 0.125 | 201 | 0.0% |

- **Gate:** tax(FF−JS) = 4.80 pp ≥ 3.0 and p < 0.05 → **PASS**.
- **Where the tax comes from:** asking for JSON *while reasoning* costs accuracy (JS), and constrained decoding on top of that costs much more for this small NF4 model (JC). Reformatting finished reasoning costs nothing (2T). The JS-vs-2T contrast agrees with Lee et al.'s prompt-driven account. The large JC loss differs from their finding that constrained decoding is mostly harmless; here JC also has 16.6% truncation, including repetition loops under the grammar.
- **FF→JS flips:** 52 flips vs 28 rescues. Flip rate rises with difficulty (P(flip | FF ✓): level 1 0% → level 5 28%). Flip rate by Δtok decile is **U-shaped** (41.7% in the most-compressed decile, 21.1% in the most-expanded), replicating Lee et al.
- **Judge validation:** the human audit agreed with the judge on 98/98 items (κ = 1.0; 48 flip items). Self-consistency was 0.78 three-way across seeds 0/1/2, rising to 0.97 when restricted to runs that all returned a verdict; 24/100 triples had at least one run hit the 2,048-token thinking cap. The judge hit that cap on 83 mv-wrong items across conditions, and these count as incorrect per §6.4.
- **Caveat:** truncation exceeds 5% in FF, JS and JC (`max_tokens` = 3072, not changed). It biases the taxes, mostly upward for JC.

## E1 — directions and H1

- **Brevity check:** median tokens C₁/D₁ = 224.5 / 525.5 = **0.43** ≤ 0.6, PASS.
- **Direction estimates:** the format direction f and the brevity direction b are highly reliable (split-half cos ≈ 0.999). The three brevity phrasings agree (cos(b⁽ʲ⁾, b⁽ᵏ⁾) = 0.88–0.93 at L*). The neutral control is nearly orthogonal to b in middle/late layers (cos(c, b) ≈ −0.01 at L*).
- **L\* = 22** (P_last, window 9–31), frozen on extract_set before any eval activation was extracted. cos(f⊥c, b) = 0.385 (1/√d = 0.020).
- **Parity:** vLLM vs HF top-1 agreement 40/40 (mean KL 0.004).

| H1 criterion (pre-registered) | result | met |
|---|---|---|
| (i) cos(f⊥c, b) at L* > sign-flip null p99 | 0.385 vs 0.399 | **no** |
| (ii-a) Spearman(s, Δtok) < 0, CI excludes 0 | −0.051 [−0.143, 0.044] | no |
| (ii-b) flip OR per SD > 1, p < 0.05 | 0.98 [0.71, 1.35], p = 0.91 | no |
| (iii) beats random-direction p95 | Spearman 70.5th pct, log-OR 45.7th pct | no |

**H1 status: not supported for Qwen3-4B.** A question's projection onto the brevity direction predicts neither its JSON compression (Δtok) nor its correct→wrong flips, and it does no better than random directions.

**Exploratory (not pre-registered; verdict unchanged):** the pre-registered sign-flip null is miscalibrated when f is strong (D9). Its draws correlate r ≈ 0.89 with the mean of the random signs, so its p99 tracks the observed value at every layer. Balanced or centred nulls give p99 ≈ 0.26 and **p = 0.001** at L*. So the format residual *does* share direction with brevity at the population level, but that shared component does not predict per-question behaviour. If criterion (i) is reused in E3, its null should be revised (needs your approval).

## E3 — steering with the brevity direction (plan: `docs/E3_plan.md`; reports `e3_dev_*.md`, `e3_*.md`)

The steering vector is added to the residual stream after block L* = 22 at every position. k = f·b̂ = 15.5 is the push the JSON prompt adds along b̂. The setup was verified before any eval run:
- **Manipulation check:** −k·b̂ removes 97% of the JS–FF gap at L* and 63–89% downstream.
- **Steered parity:** the vLLM and HF hooks agree on the top-1 next token for 40/40 prompts.
- **Hook coverage:** the hook token counters cover every prompt and generated token.
- **Prefix caching** is off for all steered runs.

**Necessity arm (JS − k·b̂): blocked by the pre-specified dev safety rule.** On dev it lengthened JS reasoning (419 → 612 tokens at α = 0.5, 1,266 at α = 1), but lenient JSON validity fell 17 and 26 pp (limit 5). b̂ also carries "terse, unformatted" style: subtracting it brings FF's markdown back inside the JSON string. By user decision it was not run on MATH-500, so necessity is **untested**, not refuted.

**Sufficiency arm (FF + k·b̂, α = 1), MATH-500, n = 500:**

| cond | acc (final) | acc (mv) | median tok | right→wrong vs FF (mv) |
|---|---|---|---|---|
| FF | 81.4% | 79.4% | 558 | — |
| JS | 76.6% | 72.2% | 380 | 67 |
| **FF + k·b̂** | **75.6%** | 72.4% | **405** | 67 |
| FF + k·r̂₀ (random, mv only) | — | 76.0% | 552 | 40 |

- **S1 met.** FF + k·b̂ is shorter than FF (median −112 tokens, p = 1e-44) and shorter than the random push (p = 6e-45). It reproduces 82% of JSON's mean compression; the random push reproduces almost none (median −5).
- **S2 met.** acc(FF) − acc(FF + k·b̂) = **+5.8 pp [+2.2, +9.4]**, McNemar p = 0.002, comparable to the JSON tax (+4.8 pp).
- **Caveat — specificity of the accuracy loss.** Against the random control on math-verify labels the difference is −3.6 pp [−7.2, 0.0], p = 0.063. So the length effect is clearly specific to b̂, but that the *accuracy* loss exceeds generic perturbation damage is not established. Judge failures (32/138, all genuinely wrong on inspection) count as incorrect.
- **Dev dose–response.** Tokens fall monotonically with α (614 → 332 at α = 2). Accuracy holds to α = 1 (78/80/77%), then collapses (60% at α = 1.5, 41% at α = 2).

**Reading.** Adding only the format-induced brevity push to a plain prompt makes Qwen3-4B behave like it does under the JSON prompt: about as short, about as accurate, with the same number of flips. So brevity is *sufficient* to reproduce the format tax's behavioural signature. Combined with E1 (the push is nearly constant across questions, so it can't predict *which* items flip), the picture is a population-level brevity bias that costs accuracy mainly on hard items that needed the room. Whether it is *necessary* remains open: removing it from JS also removes format discipline, so a cleaner, style-matched brevity direction (e.g. JS+concise vs JS+detailed) would be needed to test that.

## Deviations (details in DEVIATIONS.md)
D0 vLLM 0.31 bnb via the official `vllm-bnb-plugin` (NF4 + double quant verified); torch is a CUDA 13.2 build · D2 `gpu_memory_utilization` 0.80 (6.93 GiB free under WDDM) · D3 8B dropped as generator (user) · D4 judge in HF transformers, batch size 1 (user) · D5 FORMAT_BLOCK/REFORMAT_BLOCK revised by the user after schema echo; NEUTRAL_BLOCK then length-matched without padding · D6 2T at max_model_len 7168 (user) · D7 HF decoder layers return a tensor; hooks verified against hidden_states · D8 49/300 JS gen16 items excluded (no scaffold) · D9 null miscalibration (exploratory only) · D10 48/50 flip items available for audit · D1 LaTeX control-character repair in JSON candidates (5 items) · plus interpretations 1–10 (seeds, indexing, per-model self-consistency, figure names).

Not done (out of scope after D3): any Qwen3-8B generator results, which means no 8B rows in E0/E1. The dev set (100 ids) was used only for E3 calibration.
