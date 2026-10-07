# E3 plan: causal steering with the brevity direction (Qwen3-4B, L* = 22)

Status: **steps 1–3 done (2026-10-07); calibration waits on decisions D-E3-1…4 (see end).**

## Steps 1–3 results
- **Vectors** (`results/qwen3-4b-nf4/e3_vectors.safetensors`): L* = 22, k = 15.53, ‖f‖ = 41.94, random r̂ seeds 100–104 (⊥ b̂, f̂).
- **Manipulation check** (HF, 20 held-out dev prompts; `results/qwen3-4b-nf4/e3_manip_check.json`):
  - Mean projection on unit b_ℓ at the last prompt token:

    | ℓ | FF | JS | JS−b | FF+b | C₁ | JS−r₀ | JS–FF gap removed by −b |
    |---|---|---|---|---|---|---|---|
    | 22 (L*) | −14.8 | 1.2 | −14.3 | 0.8 | 13.1 | 1.2 | 97% |
    | 24 | −31.1 | −7.8 | −24.3 | −11.2 | 5.0 | −8.2 | 71% |
    | 27 | −26.8 | −1.8 | −24.0 | 4.0 | 29.4 | 0.0 | 89% |
    | 31 | −42.3 | −6.1 | −34.3 | 9.3 | 48.2 | −0.4 | 78% |
    | 36 | −52.1 | −4.9 | −34.5 | 32.6 | 87.1 | −1.4 | 63% |

  - The JS–FF shift on held-out dev problems (16.0) matches the extraction-set k (15.5).
  - Steering persists downstream: 63–89% of the gap stays removed, and later layers only partly re-impose it.
  - FF+b moves toward C₁. The random control leaves the b-projection unchanged.
- **Steered parity** (`reports/e3_steer_parity_qwen3-4b-nf4.md`): vLLM-hook vs HF-hook top-1 agreement **40/40** on steered prompts (JS−b KL 0.004, FF+b KL 0.00002); unsteered reference 19/20. The vLLM hook steered exactly every submitted prompt token (prefix caching **off** for all steered runs, so cached KV from another steering setting is never reused). **PASS → vLLM is usable for steered generation.**
- Still to verify at the start of calibration: the vLLM hook also fires on decode steps (token counter = prompt + generated tokens). Written 2026-10-07 after E0/E1
(`reports/SUMMARY.md`). Nothing here is pre-registered until the user approves the criteria.

## Grounding from E1 (last prompt token, layer 22, projection on the frozen b̂)

| prompt | mean projection on b̂ | SD across questions |
|---|---|---|
| D₁ ("be thorough") | −20.6 | 0.6 |
| FF | −15.4 | 1.6 |
| **JS** | **+0.2** | 2.7 |
| C₁ ("be concise") | +11.4 | 4.4 |

- The JSON prompt moves the projection by **k = f·b̂ = +15.5**, about half the concise–thorough gap.
- The push is nearly the same for every question (SD 2.7), which explains why the E1 per-question test was null.
  Causal tests ask the question correlation could not: does the *average* push matter?
- ‖f‖ = 41.9, ‖b‖ = 32.4, ‖c‖ = 10.8. Residual norm at layer 22 is ≈ 66, so a vector of size k is ≈ 23% of it.

## Hypotheses
- **H-N (necessity):** removing the brevity push from JS lengthens reasoning and recovers part of the 4.8 pp tax.
- **H-S (sufficiency):** adding the same push to FF shortens reasoning and costs accuracy, with no JSON request.

## Interventions
Add a fixed vector to the residual stream after block 22 (`model.model.layers[21]` output), at every position
(prompt and generated tokens). A random direction r̂ is Gaussian, Gram–Schmidt-orthogonalised against b̂ and f̂,
and scaled to the same norm.

| ID | condition | steering vector | tests | set |
|---|---|---|---|---|
| E3-N | JS | −α·k·b̂ | necessity | MATH-500 |
| E3-S | FF | +α·k·b̂ | sufficiency | MATH-500 |
| E3-Nr | JS | −α·k·r̂ | perturbation-damage control | MATH-500 |
| E3-Sr | FF | +α·k·r̂ | control | MATH-500 |
| E3-F | JS | −α·f (whole format shift) | positive control / upper bound | dev only |
| dose | α ∈ {0.5, 1, 1.5, 2} for N and S | — | dose–response, safety | dev only |

- **Primary dose α = 1**, meaning "undo exactly what the JSON prompt adds along b̂".
- **Optional secondary scope:** prompt-only steering (HF only; vLLM batches mix prefill and decode tokens).

## Data discipline
1. Calibrate on the **dev set** (100 MATH-train problems reserved in E1):
   - dose grid and random controls;
   - repetition/degeneration rate and JSON validity;
   - manipulation check that the projection on b̂_ℓ moves in layers 23–36.
2. Freeze `results/qwen3-4b-nf4/e3_frozen.json` (α, random seeds, hook spec) **before** any steered MATH-500 run.
3. Run the eval conditions through the E0 pipeline (scoring → judge → final label). Unsteered FF/JS are the existing E0 outputs.

## Outcomes and proposed criteria (await approval)
Length is the well-powered primary outcome. Accuracy is secondary: a full 4.8 pp recovery is detectable on 500
paired items, a half recovery likely is not (power ~30–40%).

| criterion | test |
|---|---|
| N1 (primary) | reasoning-field tokens JS−b > JS, paired Wilcoxon p < 0.05, and effect > JS−random. Report the share of FF→JS compression recovered = (len(JS−b) − len(JS)) / (len(FF) − len(JS)). |
| N2 (secondary) | acc(JS−b) − acc(JS) > 0, paired bootstrap 95% CI + exact McNemar |
| N3 (validity) | lenient JSON validity of JS−b within 5 pp of JS; otherwise N1/N2 are uninterpretable |
| S1 (primary) | tokens FF+b < FF (p < 0.05), with an effect larger than FF+random |
| S2 (secondary) | acc(FF+b) < acc(FF) with CI; compare with the 4.8 pp JS tax |
| manipulation | projection on b̂ at the last prompt token moves by ≈ ±α·k in the steered run, logged per question |

## Implementation steps
1. `src/steer.py`: vectors (b̂, k, random r̂, f) and hooks for both backends. vLLM's Qwen3 layer returns
   `(mlp_out, residual)` with the add fused into the next layer's RMSNorm, so the vector is added to the returned
   `residual` (stream = mlp_out + residual). `enforce_eager=True` keeps the hooks live. The hook is installed via
   `LLM.apply_model`, with the engine in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`).
2. Manipulation check (HF, dev set): projection of r_ℓ on b̂_ℓ, ℓ = 22…36, for JS, JS−b and FF (likewise FF+b vs C₁).
3. Steered parity: vLLM vs HF next-token distribution under the same steering on 20 dev prompts × {JS−b, FF+b};
   top-1 agreement ≥ 0.90 is required to use vLLM for steered generation.
4. Dev calibration → freeze → eval generation → scoring/judge → `src/analysis_e3.py` → `reports/e3_qwen3-4b-nf4.md`.

## Compute (estimates)
- Generation: ~3 h with vLLM + hook (~10 h HF-only) for the 4 eval conditions; dev calibration ~1.5 h (vLLM).
- Judge (HF 8B, ~1.1 min/item): ~3–4 h per fully judged condition. This dominates.

## Decisions (user, 2026-10-07)
- **D-E3-1:** vLLM + hook.
- **D-E3-2:** judge only E3-N and E3-S; the random controls E3-Nr/E3-Sr get math-verify labels only (accuracy comparisons that involve the controls use math-verify on both sides).
- **D-E3-3:** criteria N1–N3, S1–S2 approved; α = 1 primary.
  - **N1 measurement:** only ~29% of JS outputs are strictly valid JSON (LaTeX escapes), so the reasoning field is measured with tolerant span extraction: JSON parse if possible, else the raw `"reasoning": "…"` span up to `", "answer":`, or to the end. The strictly-valid subset is reported alongside.
- **D-E3-4:** not answered. Criterion (i) is not part of E3 and is not reported there.
- **Random control on eval:** r̂₀ (seed 100). r̂₁ and r̂₂ are used on dev only.
- **Safety rule for α = 1** (fixed before any steered output was inspected): on dev, for JS−b and FF+b, the degenerate rate (truncated with a repetitive or blank tail) rises by ≤ 5 pp over unsteered, and JS−b lenient validity drops by ≤ 5 pp. Otherwise STOP and report.

## Dev calibration, first readout (2026-10-07; full table in reports/e3_dev_qwen3-4b-nf4.md when done)
- **Hook on decode steps verified:** dev_JS_minus_b_a1 hook tokens 167,732 ≥ prompt + fed-back generated tokens 167,651 (the excess is vLLM preemption recompute).
- **α = 1 necessity arm FAILS the safety rule.** JS−k·b̂ on dev:

  | measure | JS | JS−k·b̂ |
  |---|---|---|
  | median output tokens (FF: 614) | 419 | 1,266 |
  | median reasoning tokens | 250 | 700 |
  | truncation | 3 | 19 (1 degenerate) |
  | lenient JSON validity | 28% | 2% (drop 26 pp; limit 5 pp) |
  | mv accuracy | 73% | 63% |

- **Why validity collapses:** steering −b̂ reinstates FF *style* inside the JSON string. Markdown headers/bold in the reasoning rise 19 → 61 (FF: 100) and raw newlines 48 → 81 (FF: 100), which strict JSON rejects. The JSON scaffold itself survives (84 vs 76 outputs start with `{"reasoning": "`). So b̂ carries "terse / unformatted" style as well as length. Subtracting it overshoots length (≈ 3× JS, 2× FF) and undoes the format's stylistic discipline, which confounds the necessity test at this dose.
- Per the rule, the freeze is blocked and nothing touches MATH-500. The rest of the dev grid (FF+b, α ∈ {0.5, 1.5, 2}, random controls, E3-F) finishes before options go to the user.

## Decisions as originally posed
- **D-E3-1 backend:** vLLM + hook (requires step 3 to pass) vs HF-only.
- **D-E3-2 eval scope:** judge all four conditions, or judge E3-N/E3-S only and score the random controls with math-verify only.
- **D-E3-3:** approve criteria N1–N3, S1–S2 and α = 1 as primary.
- **D-E3-4:** if criterion (i) is reported again, use the balanced sign-flip null (E1 D9).
