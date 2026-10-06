# Running log: interpretations and deviations from the handoff

Each item: what the handoff said → what was done → why. (Folded into SUMMARY.md at the end.)

## Interpretations of under-specified points (no change to pre-registered criteria)
1. **C_k / D_k indexing.** k = 1..3 maps to `CONCISE[k-1]` / `DETAILED[k-1]`.
2. **Calibration sample (§6.4b).** A fresh `default_rng(1)` per model × condition, drawn over items in MATH-500 dataset order, with n = round(0.10 × #mv-correct).
3. **Self-consistency (§6.5.1).** The execution order (§10) audits each model before the other finishes, so the sample is 100 judged items **per generator model**, stratified proportionally across conditions with `default_rng(2)`.
4. **Audit sampling seed.** Not specified; `default_rng(2)` (`configs/e0.yaml:audit.sample_seed`). The 30 NOT_EQUIVALENT rows come from mv-wrong items, i.e. the ones whose labels the judge decides.
5. **Unspecified seeds.** Split-half reliability uses `default_rng(4)`; the random-direction control uses `default_rng(5)`.
6. **Judge verdict parsing.** If the text after `</think>` contains both EQUIVALENT and NOT_EQUIVALENT verdict lines, the item is `judge_failed` (`conflicting_verdicts`).
7. **P_gen16 vs b.** b is defined only at prompt positions (C/D/NC have no generations). At gen16 we report cos(f_gen16, b_P_last) with its own sign-flip null, and no ⊥c.
8. **Random-direction control (§8.5 iii).** The percentile is reported both in the predicted direction (signed: −ρ, +log-OR) and by absolute value. The criterion is evaluated on the signed version, and the |·| version is shown alongside.
9. **Smoke test outputs.** Written to `outputs/{tag}/smoke/` and not reused for the full run.
10. **Figures** carry a model-tag suffix (`figs/e0_acc_by_condition_{tag}.png`) because there are two models.

## Deviations
- **D0 – vLLM bitsandbytes is now a plugin, and torch is a CUDA 13 build (§2).** vLLM 0.31.0 (latest, installed) and 0.30.0 have no in-tree bitsandbytes. It moved to the official out-of-tree `vllm-bnb-plugin` (0.0.3, vllm-project org), which I installed. VERIFY: the plugin's in-flight path calls `quantize_4bit(..., compress_statistics=True, quant_type="nf4")`, i.e. NF4 + double quant with the bnb default blocksize 64, the same as the HF `BitsAndBytesConfig`. Load args are `quantization="bitsandbytes"` only, and the plugin sets `load_format` itself. vLLM 0.31.0 pins `torch==2.13.0`, whose default wheel is **CUDA 13.2** (`+cu132`), not 12.x. The handoff's intent ("CUDA build matching the installed vLLM wheel") is met; the literal "12.x" is not. The driver is 592.82 (CUDA 13.1), and the 13.2 runtime runs under minor-version compatibility. Python is 3.11.17 via uv.
- **D1 – LaTeX-escape repair in candidate extraction (§6.2).** In JSON conditions, models often write `\frac` unescaped inside a JSON string. That is a *valid* JSON escape (`\f` = form feed), so `json.loads` silently turns `\frac{1}{2}` into `<FF>rac{1}{2}`. We map such decoded control characters back to the LaTeX command when an ASCII letter follows immediately. This is flagged per item (`latex_escape_repaired`) and counted in the E0 report. Without it the JSON tax would partly measure a parser artifact. The same guard is applied to the judge's `tail` for invalid JSON (`\n` is not unescaped when followed by a letter, e.g. `\neq`).
