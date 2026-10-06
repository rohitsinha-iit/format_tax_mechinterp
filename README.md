# format-tax-mi — E0 (format-tax replication) + E1 (format/brevity directions)

Implements `HANDOFF — Format-Tax MechInterp: E0 + E1`. Everything runs inside WSL2 (Ubuntu 24.04) in the
venv `~/ftmi-venv` (Python 3.11). One model on the GPU at a time; every stage is a separate process.

```bash
bash scripts/setup_env.sh && bash scripts/env_report.sh       # env + reports/env.txt, requirements.lock
~/ftmi-venv/bin/python -m src.data splits                       # data/extract_set.jsonl, data/dev_set_ids.json
~/ftmi-venv/bin/python -m pytest -s tests/test_templates.py tests/test_extraction.py
~/ftmi-venv/bin/python -m src.gen_vllm --model qwen3-4b --task e0 --conds FF,JS,JC --limit 10 --subdir smoke
bash scripts/run_e0.sh qwen3-4b phase1     # -> STOP: label reports/audit_qwen3-4b-nf4.csv
bash scripts/run_e0.sh qwen3-4b phase2     # after audit acceptance: analysis + gate
bash scripts/run_e0.sh qwen3-4b twoturn
python -m tests.test_parity --model qwen3-4b --stage vllm && python -m tests.test_parity --model qwen3-4b --stage hf
bash scripts/run_e1.sh qwen3-4b
```

Background jobs: `bash scripts/bg.sh <name> <cmd...>` (log in `~/ftmi_logs/<name>.log`), `bash scripts/status.sh`.

| path | content |
|---|---|
| `prompts/templates.py` | canonical templates (verbatim from the handoff) |
| `prompts/judge_prompt.txt` | canonical judge prompt (verbatim) |
| `prompts/neutral_final.json` | length-matched neutral block (frozen by `tests/test_templates.py`) |
| `outputs/{tag}/e0_{C}.jsonl` | raw generations; `_mv.jsonl` math-verify scored; `_scored.jsonl` final labels |
| `outputs/{tag}/judge/` | judge verdicts (seed 0) and self-consistency reruns |
| `acts/{tag}/` | activation dumps (float16 safetensors; git-ignored) |
| `results/{tag}/` | directions, nulls, frozen L*, summaries |
| `reports/` | env, logs, audit CSVs, E0/E1 reports, `DEVIATIONS.md`, `SUMMARY.md` |
