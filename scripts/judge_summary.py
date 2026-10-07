"""Dev helper: verdict / failure / length summary of judge files.
  python scripts/judge_summary.py outputs/qwen3-4b-nf4/judge"""
import json
import sys
from collections import Counter
from pathlib import Path

for p in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    rows = [r for r in map(json.loads, open(p, encoding="utf-8")) if "_meta" not in r]
    toks = sorted(r["judge_output_tokens"] for r in rows)
    print(f"== {p.name}: n={len(rows)}")
    for role in sorted({r["role"] for r in rows}):
        rr = [r for r in rows if r["role"] == role]
        print(f"   {role:9s} n={len(rr):4d} verdicts={dict(Counter(r['judge_verdict'] for r in rr))} "
              f"failed={dict(Counter(r['fail_reason'] for r in rr if r['judge_failed']))}")
    print(f"   judge output tokens: median={toks[len(toks) // 2]} p90={toks[int(.9 * len(toks))]} max={toks[-1]}; "
          f"tail fallback used={sum(r['tail_chars_used'] != 800 for r in rows)}")
