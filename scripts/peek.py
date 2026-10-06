"""Dev helper: summarise *_mv.jsonl files in a directory and show short samples.
  python scripts/peek.py outputs/qwen3-4b-nf4/smoke [n_samples]"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
for p in sorted(d.glob("e0_*_mv.jsonl")):
    rows = [r for r in map(json.loads, open(p, encoding="utf-8")) if "_meta" not in r]
    n = len(rows)
    acc = sum(r["mv_correct"] for r in rows) / n
    toks = sorted(r["output_token_ids_len"] for r in rows)
    sv = [r["strict_valid"] for r in rows if r["strict_valid"] is not None]
    lv = [r["lenient_valid"] for r in rows if r["lenient_valid"] is not None]
    print(f"== {p.name}: n={n} mv_acc={acc:.2f} median_tok={toks[n // 2]} "
          f"strict={sum(sv)}/{len(sv)} lenient={sum(lv)}/{len(lv)} "
          f"trunc={sum(r['finish_reason'] == 'length' for r in rows)} "
          f"methods={sorted(set(r['extract_method'] for r in rows))} "
          f"repaired={sum(r['latex_escape_repaired'] for r in rows)}")
    for r in rows[:k]:
        print(f"  [{r['uid']}] gold={r['gold']!r} cand={r['candidate']!r} ok={r['mv_correct']} tok={r['output_token_ids_len']}")
        t = r["output_text"]
        print("   ", repr(t[:300]), "...", repr(t[-200:]) if len(t) > 300 else "")
