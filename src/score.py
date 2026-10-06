"""E0 scoring (§6.1–6.3): JSON validity, candidate extraction, math-verify. Plus the final
merge with judge verdicts (§6.4 final label).

  python -m src.score --model qwen3-4b --stage mv     [--conds FF,JS,JC]  -> e0_{C}_mv.jsonl
  python -m src.score --model qwen3-4b --stage final  [--conds FF,JS,JC]  -> e0_{C}_scored.jsonl
"""
from __future__ import annotations

import argparse
import json
import re

from src.common import OUTPUTS, load_cfg, model_info, read_jsonl, write_jsonl

JSON_CONDS = ("JS", "JC", "2T")

# ----------------------------------------------------------------------------- JSON validity

_FENCE_OPEN = re.compile(r"^\s*```(?:json)?[ \t]*\n?", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\n?```\s*$")


def _as_obj(s: str):
    try:
        o = json.loads(s)
    except (json.JSONDecodeError, RecursionError, ValueError):
        return None
    if isinstance(o, dict) and isinstance(o.get("reasoning"), str) and isinstance(o.get("answer"), str):
        return o
    return None


def parse_json_output(text: str) -> dict:
    """Returns strict_valid, lenient_valid, obj (the parsed dict or None), lenient_step."""
    strict = _as_obj(text.strip())
    if strict is not None:
        return {"strict_valid": True, "lenient_valid": True, "obj": strict, "lenient_step": "strict"}
    s = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text.strip(), count=1), count=1)
    o = _as_obj(s.strip())
    if o is not None:
        return {"strict_valid": False, "lenient_valid": True, "obj": o, "lenient_step": "fence"}
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        o = _as_obj(s[i:j + 1])
        if o is not None:
            return {"strict_valid": False, "lenient_valid": True, "obj": o, "lenient_step": "braces"}
    return {"strict_valid": False, "lenient_valid": False, "obj": None, "lenient_step": None}


# ----------------------------------------------------------------------------- LaTeX-escape repair
# A model that writes `\frac` inside a JSON string produces the valid JSON escape `\f` (form feed)
# + "rac". Decoding then corrupts the LaTeX. We map such control characters back when they are
# immediately followed by an ASCII letter (i.e. they were a LaTeX command). Logged per item as
# `latex_escape_repaired`; this is a documented deviation (see reports/SUMMARY.md).
_CTRL = {"\f": "\\f", "\b": "\\b", "\t": "\\t", "\r": "\\r", "\n": "\\n", "\v": "\\v", "\a": "\\a"}
_CTRL_RE = re.compile("([\f\b\t\r\n\v\a])(?=[A-Za-z])")


def repair_latex_escapes(s: str) -> tuple[str, bool]:
    if s is None:
        return s, False
    out = _CTRL_RE.sub(lambda m: _CTRL[m.group(1)], s)
    return out, out != s


_ESC = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def json_unescape(s: str) -> str:
    """JSON-unescape a raw string body; tolerant of invalid escapes (kept literally)."""
    try:
        return json.loads('"' + s + '"')
    except (json.JSONDecodeError, ValueError):
        pass

    def rep(m):
        c = m.group(1)
        if c == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", m.group(2) or ""):
            return chr(int(m.group(2), 16))
        if c in _ESC:
            return _ESC[c] + (m.group(2) or "")
        return "\\" + c + (m.group(2) or "")

    return re.sub(r"\\(.)([0-9a-fA-F]{4})?", rep, s, flags=re.DOTALL)


# ----------------------------------------------------------------------------- extraction

def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} with balanced braces (not a regex)."""
    if text is None:
        return None
    pos = len(text)
    while True:
        i = text.rfind("\\boxed", 0, pos)
        if i < 0:
            return None
        j = i + len("\\boxed")
        while j < len(text) and text[j] == " ":
            j += 1
        if j < len(text) and text[j] == "{":
            depth, k = 0, j
            while k < len(text):
                ch = text[k]
                if ch == "\\" and k + 1 < len(text):  # skip escaped char such as \{ or \}
                    k += 2
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[j + 1:k].strip()
                k += 1
        pos = i  # unbalanced/garbled: try an earlier \boxed


_ANS_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_FINAL_RE = re.compile(r"(?i)final answer\s*(?:is|:)?\s*(.+)")


def extract_candidate(cond: str, text: str, parsed: dict | None) -> dict:
    cand, method, repaired = None, "none", False
    if cond in JSON_CONDS:
        if parsed and parsed["lenient_valid"]:
            cand, method = parsed["obj"]["answer"], "json_field"
            cand, repaired = repair_latex_escapes(cand)
        else:
            ms = _ANS_RE.findall(text or "")
            if ms:
                cand, method = json_unescape(ms[-1]), "regex_field"
                cand, repaired = repair_latex_escapes(cand)
        if cand is not None and "\\boxed" in cand:
            b = last_boxed(cand)
            if b is not None:
                cand, method = b, method + "_boxed"
    else:
        b = last_boxed(text or "")
        if b is not None:
            cand, method = b, "ff_boxed"
        else:
            lines = [ln for ln in (text or "").splitlines() if _FINAL_RE.search(ln)]
            if lines:
                cand, method = _FINAL_RE.search(lines[-1]).group(1).strip(), "ff_final_line"
    if cand is not None and cand.strip() == "":
        cand = cand.strip()
    return {"candidate": cand, "extract_method": method, "latex_escape_repaired": repaired}


# ----------------------------------------------------------------------------- math-verify

def wrap(s: str) -> str:
    return s if ("$" in s or "\\boxed" in s) else f"${s}$"


def mv_score(gold: str, cand: str | None) -> dict:
    from math_verify import parse, verify
    out = {"mv_parse_ok_gold": False, "mv_parse_ok_cand": False, "mv_correct": False, "mv_error": None}
    try:
        gold_p = parse(wrap(gold))
        out["mv_parse_ok_gold"] = bool(gold_p)
        if cand is None:
            return out
        cand_p = parse(wrap(cand))
        out["mv_parse_ok_cand"] = bool(cand_p)
        # gold FIRST: verify is not symmetric
        out["mv_correct"] = bool(gold_p) and bool(cand_p) and bool(verify(gold_p, cand_p))
    except Exception as e:  # math-verify can raise on pathological input; counted, never dropped
        out["mv_error"] = repr(e)[:300]
    return out


# ----------------------------------------------------------------------------- token counts

def reasoning_tokens(tok, parsed: dict | None) -> int | None:
    if parsed and parsed["lenient_valid"]:
        return len(tok(parsed["obj"]["reasoning"], add_special_tokens=False)["input_ids"])
    return None


def score_record(r: dict, tok) -> dict:
    cond = r["condition"]
    parsed = parse_json_output(r["output_text"]) if cond in JSON_CONDS else None
    ex = extract_candidate(cond, r["output_text"], parsed)
    mv = mv_score(r["gold"], ex["candidate"])
    out = dict(r)
    if parsed is not None:
        out.update(strict_valid=parsed["strict_valid"], lenient_valid=parsed["lenient_valid"],
                   lenient_step=parsed["lenient_step"])
    else:
        out.update(strict_valid=None, lenient_valid=None, lenient_step=None)
    out.update(ex)
    out.update(mv)
    out["reasoning_tokens"] = reasoning_tokens(tok, parsed)
    return out


def stage_mv(model: str, conds, subdir=None):
    from transformers import AutoTokenizer
    info = model_info(model)
    tok = AutoTokenizer.from_pretrained(info["hf_id"])
    d = OUTPUTS / info["tag"] / (subdir or "")
    for c in conds:
        rows = read_jsonl(d / f"e0_{c}.jsonl")
        if not rows:
            print(f"skip {c}: no outputs")
            continue
        uids = [r["uid"] for r in rows]
        assert len(uids) == len(set(uids)), f"duplicate uids in e0_{c}"
        scored = [score_record(r, tok) for r in rows]
        write_jsonl(d / f"e0_{c}_mv.jsonl", scored, meta={"stage": "mv", "condition": c, "n": len(scored)})
        acc = sum(s["mv_correct"] for s in scored) / len(scored)
        print(f"{info['tag']} {c}: n={len(scored)} mv_acc={acc:.3f}")


def final_label(mv_correct: bool, judge: dict | None) -> tuple[bool, bool]:
    """Pre-registered rule (§6.4). Returns (final_correct, judge_failed)."""
    if mv_correct:
        return True, False
    if judge is not None and not judge["judge_failed"]:
        return judge["judge_verdict"] == "EQUIVALENT", False
    return False, True


def stage_final(model: str, conds, subdir=None):
    info = model_info(model)
    d = OUTPUTS / info["tag"] / (subdir or "")
    for c in conds:
        rows = read_jsonl(d / f"e0_{c}_mv.jsonl")
        if not rows:
            print(f"skip {c}: no mv-scored outputs")
            continue
        jrows = {r["uid"]: r for r in read_jsonl(d / "judge" / f"e0_{c}_judge_seed0.jsonl")}
        out, n_missing = [], 0
        for r in rows:
            j = jrows.get(r["uid"])
            need = not r["mv_correct"]
            if need and j is None:
                n_missing += 1
            fc, jf = final_label(r["mv_correct"], j)
            out.append(dict(r, judged=j is not None, judge_role=(j or {}).get("role"),
                            judge_verdict=(j or {}).get("judge_verdict"),
                            judge_failed=bool((j or {}).get("judge_failed", False)),
                            label_from=("mv" if r["mv_correct"] else ("judge_failed" if jf else "judge")),
                            judge_raw_tail=(j or {}).get("judge_raw_tail"), final_correct=fc))
        assert n_missing == 0, f"{c}: {n_missing} mv-incorrect items lack a judge record"
        write_jsonl(d / f"e0_{c}_scored.jsonl", out, meta={"stage": "final", "condition": c, "n": len(out)})
        print(f"{info['tag']} {c}: final_acc={sum(o['final_correct'] for o in out) / len(out):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    ap.add_argument("--stage", required=True, choices=["mv", "final"])
    ap.add_argument("--conds", default="FF,JS,JC")
    ap.add_argument("--subdir", default=None)
    a = ap.parse_args()
    (stage_mv if a.stage == "mv" else stage_final)(a.model, a.conds.split(","), a.subdir)
