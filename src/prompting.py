"""Condition -> chat messages -> rendered prompt (§4). Templates come verbatim from prompts/templates.py."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from prompts import templates as T
from src.common import ROOT, sha256

NEUTRAL_FINAL_PATH = ROOT / "prompts" / "neutral_final.json"
GEN_CONDS = ("FF", "JS", "JC", "2T")
ACT_CONDS = ("FF", "JS", "NC", "C1", "C2", "C3", "D1", "D2", "D3")


def n_tokens(tok, s: str) -> int:
    return len(tok(s, add_special_tokens=False)["input_ids"])


def _sentences(s: str) -> list[str]:
    return re.split(r"(?<=\.)\s+", s.strip())


def resolve_neutral_block(tok, tol: float = 0.15) -> dict:
    """§4.2(3): length-match NEUTRAL_BLOCK to FORMAT_BLOCK within ±tol (in Qwen3 tokens)."""
    target = n_tokens(tok, T.FORMAT_BLOCK)
    lo, hi = (1 - tol) * target, (1 + tol) * target
    text, actions = T.NEUTRAL_BLOCK, []
    pads = list(T.NEUTRAL_PAD)
    while n_tokens(tok, text) < lo and pads:
        p = pads.pop(0)
        text = text + " " + p
        actions.append(f"pad:{p}")
    sents = _sentences(text)
    while n_tokens(tok, " ".join(sents)) > hi and len(sents) > 1:
        actions.append(f"trim:{sents.pop()}")
    text = " ".join(sents) if any(a.startswith("trim") for a in actions) else text
    n = n_tokens(tok, text)
    return {"format_block_tokens": target, "neutral_block_tokens": n,
            "neutral_original_tokens": n_tokens(tok, T.NEUTRAL_BLOCK),
            "ratio": n / target, "within_tol": bool(lo <= n <= hi), "actions": actions,
            "neutral_block": text}


@lru_cache(maxsize=1)
def neutral_block() -> str:
    """The frozen, length-matched neutral block (written by tests/test_templates.py)."""
    if not NEUTRAL_FINAL_PATH.exists():
        raise FileNotFoundError(f"{NEUTRAL_FINAL_PATH} missing: run pytest tests/test_templates.py first")
    d = json.loads(NEUTRAL_FINAL_PATH.read_text(encoding="utf-8"))
    assert d["within_tol"], d
    return d["neutral_block"]


def user_text(cond: str, problem: str) -> str:
    if cond == "FF":
        return T.user_ff(problem)
    if cond in ("JS", "JC"):
        return T.user_js(problem)
    if cond == "NC":
        nb = neutral_block()
        # identical to T.user_nc when no padding/trimming was needed
        return T.task(problem) + "\n\n" + nb
    if cond[0] == "C":
        return T.user_concise(problem, int(cond[1]) - 1)   # C1..C3 -> CONCISE[0..2]
    if cond[0] == "D":
        return T.user_detailed(problem, int(cond[1]) - 1)
    raise ValueError(cond)


def messages(cond: str, problem: str, ff_output: str | None = None) -> list[dict]:
    if cond == "2T":
        assert ff_output is not None
        return [{"role": "user", "content": T.user_ff(problem)},
                {"role": "assistant", "content": ff_output},
                {"role": "user", "content": T.REFORMAT_BLOCK}]
    return [{"role": "user", "content": user_text(cond, problem)}]


def render(tok, msgs, enable_thinking: bool = False) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=enable_thinking)


def encode(tok, rendered: str) -> list[int]:
    # Qwen tokenizer adds no BOS; the rendered string already contains all special tokens.
    return tok(rendered, add_special_tokens=False)["input_ids"]


def build_prompt(tok, cond: str, problem: str, ff_output: str | None = None,
                 enable_thinking: bool = False) -> dict:
    s = render(tok, messages(cond, problem, ff_output), enable_thinking)
    ids = encode(tok, s)
    return {"prompt": s, "prompt_ids": ids, "prompt_sha256": sha256(s), "prompt_n_tokens": len(ids)}


def post_instruction_span(tok, rendered: str) -> tuple[int, list[int]]:
    """Token index where the post-instruction span starts (the user turn's closing <|im_end|>)
    and the span's token ids, through the end of the generation prompt."""
    cut = rendered.rfind("<|im_end|>")
    assert cut >= 0
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    starts = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a >= cut]
    start = starts[0]
    assert enc["offset_mapping"][start][0] == cut, "span start is not a token boundary"
    return start, enc["input_ids"][start:]
