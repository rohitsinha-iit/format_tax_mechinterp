"""§4.2 template assertions. Run: pytest -s tests/test_templates.py
Side effect: freezes prompts/neutral_final.json and writes reports/template_checks.txt."""
import json

import pytest
from transformers import AutoTokenizer

from prompts import templates as T
from src.common import REPORTS, model_info, now
from src import prompting as P

MODELS = ["qwen3-4b", "qwen3-8b"]
REPORT = REPORTS / "template_checks.txt"
PROBLEMS = [
    "What is $1+1$?",
    "Find $x$ if $\\frac{x}{2} = \\{3\\}$ and {a} {problem} }}{{ hold.",
    "Let $f(x)=x^2$.\nCompute $f(f(2))$.\n\nGive the answer in \\boxed{}.",
]


def _report(line):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


@pytest.fixture(scope="module", params=MODELS)
def tok(request):
    t = AutoTokenizer.from_pretrained(model_info(request.param)["hf_id"])
    t._ftmi_key = request.param
    return t


def _problems():
    probs = list(PROBLEMS)
    try:
        from src.data import load_math500, load_extract_set
        probs += [r["problem"] for r in load_math500()[:20]]
        probs += [r["problem"] for r in load_extract_set()[:20]]
    except Exception as e:  # data may not be built yet; synthetic problems still exercise the code
        print("data not available yet:", e)
    return probs


def test_neutral_length_match(tok):
    res = P.resolve_neutral_block(tok)
    _report(f"[{now()}] {tok._ftmi_key}: FORMAT_BLOCK={res['format_block_tokens']} tok, "
            f"NEUTRAL_BLOCK original={res['neutral_original_tokens']} tok, final={res['neutral_block_tokens']} tok, "
            f"ratio={res['ratio']:.3f}, actions={res['actions']}")
    assert res["within_tol"], res
    if P.NEUTRAL_FINAL_PATH.exists():
        prev = json.loads(P.NEUTRAL_FINAL_PATH.read_text(encoding="utf-8"))
        assert prev["neutral_block"] == res["neutral_block"], "neutral block differs across tokenizers"
    else:
        P.NEUTRAL_FINAL_PATH.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    P.neutral_block.cache_clear()


def test_ff_js_differ_only_by_format_block(tok):
    for p in _problems():
        ff = P.render(tok, P.messages("FF", p))
        js = P.render(tok, P.messages("JS", p))
        i = ff.rfind("<|im_end|>")
        assert js == ff[:i] + "\n\n" + T.FORMAT_BLOCK + ff[i:]
        assert T.task(p).endswith(p)


def test_post_instruction_span_identical(tok):
    ref = None
    for p in _problems():
        for c in P.ACT_CONDS:
            s = P.render(tok, P.messages(c, p))
            _, span = P.post_instruction_span(tok, s)
            if ref is None:
                ref = span
                _report(f"{tok._ftmi_key}: post-instruction span ({len(span)} tokens) = {span} = "
                        f"{tok.convert_ids_to_tokens(span)} = {tok.decode(span)!r}")
            assert span == ref, (c, span, ref)
    assert tok.decode(ref) == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_prompt_ends_with_span_and_last_token(tok):
    p = PROBLEMS[0]
    s = P.render(tok, P.messages("FF", p))
    ids = P.encode(tok, s)
    start, span = P.post_instruction_span(tok, s)
    assert ids[start:] == span
    assert ids == tok(s)["input_ids"], "tokenizer adds special tokens unexpectedly"


def test_two_turn_contains_ff_output_verbatim(tok):
    ff_out = "We compute $1+1=2$.\n\nThe final answer is $\\boxed{2}$."
    s = P.render(tok, P.messages("2T", PROBLEMS[0], ff_out))
    expected_asst = "<|im_start|>assistant\n" + ff_out + "<|im_end|>\n"
    assert expected_asst in s, s
    assert s.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert T.REFORMAT_BLOCK in s


def test_judge_thinking_prompt_has_no_empty_think(tok):
    s = P.render(tok, [{"role": "user", "content": "hi"}], enable_thinking=True)
    assert s.endswith("<|im_start|>assistant\n"), repr(s[-60:])
