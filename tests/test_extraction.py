"""Unit tests for §6.1–6.3 (validity, extraction, math-verify wiring)."""
from src.score import (extract_candidate, json_unescape, last_boxed, mv_score, parse_json_output,
                       repair_latex_escapes)


def test_last_boxed_balanced():
    assert last_boxed(r"a \boxed{1} b \boxed{\frac{1}{2}} c") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{\{1,2\}}") == r"\{1,2\}"
    assert last_boxed(r"\boxed{x} then \boxed{unbalanced") == "x"
    assert last_boxed("no box") is None


def test_strict_and_lenient():
    ok = '{"reasoning": "r", "answer": "5"}'
    assert parse_json_output(ok)["strict_valid"]
    fenced = "```json\n" + ok + "\n```"
    p = parse_json_output(fenced)
    assert not p["strict_valid"] and p["lenient_valid"] and p["lenient_step"] == "fence"
    p = parse_json_output("Here you go: " + ok + " thanks")
    assert p["lenient_valid"] and p["lenient_step"] == "braces"
    assert not parse_json_output('{"reasoning": 3, "answer": "5"}')["lenient_valid"]
    assert not parse_json_output('{"reasoning": "a \\( x \\)", "answer": "5"}')["lenient_valid"]


def test_json_candidate_paths():
    ok = '{"reasoning": "r", "answer": "\\\\boxed{7}"}'
    p = parse_json_output(ok)
    ex = extract_candidate("JS", ok, p)
    assert ex == {"candidate": "7", "extract_method": "json_field_boxed", "latex_escape_repaired": False}
    bad = '{"reasoning": "invalid \\( escape", "answer": "\\frac{1}{2}"}'
    p = parse_json_output(bad)
    assert not p["lenient_valid"]
    ex = extract_candidate("JS", bad, p)
    assert ex["extract_method"] == "regex_field" and ex["candidate"] == r"\frac{1}{2}"
    assert extract_candidate("JC", "garbage", parse_json_output("garbage"))["candidate"] is None


def test_latex_escape_repair():
    raw = '{"reasoning": "r", "answer": "\\frac{3}{4}"}'   # JSON escape \f -> form feed
    p = parse_json_output(raw)
    assert p["strict_valid"] and p["obj"]["answer"].startswith("\f")
    ex = extract_candidate("JS", raw, p)
    assert ex["candidate"] == r"\frac{3}{4}" and ex["latex_escape_repaired"]
    assert repair_latex_escapes("a\nb")[0] == "a\\nb"  # \n followed by a letter -> LaTeX
    assert repair_latex_escapes("a\n 1")[0] == "a\n 1"
    assert json_unescape(r"\frac{1}{2} \( x \)") == "\frac{1}{2} \\( x \\)"


def test_ff_extraction():
    assert extract_candidate("FF", r"so \boxed{12}.", None)["candidate"] == "12"
    ex = extract_candidate("FF", "work\nThe final answer is 3/4\n", None)
    assert ex == {"candidate": "3/4", "extract_method": "ff_final_line", "latex_escape_repaired": False}
    assert extract_candidate("FF", "nothing here", None)["extract_method"] == "none"


def test_math_verify():
    assert mv_score(r"\frac{1}{2}", "0.5")["mv_correct"]
    assert mv_score(r"\frac{1}{2}", r"\boxed{\frac12}")["mv_correct"]
    assert not mv_score("3", "4")["mv_correct"]
    r = mv_score("3", None)
    assert r["mv_correct"] is False and r["mv_parse_ok_gold"]
