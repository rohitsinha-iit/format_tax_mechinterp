# prompts/templates.py
import json

TASK = (
    "Solve the following math problem. Reason step by step, and state the final "
    "answer clearly at the end.\n\n"
    "Problem:\n{problem}"
)

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string",
                      "description": "Step-by-step reasoning to solve the problem"},
        "answer": {"type": "string",
                   "description": "Final answer to the problem"},
    },
    "required": ["reasoning", "answer"],
}
SCHEMA_STR = json.dumps(SCHEMA, indent=2)  # still used for JC constrained decoding (SCHEMA), not in prompts

# Revised by the user on 2026-10-06 (v2): the v1 block embedded SCHEMA_STR in a ```json fence and
# Qwen3-4B echoed the schema itself in 9/10 smoke outputs. See reports/DEVIATIONS.md (D5).
FORMAT_SPEC = (
    "Format your entire response as a single JSON object with exactly two keys, "
    "in this order:\n"
    "- \"reasoning\": a string containing your step-by-step reasoning\n"
    "- \"answer\": a string containing only your final answer\n\n"
    "Your output must have this shape:\n"
    "{\"reasoning\": \"<your step-by-step reasoning>\", \"answer\": \"<your final answer>\"}\n\n"
    "Output only the JSON object, with no text before or after it."
)
FORMAT_BLOCK = FORMAT_SPEC

REFORMAT_BLOCK = (
    "Rewrite your previous response so that it follows the format below. "
    "Do not change the final answer.\n\n" + FORMAT_SPEC
)

# Neutral control: appended text that is neither a format nor a length instruction.
# Must be length-matched to FORMAT_BLOCK (see 4.2).
NEUTRAL_BLOCK = (
    "Background note: This problem is taken from a large collection of exercises "
    "that has been used in mathematics courses and competitions for many years. "
    "The collection covers several areas, including algebra, geometry, number "
    "theory, counting and probability, intermediate algebra, and precalculus. "
    "Exercises in the collection range in difficulty from introductory questions "
    "to challenging competition problems. Each exercise was written and reviewed "
    "by experienced teachers before being added to the collection."
)
NEUTRAL_PAD = [  # use only to length-match, in this order
    "The problems are presented without any hints.",
    "Some of the problems have appeared in more than one edition of the collection.",
]
NEUTRAL_TRIM_ORDER = [  # remove from the end of NEUTRAL_BLOCK, sentence by sentence, if too long
]

# Brevity contrast pairs (k = 1..3). CONCISE_k and DETAILED_k are paired by index.
CONCISE = [
    "Be concise. Keep your response as short as possible.",
    "Answer briefly, using as few words as you can.",
    "Give a short, compact solution without extra explanation.",
]
DETAILED = [
    "Be thorough. Explain every step in full detail.",
    "Answer at length, writing out all of your work explicitly.",
    "Give a long, comprehensive solution with complete explanations.",
]

def task(p):          return TASK.format(problem=p)   # braces in p are safe (argument, not template)
def user_ff(p):       return task(p)
def user_js(p):       return task(p) + "\n\n" + FORMAT_BLOCK
def user_nc(p):       return task(p) + "\n\n" + NEUTRAL_BLOCK
def user_concise(p,k):  return task(p) + "\n\n" + CONCISE[k]
def user_detailed(p,k): return task(p) + "\n\n" + DETAILED[k]
