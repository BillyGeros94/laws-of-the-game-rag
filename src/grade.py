"""Grade the search results with a language model: which passages actually answer the question?

The searches find candidates by wording; a small reranker then failed on everyday words
("dives" against a passage that says "simulation") and on questions that need two rules.
A language model reads the question and each candidate and scores it:

    3  states the rule the question asks about (or one of the rules it needs)
    2  states part of it, or something needed to apply it
    1  same topic, but does not help answer
    0  unrelated

Retrieval keeps the passages scored GRADE_MIN or more (at most MAX_SOURCES), and refuses when
there are none. The model only grades: answers are still written from the passage text.
"""
import json
import re
import sys

from openai import BadRequestError

from config import GRADE_MODEL, GRADE_PASSAGE_CHARS

GRADE_PROMPT = """You grade passages from the IFAB Laws of the Game and related IFAB documents for a question.

Score every passage:
3 = it states the rule the question asks about, or one of the rules the answer needs
2 = it states part of that rule, or something needed to apply it
1 = same topic, but it does not help answer the question
0 = unrelated

- People ask in everyday words; the Laws use their own terms (a dive is simulation, being
  booked is a caution, being sent off is a sending-off, the ref or a linesman is a match
  official). Grade by meaning, not by shared words.
- A question may need two rules (for example what happens, and how play restarts): give 3 to
  a passage for each.
- Grade only what each passage says. Do not use your own knowledge of the Laws to fill gaps.
- If the question is not about the rules of football, every passage is 0.

Reply with JSON only, a score for every passage number: {"scores": {"1": 0, "2": 3, ...}}"""


def format_passages(passages):
    """Numbered passages, each with its location line and (shortened) text."""
    blocks = []
    for number, (location, text) in enumerate(passages, start=1):
        if len(text) > GRADE_PASSAGE_CHARS:
            text = text[:GRADE_PASSAGE_CHARS].rsplit(" ", 1)[0] + " ..."
        blocks.append(f"[{number}] {location}\n{text}")
    return "\n\n".join(blocks)


def parse_scores(content, count):
    """{position: score} for positions 0..count-1, or None if the reply is unusable.

    Scores outside 0-3 and unknown numbers are ignored; passages it did not score count as 0.
    A reply that scores fewer than half of the passages is treated as a failure."""
    match = re.search(r"\{.*\}", content or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0)).get("scores")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(raw, dict):
        return None
    scores = {}
    for key, value in raw.items():
        try:
            position, score = int(key) - 1, int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= position < count and 0 <= score <= 3:
            scores[position] = score
    if len(scores) < count / 2:
        return None
    return {position: scores.get(position, 0) for position in range(count)}


def grade(question, passages):
    """Scores for passages, a list of (location line, text). Returns {position: score}, or
    None when the call fails or the reply is unusable; the caller then falls back."""
    from generate import get_client   # here, not at the top: generate imports retrieve
    request = dict(
        model=GRADE_MODEL,
        temperature=0,
        max_completion_tokens=2000,   # gpt-oss counts its reasoning here
        extra_body={"reasoning_effort": "low"},
        messages=[{"role": "system", "content": GRADE_PROMPT},
                  {"role": "user", "content": f"Question: {question}\n\nPassages:\n\n"
                                              f"{format_passages(passages)}"}],
    )
    for json_mode in (True, False):   # the provider rejects replies failing its JSON check
        try:
            extra = {"response_format": {"type": "json_object"}} if json_mode else {}
            response = get_client().chat.completions.create(**request, **extra)
        except BadRequestError:
            continue
        except Exception as error:   # rate limit, network, missing key
            print(f"grading skipped: {type(error).__name__}", file=sys.stderr)
            return None
        scores = parse_scores(response.choices[0].message.content, len(passages))
        if scores is not None:
            return scores
    print("grading skipped: the model's reply was unusable", file=sys.stderr)
    return None
