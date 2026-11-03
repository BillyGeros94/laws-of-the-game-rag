"""Question in, cited answer out: retrieve, then generate."""
import sys

from openai import RateLimitError

from config import GRADE_MIN, MAX_TOP1_DISTANCE, REFUSAL, RETRIEVAL_MODE, TOP_K
from generate import generate
from retrieve import pages, retrieve


def gate_refuses(hits):
    """True when nothing retrieved is close enough to the question to be worth answering from.

    A passage graded GRADE_MIN or more always passes: the grader found something relevant.
    Otherwise the grader is not trusted to refuse, and the closest passage must be within
    MAX_TOP1_DISTANCE (the closest of the hits, since after fusion the first hit is not
    necessarily the closest)."""
    if not hits:
        return True
    if hits[0].get("grade", -1) >= GRADE_MIN:
        return False
    return MAX_TOP1_DISTANCE is not None and min(h["distance"] for h in hits) > MAX_TOP1_DISTANCE


def answer(question, k=TOP_K, mode=RETRIEVAL_MODE):
    """Answer a question. Returns a dict: question, answer, refused, hits, finish_reason,
    token counts."""
    hits = retrieve(question, k, mode)
    result = {"question": question, "answer": "", "refused": False, "hits": hits,
              "finish_reason": None, "prompt_tokens": 0, "completion_tokens": 0}

    # Nothing retrieved is anywhere near the question: refuse without spending a model call.
    if gate_refuses(hits):
        result.update(refused=True, answer=REFUSAL)
        return result

    result.update(generate(question, hits))
    return result


def main():
    """Command line: python src/rag.py "your question" """
    sys.stdout.reconfigure(encoding="utf-8")   # the Laws text contains curly quotes and bullets
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python src/rag.py "your question"')
        sys.exit(1)

    try:
        result = answer(question)
    except RateLimitError:
        print("Rate limit reached. A per-minute limit clears within a minute; "
              "a daily limit clears the next day. Try again later.")
        sys.exit(1)

    print(f"Question: {question}\n")
    if result["refused"]:
        closest = min((h["distance"] for h in result["hits"]), default=None)
        why = ("nothing was retrieved" if closest is None else
               f"no passage was graded {GRADE_MIN} or more and the closest distance "
               f"{closest:.4f} is above {MAX_TOP1_DISTANCE}")
        print(f"(Refused without calling the model: {why})\n")
    print(f"Answer: {result['answer'] or '(empty answer)'}\n")
    if result["finish_reason"] == "length":
        print("Warning: the answer hit the token limit and may be cut off.\n")

    print("Sources:")
    for number, hit in enumerate(result["hits"], start=1):
        where = hit.get("subheading") or hit["heading"] or hit["section"] or "-"
        score = f" | grade {hit['grade']}/3" if "grade" in hit else ""
        print(f"[{number}] {hit['filename']} | {pages(hit)} | {where} | "
              f"distance {hit['distance']:.4f}{score}")
    if not result["refused"]:
        print(f"\nTokens: prompt {result['prompt_tokens']}, answer {result['completion_tokens']}")


if __name__ == "__main__":
    main()
