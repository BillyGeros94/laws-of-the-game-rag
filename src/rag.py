"""Question in, cited answer out: retrieve, then generate."""
import sys

from openai import RateLimitError

from config import MAX_TOP1_DISTANCE, REFUSAL, TOP_K
from generate import generate
from retrieve import pages, retrieve


def answer(question, k=TOP_K):
    """Answer a question. Returns a dict: question, answer, refused, hits, finish_reason, token counts."""
    hits = retrieve(question, k)
    result = {"question": question, "answer": "", "refused": False, "hits": hits,
              "finish_reason": None, "prompt_tokens": 0, "completion_tokens": 0}

    # If even the best chunk is far from the question, nothing relevant exists: refuse
    # without spending an LLM call.
    if not hits or (MAX_TOP1_DISTANCE is not None and hits[0]["distance"] > MAX_TOP1_DISTANCE):
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
        print(f"(Refused without calling the model: the best distance "
              f"{result['hits'][0]['distance']:.4f} is above {MAX_TOP1_DISTANCE})\n")
    print(f"Answer: {result['answer'] or '(empty answer)'}\n")
    if result["finish_reason"] == "length":
        print("Warning: the answer hit the token limit and may be cut off.\n")

    print("Sources:")
    for number, hit in enumerate(result["hits"], start=1):
        where = hit["heading"] or hit["section"] or "-"
        print(f"[{number}] {hit['filename']} | {pages(hit)} | {where} | distance {hit['distance']:.4f}")
    if not result["refused"]:
        print(f"\nTokens: prompt {result['prompt_tokens']}, answer {result['completion_tokens']}")


if __name__ == "__main__":
    main()
