"""Question in, cited answer out: retrieve, then generate."""
import sys

from openai import RateLimitError

from config import MAX_TOP1_DISTANCE, MIN_RERANK_SCORE, REFUSAL, RETRIEVAL_MODE, TOP_K
from generate import generate
from retrieve import pages, retrieve


def gate_refuses(hits):
    """True when nothing retrieved is relevant enough to be worth answering from.

    With reranker scores, the best score must reach MIN_RERANK_SCORE. Without them, the
    closest chunk must be within MAX_TOP1_DISTANCE (the closest of the hits, since in hybrid
    mode the first hit is not necessarily the closest)."""
    if not hits:
        return True
    if "rerank_score" in hits[0]:
        return MIN_RERANK_SCORE is not None and hits[0]["rerank_score"] < MIN_RERANK_SCORE
    return MAX_TOP1_DISTANCE is not None and min(h["distance"] for h in hits) > MAX_TOP1_DISTANCE


def answer(question, k=TOP_K, mode=RETRIEVAL_MODE):
    """Answer a question. Returns a dict: question, answer, refused, hits, finish_reason, token counts."""
    hits = retrieve(question, k, mode)
    result = {"question": question, "answer": "", "refused": False, "hits": hits,
              "finish_reason": None, "prompt_tokens": 0, "completion_tokens": 0}

    # If even the best chunk is far from the question, nothing relevant exists: refuse
    # without spending an LLM call.
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
        top = result["hits"][0] if result["hits"] else None
        if top and "rerank_score" in top:
            why = f"the best reranker score {top['rerank_score']:.2f} is below {MIN_RERANK_SCORE}"
        elif top:
            why = f"the best distance {min(h['distance'] for h in result['hits']):.4f} is above {MAX_TOP1_DISTANCE}"
        else:
            why = "nothing was retrieved"
        print(f"(Refused without calling the model: {why})\n")
    print(f"Answer: {result['answer'] or '(empty answer)'}\n")
    if result["finish_reason"] == "length":
        print("Warning: the answer hit the token limit and may be cut off.\n")

    print("Sources:")
    for number, hit in enumerate(result["hits"], start=1):
        where = hit["heading"] or hit["section"] or "-"
        score = f" | rerank {hit['rerank_score']:.2f}" if "rerank_score" in hit else ""
        print(f"[{number}] {hit['filename']} | {pages(hit)} | {where} | distance {hit['distance']:.4f}{score}")
    if not result["refused"]:
        print(f"\nTokens: prompt {result['prompt_tokens']}, answer {result['completion_tokens']}")


if __name__ == "__main__":
    main()
