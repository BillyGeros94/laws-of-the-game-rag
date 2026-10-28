"""Retrieve the chunks closest to a question."""
import sys
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION, INDEX_DIR, MODEL_NAME, QUERY_PREFIX, TOP_K


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def get_collection():
    """Open the Chroma index once and reuse it."""
    return chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)


def retrieve(question, k=TOP_K):
    """Return the k chunks closest to the question, best first.

    Each result is a dict: chunk_id, text, distance (cosine, smaller is closer),
    plus the stored metadata (title, doc_type, status, section, heading,
    page_start, page_end, filename, date).
    """
    vector = get_model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
    result = get_collection().query(
        query_embeddings=[vector.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for chunk_id, text, metadata, distance in zip(result["ids"][0], result["documents"][0],
                                                  result["metadatas"][0], result["distances"][0]):
        hits.append({"chunk_id": chunk_id, "text": text, "distance": distance, **metadata})
    return hits


def pages(hit):
    """Format a hit's page range, e.g. 'p.131-132' or 'p.131'."""
    if hit["page_start"] == hit["page_end"]:
        return f"p.{hit['page_start']}"
    return f"p.{hit['page_start']}-{hit['page_end']}"


def main():
    """Command line: python src/retrieve.py "your question" """
    sys.stdout.reconfigure(encoding="utf-8")   # the Laws text contains curly quotes and bullets
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python src/retrieve.py "your question"')
        sys.exit(1)

    print(f"Question: {question}\n")
    for rank, hit in enumerate(retrieve(question), start=1):
        preview = " ".join(hit["text"].split())[:200]
        print(f"{rank}. distance {hit['distance']:.4f} | {hit['filename']} | {pages(hit)}")
        print(f"   section: {hit['section'] or '-'} | heading: {hit['heading'] or '-'}")
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()
