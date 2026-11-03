"""Retrieve the chunks closest to a question.

Three modes (config.RETRIEVAL_MODE):
- dense:  embedding search only.
- hybrid: embedding search plus BM25 keyword search, merged by reciprocal rank fusion.
- rerank: hybrid, then a cross-encoder rescores the merged candidates.

The index holds about 225 chunks, so every search scores every chunk; no approximate
search is needed. Chroma is only the store the chunks and their vectors are read from.
"""
import re
import sys
from functools import lru_cache

import chromadb
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (CANDIDATES, COLLECTION, INDEX_DIR, MODEL_NAME, QUERY_PREFIX, RERANK_MODEL,
                    RETRIEVAL_MODE, TOP_K)
from lexical import BM25, correct_spelling, expand, rrf, terms, words

MODES = ("dense", "hybrid", "rerank")


def breadcrumb(meta):
    """Location line, e.g. 'Laws of the Game 2026/27 > Law 12 - Fouls and Misconduct >
    4. Disciplinary action'. Placed in front of a chunk for embedding, BM25 and reranking."""
    section = re.sub(r"\s*\|\s*", " - ", meta["section"])
    return " > ".join(part for part in (meta["title"], section, meta["heading"]) if part)


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder once and reuse it."""
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    """Open the Chroma index once and reuse it."""
    return chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)


@lru_cache(maxsize=1)
def get_corpus():
    """Every chunk with its vector, the BM25 index and the corpus vocabulary, loaded once."""
    stored = get_collection().get(include=["documents", "metadatas", "embeddings"])
    chunks = [{"chunk_id": i, "text": t, **m}
              for i, t, m in zip(stored["ids"], stored["documents"], stored["metadatas"])]
    full_texts = [f"{breadcrumb(c)}\n{c['text']}" for c in chunks]
    vectors = np.asarray(stored["embeddings"], dtype=np.float32)
    vocabulary = {w for text in full_texts for w in words(text)}
    return chunks, full_texts, vectors, BM25(full_texts), vocabulary


def retrieve(question, k=TOP_K, mode=RETRIEVAL_MODE):
    """Return the k best chunks for the question, best first.

    Each result is a dict: chunk_id, text, distance (cosine, smaller is closer, always from
    the embedding search), rerank_score (rerank mode only; higher is better), plus the
    stored metadata (title, doc_type, status, section, heading, page_start, page_end,
    filename, date)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    chunks, full_texts, vectors, bm25, vocabulary = get_corpus()

    query = question
    if mode != "dense":
        query, _ = correct_spelling(question, vocabulary)

    vector = get_model().encode(QUERY_PREFIX + query, normalize_embeddings=True)
    distances = 1 - vectors @ vector
    dense_order = list(np.argsort(distances))

    if mode == "dense":
        order = dense_order[:k]
        scores = {}
    else:
        keyword_scores = bm25.scores(terms(expand(query)))
        keyword_order = [i for i in np.argsort(-keyword_scores) if keyword_scores[i] > 0]
        candidates = rrf(dense_order[:CANDIDATES], keyword_order[:CANDIDATES])
        if mode == "hybrid":
            order = candidates[:k]
            scores = {}
        else:
            pool = candidates[:CANDIDATES]
            rerank_scores = get_reranker().predict([(query, full_texts[i]) for i in pool])
            ranked = sorted(zip(pool, rerank_scores), key=lambda pair: -pair[1])
            order = [i for i, _ in ranked[:k]]
            scores = dict(ranked)

    hits = []
    for i in order:
        hit = {**chunks[i], "distance": float(distances[i])}
        if i in scores:
            hit["rerank_score"] = float(scores[i])
        hits.append(hit)
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

    print(f"Question: {question} (mode: {RETRIEVAL_MODE})")
    _, fixes = correct_spelling(question, get_corpus()[4])
    if fixes and RETRIEVAL_MODE != "dense":
        print(f"Spelling corrected: {fixes}")
    print()
    for rank, hit in enumerate(retrieve(question), start=1):
        preview = " ".join(hit["text"].split())[:200]
        score = f" | rerank {hit['rerank_score']:.3f}" if "rerank_score" in hit else ""
        print(f"{rank}. distance {hit['distance']:.4f}{score} | {hit['filename']} | {pages(hit)}")
        print(f"   section: {hit['section'] or '-'} | heading: {hit['heading'] or '-'}")
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()
