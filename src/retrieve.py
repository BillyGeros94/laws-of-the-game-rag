"""Retrieve the passages that answer a question.

Three modes (config.RETRIEVAL_MODE):
- dense:  embedding search only.
- hybrid: embedding search plus BM25 keyword search, merged by reciprocal rank fusion.
- grade:  hybrid, then a language model grades the merged candidates (src/grade.py).
          This is the mode the app uses; dense and hybrid are kept for comparison
          (python src/eval.py --retrieval-only --mode dense).

The index holds about 300 chunks, so every search scores every chunk; no approximate
search is needed. Chroma is only the store the chunks and their vectors are read from.
"""
import re
import sys
from functools import lru_cache

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (CANDIDATES, COLLECTION, GRADE_MIN, GRADE_POOL, INDEX_DIR, MAX_SOURCES,
                    MODEL_NAME, QUERY_PREFIX, RETRIEVAL_MODE, TOP_K)
from grade import grade
from lexical import BM25, correct_spelling, expand, rrf, terms, words

MODES = ("dense", "hybrid", "grade")


def breadcrumb(meta):
    """Location line, e.g. 'Laws of the Game 2026/27 > Law 12 - Fouls and Misconduct >
    4. Disciplinary action > Cautions for unsporting behaviour'. Placed in front of a chunk
    for embedding, keyword search and grading."""
    section = re.sub(r"\s*\|\s*", " - ", meta["section"])
    parts = (meta["title"], section, meta["heading"], meta.get("subheading", ""))
    return " > ".join(part for part in parts if part)


def distinct(order, chunks, k):
    """The first k indices whose context the model has not already been given: two small
    chunks from the same sub-section would hand it the same text twice."""
    picked, seen = [], set()
    for i in order:
        context = chunks[i].get("context") or chunks[i]["text"]
        if context not in seen:
            seen.add(context)
            picked.append(i)
            if len(picked) == k:
                break
    return picked


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME)


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
    return chunks, vectors, BM25(full_texts), vocabulary


def retrieve(question, k=TOP_K, mode=RETRIEVAL_MODE):
    """Return the best passages for the question, best first.

    Dense and hybrid mode return k. Grade mode returns every passage graded GRADE_MIN or more,
    up to MAX_SOURCES; when none is, or grading fails, it returns the top k of the hybrid search.

    Each result is a dict: chunk_id, text (the small chunk that was matched), context (the
    surrounding text the model answers from), distance (cosine, smaller is closer), in grade
    mode grade (0-3, from the language model), plus the stored metadata (title, doc_type,
    status, section, heading, subheading, page_start, page_end, filename, date). No two
    results share a context."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    chunks, vectors, bm25, vocabulary = get_corpus()

    query = question if mode == "dense" else correct_spelling(question, vocabulary)[0]
    vector = get_model().encode(QUERY_PREFIX + query, normalize_embeddings=True)
    distances = 1 - vectors @ vector
    dense_order = list(np.argsort(distances))

    grades = {}
    if mode == "dense":
        order = distinct(dense_order, chunks, k)
    else:
        keyword_scores = bm25.scores(terms(expand(query)))
        keyword_order = [i for i in np.argsort(-keyword_scores) if keyword_scores[i] > 0]
        candidates = rrf(dense_order[:CANDIDATES], keyword_order[:CANDIDATES])
        order = distinct(candidates, chunks, k)          # hybrid, and grade mode's fallback

        if mode == "grade":
            pool = candidates[:GRADE_POOL]
            graded = grade(question, [(breadcrumb(chunks[i]), chunks[i]["text"]) for i in pool])
            if graded is not None:
                grades = {pool[n]: score for n, score in graded.items()}
                # Best grade first; equal grades keep the search order (sorted() is stable).
                kept = [i for i in sorted(pool, key=lambda i: -grades[i]) if grades[i] >= GRADE_MIN]
                # Every passage good enough, up to MAX_SOURCES. With none, the grader is not
                # trusted to refuse: it has graded a passage that answers the question 0, and
                # it judges false premises ("six seconds") rather than topics. The hybrid
                # order stands, and the answer model refuses if those passages do not answer.
                if kept:
                    order = distinct(kept, chunks, MAX_SOURCES)

    hits = []
    for i in order:
        hit = {**chunks[i], "distance": float(distances[i])}
        if i in grades:
            hit["grade"] = grades[i]
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
    _, fixes = correct_spelling(question, get_corpus()[3])
    if fixes and RETRIEVAL_MODE != "dense":
        print(f"Spelling corrected: {fixes}")
    print()
    for rank, hit in enumerate(retrieve(question), start=1):
        preview = " ".join(hit["text"].split())[:200]
        score = f" | grade {hit['grade']}/3" if "grade" in hit else ""
        print(f"{rank}. distance {hit['distance']:.4f}{score} | {hit['filename']} | {pages(hit)}")
        print(f"   section: {hit['section'] or '-'} | heading: "
              f"{hit.get('subheading') or hit['heading'] or '-'}")
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()
