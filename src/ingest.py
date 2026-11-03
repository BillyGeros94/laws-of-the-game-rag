"""Embed the chunks and build the persistent Chroma index."""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION, INDEX_DIR, MODEL_NAME
from retrieve import breadcrumb   # the same location line the searches use

IN = Path("data/processed/chunks.jsonl")

BATCH_SIZE = 32
MIN_INDEX_CHARS = 60   # chunks shorter than this carry no content (e.g. the back cover)


def load_chunks():
    """Return the chunk records as a list of dicts."""
    with open(IN, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    """Load, embed, build the index, print statistics and checks."""
    chunks = load_chunks()
    kept = [c for c in chunks if len(c["text"]) >= MIN_INDEX_CHARS]
    skipped = [c for c in chunks if len(c["text"]) < MIN_INDEX_CHARS]
    print(f"chunks loaded: {len(chunks)} | skipped: {len(skipped)} | to index: {len(kept)}")
    for c in skipped:
        print(f"  skipped {c['chunk_id']} (page {c['page_start']}): {c['text']!r}")

    # What the model reads: breadcrumb plus text. What is stored and shown: the raw text.
    embed_inputs = [f"{breadcrumb(c)}\n{c['text']}" for c in kept]

    model = SentenceTransformer(MODEL_NAME)

    # Anything beyond the model's limit would be truncated silently, so measure it.
    token_counts = [len(ids) for ids in model.tokenizer(embed_inputs)["input_ids"]]
    over = sum(1 for n in token_counts if n > model.max_seq_length)
    print(f"model: {MODEL_NAME} | max tokens: {model.max_seq_length}")
    print(f"tokens min/avg/max: {min(token_counts)}/{sum(token_counts) // len(token_counts)}/{max(token_counts)}"
          f" | over the limit: {over}")

    vectors = model.encode(embed_inputs, batch_size=BATCH_SIZE,
                           normalize_embeddings=True, show_progress_bar=True)

    client = chromadb.PersistentClient(path=INDEX_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass   # the collection did not exist yet
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    metadata_keys = ("filename", "title", "doc_type", "date", "status",
                     "section", "heading", "page_start", "page_end")
    collection.add(
        ids=[c["chunk_id"] for c in kept],
        embeddings=vectors.tolist(),
        documents=[c["text"] for c in kept],
        metadatas=[{k: c[k] for k in metadata_keys} for c in kept],
    )

    print(f"\ncollection '{COLLECTION}' count: {collection.count()} (expected {len(kept)})")
    stored = collection.get(ids=[kept[len(kept) // 2]["chunk_id"]])
    print("one stored record:")
    print("  id:", stored["ids"][0])
    print("  metadata:", stored["metadatas"][0])
    print("  text starts:", repr(stored["documents"][0][:100]))


if __name__ == "__main__":
    main()
