"""Chunk the cleaned pages into retrieval units: heading-aware, size-capped."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

IN = Path("data/processed/pages_clean.jsonl")
OUT = Path("data/processed/chunks.jsonl")

MAX_CHARS = 1500       # hard cap per chunk (about 350 tokens, below the 512-token model limit)
MIN_CHARS = 300        # a chunk must have this much new text before a heading may close it
OVERLAP_CHARS = 200    # trailing text repeated at the start of the next chunk (size breaks only)

# A numbered heading such as "4. Disciplinary action" (Laws book only).
HEADING = re.compile(r"^\d{1,2}\. [A-Z]")


def load_pages():
    """Return the cleaned page records as a list of dicts."""
    with open(IN, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fill_sections(pages):
    """Copy the section label from page N-1 to page N when page N has none.

    Only the page's own original label counts, so a label is never passed on twice.
    Returns the Laws page numbers that still have no section afterwards.
    """
    original = {(p["filename"], p["page"]): p["section"] for p in pages}
    still_empty = []
    for p in pages:
        if not p["section"]:
            p["section"] = original.get((p["filename"], p["page"] - 1), "")
        if p["doc_type"] == "laws" and not p["section"]:
            still_empty.append(p["page"])
    return still_empty


def is_heading_line(text, doc_type):
    """True for a short numbered heading line in the Laws book."""
    return (doc_type == "laws" and len(text) < 70
            and not text.endswith(".") and bool(HEADING.match(text)))


def split_block(text):
    """Split a block longer than MAX_CHARS at sentence ends (hard split as a last resort)."""
    pieces = []
    current = ""
    for sentence in re.split(r"(?<=\.) ", text):
        while len(sentence) > MAX_CHARS:
            cut = sentence.rfind(" ", 0, MAX_CHARS)
            cut = cut if cut > 0 else MAX_CHARS
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if current and len(current) + 1 + len(sentence) > MAX_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return pieces


def to_blocks(pages):
    """Turn pages into blocks, one list per file. Returns ({filename: blocks}, oversized count).

    A block starts at a line beginning with a bullet or a space, or at a numbered heading.
    Other lines continue the current block. A page that opens with a lowercase line
    continues the last block of the previous page.
    """
    by_file = defaultdict(list)
    for p in pages:
        by_file[p["filename"]].append(p)

    result = {}
    oversized = 0
    for filename, file_pages in by_file.items():
        raw = []
        heading = ""
        last_section = None
        for page in sorted(file_pages, key=lambda p: p["page"]):
            if page["section"] != last_section:       # a new Law or part: headings restart
                heading = ""
                last_section = page["section"]
            meta = {k: page[k] for k in ("filename", "title", "doc_type", "date", "status")}
            first_line = True
            for line in page["text"].split("\n"):
                text = line.strip()
                if not text:
                    continue
                is_heading = is_heading_line(text, page["doc_type"])
                marker = is_heading or text.startswith("•") or line.startswith(" ")
                continues = bool(raw) and not marker and (not first_line or text[0].islower())
                first_line = False

                if continues:
                    raw[-1]["text"] += " " + text
                    raw[-1]["page_end"] = page["page"]
                else:
                    if is_heading:
                        heading = text
                    raw.append({"text": text, "page_start": page["page"], "page_end": page["page"],
                                "section": page["section"], "heading": heading,
                                "is_heading": is_heading, "meta": meta})

        blocks = []
        for block in raw:
            if len(block["text"]) <= MAX_CHARS:
                blocks.append(block)
                continue
            oversized += 1
            for i, piece in enumerate(split_block(block["text"])):
                blocks.append({**block, "text": piece, "is_heading": block["is_heading"] and i == 0})
        result[filename] = blocks
    return result, oversized


def text_len(blocks):
    """Length of the blocks once joined with newlines."""
    return sum(len(b["text"]) for b in blocks) + max(len(blocks) - 1, 0)


def take_overlap(blocks):
    """Trailing blocks whose joined length is at most OVERLAP_CHARS."""
    overlap = []
    for block in reversed(blocks):
        if text_len([block] + overlap) > OVERLAP_CHARS:
            break
        overlap.insert(0, block)
    return overlap


def make_chunk(blocks, new_start):
    """Build a chunk record from blocks; blocks before new_start are overlap."""
    first = blocks[new_start]
    return {**first["meta"],
            "section": first["section"],
            "heading": first["heading"],
            "page_start": min(b["page_start"] for b in blocks),
            "page_end": max(b["page_end"] for b in blocks),
            "text": "\n".join(b["text"] for b in blocks)}


def build_chunks(blocks):
    """Pack one file's blocks into chunks. Returns (chunks, Counter of why chunks closed)."""
    chunks = []
    reasons = Counter()
    current = []
    new_start = 0
    for block in blocks:
        if current:
            reason = None
            if block["section"] != current[-1]["section"]:
                reason = "section"
            elif block["is_heading"] and text_len(current[new_start:]) >= MIN_CHARS:
                reason = "heading"
            elif text_len(current) + 1 + len(block["text"]) > MAX_CHARS:
                reason = "size"

            if reason:
                chunks.append(make_chunk(current, new_start))
                reasons[reason] += 1
                overlap = []
                if reason == "size" and not block["is_heading"]:
                    overlap = take_overlap(current)
                    if overlap and text_len(overlap) + 1 + len(block["text"]) > MAX_CHARS:
                        overlap = []
                current = overlap
                new_start = len(overlap)
        current.append(block)
    if len(current) > new_start:
        chunks.append(make_chunk(current, new_start))
    return chunks, reasons


def main():
    """Load, chunk every file, write JSONL, print statistics."""
    pages = load_pages()
    still_empty = fill_sections(pages)
    file_blocks, oversized = to_blocks(pages)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_reasons = Counter()
    over_max = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for filename, blocks in file_blocks.items():
            chunks, reasons = build_chunks(blocks)
            all_reasons.update(reasons)
            stem = Path(filename).stem
            lengths = []
            for n, chunk in enumerate(chunks, start=1):
                record = {"chunk_id": f"{stem}_{n:04d}", **chunk}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                lengths.append(len(chunk["text"]))
            over_max += sum(1 for n in lengths if n > MAX_CHARS)
            print(filename)
            print(f"  chunks: {len(chunks)} | chars min/avg/max: "
                  f"{min(lengths)}/{sum(lengths) // len(lengths)}/{max(lengths)}")

    print(f"\nchunks over {MAX_CHARS} chars: {over_max}")
    print(f"oversized blocks split: {oversized}")
    print(f"chunks closed by: {dict(all_reasons)}")
    print(f"Laws pages with no section after filling: {still_empty}")


if __name__ == "__main__":
    main()
