"""Chunk the cleaned pages into retrieval units that follow the structure of the Laws.

The Laws book is organised as Law > numbered section ("4. Disciplinary action") > unnumbered
sub-heading ("Cautions for unsporting behaviour") > text and lists. A chunk never crosses a
Law, a numbered section or a sub-heading (unless it is too short to stand alone), and long
lists are split into small groups of items, each repeating the list's opening line.

Small to big: each chunk is small so that it matches a question precisely, but it also
carries a "context", the surrounding text of the same sub-section up to CONTEXT_CHARS, and
the model answers from that context.
"""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

IN = Path("data/processed/pages_clean.jsonl")
OUT = Path("data/processed/chunks.jsonl")

MAX_CHARS = 1500       # hard cap per chunk (about 350 tokens, below the 512-token model limit)
MIN_CHARS = 300        # a chunk must have this much new text before a heading may close it
OVERLAP_CHARS = 200    # trailing text repeated at the start of the next chunk (size breaks only)
LIST_MAX_ITEMS = 7     # lists longer than this are split into groups...
LIST_GROUP_ITEMS = 5   # ...of at most this many items
CONTEXT_CHARS = 2500   # how much of the surrounding sub-section the model is given

# A numbered heading such as "4. Disciplinary action" (Laws book only).
HEADING = re.compile(r"^\d{1,2}\. [A-Z]")
BULLET = "•"

# Unnumbered sub-headings ("Cautions for unsporting behaviour") are found by their shape.
# Parts of the book where short lines are not sub-headings: front matter, logos, and the
# glossary, whose terms are already short self-contained definitions.
NO_SUBHEADINGS = ("", "Glossary", "FIFA Quality Programme", "Contents")
DIAGRAM_LABELS = {"Goalkeeper (GK)", "Defender", "AARs", "No AAR"}   # legends in the diagrams
ENDS_SENTENCE = (".", ":", ";", ")", "”", "’", "?", "!")


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


def sentence_case(text):
    """The Laws capitalise only the first word of a heading, plus acronyms ('(RRA)').
    Diagram labels and logos are in capitals or Title Case."""
    if text.isupper():
        return False
    for word in text.split()[1:]:
        bare = word.strip("()‘’'\"")
        if bare[:1].isupper() and not (bare.isupper() and len(bare) <= 5):
            return False
    return True


def is_subheading(line, prev, nxt, section, doc_type):
    """True for an unnumbered sub-heading line in the Laws book.

    line, prev and nxt are raw lines (prev/nxt None at the edges of a page). A sub-heading
    is short, in sentence case, without end punctuation, and stands on its own line: the
    line before it ends a sentence or the line after it starts a new paragraph (indented or
    a bullet), and the line after it does not continue it in lowercase."""
    text = line.strip()
    if doc_type != "laws" or section in NO_SUBHEADINGS:
        return False
    if not text or line.startswith(" ") or text.startswith(BULLET) or HEADING.match(text):
        return False
    if text in DIAGRAM_LABELS or ":" in text or re.search(r"\(\d+\)$", text):
        return False
    if not (3 <= len(text) <= 70) or len(text.split()) > 10:
        return False
    if not text[0].isupper() or text.endswith((".", ",", ";", ":", "–", "-")):
        return False
    if text.lower() in section.lower() or not sentence_case(text):
        return False   # a piece of the Law's own title, or a label
    if nxt is None or not nxt.strip() or nxt.strip()[0].islower():
        return False   # this line runs on into the next one
    starts_paragraph = nxt.startswith(" ") or nxt.strip().startswith(BULLET)
    if (prev is not None and not starts_paragraph and not prev.strip().endswith(ENDS_SENTENCE)
            and not HEADING.match(prev.strip())):
        return False   # the previous line runs on into this one: a wrapped sentence
    return True


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

    A block starts at a line beginning with a bullet or a space, or at a numbered heading or
    sub-heading. Other lines continue the current block. A page that opens with a lowercase
    line continues the last block of the previous page.
    """
    by_file = defaultdict(list)
    for p in pages:
        by_file[p["filename"]].append(p)

    result = {}
    oversized = 0
    for filename, file_pages in by_file.items():
        raw = []
        heading = subheading = ""
        last_section = None
        for page in sorted(file_pages, key=lambda p: p["page"]):
            if page["section"] != last_section:       # a new Law or part: headings restart
                heading = subheading = ""
                last_section = page["section"]
            meta = {k: page[k] for k in ("filename", "title", "doc_type", "date", "status")}
            lines = [line for line in page["text"].split("\n") if line.strip()]
            for i, line in enumerate(lines):
                text = line.strip()
                is_heading = is_heading_line(text, page["doc_type"])
                is_sub = not is_heading and is_subheading(
                    line, lines[i - 1] if i else None, lines[i + 1] if i + 1 < len(lines) else None,
                    page["section"], page["doc_type"])
                is_bullet = text.startswith(BULLET)
                marker = is_heading or is_sub or is_bullet or line.startswith(" ")
                continues = bool(raw) and not marker and (i > 0 or text[0].islower())

                if continues:
                    raw[-1]["text"] += " " + text
                    raw[-1]["page_end"] = page["page"]
                    continue
                if is_heading:
                    heading, subheading = text, ""
                elif is_sub:
                    subheading = text
                raw.append({"text": text, "page_start": page["page"], "page_end": page["page"],
                            "section": page["section"], "heading": heading, "subheading": subheading,
                            "is_heading": is_heading or is_sub, "is_bullet": is_bullet, "meta": meta})

        blocks = []
        for block in raw:
            if len(block["text"]) <= MAX_CHARS:
                blocks.append(block)
                continue
            oversized += 1
            for i, piece in enumerate(split_block(block["text"])):
                blocks.append({**block, "text": piece, "is_heading": block["is_heading"] and i == 0})
        mark_list_groups(blocks)
        result[filename] = blocks
    return result, oversized


def same_part(a, b):
    """True when two blocks belong to the same Law, numbered section and sub-heading."""
    return all(a[k] == b[k] for k in ("section", "heading", "subheading"))


def mark_list_groups(blocks):
    """Mark where long lists are split: block["list_break"] = index of the list's opening line
    (or None) on the first item of every group after the first.

    A list is a run of consecutive bullet blocks in the same part. Its opening line is the
    block just before it when that block ends with ':' ("A player is cautioned if guilty of:")."""
    i = 0
    while i < len(blocks):
        if not blocks[i]["is_bullet"]:
            i += 1
            continue
        end = i
        while (end + 1 < len(blocks) and blocks[end + 1]["is_bullet"]
               and same_part(blocks[end + 1], blocks[i])):
            end += 1
        count = end - i + 1
        if count > LIST_MAX_ITEMS:
            size = math.ceil(count / math.ceil(count / LIST_GROUP_ITEMS))   # balanced groups
            opener = i - 1 if i > 0 and blocks[i - 1]["text"].endswith(":") else None
            for start in range(i + size, end + 1, size):
                blocks[start]["list_break"] = opener
        i = end + 1


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


def context_for(indices, new_start, blocks):
    """The model's view of a chunk: its own blocks plus neighbouring blocks of the same part,
    added alternately after and before, up to CONTEXT_CHARS. A list's opening line that
    was carried into the chunk is put in front if the window does not reach it."""
    own = indices[new_start:]
    lo, hi = own[0], own[-1]
    part = blocks[hi]
    opener = [i for i in indices[:new_start] if i < lo - 1 and "list_break" in blocks[own[0]]]
    length = text_len(blocks[lo:hi + 1]) + sum(len(blocks[i]["text"]) + 1 for i in opener)
    if new_start and not opener:
        lo = indices[0]          # carried overlap blocks sit right before the chunk's own blocks
        length = text_len(blocks[lo:hi + 1])
    grew = True
    while grew:
        grew = False
        for step in ("after", "before"):
            j = hi + 1 if step == "after" else lo - 1
            if 0 <= j < len(blocks) and same_part(blocks[j], part) \
                    and length + 1 + len(blocks[j]["text"]) <= CONTEXT_CHARS:
                length += 1 + len(blocks[j]["text"])
                lo, hi = (lo, j) if step == "after" else (j, hi)
                grew = True
    window = [i for i in opener if i < lo] + list(range(lo, hi + 1))
    return "\n".join(blocks[i]["text"] for i in window)


def make_chunk(indices, new_start, blocks):
    """Build a chunk record from block indices; those before new_start are carried-over context
    (overlap or a list's opening line)."""
    chunk_blocks = [blocks[i] for i in indices]
    first = chunk_blocks[new_start]
    return {**first["meta"],
            "section": first["section"],
            "heading": first["heading"],
            "subheading": first["subheading"],
            "page_start": min(b["page_start"] for b in chunk_blocks),
            "page_end": max(b["page_end"] for b in chunk_blocks),
            "text": "\n".join(b["text"] for b in chunk_blocks),
            "context": context_for(indices, new_start, blocks)}


def build_chunks(blocks):
    """Pack one file's blocks into chunks. Returns (chunks, Counter of why chunks closed)."""
    chunks = []
    reasons = Counter()
    current = []        # block indices
    new_start = 0
    for index, block in enumerate(blocks):
        if current:
            last = blocks[current[-1]]
            reason = None
            if block["section"] != last["section"]:
                reason = "section"
            elif block["is_heading"] and text_len([blocks[i] for i in current[new_start:]]) >= MIN_CHARS:
                reason = "heading"
            elif "list_break" in block:
                reason = "list"
            elif text_len([blocks[i] for i in current]) + 1 + len(block["text"]) > MAX_CHARS:
                reason = "size"

            if reason:
                chunks.append(make_chunk(current, new_start, blocks))
                reasons[reason] += 1
                carried = []
                if reason == "list" and block["list_break"] is not None:
                    carried = [block["list_break"]]          # repeat the list's opening line
                elif reason == "size" and not block["is_heading"]:
                    overlap = take_overlap([blocks[i] for i in current])
                    carried = current[len(current) - len(overlap):]
                if carried and text_len([blocks[i] for i in carried]) + 1 + len(block["text"]) > MAX_CHARS:
                    carried = []
                current = carried
                new_start = len(carried)
        current.append(index)
    if len(current) > new_start:
        chunks.append(make_chunk(current, new_start, blocks))
    return chunks, reasons


def main():
    """Load, chunk every file, write JSONL, print statistics."""
    pages = load_pages()
    still_empty = fill_sections(pages)
    file_blocks, oversized = to_blocks(pages)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_reasons = Counter()
    over_max = 0
    subheadings = set()
    with open(OUT, "w", encoding="utf-8") as f:
        for filename, blocks in file_blocks.items():
            chunks, reasons = build_chunks(blocks)
            all_reasons.update(reasons)
            stem = Path(filename).stem
            lengths, contexts = [], []
            for n, chunk in enumerate(chunks, start=1):
                record = {"chunk_id": f"{stem}_{n:04d}", **chunk}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                lengths.append(len(chunk["text"]))
                contexts.append(len(chunk["context"]))
                if chunk["subheading"]:
                    subheadings.add((filename, chunk["section"], chunk["subheading"]))
            over_max += sum(1 for n in lengths if n > MAX_CHARS)
            print(filename)
            print(f"  chunks: {len(chunks)} | chars min/avg/max: "
                  f"{min(lengths)}/{sum(lengths) // len(lengths)}/{max(lengths)}"
                  f" | context avg/max: {sum(contexts) // len(contexts)}/{max(contexts)}")

    print(f"\nchunks over {MAX_CHARS} chars: {over_max}")
    print(f"oversized blocks split: {oversized}")
    print(f"chunks closed by: {dict(all_reasons)}")
    print(f"chunks under a sub-heading: {len(subheadings)} distinct sub-headings")
    print(f"Laws pages with no section after filling: {still_empty}")


if __name__ == "__main__":
    main()
