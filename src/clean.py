"""Clean the raw page records: drop empty pages, strip furniture, normalise whitespace."""
import json
import re
from collections import defaultdict
from pathlib import Path

IN = Path("data/processed/pages_raw.jsonl")
OUT = Path("data/processed/pages_clean.jsonl")

MIN_CHARS = 50

# Rule 2: running footer, optionally with the page number glued to the front.
# Group 1 captures the section label after the pipe.
FOOTER = re.compile(r"^\d*Laws of the Game \d{4}/\d{2}\s*\|\s*(.+?)\s*$")
# Rule 3: a page number on its own line.
PAGE_NUMBER = re.compile(r"^\d{1,3}$")
# Rules 4 and 5: "2/4" style page counters.
PAGE_COUNTER = re.compile(r"^\d/\d$")
# Rule 4: circular letterhead.
CIRCULAR_NUMBER = re.compile(r"^Circular no\. \d+$")
CIRCULAR_EXACT = {"The International Football Association Board", "theifab.com", "www.theifab.com"}
CIRCULAR_PREFIXES = ("Münstergasse", "+41 (0)")


def load_pages():
    """Return the raw page records as a list of dicts."""
    with open(IN, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_text(text, doc_type, page):
    """Return (cleaned text, section) for one page."""
    lines = text.splitlines()

    # Rule 3 is tested on the page's original first line, before anything is removed,
    # so a body line that happens to be a number can never be mistaken for the page number.
    if lines and PAGE_NUMBER.match(lines[0].strip()):
        lines = lines[1:]

    section = ""
    kept = []
    for line in lines:
        stripped = line.strip()

        # Rule 2: capture the section label, then drop the line.
        match = FOOTER.match(stripped)
        if match:
            if not section:
                section = match.group(1)
            continue

        # Rule 3b: in the Laws book the printed number equals the PDF page index, so any
        # remaining line equal to this page's own number is a stray page number
        # (a repeated number, or one at the end of the page).
        if doc_type == "laws" and stripped == str(page):
            continue

        # Rule 4: circular letterhead.
        if doc_type == "circular":
            if (stripped in CIRCULAR_EXACT
                    or stripped.startswith(CIRCULAR_PREFIXES)
                    or CIRCULAR_NUMBER.match(stripped)
                    or PAGE_COUNTER.match(stripped)):
                continue

        # Rule 5: trials lose only the page counter.
        if doc_type == "trial" and PAGE_COUNTER.match(stripped):
            continue

        kept.append(line)

    # Rule 6: whitespace.
    text = "\n".join(kept).replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), section


def main():
    """Load, clean, write JSONL, print per-file stats."""
    pages = load_pages()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(lambda: {"in": 0, "dropped": 0, "before": 0, "after": 0})
    with open(OUT, "w", encoding="utf-8") as f:
        for page in pages:
            s = stats[page["filename"]]
            s["in"] += 1

            # Rule 1: drop near-empty pages.
            if len(page["text"]) < MIN_CHARS:
                s["dropped"] += 1
                continue

            text, section = clean_text(page["text"], page["doc_type"], page["page"])
            record = {**page, "text": text, "section": section}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            s["before"] += len(page["text"])
            s["after"] += len(text)

    for filename, s in stats.items():
        kept = s["in"] - s["dropped"]
        print(filename)
        print(f"  pages in: {s['in']} | dropped: {s['dropped']} | kept: {kept}")
        if kept:
            print(f"  avg chars before/after: {s['before'] // kept}/{s['after'] // kept}")


if __name__ == "__main__":
    main()
