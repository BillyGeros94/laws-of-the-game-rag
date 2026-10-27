"""Parse the PDFs listed in data/manifest.csv into raw page records."""
import csv
import json
from pathlib import Path

from pypdf import PdfReader

RAW_DIR = Path('data/raw')
MANIFEST = Path('data/manifest.csv')
OUT = Path('data/processed/pages_raw.jsonl')


def load_manifest():
    """Return the manifest rows as a list of dicts."""
    with open(MANIFEST, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        manifest = list(reader)
    return manifest


def check_files(manifest):
    """Raise if a manifest file is missing from RAW_DIR,
    or a PDF in RAW_DIR is not in the manifest."""
    on_disk = {p.name for p in RAW_DIR.glob('*.pdf')}
    in_manifest = {row['filename'] for row in manifest}
    missing_disk = in_manifest - on_disk
    missing_man = on_disk - in_manifest
    if missing_disk or missing_man:
        raise ValueError(
            f'Manifest and data/raw/ disagree.\n'
            f'  In manifest, not on disk: {sorted(missing_disk)}\n'
            f'  On disk, not in manifest: {sorted(missing_man)}'
        )

def extract_pages(entry):
    """Yield one dict per page: the entry's metadata plus "page" and "text"."""
    reader = PdfReader(RAW_DIR/entry['filename'])
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ''
        yield {**entry, 'page':number, 'text':text}

def main():
    """Load, check, extract everything, write JSONL, print per-file stats."""
    manifest = load_manifest()
    check_files(manifest)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for entry in manifest:
            records = list(extract_pages(entry))
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            lengths = [len(r["text"]) for r in records]
            near_empty = [r["page"] for r in records if len(r["text"]) < 50]
            print(entry["filename"])
            print(f"  pages: {len(records)} | chars min/avg/max: {min(lengths)}/{sum(lengths) // len(lengths)}/{max(lengths)}")
            print(f"  near-empty pages: {near_empty}")


if __name__ == "__main__":
    main()