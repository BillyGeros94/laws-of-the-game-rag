"""Run the evaluation questions through the pipeline and write a results file for grading.

Usage:
    python src/eval.py                    full run: retrieval and answer for every question
    python src/eval.py --retrieval-only   retrieval only: no answers (grade mode still makes
                                          one grading call per question)
    python src/eval.py --retrieval-only --mode dense   compare retrieval modes (dense, hybrid, grade)
    python src/eval.py --questions data/eval/questions_new_draft.csv   another question set
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

from openai import RateLimitError

from config import RETRIEVAL_MODE
from generate import is_refusal
from rag import answer, gate_refuses
from retrieve import MODES, pages, retrieve

QUESTIONS = Path("data/eval/questions.csv")
# Results go next to the questions file. For questions.csv: results.csv and
# results_retrieval_<mode>.csv; for another file, its name is added, e.g. results_new_draft.csv.

# Groq's free tier allows 8,000 tokens a minute per model; grading uses about 2,500 tokens
# and an answer about 2,000.
PAUSE_SECONDS = 25   # wait between questions that called a model
RETRY_WAIT = 65      # wait after a rate-limit error before the one retry

FIELDS = ["id", "set", "difficulty", "answerable", "question", "expected_answer", "expected_phrases",
          "expected_files", "expected_pages", "question_notes",
          "model_answer", "gate_refuses", "answer_is_refusal", "hit_rank", "top_distance",
          "top_grade", "grading_failed", "sources_sent",
          "sources", "finish_reason", "prompt_tokens", "completion_tokens",
          "correct", "grounded", "cited", "grading_notes"]   # the last four are for you to fill in


def results_path(questions, retrieval_only, mode):
    """Where a run's results are written."""
    tag = "" if questions.stem == "questions" else "_" + questions.stem.removeprefix("questions_")
    name = f"results{tag}_retrieval_{mode}.csv" if retrieval_only else f"results{tag}.csv"
    return questions.with_name(name)


def load_questions(path=QUESTIONS):
    """Return the question rows as a list of dicts."""
    if not path.exists():
        raise SystemExit(f"Questions file not found: {path}.")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_expected(row):
    """Turn expected_files and expected_pages into {filename: [(first, last), ...]}.

    Files are separated by ';'. Pages are separated by ';' per file, and by ',' for several
    ranges of the same file. With one file, every ';' group belongs to that file."""
    files = [x.strip() for x in row["expected_files"].split(";") if x.strip()]
    groups = [x.strip() for x in row["expected_pages"].split(";") if x.strip()]
    if not files:
        return {}
    if len(files) == len(groups):
        per_file = dict(zip(files, [[g] for g in groups]))
    elif len(files) == 1:
        per_file = {files[0]: groups}
    else:
        raise ValueError(f"{row['id']}: {len(files)} files but {len(groups)} page groups; "
                         f"cannot tell which pages belong to which file")

    expected = {}
    for filename, group_list in per_file.items():
        ranges = []
        for group in group_list:
            for part in group.split(","):
                part = part.strip()
                if not part:
                    continue
                first, _, last = part.partition("-")
                try:
                    ranges.append((int(first), int(last or first)))
                except ValueError:
                    raise ValueError(f"{row['id']}: cannot read the page range {part!r}") from None
        expected[filename] = ranges
    return expected


def parse_phrases(row):
    """The key phrases of a question ('|'-separated), lowercased. Empty if the column is absent."""
    return [p.strip().lower() for p in row.get("expected_phrases", "").split("|") if p.strip()]


def normalise(text):
    """Lowercase, straight apostrophes, single spaces: how phrases are matched."""
    return re.sub(r"\s+", " ", text.lower().replace("’", "'"))


def hit_rank(hits, expected, phrases=()):
    """Rank (1 = best) of the first retrieved chunk that is relevant. Empty string if none.

    Relevant means: from an expected file, overlapping an expected page range and, when the
    question has key phrases, with at least one of them in the text the model is given (the
    chunk's context, or the chunk). Page overlap alone counted chunks that only shared a page
    with the answer."""
    for rank, hit in enumerate(hits, start=1):
        on_page = any(hit["page_start"] <= last and hit["page_end"] >= first
                      for first, last in expected.get(hit["filename"], []))
        if on_page and (not phrases or any(p in normalise(hit.get("context") or hit["text"])
                                           for p in phrases)):
            return rank
    return ""


def ask(question, mode):
    """Answer a question; on a rate-limit error wait and retry once."""
    try:
        return answer(question, mode=mode)
    except RateLimitError:
        print(f"  rate limit reached; waiting {RETRY_WAIT} s and retrying once")
        time.sleep(RETRY_WAIT)
        return answer(question, mode=mode)   # a second error propagates


def write_results(path, rows):
    """Write all rows so far, so a stopped run still leaves a usable file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows, retrieval_only):
    """Print the hit rate overall and by difficulty, plus refusals and tokens."""
    with_expected = [r for r in rows if r["expected_files"]]
    without = [r for r in rows if not r["expected_files"]]

    print("\nSummary")
    if with_expected:
        hits = sum(1 for r in with_expected if r["hit_rank"] != "")
        first = sum(1 for r in with_expected if r["hit_rank"] == 1)
        print(f"  retrieval hit rate: {hits}/{len(with_expected)} "
              f"({100 * hits / len(with_expected):.0f}%) | ranked first: {first}/{len(with_expected)}")
        for level in ("1", "2", "3"):
            subset = [r for r in with_expected if r["difficulty"] == level]
            if subset:
                got = sum(1 for r in subset if r["hit_rank"] != "")
                print(f"    difficulty {level}: {got}/{len(subset)}")
    sent = [int(r["sources_sent"]) for r in rows if str(r.get("sources_sent", "")).isdigit()]
    if sent:
        print(f"  sources per question: average {sum(sent) / len(sent):.1f}, max {max(sent)}")
    failed = sum(1 for r in rows if r.get("grading_failed") is True)
    if failed:
        print(f"  WARNING: grading failed for {failed} question(s); their retrieval is the fallback")
    if without:
        by_gate = sum(1 for r in without if r["gate_refuses"])
        print(f"  unanswerable questions: {len(without)} | the gate refuses {by_gate}")
        if not retrieval_only:
            refused = sum(1 for r in without if r["answer_is_refusal"])
            print(f"    refused overall (gate or model): {refused}/{len(without)}")
    if not retrieval_only:
        wrongly = sum(1 for r in with_expected if r["answer_is_refusal"])
        print(f"  answerable questions refused: {wrongly}/{len(with_expected)}")
        prompt = sum(int(r["prompt_tokens"] or 0) for r in rows)
        completion = sum(int(r["completion_tokens"] or 0) for r in rows)
        print(f"  tokens used: prompt {prompt}, answer {completion}, total {prompt + completion}")


def main():
    """Run every question and save the results."""
    parser = argparse.ArgumentParser(description="Run the evaluation questions.")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="only retrieve; no answers (grade mode still grades)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS,
                        help=f"questions file (default: {QUESTIONS})")
    parser.add_argument("--mode", choices=MODES, default=RETRIEVAL_MODE,
                        help=f"retrieval mode (default: {RETRIEVAL_MODE})")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    questions = load_questions(args.questions)
    expectations = {q["id"]: parse_expected(q) for q in questions}   # fail early on bad rows
    path = results_path(args.questions, args.retrieval_only, args.mode)

    rows = []
    for index, q in enumerate(questions):
        row = {field: q.get(field, "") for field in FIELDS}
        row["question_notes"] = q.get("notes", "")

        def run():
            if args.retrieval_only:
                return retrieve(q["question"], mode=args.mode), None
            result = ask(q["question"], args.mode)
            return result["hits"], result

        try:
            hits, result = run()
            if args.mode == "grade" and hits and "grade" not in hits[0]:
                # Grading failed (most likely a rate limit) and retrieval fell back. Measuring the
                # fallback would be misleading, so wait and try once more.
                print(f"  {q['id']}: grading failed; waiting {RETRY_WAIT} s and retrying once")
                time.sleep(RETRY_WAIT)
                hits, result = run()
        except RateLimitError:
            print(f"\n{q['id']}: the rate limit is still reached after waiting. "
                  f"A daily limit clears the next day. Saving the {len(rows)} finished rows.")
            break

        top = hits[0]["distance"] if hits else None
        row["top_distance"] = f"{top:.4f}" if top is not None else ""
        row["top_grade"] = hits[0].get("grade", "") if hits else ""
        row["grading_failed"] = args.mode == "grade" and row["top_grade"] == ""
        row["sources_sent"] = len(hits)
        row["gate_refuses"] = gate_refuses(hits)
        row["hit_rank"] = hit_rank(hits, expectations[q["id"]], parse_phrases(q))
        row["sources"] = "; ".join(f"[{n}] {h['filename']} {pages(h)} d={h['distance']:.3f}"
                                   + (f" g={h['grade']}" if "grade" in h else "")
                                   for n, h in enumerate(hits, start=1))
        if result is not None:
            row["model_answer"] = result["answer"]
            row["answer_is_refusal"] = is_refusal(result["answer"])
            row["finish_reason"] = result["finish_reason"] or ""
            row["prompt_tokens"] = result["prompt_tokens"]
            row["completion_tokens"] = result["completion_tokens"]
        rows.append(row)
        write_results(path, rows)

        status = "refused by the gate" if row["gate_refuses"] else "passed the gate"
        score = f" | grade {row['top_grade']}, {len(hits)} sources" if row["top_grade"] != "" else ""
        failed = " | GRADING FAILED (fallback)" if row["grading_failed"] else ""
        print(f"{q['id']} | hit rank {row['hit_rank'] or '-'} | distance {row['top_distance']}{score}"
              f" | {status}{failed}")

        if index < len(questions) - 1:
            if args.mode == "grade" or (result is not None and not result["refused"]):
                time.sleep(PAUSE_SECONDS)

    summarise(rows, args.retrieval_only)
    print(f"\nMode: {args.mode} | results written to {path}")


if __name__ == "__main__":
    main()
