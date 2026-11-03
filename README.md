# Laws of the Game Q&A

A retrieval-augmented generation (RAG) demo that answers questions about the IFAB **Laws of the Game 2026/27**. Every answer is built only from the retrieved passages, which are shown next to it so the answer can be checked.

**Live demo:** https://laws-of-the-game-rag-jfwbcgzktfjpzuuyobpk2e.streamlit.app/

> This is a portfolio project, not an official source. It can be wrong. Check the cited pages in the Laws before relying on an answer.

![The answer page](docs/screenshot-answer.png)

## What it covers

| Document | Used for |
|---|---|
| Laws of the Game 2026/27 (the full book) | The Laws, the summary of Law changes, the glossary, the guidelines |
| IFAB Circulars 32, 33 and 34 | AGM decisions, additional Law changes, clarifications |
| Two trial protocols | Tactical goalkeeper injuries; VAR review of attacking-team offences at set pieces |

Sources and where they come from: [SOURCES.md](data/SOURCES.md). Questions are answered in English only.

## How it works

```
PDFs → parse → clean → chunk → embed + index

question → embedding search ─┐
         → keyword search ───┴→ fuse → grade (LLM) → gate → generate → cited answer
```

| Stage | Script | What it does |
|---|---|---|
| Parse | `src/parse.py` | Extracts the text of each page with `pypdf` into `pages_raw.jsonl` (275 pages) |
| Clean | `src/clean.py` | Removes running footers, page numbers and letterheads, drops near-empty pages and the redline pages (see below), and records each page's Law section |
| Chunk | `src/chunk.py` | Splits the text into 296 chunks that follow the structure of the Laws, each with a larger context (see below); 295 are indexed |
| Index | `src/ingest.py` | Embeds the chunks with `BAAI/bge-small-en-v1.5` (384 dimensions) and stores them in a persistent Chroma index with cosine distance |
| Retrieve | `src/retrieve.py`, `src/lexical.py`, `src/grade.py` | Runs an embedding search and a BM25 keyword search and merges them; `openai/gpt-oss-20b` on Groq grades the top 12 from 0 to 3, and every passage graded 2 or more is kept (at most 6) |
| Answer | `src/rag.py`, `src/generate.py` | Applies the gate, then asks `openai/gpt-oss-120b` on Groq (OpenAI-compatible API, temperature 0) to answer from the numbered sources |
| Demo | `app.py` | Streamlit page that shows the answer and the passages it was given |

Settings (model names, the retrieval mode, the grading limits, the gate, the refusal sentence) live in one place: `src/config.py`.

### Design choices

- **Chunks that follow the Laws.** The Laws book is organised as Law > numbered section ("4. Disciplinary action") > unnumbered sub-heading ("Cautions for unsporting behaviour") > text and lists. A chunk does not cross any of these boundaries unless it is too short to stand alone (minimum 300 characters), and is capped at 1,500 characters, with a 200-character overlap only where a long block had to be split. The longest chunk is 409 tokens, under the embedding model's 512-token limit, so nothing is silently truncated.
- **Sub-headings found by their shape.** Unnumbered sub-headings are not marked in the extracted text. A line counts as one when it is short, in sentence case, has no end punctuation, and stands on its own (the previous line ends a sentence or the next one starts a paragraph). Capitalised diagram labels, table rows, the glossary and the front matter are excluded. The rule finds 117 sub-headings, checked by hand.
- **Long lists split into small groups.** Many rules are single items in long lists; the simulation rule, for example, is one of 16 cautionable offences. A whole list in one chunk matches no single question well, so lists of more than 7 items are split into groups of at most 5, each starting with the list's opening line ("A player is cautioned if they:").
- **Small to big.** Each chunk also stores a context: the surrounding text of the same sub-section, up to 2,500 characters. Searching uses the small chunk, so matches are precise; the model answers from the context, so it sees the whole rule, and no two sources it receives share a context.
- **Context for the search, not for the reader.** Each chunk is searched with a breadcrumb in front (document, Law, section, sub-heading), but the stored and displayed text is the original.
- **Query instruction.** `bge` was trained with an instruction prefix on questions, so every question is embedded with it.
- **No redline.** Pages 170–192 of the Laws book list every Law change with the deleted words struck through. Text extraction keeps struck-through words as ordinary words ("a maximum of six eight may be used"), so these pages are left out; the current wording is in the Laws themselves. The plain summary of changes on pages 167–169 is kept.
- **Hybrid retrieval.** Embedding search finds passages that mean the same as the question; BM25 keyword search finds passages that share its exact words, which matters for rule text ("touches a match official", "goes directly into the goal"). The two rankings are merged by reciprocal rank fusion.
- **Fan terms and typos.** Before keyword search, a short hand-made glossary adds the Laws' words for terms the Laws avoid ("dive" → simulation, "booked" → caution, "back-pass", "keeper"), and words not found anywhere in the documents are corrected to a close word that is ("substitues" → "substitutes"). See `src/lexical.py`.
- **A language model grades the candidates.** `gpt-oss-20b` reads the question and the 12 best search results (each shortened to 600 characters) and scores each one: 3 states the rule asked about, 2 part of it, 1 same topic, 0 unrelated. Unlike the search, it links everyday words to the Laws' terms ("dives to win a free kick" to the simulation rule). Every passage graded 2 or more goes to the answer model, at most 6, so a simple question gets one or two sources and a question needing two rules can get both.
- **A gate that the grader does not control.** A passage graded 2 or more always passes. When none is, the grader is not trusted to refuse: it sometimes grades every passage 0, including the one that answers the question, and it judges false premises ("six seconds") rather than topics. The top 4 search results then go to the answer model, which refuses if they do not answer. Only questions far from anything in the Laws (cosine distance above 0.32) are refused without a model call.
- **A prompt with strict source rules.** The model may use only the numbered sources, must refuse with a fixed sentence when they do not contain the answer, must cite source numbers, and must say when a source is a trial protocol and not part of the Laws. Incident questions get a structured answer (decision, applicable Law, analysis, referee judgment).

## Evaluation

The evaluation set is [`data/eval/questions.csv`](data/eval/questions.csv): 26 questions, each with an expected answer, source file and pages, key phrases and a difficulty. It has 21 answerable questions (6 easy, 9 medium, 6 hard), including one with a false premise, and 5 unanswerable ones. [`src/eval.py`](src/eval.py) runs every question through the system and saves the results; the graded results are in [`data/eval/results_graded.csv`](data/eval/results_graded.csv).

### Retrieval

A question counts as a hit when a passage sent to the answer model comes from the expected file, overlaps the expected pages, and contains one of the question's key phrases (`expected_phrases`), so a passage that only shares a page with the answer does not count. `python src/eval.py --retrieval-only` measures this.

| Measure (21 answerable questions) | Result |
|---|---|
| Relevant passage among those sent | 20 (95%) |
| Relevant passage ranked first | 19 (90%) |
| Sources sent per question | 3.0 on average, at most 6 |

The one miss is Q20 (see *Failure examples*).

### Answers

Each answer is scored on three checks: **correct** (1, 0.5 for true but incomplete, or 0), **grounded** (no claim beyond the retrieved text) and **cited** (the cited source numbers support what is attributed to them). Unanswerable questions score only on a correct refusal. A refusal of an answerable question scores 0 for correct and for cited, with credit for grounded.

| Measure (21 answerable questions) | Result |
|---|---|
| **Correct** | **19.5 (93%)** |
| Grounded | 21 (100%) |
| Cited | 19 (90%) |
| Refused although answerable | 1 |
| Confidently wrong | 0 |
| Unanswerable questions refused | 5 / 5 (2 by the gate, 3 by the model) |

Weighting questions by difficulty (1, 2 or 3; unanswerable 2) and averaging the three checks:

| Tier | Weighted score |
|---|---|
| Easy | 6.0 / 6 |
| Medium | 17.7 / 18 |
| Hard | 15.0 / 18 |
| Unanswerable | 10 / 10 |
| **Total** | **48.7 / 52 (94%)** |

**Read these numbers with care.** There are only 26 questions, so one question moves the score by several points. The system was tuned on these same questions, they were written with the source text open (so their wording may sit closer to the Laws than real users' questions would), the grades are a draft by an AI assistant checked by the author, and each question was run once; the same model can answer differently on a repeat run even at temperature 0. A larger set of questions the system was not tuned on is in preparation; [`data/eval/questions_new_draft.csv`](data/eval/questions_new_draft.csv) is a first draft.

### Failure examples

- **A question that needs two rules.** "If the ball touches the referee inside the penalty area and goes into the goal, how is play restarted?" is refused. The answer needs Law 9 (the ball is out of play when it touches a match official and goes directly into the goal) and the dropped-ball procedure (inside the penalty area the ball is dropped for the goalkeeper). The grader keeps only the dropped-ball page, and without Law 9 the model does not answer.
- **A false premise left standing.** "Why is a goalkeeper allowed to hold the ball for six seconds?" gets the real rule (more than eight seconds, then a corner kick), but the answer treats six seconds as simply within the limit instead of saying that the question's premise is wrong.
- **The grader grading everything 0.** On 3 of 21 answerable questions (Q01, Q04, Q18) the grader scored every candidate 0, including the passage that answers the question. The gate does not refuse on grades alone, so the top search results went to the answer model instead, and all three were answered correctly.

## Known limitations

- **The grader is a language model and makes mistakes.** It sometimes grades every passage 0 (handled by the fallback above), and it can drop a passage a question needs, as with Law 9 for Q20. Each question costs a second model call.
- **The fan-term glossary is hand-made and short.** It covers a few common terms; others ("pen", "cycling shorts") rely on the embedding search and the grader. It was written with the evaluation questions in view, so its effect on them is not independent evidence.
- **Diagrams and image-only pages are lost.** About 30 pages with images lose their diagram content, which probably includes the handball diagrams. Four pages were image-only and were dropped.
- **The Law changes lose their explanations.** Leaving out the redline pages (170–192) also leaves out IFAB's short explanation of why each Law changed. Questions about the reasons for a change rely on the circulars and the summary on pages 167–169.
- **One free API key, shared by every visitor.** Groq's free tier allows 8,000 tokens a minute for each model. A grading call uses about 2,500 tokens and an answer about 2,000, so the demo can serve roughly three questions a minute before it reports that the service is busy.
- **Sub-headings are found by a heuristic.** A few diagram captions on the signal pages ("Free kick for attacking team", "Red and yellow card") are treated as sub-headings, and a sub-heading too short to stand alone is merged with the next one.
- **The model sometimes adds detail**, misapplies a nearby rule, or omits facts that were in the retrieved text. Its native citation marks (`【1†…】`) are rewritten as `[1]`.
- **English only.** The embedding model is English-only.
- **Some extraction defects:** one page of the VAR trial protocol extracts badly.
- **The IFAB documents are not covered by this repository's license** (see below).

## Run it yourself

```bash
git clone https://github.com/BillyGeros94/laws-of-the-game-rag.git
cd laws-of-the-game-rag
python -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt
export GROQ_API_KEY="your-key"   # a free key from console.groq.com; never commit it
streamlit run app.py
```

The repository includes the built index, so the demo runs without rebuilding it. To rebuild from the PDFs, download the documents listed in [SOURCES.md](data/SOURCES.md) into `data/raw/` and run, in order:

```bash
python src/parse.py && python src/clean.py && python src/chunk.py && python src/ingest.py
```

Other commands:

```bash
python src/rag.py "How many substitutes are allowed in a match?"   # command line
python src/eval.py --retrieval-only    # retrieval hit rate only; grading still calls the model
python src/eval.py --retrieval-only --mode dense    # compare: dense, hybrid or grade
python src/eval.py                     # full evaluation; writes data/eval/results.csv
python -m unittest discover tests      # unit tests; no models or API calls
```

Changing the embedding model means changing `src/config.py` and re-running `src/ingest.py`.

## Notes

- **Shared quota.** The demo uses one free Groq key shared by all visitors, so it will sometimes answer that the service is busy. Try again later.
- **Privacy.** Questions you type are sent to Groq to grade the passages and generate the answer. Do not enter anything personal.

## License

The code in this repository is released under the [MIT License](LICENSE). The Laws of the Game and the IFAB circulars and protocols are published by The IFAB and are not covered by this license; they remain the property of their owner. See [SOURCES.md](data/SOURCES.md).
