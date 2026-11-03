# Laws of the Game Q&A

A retrieval-augmented generation (RAG) demo that answers questions about the IFAB **Laws of the Game 2026/27**. Every answer is built only from the retrieved passages, which are shown next to it so the answer can be checked.

**Live demo:** https://laws-of-the-game-rag-jfwbcgzktfjpzuuyobpk2e.streamlit.app/

> This is a portfolio project, not an official source. It can be wrong. Check the cited pages in the Laws before relying on an answer.

![The answer page](docs/screenshot-answer.png)

## What it covers

| Document | Used for |
|---|---|
| Laws of the Game 2026/27 (the full book) | The Laws, the Law changes, the glossary, the guidelines |
| IFAB Circulars 32, 33 and 34 | AGM decisions, additional Law changes, clarifications |
| Two trial protocols | Tactical goalkeeper injuries; VAR review of attacking-team offences at set pieces |

Sources and where they come from: [SOURCES.md](data/SOURCES.md). Questions are answered in English only.

## How it works

```
PDFs → parse → clean → chunk → embed + index

question → embedding search ─┐
         → keyword search ───┴→ fuse → rerank → gate → generate → cited answer
```

| Stage | Script | What it does |
|---|---|---|
| Parse | `src/parse.py` | Extracts the text of each page with `pypdf` into `pages_raw.jsonl` (275 pages) |
| Clean | `src/clean.py` | Removes running footers, page numbers and letterheads, drops near-empty pages and the redline pages (see below), and records each page's Law section |
| Chunk | `src/chunk.py` | Splits the text into 226 chunks (see below); 225 are indexed |
| Index | `src/ingest.py` | Embeds the chunks with `BAAI/bge-small-en-v1.5` (384 dimensions) and stores them in a persistent Chroma index with cosine distance |
| Retrieve | `src/retrieve.py`, `src/lexical.py` | Runs an embedding search and a BM25 keyword search, merges the top 20 of each, rescores the merged candidates with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and returns the best 4 |
| Answer | `src/rag.py`, `src/generate.py` | Applies the relevance gate, then asks `openai/gpt-oss-120b` on Groq (OpenAI-compatible API, temperature 0) to answer from the numbered sources |
| Demo | `app.py` | Streamlit page that shows the answer and the retrieved passages |

Settings (model names, the retrieval mode, k, the gate, the refusal sentence) live in one place: `src/config.py`.

### Design choices

- **Hybrid chunking.** The Laws have a clear structure (Law, section, sub-section), so chunks follow headings first and are then capped at 1,500 characters. A chunk is merged with a too-short neighbour (minimum 300 characters), and a 200-character overlap is added only where a long block had to be split. The longest chunk is 409 tokens, under the embedding model's 512-token limit, so nothing is silently truncated.
- **Context for the embedding, not for the reader.** Each chunk is embedded with a breadcrumb in front (document, Law, section heading), but the stored and displayed text is the original.
- **Query instruction.** `bge` was trained with an instruction prefix on questions, so every question is embedded with it.
- **Dropping the redline.** Pages 170–192 of the Laws book list every Law change with the deleted words struck through. Text extraction keeps struck-through words as ordinary words ("a maximum of six eight may be used"), so these pages are dropped; the current wording is in the Laws themselves. The plain summary of changes on pages 167–169 is kept.
- **Hybrid retrieval.** Embedding search finds passages that mean the same as the question; BM25 keyword search finds passages that share its exact words, which matters for rule text ("touches a match official", "goes directly into the goal"). The two rankings are merged by reciprocal rank fusion.
- **Fan terms and typos.** Before keyword search, a short hand-made glossary adds the Laws' words for terms the Laws avoid ("dive" → simulation, "booked" → caution, "back-pass", "keeper"), and words not found anywhere in the documents are corrected to a close word that is ("substitues" → "substitutes"). See `src/lexical.py`.
- **A reranker.** A cross-encoder reads the question and each candidate together, which judges relevance better than comparing two separate embeddings. It rescores the merged candidates and the best 4 are kept.
- **A relevance gate.** If the reranker's best score is below 2.0, the app refuses without calling the model. Its scores separate answerable from unanswerable questions far better than cosine distance did (see *Evaluation*).
- **A prompt with strict source rules.** The model may use only the numbered sources, must refuse with a fixed sentence when they do not contain the answer, must cite source numbers, and must say when a source is a trial protocol and not part of the Laws. Incident questions get a structured answer (decision, applicable Law, analysis, referee judgment).

## Evaluation

The evaluation set is [`data/eval/questions.csv`](data/eval/questions.csv): 26 questions, each with an expected answer, source file and pages, and a difficulty. It has 21 answerable questions (6 easy, 9 medium, 6 hard), including one with a false premise and one aimed at the strikethrough problem described below, and 5 unanswerable ones. Four questions were held out from any tuning. [`src/eval.py`](src/eval.py) runs every question through the system and saves the results; the graded results are in [`data/eval/results_v1_graded.csv`](data/eval/results_v1_graded.csv).


### Retrieval

A question counts as a hit when a relevant chunk (expected file, overlapping pages) is among the 4 retrieved. Each version 2 change was measured on its own, with `python src/eval.py --retrieval-only --mode <dense|hybrid|rerank>`, which makes no model calls:

| Version | Hit in top 4 (21 answerable) | Ranked first | Unanswerable refused by the gate (5) |
|---|---|---|---|
| v1: embedding search | 19 (90%) | 13 | 2 |
| + redline pages dropped | 19 (90%) | 15 | 2 |
| + keyword search, fused | 18 (86%) | 16 | 2 |
| **+ reranker (v2)** | **20 (95%)** | **16** | **4** |

The gate wrongly refuses one answerable question only in the keyword-search row: it uses distance, and after fusion none of the top 4 chunks for Q14 is within 0.32 even though the closest one in the whole index is. Version 2 gates on the reranker's score and refuses no answerable question.

- **Dropping the redline** did not change which questions were hits, but moved the right passage up for three of them, because the redline chunks were crowding the top 4 with near-duplicates of the rules.
- **Keyword search on its own** was a mixed result: one more question ranked first, one hit lost. Fusion gives a keyword match and an embedding match equal weight, so a keyword match on common words can push out a better passage.
- **The reranker** recovers that hit and also finds the Law 9 passage for the ball-touches-the-referee question (*Failure examples*), which both searches had found but ranked too low.
- **The gate** moved from cosine distance to the reranker's score. On these 26 questions every answerable one scores 3.4 or more and every unanswerable one 2.6 or less; with distance the two groups overlapped (answerable up to 0.31, unanswerable from 0.21). The 2.0 threshold was chosen from these same questions, so it still has to prove itself on new ones; [`data/eval/questions_new_draft.csv`](data/eval/questions_new_draft.csv) is a first draft of such a set.

### Answers

Both versions answered the same 26 questions with the same model and prompt; only retrieval and the gate differ. Graded results: [version 1](data/eval/results_v1_graded.csv), [version 2](data/eval/results_v2_graded.csv).

| Measure (21 answerable questions) | v1 | v2 |
|---|---|---|
| **Correct** (true and complete; incomplete counts as half) | **16 (76%)** | **16 (76%)** |
| Grounded (no claim beyond the retrieved text) | 17 | 20 |
| Cited (citations support what is attributed to them) | 20 | 17 |
| Refused although answerable | 1 | 4 |
| **Confidently wrong** | **3** | **0** |
| Unanswerable questions refused (5) | 5 | 5 |

Correctness did not move, but the kind of error did. Version 1 gave three convincing wrong answers (Q09, Q12, Q16); version 2 gives none, fixes two of them (Q09 fully, Q12 in part) and refuses the third. Its mistakes are now refusals and incomplete answers, which a user can see, not wrong rules stated with citations. Cited falls because a refusal earns no citation credit.

Version 2 also refuses three questions version 1 got right:

- **Q20**, the ball touching the referee inside the penalty area: the Law 9 passage fell out of the top 4, so the model saw only the dropped-ball procedure. This is a retrieval regression.
- **Q16**, "Can a player be sent off for what they say?": the sending-off passage was retrieved at rank 4, but the model still refused.
- **Q18**, the false-premise question ("six seconds"): a plain refusal where version 1 corrected the premise.

Each answer is scored on three checks: **correct** (1, 0.5 for true but incomplete, or 0), **grounded** and **cited**. Unanswerable questions score only on a correct refusal. A refusal of an answerable question scores 0 for correct and for cited, with credit for grounded; the false-premise question allows half credit for a plain refusal, as its notes say. Weighting questions by difficulty (1, 2 or 3; unanswerable 2) and averaging the three checks gives:

| Tier | v1 | v2 |
|---|---|---|
| Easy | 6.0 / 6 | 6.0 / 6 |
| Medium | 15.0 / 18 | 16.7 / 18 |
| Hard | 12.5 / 18 | 10.0 / 18 |
| Unanswerable | 10 / 10 | 10 / 10 |
| **Total** | **43.5 / 52 (84%)** | **42.7 / 52 (82%)** |

The weighted score rewards a wrong answer that cites its sources with a third of its weight, so it favours version 1's confident mistakes over version 2's refusals; correctness and the confidently-wrong count are the better guide.

**Read these numbers with care.** There are only 26 questions, so one question moves the score by several points. The author graded version 1, and version 2's grades are a draft by an AI assistant checked by the author. The questions were written with the source text open (so their wording may sit closer to the Laws than real users' questions would), version 2 was tuned on them, and each question was run once. The same model can answer differently on a repeat run even at temperature 0.

### Failure examples

These are version 1 failures, each with what version 2 changed.

- **A confident wrong rule.** Asked what happens if the ball touches the referee and goes into the goal, the system answered that the goal stands. Law 9 says the ball is out of play and play restarts with a dropped ball. The Law 9 passage was not retrieved. The closest passage was a Law 5 clause about extra balls, objects and animals, and the model treated the referee as an "object". It quoted that clause accurately, which makes the answer convincing. The sources panel shows the mismatch: the cited page is about animals and extra balls. *Version 2:* the Law 9 passage is now retrieved, ranked second.
- **A vocabulary gap.** "What is the penalty for a player who dives to win a free kick?" was refused, because the passage that says "simulation" was not retrieved. Rephrased as "simulates a fall", the system gave the right verdict, but only by linking the glossary's definition of "deceive" to unsporting behaviour. The passage that states the rule outright was still not retrieved. *Version 2:* the fan-term glossary makes keyword search rank that passage first, but the reranker then scores it low and drops it: it is a long list of cautionable offences in which "simulation" is one line. This is still a miss, and an example of a reranker overruling a correct keyword match.
- **A typo.** "How many substitues are allowed?" was refused by the distance gate, because the misspelling moved the best distance from 0.225 to 0.362. The right passages were still ranked first. *Version 2:* the spelling correction turns it into "substitutes", and it retrieves and scores exactly as the correctly spelled question does.

## Known limitations

- **The reranker is a small general-purpose model.** It was trained on web search, not on football rules, and it misjudges long list-like passages (see *Failure examples*). It also adds a model download and some CPU time to every question.
- **The fan-term glossary is hand-made and short.** It covers a few common terms; others ("pen", "cycling shorts") rely on the embedding search. It was written after seeing the "dives" failure, so its effect on that question is not independent evidence.
- **Diagrams and image-only pages are lost.** About 30 pages with images lose their diagram content, which probably includes the handball diagrams. Four pages were image-only and were dropped.
- **The Law changes lose their explanations.** Dropping the redline pages (170–192) also drops IFAB's short explanation of why each Law changed. Questions about the reasons for a change rely on the circulars and the summary on pages 167–169.
- **The gate was tuned on the questions it is measured on.** See *Retrieval*. The prompt still does the refusing for questions the gate lets through.
- **Long list-like chunks and short chunks with several topics are hard to retrieve.**
- **The model sometimes adds detail**, misapplies a nearby rule, or omits facts that were in the retrieved text. Its native citation marks (`【1†…】`) are now rewritten as `[1]`.
- **English only.** The embedding model is English-only.
- **Some extraction defects:** one page of the VAR trial protocol extracts badly, and Law 12 sub-headings are missing from the chunk metadata.
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
python src/eval.py --retrieval-only    # retrieval hit rate only, no model calls
python src/eval.py --retrieval-only --mode dense    # compare: dense, hybrid or rerank
python src/eval.py                     # full evaluation; writes data/eval/results.csv
python -m unittest discover tests      # unit tests; no models or API calls
```

Changing the embedding model means changing `src/config.py` and re-running `src/ingest.py`.

## Notes

- **Shared quota.** The demo uses one free Groq key shared by all visitors, so it will sometimes answer that the service is busy. Try again later.
- **Privacy.** Questions you type are sent to Groq to generate the answer. Do not enter anything personal.

## License

The code in this repository is released under the [MIT License](LICENSE). The Laws of the Game and the IFAB circulars and protocols are published by The IFAB and are not covered by this license; they remain the property of their owner. See [SOURCES.md](data/SOURCES.md).
