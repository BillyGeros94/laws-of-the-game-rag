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
| Demo | `app.py` | Streamlit page that shows the answer and the retrieved passages |

Settings (model names, the retrieval mode, k, the gate, the refusal sentence) live in one place: `src/config.py`.

### Design choices

- **Chunks that follow the Laws.** The Laws book is organised as Law > numbered section ("4. Disciplinary action") > unnumbered sub-heading ("Cautions for unsporting behaviour") > text and lists. A chunk does not cross any of these boundaries unless it is too short to stand alone (minimum 300 characters), and is capped at 1,500 characters, with a 200-character overlap only where a long block had to be split. The longest chunk is 409 tokens, under the embedding model's 512-token limit, so nothing is silently truncated.
- **Sub-headings found by their shape.** Unnumbered sub-headings are not marked in the extracted text. A line counts as one when it is short, in sentence case, has no end punctuation, and stands on its own (the previous line ends a sentence or the next one starts a paragraph). Capitalised diagram labels, table rows, the glossary and the front matter are excluded. The rule finds 117 sub-headings, checked by hand.
- **Long lists split into small groups.** Many rules are single items in long lists: the simulation rule was one of 16 cautionable offences in a 1,463-character chunk, so the chunk matched no single question well. Lists of more than 7 items are split into groups of at most 5, each starting with the list's opening line ("A player is cautioned if they:").
- **Small to big.** Each chunk also stores a context: the surrounding text of the same sub-section, up to 2,500 characters. Searching uses the small chunk, so matches are precise; the model answers from the context, so it sees the whole rule, and no two sources it receives share a context.
- **Context for the search, not for the reader.** Each chunk is searched with a breadcrumb in front (document, Law, section, sub-heading), but the stored and displayed text is the original.
- **Query instruction.** `bge` was trained with an instruction prefix on questions, so every question is embedded with it.
- **Dropping the redline.** Pages 170–192 of the Laws book list every Law change with the deleted words struck through. Text extraction keeps struck-through words as ordinary words ("a maximum of six eight may be used"), so these pages are dropped; the current wording is in the Laws themselves. The plain summary of changes on pages 167–169 is kept.
- **Hybrid retrieval.** Embedding search finds passages that mean the same as the question; BM25 keyword search finds passages that share its exact words, which matters for rule text ("touches a match official", "goes directly into the goal"). The two rankings are merged by reciprocal rank fusion.
- **Fan terms and typos.** Before keyword search, a short hand-made glossary adds the Laws' words for terms the Laws avoid ("dive" → simulation, "booked" → caution, "back-pass", "keeper"), and words not found anywhere in the documents are corrected to a close word that is ("substitues" → "substitutes"). See `src/lexical.py`.
- **A language model grades the candidates.** `gpt-oss-20b` reads the question and the 12 best search results (each shortened to 600 characters) and scores each one: 3 states the rule asked about, 2 part of it, 1 same topic, 0 unrelated. Every passage graded 2 or more goes to the answer model, at most 6, so a simple question gets one or two sources and a question needing two rules gets both. It replaced a small cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`, since removed), which could not link everyday words to the Laws' terms: it scored the simulation passage −0.6 for "dives to win a free kick" and preferred passages about free kicks. The grader scores it 3.
- **A gate that the grader does not control.** A passage graded 2 or more always passes. When none is, the grader is not trusted to refuse: on 3 of 21 answerable questions it graded every passage 0, including the one that answers it, and it judges false premises ("six seconds") rather than topics. The top 4 search results then go to the answer model, which refuses if they do not answer. Only questions far from anything in the Laws (cosine distance above 0.32) are refused without a model call.
- **A prompt with strict source rules.** The model may use only the numbered sources, must refuse with a fixed sentence when they do not contain the answer, must cite source numbers, and must say when a source is a trial protocol and not part of the Laws. Incident questions get a structured answer (decision, applicable Law, analysis, referee judgment).

## Evaluation

The evaluation set is [`data/eval/questions.csv`](data/eval/questions.csv): 26 questions, each with an expected answer, source file and pages, and a difficulty. It has 21 answerable questions (6 easy, 9 medium, 6 hard), including one with a false premise and one aimed at the strikethrough problem described below, and 5 unanswerable ones. Four questions were held out from any tuning. [`src/eval.py`](src/eval.py) runs every question through the system and saves the results; the graded results are in [`data/eval/results_v1_graded.csv`](data/eval/results_v1_graded.csv).


### Retrieval

A question counts as a hit when a relevant chunk (expected file, overlapping pages) is among the passages retrieved (4, or up to 6 in grade mode). Each version 2 change was measured on its own, with `python src/eval.py --retrieval-only --mode <dense|hybrid|rerank|grade>`; only grade mode calls a model:

| Version | Hit in top 4 (21 answerable) | Ranked first | Unanswerable refused by the gate (5) |
|---|---|---|---|
| v1: embedding search | 19 (90%) | 13 | 2 |
| + redline pages dropped | 19 (90%) | 15 | 2 |
| + keyword search, fused | 18 (86%) | 16 | 2 |
| **+ reranker (v2)** | **20 (95%)** | **16** | **4** |

**The table above uses a hit check that overcounted.** It counted a chunk as relevant when its pages overlapped the expected pages, so for Q16 ("Can a player be sent off for what they say?") a chunk about handball sending-offs on an overlapping page counted as a hit. Each question now also has key phrases (`expected_phrases` in `questions.csv`), and a hit must contain one of them in the text the model is given. Re-measured that way, with the structure-following chunks added as the last step:

| Version (key-phrase hit check) | Hit in top 4 | Ranked first |
|---|---|---|
| v2, embedding search only | 17 (81%) | 15 |
| v2, with reranker | 18 (86%) | 15 |
| + structure-following chunks, embedding search only | 18 (86%) | 17 |
| + structure-following chunks, with reranker | 19 (90%) | 15 |
| **+ language-model grading instead of the reranker (final)** | **20 (95%)** | **19** |

The new chunks let both searches rank the sending-off list first for Q16, which the old chunks never retrieved. (Q16's expected pages also gained p.123–124, where that list states the rule word for word.) The two remaining misses need the question reworded before searching: for Q17 the reranker scores the new simulation chunk at −0.6 against "dives to win a free kick", but at 4.5 when the question is reworded in the Laws' terms; Q20 needs two separate rules, Law 9 and the dropped-ball procedure.

Two ways to close that gap were tried:

- **Rewording the question first** (since removed): a small model rewrote each question in the Laws' terms before searching. It ranked one more question first, but it kept "to win a free kick" in Q17 (so the reranker still preferred free-kick passages), it was not consistent between runs ("simulation" in one, "simulates" in another), it sometimes put an answer into the query (a wrong restart for Q20), and it made two unanswerable questions sound answerable to the gate ("banned after a red card" became "the sanction for a player who is sent off").
- **Grading with a language model instead of the reranker** (kept): ranked first went from 15 to 19 and Q17 is found at rank 1. Run twice, retrieval was the same both times. Q19 counts as a hit because its expected pages gained p.167, the plain summary of this season's changes, which states the rule. Q20 is still missed: the grader keeps only the dropped-ball page and not Law 9.

The gate wrongly refuses one answerable question only in the keyword-search row: it uses distance, and after fusion none of the top 4 chunks for Q14 is within 0.32 even though the closest one in the whole index is. Version 2 gates on the reranker's score and refuses no answerable question.

- **Dropping the redline** did not change which questions were hits, but moved the right passage up for three of them, because the redline chunks were crowding the top 4 with near-duplicates of the rules.
- **Keyword search on its own** was a mixed result: one more question ranked first, one hit lost. Fusion gives a keyword match and an embedding match equal weight, so a keyword match on common words can push out a better passage.
- **The reranker** recovers that hit and also finds the Law 9 passage for the ball-touches-the-referee question (*Failure examples*), which both searches had found but ranked too low.
- **The gate** moved from cosine distance to the reranker's score (both the reranker and this gate were later replaced; see *A gate that the grader does not control*). On these 26 questions every answerable one scores 3.4 or more and every unanswerable one 2.6 or less; with distance the two groups overlapped (answerable up to 0.31, unanswerable from 0.21). The 2.0 threshold was chosen from these same questions, so it still has to prove itself on new ones; [`data/eval/questions_new_draft.csv`](data/eval/questions_new_draft.csv) is a first draft of such a set.

### Answers

All versions answered the same 26 questions with the same answer model and prompt; only chunking, retrieval and the gate differ. "v2 reranker" is the first version 2 (old chunks, cross-encoder reranker); "v2 final" adds the structure-following chunks and language-model grading. Graded results: [v1](data/eval/results_v1_graded.csv), [v2 reranker](data/eval/results_v2_graded.csv), [v2 final](data/eval/results_v2_grade_graded.csv).

| Measure (21 answerable questions) | v1 | v2 reranker | v2 final |
|---|---|---|---|
| **Correct** (true and complete; incomplete counts as half) | **16 (76%)** | **16 (76%)** | **19.5 (93%)** |
| Grounded (no claim beyond the retrieved text) | 17 | 20 | 21 |
| Cited (citations support what is attributed to them) | 20 | 17 | 19 |
| Refused although answerable | 1 | 4 | 1 |
| **Confidently wrong** | **3** | **0** | **0** |
| Unanswerable questions refused (5) | 5 | 5 | 5 |
| Sources sent per question (average) | 4 | 4 | 3.0 |

Correctness did not move, but the kind of error did. Version 1 gave three convincing wrong answers (Q09, Q12, Q16); version 2 gives none, fixes two of them (Q09 fully, Q12 in part) and refuses the third. Its mistakes are now refusals and incomplete answers, which a user can see, not wrong rules stated with citations. Cited falls because a refusal earns no citation credit.

Version 2 also refuses three questions version 1 got right:

- **Q20**, the ball touching the referee inside the penalty area: the Law 9 passage fell out of the top 4, so the model saw only the dropped-ball procedure. This is a retrieval regression.
- **Q16**, "Can a player be sent off for what they say?": the language rule was not retrieved (see *Retrieval*), so the refusal is correct for what the model was given. Version 1 answered it wrongly from unrelated passages.
- **Q18**, the false-premise question ("six seconds"): a plain refusal where version 1 corrected the premise.

**v2 final** answers all three of those except Q20, and also completes Q10 and Q21, which earlier versions left incomplete. Its only refusal is Q20, for the retrieval reason above. It gives no confidently wrong answer and every claim checked is in its sources. Q18 now states the real rule (eight seconds) but reads "six seconds" as within the limit instead of saying the premise is wrong.

Each answer is scored on three checks: **correct** (1, 0.5 for true but incomplete, or 0), **grounded** and **cited**. Unanswerable questions score only on a correct refusal. A refusal of an answerable question scores 0 for correct and for cited, with credit for grounded; the false-premise question allows half credit for a plain refusal, as its notes say. Weighting questions by difficulty (1, 2 or 3; unanswerable 2) and averaging the three checks gives:

| Tier | v1 | v2 reranker | v2 final |
|---|---|---|---|
| Easy | 6.0 / 6 | 6.0 / 6 | 6.0 / 6 |
| Medium | 15.0 / 18 | 16.7 / 18 | 17.7 / 18 |
| Hard | 12.5 / 18 | 10.0 / 18 | 15.0 / 18 |
| Unanswerable | 10 / 10 | 10 / 10 | 10 / 10 |
| **Total** | **43.5 / 52 (84%)** | **42.7 / 52 (82%)** | **48.7 / 52 (94%)** |

The weighted score rewards a wrong answer that cites its sources with a third of its weight, so it favours version 1's confident mistakes over version 2's refusals; correctness and the confidently-wrong count are the better guide.

**Read these numbers with care.** There are only 26 questions, so one question moves the score by several points. The author graded version 1, and version 2's grades are a draft by an AI assistant checked by the author. The questions were written with the source text open (so their wording may sit closer to the Laws than real users' questions would), version 2 was tuned on them (two questions' expected pages were also corrected along the way), and the final answers were generated once. The 94% has not yet been checked on questions the system was not tuned on; [`data/eval/questions_new_draft.csv`](data/eval/questions_new_draft.csv) is a draft of such a set. The same model can answer differently on a repeat run even at temperature 0.

### Failure examples

These are version 1 failures, each with what version 2 changed.

- **A confident wrong rule.** Asked what happens if the ball touches the referee and goes into the goal, the system answered that the goal stands. Law 9 says the ball is out of play and play restarts with a dropped ball. The Law 9 passage was not retrieved. The closest passage was a Law 5 clause about extra balls, objects and animals, and the model treated the referee as an "object". It quoted that clause accurately, which makes the answer convincing. The sources panel shows the mismatch: the cited page is about animals and extra balls. *Version 2:* the Law 9 passage is now retrieved, ranked second.
- **A vocabulary gap.** "What is the penalty for a player who dives to win a free kick?" was refused, because the passage that says "simulation" was not retrieved. Rephrased as "simulates a fall", the system gave the right verdict, but only by linking the glossary's definition of "deceive" to unsporting behaviour. The passage that states the rule outright was still not retrieved. *Version 2:* the fan-term glossary makes keyword search rank that passage first, but the reranker then scores it low and drops it: it is a long list of cautionable offences in which "simulation" is one line. Splitting that list into a small chunk under "Cautions for unsporting behaviour" did not change this: the reranker still prefers passages about free kicks. It was an example of a reranker overruling a correct keyword match. *v2 final:* the language-model grader scores the simulation passage 3, and the answer is right: a caution for unsporting behaviour.
- **A typo.** "How many substitues are allowed?" was refused by the distance gate, because the misspelling moved the best distance from 0.225 to 0.362. The right passages were still ranked first. *Version 2:* the spelling correction turns it into "substitutes", and it retrieves and scores exactly as the correctly spelled question does.

## Known limitations

- **The grader is a language model and makes mistakes.** On 3 of 21 answerable questions it graded every passage 0, including the one that answers the question; the gate then falls back to the search order, which had the right passage first each time. It can also drop a passage a question needs (Law 9 for Q20). Each question costs a second model call.
- **The fan-term glossary is hand-made and short.** It covers a few common terms; others ("pen", "cycling shorts") rely on the embedding search. It was written after seeing the "dives" failure, so its effect on that question is not independent evidence.
- **Diagrams and image-only pages are lost.** About 30 pages with images lose their diagram content, which probably includes the handball diagrams. Four pages were image-only and were dropped.
- **The Law changes lose their explanations.** Dropping the redline pages (170–192) also drops IFAB's short explanation of why each Law changed. Questions about the reasons for a change rely on the circulars and the summary on pages 167–169.
- **One free API key, shared by every visitor.** Groq's free tier allows 8,000 tokens a minute for each model. A grading call uses about 2,500 tokens and an answer about 2,000, so the demo can serve roughly three questions a minute before it reports that the service is busy.
- **Sub-headings are found by a heuristic.** A few diagram captions on the signal pages ("Free kick for attacking team", "Red and yellow card") are treated as sub-headings, and a sub-heading too short to stand alone is merged with the next one.
- **The model sometimes adds detail**, misapplies a nearby rule, or omits facts that were in the retrieved text. Its native citation marks (`【1†…】`) are now rewritten as `[1]`.
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
