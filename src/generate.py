"""Build the prompt from the retrieved chunks and ask the LLM."""
import os
import re
from functools import lru_cache

from openai import OpenAI

from config import (LLM_API_KEY_ENV, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
                    MAX_ANSWER_TOKENS, REFUSAL)
from retrieve import pages

SYSTEM_PROMPT = f"""You are a football referee assistant specializing in the IFAB Laws of the Game.
You help users understand how the Laws apply to specific match incidents, and
you answer direct questions about the Laws.

How the sources are given
- The sources are numbered [1], [2], ... Each header shows the title, document
  type, status, pages and location (Law and section).
- Use only the sources. Do not use outside knowledge. Some sources may be
  irrelevant to the question: ignore them.
- Cite every statement with its source number, like [1], and name the Law and
  section from the header.
- A source marked "status: trial" is a trial protocol, not part of the Laws.
  Say so when you rely on one. Say when a source calls a rule a competition
  option.
- Never invent a rule, citation or official interpretation.

Insufficient evidence
- Direct question: reply "{REFUSAL}"
- Incident: write "Decision: Not determinable from the provided documents."
  and state which facts or rules are missing.

Simple questions
- For a direct question about a rule, answer in one to three sentences with
  citations. Use the full format below only for incidents.

Decision procedure for incidents
1. Identify the material facts provided by the user.
2. Identify the relevant Law and its decision criteria.
3. Distinguish established facts from assumptions.
4. Explain how each material fact affects the application of the Law.
5. Identify any missing facts that could change the outcome.
6. Provide a reasoned conclusion.

Referee judgment
- Distinguish clear rule applications from situations requiring referee
  judgment. Base the discussion of judgment on the Practical guidelines for
  match officials when they appear in the sources.
- Label what the sources say separately from your own reasoning.
- Where multiple interpretations are plausible, explain the strongest argument
  for each, and which additional facts would favour one.
- Do not claim that every ambiguous incident has one certain answer, and do not
  treat uncertainty as proof that an offence occurred or did not.

Response format for incidents
Decision / Applicable Law / Analysis / Referee judgment / What could change
the decision.

Answer in the language of the question. Use clear, accessible language. Be
decisive when the sources support a conclusion and cautious when they do not."""


# gpt-oss sometimes cites in its own format, 【1】, 【1†L4-L9】 or 【1†source】, not [1].
NATIVE_CITATION = re.compile(r"【(\d+)(?:†[^】]*)?】")


def normalize_citations(text):
    """Rewrite the model's native citation marks as [1]."""
    return NATIVE_CITATION.sub(r"[\1]", text)


def is_refusal(text):
    """True when the whole answer is the refusal sentence (perhaps with a citation or
    punctuation around it). An answer that covers part of a question and refuses the rest
    is not a refusal."""
    core = re.sub(r"\[\d+\]", "", text).strip().strip('."*').strip()
    core = re.sub(r"^(?:\**\s*(?:Answer|Decision)\s*:\s*\**\s*)", "", core)   # "**Decision:** ..."
    return core == REFUSAL.rstrip(".")


@lru_cache(maxsize=1)
def get_client():
    """Create the LLM client once. The key comes from an environment variable, never from a file."""
    key = os.environ.get(LLM_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"The environment variable {LLM_API_KEY_ENV} is not set. "
                           f'In your terminal run: export {LLM_API_KEY_ENV}="your-key"')
    return OpenAI(base_url=LLM_BASE_URL, api_key=key)


def format_source(number, hit):
    """One numbered source block: a header line with its origin, then the raw chunk text."""
    section = re.sub(r"\s*\|\s*", " - ", hit["section"])
    location = " > ".join(part for part in (section, hit["heading"]) if part)
    header = (f"[{number}] {hit['title']} | {hit['doc_type']}, status: {hit['status']} "
              f"| {pages(hit)}")
    if location:
        header += f" | {location}"
    return f"{header}\n{hit['text']}"


def build_messages(question, hits):
    """The chat messages: the fixed instructions, then the sources and the question."""
    sources = "\n\n".join(format_source(i, hit) for i, hit in enumerate(hits, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"},
    ]


def generate(question, hits):
    """Ask the LLM to answer from the hits. Returns the answer and the token usage."""
    response = get_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_completion_tokens=MAX_ANSWER_TOKENS,
        messages=build_messages(question, hits),
    )
    choice = response.choices[0]
    usage = response.usage
    return {
        "answer": normalize_citations((choice.message.content or "").strip()),
        "finish_reason": choice.finish_reason,   # "length" means the token limit cut the answer
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
