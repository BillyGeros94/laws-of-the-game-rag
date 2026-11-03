"""Streamlit demo: ask a question about the Laws of the Game, get a cited answer.

Run locally from the project folder:  streamlit run app.py
"""
import os
import sys
from pathlib import Path

import streamlit as st
from openai import APIError, RateLimitError

# The pipeline modules live in src/ and use paths relative to the project folder
# (index/, data/), so make both work wherever the app is started from.
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from config import REFUSAL   # noqa: E402  (needs the path set above)
from generate import is_refusal   # noqa: E402
from rag import answer       # noqa: E402
from retrieve import get_corpus, get_model, pages   # noqa: E402

st.set_page_config(page_title="Laws of the Game Q&A", page_icon="⚽")


@st.cache_resource(show_spinner="Loading the embedding model and the index...")
def warm_up():
    """Load the embedding model and the index once per server process."""
    get_model()
    get_corpus()
    return True


def show_sources(hits, refused):
    """List the retrieved passages, each in a collapsible block."""
    st.subheader("Sources")
    if refused:
        st.caption("The closest passages found. They were judged too distant to answer from.")
    else:
        st.caption("The passages the answer was based on. The answer may cite only some of them.")
    for number, hit in enumerate(hits, start=1):
        score = f"relevance {hit['grade']}/3" if "grade" in hit else f"distance {hit['distance']:.3f}"
        label = f"[{number}] {hit['title']} | {pages(hit)} | {score}"
        with st.expander(label):
            parts = (hit["section"], hit["heading"], hit.get("subheading", ""))
            where = " > ".join(part for part in parts if part)
            st.caption(f"{hit['doc_type']}, status: {hit['status']}" + (f" | {where}" if where else ""))
            st.text(hit.get("context") or hit["text"])   # what the model was given


st.title("Laws of the Game Q&A")
st.write(
    "Ask a question about the IFAB Laws of the Game 2026/27. Answers come only from the "
    "Laws, IFAB Circulars 32, 33 and 34 and two trial protocols, with the passages shown below. "
    "This is a portfolio project, not an official source, and it can be wrong: check the cited pages."
)

warm_up()

with st.form("ask"):
    question = st.text_input("Your question", placeholder="e.g. How many substitutes are allowed?")
    submitted = st.form_submit_button("Ask")

if submitted:
    question = question.strip()
    if not question:
        st.info("Type a question first.")
    else:
        try:
            with st.spinner("Searching the Laws and writing the answer..."):
                result = answer(question)
        except RateLimitError:
            st.warning("The free model service is busy: its rate limit has been reached. "
                       "A per-minute limit clears within a minute; a daily limit clears the next day. "
                       "Please try again later.")
            st.stop()
        except RuntimeError as error:   # e.g. the API key is not set; the message names no secret
            st.error(str(error))
            st.stop()
        except APIError as error:
            st.error(f"The model service returned an error ({type(error).__name__}). Please try again later.")
            st.stop()

        # Refused by the gate (no model call) or by the model itself. An answer that covers
        # part of the question and refuses the rest is shown as an answer.
        refused = result["refused"] or is_refusal(result["answer"])
        if refused:
            st.info(REFUSAL)
        else:
            st.markdown(result["answer"] or "(empty answer)")
            if result["finish_reason"] == "length":
                st.warning("The answer hit the length limit and may be cut off.")
        show_sources(result["hits"], refused)
