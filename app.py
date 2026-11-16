"""Streamlit demo: ask a question about the Laws of the Game, get a cited answer.

Run locally from the project folder:  streamlit run app.py
"""
import os
import random
import sys
from datetime import date
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

# -------------------------
# Christmas theme: blinking lights along the top and gentle snowfall, on its own from
# 1 December to 6 January. Clicks pass through; reduced-motion settings are respected.
# -------------------------

LIGHT_COLOURS = ["#e53935", "#fdd835", "#43a047", "#1e88e5", "#fb8c00"]


def is_festive(today=None):
    """True from 1 December to 6 January."""
    today = today or date.today()
    return today.month == 12 or (today.month == 1 and today.day <= 6)


def _lights_html(n_bulbs=40):
    bulbs = "".join(
        f'<span class="xmas-bulb" style="background:{c};color:{c};animation-delay:{(i % 5) * 0.3:.1f}s"></span>'
        for i, c in enumerate(LIGHT_COLOURS[i % len(LIGHT_COLOURS)] for i in range(n_bulbs))
    )
    return f'<div class="xmas-lights">{bulbs}</div>'


def _snow_html(n_flakes=25):
    rng = random.Random(7)  # fixed seed: flakes keep their places when Streamlit reruns
    flakes = []
    for _ in range(n_flakes):
        left = rng.uniform(0, 100)
        size = rng.uniform(0.6, 1.3)
        duration = rng.uniform(9, 18)
        delay = -rng.uniform(0, 18)  # negative delay: snow is already falling on page load
        opacity = rng.uniform(0.4, 0.9)
        flakes.append(
            f'<span class="xmas-flake" style="left:{left:.1f}vw;font-size:{size:.2f}rem;'
            f'opacity:{opacity:.2f};animation-duration:{duration:.1f}s,{duration / 3:.1f}s;'
            f'animation-delay:{delay:.1f}s,{delay:.1f}s">❄</span>'
        )
    return f'<div class="xmas-snow">{"".join(flakes)}</div>'


_CSS = """
<style>
/* lights along the top edge, above Streamlit's header; clicks pass through */
.xmas-lights {
    position: fixed; top: 0; left: 0; width: 100%; height: 22px;
    display: flex; justify-content: space-around; align-items: flex-start;
    border-top: 2px solid #2e3b2e;
    pointer-events: none; z-index: 1000001;
}
.xmas-bulb {
    width: 9px; height: 13px; margin-top: 3px;
    border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
    box-shadow: 0 2px 10px 2px currentColor;
    animation: xmas-blink 1.5s ease-in-out infinite alternate;
}
@keyframes xmas-blink {
    from { opacity: 1; }
    to   { opacity: 0.25; box-shadow: none; }
}

/* falling snow over the whole page; clicks pass through */
.xmas-snow {
    position: fixed; inset: 0; overflow: hidden;
    pointer-events: none; z-index: 1000000;
}
.xmas-flake {
    position: absolute; top: -2rem;
    color: #b3d4f5; text-shadow: 0 0 3px rgba(255, 255, 255, 0.8);
    animation-name: xmas-fall, xmas-sway;
    animation-timing-function: linear, ease-in-out;
    animation-iteration-count: infinite, infinite;
    animation-direction: normal, alternate;
}
@keyframes xmas-fall {
    to { top: 105vh; }
}
@keyframes xmas-sway {
    from { transform: translateX(-15px); }
    to   { transform: translateX(15px); }
}

/* people who turned off animations in their system settings get none */
@media (prefers-reduced-motion: reduce) {
    .xmas-snow { display: none; }
    .xmas-bulb { animation: none; }
}
</style>
"""


def add_christmas_theme(force=False, snow=True, lights=True):
    """Add the theme if it is the festive season (or force=True)."""
    if not (force or is_festive()):
        return
    html = _CSS
    if lights:
        html += _lights_html()
    if snow:
        html += _snow_html()
    st.markdown(html, unsafe_allow_html=True)


add_christmas_theme()


@st.cache_resource(show_spinner="Loading the embedding model and the index...")
def warm_up():
    """Load the embedding model and the index once per server process."""
    get_model()
    get_corpus()
    return True


def show_sources(hits, refused, by_gate):
    """List the retrieved passages, each in a collapsible block."""
    st.subheader("Sources")
    if by_gate:
        st.caption("The closest passages found. They were judged too distant to answer from.")
    elif refused:
        st.caption("The passages the model was given. It found no answer to the question in them.")
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
        show_sources(result["hits"], refused, by_gate=result["refused"])
