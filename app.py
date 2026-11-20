"""Laws of the Game: ask a question about the IFAB Laws, get an answer that cites its passages.

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
import ui                    # noqa: E402

BRAND = "Laws of the Game"
GLYPH = ('<path class="s" d="M8 8.5h10M8 12.5h10M8 16.5h6" stroke-width="2" stroke-linecap="round"/>')  # a page of text
EXAMPLES = [   # (topic, question)
    ("Substitutes", "How many substitutes can a team use?"),
    ("Offside", "Is a player offside when receiving the ball from a throw-in?"),
    ("Handball", "When is handling the ball not an offence?"),
    ("Restarts", "What happens if the ball hits the referee and a goal is scored?"),
]

st.set_page_config(page_title="Laws of the Game · Answers with citations", page_icon=":material/menu_book:")

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


@st.cache_resource(show_spinner="Loading the Laws...")
def warm_up():
    """Load the embedding model and the index once per server process."""
    get_model()
    get_corpus()
    return True


def show_sources(hits, refused, by_gate):
    """The retrieved passages as numbered rows; each opens to the text the model was given."""
    ui.label("Sources")
    if by_gate:
        note = "No passage was close enough to answer from. The nearest ones are listed below."
    elif refused:
        note = "The passages reviewed for this question. None of them answers it."
    else:
        note = "The passages the answer is drawn from. Numbers in the answer refer to this list."
    st.markdown(f'<div class="lead">{note}</div>', unsafe_allow_html=True)
    for number, hit in enumerate(hits, start=1):
        score = f"relevance {hit['grade']}/3" if "grade" in hit else f"distance {hit['distance']:.3f}"
        with st.expander(f"{number:02d} {hit['title']} · {pages(hit)} · {score}"):
            parts = (hit["section"], hit["heading"], hit.get("subheading", ""))
            where = " › ".join(part for part in parts if part)
            st.markdown(f'<div class="mono">{hit["doc_type"]} · {hit["status"]}</div>'
                        + (f'<div class="note" style="margin:.3rem 0 .6rem">{where}</div>' if where else ""),
                        unsafe_allow_html=True)
            st.text(hit.get("context") or hit["text"])   # what the model was given


ui.apply(width="780px")
ui.top_bar(BRAND, GLYPH)
warm_up()

ui.hero("IFAB Laws of the Game 2026/27 · Circulars 32–34 · Trial protocols",
        "The Laws of the Game,", "answered with citations.",
        "Ask about any Law, restart or sanction. Every answer is drawn only from the official IFAB documents "
        "and cites the passages it comes from, with their pages.")

with st.form("ask", border=False):
    question = st.text_input("Your question", key="question", label_visibility="collapsed",
                             placeholder="Ask about a Law, a restart or a sanction")
    submitted = st.form_submit_button("Ask", type="primary")

asked = None
if submitted:
    asked = question.strip()
    if not asked:
        st.info("Type a question first.")
elif "pending" in st.session_state:
    asked = st.session_state.pop("pending")

if not asked:
    ui.label("Examples")
    with st.container(key="rows"):
        for i, (topic, q) in enumerate(EXAMPLES):
            if st.button(f"`{i + 1:02d}` **{topic}** {q}", key=f"row_{i}", width="stretch"):
                st.session_state.pending = q
                st.rerun()
else:
    try:
        with st.spinner("Searching the Laws..."):
            result = answer(asked)
    except RateLimitError:
        st.warning("The language model has reached its usage limit. A per-minute limit clears within a minute; "
                   "a daily limit resets the next day.")
        st.stop()
    except RuntimeError as error:   # e.g. the API key is not set; the message names no secret
        st.error(str(error))
        st.stop()
    except APIError as error:
        st.error(f"The language model service returned an error ({type(error).__name__}). Please try again shortly.")
        st.stop()

    # Refused by the gate (no model call) or by the model itself. An answer that covers
    # part of the question and refuses the rest is shown as an answer.
    refused = result["refused"] or is_refusal(result["answer"])
    ui.label("Answer")
    st.markdown(f'<div class="note" style="margin:-.3rem 0 .5rem">{asked}</div>', unsafe_allow_html=True)
    if refused:
        st.markdown(f"{REFUSAL} The question may fall outside the Laws, or need different wording.")
    else:
        st.markdown(result["answer"] or "No answer was returned.")
        if result["finish_reason"] == "length":
            st.caption("The answer reached its length limit and may be incomplete.")
    show_sources(result["hits"], refused, by_gate=result["refused"])

ui.foot("Source: The IFAB, Laws of the Game 2026/27 · Independent reference, not affiliated with The IFAB")
