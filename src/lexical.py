"""Keyword search: BM25, fan-term expansion, spelling correction and rank fusion.

Plain Python and numpy, no models, so it loads instantly and is easy to test.
"""
import difflib
import math
import re
from collections import Counter

import numpy as np

from config import RRF_K

# Words people use that the Laws do not. Each phrase found in a question adds the Laws'
# own terms to the keyword query, so BM25 can find a passage the question shares no words
# with ("dives" -> "simulation"). Kept to terms the Laws deliberately avoid.
FAN_TERMS = {
    "dive": "simulation feigns deceive",
    "dives": "simulation feigns deceive",
    "diving": "simulation feigns deceive",
    "booked": "caution cautioned",
    "booking": "caution cautioned",
    "sent off": "sending-off sent",
    "time-wasting": "delaying restart",
    "time wasting": "delaying restart",
    "keeper": "goalkeeper",
    "ref": "referee",
    "linesman": "assistant referee",
    "linesmen": "assistant referees",
    "back-pass": "deliberately kicked goalkeeper team-mate hands",
    "backpass": "deliberately kicked goalkeeper team-mate hands",
    "professional foul": "denying obvious goal-scoring opportunity",
    "shootout": "penalties penalty shoot-out kicks penalty mark",
    "shoot-out": "penalties penalty shoot-out kicks penalty mark",
}

STOPWORDS = set("""a an and are as at be been but by can could did do does for from had has have
how i if in into is it its may might must not of on or our shall should so than that the their
them then there these they this those to was we were what when where which who whom why will
with would you your""".split())

WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def words(text):
    """Lowercase words of a text, hyphenated words kept whole."""
    return WORD.findall(text.lower().replace("’", "'"))


def stem(word):
    """A deliberately light stemmer: plural and common verb endings only."""
    for suffix, keep in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ed", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4 and not word.endswith("ss"):
            return word[: -len(suffix)] + keep
    return word


def terms(text):
    """BM25 terms: words without stopwords, lightly stemmed."""
    return [stem(w) for w in words(text) if w not in STOPWORDS]


class BM25:
    """Okapi BM25 over a small fixed corpus."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(terms(d)) for d in docs]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_terms):
        """One score per document."""
        result = np.zeros(len(self.docs))
        for i, (doc, length) in enumerate(zip(self.docs, self.lengths)):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            for t in set(query_terms):
                if t in doc:
                    tf = doc[t]
                    result[i] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return result


def correct_spelling(question, vocabulary):
    """Replace words that are not in the corpus with a close corpus word, if there is one.

    Returns (corrected question, {wrong: right}). Short words and words with no close
    match are left alone, so an off-topic question stays off-topic."""
    fixes = {}
    for word in set(words(question)):
        if len(word) < 5 or word in vocabulary or word in STOPWORDS:
            continue
        match = difflib.get_close_matches(word, vocabulary, n=1, cutoff=0.85)
        if match:
            fixes[word] = match[0]
    corrected = question
    for wrong, right in fixes.items():
        corrected = re.sub(rf"\b{re.escape(wrong)}\b", right, corrected, flags=re.IGNORECASE)
    return corrected, fixes


def expand(question):
    """The keyword query: the question plus the Laws' terms for any fan terms in it."""
    lowered = question.lower()
    extra = [laws for fan, laws in FAN_TERMS.items() if re.search(rf"\b{re.escape(fan)}\b", lowered)]
    return " ".join([question, *extra])


def rrf(*rankings):
    """Reciprocal rank fusion of several best-first lists of indices."""
    fused = Counter()
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[index] += 1 / (RRF_K + rank)
    return [index for index, _ in fused.most_common()]
