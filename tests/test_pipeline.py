"""Unit tests for the pure parts of the pipeline. No models are loaded and no API is called.

Run from the project folder:  python -m unittest discover tests
(Importing generate, rag and eval loads torch, which can take a minute on a slow machine.)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chunk import MAX_CHARS, is_heading_line, split_block   # noqa: E402
from clean import clean_text   # noqa: E402
from config import MIN_RERANK_SCORE, REFUSAL   # noqa: E402
from eval import hit_rank, parse_expected, results_path   # noqa: E402
from generate import is_refusal, normalize_citations   # noqa: E402
from lexical import BM25, correct_spelling, expand, rrf, stem, terms   # noqa: E402
from rag import gate_refuses   # noqa: E402


class Clean(unittest.TestCase):
    def test_strips_footer_and_page_number(self):
        raw = "12\nLaw 1 text here.\n12Laws of the Game 2026/27 | Law 1 | The Field of Play"
        text, section = clean_text(raw, "laws", 12)
        self.assertEqual(text, "Law 1 text here.")
        self.assertEqual(section, "Law 1 | The Field of Play")

    def test_keeps_a_number_that_is_not_the_page_number(self):
        text, _ = clean_text("Intro line\n7\nmore", "laws", 30)
        self.assertIn("7", text.splitlines())


class Chunk(unittest.TestCase):
    def test_split_block_respects_the_cap(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(300))
        pieces = split_block(text)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p) <= MAX_CHARS for p in pieces))
        self.assertEqual(" ".join(pieces).split(), text.split())   # nothing lost

    def test_heading_detection(self):
        self.assertTrue(is_heading_line("4. Disciplinary action", "laws"))
        self.assertFalse(is_heading_line("4. Disciplinary action", "circular"))
        self.assertFalse(is_heading_line("4. A sentence that ends with a full stop.", "laws"))


class Lexical(unittest.TestCase):
    def test_stem_and_terms(self):
        self.assertEqual(stem("substitutes"), "substitut")
        self.assertEqual(stem("penalties"), "penalty")
        self.assertEqual(stem("pass"), "pass")
        self.assertEqual(terms("What is the penalty?"), ["penalty"])

    def test_bm25_prefers_the_matching_document(self):
        bm25 = BM25(["the referee drops the ball", "the goalkeeper holds the ball",
                     "substitutes may be used"])
        scores = bm25.scores(terms("goalkeeper"))
        self.assertEqual(scores.argmax(), 1)
        self.assertEqual(scores[2], 0)

    def test_expand_adds_laws_terms_for_fan_terms(self):
        self.assertIn("simulation", expand("Is it a foul if he dives?"))
        self.assertNotIn("goalkeeper", expand("Can a goalkeeper score?").split("?", 1)[1])

    def test_spelling_correction(self):
        vocabulary = {"substitutes", "allowed", "referee"}
        corrected, fixes = correct_spelling("How many substitues are allowed?", vocabulary)
        self.assertEqual(corrected, "How many substitutes are allowed?")
        self.assertEqual(fixes, {"substitues": "substitutes"})
        # A word with no close corpus word is left alone.
        self.assertEqual(correct_spelling("futsal rules", vocabulary)[0], "futsal rules")

    def test_rrf_rewards_agreement(self):
        self.assertEqual(rrf([1, 2, 3], [3, 4, 1])[:2], [1, 3])


class Generate(unittest.TestCase):
    def test_normalize_citations(self):
        self.assertEqual(normalize_citations("Seven players 【1†L4-L9】 and 【2】."),
                         "Seven players [1] and [2].")

    def test_is_refusal(self):
        self.assertTrue(is_refusal(REFUSAL))
        self.assertTrue(is_refusal(f"**{REFUSAL}** [2]"))
        self.assertTrue(is_refusal(f"**Decision:** {REFUSAL}"))
        self.assertFalse(is_refusal(f"A team has eleven players [1]. {REFUSAL}"))


class Gate(unittest.TestCase):
    def test_rerank_score_gate(self):
        self.assertTrue(gate_refuses([{"distance": 0.1, "rerank_score": MIN_RERANK_SCORE - 1}]))
        self.assertFalse(gate_refuses([{"distance": 0.9, "rerank_score": MIN_RERANK_SCORE + 1}]))

    def test_distance_gate_uses_the_closest_hit(self):
        self.assertFalse(gate_refuses([{"distance": 0.5}, {"distance": 0.1}]))
        self.assertTrue(gate_refuses([{"distance": 0.5}, {"distance": 0.6}]))
        self.assertTrue(gate_refuses([]))


class Eval(unittest.TestCase):
    def test_parse_expected_several_files(self):
        row = {"id": "Q", "expected_files": "a.pdf; b.pdf", "expected_pages": "1-2; 3, 5-6"}
        self.assertEqual(parse_expected(row), {"a.pdf": [(1, 2)], "b.pdf": [(3, 3), (5, 6)]})

    def test_hit_rank(self):
        expected = {"a.pdf": [(10, 12)]}
        hits = [{"filename": "a.pdf", "page_start": 1, "page_end": 2},
                {"filename": "a.pdf", "page_start": 12, "page_end": 13}]
        self.assertEqual(hit_rank(hits, expected), 2)
        self.assertEqual(hit_rank(hits[:1], expected), "")

    def test_results_path(self):
        main, draft = Path("data/eval/questions.csv"), Path("data/eval/questions_new_draft.csv")
        self.assertEqual(results_path(main, False, "rerank").name, "results.csv")
        self.assertEqual(results_path(main, True, "dense").name, "results_retrieval_dense.csv")
        self.assertEqual(results_path(draft, True, "rerank").name,
                         "results_new_draft_retrieval_rerank.csv")


if __name__ == "__main__":
    unittest.main()
