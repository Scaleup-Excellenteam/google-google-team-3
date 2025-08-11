import unittest
from ac_engine import (
    AutoCompleteEngine,
    _single_edit_descriptor,
    _score_by_rules,
    _best_prefix_score,
    normalize,
    trigrams,
    AutoCompleteData
)


class TestAcEngineHelpers(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("Hello, World!"), "hello world")
        self.assertEqual(normalize("  Multiple   Spaces "), "multiple spaces")
        self.assertEqual(normalize("No$Special#Chars"), "nospecialchars")

    def test_trigrams(self):
        self.assertEqual(list(trigrams("abc")), ["abc"])
        self.assertEqual(list(trigrams("abcd")), ["abc", "bcd"])
        self.assertEqual(list(trigrams("hi")), ["hi"])
        self.assertEqual(list(trigrams("")), [])

    def test_single_edit_descriptor(self):
        self.assertEqual(_single_edit_descriptor("cat", "cat"), (None, None))         # identical
        self.assertEqual(_single_edit_descriptor("cat", "cot"), ("replace", 1))       # replace
        self.assertEqual(_single_edit_descriptor("cat", "cart"), ("insert", 2))       # insert
        self.assertEqual(_single_edit_descriptor("cart", "cat"), ("delete", 2))       # delete
        self.assertEqual(_single_edit_descriptor("cat", "dog"), (None, None))         # >1 edit
        self.assertEqual(_single_edit_descriptor("abc", "abcdef"), (None, None))      # length >1 diff

    def test_score_by_rules(self):
        self.assertEqual(_score_by_rules("cat", "cat"), 6)  # exact
        self.assertIsNone(_score_by_rules("cat", "dog"))    # too different
        # Replace
        score_replace = _score_by_rules("cat", "cot")
        self.assertIsInstance(score_replace, int)
        # Insert
        score_insert = _score_by_rules("cat", "coat")
        self.assertIsInstance(score_insert, int)
        # Delete
        score_delete = _score_by_rules("coat", "cat")
        self.assertIsInstance(score_delete, int)

    def test_best_prefix_score(self):
        off, sc = _best_prefix_score("hello world", "hello")
        self.assertEqual(off, 0)
        self.assertEqual(sc, 10)  # exact match
        # with one typo
        off2, sc2 = _best_prefix_score("hello world", "hellp")
        self.assertIsNotNone(off2)
        self.assertIsNotNone(sc2)
        # no match
        off3, sc3 = _best_prefix_score("abc", "xyz")
        self.assertIsNone(off3)
        self.assertIsNone(sc3)


class TestAutoCompleteEngine(unittest.TestCase):
    def setUp(self):

        self.engine = AutoCompleteEngine()
        data = [
            {"sentence": "Hello world", "source": "src1"},
            {"sentence": "Hello there", "source": "src2"},
            {"sentence": "Help me please", "source": "src3"},
            {"sentence": "Completely different", "source": "src4"},
        ]
        self.engine.entries.clear()
        self.engine.tri2ids.clear()
        for idx, entry in enumerate(data):
            norm = normalize(entry["sentence"])
            self.engine.entries.append({
                "raw": entry["sentence"],
                "norm": norm,
                "source": entry["source"]
            })
            for tg in set(trigrams(norm)):
                self.engine.tri2ids[tg].add(idx)

    def test_get_exact_candidates(self):
        qnorm = normalize("hello")
        cands = self.engine._get_exact_candidates(qnorm)
        self.assertTrue(all(isinstance(c, int) for c in cands))
        self.assertIn(0, cands)
        self.assertIn(1, cands)

    def test_get_distanced_candidates(self):
        qnorm = normalize("hellp")  # typo
        cands = self.engine._get_distanced_candidates(qnorm)
        self.assertTrue(len(cands) > 0)

    def test_get_best_k_completions_exact(self):
        res = self.engine.get_best_k_completions("hello", allow_one_typo=False)
        self.assertTrue(any(isinstance(r, AutoCompleteData) for r in res))
        self.assertTrue(all("hello" in r.completed_sentence.lower() for r in res))

    def test_get_best_k_completions_with_typo(self):
        res = self.engine.get_best_k_completions("hellp", allow_one_typo=True)
        self.assertTrue(any(r.completed_sentence.lower().startswith("hello") for r in res))

    def test_get_best_k_completions_empty(self):
        res = self.engine.get_best_k_completions("zzzzzz", allow_one_typo=False)
        self.assertEqual(res, [])


if __name__ == "__main__":
    unittest.main()
