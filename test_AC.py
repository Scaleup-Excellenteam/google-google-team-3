# test_AC.py
import unittest
import os
import sys
import time
from ac_engine import load_engine, AutoCompleteEngine, AutoCompleteData, normalize, trigrams, _score_by_rules, _best_prefix_score


class TestAutoCompleteEngineWithRealData(unittest.TestCase):
    """Test autocomplete engine using real data via load_engine()."""

    @classmethod
    def setUpClass(cls):
        """Load the real engine once for all tests."""
        try:
            cls.engine = load_engine()
            print(f"Loaded engine with {len(cls.engine.entries)} entries")
        except FileNotFoundError as e:
            raise unittest.SkipTest(f"Cannot run tests: {e}")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load engine: {e}")

    def _timed_search(self, query, allow_one_typo=False, topn=5, description=""):
        """Helper method to time search operations."""
        start_time = time.perf_counter()
        results = self.engine.get_best_k_completions(query, allow_one_typo=allow_one_typo, topn=topn)
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        typo_str = "fuzzy" if allow_one_typo else "exact"
        print(f"⏱️  Query '{query}' ({typo_str}): {duration_ms:.2f}ms -> {len(results)} results {description}")

        return results, duration_ms

    def test_engine_loaded_correctly(self):
        """Test that engine loaded with real data."""
        self.assertIsInstance(self.engine, AutoCompleteEngine)
        self.assertGreater(len(self.engine.entries), 0)
        self.assertGreater(len(self.engine.tri2ids), 0)

    def test_basic_exact_search(self):
        """Test basic exact substring matching with real data."""
        print("\n=== EXACT SEARCH TIMING ===")

        # Try a simple common word that should exist
        results, duration = self._timed_search("the", allow_one_typo=False, topn=5, description="(common word)")

        # Should find some results
        self.assertGreater(len(results), 0, "Should find entries containing 'the'")

        # All results should contain 'the'
        for result in results:
            self.assertIn("the", result.completed_sentence.lower())
            self.assertIsInstance(result.score, int)
            self.assertGreater(result.score, 0)

    def test_fuzzy_search_basic(self):
        """Test fuzzy search with a known typo."""
        print("\n=== FUZZY SEARCH TIMING ===")

        # First find what exact matches exist for "and"
        exact_results, exact_duration = self._timed_search("and", allow_one_typo=False, topn=3, description="(baseline)")

        if len(exact_results) > 0:
            # Now try "abd" (substitution typo)
            fuzzy_results, fuzzy_duration = self._timed_search("abd", allow_one_typo=True, topn=10, description="(substitution typo)")

            # Should find some results
            self.assertGreater(len(fuzzy_results), 0, "Should find fuzzy matches for 'abd'")

            # Check if any contain "and"
            and_found = any("and" in result.completed_sentence.lower() for result in fuzzy_results)
            if and_found:
                print(f"✓ Successfully found 'and' entries when searching for 'abd'")

            # Compare timing
            slowdown = fuzzy_duration / exact_duration if exact_duration > 0 else 0
            print(f"📊 Fuzzy search was {slowdown:.1f}x slower than exact search")

    def test_empty_query(self):
        """Test empty query handling."""
        print("\n=== EMPTY QUERY TIMING ===")
        results, duration = self._timed_search("", allow_one_typo=True, description="(empty query)")
        self.assertEqual(len(results), 0)

    def test_no_matches_query(self):
        """Test query that should return no matches."""
        print("\n=== NO MATCHES TIMING ===")

        # Use a very unlikely combination
        results, duration = self._timed_search("xyzzqwertpoiuy", allow_one_typo=False, description="(no matches)")
        self.assertEqual(len(results), 0)

        results, duration = self._timed_search("xyzzqwertpoiuy", allow_one_typo=True, description="(no matches, fuzzy)")
        self.assertEqual(len(results), 0)

    def test_topn_parameter(self):
        """Test that topn parameter limits results correctly."""
        print("\n=== TOPN PARAMETER TIMING ===")

        # Find a query that returns multiple results
        results_unlimited, duration_100 = self._timed_search("a", allow_one_typo=False, topn=100, description="(topn=100)")

        if len(results_unlimited) > 3:
            results_limited, duration_3 = self._timed_search("a", allow_one_typo=False, topn=3, description="(topn=3)")
            self.assertEqual(len(results_limited), 3)

            # Compare timing
            if duration_100 > 0:
                speedup = duration_100 / duration_3
                print(f"📊 Limiting topn improved speed by {speedup:.1f}x")

    def test_score_ordering(self):
        """Test that results are ordered by score (highest first)."""
        print("\n=== SCORE ORDERING TIMING ===")

        results, duration = self._timed_search("test", allow_one_typo=True, topn=5, description="(score ordering)")

        if len(results) > 1:
            # Scores should be in descending order
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i].score, results[i + 1].score,
                                        f"Score at index {i} ({results[i].score}) should be >= score at index {i + 1} ({results[i + 1].score})")

    def test_data_structure_integrity(self):
        """Test that all results have required fields."""
        results, duration = self._timed_search("test", allow_one_typo=True, topn=5, description="(data integrity)")

        for result in results:
            self.assertIsInstance(result, AutoCompleteData)
            self.assertIsInstance(result.completed_sentence, str)
            self.assertIsInstance(result.source, str)
            self.assertIsInstance(result.score, int)
            self.assertGreater(len(result.completed_sentence), 0)
            self.assertGreater(len(result.source), 0)

    def test_case_insensitive_search(self):
        """Test that search is case insensitive."""
        print("\n=== CASE SENSITIVITY TIMING ===")

        results_lower, duration_lower = self._timed_search("the", allow_one_typo=False, topn=3, description="(lowercase)")
        results_upper, duration_upper = self._timed_search("THE", allow_one_typo=False, topn=3, description="(uppercase)")
        results_mixed, duration_mixed = self._timed_search("The", allow_one_typo=False, topn=3, description="(mixed case)")

        # Should return similar results (allowing for scoring differences)
        if len(results_lower) > 0:
            self.assertGreater(len(results_upper), 0)
            self.assertGreater(len(results_mixed), 0)

        avg_duration = (duration_lower + duration_upper + duration_mixed) / 3
        print(f"📊 Average case variation timing: {avg_duration:.2f}ms")

    def test_specific_real_data_queries(self):
        """Test with queries that are likely to exist in real data."""
        print("\n=== REAL DATA QUERIES TIMING ===")

        common_words = ["data", "file", "system", "user", "time", "value"]
        timings = []

        for word in common_words:
            with self.subTest(word=word):
                results, duration = self._timed_search(word, allow_one_typo=False, topn=3, description=f"(common word: {word})")
                timings.append(duration)

                if len(results) > 0:
                    # Verify the word appears in results
                    found = any(word.lower() in result.completed_sentence.lower() for result in results)
                    self.assertTrue(found, f"Word '{word}' should appear in at least one result")

        if timings:
            avg_timing = sum(timings) / len(timings)
            min_timing = min(timings)
            max_timing = max(timings)
            print(f"📊 Common words timing - Avg: {avg_timing:.2f}ms, Min: {min_timing:.2f}ms, Max: {max_timing:.2f}ms")

    def test_query_length_performance(self):
        """Test how query length affects performance."""
        print("\n=== QUERY LENGTH PERFORMANCE ===")

        test_queries = [
            ("a", "1 char"),
            ("th", "2 chars"),
            ("the", "3 chars"),
            ("test", "4 chars"),
            ("system", "6 chars"),
            ("programming", "11 chars")
        ]

        for query, desc in test_queries:
            # Test both exact and fuzzy
            exact_results, exact_duration = self._timed_search(query, allow_one_typo=False, topn=5, description=f"exact, {desc}")
            fuzzy_results, fuzzy_duration = self._timed_search(query, allow_one_typo=True, topn=5, description=f"fuzzy, {desc}")

            if exact_duration > 0:
                slowdown = fuzzy_duration / exact_duration
                print(f"  └─ Fuzzy slowdown for '{query}': {slowdown:.1f}x")


class TestSpecificTypoScenarios(unittest.TestCase):
    """Test specific typo scenarios with real data."""

    @classmethod
    def setUpClass(cls):
        """Load the real engine once for all tests."""
        try:
            # Use the same engine instance from the first test class
            if hasattr(TestAutoCompleteEngineWithRealData, 'engine'):
                cls.engine = TestAutoCompleteEngineWithRealData.engine
            else:
                cls.engine = load_engine()
        except FileNotFoundError as e:
            raise unittest.SkipTest(f"Cannot run tests: {e}")

    def _timed_search(self, query, allow_one_typo=False, topn=5, description=""):
        """Helper method to time search operations."""
        start_time = time.perf_counter()
        results = self.engine.get_best_k_completions(query, allow_one_typo=allow_one_typo, topn=topn)
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        typo_str = "fuzzy" if allow_one_typo else "exact"
        print(f"⏱️  Query '{query}' ({typo_str}): {duration_ms:.2f}ms -> {len(results)} results {description}")

        return results, duration_ms

    def test_substitution_typos(self):
        """Test various substitution typos."""
        print("\n=== SUBSTITUTION TYPOS TIMING ===")

        test_cases = [
            ("tast", "test"),  # a->e
            ("deta", "data"),  # e->a
            ("fila", "file"),  # e->a
        ]

        for typo, correct in test_cases:
            with self.subTest(typo=typo, correct=correct):
                # First check if correct word has matches
                correct_results, correct_duration = self._timed_search(correct, allow_one_typo=False, topn=3, description=f"(baseline for {correct})")

                if len(correct_results) > 0:
                    # Try the typo
                    typo_results, typo_duration = self._timed_search(typo, allow_one_typo=True, topn=10, description=f"(substitution: {typo}->{correct})")

                    # Check if we find the correct word
                    found_correct = any(correct in result.completed_sentence.lower() for result in typo_results)
                    status = "✓" if found_correct else "✗"
                    print(f"  └─ {status} Found '{correct}' when searching '{typo}'")

                    if correct_duration > 0:
                        slowdown = typo_duration / correct_duration
                        print(f"  └─ Typo search was {slowdown:.1f}x slower")

    def test_insertion_typos(self):
        """Test insertion typos (extra character)."""
        print("\n=== INSERTION TYPOS TIMING ===")

        test_cases = [
            ("fiile", "file"),  # extra 'i'
            ("tesst", "test"),  # extra 's'
            ("ddata", "data"),  # extra 'd'
        ]

        for typo, correct in test_cases:
            with self.subTest(typo=typo, correct=correct):
                correct_results, correct_duration = self._timed_search(correct, allow_one_typo=False, topn=3, description=f"(baseline for {correct})")

                if len(correct_results) > 0:
                    typo_results, typo_duration = self._timed_search(typo, allow_one_typo=True, topn=10, description=f"(insertion: {typo}->{correct})")
                    found_correct = any(correct in result.completed_sentence.lower() for result in typo_results)
                    status = "✓" if found_correct else "✗"
                    print(f"  └─ {status} Found '{correct}' when searching '{typo}'")

                    if correct_duration > 0:
                        slowdown = typo_duration / correct_duration
                        print(f"  └─ Insertion typo search was {slowdown:.1f}x slower")

    def test_deletion_typos(self):
        """Test deletion typos (missing character)."""
        print("\n=== DELETION TYPOS TIMING ===")

        test_cases = [
            ("fil", "file"),  # missing 'e'
            ("tes", "test"),  # missing 't'
            ("dat", "data"),  # missing 'a'
        ]

        for typo, correct in test_cases:
            with self.subTest(typo=typo, correct=correct):
                correct_results, correct_duration = self._timed_search(correct, allow_one_typo=False, topn=3, description=f"(baseline for {correct})")

                if len(correct_results) > 0:
                    typo_results, typo_duration = self._timed_search(typo, allow_one_typo=True, topn=10, description=f"(deletion: {typo}->{correct})")
                    found_correct = any(correct in result.completed_sentence.lower() for result in typo_results)
                    status = "✓" if found_correct else "✗"
                    print(f"  └─ {status} Found '{correct}' when searching '{typo}'")

                    if correct_duration > 0:
                        slowdown = typo_duration / correct_duration
                        print(f"  └─ Deletion typo search was {slowdown:.1f}x slower")

    def test_performance_summary(self):
        """Generate a performance summary."""
        print("\n=== PERFORMANCE SUMMARY ===")

        # Test various scenarios and collect timing data
        scenarios = [
            ("short exact", "a", False),
            ("short fuzzy", "a", True),
            ("medium exact", "test", False),
            ("medium fuzzy", "test", True),
            ("long exact", "programming", False),
            ("long fuzzy", "programming", True)
        ]

        timings = {}
        for scenario_name, query, fuzzy in scenarios:
            results, duration = self._timed_search(query, allow_one_typo=fuzzy, topn=10, description=f"({scenario_name})")
            timings[scenario_name] = duration

        print("\n📊 PERFORMANCE ANALYSIS:")
        print(f"Fastest: {min(timings, key=timings.get)} ({timings[min(timings, key=timings.get)]:.2f}ms)")
        print(f"Slowest: {max(timings, key=timings.get)} ({timings[max(timings, key=timings.get)]:.2f}ms)")

        # Compare exact vs fuzzy for same queries
        if "short exact" in timings and "short fuzzy" in timings:
            ratio = timings["short fuzzy"] / timings["short exact"] if timings["short exact"] > 0 else 0
            print(f"Short queries: Fuzzy is {ratio:.1f}x slower than exact")

        if "medium exact" in timings and "medium fuzzy" in timings:
            ratio = timings["medium fuzzy"] / timings["medium exact"] if timings["medium exact"] > 0 else 0
            print(f"Medium queries: Fuzzy is {ratio:.1f}x slower than exact")

    def test_queries_with_numbers(self):
        """Test queries containing numbers in various positions and lengths."""
        print("\n=== QUERIES WITH NUMBERS TIMING ===")

        test_queries = [
            # Numbers at beginning
            ("1", "number at start, 1 char"),
            ("2023", "number at start, 4 chars"),
            ("123abc", "number at start, 6 chars"),

            # Numbers at end
            ("abc1", "number at end, 4 chars"),
            ("test2023", "number at end, 8 chars"),
            ("programming42", "number at end, 13 chars"),

            # Numbers in middle
            ("abc123def", "number in middle, 9 chars"),
            ("test1data", "number in middle, 9 chars"),
            ("file2system", "number in middle, 11 chars"),

            # Multiple numbers
            ("123abc456", "multiple numbers, 9 chars"),
            ("v1.2.3", "version format, 6 chars"),
            ("user123test456", "multiple numbers, 14 chars"),

            # Mixed with common words
            ("the1", "common word + number, 4 chars"),
            ("data2023", "common word + number, 8 chars"),
            ("file123", "common word + number, 7 chars")
        ]

        timings = []
        exact_timings = []
        fuzzy_timings = []

        for query, description in test_queries:
            with self.subTest(query=query):
                # Test exact search
                exact_results, exact_duration = self._timed_search(
                    query,
                    allow_one_typo=False,
                    topn=5,
                    description=f"exact, {description}"
                )
                exact_timings.append(exact_duration)

                # Test fuzzy search
                fuzzy_results, fuzzy_duration = self._timed_search(
                    query,
                    allow_one_typo=True,
                    topn=5,
                    description=f"fuzzy, {description}"
                )
                fuzzy_timings.append(fuzzy_duration)

                # Calculate slowdown ratio
                if exact_duration > 0:
                    slowdown = fuzzy_duration / exact_duration
                    print(f"  └─ Fuzzy slowdown for '{query}': {slowdown:.1f}x")

                # Verify results structure
                for result in exact_results + fuzzy_results:
                    self.assertIsInstance(result, AutoCompleteData)
                    self.assertIsInstance(result.completed_sentence, str)
                    self.assertIsInstance(result.score, int)

        # Performance analysis
        if exact_timings and fuzzy_timings:
            avg_exact = sum(exact_timings) / len(exact_timings)
            avg_fuzzy = sum(fuzzy_timings) / len(fuzzy_timings)

            print(f"\n📊 Numbers Query Performance:")
            print(f"Average exact search: {avg_exact:.2f}ms")
            print(f"Average fuzzy search: {avg_fuzzy:.2f}ms")
            print(f"Overall fuzzy slowdown: {avg_fuzzy / avg_exact:.1f}x")

            # Find fastest and slowest
            fastest_exact = min(exact_timings)
            slowest_exact = max(exact_timings)
            fastest_fuzzy = min(fuzzy_timings)
            slowest_fuzzy = max(fuzzy_timings)

            print(f"Exact range: {fastest_exact:.2f}ms - {slowest_exact:.2f}ms")
            print(f"Fuzzy range: {fastest_fuzzy:.2f}ms - {slowest_fuzzy:.2f}ms")


if __name__ == "__main__":
    # Run with verbose output to see what's happening
    unittest.main(verbosity=2)