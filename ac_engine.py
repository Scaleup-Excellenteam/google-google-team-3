# ac_engine.py — prefix substring search (case-insensitive), ≤1 typo; returns ALL matching lines
import os
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Iterable
from itertools import combinations
import re

from config import JSON_PATH, INDEX_PATH

# Optional speedup
try:
    from Levenshtein import distance as lev_distance
except ImportError:
    lev_distance = None


@dataclass
class AutoCompleteData:
    completed_sentence: str
    source: str
    score: int

def _single_edit_descriptor(a: str, b: str):
    """
    Return (op, pos) describing exactly ONE edit to turn a -> b, or (None, None)
    if there are 0 edits or >1 edits.
      op ∈ {'replace','insert','delete'}
      pos: for 'replace'/'delete' is index in 'a'; for 'insert' is index in 'b'
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return None, None

    i = 0
    # advance while equal
    while i < la and i < lb and a[i] == b[i]:
        i += 1

    if la == lb:
        # either identical (0 edits) or one replace at i
        if i == la:
            return None, None  # 0 edits
        # after the first mismatch, suffix must match
        return ('replace', i) if a[i+1:] == b[i+1:] else (None, None)

    if la + 1 == lb:
        # need to INSERT one char into 'a' at position i to get 'b'
        # i.e., 'b' has an EXTRA letter (index i) vs user input
        return ('insert', i) if a[i:] == b[i+1:] else (None, None)

    # la == lb + 1:
    # need to DELETE a[i] from 'a' to get 'b'
    # i.e., 'b' is MISSING the letter at index i of user input
    return ('delete', i) if a[i+1:] == b[i:] else (None, None)

def _score_by_rules(user_input: str, window: str) -> int | None:
    """
    Scoring rules:
      exact:                    2*m
      replace at i:             2*(m-1) - (5-pen(i))
      delete (missing) at i:    2*m - 2*(5-pen(i))
      insert (extra) at i:      2*(m-1)   2*(5- pen(i))
    where pen(i) = i for i in [0..4], else 1.
    """
    m = len(user_input)
    if window == user_input:
        return 2 * m

    op, i = _single_edit_descriptor(user_input, window)
    if op is None:
        return None  # not <= 1 edit

    pen = (5-i if i < 5 else 1)

    if op == 'insert':
        # window has an extra char vs input
        return 2 * (m-1) - 2 * pen
    elif op == 'replace':
        return 2 * (m-1) - pen
    else:
        # 'delete' (missing char)
        return 2 * m - 2 * pen



def _best_prefix_score(line_norm: str, qnorm: str) -> tuple[int | None, int | None]:
    """
    Find the best (offset, score) for a prefix-at-some-position window of line_norm
    that is within <=1 edit of qnorm, scored per your rules.
    Returns (offset, score) or (None, None) if not found.
    """
    n, m = len(line_norm), len(qnorm)
    if m == 0 or n == 0:
        return None, None

    # Exact first (fast path, best score)
    off = line_norm.find(qnorm)
    if off != -1:
        return off, 2 * m

    best_off, best_score = None, None

    # Substitution windows (same length)
    if n >= m:
        for i in range(n - m + 1):
            w = line_norm[i:i+m]
            # small early check: count mismatches up to 2
            diff = 0
            for a, b in zip(w, qnorm):
                if a != b:
                    diff += 1
                    if diff > 1:
                        break
            if diff <= 1:
                sc = _score_by_rules(qnorm, w)
                if sc is not None and (best_score is None or sc > best_score or (sc == best_score and (best_off is None or i < best_off))):
                    best_off, best_score = i, sc

    # One extra char in window (insert relative to input) → window length m+1
    if n >= m + 1:
        for i in range(n - (m + 1) + 1):
            w = line_norm[i:i + m + 1]
            sc = _score_by_rules(qnorm, w)
            if sc is not None and (best_score is None or sc > best_score or (sc == best_score and (best_off is None or i < best_off))):
                best_off, best_score = i, sc

    # One missing char in window (delete relative to input) → window length m-1
    if m >= 1 and n >= m - 1:
        for i in range(n - (m - 1) + 1):
            w = line_norm[i:i + m - 1]
            sc = _score_by_rules(qnorm, w)
            if sc is not None and (best_score is None or sc > best_score or (sc == best_score and (best_off is None or i < best_off))):
                best_off, best_score = i, sc

    return best_off, best_score

def normalize(s: str) -> str:
    """Case-insensitive normalization."""
    s = re.sub(r'[^A-Za-z0-9 ]','',s)
    return s.lower()


def trigrams(s: str) -> Iterable[str]:
    """Generate 3-grams including spaces/punctuation."""
    n = len(s)
    if n < 3:
        if n > 0:
            yield s
        return
    for i in range(n - 2):
        yield s[i:i + 3]



class AutoCompleteEngine:
    """Autocomplete engine using trigram indexing with fuzzy matching."""

    def __init__(self):
        self.entries: List[Dict[str, str]] = []
        self.tri2ids: Dict[str, Set[int]] = defaultdict(set)

    def build_from_json(self, json_path: str):
        """Build index from JSON data file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.entries.clear()
        self.tri2ids.clear()

        for idx, entry in enumerate(data):
            if not isinstance(entry, dict) or "sentence" not in entry:
                continue

            raw = entry["sentence"]
            source = entry.get("source", f"line_{idx}")
            norm = normalize(raw)

            self.entries.append({
                "raw": raw,
                "norm": norm,
                "source": source
            })

            # Index trigrams
            for trigram in set(trigrams(norm)):
                self.tri2ids[trigram].add(idx)

    def save(self, path: str):
        """Save engine to pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "AutoCompleteEngine":
        """Load engine from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _get_exact_candidates(self, query_norm: str) -> Set[int]:
        """Get candidates that contain all trigrams of the query."""
        query_trigrams = list(trigrams(query_norm))
        if not query_trigrams:
            return set(range(len(self.entries)))

        # Start with trigram that has fewest matches
        trigram_sets = [(self.tri2ids.get(tg, set()), tg) for tg in query_trigrams]
        trigram_sets.sort(key=lambda x: len(x[0]))

        if not trigram_sets[0][0]:
            return set()

        candidates = trigram_sets[0][0].copy()
        for trigram_set, _ in trigram_sets[1:]:
            candidates &= trigram_set
            if not candidates:
                break

        return candidates

    def _get_fuzzy_candidates(self, query_norm: str) -> Set[int]:
        """Get additional candidates by relaxing trigram requirements."""
        query_trigrams = list(trigrams(query_norm))
        if len(query_trigrams) <= 2:
            # For very short queries, return all entries
            return set(range(len(self.entries)))

        candidates = set()

        # Try combinations requiring fewer trigrams (more permissive)
        for min_trigrams in range(max(1, len(query_trigrams) - 2), len(query_trigrams)):
            for trigram_combo in combinations(query_trigrams, min_trigrams):
                trigram_sets = [self.tri2ids.get(tg, set()) for tg in trigram_combo]
                if all(trigram_sets):
                    combo_candidates = trigram_sets[0].copy()
                    for ts in trigram_sets[1:]:
                        combo_candidates &= ts
                    candidates.update(combo_candidates)

        # If still no candidates found, be even more permissive
        if not candidates and len(query_trigrams) > 1:
            # Try requiring just 1 trigram match
            for tg in query_trigrams:
                candidates.update(self.tri2ids.get(tg, set()))

        return candidates

    def query_prefix_all(self, query: str, allow_one_typo: bool = True, topn: int = 5) -> List[AutoCompleteData]:
        """Find all sentences where query appears as substring with ≤1 typo."""
        query_norm = normalize(query)
        if not query_norm:
            return []

        # Get candidates
        candidates = self._get_exact_candidates(query_norm)
        if not len(candidates) > topn and allow_one_typo:
            fuzzy_candidates = self._get_fuzzy_candidates(query_norm)
            candidates.update(fuzzy_candidates)


        # Check candidates and collect results

        # Collect best score per line
        best: dict[tuple[str, str], tuple[int, int]] = {}
        # key -> (score, entry_idx). key is (source, norm) to dedupe identical lines.

        for idx in candidates:
            if idx >= len(self.entries):
                continue
            entry = self.entries[idx]
            line_norm = entry["norm"]

            if allow_one_typo:
                off, sc = _best_prefix_score(line_norm, query_norm)
            else:
                off = line_norm.find(query_norm)
                sc = 2 * len(query_norm) if off != -1 else None

            if sc is not None:
                key = (entry["source"], entry["norm"])
                cur = best.get(key)
                if cur is None or sc > cur[0]:
                    best[key] = (sc, idx)

        # Turn into a list we can sort by (-score, sentence alpha, source alpha)
        items = []
        for (_, _), (score, idx) in best.items():
            raw = self.entries[idx]["raw"]
            src = self.entries[idx]["source"]
            items.append((score, raw, src, idx))

        # Primary: higher score first; Tie-breaker 1: sentence A→Z; Tie-breaker 2: source A→Z
        items.sort(key=lambda t: (-t[0], t[1].lower(), t[2].lower()))

        # Take the first N
        N = int(topn) if topn is not None else len(items)
        chosen = items[:N]

        # Build results
        results = [AutoCompleteData(completed_sentence=raw, source=src, score=score)
                   for (score, raw, src, idx) in chosen]

        return results

def load_engine() -> AutoCompleteEngine:
    """Load or build autocomplete engine."""
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Missing '{JSON_PATH}'. Run json_init.py first.")

    need_build = (
            not os.path.exists(INDEX_PATH) or
            os.path.getmtime(JSON_PATH) > os.path.getmtime(INDEX_PATH)
    )

    if need_build:
        engine = AutoCompleteEngine()
        engine.build_from_json(JSON_PATH)
        engine.save(INDEX_PATH)
        return engine
    
    return AutoCompleteEngine.load(INDEX_PATH)


if __name__ == "__main__":
    print("Loading autocomplete engine...")
    engine = load_engine()
    print("Loading complete.")

    query = input("\nEnter query: (CTRL+D To exit, # to start over)").strip()
    while True:
        try:
            if not query or query == "#":
                query = input("\nEnter query: ").strip()
                continue

            results = engine.query_prefix_all(query, allow_one_typo=True)
            if not results:
                print("-> No matches found.")
            else:
                print(f"Found {len(results)} match(es):")
                for i, result in enumerate(results, 1):
                    path_part, line_number = result.source.rsplit(':', 1)
                    file_name = os.path.basename(path_part)
                    print(f"  {i}. score={result.score} {result.completed_sentence} ({file_name, line_number})")

            # Prompt for next input, showing the current query as prefix
            next_input = input(f"\n{query} ").strip()
            if next_input:
                # Append new input to the current query
                query = f"{query} {next_input}".strip()
            # If just Enter, keep the query as is (user can keep extending)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break