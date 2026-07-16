"""Tests for lookup_beliefs progressive relaxation."""

import os
import tempfile

from tools import _filter_query_terms, _relaxed_substring_search, lookup_beliefs


SAMPLE_BELIEFS = """\
### retraction-cascade-propagates [IN]
Retraction of a premise cascades to all dependent derived beliefs.

### outlist-nodes-not-tracked [IN]
Outlist nodes are not tracked in the dependents index.

### nogood-detection-works [OUT]
Nogood detection correctly identifies contradictions and triggers backtracking.

### federation-supports-multi-agent [IN]
Multi-agent federation allows separate TMS instances to share beliefs.
"""


class TestFilterQueryTerms:

    def test_removes_stop_words(self):
        terms = _filter_query_terms("what do we know about retraction")
        assert "what" not in terms
        assert "do" not in terms
        assert "we" not in terms
        assert "about" not in terms
        assert "retraction" in terms
        assert "know" in terms

    def test_removes_single_char(self):
        terms = _filter_query_terms("a b retraction")
        assert "a" not in terms
        assert "b" not in terms
        assert "retraction" in terms

    def test_all_stop_words_falls_back_to_long_tokens(self):
        terms = _filter_query_terms("what is the")
        assert terms == ["what", "is", "the"]

    def test_all_stop_words_fallback_includes_all_long(self):
        terms = _filter_query_terms("do we have")
        assert terms == ["do", "we", "have"]

    def test_preserves_content_words(self):
        terms = _filter_query_terms("retraction cascade propagation")
        assert terms == ["retraction", "cascade", "propagation"]

    def test_lowercases_terms(self):
        terms = _filter_query_terms("Retraction CASCADE")
        assert terms == ["retraction", "cascade"]


class TestRelaxedSubstringSearch:

    def _parse_beliefs(self, content):
        beliefs = []
        current = []
        for line in content.split("\n"):
            if line.startswith("### ") and "[" in line and "]" in line:
                if current:
                    beliefs.append("\n".join(current))
                current = [line]
            elif current:
                current.append(line)
        if current:
            beliefs.append("\n".join(current))
        return beliefs

    def test_all_terms_match(self):
        beliefs = self._parse_beliefs(SAMPLE_BELIEFS)
        results = _relaxed_substring_search(beliefs, ["retraction", "cascade"])
        assert len(results) == 1
        assert "retraction-cascade" in results[0]

    def test_no_terms_returns_empty(self):
        beliefs = self._parse_beliefs(SAMPLE_BELIEFS)
        assert _relaxed_substring_search(beliefs, []) == []

    def test_progressive_relaxation_drops_missing_term(self):
        beliefs = self._parse_beliefs(SAMPLE_BELIEFS)
        results = _relaxed_substring_search(
            beliefs, ["retraction", "cascade", "xyznonexistent"]
        )
        assert len(results) == 1
        assert "retraction-cascade" in results[0]

    def test_no_match_returns_empty(self):
        beliefs = self._parse_beliefs(SAMPLE_BELIEFS)
        results = _relaxed_substring_search(beliefs, ["xyznonexistent"])
        assert results == []

    def test_single_term_match(self):
        beliefs = self._parse_beliefs(SAMPLE_BELIEFS)
        results = _relaxed_substring_search(beliefs, ["nogood"])
        assert len(results) == 1
        assert "nogood-detection" in results[0]

    def test_two_terms_no_relaxation(self):
        beliefs = self._parse_beliefs(SAMPLE_BELIEFS)
        results = _relaxed_substring_search(
            beliefs, ["retraction", "xyznonexistent"]
        )
        assert results == []


class TestLookupBeliefsTool:

    def test_natural_language_query(self, tmp_path):
        beliefs_file = tmp_path / "beliefs.md"
        beliefs_file.write_text(SAMPLE_BELIEFS)
        result = lookup_beliefs.invoke(
            {"query": "what do we know about retraction cascades",
             "beliefs_file": str(beliefs_file)}
        )
        assert "retraction-cascade-propagates" in result
        assert "Found 1" in result

    def test_direct_keyword_query(self, tmp_path):
        beliefs_file = tmp_path / "beliefs.md"
        beliefs_file.write_text(SAMPLE_BELIEFS)
        result = lookup_beliefs.invoke(
            {"query": "federation multi-agent",
             "beliefs_file": str(beliefs_file)}
        )
        assert "federation-supports-multi-agent" in result

    def test_no_match(self, tmp_path):
        beliefs_file = tmp_path / "beliefs.md"
        beliefs_file.write_text(SAMPLE_BELIEFS)
        result = lookup_beliefs.invoke(
            {"query": "xyznonexistent",
             "beliefs_file": str(beliefs_file)}
        )
        assert "No beliefs found" in result

    def test_out_beliefs_filtered(self, tmp_path):
        beliefs_file = tmp_path / "beliefs.md"
        beliefs_file.write_text(SAMPLE_BELIEFS)
        result = lookup_beliefs.invoke(
            {"query": "nogood detection",
             "beliefs_file": str(beliefs_file)}
        )
        assert "No beliefs found" in result

    def test_in_beliefs_not_filtered(self, tmp_path):
        beliefs_file = tmp_path / "beliefs.md"
        beliefs_file.write_text(SAMPLE_BELIEFS)
        result = lookup_beliefs.invoke(
            {"query": "retraction cascade",
             "beliefs_file": str(beliefs_file)}
        )
        assert "retraction-cascade-propagates" in result
        assert "[IN]" in result

    def test_file_not_found(self):
        result = lookup_beliefs.invoke(
            {"query": "test", "beliefs_file": "/nonexistent/path.md"}
        )
        assert "Error" in result
