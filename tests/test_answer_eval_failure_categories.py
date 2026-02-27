from __future__ import annotations

from scripts.run_answer_eval import _classify_failure


def test_classify_failure_runtime_error():
    category = _classify_failure(
        error="timed out",
        checks={
            "required_citations": 1,
            "found_citations": 0,
            "must_include_hits": [],
            "missing_must_include": ["x"],
            "must_not_hits": [],
        },
        passed=False,
    )
    assert category == "runtime_error"


def test_classify_failure_lexical_mismatch():
    category = _classify_failure(
        error=None,
        checks={
            "required_citations": 1,
            "found_citations": 1,
            "must_include_hits": ["municip"],
            "missing_must_include": ["ajud"],
            "must_not_hits": [],
        },
        passed=False,
    )
    assert category == "lexical_mismatch"


def test_classify_failure_citation_gap():
    category = _classify_failure(
        error=None,
        checks={
            "required_citations": 2,
            "found_citations": 1,
            "must_include_hits": ["x"],
            "missing_must_include": [],
            "must_not_hits": [],
        },
        passed=False,
    )
    assert category == "citation_gap"


def test_classify_failure_content_failure():
    category = _classify_failure(
        error=None,
        checks={
            "required_citations": 1,
            "found_citations": 1,
            "must_include_hits": [],
            "missing_must_include": ["a", "b"],
            "must_not_hits": [],
        },
        passed=False,
    )
    assert category == "content_failure"
