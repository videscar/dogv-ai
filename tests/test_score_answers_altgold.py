from __future__ import annotations

from eval_v2.altgold import accept_set, citation_any_hit, citation_full_hit


def test_accept_set_unions_gold_and_siblings():
    sc = {"gold_doc_ids": [88759], "accept_doc_ids": [86267], "gold_sets": [[88759]]}
    assert accept_set(sc) == {88759, 86267}


def test_accept_set_handles_missing_fields():
    assert accept_set({"gold_sets": [[1], [2]]}) == {1, 2}
    assert accept_set({}) == set()


def test_any_hit_accepts_valid_sibling():
    # Citing the es twin of a va flagship counts as correct.
    accept = accept_set({"gold_doc_ids": [88759], "accept_doc_ids": [86267]})
    assert citation_any_hit([86267, 999], accept) is True
    assert citation_any_hit([999], accept) is False


def test_any_hit_ignores_unparseable_ids():
    assert citation_any_hit([None, "x", 11], {11}) is True
    assert citation_any_hit([None, "x"], {11}) is False


def test_full_hit_requires_every_or_group_covered():
    gold_sets = [[86529], [86200]]
    assert citation_full_hit([86529, 86200], gold_sets) is True
    assert citation_full_hit([86529], gold_sets) is False  # second group missing


def test_full_hit_or_group_satisfied_by_any_member():
    gold_sets = [[86667, 86084]]  # either is acceptable for the single group
    assert citation_full_hit([86084], gold_sets) is True


def test_full_hit_empty_goldsets_is_false():
    assert citation_full_hit([1, 2], None) is False
    assert citation_full_hit([1, 2], []) is False
