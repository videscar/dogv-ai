from __future__ import annotations

from agent.nodes.retrieve import inject_semantic_anchors
from api.retrieval import rrf_fuse


def _doc(doc_id: int, **extra):
    return {"document_id": doc_id, "title": f"doc {doc_id}", **extra}


def _lane(*doc_ids: int):
    return [_doc(d) for d in doc_ids]


def test_evicted_semantic_top_doc_is_appended_with_true_rrf_score():
    # Three correlated lexical lanes vote for docs 1..4; the semantic lane's
    # top doc (99) gets a single vote and falls past the max_docs=4 cutoff.
    lexical = _lane(1, 2, 3, 4)
    semantic = _lane(99, 1, 2)
    sources = [lexical, lexical, lexical, semantic]
    weights = [1.0, 1.0, 1.0, 1.0]
    fused = rrf_fuse(sources, max_docs=4, weights=weights)
    assert [d["document_id"] for d in fused] == [1, 2, 3, 4]

    fused, added = inject_semantic_anchors(fused, sources, weights, [semantic], anchor_top=3)
    assert added == 1
    ids = [d["document_id"] for d in fused]
    assert ids == [1, 2, 3, 4, 99]
    # true fused score: single vote at rank 1 of the semantic lane
    assert fused[-1]["rrf_score"] == 1.0 / 61


def test_noop_when_anchors_already_in_pool():
    lane = _lane(1, 2, 3)
    sources = [lane]
    fused = rrf_fuse(sources, max_docs=10, weights=[1.0])
    before = [d["document_id"] for d in fused]
    fused, added = inject_semantic_anchors(fused, sources, [1.0], [lane], anchor_top=3)
    assert added == 0
    assert [d["document_id"] for d in fused] == before


def test_anchor_top_zero_disables_injection():
    lexical = _lane(1, 2)
    semantic = _lane(99)
    sources = [lexical, lexical, semantic]
    weights = [1.0, 1.0, 1.0]
    fused = rrf_fuse(sources, max_docs=2, weights=weights)
    fused, added = inject_semantic_anchors(fused, sources, weights, [semantic], anchor_top=0)
    assert added == 0
    assert [d["document_id"] for d in fused] == [1, 2]


def test_only_top_n_of_anchor_lane_is_guaranteed():
    lexical = _lane(1, 2, 3)
    semantic = _lane(50, 51, 52, 99)  # 99 is rank 4: beyond anchor_top=3
    sources = [lexical, lexical, semantic]
    weights = [1.0, 1.0, 1.0]
    fused = rrf_fuse(sources, max_docs=3, weights=weights)
    fused, added = inject_semantic_anchors(fused, sources, weights, [semantic], anchor_top=3)
    ids = [d["document_id"] for d in fused]
    assert 99 not in ids
    assert {50, 51, 52}.issubset(set(ids))
    assert added == 3


def test_multiple_anchor_lanes_dedupe_and_preserve_pool_order():
    lexical = _lane(1, 2, 3)
    sem_a = _lane(99, 1)
    sem_b = _lane(99, 2)  # same anchor from a second lane: appended once
    sources = [lexical, lexical, lexical, sem_a, sem_b]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    fused = rrf_fuse(sources, max_docs=3, weights=weights)
    head = [d["document_id"] for d in fused]
    fused, added = inject_semantic_anchors(fused, sources, weights, [sem_a, sem_b], anchor_top=2)
    assert added == 1
    assert [d["document_id"] for d in fused] == head + [99]
    # 99 holds votes from both semantic lanes at rank 1
    assert fused[-1]["rrf_score"] == 2.0 / 61


def test_empty_pool_is_left_alone():
    fused, added = inject_semantic_anchors([], [[]], [1.0], [_lane(1)], anchor_top=3)
    assert fused == []
    assert added == 0
