from __future__ import annotations

from datetime import date

from api.edition_recency import group_editions, stale_edition_ids


def test_group_editions_unions_transitively():
    groups = group_editions([(1, 2), (2, 3), (10, 11)])
    as_sets = sorted((sorted(g) for g in groups), key=lambda g: g[0])
    assert as_sets == [[1, 2, 3], [10, 11]]


def test_group_editions_drops_singletons():
    # ids that never pair are not returned as groups
    assert group_editions([]) == []


def test_older_sibling_is_stale_newest_kept():
    pairs = [(112410, 37182)]  # 2026 vs 2025 edition of the same appointment
    dates = {112410: date(2026, 7, 1), 37182: date(2025, 6, 11)}
    assert stale_edition_ids(pairs, dates) == {37182}


def test_three_year_family_keeps_only_newest():
    pairs = [(112447, 39930), (39930, 92220), (112447, 92220)]
    dates = {112447: date(2026, 7, 1), 39930: date(2025, 9, 1), 92220: date(2024, 7, 4)}
    assert stale_edition_ids(pairs, dates) == {39930, 92220}


def test_members_sharing_newest_date_all_kept():
    # two concurrent same-day publications + one older -> only the older is stale
    pairs = [(1, 2), (2, 3), (1, 3)]
    dates = {1: date(2026, 7, 1), 2: date(2026, 7, 1), 3: date(2025, 1, 1)}
    assert stale_edition_ids(pairs, dates) == {3}


def test_unknown_date_never_marked_stale():
    pairs = [(1, 2)]
    dates = {1: date(2026, 7, 1), 2: None}
    assert stale_edition_ids(pairs, dates) == set()
