from __future__ import annotations

from datetime import date

from osspulse.github_search import Shard, build_query, iter_shards


def test_shard_date_range() -> None:
    s = Shard(2019, 2)
    start, end = s.date_range()
    assert start == date(2019, 2, 1)
    assert end == date(2019, 2, 28)

    s_dec = Shard(2020, 12)
    assert s_dec.date_range() == (date(2020, 12, 1), date(2020, 12, 31))


def test_iter_shards_count(cfg) -> None:
    shards = list(iter_shards(cfg))
    assert len(shards) == 36
    assert shards[0].label == "2019-01"
    assert shards[-1].label == "2021-12"


def test_build_query_contains_filters(cfg) -> None:
    q = build_query(cfg, Shard(2019, 1), min_stars=5)
    assert "language:Python" in q
    assert "created:2019-01-01..2019-01-31" in q
    assert "stars:>=5" in q
    assert "fork:false" in q
    assert "archived:false" in q
