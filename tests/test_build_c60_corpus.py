"""Tests for the 60%-identity corpus builder's label exclusion and member draw.

The exclusion has to find a labelled sequence wherever it sits in a 60%
cluster. An earlier version only compared labelled md5s with 90%-centroid md5s,
which caught about 120 of the 1,219 labelled sequences and left the clusters of
the rest in the pretraining corpus. The first test is the regression guard.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _load_builder():
    path = Path(__file__).parent.parent / "scripts" / "build_c60_corpus.py"
    spec = importlib.util.spec_from_file_location("build_c60_corpus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _md5(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()


def _write_orfs(directory: Path) -> None:
    """Two spine parts. Cluster 7's labelled sequence is a member, not a centroid."""
    directory.mkdir()
    pq.write_table(
        pa.table({
            "orf_id": [1, 2, 3],
            "md5": [_md5("MKV"), _md5("MKVL"), _md5("AAAA")],
            "c60_id": [7, 7, 8],
        }),
        directory / "part-0.parquet",
        row_group_size=2,
    )
    pq.write_table(
        pa.table({
            "orf_id": [4, 5],
            "md5": [_md5("GGGG"), _md5("PPPP")],
            "c60_id": [9, 10],
        }),
        directory / "part-1.parquet",
    )


def test_a_labelled_member_excludes_its_cluster(tmp_path) -> None:
    _write_orfs(tmp_path / "orfs")
    labelled = {_md5("MKVL"), _md5("PPPP"), _md5("NOT_IN_CORPUS")}

    clusters, found = builder.find_labelled_clusters(tmp_path / "orfs", labelled)

    assert clusters == {7, 10}
    assert found == {_md5("MKVL"), _md5("PPPP")}


def test_missing_spine_is_an_error(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError, match="orfs/part"):
        builder.find_labelled_clusters(tmp_path, {_md5("MKV")})


def test_label_md5s_follow_the_corpus_convention(tmp_path) -> None:
    """A lowercase sequence with a stop must hash like the corpus's ``md5``."""
    labels = tmp_path / "sequences.csv"
    labels.write_text("seq_md5,sequence\nwhatever,mkvl*\n")

    assert builder.read_labelled_md5s(labels) == {_md5("MKVL")}


def test_label_md5s_without_sequences_are_normalised(tmp_path) -> None:
    labels = tmp_path / "sequences.csv"
    labels.write_text(f"seq_md5\n {_md5('MKVL').upper()} \n")

    assert builder.read_labelled_md5s(labels) == {_md5("MKVL")}


def _centroids(path: Path) -> pq.ParquetFile:
    pq.write_table(
        pa.table({"c60_id": [7, 7, 7, 8, 8, 9, 10]}), path, row_group_size=3
    )
    return pq.ParquetFile(path)


def test_choose_members_draws_one_per_kept_cluster(tmp_path) -> None:
    parquet = _centroids(tmp_path / "centroids.parquet")

    by_group, stats = builder.choose_members(parquet, excluded={7, 10}, seed=0)

    chosen = [(group, local) for group, rows in by_group.items() for local in rows]
    assert stats["clusters_total"] == 4
    assert stats["clusters_excluded"] == 2
    assert stats["sequences_selected"] == len(chosen) == 2

    table = parquet.read().column("c60_id").to_pylist()
    picked_clusters = sorted(table[group * 3 + local] for group, local in chosen)
    assert picked_clusters == [8, 9]


def test_choose_members_is_seeded(tmp_path) -> None:
    parquet = _centroids(tmp_path / "centroids.parquet")

    first, _ = builder.choose_members(parquet, excluded=set(), seed=3)
    same, _ = builder.choose_members(parquet, excluded=set(), seed=3)

    assert first == same
