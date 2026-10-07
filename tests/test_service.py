import numpy as np

from spintrace.ui import service


def test_alignment_rows_flags_sentences_above_threshold():
    best = np.array([0.9, 0.3, 0.7])
    arg = np.array([1, 0, 2])
    rows = service.alignment_rows(best, arg, ["a", "b", "c"], ["x", "y", "z"], 0.62)
    assert [r["aligned"] for r in rows] == [True, False, True]
    assert rows[0]["source"] == "y" and rows[0]["source_i"] == 1


def test_alignment_rows_respects_limit():
    best = np.ones(5)
    rows = service.alignment_rows(best, np.zeros(5, dtype=int), list("abcde"), ["x"], 0.5, limit=3)
    assert len(rows) == 3


def test_clean_turns_nan_into_none():
    assert service._clean(float("nan")) is None
    assert service._clean(1.5) == 1.5
