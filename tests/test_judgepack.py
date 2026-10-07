import csv

from spintrace.eval import judged, judgepack

FIELDS = ["suspect", "source", "judge1", "judge2"]


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_merge_adds_missing_labels_and_keeps_existing(tmp_path, monkeypatch):
    mine, theirs = tmp_path / "mine.csv", tmp_path / "theirs.csv"
    write(mine, [dict(suspect="a", source="b", judge1="derived", judge2=""),
                 dict(suspect="c", source="d", judge1="derived", judge2="derived"),
                 dict(suspect="e", source="f", judge1="", judge2="")])
    write(theirs, [dict(suspect="a", source="b", judge1="", judge2="not_derived"),
                   dict(suspect="c", source="d", judge1="", judge2="not_derived"),
                   dict(suspect="e", source="f", judge1="", judge2="")])
    monkeypatch.setattr(judged, "SHEETS", {"live": mine})
    out = judgepack.merge(theirs, "live", 2)
    assert out == {"labels_in_file": 2, "added": 1, "clashes_kept_ours": 1}
    rows = list(csv.DictReader(mine.open()))
    assert [r["judge2"] for r in rows] == ["not_derived", "derived", ""]
    assert rows[0]["judge1"] == "derived"


def test_is_positive_either_versus_both():
    row = dict(judge1="derived", judge2="not_derived")
    assert judged.is_positive(row, "live", "any") is True
    assert judged.is_positive(row, "live", "both") is False
    assert judged.is_positive(dict(judge1="", judge2=""), "live") is None
    assert judged.is_positive(dict(judge1="not_relevant", judge2=""), "search") is False
