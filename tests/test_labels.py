"""Unit tests for medvisionqa.labels (imbalance math, missing files,
CSV hygiene). The label-noise heuristic itself is covered end-to-end on
the seeded synthetic dataset in test_synth_demo.py."""
from medvisionqa.labels import analyze_labels

from conftest import solid_image, textured_image, write_csv


def _labeled_dataset(tmp_path, n_a, n_b):
    (tmp_path / "train").mkdir(exist_ok=True)
    rows = []
    for i in range(n_a):
        name = f"train/a_{i}.png"
        textured_image(str(tmp_path / name), seed=1000 + i, size=(16, 16))
        rows.append((name, "class_a"))
    for i in range(n_b):
        name = f"train/b_{i}.png"
        textured_image(str(tmp_path / name), seed=2000 + i, size=(16, 16))
        rows.append((name, "class_b"))
    csv_path = str(tmp_path / "labels.csv")
    write_csv(csv_path, rows)
    return csv_path


def _checks(result):
    return {f["check"] for f in result["findings"]}


def test_imbalance_warning_below_threshold(tmp_path):
    # 1/21 = 4.76% < 5% -> warning
    csv_path = _labeled_dataset(tmp_path, 20, 1)
    res = analyze_labels(str(tmp_path), csv_path, run_noise_check=False)
    assert res["counts"] == {"class_a": 20, "class_b": 1}
    assert res["total"] == 21
    imb = [f for f in res["findings"] if f["check"] == "class_imbalance"]
    assert len(imb) == 1
    assert imb[0]["severity"] == "warning"
    assert imb[0]["details"]["class"] == "class_b"
    assert abs(imb[0]["details"]["ratio"] - round(1 / 21, 4)) < 1e-9


def test_imbalance_boundary_not_flagged(tmp_path):
    # 1/20 = 5.0% is NOT < 5% -> no warning (strict threshold)
    csv_path = _labeled_dataset(tmp_path, 19, 1)
    res = analyze_labels(str(tmp_path), csv_path, run_noise_check=False)
    assert "class_imbalance" not in _checks(res)


def test_balanced_classes_no_warning(tmp_path):
    csv_path = _labeled_dataset(tmp_path, 10, 10)
    res = analyze_labels(str(tmp_path), csv_path, run_noise_check=False)
    assert "class_imbalance" not in _checks(res)
    assert "class_distribution" in _checks(res)


def test_missing_labeled_file_is_error(tmp_path):
    (tmp_path / "train").mkdir()
    textured_image(str(tmp_path / "train" / "ok.png"), seed=5, size=(16, 16))
    csv_path = str(tmp_path / "labels.csv")
    write_csv(csv_path, [("train/ok.png", "a"), ("train/ghost.png", "a")])
    res = analyze_labels(str(tmp_path), csv_path, run_noise_check=False)
    missing = [f for f in res["findings"] if f["check"] == "missing_labeled_files"]
    assert len(missing) == 1
    assert missing[0]["severity"] == "error"
    assert missing[0]["files"] == ["train/ghost.png"]


def test_unreadable_labeled_file_is_warning(tmp_path):
    (tmp_path / "train").mkdir()
    textured_image(str(tmp_path / "train" / "ok.png"), seed=5, size=(16, 16))
    (tmp_path / "train" / "bad.png").write_bytes(b"junk" * 32)
    csv_path = str(tmp_path / "labels.csv")
    write_csv(csv_path, [("train/ok.png", "a"), ("train/bad.png", "a")])
    res = analyze_labels(str(tmp_path), csv_path, run_noise_check=False)
    unr = [f for f in res["findings"] if f["check"] == "unreadable_labeled_files"]
    assert len(unr) == 1
    assert unr[0]["severity"] == "warning"


def test_duplicate_csv_entries_flagged(tmp_path):
    csv_path = _labeled_dataset(tmp_path, 2, 2)
    with open(csv_path, "a", encoding="utf-8") as fh:
        fh.write("train/a_0.png,class_a\n")
    res = analyze_labels(str(tmp_path), csv_path, run_noise_check=False)
    assert "duplicate_label_entries" in _checks(res)


def test_bad_csv_header_raises(tmp_path):
    csv_path = str(tmp_path / "labels.csv")
    write_csv(csv_path, [])  # header written, then replaced
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("path,category\ntrain/a.png,x\n")
    import pytest
    with pytest.raises(ValueError, match="filepath.*label"):
        analyze_labels(str(tmp_path), csv_path, run_noise_check=False)
