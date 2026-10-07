"""End-to-end tests: synth plants issues -> audit catches every class.
Seeded and deterministic. Also a CLI `demo` smoke test."""
import os
import subprocess
import sys

from medvisionqa.labels import load_label_rows
from medvisionqa.report import health_score, render_markdown, run_audit
from medvisionqa.synth import synth_dataset

from conftest import findings_by_check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _slash(p):
    return p.replace(os.sep, "/")


def test_synth_manifest_counts(tmp_path):
    m = synth_dataset(str(tmp_path / "data"), seed=7)
    p = m["planted"]
    assert p["corrupt"] == 3
    assert p["near_duplicate_images"] == 5
    assert p["cross_split_leaks"] == 2
    assert p["mislabeled"] == 4
    assert p["degenerate"] == 2
    for f in p["corrupt_files"]:
        assert os.path.isfile(os.path.join(m["root"], f))
    for f in p["mislabeled_files"]:
        assert os.path.isfile(os.path.join(m["root"], f))
    rows = load_label_rows(m["labels_csv"])
    assert len(rows) == p["class_a"] + p["class_b"]
    assert p["minority_ratio"] < 0.05
    # minority is exactly 5/111
    assert (p["class_a"], p["class_b"]) == (106, 5)


def test_audit_catches_every_planted_issue_class(tmp_path):
    m = synth_dataset(str(tmp_path / "data"), seed=7)
    report = run_audit(m["root"], labels_csv=m["labels_csv"])
    by_check = findings_by_check(report)

    # 3 corrupt files, as errors
    corrupt = by_check["corrupt_files"][0]
    assert corrupt["severity"] == "error"
    assert len(corrupt["files"]) == 3

    # both cross-split leaks land in test/
    leaks = by_check["cross_split_leakage"]
    leak_test_files = {_slash(f) for g in leaks for f in g["files"]
                       if _slash(f).startswith("test/")}
    assert leak_test_files == {"test/leak_near.png", "test/leak_exact.png"}

    # the 3 within-train near-dups are found too (merged into the leakage
    # group, since leak_exact.png is an exact copy of the train base image)
    dup_files = {_slash(f) for g in leaks for f in g["files"]}
    assert {_slash(f"train/dup_near_{i}.png") for i in range(3)} <= dup_files

    # imbalance warning on the 4.5% minority class
    imb = [f for f in by_check["class_imbalance"]]
    assert imb and imb[0]["details"]["class"] == "class_b"

    # all 4 planted mislabels flagged, nothing else
    noise = by_check["label_noise_suspects"][0]
    flagged = {_slash(f) for f in noise["files"]}
    assert flagged == {_slash(f) for f in m["planted"]["mislabeled_files"]}

    # degenerate pair
    assert len(by_check["degenerate_images"][0]["files"]) == 2

    # health score reflects a messy dataset
    assert report["health_score"] < 60
    assert report["n_errors"] >= 3  # corrupt + 2 leakage groups


def test_health_score_math():
    assert health_score([]) == 100
    f = lambda s: {"severity": s}
    assert health_score([f("error")]) == 85
    assert health_score([f("warning")]) == 95
    assert health_score([f("error"), f("warning"), f("info")]) == 79
    assert health_score([f("error")] * 10) == 0  # floored


def test_render_markdown_writes_report(tmp_path):
    m = synth_dataset(str(tmp_path / "data"), seed=7)
    report = run_audit(m["root"], labels_csv=m["labels_csv"])
    out = str(tmp_path / "report.md")
    text = render_markdown(report, out)
    assert os.path.isfile(out)
    assert "health score" in text.lower()
    assert "cross_split_leakage" in text
    assert "label_noise_suspects" in text


def test_cli_demo_smoke(tmp_path):
    out = str(tmp_path / "demo")
    proc = subprocess.run(
        [sys.executable, "-m", "medvisionqa", "demo", "--out", out],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "demo OK" in proc.stdout
    assert "[PASS] mislabeled samples flagged" in proc.stdout
    assert os.path.isfile(os.path.join(out, "audit_report.md"))


def test_cli_audit_smoke(tmp_path):
    m = synth_dataset(str(tmp_path / "data"), seed=7)
    out = str(tmp_path / "r.md")
    proc = subprocess.run(
        [sys.executable, "-m", "medvisionqa", "audit", m["root"],
         "--labels", m["labels_csv"], "--out", out, "--quiet"],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert os.path.isfile(out)
