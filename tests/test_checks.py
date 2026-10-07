"""Unit tests for medvisionqa.checks (dHash, corruption, degenerate,
geometry, cross-split leakage)."""
import numpy as np
from PIL import Image

from medvisionqa import checks
from medvisionqa.checks import (
    audit_images,
    detect_degenerate,
    dhash,
    find_duplicate_groups,
    hamming,
    load_image,
)

from conftest import solid_image, textured_image


def _open(p):
    im = Image.open(p)
    im.load()
    return im


# --- dHash discrimination -------------------------------------------------- #
def test_dhash_identical_images_hash_equal(tmp_path):
    p = textured_image(str(tmp_path / "a.png"), seed=1)
    assert hamming(dhash(_open(p)), dhash(_open(p))) == 0


def test_dhash_exact_copy_matches(tmp_path):
    src = textured_image(str(tmp_path / "a.png"), seed=2)
    dst = str(tmp_path / "b.png")
    _open(src).save(dst)
    assert hamming(dhash(_open(src)), dhash(_open(dst))) == 0


def test_dhash_near_duplicate_within_threshold(tmp_path):
    rng = np.random.default_rng(3)
    base = (rng.integers(0, 256, size=(48, 48, 3))).astype(np.int16)
    noisy = np.clip(base + rng.integers(-3, 4, size=base.shape), 0, 255).astype(np.uint8)
    p1, p2 = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    Image.fromarray(base.astype(np.uint8)).save(p1)
    Image.fromarray(noisy).save(p2)
    assert hamming(dhash(_open(p1)), dhash(_open(p2))) <= 5


def test_dhash_different_images_far_apart(tmp_path):
    p1 = textured_image(str(tmp_path / "a.png"), seed=10)
    p2 = textured_image(str(tmp_path / "b.png"), seed=11)
    assert hamming(dhash(_open(p1)), dhash(_open(p2))) > 5


# --- corruption ------------------------------------------------------------ #
def test_corrupt_file_detected(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64 + b"garbage" * 20)
    ok, err, im = load_image(str(p))
    assert not ok
    assert err
    assert im is None


def test_truncated_file_detected(tmp_path):
    p = textured_image(str(tmp_path / "a.png"), seed=4)
    data = p.read_bytes() if hasattr(p, "read_bytes") else open(p, "rb").read()
    with open(str(tmp_path / "cut.png"), "wb") as fh:
        fh.write(data[: len(data) // 3])
    ok, _err, _im = load_image(str(tmp_path / "cut.png"))
    assert not ok


def test_audit_flags_corrupt_files(tmp_path):
    (tmp_path / "train").mkdir()
    textured_image(str(tmp_path / "train" / "ok.png"), seed=5)
    (tmp_path / "train" / "bad.png").write_bytes(b"not an image at all")
    res = audit_images(str(tmp_path))
    by_check = {f["check"]: f for f in res["findings"]}
    assert "corrupt_files" in by_check
    assert by_check["corrupt_files"]["severity"] == "error"
    assert any("bad.png" in f for f in by_check["corrupt_files"]["files"])
    assert res["stats"]["n_corrupt"] == 1


# --- degenerate ------------------------------------------------------------ #
def test_degenerate_variants():
    assert detect_degenerate(Image.new("L", (8, 8), 0)) == "all_black"
    assert detect_degenerate(Image.new("L", (8, 8), 255)) == "all_white"
    assert detect_degenerate(Image.new("L", (8, 8), 128)) == "constant"
    assert detect_degenerate(Image.new("RGB", (8, 8), (0, 0, 0))) == "all_black"


def test_normal_image_not_degenerate(tmp_path):
    p = textured_image(str(tmp_path / "a.png"), seed=6)
    assert detect_degenerate(_open(p)) is None


def test_audit_flags_degenerate(tmp_path):
    (tmp_path / "train").mkdir()
    textured_image(str(tmp_path / "train" / "ok.png"), seed=7)
    solid_image(str(tmp_path / "train" / "black.png"), (0, 0, 0))
    res = audit_images(str(tmp_path))
    by_check = {f["check"]: f for f in res["findings"]}
    assert "degenerate_images" in by_check
    assert by_check["degenerate_images"]["severity"] == "warning"


# --- geometry / mode -------------------------------------------------------- #
def test_audit_flags_geometry_outlier(tmp_path):
    (tmp_path / "train").mkdir()
    for i in range(3):
        textured_image(str(tmp_path / "train" / f"ok{i}.png"), seed=20 + i,
                       size=(48, 48))
    textured_image(str(tmp_path / "train" / "odd.png"), seed=99, size=(24, 24))
    res = audit_images(str(tmp_path))
    by_check = {f["check"]: f for f in res["findings"]}
    assert "geometry_mode_inconsistency" in by_check
    assert any("odd.png" in f for f in by_check["geometry_mode_inconsistency"]["files"])


# --- duplicates & cross-split leakage --------------------------------------- #
def _leak_fixture(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    textured_image(str(tmp_path / "train" / "base.png"), seed=42)
    _open(str(tmp_path / "train" / "base.png")).save(str(tmp_path / "test" / "leak.png"))
    textured_image(str(tmp_path / "train" / "other.png"), seed=43)
    rng = np.random.default_rng(44)
    arr = np.asarray(_open(str(tmp_path / "train" / "other.png"))).astype(np.int16)
    arr = np.clip(arr + rng.integers(-3, 4, size=arr.shape), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(str(tmp_path / "train" / "other_near.png"))
    return tmp_path


def test_cross_split_leakage_caught(tmp_path):
    _leak_fixture(tmp_path)
    res = audit_images(str(tmp_path))
    leaks = [f for f in res["findings"] if f["check"] == "cross_split_leakage"]
    assert len(leaks) == 1
    assert leaks[0]["severity"] == "error"
    files = " ".join(leaks[0]["files"])
    assert "test/leak.png" in files.replace("\\", "/")
    assert "train/base.png" in files.replace("\\", "/")


def test_within_split_duplicates_are_warning(tmp_path):
    _leak_fixture(tmp_path)
    res = audit_images(str(tmp_path))
    dups = [f for f in res["findings"] if f["check"] == "near_duplicates_within_split"]
    assert len(dups) == 1
    assert dups[0]["severity"] == "warning"
    files = " ".join(dups[0]["files"]).replace("\\", "/")
    assert "train/other.png" in files and "train/other_near.png" in files


def test_no_false_duplicate_groups(tmp_path):
    (tmp_path / "train").mkdir()
    for i in range(4):
        textured_image(str(tmp_path / "train" / f"img{i}.png"), seed=100 + i)
    groups = find_duplicate_groups(checks.scan_dataset(str(tmp_path)))
    assert groups == []


def test_split_detection_defaults_to_all(tmp_path):
    textured_image(str(tmp_path / "a.png"), seed=8)
    recs = checks.scan_dataset(str(tmp_path))
    assert recs and all(r.split == "all" for r in recs)
