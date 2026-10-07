"""Per-image integrity checks: corruption, degenerate pixels, dHash
near-duplicates (incl. cross-split leakage), and geometry/mode consistency.

Offline, CPU-only. Depends on numpy + Pillow only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "test", "val", "valid", "validation"}


@dataclass
class ImageRecord:
    path: str
    rel: str
    split: str
    ok: bool = True
    error: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mode: Optional[str] = None
    degenerate: Optional[str] = None  # 'all_black' | 'all_white' | 'constant'
    dhash: Optional[int] = None


# --------------------------------------------------------------------------- #
# perceptual hash
# --------------------------------------------------------------------------- #
def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash: 64-bit int. Robust to brightness/contrast shifts and
    small resizes; near-identical images get tiny Hamming distances."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    px = np.asarray(gray, dtype=np.int16)
    diff = (px[:, 1:] > px[:, :-1]).flatten()
    h = 0
    for bit in diff:
        h = (h << 1) | int(bit)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# single-file checks
# --------------------------------------------------------------------------- #
def load_image(path: str) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """Returns (ok, error, image). verify() catches truncated headers; the
    second open+load() catches truncated pixel data."""
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as exc:  # noqa: BLE001 - any decode failure = corrupt
        return False, f"{type(exc).__name__}: {exc}", None
    try:
        im = Image.open(path)
        im.load()
        return True, None, im
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", None


def detect_degenerate(image: Image.Image) -> Optional[str]:
    """Flag images carrying no information: all-black, all-white, constant."""
    arr = np.asarray(image.convert("L"))
    if arr.size == 0:
        return "empty"
    mn, mx = int(arr.min()), int(arr.max())
    if mn == mx:
        if mx == 0:
            return "all_black"
        if mx == 255:
            return "all_white"
        return "constant"
    return None


def split_of(rel_path: str) -> str:
    first = rel_path.split(os.sep)[0].lower()
    return first if first in SPLIT_NAMES else "all"


def scan_dataset(root: str) -> List[ImageRecord]:
    """Walk *root*, run per-file checks, return one record per image file."""
    records: List[ImageRecord] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            rec = ImageRecord(path=path, rel=rel, split=split_of(rel))
            ok, error, im = load_image(path)
            if not ok:
                rec.ok = False
                rec.error = error
                records.append(rec)
                continue
            rec.width, rec.height = im.size
            rec.mode = im.mode
            rec.degenerate = detect_degenerate(im)
            if rec.degenerate is None:
                rec.dhash = dhash(im)
            records.append(rec)
    return records


# --------------------------------------------------------------------------- #
# duplicate grouping (union-find over pairwise Hamming distances)
# --------------------------------------------------------------------------- #
def find_duplicate_groups(
    records: List[ImageRecord], threshold: int = 5
) -> List[List[ImageRecord]]:
    """Group images whose dHash Hamming distance <= threshold. Degenerate and
    corrupt files are excluded (constant images trivially hash alike)."""
    cands = [r for r in records if r.ok and r.dhash is not None]
    parent = list(range(len(cands)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(cands)):
        hi = cands[i].dhash
        for j in range(i + 1, len(cands)):
            if hamming(hi, cands[j].dhash) <= threshold:
                union(i, j)

    buckets: Dict[int, List[ImageRecord]] = {}
    for i, rec in enumerate(cands):
        buckets.setdefault(find(i), []).append(rec)
    return [sorted(g, key=lambda r: r.rel) for g in buckets.values() if len(g) > 1]


# --------------------------------------------------------------------------- #
# audit entry point
# --------------------------------------------------------------------------- #
def _finding(severity: str, check: str, message: str,
             files: Optional[List[str]] = None,
             details: Optional[dict] = None) -> dict:
    return {
        "severity": severity,
        "check": check,
        "message": message,
        "files": sorted(files or []),
        "details": details or {},
    }


def audit_images(root: str, hash_threshold: int = 5) -> dict:
    """Run all per-image checks. Returns {'records', 'findings', 'stats'}."""
    records = scan_dataset(root)
    findings: List[dict] = []

    corrupt = [r for r in records if not r.ok]
    if corrupt:
        findings.append(_finding(
            "error", "corrupt_files",
            f"{len(corrupt)} corrupt/unreadable image file(s) — "
            "they will crash or silently poison a data loader.",
            [r.rel for r in corrupt],
            {"errors": {r.rel: r.error for r in corrupt}},
        ))

    degen: Dict[str, List[str]] = {}
    for r in records:
        if r.degenerate:
            degen.setdefault(r.degenerate, []).append(r.rel)
    if degen:
        total = sum(len(v) for v in degen.values())
        kinds = ", ".join(f"{len(v)} {k}" for k, v in sorted(degen.items()))
        findings.append(_finding(
            "warning", "degenerate_images",
            f"{total} degenerate image(s) with no usable signal ({kinds}).",
            [f for v in degen.values() for f in v],
            {"by_type": {k: sorted(v) for k, v in degen.items()},
             "hint": "all-black scans are often failed acquisitions; "
                     "all-white are often overexposed/blank."},
        ))

    # geometry / mode consistency vs. the majority
    good = [r for r in records if r.ok]
    if good:
        from collections import Counter
        geom = Counter((r.width, r.height, r.mode) for r in good)
        (mw, mh, mmode), _n = geom.most_common(1)[0]
        odd = [r for r in good
               if (r.width, r.height, r.mode) != (mw, mh, mmode)]
        if odd:
            findings.append(_finding(
                "warning", "geometry_mode_inconsistency",
                f"{len(odd)} image(s) differ from the majority geometry/mode "
                f"({mw}x{mh}, mode {mmode}) — resize/normalize before batching.",
                [r.rel for r in odd],
                {"majority": {"width": mw, "height": mh, "mode": mmode},
                 "offenders": {r.rel: {"width": r.width, "height": r.height,
                                       "mode": r.mode} for r in odd}},
            ))

    # near-duplicates: within-split vs cross-split (leakage)
    for group in find_duplicate_groups(records, threshold=hash_threshold):
        splits = sorted({r.split for r in group})
        files = [r.rel for r in group]
        if len(splits) > 1:
            findings.append(_finding(
                "error", "cross_split_leakage",
                f"DATA LEAKAGE: {len(group)} near-duplicate image(s) appear in "
                f"multiple splits {splits} (max Hamming "
                f"{max(hamming(a.dhash, b.dhash) for a in group for b in group if a is not b)}). "
                "Test metrics measured on these are inflated.",
                files,
                {"splits": splits, "group_size": len(group)},
            ))
        else:
            findings.append(_finding(
                "warning", "near_duplicates_within_split",
                f"{len(group)} near-duplicate image(s) inside split "
                f"'{splits[0]}' — redundant samples bias training.",
                files,
                {"split": splits[0], "group_size": len(group)},
            ))

    stats = {
        "n_files": len(records),
        "n_ok": sum(1 for r in records if r.ok),
        "n_corrupt": len(corrupt),
        "n_degenerate": sum(len(v) for v in degen.values()),
        "splits": sorted({r.split for r in records}),
    }
    return {"records": records, "findings": findings, "stats": stats}
