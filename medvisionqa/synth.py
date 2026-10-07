"""Synthetic dataset generator with *planted* quality issues, for demos
and tests.

Planted issues (seeded, deterministic):
  - 3 corrupt files (random bytes with image extensions)
  - 5 near-duplicate images, incl. 2 leaking across train/test
  - severe class imbalance (minority class < 5%)
  - 4 mislabeled samples (true pattern of the other class)
  - 2 degenerate images (all-black, all-white) + 1 odd-geometry image

Classes are visually distinct on purpose (dark bg + bright disc  vs.
bright bg + dark square) so the label-noise heuristic has a fair signal.
"""
from __future__ import annotations

import csv
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

SIZE = (96, 96)
ODD_SIZE = (64, 64)


def _pattern(rng: np.random.Generator, cls: int, size: Tuple[int, int],
             noise: int = 14) -> Image.Image:
    """cls 0: dark background, bright disc. cls 1: bright bg, dark square.

    Position, size and brightness are jittered so that distinct images are
    NOT dHash-identical, while tiny-noise copies still match.
    """
    w, h = size
    base = int(rng.integers(20, 40)) if cls == 0 else int(rng.integers(215, 240))
    arr = np.full((h, w), base, dtype=np.int16)
    arr += rng.integers(-noise, noise + 1, size=(h, w))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    d = ImageDraw.Draw(img)
    cx = int(w / 2 + rng.integers(-w // 6, w // 6 + 1))
    cy = int(h / 2 + rng.integers(-h // 6, h // 6 + 1))
    if cls == 0:
        r = int(min(w, h) * rng.uniform(0.15, 0.22))
        fill = int(rng.integers(225, 245))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    else:
        s = int(min(w, h) * rng.uniform(0.30, 0.38))
        fill = int(rng.integers(8, 30))
        d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=fill)
    return img.convert("RGB")


def _near_dup(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """A copy with tiny pixel noise — dHash should barely move."""
    arr = np.asarray(img).astype(np.int16)
    arr += rng.integers(-4, 5, size=arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _write(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def synth_dataset(out_dir: str, seed: int = 7) -> Dict:
    """Create the demo dataset. Returns a manifest describing planted issues."""
    rng = np.random.default_rng(seed)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    train = os.path.join(out_dir, "train")
    test = os.path.join(out_dir, "test")

    labeled: List[Tuple[str, str]] = []  # (relpath, label)

    def add(img: Image.Image, split: str, name: str, label: str) -> str:
        rel = os.path.join(split, name)
        _write(img, os.path.join(out_dir, rel))
        labeled.append((rel, label))
        return rel

    # --- clean majority class (true-A, label A): 96 images -------------------
    for i in range(64):
        add(_pattern(rng, 0, SIZE), "train", f"a_clean_{i:02d}.png", "class_a")
    for i in range(32):
        add(_pattern(rng, 0, SIZE), "test", f"a_clean_{i:02d}.png", "class_a")

    # --- mislabeled: 2x true-A labeled B, 2x true-B labeled A -----------------
    for i in range(2):  # prototypical, low noise -> confidently catchable
        add(_pattern(rng, 0, SIZE, noise=4), "train", f"a_mis_{i}.png", "class_b")
    for i in range(2):
        add(_pattern(rng, 1, SIZE, noise=4), "train", f"b_mis_{i}.png", "class_a")

    # --- clean minority (true-B, label B): 3 images -> imbalance --------------
    for i in range(3):
        add(_pattern(rng, 1, SIZE), "test", f"b_clean_{i:02d}.png", "class_b")

    # --- near-duplicates within train: base + 3 near copies -------------------
    base = _pattern(rng, 0, SIZE)
    add(base, "train", "dup_base.png", "class_a")
    for i in range(3):
        add(_near_dup(base, rng), "train", f"dup_near_{i}.png", "class_a")

    # --- cross-split leaks: 2 images in test duplicating train images ---------
    leak_src = _pattern(rng, 0, SIZE)
    add(leak_src, "train", "leak_src.png", "class_a")
    add(_near_dup(leak_src, rng), "test", "leak_near.png", "class_a")  # leak 1
    add(base.copy(), "test", "leak_exact.png", "class_a")              # leak 2

    # --- odd geometry: one 64x64 among 96x96 ----------------------------------
    add(_pattern(rng, 0, ODD_SIZE), "train", "odd_size.png", "class_a")

    # --- degenerate (unlabeled): all-black + all-white ------------------------
    _write(Image.new("RGB", SIZE, (0, 0, 0)), os.path.join(train, "degen_black.png"))
    _write(Image.new("RGB", SIZE, (255, 255, 255)), os.path.join(train, "degen_white.png"))

    # --- corrupt (unlabeled): random bytes ------------------------------------
    corrupt_files = []
    for i, name in enumerate(["corrupt_0.png", "corrupt_1.jpg", "corrupt_2.png"]):
        split = "train" if i < 2 else "test"
        p = os.path.join(out_dir, split, name)
        with open(p, "wb") as fh:
            fh.write(rng.integers(0, 256, size=512).astype(np.uint8).tobytes())
        corrupt_files.append(os.path.join(split, name))

    # --- labels.csv ------------------------------------------------------------
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filepath", "label"])
        w.writerows(sorted(labeled))

    n_a = sum(1 for _, lb in labeled if lb == "class_a")
    n_b = sum(1 for _, lb in labeled if lb == "class_b")
    total = n_a + n_b
    return {
        "root": os.path.abspath(out_dir),
        "labels_csv": os.path.abspath(os.path.join(out_dir, "labels.csv")),
        "planted": {
            "corrupt": 3,
            "corrupt_files": corrupt_files,
            "near_duplicate_images": 5,   # 3 within-train + 2 cross-split
            "cross_split_leaks": 2,       # leak_near.png, leak_exact.png (in test)
            "mislabeled": 4,              # a_mis_0/1, b_mis_0/1
            "mislabeled_files": [os.path.join("train", f"a_mis_{i}.png") for i in range(2)]
            + [os.path.join("train", f"b_mis_{i}.png") for i in range(2)],
            "degenerate": 2,
            "class_a": n_a,
            "class_b": n_b,
            "minority_ratio": round(n_b / total, 4),
        },
    }
