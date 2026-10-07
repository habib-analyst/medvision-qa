"""Shared fixtures/helpers for the medvision-qa test suite (no network)."""
import os

import numpy as np
from PIL import Image


def textured_image(path, seed, size=(48, 48)):
    """A high-entropy image with a dHash far from other seeds' images."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def solid_image(path, color, size=(48, 48), mode="RGB"):
    Image.new(mode, size, color).save(path)
    return path


def write_csv(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filepath", "label"])
        w.writerows(rows)


def findings_by_check(report):
    out = {}
    for f in report["findings"]:
        out.setdefault(f["check"], []).append(f)
    return out


def rel(p):
    return p.replace(os.sep, "/")
