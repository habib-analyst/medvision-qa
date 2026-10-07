"""Label-file analysis: class imbalance stats and a lightweight,
honest label-noise heuristic.

The noise heuristic is *confident-learning-lite*: thumbnails (32x32 gray)
are classified with cross-validated logistic regression; a sample is
flagged only when the model predicts a *different* class with high
confidence. It is a triage signal, not ground truth — every flag needs a
human look.
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

MINORITY_WARN_RATIO = 0.05  # warn when a class is < 5% of labeled data


@dataclass
class LabelRow:
    filepath: str   # as written in the CSV
    label: str
    lineno: int


def load_label_rows(csv_path: str) -> List[LabelRow]:
    rows: List[LabelRow] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("labels CSV is empty")
        cols = {c.strip().lower() for c in reader.fieldnames}
        if "filepath" not in cols or "label" not in cols:
            raise ValueError(
                f"labels CSV needs 'filepath' and 'label' columns, "
                f"found: {reader.fieldnames}"
            )
        # map back to the actual header spellings
        fp_col = next(c for c in reader.fieldnames if c.strip().lower() == "filepath")
        lb_col = next(c for c in reader.fieldnames if c.strip().lower() == "label")
        for i, row in enumerate(reader, start=2):
            fp = (row.get(fp_col) or "").strip()
            lb = (row.get(lb_col) or "").strip()
            if not fp:
                continue
            rows.append(LabelRow(filepath=fp, label=lb, lineno=i))
    return rows


def resolve_label_path(dataset_root: str, filepath: str) -> str:
    if os.path.isabs(filepath):
        return filepath
    return os.path.normpath(os.path.join(dataset_root, filepath))


def class_distribution(rows: List[LabelRow]) -> Tuple[Counter, int]:
    counts = Counter(r.label for r in rows)
    return counts, len(rows)


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


def _thumbnail(path: str, size: int = 32) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as im:
            im.load()
            g = im.convert("L").resize((size, size), Image.LANCZOS)
            return np.asarray(g, dtype=np.float32).flatten() / 255.0
    except Exception:  # noqa: BLE001 - unreadable here, reported elsewhere
        return None


def flag_label_noise(
    rows: List[LabelRow],
    dataset_root: str,
    thumb_size: int = 16,
    cv_folds: int = 3,
    confidence: float = 0.80,
    random_state: int = 42,
) -> Tuple[List[dict], dict]:
    """Cross-validated predicted-probability flagging.

    Thumbnails are classified with a regularized, class-balanced logistic
    regression under stratified K-fold CV (``class_weight='balanced'`` so a
    tiny minority class still gets a fair decision boundary — the very
    imbalance this tool reports must not blind the noise check itself).

    Returns (flags, info). Each flag: filepath, label, predicted_label,
    confidence. Samples the model gets *wrong-but-confident* are the ones
    most worth a human re-look.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    # unique, loadable files only
    seen: Dict[str, str] = {}
    for r in rows:
        seen.setdefault(resolve_label_path(dataset_root, r.filepath), r.label)
    paths = sorted(seen)
    X_list, y_list, kept = [], [], []
    for p in paths:
        t = _thumbnail(p, thumb_size)
        if t is not None:
            X_list.append(t)
            y_list.append(seen[p])
            kept.append(p)
    info = {"n_labeled_files": len(paths), "n_usable": len(kept)}
    if len(kept) < 6:
        info["skipped"] = "too few usable labeled images for cross-validation"
        return [], info

    labels = sorted(set(y_list))
    counts = Counter(y_list)
    info["class_counts"] = {l: counts[l] for l in labels}
    folds = min(cv_folds, min(counts.values()))
    if folds < 2:
        info["skipped"] = "a class has < 2 samples; cannot cross-validate"
        return [], info
    info["cv_folds"] = folds

    X = np.stack(X_list)
    y = np.array([labels.index(v) for v in y_list])
    clf = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced",
                             random_state=random_state)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)

    flags = []
    for p, true_i, pred_i, c in zip(kept, y, pred, conf):
        if pred_i != true_i and float(c) >= confidence:
            flags.append({
                "filepath": os.path.relpath(p, dataset_root)
                if p.startswith(dataset_root) else p,
                "label": labels[int(true_i)],
                "predicted_label": labels[int(pred_i)],
                "confidence": round(float(c), 3),
            })
    info["n_flags"] = len(flags)
    return flags, info


def analyze_labels(
    dataset_root: str,
    csv_path: str,
    minority_warn_ratio: float = MINORITY_WARN_RATIO,
    noise_confidence: float = 0.80,
    run_noise_check: bool = True,
) -> dict:
    """Full label-file audit. Returns {'rows','counts','total','findings','noise'}."""
    rows = load_label_rows(csv_path)
    findings: List[dict] = []
    counts, total = class_distribution(rows)

    # duplicates in the CSV itself
    dup_paths = [p for p, c in Counter(r.filepath for r in rows).items() if c > 1]
    if dup_paths:
        findings.append(_finding(
            "warning", "duplicate_label_entries",
            f"{len(dup_paths)} filepath(s) appear more than once in labels.csv.",
            dup_paths))

    # missing / unreadable
    missing, unreadable = [], []
    for r in rows:
        p = resolve_label_path(dataset_root, r.filepath)
        if not os.path.isfile(p):
            missing.append(r.filepath)
        elif _thumbnail(p) is None:
            unreadable.append(r.filepath)
    # dedupe (a corrupt file listed twice should count once)
    missing, unreadable = sorted(set(missing)), sorted(set(unreadable))
    if missing:
        findings.append(_finding(
            "error", "missing_labeled_files",
            f"{len(missing)} file(s) listed in labels.csv do not exist on disk.",
            missing))
    if unreadable:
        findings.append(_finding(
            "warning", "unreadable_labeled_files",
            f"{len(unreadable)} labeled file(s) could not be decoded "
            "(counted as corrupt by the image checks).",
            unreadable))

    # imbalance
    dist_info = {l: {"count": c, "ratio": round(c / total, 4)} for l, c in counts.items()}
    for label in sorted(counts):
        ratio = counts[label] / total
        if ratio < minority_warn_ratio:
            findings.append(_finding(
                "warning", "class_imbalance",
                f"class '{label}': {counts[label]}/{total} "
                f"({ratio:.1%}) < {minority_warn_ratio:.0%} — severe imbalance; "
                "consider stratified sampling, class weights, or more data.",
                [],
                {"class": label, "count": counts[label],
                 "ratio": round(ratio, 4), "threshold": minority_warn_ratio}))
    findings.append(_finding(
        "info", "class_distribution",
        "Class distribution: " + ", ".join(
            f"{l}={counts[l]} ({counts[l]/total:.1%})" for l in sorted(counts)),
        [], {"distribution": dist_info}))

    # label noise heuristic
    noise = {"flags": [], "info": {"skipped": "disabled"}}
    if run_noise_check:
        flags, info = flag_label_noise(rows, dataset_root, confidence=noise_confidence)
        noise = {"flags": flags, "info": info}
        if flags:
            findings.append(_finding(
                "warning", "label_noise_suspects",
                f"{len(flags)} sample(s) where a cross-validated model confidently "
                f"predicts a DIFFERENT class than the label — review these by hand.",
                [f["filepath"] for f in flags],
                {"flags": flags,
                 "note": "heuristic triage only; a flag is not proof of mislabeling."}))

    return {"rows": rows, "counts": dict(counts), "total": total,
            "findings": findings, "noise": noise}
