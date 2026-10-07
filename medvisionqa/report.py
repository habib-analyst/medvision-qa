"""Audit orchestration + Markdown/JSON report rendering with severity
levels and a 0-100 dataset health score."""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import List

from . import checks, labels

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
SEVERITY_WEIGHT = {"error": 15, "warning": 5, "info": 1}


def health_score(findings: List[dict]) -> int:
    """Start at 100, deduct per finding by severity. Floor at 0."""
    penalty = sum(SEVERITY_WEIGHT.get(f.get("severity", "info"), 1)
                  for f in findings)
    return max(0, 100 - penalty)


def run_audit(dataset_root: str, labels_csv: Optional[str] = None,
              hash_threshold: int = 5, noise_confidence: float = 0.80,
              run_noise_check: bool = True) -> dict:
    """Run image checks (+ label analysis when a CSV is given)."""
    img = checks.audit_images(dataset_root, hash_threshold=hash_threshold)
    label_result = None
    if labels_csv:
        label_result = labels.analyze_labels(
            dataset_root, labels_csv, noise_confidence=noise_confidence,
            run_noise_check=run_noise_check)
    findings = sorted(
        img["findings"] + (label_result["findings"] if label_result else []),
        key=lambda f: (SEVERITY_ORDER.get(f["severity"], 9), f["check"]),
    )
    return {
        "dataset": os.path.abspath(dataset_root),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "image_stats": img["stats"],
        "label_stats": ({k: v for k, v in label_result.items()
                         if k in ("counts", "total", "noise")}
                        if label_result else None),
        "findings": findings,
        "health_score": health_score(findings),
        "n_errors": sum(1 for f in findings if f["severity"] == "error"),
        "n_warnings": sum(1 for f in findings if f["severity"] == "warning"),
        "n_info": sum(1 for f in findings if f["severity"] == "info"),
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
_SEV_ICON = {"error": "🔴", "warning": "🟡", "info": "🔵"}


def _short_files(files: List[str], limit: int = 12) -> str:
    shown = [f"`{f}`" for f in files[:limit]]
    extra = len(files) - limit
    if extra > 0:
        shown.append(f"*…and {extra} more*")
    return "<br>".join(shown) if shown else "—"


def render_markdown(report: dict, out_path: str) -> str:
    L: List[str] = []
    ds = os.path.basename(report["dataset"].rstrip(os.sep)) or report["dataset"]
    L.append(f"# medvision-qa audit — `{ds}`")
    L.append("")
    L.append(f"Generated (UTC): {report['generated_utc']}")
    L.append("")
    L.append("## Dataset health score")
    L.append("")
    score = report["health_score"]
    bar = "█" * (score // 5) + "░" * (20 - score // 5)
    L.append(f"**{score}/100** `{bar}`")
    L.append("")
    L.append(f"- 🔴 {report['n_errors']} error(s) · "
             f"🟡 {report['n_warnings']} warning(s) · "
             f"🔵 {report['n_info']} info")
    L.append("")
    L.append("## Summary")
    L.append("")
    st = report["image_stats"]
    L.append(f"- Images scanned: **{st['n_files']}** "
             f"({st['n_ok']} ok, {st['n_corrupt']} corrupt, "
             f"{st['n_degenerate']} degenerate)")
    L.append(f"- Splits detected: {', '.join(st['splits'])}")
    if report["label_stats"]:
        ls = report["label_stats"]
        L.append(f"- Labeled samples: **{ls['total']}** "
                 f"({', '.join(f'{k}={v}' for k, v in sorted(ls['counts'].items()))})")
        ni = ls["noise"]["info"]
        if ls["noise"]["flags"]:
            L.append(f"- Label-noise suspects flagged: **{len(ls['noise']['flags'])}** "
                     f"(cross-validated, {ni.get('cv_folds', '?')}-fold)")
    L.append("")
    L.append("## Findings")
    L.append("")
    if not report["findings"]:
        L.append("No issues found. 🎉")
    else:
        L.append("| Severity | Check | Finding | Files |")
        L.append("|---|---|---|---|")
        for f in report["findings"]:
            icon = _SEV_ICON.get(f["severity"], "⚪")
            L.append(f"| {icon} {f['severity']} | `{f['check']}` | "
                     f"{f['message']} | {_short_files(f['files'])} |")
        L.append("")
        L.append("### Flagged label-noise samples")
        L.append("")
        flags = (report["label_stats"] or {}).get("noise", {}).get("flags", [])
        if flags:
            L.append("| File | Labeled as | Model predicts | Confidence |")
            L.append("|---|---|---|---|")
            for fl in flags:
                L.append(f"| `{fl['filepath']}` | {fl['label']} | "
                         f"{fl['predicted_label']} | {fl['confidence']} |")
            L.append("")
            L.append("> Heuristic triage only — every flag needs human review "
                     "before relabeling.")
        else:
            L.append("None flagged.")
    L.append("")
    text = "\n".join(L)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def render_json(report: dict, out_path: str) -> None:
    # ImageRecord objects are not JSON-serializable; the report dict itself
    # only carries plain data (records stay out of it by design).
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
