"""Command-line interface: audit | synth | demo."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

from . import __version__
from .report import render_json, render_markdown, run_audit
from .synth import synth_dataset


def _resolve_labels(dataset: str, labels: str | None) -> str | None:
    if labels:
        cand = labels if os.path.isabs(labels) else os.path.join(dataset, labels)
        if os.path.isfile(cand):
            return cand
        if os.path.isfile(labels):  # as given, relative to CWD
            return labels
        raise SystemExit(f"labels file not found: {labels}")
    default = os.path.join(dataset, "labels.csv")
    return default if os.path.isfile(default) else None


def cmd_audit(args: argparse.Namespace) -> int:
    labels_csv = _resolve_labels(args.dataset, args.labels)
    report = run_audit(
        args.dataset,
        labels_csv=labels_csv,
        hash_threshold=args.hash_threshold,
        noise_confidence=args.noise_confidence,
        run_noise_check=not args.no_noise_check,
    )
    md_path = args.out
    text = render_markdown(report, md_path)
    if args.json:
        render_json(report, args.json)
        print(f"JSON report: {args.json}")
    print(f"Markdown report: {md_path}")
    print(f"Health score: {report['health_score']}/100 "
          f"({report['n_errors']} errors, {report['n_warnings']} warnings)")
    if not args.quiet:
        print()
        print(_findings_table(report["findings"]))
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    manifest = synth_dataset(args.out, seed=args.seed)
    p = manifest["planted"]
    print(f"Synthetic dataset written to {manifest['root']}")
    print(f"labels: {manifest['labels_csv']}")
    print(f"planted issues: {p['corrupt']} corrupt, "
          f"{p['near_duplicate_images']} near-dups "
          f"({p['cross_split_leaks']} cross-split leaks), "
          f"{p['mislabeled']} mislabeled, {p['degenerate']} degenerate, "
          f"class imbalance {p['class_a']}/{p['class_b']} "
          f"(minority {p['minority_ratio']:.1%})")
    return 0


def _findings_table(findings: list) -> str:
    icon = {"error": "ERR ", "warning": "WARN", "info": "INFO"}
    lines = [f"{'SEV':4}  {'CHECK':32} FINDING",
             f"{'-'*4}  {'-'*32} {'-'*60}"]
    for f in findings:
        msg = f["message"]
        if len(msg) > 90:
            msg = msg[:87] + "..."
        lines.append(f"{icon.get(f['severity'], '?'):4}  {f['check']:32} {msg}")
        if f["files"]:
            shown = ", ".join(f["files"][:4])
            extra = f" (+{len(f['files'])-4} more)" if len(f["files"]) > 4 else ""
            lines.append(f"        files: {shown}{extra}")
    return "\n".join(lines)


def _checklist(report: dict, planted: dict) -> list:
    """(label, planted_n, detected_n, ok) rows for the demo summary."""
    findings = report["findings"]

    def files_of(*check_names):
        out = []
        for f in findings:
            if f["check"] in check_names:
                out.extend(f["files"])
        return out

    corrupt = files_of("corrupt_files")
    dup_files = files_of("near_duplicates_within_split", "cross_split_leakage")
    leak_test = [fl for f in findings if f["check"] == "cross_split_leakage"
                 for fl in f["files"] if fl.startswith("test" + os.sep)]
    noise = files_of("label_noise_suspects")
    imbalance = any(f["check"] == "class_imbalance" for f in findings)
    degen = files_of("degenerate_images")

    rows = [
        ("corrupt files", planted["corrupt"], len(set(corrupt)),
         len(set(corrupt)) == planted["corrupt"]),
        ("near-duplicate images", planted["near_duplicate_images"], len(set(dup_files)),
         len(set(dup_files)) >= planted["near_duplicate_images"]),
        ("cross-split leaks (in test/)", planted["cross_split_leaks"], len(set(leak_test)),
         len(set(leak_test)) == planted["cross_split_leaks"]),
        ("mislabeled samples flagged", planted["mislabeled"], len(set(noise)),
         len(set(noise)) >= planted["mislabeled"]),
        ("class imbalance <5% warning", 1, 1 if imbalance else 0, imbalance),
        ("degenerate images", planted["degenerate"], len(set(degen)),
         len(set(degen)) == planted["degenerate"]),
    ]
    # also verify the *right* mislabeled files were caught, not just the count
    caught = set(noise)
    want = set(planted["mislabeled_files"])
    rows.append(("planted mislabels among flags", len(want), len(caught & want),
                 want <= caught))
    return rows


def cmd_demo(args: argparse.Namespace) -> int:
    out = args.out or tempfile.mkdtemp(prefix="medvisionqa_demo_")
    manifest = synth_dataset(out, seed=args.seed)
    planted = manifest["planted"]
    print(f"1) synthesized demo dataset -> {out}")
    report = run_audit(out, labels_csv=manifest["labels_csv"])
    md = os.path.join(out, "audit_report.md")
    render_markdown(report, md)
    print(f"2) audit complete -> {md}  (health score {report['health_score']}/100)")
    print()
    print("FINDINGS")
    print(_findings_table(report["findings"]))
    print()
    print("PLANTED vs DETECTED")
    all_ok = True
    for label, n_planted, n_found, ok in _checklist(report, planted):
        mark = "PASS" if ok else "FAIL"
        all_ok &= ok
        print(f"  [{mark}] {label}: planted={n_planted} detected={n_found}")
    print()
    print("demo " + ("OK: every planted issue class was caught."
                     if all_ok else "INCOMPLETE: see FAIL rows above."))
    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="medvision-qa",
        description="Audit medical image datasets for corruption, duplicates, "
                    "leakage, imbalance and label noise.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="audit a dataset directory")
    a.add_argument("dataset", help="dataset root (expects train/ test/ ... subdirs)")
    a.add_argument("--labels", default=None,
                   help="labels CSV (filepath,label); default: <dataset>/labels.csv if present")
    a.add_argument("--out", default="audit_report.md", help="markdown report path")
    a.add_argument("--json", default=None, help="also write a JSON report here")
    a.add_argument("--hash-threshold", type=int, default=5,
                   help="max dHash Hamming distance for near-duplicates")
    a.add_argument("--noise-confidence", type=float, default=0.80,
                   help="min model confidence to flag a label-noise suspect")
    a.add_argument("--no-noise-check", action="store_true",
                   help="skip the label-noise heuristic")
    a.add_argument("--quiet", action="store_true", help="only print report paths")
    a.set_defaults(func=cmd_audit)

    s = sub.add_parser("synth", help="generate a synthetic dataset with planted issues")
    s.add_argument("--out", default="./demo_data", help="output directory")
    s.add_argument("--seed", type=int, default=7, help="random seed")
    s.set_defaults(func=cmd_synth)

    d = sub.add_parser("demo", help="synthesize + audit + print findings table")
    d.add_argument("--out", default=None, help="output dir (default: temp dir)")
    d.add_argument("--seed", type=int, default=7, help="random seed")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
