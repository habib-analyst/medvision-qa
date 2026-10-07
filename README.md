# medvision-qa

[![CI](https://github.com/habib-analyst/medvision-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/habib-analyst/medvision-qa/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/habib-analyst/medvision-qa/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/habib-analyst/medvision-qa)

A medical image dataset quality auditor. Run it **before** training — it catches the silent dataset bugs that ruin medical-imaging experiments: corrupt files, degenerate scans, near-duplicate images leaking across train/test splits, geometry inconsistencies, severe class imbalance, and likely mislabeled samples.

## The problem

Medical image datasets are assembled from scanners, PACS exports, and manual annotation — and they rot quietly:

- A few **corrupt files** crash a `DataLoader` at 3 AM, or worse, get silently skipped, shifting your effective sample count.
- **Near-duplicate images** (same patient, adjacent slice, re-exported study) end up in both `train/` and `test/` → test metrics are inflated and nobody knows.
- **All-black / all-white scans** from failed acquisitions train the model on nothing.
- **Severe class imbalance** (a rare pathology at 4% of the data) goes unnoticed until the model predicts only the majority class.
- **Label noise** — a tired annotator's misclick — poisons the loss for every epoch.

`medvision-qa` audits the dataset folder and produces a Markdown + JSON report with severity levels and a 0–100 health score. Offline, CPU-only, three light dependencies.

## Architecture

```
                        ┌─────────────────────┐
                        │    medvision-qa     │
                        │   audit <dataset>   │
                        └─────────┬───────────┘
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
     ┌────────────────┐  ┌───────────────┐  ┌──────────────────┐
     │   checks.py    │  │   labels.py   │  │    report.py     │
     │ per-image      │  │ CSV analysis  │  │ Markdown + JSON  │
     │                │  │               │  │ health score     │
     │ • corrupt file │  │ • class counts│  │                  │
     │   detection    │  │ • <5% minority│  │ 100 − Σ severity │
     │ • degenerate   │  │   warnings    │  │ penalties        │
     │   (black/white │  │ • missing /   │  │ (error 15,       │
     │   /constant)   │  │   unreadable  │  │  warning 5,      │
     │ • dHash near-  │  │ • label-noise │  │  info 1)         │
     │   duplicates   │  │   suspects *  │  └──────────────────┘
     │ • cross-split  │  └───────────────┘
     │   leakage      │           ▲
     │ • geometry/    │           │
     │   mode drift   │  ┌────────┴────────┐
     └────────────────┘  │    synth.py     │
                         │ seeded dataset  │
                         │ generator with  │
                         │ planted issues  │
                         │ (for demo/tests)│
                         └─────────────────┘

  * confident-learning-lite: 3-fold CV logistic regression on 16×16
    thumbnails (class-balanced); flags samples the model predicts as a
    DIFFERENT class with ≥80% confidence. Triage signal, not ground truth.
```

Split membership is inferred from top-level directory names (`train/`, `test/`, `val/`).

## Quickstart (< 5 min)

```bash
pip install -r requirements.txt

# 1. See it work on a synthetic dataset with planted issues (~5 s)
python -m medvisionqa demo

# 2. Audit your own dataset (labels.csv with filepath,label columns)
medvision-qa audit ./my_dataset --labels labels.csv --out report.md --json report.json

# 3. Just generate the synthetic dataset
medvision-qa synth --out ./demo_data
```

Install as a CLI via pip: `pip install .` exposes the `medvision-qa` command.

## Real example output

`python -m medvisionqa demo` synthesizes a dataset with **planted** issues (3 corrupt files, 5 near-duplicates incl. 2 leaking across train/test, 4.5% minority class, 4 mislabeled samples, 2 degenerate scans) and audits it. Actual output:

```
1) synthesized demo dataset -> /tmp/mvqa_readme
2) audit complete -> /tmp/mvqa_readme/audit_report.md  (health score 34/100)

FINDINGS
SEV   CHECK                            FINDING
----  -------------------------------- ------------------------------------------------------------
ERR   corrupt_files                    3 corrupt/unreadable image file(s) — they will crash or silently poison a data loader.
        files: test/corrupt_2.png, train/corrupt_0.png, train/corrupt_1.jpg
ERR   cross_split_leakage              DATA LEAKAGE: 5 near-duplicate image(s) appear in multiple splits ['test', 'train'] (ma...
        files: test/leak_exact.png, train/dup_base.png, train/dup_near_0.png, train/dup_near_1.png (+1 more)
ERR   cross_split_leakage              DATA LEAKAGE: 2 near-duplicate image(s) appear in multiple splits ['test', 'train'] (ma...
        files: test/leak_near.png, train/leak_src.png
WARN  class_imbalance                  class 'class_b': 5/111 (4.5%) < 5% — severe imbalance; consider stratified sampling, cl...
WARN  degenerate_images                2 degenerate image(s) with no usable signal (1 all_black, 1 all_white).
        files: train/degen_black.png, train/degen_white.png
WARN  geometry_mode_inconsistency      1 image(s) differ from the majority geometry/mode (96x96, mode RGB) — resize/normalize ...
        files: train/odd_size.png
WARN  label_noise_suspects             4 sample(s) where a cross-validated model confidently predicts a DIFFERENT class than t...
        files: train/a_mis_0.png, train/a_mis_1.png, train/b_mis_0.png, train/b_mis_1.png
INFO  class_distribution               Class distribution: class_a=106 (95.5%), class_b=5 (4.5%)

PLANTED vs DETECTED
  [PASS] corrupt files: planted=3 detected=3
  [PASS] near-duplicate images: planted=5 detected=7
  [PASS] cross-split leaks (in test/): planted=2 detected=2
  [PASS] mislabeled samples flagged: planted=4 detected=4
  [PASS] class imbalance <5% warning: planted=1 detected=1
  [PASS] degenerate images: planted=2 detected=2
  [PASS] planted mislabels among flags: planted=4 detected=4

demo OK: every planted issue class was caught.
```

(The near-dup count reads 7 because the two "base" images each group is anchored on are listed too — all 5 planted near-duplicates are inside those groups.)

## CLI reference

```
medvision-qa audit ./dataset --labels labels.csv --out report.md [--json report.json]
           [--hash-threshold 5] [--noise-confidence 0.80] [--no-noise-check] [--quiet]
medvision-qa synth --out ./demo_data [--seed 7]
medvision-qa demo [--out DIR] [--seed 7]
```

- `--hash-threshold`: max dHash Hamming distance to call two images near-duplicates (default 5 of 64 bits).
- `--noise-confidence`: min model confidence to flag a label-noise suspect (default 0.80).
- Label paths in the CSV are resolved relative to the dataset root (absolute paths also accepted).

## Honest limitations

- **dHash is perceptual, not semantic.** It catches re-exports, adjacent slices, and re-compressed copies — it will *miss* two genuinely different scans of the same pathology, and on highly stereotyped acquisitions (same scanner, centered anatomy) distinct images can hash closely. Tune `--hash-threshold` on your data.
- **Label-noise flags are triage, not truth.** The heuristic assumes most labels are right and classes are thumbnail-separable; a confident model can be confidently wrong. Every flag needs a human look before relabeling — the report says this on every run.
- **No DICOM metadata checks** (patient-ID de-duplication, slice-spacing consistency) — file-level only for now; see roadmap.
- The imbalance rule (< 5% minority) is a heuristic default, not a statistical test.

## Roadmap

- [ ] DICOM tag audit: duplicate PatientID across splits, inconsistent spacing/modality
- [ ] Embedding-based semantic duplicate detection (self-supervised features) alongside dHash
- [ ] Full confident-learning joint-distribution noise estimation
- [ ] HTML report with thumbnail grids of flagged samples
- [ ] Pre-commit / CI gate mode (`--fail-on error` exit codes)

## Tests

```bash
python -m pytest tests/ -q   # 28 tests, no network, ~7 s
```

Covers: dHash identical/near-duplicate/different discrimination, corrupt + truncated + degenerate detection, cross-split leakage and within-split duplicate fixtures, imbalance math incl. the exact 5% boundary, missing/unreadable/duplicate label entries, and an end-to-end seeded test asserting every planted issue class is caught (plus a CLI `demo` smoke test).

## Citations

- Zauner, C. *Implementation and Benchmarking of Perceptual Image Hash Functions* (dHash), 2010.
- Northcutt, C. G., Jiang, L., Chuang, I. *Confident Learning: Estimating Uncertainty in Dataset Labels.* JAIR, 2021. (label-noise heuristic inspiration)
- The class-imbalance reporting follows the stratification guidance in most medical-imaging challenge baselines (e.g., MONAI tutorials).

## License

MIT — see [LICENSE](LICENSE). © 2026 Habib Ur Rehman.
