"""medvision-qa: a medical image dataset quality auditor.

Catch corrupt files, degenerate images, near-duplicate leakage across
train/test splits, geometry inconsistencies, class imbalance, and likely
mislabeled samples *before* they silently ruin a training run.
"""

from .checks import audit_images
from .labels import analyze_labels
from .report import health_score, render_json, render_markdown, run_audit
from .synth import synth_dataset

__version__ = "0.1.0"
__all__ = [
    "audit_images",
    "analyze_labels",
    "health_score",
    "render_json",
    "render_markdown",
    "run_audit",
    "synth_dataset",
]
