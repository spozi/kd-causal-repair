"""UrbanCars alignment slices and gaps, matching the author's sign convention.

Reference: facebookresearch/Whac-A-Mole/urbancars_trainers/base_trainer.py,
_eval_split. Gaps are shifted-slice accuracy minus training-mixture ID accuracy.
"""

import numpy as np

from .group_metrics import group_accuracy


def urban_metrics(labels, predictions, nuisance, train_counts):
    labels, predictions, nuisance = np.asarray(labels), np.asarray(predictions), np.asarray(nuisance)
    if nuisance.shape != (len(labels), 2) or np.any((nuisance != 0) & (nuisance != 1)):
        raise ValueError("UrbanCars needs two aligned binary nuisance columns")
    groups = labels * 4 + nuisance[:, 0] * 2 + nuisance[:, 1]
    summary = group_accuracy(labels, predictions, groups, 8)
    values = summary["per_group"]
    slices = {}
    for bg_align in (0, 1):
        for co_align in (0, 1):
            indices = [label * 4 + (label if bg_align else 1 - label) * 2
                       + (label if co_align else 1 - label) for label in (0, 1)]
            scores = [values[group]["accuracy"] for group in indices]
            slices[f"bg_{'present' if bg_align else 'absent'}_coobject_{'present' if co_align else 'absent'}"] = (
                None if any(score is None for score in scores) else float(np.mean(scores)))
    counts = np.asarray(train_counts, dtype=float)
    if counts.shape != (8,) or not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("UrbanCars weighted metrics require positive training counts for all eight groups")
    available = summary["group_metrics_available"]
    id_acc = (sum(slices[f"bg_{'present' if bg else 'absent'}_coobject_{'present' if co else 'absent'}"]
                  * (0.95 if bg else 0.05) * (0.95 if co else 0.05)
                  for bg in (0, 1) for co in (0, 1)) if available else None)
    weighted = float(np.dot(counts / counts.sum(), [row["accuracy"] for row in values])) if available else None
    return {"urbancars_alignment_slices": slices, "urbancars_id_accuracy": id_acc,
            "train_group_weighted_accuracy": weighted,
            "urbancars_background_gap": slices["bg_absent_coobject_present"] - id_acc if available else None,
            "urbancars_coobject_gap": slices["bg_present_coobject_absent"] - id_acc if available else None,
            "urbancars_joint_gap": slices["bg_absent_coobject_absent"] - id_acc if available else None,
            "urbancars_gap_definition": "shifted equal-class slice minus ID accuracy; ID uses independent 0.95/0.95 alignment weights",
            "urbancars_metric_reference": "facebookresearch/Whac-A-Mole/urbancars_trainers/base_trainer.py::_eval_split"}
