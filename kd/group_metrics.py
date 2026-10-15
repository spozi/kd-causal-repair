"""Fail-closed group metadata and metrics; annotations never enter model inputs."""

from __future__ import annotations

import numpy as np
from torch.utils.data import SequentialSampler, Subset


def dataset_metadata(dataset, name):
    if isinstance(dataset, Subset):
        value = dataset_metadata(dataset.dataset, name)
        return None if value is None else np.asarray(value)[np.asarray(dataset.indices, dtype=np.int64)]
    value = getattr(dataset, name, None)
    return None if value is None else np.asarray(value)


def group_count(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return getattr(dataset, "num_groups", None)


def loader_groups(loader):
    groups = dataset_metadata(loader.dataset, "group_ids")
    if groups is None:
        return None
    if (not isinstance(loader.sampler, SequentialSampler) or loader.drop_last
            or len(groups) != len(loader.dataset)):
        raise ValueError("Group evaluation requires complete sequential dataset order")
    return groups


def group_accuracy(labels, predictions, groups, required_groups):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    groups = np.asarray(groups)
    if (labels.ndim != 1 or predictions.shape != labels.shape or groups.shape != labels.shape
            or not len(labels) or groups.dtype.kind not in "iu"
            or type(required_groups) is not int or required_groups < 1
            or np.any(groups < 0) or np.any(groups >= required_groups)):
        raise ValueError("Invalid aligned group predictions or group vocabulary")
    counts = np.bincount(groups, minlength=required_groups)
    correct = np.bincount(groups, weights=(predictions == labels), minlength=required_groups).astype(int)
    values = [float(correct[g] / counts[g]) if counts[g] else None for g in range(required_groups)]
    missing = [g for g in range(required_groups) if not counts[g]]
    return {"required_groups": required_groups,
            "group_counts": counts.tolist(), "group_correct": correct.tolist(),
            "per_group": [{"group": g, "samples": int(counts[g]), "correct": int(correct[g]),
                           "accuracy": values[g]} for g in range(required_groups)],
            "missing_groups": missing,
            "worst_group_accuracy": None if missing else min(values),
            "group_balanced_accuracy": None if missing else float(np.mean(values)),
            "group_metrics_available": not missing}


def add_group_metrics(result, labels, probabilities, loader):
    groups = loader_groups(loader)
    if groups is None:
        return result
    expected = dataset_metadata(loader.dataset, "targets")
    if expected is None or not np.array_equal(expected, labels):
        raise ValueError("Prediction labels do not align with the recorded group metadata")
    result.update(group_accuracy(labels, np.asarray(probabilities).argmax(1), groups, group_count(loader.dataset)))
    return result


def selection_key(metrics, metric):
    value = metrics.get(metric)
    if value is None or not np.isfinite(value):
        raise ValueError(f"Selection metric {metric} is unavailable; check required group coverage")
    if metric == "worst_group_accuracy":
        return (float(value), float(metrics["accuracy"]))
    return (float(value),)
