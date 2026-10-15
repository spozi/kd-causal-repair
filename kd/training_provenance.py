"""Checkpoint-bound identities for recorded training/validation provenance.

These identities cover the metadata recorded by the data adapter, not an
independent scan or content hash of every dataset example.
"""

import hashlib
import json
from pathlib import Path

import torch

from .checkpoints import fingerprint


def training_provenance(value: dict) -> dict:
    """Normalize recorded identity, excluding evaluation-only loader sizes."""
    if not isinstance(value, dict):
        raise ValueError("Training data provenance must be a recorded mapping")
    result = json.loads(json.dumps(value, allow_nan=False))
    if not isinstance(result.get("source"), str) or not result["source"]:
        raise ValueError("Training data provenance requires a dataset source")
    sizes = result.get("split_sizes")
    if not isinstance(sizes, dict) or any(
            isinstance(sizes.get(name), bool) or not isinstance(sizes.get(name), int)
            or sizes[name] < 1 for name in ("train", "val")):
        raise ValueError("Training data provenance requires train/validation split sizes")
    sizes.pop("test", None)
    sizes.pop("confirmation", None)
    if result["source"] in {"colored_mnist", "waterbirds", "multiceleba", "urbancars"}:
        for name in list(sizes):
            if name.startswith("test_"):
                sizes.pop(name)
        result.pop("evaluation_only", None)
    return result


def _digest(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def bind_training_data(value: dict) -> dict:
    """Return a self-consistent recorded identity to persist inside checkpoints."""
    provenance = training_provenance(value)
    return {"format_version": 1, "provenance": provenance,
            "sha256": _digest(provenance)}


def validate_training_data(state: dict, current: dict) -> None:
    """Reject missing, corrupt, or changed provenance before adopting resume state."""
    recorded = state.get("training_data_identity")
    if not isinstance(recorded, dict) or set(recorded) != {
            "format_version", "provenance", "sha256"} or type(recorded["format_version"]) is not int \
            or recorded["format_version"] != 1:
        raise ValueError(
            "Resume requires checkpoint-bound training data provenance; legacy "
            "checkpoints cannot be verified from an unbound adjacent data.json")
    normalized = training_provenance(recorded["provenance"])
    if recorded["provenance"] != normalized or recorded["sha256"] != _digest(normalized):
        raise ValueError("Checkpoint-bound training data provenance is inconsistent")
    if normalized != training_provenance(current):
        raise ValueError("Training/validation data provenance changed since the saved epoch")


def load_training_checkpoint(path: str | Path, expected: dict, current: dict) -> dict:
    """Read/validate a checkpoint without mutating a model, optimizer, or RNG."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"Incompatible checkpoint {path}: {key} does not match "
                f"({state.get(key)!r} != {value!r})")
    validate_training_data(state, current)
    return state


def load_selected_checkpoint(path: str | Path, expected: dict, current: dict,
                             training_state: dict) -> dict:
    """Validate the selected artifact against its committed training checkpoint.

    ``last.pt`` is the selection anchor, including when resuming from ``best.pt``.
    A best-save/last-save interruption fails closed: a self-consistent replacement
    best is not evidence that the previous committed epoch selected it.
    """
    recorded = training_state.get("selected_best_checkpoint_sha256")
    if (not isinstance(recorded, str) or len(recorded) != 64
            or any(character not in "0123456789abcdef" for character in recorded)):
        raise ValueError(
            "Selected best checkpoint requires a committed last.pt SHA256 binding; "
            "legacy unsealed checkpoints cannot be safely resumed")
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Selected best checkpoint is missing or a symlink: {path}")
    state = load_training_checkpoint(path, expected, current)
    if fingerprint(path) != recorded:
        raise ValueError(
            "Selected best checkpoint differs from committed last.pt; the best/last "
            "pair may have changed or saving was interrupted between commits. "
            "Restore the matching checkpoint pair or use a fresh run; do not rebind it")
    if (type(state.get("epoch")) is not int or state["epoch"] < 0
            or type(training_state.get("epoch")) is not int
            or state["epoch"] > training_state["epoch"]):
        raise ValueError("Selected best checkpoint epoch exceeds its committed training epoch")
    return state
