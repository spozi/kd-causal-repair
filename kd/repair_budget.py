"""Parameter-normalized channel budgets and held-out behavioral preservation.

Budgets count the *union* of editable parameter coordinates, not parameters that
eventually change numerically and not the sum of independent channel costs.
BatchNorm buffers are deliberately excluded from the model-size denominator.
"""

from __future__ import annotations

import math
import hashlib
import json
from numbers import Integral, Real

import numpy as np

from .neuron_surgery import channel_parameter_masks


def parameter_support_sha256(masks) -> str:
    """Fingerprint actual editable coordinates, independent of channel ordering.

    Include names and shapes so equal counts/different coordinates cannot alias.
    Buffers and tensor values are deliberately not part of an editable support.
    """
    digest = hashlib.sha256()
    for name, mask in sorted(masks.items()):
        values = mask.detach().cpu().numpy()
        if values.dtype != np.bool_:
            raise ValueError("Editable support masks must be boolean")
        digest.update(json.dumps([name, list(values.shape)], separators=(",", ":")).encode())
        digest.update(b"\n")
        digest.update(np.packbits(values.reshape(-1), bitorder="little").tobytes())
    return digest.hexdigest()


def _validated_fractions(values) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("parameter_budget_fractions must be a list or tuple")
    fractions = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Parameter budget fractions must be finite numbers, not booleans")
        fraction = float(value)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("Parameter budget fractions must satisfy 0 < fraction <= 1")
        if fractions and fraction <= fractions[-1]:
            raise ValueError("Parameter budget fractions must be strictly increasing and unique")
        fractions.append(fraction)
    return tuple(fractions)


def validate_budget_options(parameter_budget_fractions, max_preservation_kl) -> None:
    """Validate opt-in normalized budgets without changing legacy empty budgets.

    An explicit nonnegative validation KL ceiling is mandatory in normalized
    mode. Fixed-channel legacy mode may also opt in to this preservation guard.
    """
    fractions = _validated_fractions(parameter_budget_fractions)
    if max_preservation_kl is None:
        if fractions:
            raise ValueError("Parameter-normalized repair requires an explicit max_preservation_kl")
        return
    if (isinstance(max_preservation_kl, bool) or not isinstance(max_preservation_kl, Real)
            or not math.isfinite(float(max_preservation_kl)) or max_preservation_kl < 0):
        raise ValueError("max_preservation_kl must be finite and nonnegative, not a boolean")


def plan_parameter_budgets(model, ranking, fractions) -> dict:
    """Choose the longest ranked channel prefix below each parameter cap.

    Costs are obtained from the same mask implementation used by repair
    training. Monotone prefix costs are cached and binary-searched, avoiding a
    full-model mask allocation for every channel in a large ResNet ranking.
    Requested caps resolving to one prefix share a single candidate. A capped
    prefix is never extended using channels outside the supplied ranking.
    """
    fractions = _validated_fractions(fractions)
    total = sum(parameter.numel() for parameter in model.parameters())
    if total <= 0:
        raise ValueError("A repair teacher must contain parameters")
    channels = []
    seen = set()
    for row in ranking:
        if not isinstance(row, dict) or not isinstance(row.get("stage"), str):
            raise ValueError("Each ranked channel requires a string stage and integer channel")
        channel = row.get("channel")
        if isinstance(channel, bool) or not isinstance(channel, Integral):
            raise ValueError("Ranked channel indices must be integers, not booleans")
        key = (row["stage"], int(channel))
        if key in seen:
            raise ValueError(f"Duplicate ranked repair channel {key[0]}:{key[1]}")
        seen.add(key)
        channels.append({"stage": key[0], "channel": key[1]})

    counts = {0: 0}

    def prefix_count(size):
        if size not in counts:
            masks = channel_parameter_masks(model, channels[:size])
            counts[size] = sum(int(mask.count_nonzero().item()) for mask in masks.values())
        return counts[size]

    # Validate even low-ranked channels, not just the prefixes reached by caps.
    available = prefix_count(len(channels))
    requests, candidates = [], []
    by_budget = {}
    for fraction in fractions:
        cap = math.floor(total * fraction)
        lower, upper = 0, len(channels)
        while lower < upper:
            middle = (lower + upper + 1) // 2
            if prefix_count(middle) <= cap:
                lower = middle
            else:
                upper = middle - 1
        budget = lower
        editable = prefix_count(budget)
        request = {"requested_fraction": fraction, "coordinate_cap": cap,
                   "achieved_fraction": editable / total,
                   "editable_coordinates": editable, "channel_budget": budget,
                   "status": "infeasible" if budget == 0 else "unique",
                   "saturated": bool(channels) and budget == len(channels)}
        request["shortfall_reason"] = (
            "empty_ranked_pool" if not channels else
            "first_channel_exceeds_cap" if budget == 0 else
            "ranked_pool_exhausted" if budget == len(channels) and editable < cap else
            "next_channel_exceeds_cap" if editable < cap else None)
        if budget == 0:
            requests.append(request)
            continue
        if budget in by_budget:
            request.update(status="duplicate", duplicate_of=budget)
            by_budget[budget]["parameter_budget"]["requested_fractions"].append(fraction)
        else:
            selected = channels[:budget]
            stages = {}
            for row in selected:
                stage = row["stage"]
                if stage not in stages:
                    stages[stage] = {"channels": 0,
                                     "available_channels": int(model.feature_channels[stage])}
                stages[stage]["channels"] += 1
            for entry in stages.values():
                entry["channel_fraction"] = entry["channels"] / entry["available_channels"]
            candidate = {"budget": budget, "channels": selected,
                         "parameter_budget": {"requested_fraction": fraction,
                             "requested_fractions": [fraction],
                             "editable_coordinates": editable, "total_parameters": total,
                             "achieved_fraction": editable / total, "per_stage": stages,
                             "support_sha256": parameter_support_sha256(
                                 channel_parameter_masks(model, selected))}}
            candidates.append(candidate)
            by_budget[budget] = candidate
        requests.append(request)
    return {"total_parameters": total, "ranked_channels": len(channels),
            "available_editable_coordinates": available,
            "requests": requests, "candidates": candidates}


def preservation_metrics(labels, original_probs, candidate_probs) -> dict:
    """Measure held-out T=1 prediction drift relative to the original teacher.

    KL is forward KL(original || candidate), in nats, averaged across images.
    Float64 arithmetic and a float64-tiny floor on candidate probabilities keep
    exact zeros safe. Input rows must already be normalized; re-normalization
    only removes permitted floating-point summation error, not arbitrary logits.
    """
    labels = np.asarray(labels)
    original = np.asarray(original_probs, dtype=np.float64)
    candidate = np.asarray(candidate_probs, dtype=np.float64)
    if (original.ndim != 2 or original.shape != candidate.shape
            or original.shape[0] == 0 or original.shape[1] == 0):
        raise ValueError("Preservation probabilities must be matching nonempty [images, classes] arrays")
    if (labels.ndim != 1 or len(labels) != len(original)
            or not np.issubdtype(labels.dtype, np.integer)
            or (labels < 0).any() or (labels >= original.shape[1]).any()):
        raise ValueError("Preservation labels must be matching integer class indices")
    for values in (original, candidate):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Preservation probabilities must be finite and nonnegative")
        if not np.allclose(values.sum(axis=1), 1, rtol=1e-5, atol=1e-7):
            raise ValueError("Preservation probability rows must be normalized")
    original = original / original.sum(axis=1, keepdims=True)
    candidate = candidate / candidate.sum(axis=1, keepdims=True)
    positive = original > 0
    terms = np.zeros_like(original)
    terms[positive] = original[positive] * (
        np.log(original[positive])
        - np.log(np.maximum(candidate[positive], np.finfo(np.float64).tiny)))
    original_predictions = original.argmax(axis=1)
    candidate_predictions = candidate.argmax(axis=1)
    return {"mean_forward_kl": max(0.0, float(terms.sum(axis=1).mean())),
            "samples": len(labels),
            "prediction_flip_rate": float(np.mean(original_predictions != candidate_predictions)),
            "original_accuracy": float(np.mean(original_predictions == labels)),
            "candidate_accuracy": float(np.mean(candidate_predictions == labels))}
