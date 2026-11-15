"""Deterministic, opt-in channel-pool expansion for normalized repair.

This module does not measure channels or use validation/test predictions.  It
combines existing localization and channel-ablation diagnostics into a ranked
pool, leaving support allocation and final repair acceptance to their existing
implementations.
"""

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Integral, Real


SCREENING_POLICIES = ("strict", "screened_then_ranked")


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, not a boolean")
    return float(value)


def _channel_key(row, allowed_channels, label):
    if not isinstance(row, Mapping):
        raise ValueError(f"{label} channel must be a mapping")
    stage, channel = row.get("stage"), row.get("channel")
    if (not isinstance(stage, str) or stage not in allowed_channels
            or isinstance(channel, bool) or not isinstance(channel, Integral)
            or not 0 <= channel < allowed_channels[stage]):
        raise ValueError(f"Invalid {label} channel {stage}:{channel}")
    return stage, int(channel)


def _unique_rows(rows, allowed_channels, label):
    if not isinstance(rows, (list, tuple)):
        raise ValueError(f"{label} channels must be a list or tuple")
    result = {}
    for row in rows:
        key = _channel_key(row, allowed_channels, label)
        if key in result:
            raise ValueError(f"Duplicate {label} channel {key[0]}:{key[1]}")
        result[key] = row
    return result


def build_screening_pool(consensus: dict, causal: dict, *,
                         policy: str = "strict", allowed_channels: Mapping) -> dict:
    """Prioritize the exact strict prefix, optionally followed by all other channels.

    ``allowed_channels`` maps the *configured supported stages* to their widths.
    ``consensus`` must describe that complete channel universe, and ``causal``
    must contain one ablation result per bootstrap-eligible channel plus the
    original strict ranking.  Ineligible channels have ``not_evaluated`` causal
    status, never ``failed``.  Inputs are not mutated.

    Expanded fallback ordering is descending finite consensus score, then stage
    and channel index.  Bootstrap eligibility does not restrict that fallback.
    A strict pool may be empty; an expanded pool can still use the localization
    ranking.  No validation-dependent adaptive threshold or fitting occurs here.
    """
    if not isinstance(policy, str) or policy not in SCREENING_POLICIES:
        raise ValueError(f"screening_policy must be one of {SCREENING_POLICIES}")
    if (not isinstance(allowed_channels, Mapping) or not allowed_channels
            or any(not isinstance(stage, str) or not stage
                   or isinstance(count, bool) or not isinstance(count, Integral) or count < 1
                   for stage, count in allowed_channels.items())):
        raise ValueError("allowed_channels must map configured stage names to positive widths")
    if not isinstance(consensus, Mapping) or not isinstance(causal, Mapping):
        raise ValueError("Consensus and causal diagnostics must be mappings")

    universe = _unique_rows(consensus.get("channels"), allowed_channels, "localized")
    expected = {(stage, index) for stage, count in allowed_channels.items() for index in range(count)}
    if set(universe) != expected:
        raise ValueError("Localization must contain the complete configured channel universe")
    eligible = set()
    scores, stabilities = {}, {}
    for key, row in universe.items():
        scores[key] = _finite_number(row.get("consensus_score"), "consensus_score")
        stabilities[key] = _finite_number(row.get("bootstrap_stability"), "bootstrap_stability")
        if not 0 <= stabilities[key] <= 1:
            raise ValueError("bootstrap_stability must be in [0, 1]")
        if type(row.get("eligible")) is not bool:
            raise ValueError("Bootstrap eligibility must be a boolean")
        if row["eligible"]:
            eligible.add(key)

    measured = _unique_rows(causal.get("channels"), allowed_channels, "causal")
    if set(measured) != eligible:
        raise ValueError("Causal diagnostics must cover exactly the bootstrap-eligible channels")
    for row in measured.values():
        if type(row.get("retained")) is not bool:
            raise ValueError("Causal retained flag must be a boolean")
        for field in ("target_margin_delta", "preservation_accuracy_delta", "preservation_nll_delta"):
            if field in row:
                _finite_number(row[field], field)
    strict_rows = _unique_rows(causal.get("ranking"), allowed_channels, "strict-ranked")
    if set(strict_rows) != {key for key, row in measured.items() if row["retained"]}:
        raise ValueError("Strict ranking must contain exactly the retained causal channels")
    strict_keys = list(strict_rows)  # Preserve upstream margin/NLL ordering exactly.
    fallback_keys = sorted((key for key in universe if key not in strict_rows),
                           key=lambda key: (-scores[key], key[0], key[1]))
    selected_keys = strict_keys + (fallback_keys if policy == "screened_then_ranked" else [])

    def identity(key):
        return {"stage": key[0], "channel": key[1]}

    diagnostics = []
    for key in strict_keys + fallback_keys:
        status = ("not_evaluated" if key not in measured else
                  "passed" if measured[key]["retained"] else "failed")
        source = ("strict" if key in strict_rows else
                  "fallback" if policy == "screened_then_ranked" else "excluded")
        diagnostics.append({**identity(key), "consensus_score": scores[key],
                            "bootstrap_stability": stabilities[key],
                            "bootstrap_eligible": key in eligible,
                            "causal_status": status, "selection_source": source,
                            "causal_diagnostics": dict(measured[key]) if key in measured else None})
    ranking = [identity(key) for key in selected_keys]
    hash_input = {"policy": policy, "ranking": ranking}
    digest = hashlib.sha256(json.dumps(hash_input, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    return {"policy": policy, "ranking": ranking,
            "strict_ranking": [identity(key) for key in strict_keys],
            "strict_ranked_channels": len(strict_keys), "ranked_channels": len(selected_keys),
            "universe_channels": len(universe), "channels": diagnostics,
            "counts": {"passed": len(strict_keys), "failed": len(measured) - len(strict_keys),
                       "not_evaluated": len(universe) - len(measured),
                       "strict": len(strict_keys),
                       "fallback": len(fallback_keys) if policy == "screened_then_ranked" else 0,
                       "excluded": len(fallback_keys) if policy == "strict" else 0},
            "pool_sha256": digest}
