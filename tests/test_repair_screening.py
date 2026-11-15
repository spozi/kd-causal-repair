"""Pure screening-pool policy invariants, independent of model inference."""

from copy import deepcopy

import pytest

from kd.repair_screening import build_screening_pool


WIDTHS = {"stage3": 2, "stage4": 2}


def diagnostics():
    consensus = {"channels": [
        {"stage": "stage3", "channel": 0, "consensus_score": 8.0,
         "bootstrap_stability": 0.9, "eligible": True},
        {"stage": "stage3", "channel": 1, "consensus_score": 9.0,
         "bootstrap_stability": 0.8, "eligible": True},
        {"stage": "stage4", "channel": 0, "consensus_score": 9.0,
         "bootstrap_stability": 0.4, "eligible": False},
        {"stage": "stage4", "channel": 1, "consensus_score": 1.0,
         "bootstrap_stability": 0.75, "eligible": True},
    ]}
    causal = {"channels": [
        {"stage": "stage3", "channel": 0, "retained": True},
        {"stage": "stage3", "channel": 1, "retained": False},
        {"stage": "stage4", "channel": 1, "retained": True},
    ], "ranking": [{"stage": "stage4", "channel": 1}, {"stage": "stage3", "channel": 0}]}
    return consensus, causal


def build(consensus, causal, policy="screened_then_ranked", widths=WIDTHS):
    return build_screening_pool(consensus, causal, policy=policy, allowed_channels=widths)


def test_strict_retains_upstream_order_without_reordering_by_consensus_score():
    consensus, causal = diagnostics()
    actual = build(consensus, causal, "strict")
    assert actual["ranking"] == causal["ranking"]
    assert actual["strict_ranked_channels"] == actual["ranked_channels"] == 2
    assert actual["universe_channels"] == 4
    assert actual["counts"] == {"passed": 2, "failed": 1, "not_evaluated": 1,
                                "strict": 2, "fallback": 0, "excluded": 2}


def test_expansion_preserves_strict_prefix_then_ranks_all_remaining_channels():
    consensus, causal = diagnostics()
    causal["channels"][0].update(target_margin_delta=0.1, preservation_accuracy_delta=0.0,
                                  preservation_nll_delta=-0.01)
    saved = deepcopy((consensus, causal))
    actual = build(consensus, causal)
    assert actual["ranking"] == causal["ranking"] + [
        {"stage": "stage3", "channel": 1}, {"stage": "stage4", "channel": 0}]
    assert actual["strict_ranking"] == causal["ranking"]
    assert actual["counts"] == {"passed": 2, "failed": 1, "not_evaluated": 1,
                                "strict": 2, "fallback": 2, "excluded": 0}
    details = {(r["stage"], r["channel"]): r for r in actual["channels"]}
    assert details["stage3", 0]["causal_status"] == "passed"
    assert details["stage3", 0]["causal_diagnostics"] == causal["channels"][0]
    assert details["stage3", 0]["causal_diagnostics"] is not causal["channels"][0]
    assert details["stage3", 1]["causal_status"] == "failed"
    assert details["stage4", 0]["causal_status"] == "not_evaluated"
    assert details["stage4", 0]["causal_diagnostics"] is None
    assert details["stage4", 0]["selection_source"] == "fallback"
    assert details["stage4", 1]["selection_source"] == "strict"
    assert (consensus, causal) == saved


def test_tied_fallback_scores_sort_by_stage_and_channel_independent_of_input_order():
    consensus, causal = diagnostics()
    for row in consensus["channels"]:
        row.update(consensus_score=1.0, eligible=False)
    causal = {"channels": [], "ranking": []}
    original = build(consensus, causal)
    consensus["channels"].reverse()
    assert build(consensus, causal) == original
    assert original["ranking"] == [
        {"stage": stage, "channel": i} for stage in WIDTHS for i in range(WIDTHS[stage])]


def test_empty_strict_pool_does_not_prevent_expanded_repair():
    consensus, causal = diagnostics()
    for row in causal["channels"]:
        row["retained"] = False
    causal["ranking"] = []
    strict, expanded = build(consensus, causal, "strict"), build(consensus, causal)
    assert strict["ranking"] == []
    assert expanded["strict_ranked_channels"] == 0
    assert expanded["ranked_channels"] == 4
    assert expanded["pool_sha256"] != strict["pool_sha256"]


def test_pool_hash_binds_policy_and_order_but_not_unrelated_diagnostic_metadata():
    consensus, causal = diagnostics()
    before = build(consensus, causal)
    consensus["unused_metadata"] = "unrelated"
    assert build(consensus, causal)["pool_sha256"] == before["pool_sha256"]
    causal["ranking"].reverse()
    assert build(consensus, causal)["pool_sha256"] != before["pool_sha256"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), True, "1", None])
def test_nonfinite_or_nonnumeric_scores_fail_explicitly(score):
    consensus, causal = diagnostics()
    consensus["channels"][0]["consensus_score"] = score
    with pytest.raises(ValueError, match="consensus_score"):
        build(consensus, causal)


@pytest.mark.parametrize("stability", [float("nan"), float("inf"), -0.1, 1.1, True, None])
def test_invalid_stability_is_not_silently_ignored(stability):
    consensus, causal = diagnostics()
    consensus["channels"][0]["bootstrap_stability"] = stability
    with pytest.raises(ValueError, match="bootstrap_stability"):
        build(consensus, causal)


@pytest.mark.parametrize("section", ["localized", "causal", "strict-ranked"])
def test_duplicate_channel_diagnostics_are_rejected(section):
    consensus, causal = diagnostics()
    target = consensus["channels"] if section == "localized" else causal[
        "channels" if section == "causal" else "ranking"]
    target.append(dict(target[0]))
    with pytest.raises(ValueError, match="Duplicate " + section):
        build(consensus, causal)


@pytest.mark.parametrize("changes", [
    {"channel": True}, {"channel": -1}, {"channel": 2}, {"channel": 0.0},
    {"stage": "stage2"}, {"stage": None},
])
def test_localized_identities_must_belong_to_supported_configured_stages(changes):
    consensus, causal = diagnostics()
    consensus["channels"][0].update(changes)
    with pytest.raises(ValueError, match="Invalid localized"):
        build(consensus, causal)


def test_missing_localized_channels_is_not_treated_as_a_complete_universe():
    consensus, causal = diagnostics()
    consensus["channels"].pop()
    with pytest.raises(ValueError, match="complete configured"):
        build(consensus, causal)


@pytest.mark.parametrize("mutation", ["missing", "unstable_tested", "retained_without_ranking", "ranking_failed"])
def test_inconsistent_causal_diagnostics_fail_explicitly(mutation):
    consensus, causal = diagnostics()
    if mutation == "missing":
        causal["channels"].pop()
    elif mutation == "unstable_tested":
        causal["channels"].append({"stage": "stage4", "channel": 0, "retained": False})
    elif mutation == "retained_without_ranking":
        causal["ranking"].pop()
    else:
        causal["ranking"].append({"stage": "stage3", "channel": 1})
    with pytest.raises(ValueError, match="Causal diagnostics|Strict ranking"):
        build(consensus, causal)


def test_nonfinite_measured_causal_delta_cannot_be_reused():
    consensus, causal = diagnostics()
    causal["channels"][0]["target_margin_delta"] = float("nan")
    with pytest.raises(ValueError, match="target_margin_delta"):
        build(consensus, causal)


@pytest.mark.parametrize("policy", ["adaptive", None, [], True])
def test_unknown_or_invalid_policy_is_rejected(policy):
    consensus, causal = diagnostics()
    with pytest.raises(ValueError, match="screening_policy"):
        build(consensus, causal, policy)


@pytest.mark.parametrize("widths", [{}, {"stage3": 0}, {"stage3": True}, {"stage3": 2.0}, {1: 2}])
def test_invalid_allowed_universe_is_rejected(widths):
    consensus, causal = diagnostics()
    with pytest.raises(ValueError, match="allowed_channels"):
        build(consensus, causal, widths=widths)
