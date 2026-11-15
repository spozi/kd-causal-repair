"""CPU integration for opt-in expansion; localization is controlled, updates are real.

These tests do not run downstream KD or load test data. They exercise the actual
candidate optimizer, frozen-coordinate checks, validation gates and cache binding.
"""

from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from kd import neuron_surgery_study as study
from kd.checkpoints import fingerprint
from kd.data import build_data
from kd.models import create_model
from kd.neuron_surgery import channel_parameter_masks
from kd.repair_budget import parameter_support_sha256, plan_parameter_budgets
from kd.repair_screening import build_screening_pool
from tests.test_normalized_repair_study import trained_anchor
from tests.test_teacher_baselines import _TinyCifar


POLICY = "screened_then_ranked"


def _read(path):
    return json.loads(Path(path).read_text())


def _localization(spec, *, empty_strict):
    model = create_model(spec.teacher_model, spec.dataset_num_classes)
    leading = [("stage2", 0), ("stage3", 0), ("stage3", 1)]
    channels = []
    for stage, width in model.feature_channels.items():
        if stage not in spec.stages:
            continue
        for index in range(width):
            key = (stage, index)
            eligible = not empty_strict and key in {leading[0], leading[2]}
            channels.append({"stage": stage, "channel": index,
                             "consensus_score": float(10 - leading.index(key)) if key in leading else 0.0,
                             "bootstrap_stability": 1.0 if eligible else 0.0,
                             "eligible": eligible})
    channels.sort(key=lambda row: (-row["consensus_score"], row["stage"], row["channel"]))
    consensus = {"channels": channels, "eligible_count": sum(row["eligible"] for row in channels),
                 "stability_threshold": 0.7, "top_fraction": 0.25, "repetitions": 20, "seed": 2026}
    causal_rows = []
    for row in channels:
        if row["eligible"]:
            passed = (row["stage"], row["channel"]) == leading[2]
            causal_rows.append({**row, "retained": passed,
                                "target_margin_delta": 0.1 if passed else -0.1,
                                "preservation_accuracy_delta": 0.0,
                                "preservation_nll_delta": 0.0})
    strict = [] if empty_strict else [{"stage": "stage3", "channel": 1}]
    return consensus, {"channels": causal_rows, "ranking": strict}, model


@contextmanager
def _controlled_run(consensus, causal, *, accepted=True):
    """Keep real training/inference, replace only localization and gate observations."""
    original_report = study._validation_report
    candidate_observations = []

    def validation(*args, **kwargs):
        report = original_report(*args, **kwargs)
        is_candidate = kwargs.get("preservation_reference") is not None
        report.update(accuracy=0.8, target_macro_recall=0.5 if is_candidate else 0.4,
                      tail_macro_recall=0.5 if is_candidate else 0.4,
                      target_balanced_nll=0.9 if is_candidate else 1.0,
                      tail_balanced_nll=0.9 if is_candidate else 1.0)
        if is_candidate:
            report["preservation"]["mean_forward_kl"] = 0.01 if accepted else 0.02
            candidate_observations.append(report)
        return report

    def no_test(*args, **kwargs):
        assert kwargs.get("include_test") is False
        return build_data(*args, **kwargs)

    with ExitStack() as stack, redirect_stdout(io.StringIO()):
        for context in (
                patch("kd.data.datasets.CIFAR10", _TinyCifar),
                patch.object(study, "build_data", side_effect=no_test),
                patch.object(study, "run_experiment", side_effect=AssertionError("downstream KD")),
                patch.object(study, "differential_channel_scores", return_value={"fixture": np.array([1.0])}),
                patch.object(study, "consensus_channel_ranking", return_value=consensus),
                patch.object(study, "causal_channel_validation", return_value=causal),
                patch.object(study, "_validation_report", side_effect=validation)):
            stack.enter_context(context)
        yield candidate_observations


def _assert_frozen_except_support(baseline, spec, rows):
    anchor_path = baseline / "teacher_control_seed41" / "student.pt"
    digest = fingerprint(anchor_path)
    anchor = torch.load(anchor_path, weights_only=True)["student"]
    model = create_model(spec.teacher_model, spec.dataset_num_classes)
    for row in rows:
        state = torch.load(row["checkpoint"], weights_only=True)
        masks = channel_parameter_masks(model, row["channels"])
        count = sum(int(mask.count_nonzero()) for mask in masks.values())
        assert count == row["parameter_budget"]["editable_coordinates"]
        assert count == row["verification"]["selected_parameter_coordinates"]
        assert state["repair"]["parameter_budget"] == row["parameter_budget"]
        assert state["repair"]["screening_policy"] == POLICY
        assert row["screening_policy"] == POLICY
        support = row["screening_support"]
        assert state["repair"]["screening_support"] == support
        assert support["support_sha256"] == row["parameter_budget"]["support_sha256"]
        assert support["support_sha256"] == parameter_support_sha256(masks)
        assert support["strict_channels"] + support["fallback_channels"] == row["budget"]
        assert support["strict_channels"] == sum(
            channel["selection_source"] == "strict" for channel in support["channels"])
        assert support["fallback_channels"] == sum(
            channel["selection_source"] == "fallback" for channel in support["channels"])
        assert [{"stage": channel["stage"], "channel": channel["channel"]}
                for channel in support["channels"]] == row["channels"]
        for name, tensor in anchor.items():
            actual = state["student"][name]
            if name in masks:
                assert torch.equal(actual[~masks[name]], tensor[~masks[name]]), name
            else:
                assert torch.equal(actual, tensor), name  # also freezes every BN buffer
        assert any(not torch.equal(state["student"][name], value) for name, value in anchor.items())
        assert len(_read(Path(row["checkpoint"]).parent / "history.json")) == spec.repair_epochs
    assert fingerprint(anchor_path) == digest


@pytest.mark.parametrize("accepted", [True, False])
def test_expansion_can_train_with_no_strict_channels_without_relaxing_kl_gate(trained_anchor, accepted):
    root, baseline, _, original, _ = trained_anchor
    spec = replace(original, screening_policy=POLICY)
    consensus, causal, _ = _localization(spec, empty_strict=True)
    output = root / f"expanded-empty-{accepted}"
    with _controlled_run(consensus, causal, accepted=accepted) as observations:
        result = study.run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds, spec=spec)
    assert result["status"] == ("repair_only_complete" if accepted else "no_repair_selected")
    assert len(observations) == 2
    assert (output / "teacher_repaired.pt").exists() is accepted
    assert not (output / "teacher_evaluation").exists()
    assert not list(output.glob("kd_*"))
    localization = _read(output / "localization.json")["payload"]
    assert localization["screening_policy"] == POLICY
    assert localization["ranking"][:2] == [
        {"stage": "stage2", "channel": 0}, {"stage": "stage3", "channel": 0}]
    rows = _read(output / "candidate_results.json")
    assert [row["budget"] for row in rows] == [1, 2]
    assert all(row["screening_support"]["pool_sha256"] == localization["screening_pool"]["pool_sha256"]
               for row in rows)
    assert all(row["screening_support"]["strict_channels"] == 0 for row in rows)
    assert all(channel["causal_status"] == "not_evaluated"
               for row in rows for channel in row["screening_support"]["channels"])
    _assert_frozen_except_support(baseline, spec, rows)
    selection = _read(output / "teacher_selection.json")
    assert selection["screening_policy"] == POLICY
    assert selection["eligible_count"] == (2 if accepted else 0)
    assert all(row["preservation_kl_guard_pass"] is accepted for row in selection["candidate_constraints"])
    protocol = _read(output / "protocol.json")
    assert protocol["study"]["screening_policy"] == POLICY
    assert "repair_screening.py" in protocol["source_sha256"]
    report = (output / "report.md").read_text()
    assert "screened_then_ranked" in report
    assert "not strict causal screening" in report
    # Neither a cached completion nor a rejected run may be reinterpreted under strict screening.
    with patch.object(study, "build_data", side_effect=AssertionError("cached data access")):
        assert study.run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds,
                                              spec=spec) == json.loads(json.dumps(result))
        with pytest.raises(ValueError, match="new output directory"):
            study.run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds,
                                           spec=replace(spec, screening_policy="strict"))
        with pytest.raises(ValueError, match="new output directory"):
            study.run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds,
                                           spec=replace(spec, max_preservation_kl=0.016))


def test_strict_empty_screen_still_stops_without_any_candidate_fit(trained_anchor):
    root, baseline, _, spec, _ = trained_anchor
    consensus, causal, _ = _localization(spec, empty_strict=True)
    output = root / "strict-empty"
    with _controlled_run(consensus, causal) as observations, patch.object(
            study, "_train_candidate", side_effect=AssertionError("candidate training")):
        result = study.run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds, spec=spec)
    assert result["status"] == "no_repair_selected"
    assert observations == []
    localization = _read(output / "localization.json")["payload"]
    assert localization["ranking"] == []
    assert localization["available_channel_budgets"] == []
    assert not (output / "candidate_results.json").exists()


def test_expanded_prefix_retains_strict_ranking_then_reaches_larger_actual_support(trained_anchor):
    root, baseline, _, original, _ = trained_anchor
    consensus, causal, model = _localization(original, empty_strict=False)
    expanded = [{"stage": "stage3", "channel": 1},
                {"stage": "stage2", "channel": 0}, {"stage": "stage3", "channel": 0}]
    counts = [sum(int(mask.count_nonzero()) for mask in channel_parameter_masks(model, expanded[:n]).values())
              for n in (1, 3)]
    total = sum(parameter.numel() for parameter in model.parameters())
    fractions = tuple((count + 0.25) / total for count in counts)
    spec = replace(original, screening_policy=POLICY, parameter_budget_fractions=fractions)
    output = root / "expanded-mixed"
    with _controlled_run(consensus, causal):
        result = study.run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds, spec=spec)
    assert result["status"] == "repair_only_complete"
    localization = _read(output / "localization.json")["payload"]
    assert localization["ranking"][:3] == expanded
    rows = _read(output / "candidate_results.json")
    assert [row["budget"] for row in rows] == [1, 3]
    assert rows[0]["channels"] == causal["ranking"]
    assert rows[-1]["channels"] == expanded
    assert rows[-1]["screening_support"]["strict_channels"] == 1
    assert rows[-1]["screening_support"]["fallback_channels"] == 2
    assert [channel["causal_status"] for channel in rows[-1]["screening_support"]["channels"]] == [
        "passed", "failed", "not_evaluated"]
    assert all(row["screening_support"]["pool_sha256"] == localization["screening_pool"]["pool_sha256"]
               for row in rows)
    assert rows[-1]["parameter_budget"]["editable_coordinates"] > rows[0]["parameter_budget"]["editable_coordinates"]
    assert all(row["parameter_budget"]["achieved_fraction"] <= fractions[i] for i, row in enumerate(rows))
    _assert_frozen_except_support(baseline, spec, rows)
    # The reference policy remains the same screened prefix and deduplicates the larger cap.
    strict_output = root / "strict-mixed"
    strict = replace(spec, screening_policy="strict")
    with _controlled_run(consensus, causal):
        strict_result = study.run_neuron_surgery_study(
            baseline, strict_output, "cpu", strict.student_seeds, spec=strict)
    assert strict_result["status"] == "repair_only_complete"
    strict_rows = _read(strict_output / "candidate_results.json")
    assert len(strict_rows) == 1
    assert strict_rows[0]["channels"] == causal["ranking"]
    assert strict_rows[0]["parameter_budget"]["editable_coordinates"] == counts[0]
    assert "screened_then_ranked" not in (strict_output / "report.md").read_text()


def test_expanded_dry_run_records_policy_without_loading_data(trained_anchor):
    root, baseline, _, original, _ = trained_anchor
    spec = replace(original, screening_policy=POLICY)
    output = root / "expanded-dry"
    with patch.object(study, "build_data", side_effect=AssertionError("dry-run data access")):
        plan = study.run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds,
                                             dry_run=True, spec=spec)
    assert plan["protocol"]["study"]["screening_policy"] == POLICY
    assert plan["repair_candidate_count_is_upper_bound"]
    assert plan["student_run_count"] == 0
    assert not output.exists()


@pytest.mark.parametrize("changes", [
    {"screening_policy": "unknown"}, {"screening_policy": True},
    {"screening_policy": [POLICY]}, {"parameter_budget_fractions": ()},
    {"max_preservation_kl": None}, {"causal_validation": False},
    {"channel_selection": "random"}, {"channel_selection": "stage_matched_random"},
    {"channel_selection": "antiscreened"},
])
def test_standalone_expanded_policy_rejects_invalid_or_unsupported_options(changes):
    values = {"screening_policy": POLICY, "parameter_budget_fractions": (0.004, 0.008),
              "max_preservation_kl": 0.01, **changes}
    with pytest.raises(ValueError):
        replace(study.NeuronSurgerySpec(), **values).validate()


def test_standalone_policy_roundtrip_preserves_compact_strict_defaults():
    default = study.NeuronSurgerySpec()
    assert default.screening_policy == "strict"
    assert "screening_policy" not in default.to_dict()
    assert study.neuron_surgery_spec_from_dict(default.to_dict()) == default
    explicit = study.neuron_surgery_spec_from_dict({**default.to_dict(), "screening_policy": "strict"})
    assert explicit.to_dict() == default.to_dict()
    expanded = replace(default, screening_policy=POLICY,
                       parameter_budget_fractions=(0.004, 0.008, 0.016, 0.032), max_preservation_kl=0.01)
    expanded.validate()
    assert study.neuron_surgery_spec_from_dict(json.loads(json.dumps(expanded.to_dict()))) == expanded


def test_resnet34_expansion_makes_large_parameter_caps_reachable_on_cpu():
    # No weights download, input data, optimizer or inference is involved: this
    # tests actual ResNet-34 coordinate masks against the requested budget grid.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = create_model("resnet34_smallstem", 100)
        allowed = {stage: model.feature_channels[stage] for stage in ("stage3", "stage4")}
        strict = [{"stage": "stage4", "channel": 0}, {"stage": "stage4", "channel": 1}]
        strict_keys = {(row["stage"], row["channel"]) for row in strict}
        channels = [{"stage": stage, "channel": channel,
                     "consensus_score": float(width - channel),
                     "bootstrap_stability": 1.0 if (stage, channel) in strict_keys else 0.0,
                     "eligible": (stage, channel) in strict_keys}
                    for stage, width in allowed.items() for channel in range(width)]
        causal = {"channels": [dict(row, retained=True) for row in strict], "ranking": strict}
        pool = build_screening_pool({"channels": channels}, causal,
                                    policy=POLICY, allowed_channels=allowed)
        fractions = (0.004, 0.008, 0.016, 0.032)
        old = plan_parameter_budgets(model, strict, fractions)
        new = plan_parameter_budgets(model, pool["ranking"], fractions)
        assert len(old["candidates"]) == 1
        assert all(request["saturated"] for request in old["requests"])
        assert len(new["candidates"]) == 4
        assert all(not request["saturated"] for request in new["requests"])
        assert all(request["status"] == "unique" for request in new["requests"])
        for request in new["requests"]:
            assert request["editable_coordinates"] <= request["coordinate_cap"]
            assert request["achieved_fraction"] <= request["requested_fraction"]
        # Near-cap support is genuinely above 1.5% and 3.1%, respectively, not
        # the earlier <1% screened support repeated under nominal larger caps.
        assert new["requests"][2]["achieved_fraction"] > 0.015
        assert new["requests"][3]["achieved_fraction"] > 0.031
        assert len({row["parameter_budget"]["support_sha256"] for row in new["candidates"]}) == 4
    finally:
        torch.set_num_threads(previous_threads)

