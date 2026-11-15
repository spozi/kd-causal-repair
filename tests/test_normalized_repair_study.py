"""Selection gates and real CPU integration for opt-in normalized repair.

Integration fixtures control localization and selection scores, not optimizer
updates: each deduplicated support goes through the actual repair trainer.
"""

from contextlib import ExitStack, redirect_stdout
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
from kd.teacher_baselines import study_configs
from kd.teacher_repair_baseline import repair_spec
from kd.teacher_training import RECIPES, train_teacher
from tests.test_teacher_baselines import TeacherRepairIntegrationTests as _RepairFixture, _TinyCifar


BASELINE = {"accuracy": 0.8, "target_macro_recall": 0.4}


def candidate(*, kl=0.01, accuracy=0.8, recall=0.5, nll=1.0, budget=2, coordinates=100):
    return {"budget": budget, "learning_rate": 0.001,
            "parameter_budget": {"editable_coordinates": coordinates},
            "validation": {"accuracy": accuracy, "target_macro_recall": recall,
                           "target_balanced_nll": nll,
                           "preservation": {"mean_forward_kl": kl}}}


@pytest.mark.parametrize("kl,passed", [
    (0.0, True), (0.02, True), (0.020000001, False), (-0.01, False),
    (float("nan"), False), (float("inf"), False), (-float("inf"), False),
])
def test_preservation_guard_is_closed_and_includes_exact_boundary(kl, passed):
    result = study._candidate_constraints(candidate(kl=kl), BASELINE, 0.02)
    assert result["accuracy_guard_pass"]
    assert result["target_recall_guard_pass"]
    assert result["preservation_kl_guard_pass"] is passed
    assert result["accepted"] is passed


def test_missing_preservation_cannot_pass_but_legacy_gate_is_unchanged():
    row = candidate()
    del row["validation"]["preservation"]
    assert not study._candidate_constraints(row, BASELINE, 0.02)["accepted"]
    legacy = study._candidate_constraints(row, BASELINE)
    assert legacy["accepted"]
    assert "preservation_kl_guard_pass" not in legacy


def test_kl_does_not_replace_accuracy_or_strict_target_recall_guards():
    assert study._candidate_constraints(candidate(accuracy=0.795), BASELINE, 0.02)["accepted"]
    assert not study._candidate_constraints(candidate(accuracy=0.794), BASELINE, 0.02)["accepted"]
    assert not study._candidate_constraints(candidate(recall=0.4), BASELINE, 0.02)["accepted"]
    assert not study._candidate_constraints(candidate(recall=0.39), BASELINE, 0.02)["accepted"]


def test_normalized_selection_ties_use_coordinates_not_channel_count():
    fewer_channels = candidate(budget=1, coordinates=200)
    fewer_coordinates = candidate(budget=2, coordinates=100)
    normalized = replace(study.NeuronSurgerySpec(), parameter_budget_fractions=(0.01,),
                         max_preservation_kl=0.02)
    assert study._eligible_candidates([fewer_channels, fewer_coordinates], BASELINE, normalized)[0] is fewer_coordinates
    assert study._eligible_candidates([fewer_coordinates, fewer_channels], BASELINE,
                                      study.NeuronSurgerySpec())[0] is fewer_channels
    rejected = candidate(kl=0.03, recall=0.9)
    assert rejected not in study._eligible_candidates([fewer_channels, rejected], BASELINE, normalized)


def test_normalized_spec_round_trip_does_not_modify_legacy_serialization():
    original = study.NeuronSurgerySpec()
    assert "parameter_budget_fractions" not in original.to_dict()
    assert "max_preservation_kl" not in original.to_dict()
    assert study.neuron_surgery_spec_from_dict(original.to_dict()) == original
    normalized = replace(original, parameter_budget_fractions=(0.004, 0.008, 0.016, 0.032),
                         max_preservation_kl=0.02)
    encoded = json.loads(json.dumps(normalized.to_dict()))
    assert study.neuron_surgery_spec_from_dict(encoded) == normalized
    assert normalized.to_dict()["parameter_budget_fractions"] == (0.004, 0.008, 0.016, 0.032)


@pytest.mark.parametrize("changes", [
    {"causal_validation": False}, {"channel_selection": "random"},
    {"channel_selection": "stage_matched_random"}, {"channel_selection": "antiscreened"},
])
def test_normalized_spec_requires_causally_screened_channels(changes):
    spec = replace(study.NeuronSurgerySpec(), parameter_budget_fractions=(0.01,),
                   max_preservation_kl=0.02, **changes)
    with pytest.raises(ValueError, match="causally screened"):
        spec.validate()


def test_validation_report_measures_drift_from_original_anchor_not_labels():
    labels = np.array([0, 1], dtype=np.int64)
    anchor = np.array([[0.8, 0.2], [0.1, 0.9]])
    changed = np.array([[0.7, 0.3], [0.2, 0.8]])
    expected = float(np.sum(anchor * np.log(anchor / changed), axis=1).mean())
    with patch.object(study, "collect_predictions", return_value=(labels, changed)):
        report = study._validation_report(None, None, "cpu", ["a", "b"], (0, 1),
                                          preservation_reference=(labels, anchor))
        assert report["preservation"]["mean_forward_kl"] == pytest.approx(expected)
        assert report["preservation"]["samples"] == 2
        with pytest.raises(ValueError, match="order/labels"):
            study._validation_report(None, None, "cpu", ["a", "b"], (0, 1),
                                     preservation_reference=(labels[::-1], anchor))


@pytest.fixture(scope="module")
def trained_anchor(tmp_path_factory):
    root = tmp_path_factory.mktemp("normalized-repair")
    base = _RepairFixture().spec(str(root), teacher_model="convnet6_w32", student_model="convnet6_w16")
    base = replace(base, teacher=replace(base.teacher, train=replace(base.teacher.train, threads=1)))
    output = root / "baseline"
    teachers, _, _ = study_configs(base, output)
    with patch("kd.data.datasets.CIFAR10", _TinyCifar), redirect_stdout(io.StringIO()):
        control = next(config for recipe, config in teachers if recipe == "control")
        train_teacher(control, RECIPES["control"], selection_metric=base.teacher_selection_metric)
    ranking = [{"stage": "stage2", "channel": 0},
               {"stage": "stage3", "channel": 0},
               {"stage": "stage3", "channel": 1}]
    model = create_model(base.teacher.student.name, 10)
    total = sum(parameter.numel() for parameter in model.parameters())
    counts = [sum(int(mask.count_nonzero()) for mask in channel_parameter_masks(model, ranking[:i]).values())
              for i in (1, 2)]
    # Four requests: infeasible, unique one-channel support, duplicate support,
    # unique two-channel support. Fractions avoid integer-floor boundary noise.
    fractions = ((counts[0] - 1) / total, (counts[0] + 0.25) / total,
                 (counts[0] + 0.75) / total, (counts[1] + 0.25) / total)
    spec = replace(repair_spec(base, 41), parameter_budget_fractions=fractions,
                   max_preservation_kl=0.015)
    return root, output, base, spec, ranking


def test_dry_run_records_budget_denominator_and_held_out_constraint_without_data_access(trained_anchor):
    root, baseline, _, spec, _ = trained_anchor
    output = root / "dry"
    with patch.object(study, "build_data", side_effect=AssertionError("dataset access")):
        result = study.run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds,
                                               dry_run=True, spec=spec)
    assert result["repair_candidate_count"] == 4  # upper bound before infeasibility/deduplication
    assert result["student_run_count"] == 0
    protocol = result["protocol"]
    assert protocol["repair"]["budget_mode"] == "parameter_fraction"
    assert protocol["repair"]["channel_budgets"] is None
    assert protocol["repair"]["parameter_budget_fractions"] == list(spec.parameter_budget_fractions)
    constraint = protocol["teacher_selection"]["preservation_constraint"]
    assert constraint["direction"] == "original||candidate"
    assert constraint["temperature"] == 1.0
    assert constraint["maximum"] == 0.015
    assert constraint["population"] == "all selection-validation images, including target classes"
    assert "repair_budget.py" in protocol["source_sha256"]
    assert not output.exists()


@pytest.mark.parametrize("accepted", [True, False])
def test_actual_masked_training_deduplicates_supports_preserves_state_and_gates_export(trained_anchor, accepted):
    root, baseline, _, spec, ranking = trained_anchor
    output = root / ("accepted" if accepted else "rejected")
    original_report = study._validation_report
    reports = []

    def controlled_selection(*args, **kwargs):
        report = original_report(*args, **kwargs)
        is_candidate = kwargs.get("preservation_reference") is not None
        report.update(accuracy=0.8, target_macro_recall=0.5 if is_candidate else 0.4,
                      tail_macro_recall=0.5 if is_candidate else 0.4,
                      target_balanced_nll=0.9 if is_candidate else 1.0,
                      tail_balanced_nll=0.9 if is_candidate else 1.0)
        if is_candidate:
            report["preservation"]["mean_forward_kl"] = 0.01 if accepted else 0.02
        reports.append(is_candidate)
        return report

    def no_test(*args, **kwargs):
        assert kwargs.get("include_test") is False
        return build_data(*args, **kwargs)

    patches = [
        patch("kd.data.datasets.CIFAR10", _TinyCifar),
        patch.object(study, "build_data", side_effect=no_test),
        patch.object(study, "run_experiment", side_effect=AssertionError("downstream KD")),
        patch.object(study, "differential_channel_scores", return_value={"fixture": np.array([1.0])}),
        patch.object(study, "consensus_channel_ranking", return_value={"channels": [dict(row, eligible=True) for row in ranking]}),
        patch.object(study, "causal_channel_validation", return_value={"channels": [], "ranking": ranking}),
        patch.object(study, "_validation_report", side_effect=controlled_selection),
    ]
    with ExitStack() as stack, redirect_stdout(io.StringIO()):
        for context in patches:
            stack.enter_context(context)
        result = study.run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds, spec=spec)
    assert result["status"] == ("repair_only_complete" if accepted else "no_repair_selected")
    assert reports == [False, True, True]
    assert (output / "teacher_repaired.pt").exists() is accepted
    assert not (output / "teacher_evaluation").exists()
    assert not list(output.glob("kd_*"))
    localization = json.loads((output / "localization.json").read_text())["payload"]
    assert [row["status"] for row in localization["parameter_budget_plan"]["requests"]] == [
        "infeasible", "unique", "duplicate", "unique"]
    candidates = json.loads((output / "candidate_results.json").read_text())
    assert len(candidates) == 2
    selection = json.loads((output / "teacher_selection.json").read_text())
    assert selection["eligible_count"] == (2 if accepted else 0)
    assert all(row["preservation_kl_guard_pass"] is accepted for row in selection["candidate_constraints"])
    anchor_path = baseline / "teacher_control_seed41" / "student.pt"
    anchor_digest = fingerprint(anchor_path)
    anchor = torch.load(anchor_path, weights_only=True)["student"]
    model = create_model(spec.teacher_model, 10)
    for row in candidates:
        checkpoint = torch.load(row["checkpoint"], weights_only=True)
        metadata = row["parameter_budget"]
        assert checkpoint["repair"]["parameter_budget"] == metadata
        assert checkpoint["repair"]["max_preservation_kl"] == 0.015
        masks = channel_parameter_masks(model, row["channels"])
        assert sum(int(mask.count_nonzero()) for mask in masks.values()) == metadata["editable_coordinates"]
        assert row["verification"]["selected_parameter_coordinates"] == metadata["editable_coordinates"]
        for name, value in anchor.items():
            actual = checkpoint["student"][name]
            if name in masks:
                assert torch.equal(actual[~masks[name]], value[~masks[name]]), name
            else:
                assert torch.equal(actual, value), name  # includes every BN buffer
        assert any(not torch.equal(checkpoint["student"][name], value) for name, value in anchor.items())
        assert len(json.loads((Path(row["checkpoint"]).parent / "history.json").read_text())) == 1
    assert fingerprint(anchor_path) == anchor_digest
    # Cached identical protocols are reusable; changing a cap or KL threshold
    # must never silently reinterpret an already selected experiment.
    with patch.object(study, "build_data", side_effect=AssertionError("cached dataset access")):
        assert study.run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds,
                                              spec=spec) == json.loads(json.dumps(result))
        with pytest.raises(ValueError, match="new output directory"):
            study.run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds,
                                           spec=replace(spec, max_preservation_kl=0.016))

