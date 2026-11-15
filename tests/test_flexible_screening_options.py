"""Benchmark configuration/provenance checks for opt-in flexible screening."""

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kd.checkpoints import fingerprint, write_json
from kd.matrix_binding import STUDY_SOURCE_FILES
from kd.teacher_baselines import load_spec, plan_study, study_configs
from kd.teacher_repair_baseline import TeacherRepairOptions, repair_options, repair_spec
from tests.test_normalized_repair_options import FRACTIONS, LEGACY_OPTIONS, _spec


def expanded(options=None):
    return replace(options or TeacherRepairOptions(), screening_policy="screened_then_ranked",
                   parameter_budget_fractions=FRACTIONS, max_preservation_kl=0.01)


def test_strict_policy_keeps_compact_historical_defaults():
    options = TeacherRepairOptions()
    assert options.screening_policy == "strict"
    assert "screening_policy" not in options.to_dict()
    assert json.loads(json.dumps(options.to_dict())) == LEGACY_OPTIONS
    assert repair_options(LEGACY_OPTIONS) == options
    assert repair_options({**LEGACY_OPTIONS, "screening_policy": "strict"}) == options


@pytest.mark.parametrize("policy", [None, True, "", "ranked", "adaptive", []])
def test_unknown_or_malformed_policy_is_rejected(policy):
    with pytest.raises(ValueError, match="screening_policy"):
        replace(expanded(), screening_policy=policy).validate()


def test_expansion_requires_normalized_budgets_and_explicit_preservation_gate():
    with pytest.raises(ValueError, match="parameter-normalized"):
        TeacherRepairOptions(screening_policy="screened_then_ranked").validate()
    with pytest.raises(ValueError, match="max_preservation_kl"):
        TeacherRepairOptions(screening_policy="screened_then_ranked",
                             parameter_budget_fractions=FRACTIONS).validate()
    expanded().validate()


def test_expanded_benchmark_roundtrip_and_standalone_propagation(tmp_path):
    original = _spec(tmp_path)
    spec = replace(original, repair=expanded(original.repair))
    path = tmp_path / "expanded.json"
    write_json(path, spec.to_dict())
    loaded = load_spec(path)
    assert loaded == spec
    standalone = repair_spec(loaded, 41)
    assert standalone.screening_policy == "screened_then_ranked"
    assert standalone.parameter_budget_fractions == FRACTIONS
    assert standalone.max_preservation_kl == 0.01
    assert standalone.channel_selection == "screened"
    assert standalone.causal_validation is True
    assert loaded.to_dict()["repair"]["screening_policy"] == "screened_then_ranked"


def test_expanded_plan_freezes_policy_and_keeps_candidate_grid_bound(tmp_path):
    original = _spec(tmp_path)
    strict = replace(original, repair=replace(original.repair,
        parameter_budget_fractions=FRACTIONS, max_preservation_kl=0.01))
    spec = replace(strict, repair=expanded(original.repair))
    with patch("kd.teacher_baselines.build_data", side_effect=AssertionError("data access")):
        strict_plan = plan_study(strict, tmp_path / "strict")
        plan = plan_study(spec, tmp_path / "expanded")
    assert plan["counts"]["repair_candidate_fits_max"] == 8
    assert plan["counts"] == strict_plan["counts"]
    assert plan["repair_protocols"][0]["screening_policy"] == "screened_then_ranked"
    assert plan["repair_protocols"][0]["max_preservation_kl"] == 0.01
    assert "screening_policy" not in strict_plan["repair_protocols"][0]
    assert "repair_screening.py" in STUDY_SOURCE_FILES
    assert plan["source_sha256"]["repair_screening.py"] == fingerprint(Path("kd/repair_screening.py"))


def test_policy_changes_no_student_kd_configuration(tmp_path):
    original = _spec(tmp_path)
    strict = replace(original, repair=replace(original.repair,
        parameter_budget_fractions=FRACTIONS, max_preservation_kl=0.01))
    spec = replace(strict, repair=expanded(original.repair))
    strict_configs = study_configs(strict, tmp_path / "same-output")
    expanded_configs = study_configs(spec, tmp_path / "same-output")
    # Teacher/KD runtime settings are unchanged; only the frozen repair protocol differs.
    assert strict_configs == expanded_configs


@pytest.mark.parametrize("source", ["colored_mnist", "waterbirds"])
def test_shortcut_family_rejects_expansion_before_data_or_training(tmp_path, source):
    original = _spec(tmp_path, source)
    spec = replace(original, repair=expanded(original.repair))
    with patch("kd.teacher_baselines.build_data", side_effect=AssertionError("data access")):
        with pytest.raises(ValueError, match="not implemented for shortcut datasets"):
            plan_study(spec, tmp_path / "unsupported")


def test_expanded_options_are_not_silently_ignored_without_repair_condition(tmp_path):
    original = _spec(tmp_path)
    spec = replace(original, repair=expanded(original.repair),
                   recipes=tuple(r for r in original.recipes if r != "teacher_repair"))
    with pytest.raises(ValueError, match="Repair options require"):
        plan_study(spec, tmp_path / "disabled")
