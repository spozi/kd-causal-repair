from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kd.baseline_artifacts import write_manifest
from kd.checkpoints import fingerprint, write_json
from kd.teacher_baselines import plan_study
from kd.teacher_baseline_phases import main, run_baseline_then_repair
from kd.teacher_training import RECIPES


def _fixture(root):
    from tests.test_teacher_baselines import TeacherRepairIntegrationTests
    fixture = TeacherRepairIntegrationTests()
    spec = fixture.spec(str(root / "data"))
    return fixture, replace(spec, recipes=(*RECIPES, "teacher_repair"))


def test_dry_run_freezes_nothing_and_never_accesses_data_or_trains(tmp_path):
    _, spec = _fixture(tmp_path)
    output, repairs = tmp_path / "baseline", tmp_path / "repair"
    with patch("kd.teacher_baseline_phases.run_study", side_effect=AssertionError("training")), \
            patch("kd.teacher_baselines.build_data", side_effect=AssertionError("dataset access")), \
            patch("kd.teacher_baseline_phases.train_repair_teacher", side_effect=AssertionError("repair")):
        result = run_baseline_then_repair(spec, output, repair_output=repairs, dry_run=True)
    workflow = result["workflow"]
    assert result["dry_run"]
    assert workflow["spec"] == spec.to_dict()
    assert workflow["baseline_protocol"]["spec"]["repair"] is None
    assert workflow["baseline_protocol"]["spec"]["recipes"] == list(RECIPES)
    assert "teacher_baseline_phases.py" in workflow["source_sha256"]
    assert workflow["counts"]["baseline_fits"] == 23
    assert workflow["counts"]["repair_additional_fits_max"] == 2
    assert workflow["counts"]["total_fits_max"] == 25
    assert not output.exists() and not repairs.exists()


@pytest.mark.parametrize("invalid", ["missing_repair", "missing_condition", "matrix", "shortcut"])
def test_invalid_scope_fails_without_writes(tmp_path, invalid):
    _, spec = _fixture(tmp_path)
    if invalid == "missing_repair":
        spec = replace(spec, repair=None)
    elif invalid == "missing_condition":
        spec = replace(spec, recipes=tuple(RECIPES), repair=None)
    elif invalid == "matrix":
        spec = replace(spec, matrix_binding={"untrusted": "binding"})
    else:
        spec = replace(spec, teacher=replace(spec.teacher, data=replace(spec.teacher.data, source="colored_mnist")))
    with pytest.raises(ValueError):
        run_baseline_then_repair(spec, tmp_path / "baseline", repair_output=tmp_path / "repair")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("paths", [
    ("baseline", "baseline"), ("baseline", "baseline/repair"), ("repair/base", "repair"),
    ("data", "repair"), ("data/models", "repair"), ("baseline", "data/predictions"),
    (".", "repair"),
])
def test_overlapping_output_and_dataset_paths_are_rejected_before_writes(tmp_path, paths):
    _, spec = _fixture(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        run_baseline_then_repair(spec, tmp_path / paths[0], repair_output=tmp_path / paths[1])
    assert list(tmp_path.iterdir()) == []


def test_nonprotocol_directories_are_not_adopted(tmp_path):
    _, spec = _fixture(tmp_path)
    output, repairs = tmp_path / "baseline", tmp_path / "repair"
    output.mkdir()
    (output / "unrelated.txt").write_text("preserve me")
    with pytest.raises(FileExistsError, match="no protocol"):
        run_baseline_then_repair(spec, output, repair_output=repairs)
    assert not repairs.exists()
    assert (output / "unrelated.txt").read_text() == "preserve me"


def test_existing_matching_baseline_cannot_freeze_repair_settings_posthoc(tmp_path):
    _, spec = _fixture(tmp_path)
    output, repairs = tmp_path / "baseline", tmp_path / "repair"
    baseline = replace(spec, recipes=tuple(RECIPES), repair=None)
    output.mkdir()
    write_json(output / "protocol.json", plan_study(baseline, output))
    before = fingerprint(output / "protocol.json")
    with pytest.raises(ValueError, match="Cannot adopt an existing baseline"):
        run_baseline_then_repair(spec, output, repair_output=repairs)
    assert not repairs.exists()
    assert fingerprint(output / "protocol.json") == before


def test_cli_dry_run(tmp_path):
    _, spec = _fixture(tmp_path)
    config = tmp_path / "study.json"
    write_json(config, spec.to_dict())
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert main(["--config", str(config), "--output", str(tmp_path / "baseline"),
                     "--repair-output", str(tmp_path / "repair"), "--dry-run"]) == 0
    assert json.loads(stdout.getvalue())["dry_run"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["study.json"]


@pytest.mark.parametrize("selected", [True, False])
def test_real_baselines_complete_before_repair_and_resume_without_refitting(tmp_path, selected):
    from tests.test_teacher_baselines import _TinyCifar
    import kd.teacher_baselines as baseline_module

    fixture, spec = _fixture(tmp_path)
    output, repairs = tmp_path / "baseline", tmp_path / "repair"
    phases, parent_hashes, new_students = [], {}, []
    fake_repair = fixture.fake_repair(selected)

    def repair_after_evaluation(*args, **kwargs):
        assert (output / "evaluation_completion.json").exists()
        workflow = json.loads((repairs / "workflow.json").read_text())
        assert workflow["spec"] == spec.to_dict()
        for name in ("protocol.json", "validation.json", "selection.json", "comparison.json", "evaluation_completion.json"):
            parent_hashes[name] = fingerprint(output / name)
        for checkpoint in output.glob("*/student.pt"):
            parent_hashes[checkpoint.relative_to(output).as_posix()] = fingerprint(checkpoint)
        return fake_repair(*args, **kwargs)

    real_study = baseline_module.run_study
    def baseline_after_prespecification(*args, **kwargs):
        assert (repairs / "workflow.json").exists()
        return real_study(*args, **kwargs)

    real_student = baseline_module._student_run
    def matched_student(config, **kwargs):
        if ((output / "evaluation_completion.json").exists() and not kwargs.get("verify_only", False)
                and not (output / config.name / "summary.json").exists()):
            assert config.name.startswith("kd_teacher_repair_")
            new_students.append(config.name)
        return real_student(config, **kwargs)

    real_experiment = baseline_module.run_experiment
    fitted_after_baseline = []
    def no_baseline_refit(config, *args, **kwargs):
        if (output / "evaluation_completion.json").exists():
            assert config.name.startswith("kd_teacher_repair_")
            fitted_after_baseline.append(config.name)
        return real_experiment(config, *args, **kwargs)

    with redirect_stdout(io.StringIO()), patch("kd.data.datasets.CIFAR10", _TinyCifar), \
            patch("kd.teacher_baseline_phases.run_study", side_effect=baseline_after_prespecification), \
            patch("kd.teacher_repair_baseline.run_neuron_surgery_study", side_effect=repair_after_evaluation), \
            patch("kd.teacher_baseline_phases._student_run", side_effect=matched_student), \
            patch("kd.teacher_baselines.run_experiment", side_effect=no_baseline_refit):
        report = run_baseline_then_repair(spec, output, repair_output=repairs, callback=phases.append)
    assert phases == ["baseline_train", "baseline_evaluate", "repair_and_matched_kd"]
    assert len(report["rows"]) == 12
    assert len(new_students) == (1 if selected else 0)
    assert fitted_after_baseline == new_students
    assert all(fingerprint(output / name) == digest for name, digest in parent_hashes.items())
    assert len(json.loads((output / "comparison.json").read_text())["rows"]) == 11
    if not selected:
        row = next(row for row in report["rows"] if row["recipe"] == "teacher_repair")
        assert row["paired_student_delta_vs_control"]["accuracy"]["mean"] == 0

    # Completed baseline and extension are verified without any training or evaluation.
    with patch("kd.teacher_baseline_phases.run_study", side_effect=AssertionError("baseline refit")), \
            patch("kd.teacher_baseline_phases.train_repair_teacher", side_effect=AssertionError("repair refit")), \
            patch("kd.teacher_baseline_phases._report", side_effect=AssertionError("re-evaluation")):
        resumed = run_baseline_then_repair(spec, output, repair_output=repairs)
    assert resumed == report
    assert all(fingerprint(output / name) == digest for name, digest in parent_hashes.items())
    # A changed prespecification is rejected, even if baseline model configs match.
    changed = replace(spec, repair=replace(spec.repair, repair_epochs=2))
    with pytest.raises(ValueError, match="Frozen phased baseline record changed"):
        run_baseline_then_repair(changed, output, repair_output=repairs)
    # Even a re-sealed record must obey the prespecified validation selection rule.
    selection = json.loads((repairs / "selection.json").read_text())
    selection["recipe"] = next(recipe for recipe in spec.recipes if recipe != selection["recipe"])
    write_json(repairs / "selection.json", selection)
    write_manifest(repairs, "training_completion.json",
        ("workflow.json", "protocol.json", "validation.json", "validation.md", "selection.json"))
    with patch("kd.teacher_baseline_phases._report", side_effect=AssertionError("test access")):
        with pytest.raises(ValueError, match="Selection differs from the prespecified validation rule"):
            run_baseline_then_repair(spec, output, repair_output=repairs)


def test_balanced_and_lt_configs_keep_counts_and_selected_recipe_overrides(tmp_path):
    from kd.teacher_baselines import load_spec
    for filename in ("cifar10_lt100.json", "cifar100_balanced.json"):
        spec = load_spec(Path("configs/teacher_baselines") / filename)
        result = run_baseline_then_repair(spec, tmp_path / filename, repair_output=tmp_path / (filename + "-repair"), dry_run=True)
        expected = plan_study(spec, tmp_path / filename)["counts"]["total_fits_max"]
        assert result["workflow"]["counts"]["total_fits_max"] == expected
        assert result["workflow"]["spec"] == spec.to_dict()
