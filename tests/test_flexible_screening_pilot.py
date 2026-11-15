"""Preparation is local-only, source-frozen and paired; no GPU experiments."""

from dataclasses import replace
import json
from pathlib import Path
import tarfile
from unittest.mock import patch

import pytest

from kd.neuron_surgery_study import load_neuron_surgery_spec
from scripts import prepare_flexible_screening_pilot as preparation
from scripts.flexible_screening_pilot_preflight import verify_snapshot


def test_pilot_snapshot_contains_four_paired_configs_and_no_training_or_external_material(tmp_path):
    root = tmp_path / "pilot"
    with patch("kd.neuron_surgery_study.run_neuron_surgery_study", side_effect=AssertionError("training")):
        receipt = preparation.prepare(root)
    assert receipt["jobs"] == 4
    assert receipt["maximum_candidate_fits"] == 32
    assert receipt["maximum_new_student_fits"] == 12
    assert not receipt["training_started"] and not receipt["remote_contacted"]
    manifest = verify_snapshot(root)
    assert "code/kd/repair_screening.py" in manifest
    assert "code/kd/catalog/v1.json" in manifest
    assert "code/pyproject.toml" in manifest
    assert "preflight.py" in manifest
    plan = json.loads((root / "experiment.json").read_text())
    assert len({job["output"] for job in plan["jobs"]}) == 4
    for i in (0, 2):
        strict_job, expanded_job = plan["jobs"][i:i + 2]
        strict = load_neuron_surgery_spec(root / strict_job["config"])
        expanded = load_neuron_surgery_spec(root / expanded_job["config"])
        assert strict.screening_policy == "strict"
        assert expanded.screening_policy == "screened_then_ranked"
        assert replace(expanded, name=strict.name, screening_policy="strict") == strict
        assert strict_job["baseline"] == expanded_job["baseline"]
    assert preparation.sha(root / "bundle.tar.gz") == receipt["bundle_sha256"]
    with tarfile.open(root / "bundle.tar.gz") as archive:
        names = archive.getnames()
        assert set(names) == set(manifest) | {"manifest.json"}
        assert all(m.isfile() for m in archive)
        assert not any(name.endswith((".pt", ".pth", ".npz")) or ".git/" in name
                       or "proposal/" in name or "app/" in name for name in names)
    # Shell syntax is checked without executing it (no preflight and no launch).
    import subprocess
    subprocess.run(["bash", "-n", str(root / "run-serial.sh")], check=True)
    commands = (root / "run-serial.sh").read_text()
    assert commands.count('"$PYTHON" -m kd neuron-surgery-study') == 4
    assert "teacher-baselines" not in commands
    assert "export PYTHONPATH" in commands


def test_preparation_refuses_existing_directory_without_modifying_it(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("untouched")
    with pytest.raises(FileExistsError):
        preparation.prepare(output)
    assert sentinel.read_text() == "untouched"
    assert list(output.iterdir()) == [sentinel]


def test_preflight_snapshot_fails_closed_on_source_or_config_tampering(tmp_path):
    output = tmp_path / "pilot"
    preparation.prepare(output)
    config = next((output / "configs").glob("*.json"))
    config.write_text("{}")
    with pytest.raises(ValueError, match="snapshot changed"):
        verify_snapshot(output)


@pytest.mark.parametrize("change", ["missing_model_pin", "invalid_hash", "missing_data_pin", "wrong_policies"])
def test_invalid_pilot_definition_fails_before_any_outputs(tmp_path, change):
    plan = json.loads(preparation.DEFINITION.read_text())
    if change == "missing_model_pin":
        del plan["cases"][0]["expected_model_sha256"]["teacher_control_seed41"]
    elif change == "invalid_hash":
        plan["cases"][0]["expected_model_sha256"]["teacher_control_seed41"] = "invalid"
    elif change == "missing_data_pin":
        del plan["cases"][0]["expected_data"]["train_index_sha256"]
    else:
        plan["policies"] = ["strict"]
    definition = tmp_path / "definition.json"
    definition.write_text(json.dumps(plan))
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        preparation.prepare(output, definition)
    assert not output.exists()
