"""Adopted baseline contracts and standalone repair protocol coverage."""

import ast
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kd.checkpoints import fingerprint, write_json
from kd.config import CalibrationConfig, CPCConfig, SupervisedLossConfig
from kd.neuron_surgery_baselines import baseline_configs
from kd.neuron_surgery_study import (
    COMPUTE_SOURCES, NeuronSurgerySpec, _load_inputs, _protocol,
    run_neuron_surgery_study,
)


def adopted_baseline(root, spec):
    """Minimal saved artifacts; adoption reads identities, not model tensors."""
    root = Path(root)
    teacher, students = baseline_configs(root, root / "data", "cpu", spec)
    teacher_directory = root / teacher.name
    teacher_directory.mkdir(parents=True)
    write_json(teacher_directory / "config.json", teacher.to_dict())
    (teacher_directory / "student.pt").write_bytes(b"completed teacher identity")
    teacher_digest = fingerprint(teacher_directory / "student.pt")
    for config in students:
        directory = root / config.name
        directory.mkdir()
        write_json(directory / "config.json", config.to_dict())
        (directory / "student.pt").write_bytes(f"completed student {config.train.seed}".encode())
        write_json(directory / "summary.json", {
            "initial_student_sha256": f"initial-student-{config.train.seed}",
            "teacher": {"checkpoint_sha256": teacher_digest},
        })
    return teacher, students


class RepairBaselineContractTests(unittest.TestCase):
    def test_plain_static_and_neutral_generic_baselines_remain_accepted(self):
        specs = (
            NeuronSurgerySpec(),
            NeuronSurgerySpec(
                dataset_source="bloodmnist", dataset_num_classes=8,
                dataset_version="2.0", dataset_profile="lt-if10", imbalance_factor=10,
                teacher_model="convnet6_w32", student_model="convnet6_w16",
                target_classes=None, target_class_policy="frequency_tail",
                expected_train_counts=None, expected_target_counts=None,
                allow_sparse_class_fallback=True),
        )
        for spec in specs:
            with self.subTest(source=spec.dataset_source), tempfile.TemporaryDirectory() as root:
                teacher, _ = adopted_baseline(root, spec)
                _, adopted, _, students = _load_inputs(root, spec.student_seeds, spec)
                self.assertEqual(adopted, teacher)
                self.assertEqual(tuple(students), spec.student_seeds)

    def test_adopted_kd_controls_reject_undeclared_loss_interventions(self):
        interventions = (
            {"calibration": CalibrationConfig(enabled=True)},
            {"cpc": CPCConfig(enabled=True)},
            {"supervised_loss": SupervisedLossConfig(method="adafocal")},
            {"supervised_loss": SupervisedLossConfig(method="adadualfocal")},
        )
        for intervention in interventions:
            with self.subTest(intervention=intervention), tempfile.TemporaryDirectory() as root:
                spec = NeuronSurgerySpec()
                _, students = adopted_baseline(root, spec)
                config = replace(students[0], **intervention)
                # These combinations are valid engine configurations, but not
                # valid classical-KD controls for this repair experiment.
                config.validate()
                path = Path(root) / config.name / "config.json"
                write_json(path, config.to_dict())
                before = fingerprint(path)
                with self.assertRaisesRegex(ValueError, "Seed 42 KD baseline requires plain cross-entropy"):
                    _load_inputs(root, spec.student_seeds, spec)
                self.assertEqual(fingerprint(path), before)

    def test_adopted_teacher_rejects_undeclared_loss_interventions(self):
        interventions = (
            {"calibration": CalibrationConfig(enabled=True)},
            {"cpc": CPCConfig(enabled=True)},
            {"supervised_loss": SupervisedLossConfig(method="adafocal")},
            {"supervised_loss": SupervisedLossConfig(method="adadualfocal")},
        )
        for intervention in interventions:
            with self.subTest(intervention=intervention), tempfile.TemporaryDirectory() as root:
                spec = NeuronSurgerySpec()
                teacher, _ = adopted_baseline(root, spec)
                config = replace(teacher, **intervention)
                path = Path(root) / config.name / "config.json"
                write_json(path, config.to_dict())
                before = fingerprint(path)
                # Supervised+CPC is already rejected by generic config parsing;
                # valid supervised intervention combinations hit the repair guard.
                with self.assertRaisesRegex(
                        ValueError, "Teacher baseline requires plain cross-entropy|CPC currently requires"):
                    _load_inputs(root, spec.student_seeds, spec)
                self.assertEqual(fingerprint(path), before)

    def test_disabled_intervention_settings_do_not_change_the_contract(self):
        with tempfile.TemporaryDirectory() as root:
            spec = NeuronSurgerySpec()
            teacher, students = adopted_baseline(root, spec)
            for config in (teacher, *students):
                config = replace(
                    config, calibration=CalibrationConfig(enabled=False, max_weight=2.0),
                    cpc=CPCConfig(enabled=False, discrimination_weight=0.3),
                    supervised_loss=SupervisedLossConfig(method="cross_entropy", bins=7))
                write_json(Path(root) / config.name / "config.json", config.to_dict())
            _, _, _, adopted = _load_inputs(root, spec.student_seeds, spec)
            self.assertEqual(tuple(adopted), spec.student_seeds)

    def test_present_teacher_path_does_not_override_recorded_training_hash(self):
        with tempfile.TemporaryDirectory() as root:
            spec = NeuronSurgerySpec()
            _, students = adopted_baseline(root, spec)
            config = students[0]
            self.assertTrue(Path(config.teacher.checkpoint).is_file())
            summary_path = Path(root) / config.name / "summary.json"
            write_json(summary_path, {
                "teacher": {"checkpoint_sha256": "different-training-teacher"},
            })
            before = fingerprint(summary_path)
            with self.assertRaisesRegex(ValueError, "recorded training teacher differs"):
                _load_inputs(root, spec.student_seeds, spec)
            self.assertEqual(fingerprint(summary_path), before)

    def test_missing_recorded_training_hash_is_rejected_even_with_present_teacher(self):
        for summary in ({}, {"teacher": {}}, {"teacher": None}, {"teacher": []},
                        {"teacher": "invalid"}, {"teacher": True}):
            with self.subTest(summary=summary), tempfile.TemporaryDirectory() as root:
                spec = NeuronSurgerySpec()
                _, students = adopted_baseline(root, spec)
                self.assertTrue(Path(students[0].teacher.checkpoint).is_file())
                summary_path = Path(root) / students[0].name / "summary.json"
                write_json(summary_path, summary)
                before = fingerprint(summary_path)
                with self.assertRaisesRegex(ValueError, "recorded training teacher differs"):
                    _load_inputs(root, spec.student_seeds, spec)
                self.assertEqual(fingerprint(summary_path), before)

    def test_present_teacher_checkpoint_must_match_recorded_training_identity(self):
        with tempfile.TemporaryDirectory() as root:
            spec = NeuronSurgerySpec()
            _, students = adopted_baseline(root, spec)
            source = Path(root) / "different_teacher.pt"
            source.write_bytes(b"different teacher currently at the recorded path")
            config = replace(students[0], teacher=replace(
                students[0].teacher, checkpoint=str(source)))
            config_path = Path(root) / config.name / "config.json"
            write_json(config_path, config.to_dict())
            before = fingerprint(source)
            with self.assertRaisesRegex(ValueError, "teacher checkpoint differs"):
                _load_inputs(root, spec.student_seeds, spec)
            self.assertEqual(fingerprint(source), before)

    def test_relocated_missing_teacher_path_uses_recorded_training_identity(self):
        with tempfile.TemporaryDirectory() as root:
            spec = NeuronSurgerySpec()
            _, students = adopted_baseline(root, spec)
            old_teacher_path = Path(root) / "unavailable_old_host" / "teacher.pt"
            self.assertFalse(old_teacher_path.exists())
            for config in students:
                config = replace(config, teacher=replace(
                    config.teacher, checkpoint=str(old_teacher_path)))
                write_json(Path(root) / config.name / "config.json", config.to_dict())
            _, _, _, adopted = _load_inputs(root, spec.student_seeds, spec)
            self.assertEqual(tuple(adopted), spec.student_seeds)

    def test_repair_only_anchors_enforce_plain_ce_without_student_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            spec = replace(NeuronSurgerySpec(), downstream_kd=False)
            teacher, _ = adopted_baseline(root, spec)
            teacher = replace(teacher, calibration=CalibrationConfig(enabled=True))
            write_json(Path(root) / teacher.name / "config.json", teacher.to_dict())
            with self.assertRaisesRegex(ValueError, "Teacher baseline requires plain cross-entropy"):
                _load_inputs(root, spec.student_seeds, spec)


class RepairSourceCoverageTests(unittest.TestCase):
    def test_local_import_closure_is_frozen_by_standalone_protocol(self):
        source_root = Path("kd")
        pending, seen = ["neuron_surgery_study.py"], set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            tree = ast.parse((source_root / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                    dependency = node.module.replace(".", "/") + ".py"
                    if (source_root / dependency).is_file():
                        pending.append(dependency)
        self.assertFalse(seen - set(COMPUTE_SOURCES),
                         f"Unfrozen repair computational dependencies: {seen - set(COMPUTE_SOURCES)}")

    def test_protocol_hashes_the_training_data_and_reporting_dependencies(self):
        required = {
            "config.py", "data.py", "dataset_registry.py", "catalog/v1.json",
            "checkpoints.py", "engine.py", "training_provenance.py", "surgery.py",
            "losses.py", "calibration.py", "adaptive_focal.py", "cpc.py", "models.py",
            "runtime.py", "metrics.py", "evaluation.py", "calibration_metrics.py",
            "neuron_surgery.py", "neuron_surgery_study.py",
        }
        self.assertTrue(required <= set(COMPUTE_SOURCES))
        self.assertEqual(len(COMPUTE_SOURCES), len(set(COMPUTE_SOURCES)))
        with tempfile.TemporaryDirectory() as root:
            spec = NeuronSurgerySpec()
            adopted_baseline(root, spec)
            baseline, teacher, checkpoint, students = _load_inputs(root, spec.student_seeds, spec)
            protocol = _protocol(baseline, Path(root) / "new-study", "cpu",
                                 teacher, checkpoint, students, spec)
            for name in required:
                self.assertEqual(protocol["source_sha256"][name], fingerprint(Path("kd") / name))

    def test_execution_rejects_incomplete_historical_protocol_without_rewriting(self):
        with tempfile.TemporaryDirectory() as root:
            spec = NeuronSurgerySpec()
            adopted_baseline(root, spec)
            baseline, teacher, checkpoint, students = _load_inputs(root, spec.student_seeds, spec)
            output = Path(root) / "historical-study"
            output.mkdir()
            protocol = _protocol(baseline, output, "cpu", teacher, checkpoint, students, spec)
            protocol["source_sha256"] = {
                name: protocol["source_sha256"][name]
                for name in ("config.py", "data.py", "neuron_surgery.py",
                             "neuron_surgery_study.py", "losses.py", "models.py", "runtime.py")
            }
            write_json(output / "protocol.json", protocol)
            write_json(output / "report_manifest.json", {"historical": "preserve"})
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with patch("kd.neuron_surgery_study._data_available"), \
                    patch("kd.neuron_surgery_study.build_data", side_effect=AssertionError("data access")):
                with self.assertRaisesRegex(ValueError, "protocol or computational source changed"):
                    run_neuron_surgery_study(baseline, output, "cpu", spec.student_seeds, spec=spec)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)

    def test_changed_engine_hash_changes_new_protocol_identity(self):
        with tempfile.TemporaryDirectory() as root:
            spec = NeuronSurgerySpec()
            adopted_baseline(root, spec)
            baseline, teacher, checkpoint, students = _load_inputs(root, spec.student_seeds, spec)
            output = Path(root) / "new-study"
            original = _protocol(baseline, output, "cpu", teacher, checkpoint, students, spec)
            def changed_source(path):
                return "0" * 64 if Path(path).name == "engine.py" else fingerprint(path)
            with patch("kd.neuron_surgery_study.fingerprint", side_effect=changed_source):
                changed = _protocol(baseline, output, "cpu", teacher, checkpoint, students, spec)
            self.assertNotEqual(original, changed)
            self.assertEqual(changed["source_sha256"]["engine.py"], "0" * 64)


if __name__ == "__main__":
    unittest.main()
