from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kd.cli import main as cli_main
from kd.checkpoints import write_json
from kd.config import (DataConfig, DistillationConfig, ExperimentConfig, ModelConfig,
                       TrainConfig)
from kd.neuron_surgery_study import NeuronSurgerySpec
from kd.student_surgery_study import _direct_spec, _source_inputs, _supervised_config


class StudentSurgeryConfigurationTests(unittest.TestCase):
    def test_new_teacher_budget_constraints_are_not_silently_reused_for_student_surgery(self):
        source = NeuronSurgerySpec(student_seeds=(142,), confirmation_fraction=0.1,
                                   evaluation_split="confirmation")
        for changed in (replace(source, parameter_budget_fractions=(0.008,),
                                max_preservation_kl=0.02),
                        replace(source, max_preservation_kl=0.02)):
            with self.subTest(spec=changed), self.assertRaisesRegex(
                    ValueError, "not implemented for the direct-student comparison"):
                _direct_spec(changed, 142, "supervised_seed142")
            with tempfile.TemporaryDirectory() as root:
                directory = Path(root)
                (directory / "studies" / "confirmatory").mkdir(parents=True)
                write_json(directory / "progress.json", {"phase": "complete"})
                write_json(directory / "studies" / "confirmatory" / "comparison.json",
                           {"status": "complete", "success": {"passed": True},
                            "study": changed.to_dict()})
                with self.assertRaisesRegex(ValueError, "not implemented for the direct-student comparison"):
                    _source_inputs(directory, (142,))

    def test_direct_spec_disables_all_distillation(self):
        source = NeuronSurgerySpec(
            name="source", teacher_run="teacher", student_run_template="kd_seed{seed}",
            student_seeds=(142,), confirmation_fraction=0.1,
            evaluation_split="confirmation", expected_train_counts=None,
            expected_target_counts=None)
        spec = _direct_spec(source, 142, "supervised_seed142")
        self.assertEqual(spec.teacher_model, "cifar_student")
        self.assertEqual(spec.repair_subject, "student")
        self.assertEqual(spec.kd_weight, 0.0)
        self.assertEqual(spec.feature_weight, 0.0)
        self.assertEqual(spec.preservation_ce_weight, 1.0)
        self.assertFalse(spec.downstream_kd)

    def test_supervised_config_preserves_matched_seed_and_recipe(self):
        with tempfile.TemporaryDirectory() as root:
            source = ExperimentConfig(
                name="kd_seed142", output_dir=root,
                data=DataConfig(source="cifar10", root="data", num_classes=10,
                                image_size=32, imbalance_factor=100,
                                confirmation_fraction=0.1),
                student=ModelConfig("cifar_student"),
                teacher=ModelConfig("cifar_teacher", "teacher.pt"),
                distillation=DistillationConfig(method="kd"),
                train=TrainConfig(seed=142, epochs=81))
            result = _supervised_config(source, Path(root) / "supervised", 142, "cpu")
        self.assertEqual(result.distillation.method, "supervised")
        self.assertEqual(result.student.name, "cifar_student")
        self.assertIsNone(result.teacher.checkpoint)
        self.assertEqual(result.train.seed, 142)
        self.assertEqual(result.train.device, "cpu")

    def test_cli_exposes_student_surgery_dry_run(self):
        result = {"dry_run": True}
        with patch("kd.student_surgery_study.run_student_surgery_study",
                   return_value=result) as run, redirect_stdout(io.StringIO()) as output:
            cli_main(["student-surgery-study", "--source", "source", "--output", "output",
                      "--device", "cpu", "--seeds", "1", "2", "3", "--dry-run"])
        run.assert_called_once_with("source", "output", "cpu", (1, 2, 3), dry_run=True)
        self.assertIn('"dry_run": true', output.getvalue())


if __name__ == "__main__":
    unittest.main()
