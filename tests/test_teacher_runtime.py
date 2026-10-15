"""Preflight and live-data identity guards for the teacher baseline study."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from kd.checkpoints import metadata, save_checkpoint, write_json
from kd.data import build_data
from kd.models import create_model
from kd.teacher_baselines import _validate_checkpoint, run_study, smoke_spec


class TeacherRuntimePreflightTests(unittest.TestCase):
    def spec(self, precision="auto"):
        base = smoke_spec()
        return replace(base, recipes=("control", "sam"), teacher=replace(
            base.teacher, train=replace(base.teacher.train, precision=precision, device="auto")))

    def test_explicit_float16_sam_is_invalid_without_resolving_a_device(self):
        with patch("kd.teacher_baselines.resolve_device", side_effect=AssertionError("device access")):
            with self.assertRaisesRegex(ValueError, "SAM supports float32/bfloat16"):
                self.spec("float16").validate()

    def test_auto_cuda_float16_rejects_before_writes_data_or_training(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "study"
            device = torch.device("cuda:0")
            with patch("kd.teacher_baselines.resolve_device", return_value=device), \
                    patch("kd.teacher_baselines.resolve_precision", return_value="float16") as precision, \
                    patch("kd.teacher_baselines.train_teacher", side_effect=AssertionError("training")), \
                    patch("kd.teacher_baselines.build_data", side_effect=AssertionError("data access")), \
                    patch("kd.teacher_training.build_data", side_effect=AssertionError("data access")):
                with self.assertRaisesRegex(ValueError, "SAM supports float32/bfloat16"):
                    run_study(self.spec(), output)
            precision.assert_called_once_with("auto", device)
            self.assertFalse(output.exists())

    def test_supported_precision_reaches_protocol_freeze(self):
        for requested, resolved in (("float32", "float32"), ("auto", "bfloat16"),
                                    ("bfloat16", "bfloat16")):
            with self.subTest(precision=requested), tempfile.TemporaryDirectory() as root:
                with patch("kd.teacher_baselines.resolve_device", return_value=torch.device("cuda:0")), \
                        patch("kd.teacher_baselines.resolve_precision", return_value=resolved), \
                        patch("kd.teacher_baselines._freeze_protocol", side_effect=RuntimeError("preflight passed")):
                    with self.assertRaisesRegex(RuntimeError, "preflight passed"):
                        run_study(self.spec(requested), Path(root) / "study")

    def test_unsupported_explicit_bfloat16_rejects_before_any_writes(self):
        for device in (torch.device("cpu"), torch.device("cuda:0")):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "study"
                with patch("kd.teacher_baselines.resolve_device", return_value=device), \
                        patch("kd.runtime.cuda_bf16_supported", return_value=False), \
                        patch("kd.teacher_baselines._freeze_protocol", side_effect=AssertionError("writes")), \
                        patch("kd.teacher_baselines.build_data", side_effect=AssertionError("data access")):
                    with self.assertRaisesRegex(ValueError, "bfloat16"):
                        run_study(self.spec("bfloat16"), output)
                self.assertFalse(output.exists())

    def test_plan_and_dry_run_do_not_resolve_runtime(self):
        with tempfile.TemporaryDirectory() as root, \
                patch("kd.teacher_baselines.resolve_device", side_effect=AssertionError("device access")), \
                patch("kd.teacher_baselines.resolve_precision", side_effect=AssertionError("precision access")):
            for phase in ("plan", "train", "evaluate"):
                output = Path(root) / f"dry-{phase}"
                result = run_study(self.spec(), output, phase=phase, dry_run=True)
                self.assertTrue(result["dry_run"])
                self.assertFalse(output.exists())
            result = run_study(self.spec(), Path(root) / "plan", phase="plan")
            self.assertEqual(result["status"], "planned")


class TeacherEvaluationDataIdentityTests(unittest.TestCase):
    def config(self, root):
        config = replace(smoke_spec().teacher, output_dir=root, name="teacher")
        data = build_data(config.data, config.train, include_test=False)
        model = create_model(config.student.name, config.data.num_classes)
        directory = Path(root) / config.name
        directory.mkdir()
        save_checkpoint(directory / "student.pt", {
            **metadata(config.student.name, data.classes, config.data.image_size, config.data.source),
            "kind": "inference", "epoch": 0, "student": model.state_dict()})
        write_json(directory / "summary.json", {"data": data.provenance})
        return config

    def test_changed_live_provenance_rejects_before_predictions(self):
        changes = ({"source": "different-source"},
                   {"val_index_sha256": "different-validation-indices"},
                   {"artifact_sha256": {"archive": "different-archive"}},
                   {"split_sizes": {"train": 32, "val": 15}})
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root:
                config = self.config(root)

                def changed(*args, **kwargs):
                    bundle = build_data(*args, **kwargs)
                    bundle.provenance.update(change)
                    return bundle

                with patch("kd.teacher_baselines.build_data", side_effect=changed), \
                        patch("kd.teacher_baselines.collect_predictions", side_effect=AssertionError("predictions")):
                    with self.assertRaisesRegex(ValueError, "frozen training/validation provenance"):
                        _validate_checkpoint(config, targets=(0, 1, 2, 3))

    def test_test_and_confirmation_only_additions_are_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            config = self.config(root)

            def evaluation_only(*args, **kwargs):
                bundle = build_data(*args, **kwargs)
                bundle.provenance["split_sizes"]["confirmation"] = 8
                bundle.provenance.update(
                    test_index_sha256="test-only-indices", test_per_class=[4] * 4,
                    test_policy="explicit final test", confirmation_index_sha256="confirmation-only-indices",
                    confirmation_per_class=[2] * 4, confirmation_policy="sealed holdout")
                return bundle

            with patch("kd.teacher_baselines.build_data", side_effect=evaluation_only), \
                    patch("kd.teacher_baselines.collect_predictions", return_value=(None, None)) as collect, \
                    patch("kd.teacher_baselines.prediction_metrics", return_value={
                        "accuracy": 0.5, "per_class": [{"recall": 0.5}] * 4}):
                metrics = _validate_checkpoint(config, split="test", targets=(0, 1, 2, 3))
            collect.assert_called_once()
            self.assertEqual(metrics["target_macro_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
