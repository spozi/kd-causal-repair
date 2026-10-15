"""BF16 configuration contracts plus a hardware-gated full baseline smoke run."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from kd.evaluation import collect_predictions
from kd.losses import LogitsKD
from kd.runtime import cuda_bf16_supported, cuda_preflight
from kd.teacher_baseline_matrix import main, matrix_jobs
from kd.teacher_baselines import load_spec, main as study_main, run_study, smoke_spec, study_configs
from kd.teacher_training import RECIPES, teacher_loss


class Bfloat16BaselineTests(unittest.TestCase):
    def test_all_dataset_families_share_explicit_bfloat16_defaults(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = Path(root) / "spec.json"
            for job in matrix_jobs():
                with self.subTest(job=job["id"]):
                    config_path.write_text(json.dumps(job["study"]))
                    spec = load_spec(config_path)
                    teachers, students, supervised = study_configs(spec, Path(root) / "study")
                    configs = [*(config for _, config in teachers),
                               *(config for _, _, config in students), *supervised]
                    self.assertTrue(configs)
                    self.assertEqual({config.train.precision for config in configs}, {"bfloat16"})

    def test_cpu_debug_plan_has_an_explicit_float32_override(self):
        for job in matrix_jobs(device="cpu", precision="float32"):
            self.assertEqual(job["study"]["teacher"]["train"]["precision"], "float32")

    def test_cli_uses_same_bfloat16_default_as_python_planner(self):
        with patch("kd.teacher_baseline_matrix.materialize_matrix_plan", return_value={}) as planner, \
                redirect_stdout(io.StringIO()):
            main(["plan", "--output", "unused-bf16-plan", "--dry-run"])
        self.assertEqual(planner.call_args.kwargs["precision"], "bfloat16")
        self.assertTrue(planner.call_args.kwargs["dry_run"])

    def test_standalone_cli_can_explicitly_select_float32_for_cpu(self):
        with patch("kd.teacher_baselines.run_study", return_value={}) as runner, \
                redirect_stdout(io.StringIO()):
            study_main(["plan", "--config", "configs/teacher_baselines/cifar100_balanced.json",
                        "--output", "unused-fp32-plan", "--device", "cpu",
                        "--precision", "float32", "--dry-run"])
        spec = runner.call_args.args[0]
        self.assertEqual((spec.teacher.train.device, spec.teacher.train.precision), ("cpu", "float32"))

    def test_teacher_loss_reductions_remain_float32_for_bfloat16_logits(self):
        targets = F.one_hot(torch.tensor([0, 1]), 2).float()
        for name, recipe in RECIPES.items():
            with self.subTest(recipe=name):
                logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], dtype=torch.bfloat16,
                                      requires_grad=True)
                loss = teacher_loss(logits, targets, recipe, [3, 1])
                self.assertEqual(loss.dtype, torch.float32)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_kd_reductions_remain_float32_for_bfloat16_logits(self):
        student = torch.tensor([[80.0, -80.0], [0.5, -0.5]], dtype=torch.bfloat16,
                               requires_grad=True)
        teacher = torch.tensor([[70.0, -70.0], [-0.5, 0.5]], dtype=torch.bfloat16,
                               requires_grad=True)
        loss = LogitsKD(4)(student, teacher, torch.tensor([0, 1]))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertIsNone(teacher.grad)

    def test_bfloat16_predictions_export_float32_probabilities(self):
        class BF16Output(torch.nn.Module):
            def forward(self, images):
                return SimpleNamespace(logits=images[:, :2].to(torch.bfloat16))

        model = BF16Output().train()
        loader = DataLoader(TensorDataset(torch.tensor([[2.0, -1.0], [-0.5, 1.5]]),
                                         torch.tensor([0, 1])), batch_size=2)
        labels, probabilities = collect_predictions(model, loader, torch.device("cpu"))
        self.assertTrue(model.training)
        self.assertEqual(probabilities.dtype, np.float32)
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)


@unittest.skipUnless(torch.cuda.is_available() and cuda_bf16_supported(torch.device("cuda")),
                     "Native CUDA BF16 is required; CPU tests do not validate CUDA kernels")
class CudaBfloat16BaselineTests(unittest.TestCase):
    def test_real_cuda_autocast_backward_and_optimizer(self):
        result = cuda_preflight("cuda:0", "bfloat16", batch_size=4, iterations=2)
        self.assertEqual(result["runtime"]["precision"], "bfloat16")
        self.assertEqual(result["output_dtype"], "torch.bfloat16")
        self.assertFalse(result["grad_scaler_enabled"])

    def test_all_teacher_recipes_supervised_kd_and_test_export(self):
        with tempfile.TemporaryDirectory() as root:
            base = smoke_spec()
            spec = replace(base, recipes=tuple(RECIPES), teacher=replace(base.teacher,
                train=replace(base.teacher.train, device="cuda:0", precision="bfloat16")))
            with redirect_stdout(io.StringIO()):
                run_study(spec, root)
                run_study(spec, root, phase="evaluate")
            summaries = list(Path(root).glob("*/summary.json"))
            self.assertEqual(len(summaries), 23)  # 11 teachers + 11 KD + supervised
            for path in summaries:
                summary = json.loads(path.read_text())
                self.assertEqual(summary["runtime"]["precision"], "bfloat16")
                state = torch.load(path.parent / "student.pt", map_location="cpu", weights_only=True)
                self.assertEqual({value.dtype for value in state["student"].values()
                                  if value.is_floating_point()}, {torch.float32})
            predictions = list((Path(root) / "test").rglob("predictions.npz"))
            self.assertEqual(len(predictions), 23)
            for path in predictions:
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(archive["probabilities"].dtype, np.float32)
                    self.assertTrue(np.isfinite(archive["probabilities"]).all())


if __name__ == "__main__":
    unittest.main()
