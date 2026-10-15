"""CPU-only tests for non-resumable repair failure artifact storage."""

import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from kd.checkpoints import fingerprint, restore_rng, save_checkpoint
from kd.models import ModelOutput
from kd.repair_failure import RepairFailureRecorder


class RepairFailureStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "failures"
        self.candidate = torch.nn.Linear(2, 2)
        self.anchor = torch.nn.Linear(2, 2).requires_grad_(False)
        self.optimizer = torch.optim.SGD(self.candidate.parameters(), lr=0.1, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.images = torch.tensor([[1.0, 2.0]])
        self.candidate(self.images).sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.recorder = RepairFailureRecorder(self.directory, context={"source": "unit-test"})

    def begin(self):
        self.recorder.begin_step(self.candidate, self.optimizer, self.scheduler, None,
                                 epoch=2, step=3, global_step=8,
                                 samplers={"target": {"consumed": 1}})
        self.recorder.before_forward()

    def save(self, error=None, **overrides):
        args = dict(phase="loss", candidate=self.candidate, anchor=self.anchor,
                    optimizer=self.optimizer, scheduler=self.scheduler, scaler=None,
                    losses={"total": torch.tensor(float("nan")),
                            "kd": float("inf"), "feature": float("-inf")},
                    batches={"target": {"images": self.images,
                                        "labels": torch.tensor([1]), "indices": [4]}},
                    outputs={"target": ModelOutput(torch.tensor([[0.0, float("nan")]]),
                                                    {"feature": torch.ones(1, 2, 3, 3)})},
                    history=[{"loss": float("inf")}], masks={"layer": torch.tensor([True])},
                    settings={"learning_rate": 0.1})
        args.update(overrides)
        return self.recorder.save(error or FloatingPointError("non-finite repair loss"), **args)

    def test_nonfinite_receipt_and_weights_only_payload(self):
        self.begin()
        snapshot = self.save()
        receipt = json.loads((snapshot / "failure.json").read_text(),
                             parse_constant=lambda value: self.fail(f"Non-standard JSON: {value}"))
        state = torch.load(snapshot / "state.pt", weights_only=True, map_location="cpu")
        self.assertEqual(receipt["kind"], "repair_failure_snapshot")
        self.assertEqual(receipt["capture_status"], "committed")
        self.assertFalse(receipt["resumable"])
        self.assertEqual(receipt["losses"]["total"]["value"], "nan")
        self.assertEqual(receipt["losses"]["kd"], "+inf")
        self.assertEqual(receipt["losses"]["feature"], "-inf")
        self.assertEqual(receipt["checkpoint_sha256"], fingerprint(snapshot / "state.pt"))
        self.assertEqual(receipt["pre_step_checkpoint_sha256"], fingerprint(snapshot / "pre_step.pt"))
        self.assertEqual(receipt["pre_step_checkpoint_scope"], "pre_step_only")
        self.assertFalse(receipt["pre_step_checkpoint_resumable"])
        logits = state["outputs"]["target"]["logits"]
        self.assertEqual(logits["nan_count"], 1)
        self.assertEqual(logits["min"], 0.0)
        self.assertEqual(logits["max"], 0.0)
        self.assertEqual(logits["l2_norm"], "nan")
        self.assertEqual(receipt["diagnostics"]["outputs"], state["outputs"])
        self.assertAlmostEqual(receipt["diagnostics"]["batches"]["target"]["images"]["l2_norm"], math.sqrt(5))
        self.assertIsInstance(state["outputs"]["target"]["features"]["feature"], dict)
        self.assertTrue(torch.isnan(state["losses"]["total"]))
        self.assertNotIn("student", state)
        self.assertFalse((snapshot / "completion.json").exists())

    def test_pre_step_detaches_model_optimizer_and_sampler_aliases(self):
        samplers = {"target": {"generator_state": torch.tensor([1, 2]), "epoch_draws": [0, 1]}}
        self.recorder.begin_step(self.candidate, self.optimizer, self.scheduler, None,
                                 epoch=1, step=1, global_step=1, samplers=samplers)
        before = self.candidate.weight.detach().clone()
        buffer = self.optimizer.state[self.candidate.weight]["momentum_buffer"]
        before_buffer = buffer.clone()
        with torch.no_grad():
            self.candidate.weight.add_(10)
            buffer.add_(20)
            samplers["target"]["generator_state"].fill_(9)
            samplers["target"]["epoch_draws"].append(9)
        self.candidate(self.images).sum().backward()
        snapshot = self.save()
        state = torch.load(snapshot / "state.pt", weights_only=True)
        self.assertTrue(torch.equal(state["pre_step"]["candidate"]["weight"], before))
        self.assertTrue(torch.equal(state["pre_step"]["optimizer"]["state"][0]["momentum_buffer"], before_buffer))
        self.assertEqual(state["pre_step"]["samplers"]["target"]["epoch_draws"], [0, 1])
        self.assertEqual(state["pre_step"]["samplers"]["target"]["generator_state"].tolist(), [1, 2])
        self.assertTrue(torch.equal(state["observed"]["candidate"]["weight"], before + 10))
        self.assertTrue(torch.equal(state["observed"]["gradients"]["weight"], self.candidate.weight.grad))
        self.assertEqual(state["batches"]["target"]["indices"], [4])

    def test_unique_failure_directories_never_overwrite(self):
        self.begin()
        first = self.save()
        original_digest = fingerprint(first / "state.pt")
        second = self.save()
        self.assertNotEqual(first, second)
        self.assertEqual(fingerprint(first / "state.pt"), original_digest)
        self.assertEqual(len(list(self.directory.iterdir())), 2)

    def test_capture_has_no_writes_and_rng_can_replay(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        self.begin()
        self.assertFalse(self.directory.exists())
        expected = (random.random(), float(np.random.random()), torch.rand(3))
        snapshot = self.save()
        state = torch.load(snapshot / "state.pt", weights_only=True)
        restore_rng(state["pre_step"]["before_forward_rng"], {})
        observed = (random.random(), float(np.random.random()), torch.rand(3))
        self.assertEqual(expected[:2], observed[:2])
        self.assertTrue(torch.equal(expected[2], observed[2]))

    def test_checkpoint_io_failure_retains_preliminary_receipt_and_original_error(self):
        self.begin()
        original = FloatingPointError("original failure")
        def fail_full_checkpoint(path, state):
            if path.name == "state.pt":
                raise OSError("disk full")
            save_checkpoint(path, state)
        with patch("kd.repair_failure.save_checkpoint", side_effect=fail_full_checkpoint):
            self.assertIsNone(self.save(error=original))
        receipt_path, = self.directory.glob("*/failure.json")
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["capture_status"], "capture_failed")
        self.assertEqual(receipt["capture_error"], {"type": "OSError", "message": "disk full"})
        self.assertEqual(receipt["exception"]["message"], "original failure")
        self.assertEqual(receipt["losses"]["total"]["value"], "nan")
        self.assertTrue(any("disk full" in note for note in original.__notes__))
        self.assertEqual(str(original), "original failure")

    def test_unwritable_directory_does_not_mask_original_error(self):
        self.directory.write_text("not a directory")
        self.begin()
        original = RuntimeError("original")
        self.assertIsNone(self.save(error=original))
        self.assertTrue(any("snapshot could not be completed" in note for note in original.__notes__))

    def test_failed_pre_step_capture_is_recorded_not_raised(self):
        with patch("kd.repair_failure.capture_rng", side_effect=RuntimeError("rng unavailable")):
            self.begin()
        snapshot = self.save()
        state = torch.load(snapshot / "state.pt", weights_only=True)
        self.assertIsNone(state["pre_step"])
        self.assertIn("rng unavailable", state["capture_errors"][0])

    def test_finite_gradient_norm_uses_float64_not_overflowing_float32(self):
        self.begin()
        self.candidate.weight.grad = torch.full_like(self.candidate.weight, 1e30)
        self.candidate.bias.grad = torch.full_like(self.candidate.bias, 1e30)
        self.assertTrue(torch.isinf(self.candidate.weight.grad.norm()))
        snapshot = self.save(losses={"total": torch.tensor(0.25)})
        receipt = json.loads((snapshot / "failure.json").read_text())
        state = torch.load(snapshot / "state.pt", weights_only=True)
        diagnostics = receipt["diagnostics"]
        self.assertTrue(math.isfinite(diagnostics["global_gradient_l2"]))
        self.assertAlmostEqual(diagnostics["global_gradient_l2"] / 1e30, math.sqrt(6), places=6)
        self.assertAlmostEqual(diagnostics["gradients"]["weight"]["l2_norm"] / 1e30, 2, places=6)
        self.assertEqual(diagnostics["nonfinite_candidate_tensors"], [])
        self.assertEqual(diagnostics["nonfinite_optimizer_momentum"], [])
        self.assertEqual(state["diagnostics"], diagnostics)

    def test_nonfinite_gradient_and_momentum_names_are_json_safe(self):
        self.begin()
        self.candidate.weight.grad = torch.full_like(self.candidate.weight, float("inf"))
        self.candidate.bias.grad = torch.full_like(self.candidate.bias, float("nan"))
        with torch.no_grad():
            self.candidate.weight[0, 0] = float("nan")
            self.optimizer.state[self.candidate.weight]["momentum_buffer"][0, 0] = float("inf")
        snapshot = self.save()
        receipt = json.loads((snapshot / "failure.json").read_text(),
                             parse_constant=lambda value: self.fail(f"Non-standard JSON: {value}"))
        diagnostics = receipt["diagnostics"]
        self.assertEqual(diagnostics["global_gradient_l2"], "nan")
        self.assertEqual(diagnostics["gradients"]["weight"]["l2_norm"], "+inf")
        self.assertEqual(diagnostics["gradients"]["bias"]["l2_norm"], "nan")
        self.assertEqual(diagnostics["nonfinite_candidate_tensors"], ["weight"])
        self.assertEqual(diagnostics["nonfinite_optimizer_momentum"], ["weight.momentum_buffer"])

    def test_preliminary_receipt_survives_state_inspection_failure(self):
        self.begin()
        original = RuntimeError("original computation failure")
        with patch.object(self.candidate, "state_dict", side_effect=RuntimeError("device poisoned")):
            self.assertIsNone(self.save(error=original))
        receipt_path, = self.directory.glob("*/failure.json")
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["capture_status"], "capture_failed")
        self.assertEqual(receipt["exception"]["message"], "original computation failure")
        self.assertIn("device poisoned", receipt["capture_error"]["message"])
        self.assertFalse((receipt_path.parent / "state.pt").exists())
        fallback = receipt_path.parent / receipt["pre_step_checkpoint"]
        self.assertEqual(receipt["pre_step_checkpoint_sha256"], fingerprint(fallback))
        state = torch.load(fallback, weights_only=True, map_location="cpu")
        self.assertTrue(torch.equal(state["pre_step"]["candidate"]["weight"],
                                    self.recorder._pre_step["candidate"]["weight"]))
        self.assertEqual(state["snapshot_scope"], "pre_step_only")
        self.assertFalse(state["resumable"])
        self.assertNotIn("observed", state)

    def test_pre_step_fallback_survives_poisoned_loss_inspection(self):
        self.begin()
        original = FloatingPointError("original loss failure")
        with patch("kd.repair_failure._summary", side_effect=RuntimeError("loss device poisoned")):
            self.assertIsNone(self.save(error=original))
        receipt_path, = self.directory.glob("*/failure.json")
        receipt = json.loads(receipt_path.read_text())
        fallback = receipt_path.parent / receipt["pre_step_checkpoint"]
        self.assertEqual(receipt["pre_step_checkpoint_sha256"], fingerprint(fallback))
        state = torch.load(fallback, weights_only=True, map_location="cpu")
        self.assertEqual(state["kind"], "repair_failure_snapshot")
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["context"], {"source": "unit-test"})
        self.assertEqual(state["settings"], {"learning_rate": 0.1})
        self.assertEqual(state["source_sha256"], receipt["source_sha256"])
        self.assertEqual(state["pre_step"]["candidate"]["weight"].device.type, "cpu")
        self.assertEqual(state["pre_step"]["optimizer"]["state"][0]["momentum_buffer"].device.type, "cpu")
        self.assertIsNone(receipt["checkpoint"])
        self.assertEqual(receipt["capture_status"], "capture_failed")
        self.assertTrue(any("loss device poisoned" in note for note in original.__notes__))


if __name__ == "__main__":
    unittest.main()
