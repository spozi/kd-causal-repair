"""CPU-only fault injection for repair failure snapshots, not repair experiments."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from kd.models import create_model
from kd.neuron_surgery import repair_loss, train_repair_candidate


class _AugmentedTensorDataset(TensorDataset):
    """A transform consuming all three RNGs, with an observable fetch count."""

    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.fetches = []

    def __getitem__(self, index):
        self.fetches.append(index)
        image, label = super().__getitem__(index)
        noise = torch.rand(()) + random.random() + float(np.random.random())
        return image + noise / 100, label


class RepairFailureIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.manual_seed(311)
        self.anchor = create_model("convnet6_w16", 2).eval()
        self.candidate = deepcopy(self.anchor)
        self.anchor_before = deepcopy(self.anchor.state_dict())
        # Each example has a recognizable, deterministic image and both pools
        # contain both labels. Indices are intentionally not contiguous subsets.
        self.images = torch.linspace(-1, 1, 8 * 3 * 32 * 32).reshape(8, 3, 32, 32)
        self.labels = np.asarray([0, 1, 1, 0, 0, 1, 1, 0])
        self.dataset = TensorDataset(self.images, torch.tensor(self.labels))
        self.target_indices = [1, 4, 6, 7]
        self.preservation_indices = [0, 2, 3, 5]
        self.failure_dir = self.root / "failures"

    def run_candidate(self, *, candidate=None, failure_dir=True, **overrides):
        options = dict(epochs=3, samples_per_epoch=4, batch_size=2,
                       learning_rate=0.01, momentum=0.9, seed=23,
                       precision="float32", channels_last=False,
                       full_finetune=True)
        options.update(overrides)
        if failure_dir is not False:
            options.update(failure_dir=self.failure_dir,
                           failure_context={"candidate": "synthetic-injected-failure"})
        return train_repair_candidate(
            candidate if candidate is not None else self.candidate,
            self.anchor, self.dataset, self.labels, self.target_indices,
            self.preservation_indices, [{"stage": "stage3", "channel": 0}],
            torch.device("cpu"), **options)

    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0,
                                       equal_nan=True)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for item, reference in zip(actual, expected):
                self.assert_tree_equal(item, reference)
        else:
            self.assertEqual(actual, expected)

    def load_failure(self, phase, *, epoch=2, step=2, global_step=4):
        summaries = list(self.failure_dir.rglob("failure.json"))
        self.assertEqual(len(summaries), 1)
        summary_path = summaries[0]
        summary = json.loads(summary_path.read_text())
        state_path = summary_path.parent / "state.pt"
        self.assertEqual(summary["checkpoint"], "state.pt")
        self.assertEqual(summary["checkpoint_sha256"],
                         hashlib.sha256(state_path.read_bytes()).hexdigest())
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        for key, expected in {"phase": phase, "epoch": epoch, "step": step,
                              "global_step": global_step, "status": "failed",
                              "resumable": False}.items():
            self.assertEqual(state[key], expected)
            self.assertEqual(summary[key], expected)
        self.assertEqual(state["context"]["candidate"], "synthetic-injected-failure")
        self.assert_tree_equal(state["observed"]["anchor"], self.anchor_before)
        self.assert_tree_equal(self.anchor.state_dict(), self.anchor_before)
        self.assertIn("before_forward_rng", state["pre_step"])
        self.assertEqual(len(state["history"]), epoch - 1)
        # Diagnostic artifacts must never masquerade as valid selected/completed
        # training artifacts, even if the enclosing runner aborts immediately.
        for filename in ("student.pt", "best.pt", "last.pt", "completion.json",
                         "summary.json", "result.json"):
            self.assertEqual(list(self.root.rglob(filename)), [])
        for pool_name, allowed in (("target", self.target_indices),
                                   ("preservation", self.preservation_indices)):
            batch = state["batches"][pool_name]
            indices = torch.as_tensor(batch["indices"], dtype=torch.long)
            self.assertEqual(indices.numel(), 2)
            self.assertTrue(set(indices.tolist()).issubset(allowed))
            self.assert_tree_equal(batch["images"], self.images[indices])
            self.assert_tree_equal(batch["labels"], torch.tensor(self.labels)[indices])
            self.assertEqual(batch["images"].device.type, "cpu")
        return state

    @staticmethod
    def loss_with_nan_at(call_number):
        calls = 0

        def injected(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = repair_loss(*args, **kwargs)
            if calls == call_number:
                result["total"] = result["total"] * float("nan")
            return result

        return injected

    def test_nonfinite_loss_saves_exact_failing_step_and_batches(self):
        with patch("kd.neuron_surgery.repair_loss", self.loss_with_nan_at(4)):
            with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
                self.run_candidate()
        state = self.load_failure("loss_validation")
        self.assertTrue(torch.isnan(state["losses"]["total"]).all())
        self.assertTrue(torch.isfinite(state["losses"]["ce"]).all())
        self.assert_tree_equal(state["pre_step"]["candidate"],
                               state["observed"]["candidate"])
        self.assert_tree_equal(state["pre_step"]["optimizer"],
                               state["observed"]["optimizer"])
        self.assertTrue(state["pre_step"]["optimizer"]["state"])
        self.assertTrue(any(not torch.equal(value, self.anchor_before[name])
                            for name, value in state["pre_step"]["candidate"].items()))
        self.assertIn("rng", state["pre_step"])
        self.assertIn("samplers", state["pre_step"])

    def test_backward_exception_is_saved_and_original_exception_is_reraised(self):
        original_backward = torch.Tensor.backward
        calls = 0
        sentinel = RuntimeError("injected backward failure")

        def injected(tensor, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise sentinel
            return original_backward(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "backward", injected):
            with self.assertRaises(RuntimeError) as raised:
                self.run_candidate()
        self.assertIs(raised.exception, sentinel)
        self.load_failure("backward")

    def test_epoch_boundary_fetch_failure_does_not_save_previous_preservation_draws(self):
        sentinel = RuntimeError("injected first target fetch of second epoch")

        class FailingDataset(TensorDataset):
            def __init__(self, *tensors):
                super().__init__(*tensors)
                self.fetches = []

            def __getitem__(self, index):
                self.fetches.append(index)
                if len(self.fetches) == 9:
                    raise sentinel
                return super().__getitem__(index)

        self.dataset = FailingDataset(self.images, torch.tensor(self.labels))
        with self.assertRaises(RuntimeError) as raised:
            self.run_candidate()
        self.assertIs(raised.exception, sentinel)
        summaries = list(self.failure_dir.rglob("failure.json"))
        self.assertEqual(len(summaries), 1)
        state = torch.load(summaries[0].parent / "state.pt", map_location="cpu", weights_only=True)
        self.assertEqual((state["phase"], state["epoch"], state["step"], state["global_step"]),
                         ("target_data", 2, 1, 3))
        self.assertEqual(len(self.dataset.fetches), 9)
        target = state["batches"]["target"]
        self.assertEqual(len(target["indices"]), 2)
        self.assertEqual(target["indices"][0], self.dataset.fetches[-1])
        self.assertTrue(set(target["indices"]).issubset(self.target_indices))
        self.assertNotIn("images", target)
        self.assertNotIn("labels", target)
        self.assertNotIn("preservation", state["batches"])
        for stage in ("pre_step", "observed"):
            preservation = state[stage]["samplers"]["preservation"]
            self.assertEqual(preservation["epoch_draws"], [])
            self.assertEqual(preservation["consumed"], 0)
        self.assertEqual(state["observed"]["samplers"]["target"]["consumed"], 2)
        self.assert_tree_equal(state["observed"]["anchor"], self.anchor_before)
        self.assert_tree_equal(self.anchor.state_dict(), self.anchor_before)

    def test_forward_failure_saves_exact_input_batches_and_original_exception(self):
        original_forward = self.candidate.forward
        calls = 0
        sentinel = RuntimeError("injected candidate target forward failure")

        def injected(*args, **kwargs):
            nonlocal calls
            calls += 1
            # Two candidate forwards per completed step; the anchor is separate.
            if calls == 7:
                raise sentinel
            return original_forward(*args, **kwargs)

        with patch.object(self.candidate, "forward", injected):
            with self.assertRaises(RuntimeError) as raised:
                self.run_candidate()
        self.assertIs(raised.exception, sentinel)
        state = self.load_failure("forward_target")
        self.assertEqual(state["losses"], {})
        self.assertEqual(state["outputs"], {})
        self.assert_tree_equal(state["pre_step"]["candidate"],
                               state["observed"]["candidate"])

    def test_nonfinite_gradient_is_saved_before_optimizer_mutation(self):
        for fail_fast in (True, False):
            with self.subTest(fail_fast=fail_fast):
                self.candidate = deepcopy(self.anchor).requires_grad_(True)
                self.failure_dir = self.root / f"gradient-failures-{fail_fast}"
                calls = 0

                def injected(gradient):
                    nonlocal calls
                    calls += 1
                    return torch.full_like(gradient, float("nan")) if calls == 4 else gradient

                handle = self.candidate.classifier.bias.register_hook(injected)
                try:
                    with self.assertRaises((RuntimeError, FloatingPointError)):
                        self.run_candidate(fail_fast=fail_fast)
                finally:
                    handle.remove()
                state = self.load_failure("gradient_validation")
                self.assertTrue(torch.isnan(
                    state["observed"]["gradients"]["classifier.bias"]).all())
                self.assert_tree_equal(state["pre_step"]["candidate"],
                                       state["observed"]["candidate"])
                self.assert_tree_equal(state["pre_step"]["optimizer"],
                                       state["observed"]["optimizer"])

    def test_optimizer_exception_preserves_pre_step_and_partially_mutated_state(self):
        original_step = torch.optim.SGD.step
        calls = 0
        sentinel = RuntimeError("injected partially completed SGD step")

        def injected(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            result = original_step(optimizer, *args, **kwargs)
            if calls == 4:
                raise sentinel
            return result

        with patch.object(torch.optim.SGD, "step", injected):
            with self.assertRaises(RuntimeError) as raised:
                self.run_candidate()
        self.assertIs(raised.exception, sentinel)
        state = self.load_failure("optimizer_step")
        self.assertTrue(any(not torch.equal(value, state["observed"]["candidate"][name])
                            for name, value in state["pre_step"]["candidate"].items()))
        self.assertTrue(any(not torch.equal(
            row["momentum_buffer"],
            state["observed"]["optimizer"]["state"][index]["momentum_buffer"])
            for index, row in state["pre_step"]["optimizer"]["state"].items()))

    def test_nonfinite_optimizer_output_is_caught_per_step_even_without_fail_fast(self):
        original_step = torch.optim.SGD.step
        for fail_fast in (True, False):
            with self.subTest(fail_fast=fail_fast):
                self.candidate = deepcopy(self.anchor)
                self.failure_dir = self.root / f"failures-{fail_fast}"
                calls = 0

                def injected(optimizer, *args, **kwargs):
                    nonlocal calls
                    calls += 1
                    result = original_step(optimizer, *args, **kwargs)
                    if calls == 4:
                        with torch.no_grad():
                            optimizer.param_groups[0]["params"][0].fill_(float("inf"))
                    return result

                with patch.object(torch.optim.SGD, "step", injected):
                    with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
                        self.run_candidate(fail_fast=fail_fast)
                state = self.load_failure("post_step_validation")
                self.assertTrue(all(torch.isfinite(value).all()
                                    for value in state["pre_step"]["candidate"].values()))
                self.assertTrue(any(not torch.isfinite(value).all()
                                    for value in state["observed"]["candidate"].values()))

    def test_nonfinite_optimizer_state_is_caught_even_when_model_stays_finite(self):
        original_step = torch.optim.SGD.step
        calls = 0

        def injected(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            result = original_step(optimizer, *args, **kwargs)
            if calls == 4:
                first_parameter = optimizer.param_groups[0]["params"][0]
                optimizer.state[first_parameter]["momentum_buffer"].fill_(float("nan"))
            return result

        with patch.object(torch.optim.SGD, "step", injected):
            with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
                self.run_candidate(fail_fast=False)
        state = self.load_failure("post_step_validation")
        self.assertTrue(all(torch.isfinite(value).all()
                            for value in state["observed"]["candidate"].values()))
        self.assertTrue(all(torch.isfinite(row["momentum_buffer"]).all()
                            for row in state["pre_step"]["optimizer"]["state"].values()))
        self.assertTrue(any(not torch.isfinite(row["momentum_buffer"]).all()
                            for row in state["observed"]["optimizer"]["state"].values()))

    def test_loss_failure_is_caught_per_step_when_fail_fast_is_disabled(self):
        with patch("kd.neuron_surgery.repair_loss", self.loss_with_nan_at(4)):
            with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
                self.run_candidate(fail_fast=False)
        self.load_failure("loss_validation")

    def test_snapshot_write_failure_does_not_replace_original_exception(self):
        sentinel = RuntimeError("original backward failure")
        with patch.object(torch.Tensor, "backward", side_effect=sentinel):
            with patch("kd.repair_failure.RepairFailureRecorder.save",
                       side_effect=OSError("diagnostic disk is full")):
                with self.assertRaises(RuntimeError) as raised:
                    self.run_candidate()
        self.assertIs(raised.exception, sentinel)

    def test_success_has_bitwise_model_history_and_rng_parity_without_artifacts(self):
        self.dataset = _AugmentedTensorDataset(self.images, torch.tensor(self.labels))
        for full_finetune in (True, False):
            with self.subTest(full_finetune=full_finetune):
                outcomes = []
                for recording in (False, True):
                    candidate = deepcopy(self.anchor)
                    torch.manual_seed(101)
                    random.seed(102)
                    np.random.seed(103)
                    self.dataset.fetches.clear()
                    history, verification = self.run_candidate(
                        candidate=candidate, failure_dir=recording,
                        full_finetune=full_finetune)
                    outcomes.append((deepcopy(candidate.state_dict()), history,
                                     verification, torch.get_rng_state(),
                                     random.getstate(), np.random.get_state(),
                                     self.dataset.fetches.copy()))
                    self.assertEqual(len(self.dataset.fetches), 3 * 4 * 2)
                self.assert_tree_equal(outcomes[0], outcomes[1])
                self.assertFalse(self.failure_dir.exists() and
                                 any(path.is_file() for path in self.failure_dir.rglob("*")))
                self.assert_tree_equal(self.anchor.state_dict(), self.anchor_before)


if __name__ == "__main__":
    unittest.main()
