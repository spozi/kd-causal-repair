import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from kd.models import CifarCNN, ResNetAdapter
from kd.neuron_surgery import channel_parameter_masks
from kd.repair_budget import (plan_parameter_budgets, preservation_metrics,
                              validate_budget_options, parameter_support_sha256)


class RepairBudgetValidationTests(unittest.TestCase):
    def test_legacy_and_explicit_normalized_options(self):
        validate_budget_options((), None)
        validate_budget_options([], 0)
        validate_budget_options((0.004, 0.008, 0.016, 0.032), 0.01)

    def test_invalid_fractions_are_rejected(self):
        for fractions in (None, "0.01", [True], [np.bool_(True)], [0], [-0.1], [1.01],
                          [float("nan")], [float("inf")], [0.1, 0.1], [0.2, 0.1], ["0.1"]):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                validate_budget_options(fractions, 0.01)

    def test_normalized_mode_requires_explicit_preservation_constraint(self):
        with self.assertRaisesRegex(ValueError, "explicit max_preservation_kl"):
            validate_budget_options([0.01], None)
        for threshold in (True, np.bool_(True), -0.01, float("nan"), float("inf"), "0.1"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                validate_budget_options([], threshold)


class ParameterBudgetPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model = CifarCNN(width=2, num_classes=3)
        self.ranking = [{"stage": "stage2", "channel": 0, "score": 4.0},
                        {"stage": "stage3", "channel": 1, "score": 3.0},
                        {"stage": "stage2", "channel": 2, "score": 2.0}]

    def count(self, ranking):
        return sum(int(mask.count_nonzero()) for mask in
                   channel_parameter_masks(self.model, ranking).values())

    def test_cnn_prefixes_use_actual_masks_and_parameter_denominator(self):
        total = sum(parameter.numel() for parameter in self.model.parameters())
        costs = [self.count(self.ranking[:i]) for i in (1, 2, 3)]
        fractions = [(cost + 0.25) / total for cost in costs]
        result = plan_parameter_budgets(self.model, self.ranking, fractions)
        self.assertEqual(result["total_parameters"], total)
        self.assertNotEqual(total, sum(value.numel() for value in self.model.state_dict().values()))
        self.assertEqual([row["budget"] for row in result["candidates"]], [1, 2, 3])
        self.assertEqual([row["editable_coordinates"] for row in result["requests"]], costs)
        for request, candidate in zip(result["requests"], result["candidates"]):
            self.assertLessEqual(request["editable_coordinates"], request["coordinate_cap"])
            self.assertEqual(candidate["channels"], [
                {"stage": row["stage"], "channel": row["channel"]}
                for row in self.ranking[:candidate["budget"]]])
            self.assertEqual(candidate["parameter_budget"]["achieved_fraction"],
                             request["editable_coordinates"] / total)
        self.assertEqual(result["candidates"][1]["parameter_budget"]["per_stage"], {
            "stage2": {"channels": 1, "available_channels": 4, "channel_fraction": 0.25},
            "stage3": {"channels": 1, "available_channels": 8, "channel_fraction": 0.125}})

    def test_cap_infeasible_duplicate_saturated_and_empty(self):
        total = sum(parameter.numel() for parameter in self.model.parameters())
        first = self.count(self.ranking[:1])
        result = plan_parameter_budgets(self.model, self.ranking,
                                        [(first - 1) / total, (first + .1) / total,
                                         (first + .2) / total, 1.0])
        self.assertEqual([row["status"] for row in result["requests"]],
                         ["infeasible", "unique", "duplicate", "unique"])
        self.assertEqual(result["requests"][2]["duplicate_of"], 1)
        self.assertEqual(result["requests"][0]["shortfall_reason"], "first_channel_exceeds_cap")
        self.assertEqual(result["requests"][-1]["shortfall_reason"], "ranked_pool_exhausted")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(len(result["candidates"][0]["parameter_budget"]["requested_fractions"]), 2)
        self.assertTrue(result["requests"][-1]["saturated"])
        self.assertEqual(result["requests"][-1]["channel_budget"], len(self.ranking))
        empty = plan_parameter_budgets(self.model, [], [0.01, 1.0])
        self.assertEqual(empty["candidates"], [])
        self.assertEqual([row["status"] for row in empty["requests"]], ["infeasible"] * 2)
        self.assertTrue(all(not row["saturated"] for row in empty["requests"]))
        self.assertTrue(all(row["shortfall_reason"] == "empty_ranked_pool" for row in empty["requests"]))
        self.assertEqual(plan_parameter_budgets(self.model, self.ranking, [])["requests"], [])
        zero_cap = plan_parameter_budgets(self.model, self.ranking, [0.01 / total])["requests"][0]
        self.assertEqual(zero_cap["coordinate_cap"], 0)
        self.assertEqual(zero_cap["shortfall_reason"], "first_channel_exceeds_cap")

    def test_actual_support_hash_is_coordinate_not_count_or_channel_order_identity(self):
        first = channel_parameter_masks(self.model, self.ranking[:1])
        other = channel_parameter_masks(self.model, [{"stage": "stage2", "channel": 1}])
        self.assertEqual(sum(int(mask.count_nonzero()) for mask in first.values()),
                         sum(int(mask.count_nonzero()) for mask in other.values()))
        self.assertNotEqual(parameter_support_sha256(first), parameter_support_sha256(other))
        both = channel_parameter_masks(self.model, self.ranking)
        reverse = channel_parameter_masks(self.model, list(reversed(self.ranking)))
        self.assertEqual(parameter_support_sha256(both), parameter_support_sha256(reverse))
        self.assertEqual(parameter_support_sha256(both),
                         parameter_support_sha256(dict(reversed(list(both.items())))))
        plan = plan_parameter_budgets(self.model, self.ranking, [1.0])
        self.assertEqual(plan["candidates"][0]["parameter_budget"]["support_sha256"],
                         parameter_support_sha256(both))

    def test_support_hash_binds_tensor_name_shape_and_requires_boolean_masks(self):
        self.assertNotEqual(parameter_support_sha256({"weight": torch.tensor([True, False])}),
                            parameter_support_sha256({"other": torch.tensor([True, False])}))
        self.assertNotEqual(parameter_support_sha256({"weight": torch.tensor([True, False])}),
                            parameter_support_sha256({"weight": torch.tensor([[True, False]])}))
        with self.assertRaisesRegex(ValueError, "boolean"):
            parameter_support_sha256({"weight": torch.tensor([1, 0])})

    def test_overlapping_masks_are_counted_as_a_union_and_longest_prefix_wins(self):
        model = torch.nn.Linear(4, 1, bias=False)
        model.feature_channels = {"stage2": 3}
        ranking = [{"stage": "stage2", "channel": index} for index in range(3)]

        def masks(_model, selected):
            mask = torch.zeros(4, dtype=torch.bool)
            for row in selected:
                # Second channel contributes no extra coordinates: plateau.
                indices = ((0, 1), (1,), (1, 2))[row["channel"]]
                mask[list(indices)] = True
            return {"weight": mask}

        with patch("kd.repair_budget.channel_parameter_masks", side_effect=masks):
            result = plan_parameter_budgets(model, ranking, [0.5, 0.75])
        self.assertEqual([row["budget"] for row in result["candidates"]], [2, 3])
        self.assertEqual([row["editable_coordinates"] for row in result["requests"]], [2, 3])

    def test_actual_resnet_mask_cost_is_used_without_pretrained_download(self):
        model = ResNetAdapter("resnet18", 3, cifar_stem=True)
        ranking = [{"stage": "stage3", "channel": 0}, {"stage": "stage4", "channel": 1}]
        masks = channel_parameter_masks(model, ranking)
        expected = sum(int(mask.count_nonzero()) for mask in masks.values())
        result = plan_parameter_budgets(model, ranking, [1.0])
        self.assertEqual(result["requests"][0]["editable_coordinates"], expected)
        self.assertEqual(result["candidates"][0]["budget"], 2)

    def test_invalid_and_duplicate_channels_are_rejected_even_below_cap(self):
        for ranking in ([self.ranking[0], self.ranking[0]],
                        [{"stage": "stage2", "channel": True}],
                        [{"stage": "stage2", "channel": 0.5}],
                        [{"stage": "stage1", "channel": 0}],
                        [{"stage": "stage3", "channel": 8}],
                        [self.ranking[0], {"stage": "stage3", "channel": -1}]):
            with self.subTest(ranking=ranking), self.assertRaises(ValueError):
                plan_parameter_budgets(self.model, ranking, [1 / 1000000])


class PreservationMetricsTests(unittest.TestCase):
    def test_identical_predictions_have_exact_zero_drift_including_zeros(self):
        probabilities = np.asarray([[1, 0], [0.25, 0.75]], dtype=np.float32)
        result = preservation_metrics([0, 1], probabilities, probabilities)
        self.assertEqual(result, {"mean_forward_kl": 0.0, "samples": 2,
                                  "prediction_flip_rate": 0.0, "original_accuracy": 1.0,
                                  "candidate_accuracy": 1.0})

    def test_forward_kl_is_image_average_at_t1_not_reverse_or_class_average(self):
        original = np.asarray([[0.8, 0.2], [0.3, 0.7]])
        candidate = np.asarray([[0.4, 0.6], [0.3, 0.7]])
        result = preservation_metrics([0, 1], original, candidate)
        expected = (0.8 * math.log(0.8 / 0.4) + 0.2 * math.log(0.2 / 0.6)) / 2
        self.assertAlmostEqual(result["mean_forward_kl"], expected)
        self.assertEqual(result["prediction_flip_rate"], 0.5)
        self.assertEqual(result["original_accuracy"], 1.0)
        self.assertEqual(result["candidate_accuracy"], 0.5)

    def test_zero_candidate_probability_is_safe_and_float32_sums_are_tolerated(self):
        result = preservation_metrics([0], [[1, 0]], [[0, 1]])
        self.assertTrue(math.isfinite(result["mean_forward_kl"]))
        self.assertGreater(result["mean_forward_kl"], 700)
        probabilities = np.asarray([[0.1, 0.2, 0.7]], dtype=np.float32)
        self.assertEqual(preservation_metrics([2], probabilities, probabilities)["mean_forward_kl"], 0)

    def test_invalid_probability_or_label_inputs_are_rejected(self):
        valid = np.asarray([[0.8, 0.2], [0.3, 0.7]])
        for labels, original, candidate in (
                ([], np.empty((0, 2)), np.empty((0, 2))),
                ([0, 1], valid, valid[:, :1]),
                ([0, 1], valid[0], valid[0]),
                ([0], valid, valid),
                ([0.0, 1.0], valid, valid),
                ([False, True], valid, valid),
                ([0, 2], valid, valid),
                ([0, -1], valid, valid),
                ([0, 1], [[float("nan"), 0], [0, 1]], valid),
                ([0, 1], valid, [[float("inf"), 0], [0, 1]]),
                ([0, 1], [[1.1, -0.1], [0, 1]], valid),
                ([0, 1], valid, valid * 2)):
            with self.subTest(labels=labels, original=original, candidate=candidate), \
                    self.assertRaises(ValueError):
                preservation_metrics(labels, original, candidate)


if __name__ == "__main__":
    unittest.main()
