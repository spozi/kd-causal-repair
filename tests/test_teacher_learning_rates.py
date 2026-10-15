"""Frozen, teacher-only recipe rates do not leak into matched students."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from kd.teacher_baselines import TeacherBaselineSpec, load_spec, study_configs
from kd.teacher_baseline_matrix import matrix_jobs, materialize_matrix_plan
from kd.config import from_dict


def spec_for(source):
    raw = matrix_jobs(datasets=(source,))[0]["study"]
    return TeacherBaselineSpec(**{**raw, "teacher": from_dict(raw["teacher"]), "repair": None,
        "recipes": tuple(name for name in raw["recipes"] if name != "teacher_repair"),
        "group_policy": {**raw["group_policy"], "mechanism_controls": False}})


class TeacherLearningRateTests(unittest.TestCase):
    def test_colored_mnist_only_mse_teacher_changes(self):
        spec = spec_for("colored_mnist")
        teachers, students, supervised = study_configs(spec, "runs/rate-check")
        self.assertEqual(spec.teacher_learning_rates, {"mse_logits": 0.005})
        for recipe, config in teachers:
            self.assertEqual(config.train.learning_rate, 0.005 if recipe == "mse_logits" else 0.05)
        self.assertTrue(all(config.train.learning_rate == 0.05 for _, _, config in students))
        self.assertTrue(all(config.train.learning_rate == 0.05 for config in supervised))
        for source in ("waterbirds", "multiceleba", "urbancars"):
            self.assertIsNone(spec_for(source).teacher_learning_rates)

    def test_invalid_rates_and_recipes_rejected(self):
        base = spec_for("colored_mnist")
        for mapping in ([], {"mse_logits": True}, {"mse_logits": 0}, {"mse_logits": -1},
                        {"mse_logits": float("nan")}, {"mse_logits": float("inf")},
                        {"unknown": 0.005}, {"teacher_repair": 0.005}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                replace(base, teacher_learning_rates=mapping).validate()

    def test_matrix_roundtrip_binds_rate_and_legacy_omits_optional_field(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            materialize_matrix_plan(root / "plan", datasets=("colored_mnist",), root=root / "data")
            protocol = json.loads((root / "plan" / "protocol.json").read_text())
            job = protocol["jobs"][0]
            spec = load_spec(job["config"])
            self.assertEqual(spec.teacher_learning_rates, {"mse_logits": 0.005})
            legacy = replace(spec_for("colored_mnist"), teacher_learning_rates=None)
            self.assertNotIn("teacher_learning_rates", legacy.to_dict())
            changed = replace(spec, teacher_learning_rates={"mse_logits": 0.01})
            from kd.teacher_baselines import plan_study
            with self.assertRaises(ValueError):
                plan_study(changed, job["output"])


if __name__ == "__main__":
    unittest.main()
