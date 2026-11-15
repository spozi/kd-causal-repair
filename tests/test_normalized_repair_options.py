"""Configuration and benchmark accounting for parameter-normalized repair."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kd.checkpoints import fingerprint, write_json
from kd.matrix_binding import STUDY_SOURCE_FILES
from kd.teacher_baseline_matrix import matrix_jobs
from kd.teacher_baselines import load_spec, plan_study
from kd.teacher_repair_baseline import TeacherRepairOptions, repair_options, repair_spec


FRACTIONS = (0.004, 0.008, 0.016, 0.032)
LEGACY_OPTIONS = {
    "channel_budgets": [2, 4, 8], "learning_rates": [0.001, 0.005],
    "repair_epochs": 20, "repair_samples_per_epoch": 640, "repair_batch_size": 64,
    "target_fraction": 0.2, "companion_count": 5, "allow_sparse_class_fallback": False,
}


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _spec(directory, source="cifar10"):
    kwargs = {"datasets": (source,), "root": Path(directory) / "data"}
    if source == "cifar10":
        kwargs.update(pairs=("convnet6",), profiles=("lt-if10",))
    raw = matrix_jobs(**kwargs)[0]["study"]
    path = Path(directory) / "study.json"
    write_json(path, raw)
    return load_spec(path)


class NormalizedRepairOptionsTests(unittest.TestCase):
    def test_fixed_channel_defaults_keep_legacy_serialization(self):
        options = TeacherRepairOptions()
        self.assertEqual(_json(options.to_dict()), LEGACY_OPTIONS)
        self.assertEqual(repair_options(LEGACY_OPTIONS), options)
        with tempfile.TemporaryDirectory() as directory:
            spec = _spec(directory)
            expected = {**LEGACY_OPTIONS, "allow_sparse_class_fallback": True}
            self.assertEqual(spec.to_dict()["repair"], expected)

    def test_normalized_options_roundtrip_and_reach_standalone_repair(self):
        raw = {**LEGACY_OPTIONS, "parameter_budget_fractions": list(FRACTIONS),
               "max_preservation_kl": 0.02}
        options = repair_options(raw)
        self.assertEqual(options.parameter_budget_fractions, FRACTIONS)
        self.assertEqual(_json(options.to_dict()), raw)
        with tempfile.TemporaryDirectory() as directory:
            spec = replace(_spec(directory), repair=options)
            path = Path(directory) / "normalized.json"
            write_json(path, spec.to_dict())
            loaded = load_spec(path)
            self.assertEqual(loaded.to_dict(), spec.to_dict())
            standalone = repair_spec(loaded, 41)
            self.assertEqual(standalone.parameter_budget_fractions, FRACTIONS)
            self.assertEqual(standalone.max_preservation_kl, 0.02)

    def test_fixed_channel_mode_can_explicitly_request_kl_constraint(self):
        options = repair_options({**LEGACY_OPTIONS, "max_preservation_kl": 0.0})
        self.assertEqual(options.max_preservation_kl, 0.0)
        self.assertNotIn("parameter_budget_fractions", options.to_dict())
        self.assertEqual(options.to_dict()["max_preservation_kl"], 0.0)

    def test_normalized_mode_requires_explicit_valid_preservation_constraint(self):
        with self.assertRaises(ValueError):
            TeacherRepairOptions(parameter_budget_fractions=FRACTIONS).validate()
        for invalid in (-0.1, True, float("inf"), float("nan")):
            with self.subTest(kl=invalid), self.assertRaises(ValueError):
                TeacherRepairOptions(parameter_budget_fractions=FRACTIONS,
                                     max_preservation_kl=invalid).validate()

    def test_fractions_must_be_unique_increasing_finite_proportions(self):
        for fractions in ((0,), (-0.1,), (1.1,), (True,), (float("nan"),),
                          (0.01, 0.01), (0.02, 0.01)):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                TeacherRepairOptions(parameter_budget_fractions=fractions,
                                     max_preservation_kl=0.02).validate()

    def test_plan_counts_normalized_grid_not_inactive_fixed_channel_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            original = _spec(directory)
            normalized = replace(original, repair=replace(original.repair,
                parameter_budget_fractions=FRACTIONS, max_preservation_kl=0.02))
            with patch("kd.teacher_baselines.build_data", side_effect=AssertionError("data access")):
                baseline_plan = plan_study(original, Path(directory) / "old")
                normalized_plan = plan_study(normalized, Path(directory) / "new")
            self.assertEqual(baseline_plan["counts"]["repair_candidate_fits_max"], 6)
            self.assertEqual(normalized_plan["counts"]["repair_candidate_fits_max"], 8)
            self.assertEqual(normalized_plan["counts"]["total_fits_max"],
                             baseline_plan["counts"]["total_fits_max"] + 2)
            self.assertEqual(normalized_plan["repair_protocols"][0]["parameter_budget_fractions"],
                             list(FRACTIONS))

    def test_shortcut_settings_reject_unsupported_new_constraints_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            original = _spec(directory, "colored_mnist")
            for options in (replace(original.repair, parameter_budget_fractions=FRACTIONS,
                                    max_preservation_kl=0.02),
                            replace(original.repair, max_preservation_kl=0.02)):
                spec = replace(original, repair=options)
                with self.subTest(options=options), self.assertRaisesRegex(
                        ValueError, "not implemented for shortcut datasets"):
                    plan_study(spec, Path(directory) / "unsupported")

    def test_budget_implementation_is_bound_to_benchmark_source_identity(self):
        self.assertIn("repair_budget.py", STUDY_SOURCE_FILES)
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_study(_spec(directory), Path(directory) / "run")
            self.assertEqual(plan["source_sha256"]["repair_budget.py"],
                             fingerprint(Path("kd") / "repair_budget.py"))


if __name__ == "__main__":
    unittest.main()
