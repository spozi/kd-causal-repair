"""Checks the supervised-only imbalance repair protocol and report."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.imbalance_repair_study import _paired_scores, load, render
from scripts.vanilla_imbalance_repair import plan


def metrics(recalls):
    return {"samples": 140, "accuracy": 0.6, "macro_f1": 0.5,
            "target_macro_recall": sum(recalls[1:]) / 2,
            "per_class": [{"class": name, "support": support, "recall": recall}
                          for name, support, recall in zip(
                              ("head", "rare_a", "rare_b"), (100, 20, 20), recalls)]}


class ImbalanceRepairStudyTests(unittest.TestCase):
    def test_minority_recall_is_class_macro_and_support_must_match(self):
        original = metrics((0.8, 0.1, 0.3))
        repaired = metrics((0.75, 0.2, 0.5))
        pair = _paired_scores(original, repaired, {1, 2})
        self.assertAlmostEqual(pair["delta"]["target_recall"], 0.15)
        self.assertAlmostEqual(pair["delta"]["head_recall"], -0.05)
        repaired["per_class"][1]["support"] = 19
        with self.assertRaisesRegex(ValueError, "different test class vocabulary"):
            _paired_scores(original, repaired, {1, 2})

    def test_plan_disables_kd_and_report_rejects_kd_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            matrix, root = Path(temporary) / "matrix", Path(temporary) / "vanilla"
            matrix.mkdir()
            profiles = ("balanced", "lt-if10", "lt-if50", "lt-if100")
            jobs = [{"id": f"toy-{profile}", "dataset": "toy", "profile": profile,
                     "status": "no_repair_selected"} for profile in profiles]
            (matrix / "matrix_summary.json").write_text(json.dumps({
                "jobs": jobs, "job_count": 4, "all_jobs_finished": True}))
            config = matrix / "matrix-plan/configs/toy-lt-if10.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"name": "toy-lt-if10", "teacher_run": "teacher_toy-lt-if10",
                                          "kd_weight": 1.0, "feature_weight": 0.25,
                                          "preservation_ce_weight": 0.0, "downstream_kd": True}))
            checkpoint = matrix / "toy/lt-if10/baselines/confirmatory/teacher_toy-lt-if10/student.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic checkpoint")
            item = plan(matrix, "toy", "lt-if10", False)[0]
            self.assertEqual((item["study"]["kd_weight"], item["study"]["feature_weight"],
                              item["study"]["preservation_ce_weight"], item["study"]["downstream_kd"]),
                             (0.0, 0.0, 1.0, False))
            baseline, candidate = metrics((0.8, 0.1, 0.3)), metrics((0.75, 0.2, 0.5))
            for profile in profiles[1:]:
                target = root / "toy" / profile / "test_comparison.json"
                target.parent.mkdir(parents=True)
                payload = {"status": "no_repair_selected",
                           "study": {**item["study"], "name": f"toy-{profile}-vanilla-ce",
                                     "teacher_model": "cifar_teacher"}}
                if profile == "lt-if10":
                    payload = {"status": "complete", "evaluation_split": "test",
                               "target_classes": [1, 2],
                               "study": {**item["study"], "teacher_model": "cifar_teacher"},
                               "teacher": {"baseline": baseline, "candidate": candidate}}
                target.write_text(json.dumps(payload))
            loaded_jobs, rows = load(matrix, root)
            self.assertEqual(len(loaded_jobs), 3)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["model"]["delta"]["target_recall"], 0.15)
            report = render(loaded_jobs, rows)
            self.assertIn("No student model was trained", report)
            self.assertIn("| **All long-tail** | 3 | 1 | 2 |", report)
            bad = root / "toy/lt-if10/test_comparison.json"
            payload = json.loads(bad.read_text())
            payload["study"]["kd_weight"] = 1.0
            bad.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "not from the frozen supervised-only protocol"):
                load(matrix, root)

    def test_report_waits_for_every_setting(self):
        with tempfile.TemporaryDirectory() as temporary:
            matrix = Path(temporary) / "matrix"
            matrix.mkdir()
            (matrix / "matrix_summary.json").write_text(json.dumps({
                "jobs": [{"id": f"toy-{profile}", "dataset": "toy", "profile": profile}
                         for profile in ("lt-if10", "lt-if50", "lt-if100")],
                "all_jobs_finished": True}))
            with self.assertRaisesRegex(FileNotFoundError, "3 supervised-only settings remain pending"):
                load(matrix, Path(temporary) / "vanilla")


if __name__ == "__main__":
    unittest.main()
