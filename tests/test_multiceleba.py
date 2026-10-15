"""Offline checks of the author two-bias selection and held-out boundary."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kd.config import DataConfig, TrainConfig
from kd.data import build_data
from kd.shortcut_data import _multiceleba_rows
from kd.teacher_baselines import load_spec, study_configs, plan_study
from kd.teacher_baseline_matrix import materialize_matrix_plan, matrix_jobs
from kd.teacher_repair_baseline import TeacherRepairOptions
from kd.training_provenance import training_provenance
from test_shortcut_benchmark import multiceleba_release


class MultiCelebATests(unittest.TestCase):
    def config(self, root):
        return DataConfig(source="multiceleba", root=str(root), image_size=224,
                          num_classes=2, dataset_version="1.0", dataset_profile="native-shortcut")

    def test_shipped_bf16_config_matches_generator_and_plans_without_data_or_models(self):
        path = Path(__file__).resolve().parents[1] / "configs/teacher_baselines/multiceleba_bf16.json"
        spec = load_spec(path)
        generated = matrix_jobs(datasets=("multiceleba",), workers=4)[0]["study"]
        actual = spec.to_dict()
        generated["teacher"]["output_dir"] = actual["teacher"]["output_dir"]
        actual["teacher"]["data"]["root"] = str(Path(actual["teacher"]["data"]["root"]).resolve())
        self.assertEqual(actual, generated)
        with tempfile.TemporaryDirectory() as temp, \
             patch("kd.data.build_data", side_effect=AssertionError("Dataset access during plan")), \
             patch("kd.models.create_model", side_effect=AssertionError("Model construction during plan")):
            output = Path(temp) / "study"
            plan = plan_study(spec, output)
            self.assertFalse(output.exists())
            self.assertEqual(plan["counts"]["total_fits_max"], 47)
            self.assertEqual(plan["counts"]["repair_candidate_fits_max"], 0)
            self.assertEqual(plan["mechanism_conditions"], [])
            self.assertEqual(plan["test_endpoints"], ["released"])
            self.assertIsNone(spec.teacher_learning_rates)
            for run in plan["teacher_runs"]:
                self.assertEqual(run["config"]["train"]["learning_rate"], 0.001)

    def test_groups_geometry_provenance_and_no_test_image_access_when_fitting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            multiceleba_release(root / "multiceleba" / "1.0")
            config, train = self.config(root), TrainConfig(device="cpu", workers=0)
            full = build_data(config, train, diagnostic=True)
            self.assertEqual(full.provenance["split_sizes"], {"train": 8, "val": 8, "test": 8})
            self.assertEqual(full.provenance["train_per_group"], [1]*8)
            self.assertEqual(full.train.dataset[0][0].shape, (3, 224, 224))
            self.assertEqual(full.train.dataset.group_ids.tolist(), list(range(8)))
            self.assertEqual(full.provenance["construction"]["nuisance_order"], ["Young", "Male"])
            for image in (root / "multiceleba" / "1.0" / "img_align_celeba").glob("*.jpg"):
                if int(image.stem) > 16:
                    image.unlink()
            fitting = build_data(config, train, include_test=False, diagnostic=True)
            self.assertEqual(training_provenance(full.provenance), training_provenance(fitting.provenance))
            with self.assertRaisesRegex(ValueError, "Missing dataset image"):
                build_data(config, train, include_test=True)

    def test_author_caps_are_deterministic_and_do_not_resample_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            multiceleba_release(root)
            path = root / "list_attr_celeba.csv"
            with path.open() as handle:
                attrs = list(csv.DictReader(handle))
            splits_path = root / "list_eval_partition.csv"
            with splits_path.open() as handle:
                splits = list(csv.DictReader(handle))
            # The (y=0, young=1, male=0) cell is capped at 40 training images.
            for i in range(100, 150):
                name = f"{i:06d}.jpg"
                attrs.append(dict(image_id=name, High_Cheekbones="-1", Young="1", Male="-1"))
                splits.append(dict(image_id=name, partition="0"))
            for target, rows in ((path, list(reversed(attrs))), (splits_path, list(reversed(splits)))):
                with target.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            rows, _ = _multiceleba_rows(root, include_test=True)
            cell = [row for row in rows["train"] if row["label"] == 0 and row["nuisance"] == [1, 0]]
            self.assertEqual(len(cell), 40)
            self.assertEqual(cell[0]["foreground_id"], "000003.jpg")
            self.assertEqual(cell[-1]["foreground_id"], "000138.jpg")
            self.assertEqual((len(rows["val"]), len(rows["test"])), (8, 8))

    def test_original_txt_annotations_match_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            multiceleba_release(root)
            expected, _ = _multiceleba_rows(root, include_test=True)
            for name in ("list_attr_celeba", "list_eval_partition"):
                path = root / (name + ".csv")
                with path.open() as handle:
                    reader = csv.reader(handle)
                    header, rows = next(reader), list(reader)
                prefix = f"{len(rows)}\n{' '.join(header[1:])}\n" if name == "list_attr_celeba" else ""
                (root / (name + ".txt")).write_text(prefix + "\n".join(" ".join(row) for row in rows) + "\n")
                path.unlink()
            actual, _ = _multiceleba_rows(root, include_test=True)
            self.assertEqual(actual, expected)

    def test_invalid_metadata_rejected(self):
        for case in ("duplicate", "partition", "attribute", "path", "missing_group"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                multiceleba_release(root / "multiceleba" / "1.0")
                release = root / "multiceleba" / "1.0"
                attrs = release / "list_attr_celeba.csv"
                splits = release / "list_eval_partition.csv"
                if case == "duplicate":
                    attrs.write_text(attrs.read_text() + attrs.read_text().splitlines()[1] + "\n")
                elif case == "partition":
                    splits.write_text(splits.read_text().replace("000001.jpg,0", "000001.jpg,9"))
                elif case == "attribute":
                    attrs.write_text(attrs.read_text().replace("000001.jpg,-1", "000001.jpg,0"))
                elif case == "path":
                    for path in (attrs, splits):
                        path.write_text(path.read_text().replace("000001.jpg", "../bad.jpg"))
                else:
                    splits.write_text(splits.read_text().replace("000001.jpg,0", "000001.jpg,1"))
                with self.assertRaises(ValueError):
                    build_data(self.config(root), TrainConfig(device="cpu"), include_test=False)

    def test_frozen_plan_baseline_only_and_named_pretrained_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            materialize_matrix_plan(root / "plan", datasets=("multiceleba",), root=root / "data")
            plan = json.loads((root / "plan" / "protocol.json").read_text())
            job = plan["jobs"][0]
            spec = load_spec(job["config"])
            self.assertIsNone(spec.repair)
            self.assertFalse(spec.group_policy["mechanism_controls"])
            self.assertEqual(job["counts"]["total_fits_max"], 47)
            self.assertEqual(spec.teacher.train.precision, "bfloat16")
            self.assertEqual(spec.teacher.train.epochs, 100)
            self.assertEqual(spec.teacher.student.name, "resnet34")
            self.assertEqual(spec.student_weights, "IMAGENET1K_V1")
            self.assertEqual(spec.teacher.data.image_size, 224)
            self.assertFalse(spec.teacher.data.horizontal_flip)
            with self.assertRaisesRegex(ValueError, "facial-attribute repair"):
                replace(spec, recipes=(*spec.recipes, "teacher_repair"), repair=TeacherRepairOptions()).validate()
            teachers, students, supervised = study_configs(spec, job["output"])
            self.assertEqual((len(teachers), len(students), len(supervised)), (11, 33, 3))
            self.assertTrue(all(config.student.weights == "IMAGENET1K_V1" for _, _, config in students))


if __name__ == "__main__":
    unittest.main()
