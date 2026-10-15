"""Offline fixtures for all 16 datasets' planning and the new group path."""

import contextlib
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models

from kd.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, from_dict
from kd.data import build_data
from kd.group_metrics import add_group_metrics, dataset_metadata, group_accuracy, loader_groups, selection_key
from kd.models import create_model
from kd.shortcut_baselines import DEFAULT_GROUP_POLICY, mechanism_conditions
from kd.shortcut_data import SHORTCUT_DATASETS
from kd.shortcut_repair import candidate_eligible, group_guardrails, hard_group_targets, nuisance_companions
from kd.teacher_baseline_matrix import DATASETS, PAIRINGS, matrix_jobs, materialize_matrix_plan, plan_matrix
from kd.teacher_baselines import TeacherBaselineSpec, load_spec, run_study, study_configs
from kd.teacher_repair_baseline import TeacherRepairOptions
from kd.training_provenance import bind_training_data, training_provenance, validate_training_data
from kd.urban_metrics import urban_metrics
from kd.full_benchmark import _report_rows, execute, preflight, summarize


class FakeMNIST:
    accesses = []

    def __init__(self, root, train, download=False):
        if download:
            raise AssertionError("Offline test must not download")
        self.accesses.append(train)
        n = 1000 if train else 200
        self.targets = torch.arange(n) % 10
        self.data = (torch.arange(n * 28 * 28).reshape(n, 28, 28) % 256).to(torch.uint8)


def waterbirds_release(root, per_group=3):
    root.mkdir(parents=True)
    rows = []
    for split in range(3):
        for label in (0, 1):
            for background in (0, 1):
                for index in range(per_group):
                    name = f"split{split}-bird{label}-bg{background}-i{index}.jpg"
                    Image.new("RGB", (40, 36), (split * 60 + index, label * 100, background * 100)).save(root / name)
                    rows.append({"img_filename": name, "y": label, "place": background, "split": split})
    with (root / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("img_filename", "y", "place", "split"))
        writer.writeheader()
        writer.writerows(rows)


def urban_release(root):
    for split in ("train", "val", "test"):
        ratio = "0.95" if split == "train" else "0.5"
        for label, obj in enumerate(("urban", "country")):
            for bg, background in enumerate(("urban", "country")):
                for co, coobject in enumerate(("urban", "country")):
                    directory = root / f"bg-{ratio}_co_occur_obj-{ratio}" / split / f"obj-{obj}_bg-{background}_co_occur_obj-{coobject}"
                    directory.mkdir(parents=True)
                    Image.new("RGB", (256, 256), (("train", "val", "test").index(split) * 80 + label, bg * 100, co * 100)).save(directory / "000.jpg")
                    Image.new("L", (256, 256), 255).save(directory / "000_mask.png")


def small_cmnist_spec(root, *, repair=False):
    from kd.teacher_baseline_matrix import _shortcut_job
    from kd.config import BenchmarkConfig
    raw = _shortcut_job("colored_mnist", root=root, device="cpu", precision="float32", workers=0,
                        epochs=1, teacher_seeds=(41,), student_seeds=(42,))["study"]
    teacher = replace(from_dict(raw["teacher"]), benchmark=BenchmarkConfig(warmup=0, iterations=1),
                      train=replace(from_dict(raw["teacher"]).train, threads=2))
    policy = {**DEFAULT_GROUP_POLICY, "mechanism_controls": repair, "screen_limit": 1,
              "localization_batch_size": 16, "bootstrap_repetitions": 2, "preservation_per_group": 2}
    return TeacherBaselineSpec(teacher=teacher, student_model="convnet6_w16", teacher_seeds=(41,), student_seeds=(42,),
        recipes=("control", "teacher_repair") if repair else ("control", "label_smoothing", "balanced_softmax"),
        teacher_selection_metric="worst_group_accuracy", student_selection_metric="worst_group_accuracy",
        group_policy=policy, repair=TeacherRepairOptions(channel_budgets=(1,), learning_rates=(0.001,),
            repair_epochs=1, repair_samples_per_epoch=8, repair_batch_size=4, companion_count=1) if repair else None)


class ShortcutDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        FakeMNIST.accesses = []

    def test_cmnist_splits_before_rendering_and_test_endpoints_are_paired(self):
        with tempfile.TemporaryDirectory() as root, patch("kd.shortcut_data.datasets.MNIST", FakeMNIST):
            data = DataConfig(source="colored_mnist", root=root, num_classes=2, image_size=32, dataset_version="1.0")
            train = TrainConfig(device="cpu", batch_size=32, workers=0)
            fitting = build_data(data, train, include_test=False)
            self.assertEqual(FakeMNIST.accesses, [True])
            self.assertFalse(set(fitting.train.dataset.source_ids) & set(fitting.val.dataset.source_ids))
            self.assertEqual(fitting.train.dataset[0][0].shape, (3, 32, 32))
            self.assertTrue(all(n > 0 for n in fitting.provenance["val_per_group"]))
            other = build_data(data, replace(train, seed=99), include_test=False, diagnostic=True)
            self.assertEqual(fitting.provenance, other.provenance)
            full = build_data(data, train, include_test=True)
            left, right = (full.test_endpoints[name].dataset for name in ("reversed", "unbiased"))
            np.testing.assert_array_equal(left.source_ids, right.source_ids)
            np.testing.assert_array_equal(left.targets, right.targets)
            self.assertTrue(np.any(left.colors != right.colors))
            self.assertEqual(training_provenance(fitting.provenance), training_provenance(full.provenance))
            validate_training_data({"training_data_identity": bind_training_data(fitting.provenance)}, full.provenance)

    def test_cmnist_color_companions_preserve_foreground_label(self):
        from kd.shortcut_repair import ColorVariants
        with tempfile.TemporaryDirectory() as root, patch("kd.shortcut_data.datasets.MNIST", FakeMNIST):
            data = build_data(DataConfig(source="colored_mnist", root=root, num_classes=2), TrainConfig(device="cpu"), include_test=False, diagnostic=True)
            original = data.train.dataset
            variants = ColorVariants(original)
            image, label = original[0]
            other, other_label = variants[len(original)]
            self.assertEqual(label, other_label)
            torch.testing.assert_close(other, image[[1, 0, 2]])

    def test_waterbirds_released_split_groups_and_native_model_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "waterbirds" / "1.0"
            waterbirds_release(root)
            config = DataConfig(source="waterbirds", root=temp, num_classes=2, image_size=224, dataset_version="1.0")
            data = build_data(config, TrainConfig(device="cpu"), include_test=False)
            self.assertEqual(data.train.dataset[0][0].shape, (3, 224, 224))
            self.assertEqual(data.provenance["train_per_group"], [3] * 4)
            self.assertEqual(data.provenance["split_sizes"], {"train": 12, "val": 12})
            self.assertFalse(set(data.train.dataset.source_ids) & set(data.val.dataset.source_ids))
            full = build_data(config, TrainConfig(device="cpu"))
            self.assertEqual(training_provenance(data.provenance), training_provenance(full.provenance))

    def test_waterbirds_changed_image_changes_bound_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "waterbirds" / "1.0"
            waterbirds_release(root)
            config = DataConfig(source="waterbirds", root=temp, num_classes=2, image_size=224)
            data = build_data(config, TrainConfig(device="cpu"), include_test=False)
            Image.new("RGB", (40, 36), (255, 255, 255)).save(root / "split0-bird0-bg0-i0.jpg")
            changed = build_data(config, TrainConfig(device="cpu"), include_test=False)
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                validate_training_data({"training_data_identity": bind_training_data(data.provenance)}, changed.provenance)

    def test_waterbirds_rejects_traversal_duplicate_ids_and_missing_groups(self):
        for case in ("path", "duplicate", "missing"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "waterbirds"
                waterbirds_release(root)
                csv_path = root / "metadata.csv"
                text = csv_path.read_text()
                if case == "path":
                    text = text.replace("split0-bird0-bg0-i0.jpg", "../outside.jpg")
                elif case == "duplicate":
                    text = text.replace("split1-bird0-bg0-i0.jpg", "split0-bird0-bg0-i0.jpg")
                else:
                    text = "\n".join(line for line in text.splitlines() if not line.startswith("split1-bird0-bg1-")) + "\n"
                csv_path.write_text(text)
                with self.assertRaises(ValueError):
                    build_data(DataConfig(source="waterbirds", root=temp, num_classes=2, image_size=224), TrainConfig(device="cpu"), include_test=False)

    def test_urbancars_author_folder_layout_eight_groups_masks_not_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "urbancars" / "1.0"
            urban_release(root)
            data = build_data(DataConfig(source="urbancars", root=temp, num_classes=2, image_size=256, dataset_version="1.0"), TrainConfig(device="cpu"))
            self.assertEqual(data.provenance["train_per_group"], [1] * 8)
            self.assertEqual(len(data.train.dataset), 8)
            self.assertEqual(data.train.dataset[0][0].shape, (3, 256, 256))
            self.assertIn("does not export original source IDs", data.provenance["constituent_audit"])

    def test_group_config_rejects_if_profiles_wrong_dimensions_and_strong_base(self):
        for source, info in SHORTCUT_DATASETS.items():
            base = DataConfig(source=source, num_classes=2, image_size=info["image_size"])
            for changed in (replace(base, imbalance_factor=10), replace(base, dataset_profile="lt-if10"),
                            replace(base, image_size=16), replace(base, augmentation="strong")):
                with self.subTest(source=source, changed=changed), self.assertRaises(ValueError):
                    ExperimentConfig(data=changed).validate(require_teacher=False)


class GroupMetricTests(unittest.TestCase):
    def test_wga_and_group_mean_are_not_class_balanced_accuracy(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        groups = np.array([0, 0, 0, 1, 2, 3, 3, 3])
        predictions = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        result = group_accuracy(labels, predictions, groups, 4)
        self.assertEqual(result["worst_group_accuracy"], 0)
        self.assertEqual(result["group_balanced_accuracy"], 0.75)
        missing = group_accuracy(labels[:4], predictions[:4], groups[:4], 4)
        self.assertIsNone(missing["worst_group_accuracy"])
        self.assertEqual(missing["missing_groups"], [2, 3])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            selection_key(missing, "worst_group_accuracy")

    def test_wga_selection_ties_use_raw_accuracy(self):
        self.assertGreater(selection_key({"worst_group_accuracy": 0.5, "accuracy": 0.8}, "worst_group_accuracy"),
                           selection_key({"worst_group_accuracy": 0.5, "accuracy": 0.7}, "worst_group_accuracy"))

    def test_urban_gaps_use_author_shift_minus_id_sign(self):
        labels = np.repeat([0, 1], 4)
        nuisance = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (2, 1))
        predictions = nuisance[:, 0]
        metrics = urban_metrics(labels, predictions, nuisance, [1] * 8)
        self.assertAlmostEqual(metrics["urbancars_id_accuracy"], 0.95)
        self.assertAlmostEqual(metrics["urbancars_background_gap"], -0.95)
        self.assertAlmostEqual(metrics["urbancars_coobject_gap"], 0.05)
        self.assertAlmostEqual(metrics["urbancars_joint_gap"], -0.95)
        self.assertAlmostEqual(metrics["train_group_weighted_accuracy"], 0.5)

    def test_group_guardrails_reject_local_damage_even_when_overall_improves(self):
        base = {"accuracy": 0.8, "worst_group_accuracy": 0.5, "group_metrics_available": True,
                "per_group": [{"accuracy": 0.5}, {"accuracy": 0.9}]}
        damaged = {"accuracy": 0.81, "worst_group_accuracy": 0.6, "group_metrics_available": True,
                   "per_group": [{"accuracy": 0.6}, {"accuracy": 0.88}]}
        self.assertFalse(candidate_eligible(damaged, base, DEFAULT_GROUP_POLICY))
        self.assertFalse(candidate_eligible(base, base, DEFAULT_GROUP_POLICY))
        better = {**damaged, "per_group": [{"accuracy": 0.6}, {"accuracy": 0.9}]}
        self.assertTrue(candidate_eligible(better, base, DEFAULT_GROUP_POLICY))

    def test_group_metadata_survives_subsets_and_random_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, patch("kd.shortcut_data.datasets.MNIST", FakeMNIST):
            data = build_data(DataConfig(source="colored_mnist", root=root, num_classes=2), TrainConfig(device="cpu"), include_test=False)
            subset = Subset(data.val.dataset, [3, 0, 2])
            np.testing.assert_array_equal(dataset_metadata(subset, "group_ids"), data.val.dataset.group_ids[[3, 0, 2]])
            with self.assertRaisesRegex(ValueError, "sequential"):
                loader_groups(DataLoader(data.val.dataset, batch_size=16, shuffle=True))


class FullPlannerTests(unittest.TestCase):
    def test_full_default_scope_55_and_expanded_scope_107_without_access_or_writes(self):
        with tempfile.TemporaryDirectory() as temp, patch("kd.data.build_data", side_effect=AssertionError("dataset access")), \
             patch("kd.shortcut_data.datasets.MNIST", side_effect=AssertionError("MNIST access")), \
             patch("kd.models.create_model", side_effect=AssertionError("model construction")):
            root = Path(temp)
            plan = materialize_matrix_plan(root / "plan", root=root / "data", dry_run=True)
            self.assertEqual(len(plan["datasets"]), 16)
            self.assertEqual(plan["job_count"], 55)
            self.assertEqual(plan["counts"]["total_fits_max"], 3153)
            expanded = plan_matrix(root / "expanded", root=root / "data", pairs=("resnet34_resnet18_smallstem_imagenet", "convnet6"))
            self.assertEqual(expanded["job_count"], 107)
            self.assertEqual(list(root.iterdir()), [])
            self.assertEqual(len([job for job in expanded["jobs"] if job.get("evaluation_family") == "shortcut"]), 3)

    def test_shortcut_frozen_specs_roundtrip_and_bind_named_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            materialize_matrix_plan(root / "plan", datasets=tuple(SHORTCUT_DATASETS), root=root / "data")
            protocol = json.loads((root / "plan" / "protocol.json").read_text())
            for job in protocol["jobs"]:
                spec = load_spec(job["config"])
                self.assertEqual(spec.to_dict(), job["study"])
                self.assertEqual(spec.teacher.train.selection_metric, "worst_group_accuracy")
                self.assertEqual(len(mechanism_conditions(spec)), 3)
                _, students, supervised = study_configs(spec, job["output"])
                if job["dataset"] != "colored_mnist":
                    self.assertEqual(spec.teacher.student.weights, "IMAGENET1K_V1")
                    self.assertTrue(all(config.student.weights == "IMAGENET1K_V1" for _, _, config in students))
                    self.assertTrue(all(config.student.weights == "IMAGENET1K_V1" for config in supervised))
                    self.assertTrue(all(config.teacher.weights is None for _, _, config in students))

    def test_legacy_optional_fields_are_omitted_from_serialization(self):
        config = ExperimentConfig()
        raw = config.to_dict()
        self.assertNotIn("weights", raw["student"])
        self.assertNotIn("selection_metric", raw["train"])
        self.assertNotIn("test_environment", raw["data"])
        self.assertEqual(from_dict(raw).to_dict(), raw)

    def test_missing_dataset_preflight_prevents_any_training(self):
        with tempfile.TemporaryDirectory() as temp, patch("kd.full_benchmark.run_study", side_effect=AssertionError("training")):
            root = Path(temp)
            materialize_matrix_plan(root / "plan", datasets=("waterbirds",), root=root / "data")
            readiness = preflight(root / "plan")
            self.assertEqual(readiness["failed_datasets"], ["waterbirds"])
            with self.assertRaisesRegex(ValueError, "preflight failed"):
                execute(root / "plan")
            self.assertFalse((root / "studies").exists())

    def test_full_summary_preserves_all_56_endpoint_rows_without_fabricated_zeros(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            materialize_matrix_plan(root / "plan", root=root / "data")
            summarize(root / "plan", root / "summary")
            report = json.loads((root / "summary" / "overview.json").read_text())
            self.assertEqual(report["endpoint_rows"], 56)
            self.assertEqual(len(report["datasets"]), 16)
            self.assertTrue(all(row["selected_kd"] is None for row in report["rows"]))
            self.assertTrue(all(row["status"] == "pending" for row in report["rows"]))
            with self.assertRaisesRegex(ValueError, "separate"):
                summarize(root / "plan", root / "plan" / "summary")

    def test_pretrained_backbone_loads_named_weights_then_seeds_two_class_head(self):
        # Supply a local fixture through the actual weights-loading API; no network.
        torch.manual_seed(7)
        fixture = models.resnet18(weights=None).state_dict()
        fixture["conv1.weight"].fill_(0.0125)
        with patch.object(models.ResNet18_Weights, "get_state_dict", return_value=fixture):
            torch.manual_seed(41)
            left = create_model("resnet18", 2, weights="IMAGENET1K_V1")
            torch.manual_seed(41)
            right = create_model("resnet18", 2, weights="IMAGENET1K_V1")
        torch.testing.assert_close(left.backbone.conv1.weight, fixture["conv1.weight"])
        torch.testing.assert_close(left.backbone.fc.weight, right.backbone.fc.weight)
        self.assertEqual(left.backbone.fc.out_features, 2)
        self.assertIn("weights_tensor_sha256", left.initialization)
        with self.assertRaises(ValueError):
            create_model("convnet6_w16", 2, weights="IMAGENET1K_V1")


class ShortcutLifecycleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        FakeMNIST.accesses = []

    def test_train_freeze_evaluate_two_environments_and_resume_no_retraining(self):
        with tempfile.TemporaryDirectory() as temp, patch("kd.shortcut_data.datasets.MNIST", FakeMNIST), contextlib.redirect_stdout(io.StringIO()):
            spec = small_cmnist_spec(str(Path(temp) / "data"))
            output = Path(temp) / "study"
            with self.assertRaisesRegex(ValueError, "completed"):
                run_study(spec, output, phase="evaluate")
            self.assertFalse(output.exists())
            result = run_study(spec, output)
            self.assertEqual(result["status"], "trained")
            self.assertTrue(all(FakeMNIST.accesses))
            self.assertEqual(run_study(spec, output)["status"], "trained")
            evaluation = run_study(spec, output, phase="evaluate")
            self.assertEqual(evaluation["status"], "evaluated")
            report = json.loads((output / "comparison.json").read_text())
            self.assertEqual(set(report["endpoint_reports"]), {"reversed", "unbiased"})
            for endpoint in report["endpoint_reports"].values():
                self.assertEqual(endpoint["validation_selected_recipe"], result["selection"])
            self.assertTrue(report["paired_nuisance_checks"])
            self.assertIn(False, FakeMNIST.accesses)

    def test_repair_and_three_mechanism_controls_complete_with_frozen_buffers(self):
        with tempfile.TemporaryDirectory() as temp, patch("kd.shortcut_data.datasets.MNIST", FakeMNIST), contextlib.redirect_stdout(io.StringIO()):
            spec = small_cmnist_spec(str(Path(temp) / "data"), repair=True)
            output = Path(temp) / "study"
            result = run_study(spec, output)
            self.assertEqual(result["status"], "trained")
            report = json.loads((output / "validation.json").read_text())
            self.assertEqual(len([row for row in report["rows"] if row["role"] == "mechanism_control"]), 3)
            self.assertIn(result["selection"], spec.recipes)
            for row in report["rows"]:
                for decision in row.get("repair_outcomes", {}).values():
                    for candidate in decision["candidates"]:
                        if "verification" in candidate:
                            self.assertTrue(candidate["verification"]["all_buffers_unchanged"])
            self.assertEqual(run_study(spec, output)["status"], "trained")

    def test_native_image_pretrained_teacher_student_smoke_both_datasets(self):
        # Real ResNet-34/18 architecture and training; local fake pretrained states.
        # This tests plumbing, not the effectiveness of actual ImageNet weights.
        torch.manual_seed(7)
        state34, state18 = models.resnet34(weights=None).state_dict(), models.resnet18(weights=None).state_dict()
        from kd.config import BenchmarkConfig
        from kd.teacher_baseline_matrix import _shortcut_job
        for source in ("waterbirds", "urbancars"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(models.ResNet34_Weights, "get_state_dict", return_value=state34), \
                 patch.object(models.ResNet18_Weights, "get_state_dict", return_value=state18):
                root = Path(temp)
                if source == "waterbirds":
                    waterbirds_release(root / "data" / source / "1.0", per_group=1)
                else:
                    urban_release(root / "data" / source / "1.0")
                raw = _shortcut_job(source, root=root / "data", device="cpu", precision="float32", workers=0,
                                    epochs=1, teacher_seeds=(41,), student_seeds=(42,))["study"]
                teacher = from_dict(raw["teacher"])
                teacher = replace(teacher, train=replace(teacher.train, batch_size=2, threads=2),
                                  benchmark=BenchmarkConfig(warmup=0, iterations=1))
                spec = TeacherBaselineSpec(teacher=teacher, student_model="resnet18", student_weights="IMAGENET1K_V1",
                    recipes=("control",), student_seeds=(42,), teacher_selection_metric="worst_group_accuracy",
                    student_selection_metric="worst_group_accuracy", group_policy={**DEFAULT_GROUP_POLICY, "mechanism_controls": False})
                output = root / "study"
                self.assertEqual(run_study(spec, output)["status"], "trained")
                self.assertEqual(run_study(spec, output, phase="evaluate")["status"], "evaluated")
                summary = json.loads((output / "kd_control_teacher41_seed42" / "summary.json").read_text())
                self.assertEqual(summary["initialization"]["weights"], "IMAGENET1K_V1")
                report = json.loads((output / "comparison.json").read_text())
                self.assertEqual(report["primary_metric"], "worst_group_accuracy")
                if source == "urbancars":
                    self.assertTrue(all("urbancars_background_gap" in result for result in report["per_run"].values()))

    def test_unsealed_invalid_winner_is_not_blessed_by_a_new_manifest(self):
        from kd.checkpoints import fingerprint
        with tempfile.TemporaryDirectory() as temp, patch("kd.shortcut_data.datasets.MNIST", FakeMNIST), contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            spec = small_cmnist_spec(str(root / "data"))
            output = root / "study"
            run_study(spec, output)
            manifest = output / "study_completion.json"
            manifest.unlink()
            path = output / "selection.json"
            selection = json.loads(path.read_text())
            selection["recipe"] = "not_a_recipe"
            path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, "winner changed"):
                run_study(spec, output)
            self.assertFalse(manifest.exists())

    def test_interrupted_repair_reuses_verified_localization_and_candidate_cache(self):
        from kd.shortcut_repair import train_shortcut_repair
        with tempfile.TemporaryDirectory() as temp, patch("kd.shortcut_data.datasets.MNIST", FakeMNIST), contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            spec = small_cmnist_spec(str(root / "data"), repair=True)
            output = root / "study"
            run_study(spec, output)
            teachers, _, _ = study_configs(spec, output)
            config = next(config for condition, config in teachers if condition == "repair_random_channels")
            directory = output / config.name
            (directory / "completion.json").unlink()
            with patch("kd.shortcut_repair.collect_teacher_diagnostics", side_effect=AssertionError("localization repeated")), \
                 patch("kd.shortcut_repair.train_repair_candidate", side_effect=AssertionError("candidate fit repeated")):
                summary = train_shortcut_repair(spec, "repair_random_channels", config)
            self.assertTrue((directory / "completion.json").exists())
            self.assertIn(summary["selection_metric"], ("worst_group_accuracy",))

    def test_group_report_summary_rejects_tampered_final_metrics(self):
        with tempfile.TemporaryDirectory() as temp, patch("kd.shortcut_data.datasets.MNIST", FakeMNIST), contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            from kd.teacher_baseline_matrix import _shortcut_job
            fixture_study = small_cmnist_spec(str(root / "data")).to_dict()
            def fixture_job(dataset, **settings):
                job = _shortcut_job(dataset, **settings)
                job["study"] = fixture_study
                return job
            with patch("kd.teacher_baseline_matrix._shortcut_job", side_effect=fixture_job):
                materialize_matrix_plan(root / "plan", datasets=("colored_mnist",), root=root / "data", epochs=1,
                                        student_seeds=(42,))
            protocol = json.loads((root / "plan" / "protocol.json").read_text())
            job = protocol["jobs"][0]
            spec = load_spec(job["config"])
            run_study(spec, job["output"])
            run_study(spec, job["output"], phase="evaluate")
            summarize(root / "plan", root / "summary")
            report = json.loads((root / "summary" / "overview.json").read_text())
            self.assertTrue(all(row["status"] == "complete" for row in report["rows"]))
            from kd.teacher_baselines import plan_study
            foreign_protocol = plan_study(replace(spec, matrix_binding=None), job["output"])
            with patch("kd.full_benchmark.plan_study", return_value=foreign_protocol):
                summarize(root / "plan", root / "summary")
            rejected = json.loads((root / "summary" / "overview.json").read_text())
            self.assertTrue(all(row["status"] == "invalid" for row in rejected["rows"]))
            self.assertTrue(all("frozen matrix job" in row["reason"] for row in rejected["rows"]))
            (Path(job["output"]) / "comparison.json").write_text("{}")
            summarize(root / "plan", root / "summary")
            report = json.loads((root / "summary" / "overview.json").read_text())
            self.assertTrue(all(row["status"] == "invalid" for row in report["rows"]))
            self.assertTrue(all(row["selected_kd"] is None for row in report["rows"]))


class FullReportContractTests(unittest.TestCase):
    def report(self):
        values = {"mean": 0.5, "std": 0.0}
        return {"split": "test", "endpoint": "reversed", "validation_selected_recipe": "control",
                "rows": [{"recipe": "control", "role": "recipe", "student": {"worst_group_accuracy": values},
                          "paired_student_delta_vs_control": {"worst_group_accuracy": {"mean": 0.0, "std": 0.0}}}]}

    def test_missing_color_endpoint_is_not_replaced_by_the_other_endpoint(self):
        report = self.report()
        report["endpoint_reports"] = {"reversed": self.report()}
        with self.assertRaisesRegex(ValueError, "endpoint coverage"):
            _report_rows(report, ["reversed", "unbiased"], ["control"], [], "worst_group_accuracy", "control")

    def test_duplicate_method_rows_fail_coverage(self):
        report = self.report()
        report["rows"].append(report["rows"][0])
        with self.assertRaisesRegex(ValueError, "coverage"):
            _report_rows(report, ["test"], ["control"], [], "worst_group_accuracy", "control")

    def test_nonfinite_measurements_are_not_reported_as_complete(self):
        report = self.report()
        report["rows"][0]["student"]["worst_group_accuracy"]["mean"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid aggregate"):
            _report_rows(report, ["test"], ["control"], [], "worst_group_accuracy", "control")

    def test_unavailable_groups_keep_empty_measurements(self):
        report = self.report()
        report["rows"][0]["student"]["worst_group_accuracy"] = {"mean": None, "std": None}
        result = _report_rows(report, ["test"], ["control"], [], "worst_group_accuracy", "control")
        self.assertIsNone(result["test"]["control"]["student"]["worst_group_accuracy"]["mean"])


if __name__ == "__main__":
    unittest.main()
