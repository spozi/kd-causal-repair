"""Offline contracts for the adapted pretrained 32-pixel baseline protocol."""

import contextlib
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import models, transforms

from kd.checkpoints import load_model_checkpoint, metadata, save_checkpoint, tensor_state_fingerprint
from kd.config import BenchmarkConfig, DataConfig, ExperimentConfig, ModelConfig, from_dict
from kd.data import MEAN, STD, build_data, image_transform, normalization
from kd.full_benchmark import preflight, summarize
from kd.models import create_model
from kd.neuron_surgery import train_repair_candidate
from kd.neuron_surgery_matrix import DATASETS as CLASS_DATASETS
from kd.neuron_surgery_study import NeuronSurgerySpec
from kd.teacher_baseline_matrix import DEFAULT_PAIRS, PAIRINGS, matrix_jobs, materialize_matrix_plan, plan_matrix
from kd.teacher_baselines import TeacherBaselineSpec, load_spec, run_study, study_configs
from kd.teacher_repair_baseline import REPAIR_NAME, TeacherRepairOptions, repair_options, repair_spec


class CatalogFixture(Dataset):
    def __init__(self, source, split, transform, per_class=10):
        count = CLASS_DATASETS[source]["classes"]
        self.classes = self.categories = [f"class_{i}" for i in range(count)]
        self.targets = np.repeat(np.arange(count), per_class).tolist()
        self.transform = transform
        self.gray = source in {"fashionmnist", "organamnist"}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.new("L" if self.gray else "RGB", (32, 32), 128 if self.gray else (128, 96, 64))
        return self.transform(image), self.targets[index]


def fixture_factory(source, root, split, transform, download):
    if download:
        raise AssertionError("Offline fixtures must not download")
    return CatalogFixture(source, split, transform)


def study_spec(job):
    raw = dict(job["study"])
    raw["teacher"] = from_dict(raw["teacher"])
    if raw["repair"] is not None:
        raw["repair"] = repair_options(raw["repair"])
    for key in ("recipes", "teacher_seeds", "student_seeds"):
        raw[key] = tuple(raw[key])
    return TeacherBaselineSpec(**raw)


class PretrainedTransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_default_52_class_settings_match_pretrained_teacher_student_contract(self):
        self.assertEqual(DEFAULT_PAIRS, ("resnet34_resnet18_smallstem_imagenet",))
        jobs = matrix_jobs(datasets=tuple(CLASS_DATASETS))
        self.assertEqual(len(jobs), 52)
        for job in jobs:
            with self.subTest(job=job["id"]):
                spec = study_spec(job)
                self.assertEqual(spec.teacher.data.image_size, 32)
                self.assertEqual(spec.teacher.data.preprocessing, "imagenet-smallstem-v1")
                self.assertEqual(spec.teacher.student, ModelConfig("resnet34_smallstem", weights="IMAGENET1K_V1"))
                self.assertEqual((spec.student_model, spec.student_weights), ("resnet18_smallstem", "IMAGENET1K_V1"))
                self.assertEqual((spec.teacher.train.batch_size, spec.teacher.train.learning_rate,
                                  spec.teacher.train.epochs), (128, 0.001, 100))
                teachers, students, supervised = study_configs(spec, "/private/tmp/transfer-contract")
                for _, _, student in students:
                    self.assertEqual(student.data, spec.teacher.data)
                    self.assertEqual(student.student.weights, "IMAGENET1K_V1")
                    self.assertIsNone(student.teacher.weights)
                    self.assertTrue(student.teacher.checkpoint)
                for student in supervised:
                    self.assertEqual(student.student.weights, "IMAGENET1K_V1")
                    self.assertEqual(student.data, spec.teacher.data)
                repair = repair_spec(spec, 41)
                self.assertEqual((repair.dataset_image_size, repair.dataset_preprocessing), (32, "imagenet-smallstem-v1"))
                self.assertEqual((repair.localization_batch_size, repair.screening_batch_size,
                                  repair.repair_batch_size, repair.repair_epochs), (16, 128, 64, 20))

    def test_55_default_107_two_pairs_and_159_all_pairs_are_distinct_protocols(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            default = plan_matrix(root / "default", root=root / "data")
            self.assertEqual((default["job_count"], default["counts"]["total_fits_max"]), (55, 3153))
            two = plan_matrix(root / "two", root=root / "data", pairs=("resnet34_resnet18_smallstem_imagenet", "convnet6"))
            self.assertEqual((two["job_count"], two["counts"]["total_fits_max"]), (107, 6057))
            all_pairs = plan_matrix(root / "all", root=root / "data", pairs=tuple(PAIRINGS))
            self.assertEqual((all_pairs["job_count"], all_pairs["counts"]["total_fits_max"]), (159, 8961))
            self.assertEqual(list(root.iterdir()), [])

    def test_compact_presets_and_native_shortcut_inputs_remain_available(self):
        for job in matrix_jobs(datasets=tuple(CLASS_DATASETS), pairs=("convnet6", "resnet34_resnet18")):
            spec = study_spec(job)
            self.assertEqual((spec.teacher.data.image_size, spec.teacher.data.preprocessing), (32, "legacy"))
            self.assertIsNone(spec.teacher.student.weights)
            self.assertIsNone(spec.student_weights)
        shortcuts = {job["dataset"]: study_spec(job) for job in matrix_jobs(datasets=("colored_mnist", "waterbirds", "urbancars"))}
        self.assertEqual(shortcuts["colored_mnist"].teacher.data.image_size, 32)
        self.assertIsNone(shortcuts["colored_mnist"].student_weights)
        for source, size in (("waterbirds", 224), ("urbancars", 256)):
            self.assertEqual(shortcuts[source].teacher.data.image_size, size)
            self.assertEqual(shortcuts[source].student_weights, "IMAGENET1K_V1")

    def test_all_catalog_inputs_use_rgb_imagenet_normalization_and_32_views(self):
        for job in matrix_jobs(datasets=tuple(CLASS_DATASETS), profiles=("balanced",)):
            config = study_spec(job).teacher.data
            with self.subTest(source=config.source):
                canonical = image_transform(config, training=False)
                self.assertIsInstance(canonical.transforms[0], transforms.Lambda)
                for operation in canonical.transforms:
                    if isinstance(operation, transforms.Resize):
                        self.assertIn(operation.size, (36, (32, 32)))
                self.assertEqual(normalization(config.source, config.preprocessing), (MEAN, STD))
                image = canonical(Image.new("L", (32, 32), 128))
                self.assertEqual(tuple(image.shape), (3, 32, 32))
                torch.testing.assert_close(image[:, 0, 0],
                    (torch.full((3,), 128 / 255) - torch.tensor(MEAN)) / torch.tensor(STD))
                training = image_transform(config, training=True)
                self.assertTrue(any(isinstance(op, (transforms.RandomCrop, transforms.RandomResizedCrop)) for op in training.transforms))
                self.assertEqual(any(isinstance(op, transforms.RandomHorizontalFlip) for op in training.transforms), config.horizontal_flip)

    def test_resolution_and_normalization_change_never_changes_split_membership(self):
        with tempfile.TemporaryDirectory() as temp, patch("kd.data._native_dataset", side_effect=fixture_factory):
            for source in CLASS_DATASETS:
                for profile in ("balanced", "lt-if100"):
                    transfer = study_spec(matrix_jobs(datasets=(source,), profiles=(profile,), root=temp)[0]).teacher
                    compact = replace(transfer, data=replace(transfer.data, image_size=32, preprocessing="legacy"))
                    old, new = (build_data(config.data, config.train, include_test=False, diagnostic=True) for config in (compact, transfer))
                    for key in ("train_index_sha256", "val_index_sha256", "split_indices_sha256", "train_per_class", "val_per_class"):
                        self.assertEqual(old.provenance[key], new.provenance[key])
                    self.assertEqual(tuple(new.val.dataset[0][0].shape), (3, 32, 32))
                    self.assertEqual(new.provenance["input_protocol"]["name"], "imagenet-smallstem-v1")

    def test_legacy_config_and_study_serialization_do_not_gain_new_default_fields(self):
        self.assertNotIn("preprocessing", ExperimentConfig().to_dict()["data"])
        old = NeuronSurgerySpec().to_dict()
        self.assertNotIn("dataset_preprocessing", old)
        self.assertNotIn("localization_batch_size", old)
        self.assertNotIn("screening_batch_size", old)
        with self.assertRaises(ValueError):
            replace(NeuronSurgerySpec(), dataset_image_size=224).validate()

    def test_transfer_validation_rejects_implicit_resolution_or_one_role_scratch(self):
        spec = study_spec(matrix_jobs(datasets=("cifar10",), profiles=("balanced",))[0])
        for data in (replace(spec.teacher.data, image_size=224),
                     replace(spec.teacher.data, preprocessing="legacy"),
                     replace(spec.teacher.data, augmentation="strong")):
            with self.assertRaises(ValueError):
                replace(spec.teacher, data=data).validate(require_teacher=False)
        with self.assertRaisesRegex(ValueError, "both teacher and student"):
            replace(spec, student_weights=None).validate()

    def test_checkpoint_binds_actual_input_protocol_not_legacy_source_statistics(self):
        classes = ["a", "b"]
        expected = metadata("resnet18_smallstem", classes, 32, "cifar10", preprocessing="imagenet-smallstem-v1")
        self.assertEqual(expected["preprocessing"]["mean"], list(MEAN))
        model = Mock()
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.pt"
            save_checkpoint(checkpoint, {**expected, "student": {}})
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                load_model_checkpoint(model, checkpoint, metadata("resnet18_smallstem", classes, 32, "cifar10"))
            model.load_state_dict.assert_not_called()
            load_model_checkpoint(model, checkpoint, expected)
            model.load_state_dict.assert_called_once()

    def test_preflight_checks_each_input_contract_and_reports_overlap_warnings(self):
        with tempfile.TemporaryDirectory() as temp, patch("kd.data._native_dataset", side_effect=fixture_factory):
            root = Path(temp)
            materialize_matrix_plan(root / "plan", root=root / "data", datasets=("cinic10", "stl10"),
                                    profiles=("balanced",), pairs=("resnet34_resnet18_smallstem_imagenet", "convnet6"))
            checked = preflight(root / "plan")
            self.assertEqual(checked["status"], "ready")
            self.assertEqual({row["dataset"] for row in checked["warnings"]}, {"cinic10", "stl10"})
            for source in ("cinic10", "stl10"):
                self.assertEqual({tuple(row["input_shape"]) for row in checked["datasets"][source]["input_contracts"]},
                                 {(3, 32, 32)})
                self.assertEqual({row["preprocessing"] for row in checked["datasets"][source]["input_contracts"]},
                                 {"imagenet-smallstem-v1", "legacy"})

    def test_overview_retains_overlap_disclosure_without_claiming_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            materialize_matrix_plan(root / "plan", root=root / "data")
            summarize(root / "plan", root / "summary")
            report = json.loads((root / "summary/overview.json").read_text())
            self.assertEqual(report["endpoint_rows"], 56)
            for row in report["rows"]:
                if row["dataset"] in {"cinic10", "stl10"}:
                    self.assertEqual(row["pretraining_overlap"]["audit_status"], "not_audited")
                    self.assertTrue(row["pretraining_overlap"]["imagenet_derived_dataset"])
                self.assertIsNone(row["selected_kd"])
            self.assertIn("Pretraining-overlap warning", (root / "summary/overview.md").read_text())

    def test_bound_transfer_study_repairs_distills_resumes_and_aggregates_offline(self):
        # Actual ResNet architectures; locally supplied states are not actual
        # ImageNet weights. This checks loading/plumbing, not transfer efficacy.
        # Force a one-channel screening result to exercise the candidate fitter;
        # repair optimization, validation selection and checkpoint/KD IO are real.
        torch.manual_seed(7)
        state34, state18 = models.resnet34(weights=None).state_dict(), models.resnet18(weights=None).state_dict()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
             patch("kd.data._native_dataset", side_effect=lambda source, root, split, transform, download:
                   CatalogFixture(source, split, transform, per_class=4)), \
             patch.object(models.ResNet34_Weights, "get_state_dict", return_value=state34), \
             patch.object(models.ResNet18_Weights, "get_state_dict", return_value=state18), \
             patch("kd.neuron_surgery_study._data_available"), \
             patch("kd.neuron_surgery_study.causal_channel_validation", return_value={
                 "performed": True, "channels": [], "ranking": [{"stage": "stage4", "channel": 0}]}):
            root = Path(temp)
            job = matrix_jobs(datasets=("cifar10",), profiles=("balanced",), root=root / "data",
                              epochs=1, device="cpu", student_seeds=(42,))[0]
            spec = study_spec(job)
            spec = replace(spec, recipes=("control", REPAIR_NAME),
                repair=TeacherRepairOptions(channel_budgets=(1,), learning_rates=(0.001,),
                    repair_epochs=1, repair_samples_per_epoch=2, repair_batch_size=2,
                    companion_count=1, allow_sparse_class_fallback=True),
                teacher=replace(spec.teacher, data=replace(spec.teacher.data, validation_fraction=0.5),
                                train=replace(spec.teacher.train, batch_size=2, threads=2),
                                benchmark=BenchmarkConfig(warmup=0, iterations=1)))
            job["study"] = spec.to_dict()
            with patch("kd.teacher_baseline_matrix.matrix_jobs", return_value=[job]):
                materialize_matrix_plan(root / "plan", root=root / "data")
            protocol = json.loads((root / "plan/protocol.json").read_text())
            job = protocol["jobs"][0]
            spec = load_spec(job["config"])
            self.assertEqual(run_study(spec, job["output"])["status"], "trained")
            repair_root = Path(job["output"]) / "teacher_repair_seed41"
            candidate = torch.load(repair_root / "repair/repair_candidates/budget1_lr0.001/teacher.pt", weights_only=True)
            self.assertEqual(candidate["preprocessing"]["input_protocol"], "imagenet-smallstem-v1")
            repaired = json.loads((repair_root / "summary.json").read_text())
            control = json.loads((Path(job["output"]) / "teacher_control_seed41/summary.json").read_text())
            self.assertEqual(repaired["initialization"], control["initialization"])
            self.assertFalse((repair_root / "repair/teacher_evaluation").exists())
            with patch("kd.teacher_training.train_teacher", side_effect=AssertionError("retraining")), \
                 patch("kd.teacher_baselines.run_experiment", side_effect=AssertionError("retraining")):
                self.assertEqual(run_study(spec, job["output"])["status"], "trained")
            self.assertEqual(run_study(spec, job["output"], phase="evaluate")["status"], "evaluated")
            summarize(root / "plan", root / "summary")
            row = json.loads((root / "summary/overview.json").read_text())["rows"][0]
            self.assertEqual(row["status"], "complete")
            self.assertIsNotNone(row["control_kd"])
            self.assertIsNotNone(row["repaired_kd"])
            checkpoint = torch.load(Path(job["output"]) / "kd_control_teacher41_seed42/student.pt", weights_only=True)
            self.assertEqual(checkpoint["preprocessing"]["input_protocol"], "imagenet-smallstem-v1")

    def test_resnet_channel_repair_at_32_preserves_unselected_coordinates_and_buffers(self):
        data = DataConfig(source="cifar10", num_classes=10, image_size=32, preprocessing="imagenet-smallstem-v1")
        dataset = CatalogFixture("cifar10", "train", image_transform(data, training=False))
        anchor = create_model("resnet34_smallstem", 10).requires_grad_(False).eval()
        candidate = deepcopy(anchor)
        _, verification = train_repair_candidate(candidate, anchor, dataset, np.asarray(dataset.targets),
            [0, 2], [4, 6], [{"stage": "stage4", "channel": 0}], torch.device("cpu"),
            epochs=1, samples_per_epoch=2, batch_size=2, learning_rate=0.001,
            precision="float32", channels_last=False)
        self.assertTrue(verification["all_buffers_unchanged"])
        self.assertTrue(verification["all_unselected_coordinates_unchanged"])

    def test_pretrained_stem_adaptation_and_retained_backbones_for_all_resnet_depths(self):
        for depth in (18, 34, 50):
            with self.subTest(depth=depth):
                torch.manual_seed(7)
                fixture = getattr(models, f"resnet{depth}")(weights=None).state_dict()
                original = fixture["conv1.weight"].clone()
                enum = getattr(models, f"ResNet{depth}_Weights")
                with patch.object(enum, "get_state_dict", return_value=fixture):
                    torch.manual_seed(41)
                    model = create_model(f"resnet{depth}_smallstem", 10, weights="IMAGENET1K_V1").eval()
                resized = torch.nn.functional.interpolate(original, size=(3, 3), mode="bilinear", align_corners=False)
                old_norm = original.square().sum((1, 2, 3), keepdim=True).sqrt()
                new_norm = resized.square().sum((1, 2, 3), keepdim=True).sqrt()
                expected = resized * (old_norm / new_norm)
                torch.testing.assert_close(model.backbone.conv1.weight, expected)
                torch.testing.assert_close(model.backbone.conv1.weight.square().sum((1, 2, 3)).sqrt(), old_norm.flatten())
                self.assertEqual(model.backbone.conv1.stride, (1, 1))
                self.assertEqual(model.backbone.conv1.padding, (1, 1))
                self.assertIsInstance(model.backbone.maxpool, torch.nn.Identity)
                for key, value in model.backbone.state_dict().items():
                    if key != "conv1.weight" and not key.startswith("fc."):
                        torch.testing.assert_close(value, fixture[key], rtol=0, atol=0)
                info = model.initialization["stem_adaptation"]
                self.assertEqual(info["name"], "bilinear_l2_7to3_v1")
                self.assertEqual(info["adapted_stem_tensor_sha256"],
                                 tensor_state_fingerprint({"conv1.weight": model.backbone.conv1.weight}))
                with torch.no_grad():
                    output = model(torch.zeros(2, 3, 32, 32), return_features=True)
                self.assertEqual(tuple(output.logits.shape), (2, 10))
                self.assertEqual([features.shape[-1] for features in output.features.values()], [32, 16, 8, 4])
                del model, fixture

    def test_pretrained_smallstem_aliases_have_identical_seeded_initialization(self):
        fixture = models.resnet18(weights=None).state_dict()
        with patch.object(models.ResNet18_Weights, "get_state_dict", return_value=fixture):
            torch.manual_seed(42)
            neutral = create_model("resnet18_smallstem", 10, weights="IMAGENET1K_V1")
            torch.manual_seed(42)
            legacy = create_model("cifar_resnet18", 10, weights="IMAGENET1K_V1")
        self.assertEqual(tensor_state_fingerprint(neutral.state_dict()), tensor_state_fingerprint(legacy.state_dict()))

    def test_zero_pretrained_stem_filter_is_kept_zero_without_nan(self):
        fixture = models.resnet18(weights=None).state_dict()
        fixture["conv1.weight"][0].zero_()
        with patch.object(models.ResNet18_Weights, "get_state_dict", return_value=fixture):
            model = create_model("resnet18_smallstem", 2, weights="IMAGENET1K_V1")
        self.assertTrue(torch.isfinite(model.backbone.conv1.weight).all())
        self.assertEqual(model.backbone.conv1.weight[0].count_nonzero().item(), 0)

    def test_nonzero_filter_that_resizes_to_zero_is_rejected_not_randomly_replaced(self):
        fixture = models.resnet18(weights=None).state_dict()
        fixture["conv1.weight"][0].zero_()
        fixture["conv1.weight"][0, 0, 2, 2] = 1.0
        with patch.object(models.ResNet18_Weights, "get_state_dict", return_value=fixture):
            with self.assertRaisesRegex(ValueError, "preserving filter norms"):
                create_model("resnet18_smallstem", 2, weights="IMAGENET1K_V1")

    def test_smallstem_pretrained_config_rejects_legacy_normalization_or_ordinary_stem(self):
        spec = study_spec(matrix_jobs(datasets=("cifar10",), profiles=("balanced",))[0])
        with self.assertRaisesRegex(ValueError, "explicit imagenet-smallstem"):
            replace(spec.teacher, data=replace(spec.teacher.data, preprocessing="legacy")).validate(require_teacher=False)
        with self.assertRaisesRegex(ValueError, "adapted small-stem"):
            replace(spec.teacher, student=ModelConfig("resnet34", weights="IMAGENET1K_V1")).validate(require_teacher=False)


if __name__ == "__main__":
    unittest.main()
