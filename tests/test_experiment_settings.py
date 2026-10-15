"""Offline conformance checks for docs/experiment-settings.md's default protocol.

Fixtures exercise the real split/config/loss paths, not benchmark effectiveness.
Saved run directories are deliberately not test dependencies.
"""

from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from kd.config import DistillationConfig, from_dict
from kd.data import build_data, image_transform, normalization
from kd.losses import DistillationObjective
from kd.models import ModelOutput
from kd.neuron_surgery import repair_loss, same_model_feature_loss
from kd.neuron_surgery_baselines import baseline_configs
from kd.neuron_surgery_matrix import DATASETS, matrix_spec
from kd.teacher_baseline_matrix import matrix_jobs
from kd.teacher_baselines import TeacherBaselineSpec, _target_classes, study_configs
from kd.teacher_repair_baseline import repair_options


PROFILES = {"balanced": (1, 100), "lt-if10": (10, 100),
            "lt-if50": (50, 100), "lt-if100": (100, 100)}
SMALL_NATIVE = {"cifar10", "cifar100", "svhn", "cinic10"}
RESIZED = {"fashionmnist", "pathmnist", "bloodmnist", "dermamnist", "organamnist"}
FLIPPED = {"cifar10", "cifar100", "cinic10"}


class NativeFixture(Dataset):
    """Naturally unequal classes with distinct native train/val/test partitions."""

    def __init__(self, source, split, transform):
        classes = DATASETS[source]["classes"]
        base = {"train": 120, "val": 40, "test": 25}[split]
        self.targets = np.repeat(np.arange(classes),
                                 [base + label % 5 for label in range(classes)]).tolist()
        self.classes = [f"class_{label}" for label in range(classes)]
        self.categories = self.classes
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.new("RGB", (32, 32), (index % 256, 100, 50))
        return self.transform(image), self.targets[index]


def fixture_factory(source, root, split, transform, download):
    if download:
        raise AssertionError("Conformance tests must not download datasets")
    return NativeFixture(source, split, transform)


def dataset_labels(dataset):
    if isinstance(dataset, Subset):
        return np.asarray(dataset.dataset.targets)[dataset.indices]
    return np.asarray(dataset.targets)


def study_spec(job):
    values = dict(job["study"])
    values["teacher"] = from_dict(values["teacher"])
    values["repair"] = repair_options(values["repair"])
    for name in ("teacher_seeds", "student_seeds", "recipes"):
        values[name] = tuple(values[name])
    return TeacherBaselineSpec(**values)


class ExperimentSettingsTests(unittest.TestCase):
    def test_every_supported_pair_has_matched_default_teacher_student_settings(self):
        for job in matrix_jobs(datasets=tuple(DATASETS), pairs=("convnet6", "resnet34_resnet18")):
            with self.subTest(job=job["id"]):
                spec = study_spec(job)
                teachers, students, supervised = study_configs(spec, "/private/tmp/settings-contract")
                self.assertEqual(spec.teacher_selection_metric, "accuracy")
                self.assertEqual(spec.teacher_seeds, (41,))
                self.assertEqual(spec.student_seeds, (42, 43, 44))
                self.assertEqual(spec.student_selection_metric,
                                 "accuracy" if job["profile"] == "balanced" else "balanced_accuracy")
                all_configs = [*(config for _, config in teachers),
                               *(config for _, _, config in students), *supervised]
                for config in all_configs:
                    train = config.train
                    self.assertEqual(train.epochs, PROFILES[job["profile"]][1])
                    self.assertEqual((train.batch_size, train.learning_rate, train.momentum,
                                      train.weight_decay, train.threads), (128, 0.05, 0.9, 0.0005, 4))
                    self.assertEqual((train.device, train.precision, train.workers), ("auto", "float32", 0))
                    self.assertEqual((train.cuda_mode, train.channels_last, train.persistent_workers,
                                      train.prefetch_factor), ("fast", True, False, 2))
                    self.assertEqual((config.benchmark.warmup, config.benchmark.iterations), (20, 100))
                    self.assertEqual(config.data, spec.teacher.data)
                    self.assertFalse(config.calibration.enabled or config.cpc.enabled)
                    self.assertEqual(config.surgery.action, "none")
                    self.assertEqual(config.supervised_loss.method, "cross_entropy")
                for _, _, config in students:
                    self.assertEqual(config.distillation, DistillationConfig(method="kd"))
                self.assertTrue(all(config.distillation.method == "supervised" for config in supervised))

    def test_all_sources_use_the_documented_base_views_and_normalization(self):
        default_norm = ((0.5,) * 3, (0.5,) * 3)
        norms = {
            "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            "cinic10": ((0.47889522, 0.47227842, 0.43047404),
                        (0.24205776, 0.23828046, 0.25874835)),
            "fashionmnist": ((0.2860,) * 3, (0.3530,) * 3),
        }
        for source in DATASETS:
            with self.subTest(source=source):
                teacher, _ = baseline_configs("/private/tmp/settings-contract", "unused", "cpu",
                                             matrix_spec(source, "balanced"))
                train = image_transform(teacher.data, training=True).transforms
                canonical = image_transform(teacher.data, training=False).transforms
                expected_norm = norms.get(source, default_norm if source in SMALL_NATIVE | RESIZED else
                                          ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
                self.assertEqual(normalization(source), expected_norm)
                if source in SMALL_NATIVE | RESIZED:
                    expected_train = ([transforms.Resize] if source in RESIZED else []) + [transforms.RandomCrop]
                    if source in FLIPPED:
                        expected_train.append(transforms.RandomHorizontalFlip)
                    expected_val = [transforms.Resize] if source in RESIZED else []
                    crop = next(operation for operation in train if isinstance(operation, transforms.RandomCrop))
                    self.assertEqual((crop.size, crop.padding), ((32, 32), 4))
                else:
                    expected_train = [transforms.RandomResizedCrop, transforms.Lambda]
                    expected_val = [transforms.Resize, transforms.CenterCrop, transforms.Lambda]
                    self.assertEqual((train[0].size, train[0].scale, train[0].ratio),
                                     ((32, 32), (0.85, 1.0), (0.9, 1.1)))
                    self.assertEqual(canonical[0].size, 36)
                    self.assertEqual(canonical[1].size, (32, 32))
                self.assertEqual([type(operation) for operation in train],
                                 expected_train + [transforms.ToTensor, transforms.Normalize])
                self.assertEqual([type(operation) for operation in canonical],
                                 expected_val + [transforms.ToTensor, transforms.Normalize])
                for operation in train:
                    if isinstance(operation, transforms.RandomHorizontalFlip):
                        self.assertEqual(operation.p, 0.5)

    def test_all_52_split_profiles_sample_train_and_val_but_never_test(self):
        with tempfile.TemporaryDirectory() as root, patch("kd.data._native_dataset", side_effect=fixture_factory):
            for source in DATASETS:
                balanced_counts, test_labels, test_indices = None, None, None
                for profile, (factor, _) in PROFILES.items():
                    with self.subTest(source=source, profile=profile):
                        spec = matrix_spec(source, profile)
                        config, _ = baseline_configs("/private/tmp/settings-contract", root, "cpu", spec)
                        data = build_data(config.data, config.train, include_test=True)
                        counts = (data.provenance["train_per_class"], data.provenance["val_per_class"])
                        if balanced_counts is None:
                            balanced_counts = counts
                            test_labels = dataset_labels(data.test.dataset)
                            test_indices = getattr(data.test.dataset, "indices", None)
                            if source in {"cinic10", "pathmnist", "bloodmnist", "dermamnist", "organamnist"}:
                                expected_train = [120 + label % 5 for label in range(spec.dataset_num_classes)]
                                expected_val = [40 + label % 5 for label in range(spec.dataset_num_classes)]
                            else:
                                native = [120 + label % 5 for label in range(spec.dataset_num_classes)]
                                expected_val = [round(n * 0.1) for n in native]
                                test_fraction = {"caltech101": 0.2, "eurosat": 0.1}.get(source, 0)
                                expected_train = [n - v - round(n * test_fraction)
                                                  for n, v in zip(native, expected_val)]
                            self.assertEqual(counts, (expected_train, expected_val))
                        for actual, original in zip(counts, balanced_counts):
                            expected = [max(1, round(n * factor ** (-k / (len(original) - 1))))
                                        for k, n in enumerate(original)]
                            self.assertEqual(actual, expected)
                        np.testing.assert_array_equal(dataset_labels(data.test.dataset), test_labels)
                        self.assertEqual(getattr(data.test.dataset, "indices", None), test_indices)
                        self.assertIsNone(data.confirmation)
                        # Model seed changes neither training nor selection membership.
                        other = build_data(config.data, replace(config.train, seed=44), include_test=False,
                                           diagnostic=True)
                        self.assertEqual(data.train.dataset.indices, other.train.dataset.indices)
                        self.assertEqual(data.val.dataset.indices, other.val.dataset.indices)
                        targets = _target_classes(counts[0], long_tail=factor > 1)
                        self.assertEqual(len(targets), (spec.dataset_num_classes + 1) // 2
                                         if factor > 1 else spec.dataset_num_classes)

    def test_exact_documented_kd_formula_and_five_epoch_warmup(self):
        student = torch.tensor([[1.2, -0.3, 0.5], [-0.1, 0.7, -0.2]], requires_grad=True)
        teacher = torch.tensor([[0.1, 0.8, -0.4], [1.1, -0.2, 0.3]], requires_grad=True)
        labels = torch.tensor([0, 2])
        objective = DistillationObjective(DistillationConfig(method="kd"))
        ce = F.cross_entropy(student, labels)
        teacher_probability = (teacher.detach() / 4).softmax(1)
        kl = (teacher_probability * (teacher_probability.log() - (student / 4).log_softmax(1))).sum() / 2
        for epoch in range(7):
            result = objective(ModelOutput(student), ModelOutput(teacher), labels, epoch)
            expected = 0.5 * ce + 0.5 * min((epoch + 1) / 5, 1) * 16 * kl
            torch.testing.assert_close(result["total"], expected)
            self.assertEqual(result["features"].item(), 0)
        result["total"].backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_repair_preservation_has_documented_reductions_and_coefficients(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            candidate_features = {"stage2": torch.randn(2, 3, 2, 2, requires_grad=True),
                                  "stage3": torch.randn(2, 5, 1, 1, requires_grad=True)}
            anchor_features = {name: torch.randn_like(value, requires_grad=True)
                               for name, value in candidate_features.items()}
            stages = ("stage2", "stage3")
            manual_feature = torch.stack([
                ((value / value.norm(dim=1, keepdim=True)) -
                 (anchor_features[name].detach() / anchor_features[name].detach().norm(dim=1, keepdim=True)))
                .square().sum(dim=1).mean() for name, value in candidate_features.items()]).mean()
            torch.testing.assert_close(same_model_feature_loss(candidate_features, anchor_features, stages),
                                       manual_feature)
            target_logits = torch.randn(2, 4, requires_grad=True)
            preservation_logits = torch.randn(2, 4, requires_grad=True)
            anchor_logits = torch.randn(2, 4, requires_grad=True)
            labels = torch.tensor([0, 3])
            result = repair_loss(ModelOutput(target_logits), labels,
                                 ModelOutput(preservation_logits, candidate_features),
                                 ModelOutput(anchor_logits, anchor_features), labels,
                                 feature_stages=stages)
            probability = (anchor_logits.detach() / 4).softmax(1)
            manual_kl = (probability * (probability.log() - (preservation_logits / 4).log_softmax(1))).sum() / 2
            expected = F.cross_entropy(target_logits, labels) + 16 * manual_kl + 0.25 * manual_feature
            torch.testing.assert_close(result["total"], expected)
            self.assertEqual(result["preservation_ce"].item(), 0)
            result["total"].backward()
            self.assertIsNone(anchor_logits.grad)
            self.assertTrue(all(value.grad is None for value in anchor_features.values()))


if __name__ == "__main__":
    unittest.main()
