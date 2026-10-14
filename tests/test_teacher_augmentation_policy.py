from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from kd.config import (BenchmarkConfig, DataConfig, DistillationConfig,
                       ExperimentConfig, ModelConfig, TrainConfig)
from kd.data import CATALOG_SOURCES, MEDMNIST_SOURCES, image_transform, normalization
from kd.teacher_training import (PolicyRandAugment, RECIPES, add_randaugment,
                                 randaugment_policy, train_teacher)


STANDARD_OPERATIONS = (
    "Identity", "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate",
    "Brightness", "Color", "Contrast", "Sharpness", "Posterize", "Solarize",
    "AutoContrast", "Equalize",
)
DIGIT_SIGN_OPERATIONS = (
    "Identity", "Brightness", "Contrast", "Sharpness", "AutoContrast", "Equalize",
)
MEDICAL_OPERATIONS = ("Identity", "Brightness", "Contrast", "Sharpness")
STANDARD_SOURCES = {
    "cifar10", "cifar100", "cinic10", "fashionmnist", "caltech101", "eurosat", "stl10",
}


class PILImages(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self.transform(self.image.copy()), index % 2


class TeacherAugmentationPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_all_thirteen_sources_have_the_declared_operation_spaces(self):
        self.assertEqual(len(CATALOG_SOURCES), 13)
        self.assertEqual(CATALOG_SOURCES, STANDARD_SOURCES | {"svhn", "gtsrb"} | MEDMNIST_SOURCES)
        vanilla = transforms.RandAugment()._augmentation_space(31, (32, 32))
        self.assertEqual(tuple(vanilla), STANDARD_OPERATIONS)
        for source in sorted(CATALOG_SOURCES):
            with self.subTest(source=source):
                if source in {"svhn", "gtsrb"}:
                    name, operations = "restricted_digit_sign_photometric", DIGIT_SIGN_OPERATIONS
                elif source in MEDMNIST_SOURCES:
                    name, operations = "restricted_medical_photometric", MEDICAL_OPERATIONS
                else:
                    name, operations = "torchvision_standard", STANDARD_OPERATIONS
                self.assertEqual(randaugment_policy(source), {"name": name, "operations": list(operations)})
                augmentation = PolicyRandAugment(operations=operations)
                self.assertEqual(tuple(augmentation._augmentation_space(31, (32, 32))), operations)
                if source not in STANDARD_SOURCES:
                    excluded = {"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate",
                                "Color", "Posterize", "Solarize"}
                    self.assertFalse(set(operations) & excluded)
        # Policy callers cannot mutate subsequent source-specific declarations.
        changed = randaugment_policy("svhn")
        changed["operations"].append("Rotate")
        self.assertEqual(randaugment_policy("svhn")["operations"], list(DIGIT_SIGN_OPERATIONS))

    def test_invalid_operation_selections_fail(self):
        for operations in ((), ("Identity", "Identity")):
            with self.subTest(operations=operations), self.assertRaisesRegex(ValueError, "nonempty and unique"):
                PolicyRandAugment(operations=operations)
        unknown = PolicyRandAugment(operations=("Identity", "UnknownOperation"))
        with self.assertRaisesRegex(ValueError, "Unsupported RandAugment operations"):
            unknown._augmentation_space(31, (32, 32))
        image = Image.new("RGB", (32, 32))
        with self.assertRaisesRegex(ValueError, "UnknownOperation"):
            unknown(image)

    def test_every_chosen_torchvision_pil_operation_executes_reproducibly(self):
        image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
        with torch.random.fork_rng(devices=[]):
            for operation in STANDARD_OPERATIONS:
                with self.subTest(operation=operation):
                    augmentation = PolicyRandAugment(operations=(operation,), num_ops=2,
                                                      magnitude=9, fill=(128, 128, 128))
                    torch.manual_seed(2026)
                    first = augmentation(image.copy())
                    torch.manual_seed(2026)
                    second = augmentation(image.copy())
                    self.assertIsInstance(first, Image.Image)
                    self.assertEqual(first.size, (32, 32))
                    self.assertEqual(first.mode, "RGB")
                    self.assertEqual(first.tobytes(), second.tobytes())

    def test_standard_policy_matches_vanilla_randaugment_exactly(self):
        image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
        arguments = {"num_ops": 2, "magnitude": 9, "fill": (128, 128, 128)}
        vanilla = transforms.RandAugment(**arguments)
        selected = PolicyRandAugment(operations=randaugment_policy("cifar10")["operations"], **arguments)
        with torch.random.fork_rng(devices=[]):
            for seed in range(32):
                with self.subTest(seed=seed):
                    torch.manual_seed(seed)
                    expected = vanilla(image.copy())
                    torch.manual_seed(seed)
                    actual = selected(image.copy())
                    self.assertEqual(actual.tobytes(), expected.tobytes())
                    # Selecting operations in the same order also consumes exactly the same RNG.
                    torch.manual_seed(seed)
                    vanilla(image.copy())
                    expected_rng = torch.get_rng_state()
                    torch.manual_seed(seed)
                    selected(image.copy())
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_insertion_is_training_only_and_preserves_canonical_transforms(self):
        with torch.random.fork_rng(devices=[]):
            for source in sorted(CATALOG_SOURCES):
                with self.subTest(source=source):
                    data_config = DataConfig(source=source, image_size=32, horizontal_flip=False)
                    original = image_transform(data_config, training=True)
                    original_operations = tuple(original.transforms)
                    base = PILImages(original)
                    validation = PILImages(image_transform(data_config, training=False))
                    validation_transform = validation.transform
                    expected_validation = validation[0][0]
                    nested = Subset(Subset(base, (0, 1, 2, 3)), (0, 1))
                    data = SimpleNamespace(train=DataLoader(nested, batch_size=2),
                                           val=DataLoader(validation, batch_size=2))
                    config = SimpleNamespace(data=data_config)
                    add_randaugment(data, config, RECIPES["randaugment"])
                    operations = base.transform.transforms
                    index = next(i for i, operation in enumerate(original_operations)
                                 if isinstance(operation, transforms.ToTensor))
                    self.assertIsInstance(operations[index], transforms.RandAugment)
                    self.assertIsInstance(operations[index + 1], transforms.ToTensor)
                    self.assertEqual(tuple(original.transforms), original_operations)
                    self.assertEqual(tuple(operations[:index]), original_operations[:index])
                    self.assertEqual(tuple(operations[index + 1:]), original_operations[index:])
                    self.assertIs(validation.transform, validation_transform)
                    self.assertTrue(torch.equal(validation[0][0], expected_validation))
                    if source in STANDARD_SOURCES:
                        self.assertIs(type(operations[index]), transforms.RandAugment)
                    else:
                        self.assertIsInstance(operations[index], PolicyRandAugment)
                        self.assertEqual(operations[index].operations,
                                         tuple(randaugment_policy(source)["operations"]))
                    mean, _ = normalization(source)
                    self.assertEqual(operations[index].fill, tuple(round(255 * value) for value in mean))
                    images, _ = next(iter(data.train))
                    self.assertEqual(images.shape, (2, 3, 32, 32))
                    self.assertTrue(torch.isfinite(images).all())

    def test_cifar_training_transform_matches_vanilla_insertion(self):
        data_config = DataConfig(source="cifar10", image_size=32)
        original = image_transform(data_config, training=True)
        base = PILImages(original)
        data = SimpleNamespace(train=SimpleNamespace(dataset=base))
        add_randaugment(data, SimpleNamespace(data=data_config), RECIPES["randaugment"])
        operations = list(original.transforms)
        index = next(i for i, operation in enumerate(operations) if isinstance(operation, transforms.ToTensor))
        operations.insert(index, transforms.RandAugment(num_ops=2, magnitude=9, fill=(128, 128, 128)))
        expected_transform = transforms.Compose(operations)
        with torch.random.fork_rng(devices=[]):
            for seed in range(16):
                torch.manual_seed(seed)
                expected = expected_transform(base.image.copy())
                torch.manual_seed(seed)
                actual = base.transform(base.image.copy())
                self.assertTrue(torch.equal(actual, expected), seed)

    def test_control_does_not_change_the_training_transform(self):
        config = SimpleNamespace(data=DataConfig(image_size=32))
        dataset = PILImages(image_transform(config.data, training=True))
        original = dataset.transform
        data = SimpleNamespace(train=SimpleNamespace(dataset=dataset))
        add_randaugment(data, config, RECIPES["control"])
        self.assertIs(dataset.transform, original)

    def test_one_epoch_protocol_and_summary_record_the_policy(self):
        base = ExperimentConfig(
            name="teacher", data=DataConfig(image_size=32, train_samples=16, val_samples=8,
                                            test_samples=8),
            student=ModelConfig("tiny_small"), distillation=DistillationConfig(method="supervised"),
            train=TrainConfig(epochs=1, batch_size=8, workers=0, threads=1, device="cpu",
                              precision="float32"),
            benchmark=BenchmarkConfig(warmup=0, iterations=1),
        )
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            for recipe_name in ("control", "randaugment"):
                with self.subTest(recipe=recipe_name):
                    config = replace(base, name=recipe_name, output_dir=directory)
                    result = train_teacher(config, RECIPES[recipe_name])
                    path = Path(directory) / recipe_name
                    protocol = json.loads((path / "teacher_protocol.json").read_text())
                    summary = json.loads((path / "summary.json").read_text())
                    expected = randaugment_policy("synthetic") if recipe_name == "randaugment" else None
                    self.assertEqual(protocol["augmentation_policy"], expected)
                    self.assertEqual(summary["augmentation_policy"], expected)
                    self.assertEqual(result["augmentation_policy"], expected)
                    self.assertTrue(summary["synthetic_smoke_only"])
                    self.assertEqual(len(json.loads((path / "history.json").read_text())), 1)


if __name__ == "__main__":
    unittest.main()
