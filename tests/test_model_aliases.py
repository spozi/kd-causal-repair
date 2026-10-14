from pathlib import Path
import tempfile
import unittest

import torch

from kd.checkpoints import load_model_checkpoint, metadata, save_checkpoint
from kd.models import MODEL_NAMES, canonical_model_name, create_model


ALIASES = {
    "convnet6_w32": "cifar_teacher",
    "convnet6_w16": "cifar_student",
    "resnet18_smallstem": "cifar_resnet18",
    "resnet34_smallstem": "cifar_resnet34",
    "resnet50_smallstem": "cifar_resnet50",
}

LEGACY_NAMES = (
    "tiny_small", "tiny_medium", "tiny_large", "resnet18", "resnet34", "resnet50",
    "cifar_student", "cifar_teacher", "cifar_resnet18", "cifar_resnet34", "cifar_resnet50",
    "resnet8", "resnet20",
)


class ModelAliasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_legacy_names_are_preserved_and_aliases_have_explicit_routing(self):
        self.assertEqual(MODEL_NAMES[:len(LEGACY_NAMES)], LEGACY_NAMES)
        self.assertEqual(set(MODEL_NAMES), set(LEGACY_NAMES) | set(ALIASES))
        self.assertEqual(len(MODEL_NAMES), len(set(MODEL_NAMES)))
        for name in LEGACY_NAMES:
            with self.subTest(name=name):
                self.assertEqual(canonical_model_name(name), name)
        for alias, legacy in ALIASES.items():
            with self.subTest(alias=alias):
                self.assertEqual(canonical_model_name(alias), legacy)

    def test_alias_and_legacy_have_identical_initialized_state(self):
        for alias, legacy in ALIASES.items():
            with self.subTest(alias=alias), torch.random.fork_rng(devices=[]):
                torch.manual_seed(2026)
                original = create_model(legacy, 17)
                torch.manual_seed(2026)
                neutral = create_model(alias, 17)
                self.assertIs(type(neutral), type(original))
                self.assertEqual(neutral.feature_channels, original.feature_channels)
                original_state, neutral_state = original.state_dict(), neutral.state_dict()
                self.assertEqual(tuple(neutral_state), tuple(original_state))
                for name, expected in original_state.items():
                    self.assertTrue(torch.equal(neutral_state[name], expected), name)

    def test_aliases_keep_feature_contract_and_dataset_sized_heads(self):
        widths = {
            "convnet6_w32": (32, 64, 128),
            "convnet6_w16": (16, 32, 64),
            "resnet18_smallstem": (64, 128, 256, 512),
            "resnet34_smallstem": (64, 128, 256, 512),
            "resnet50_smallstem": (256, 512, 1024, 2048),
        }
        with torch.random.fork_rng(devices=[]):
            images = torch.randn(1, 3, 32, 32)
            for alias in ALIASES:
                for classes in (2, 10, 100):
                    with self.subTest(alias=alias, classes=classes):
                        model = create_model(alias, classes).eval()
                        channels = {f"stage{i}": width
                                    for i, width in enumerate(widths[alias], start=1)}
                        self.assertEqual(model.feature_channels, channels)
                        with torch.no_grad():
                            result = model(images, return_features=True)
                            logits_only = model(images)
                        self.assertEqual(result.logits.shape, (1, classes))
                        self.assertEqual(tuple(result.features), tuple(channels))
                        self.assertEqual(logits_only.features, {})
                        self.assertTrue(torch.equal(logits_only.logits, result.logits))
                        for stage, width in channels.items():
                            self.assertEqual(result.features[stage].shape[1], width)
                        if alias.startswith("resnet"):
                            self.assertEqual(model.backbone.conv1.kernel_size, (3, 3))
                            self.assertEqual(model.backbone.conv1.stride, (1, 1))
                            self.assertIsInstance(model.backbone.maxpool, torch.nn.Identity)

    def test_unknown_names_have_the_same_validation_error(self):
        for name in ("unknown", "convnet6_w64", "resnet18_small_stem"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as canonical_error:
                    canonical_model_name(name)
                with self.assertRaises(ValueError) as factory_error:
                    create_model(name, 10)
                self.assertEqual(str(factory_error.exception), str(canonical_error.exception))
                self.assertIn(repr(name), str(factory_error.exception))
                self.assertIn("choose from", str(factory_error.exception))

    def test_trained_alias_and_legacy_checkpoints_roundtrip_without_renaming(self):
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng(devices=[]):
            path = Path(directory) / "model.pt"
            images = torch.randn(2, 3, 16, 16)
            labels = torch.tensor([0, 2])
            classes = ["first", "second", "third"]
            for name in (*ALIASES, *ALIASES.values()):
                with self.subTest(name=name):
                    model = create_model(name, len(classes))
                    initial_parameter = next(model.parameters()).detach().clone()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    loss = torch.nn.functional.cross_entropy(model(images).logits, labels)
                    loss.backward()
                    optimizer.step()
                    self.assertFalse(torch.equal(next(model.parameters()), initial_parameter))
                    model.eval()
                    expected = metadata(name, classes, 16)
                    save_checkpoint(path, {**expected, "epoch": 0, "kind": "inference",
                                           "student": model.state_dict()})
                    restored = create_model(name, len(classes)).eval()
                    checkpoint = load_model_checkpoint(restored, path, expected)
                    self.assertEqual(checkpoint["model_name"], name)
                    for parameter, tensor in model.state_dict().items():
                        self.assertTrue(torch.equal(restored.state_dict()[parameter], tensor), parameter)
                    with torch.no_grad():
                        self.assertTrue(torch.equal(restored(images).logits, model(images).logits))
                    # Alias routing must not relax the historical metadata identity check.
                    other_name = ALIASES.get(name)
                    if other_name is not None:
                        with self.assertRaisesRegex(ValueError, "model_name does not match"):
                            load_model_checkpoint(restored, path, metadata(other_name, classes, 16))


if __name__ == "__main__":
    unittest.main()
