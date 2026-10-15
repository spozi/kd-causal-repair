from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch.nn import functional as F

from kd.baseline_artifacts import STUDENT_FILES, write_manifest
from kd.checkpoints import fingerprint, save_checkpoint, tensor_state_fingerprint, write_json
from kd.data import SyntheticImages, build_data
from kd.engine import seed_everything
from kd.models import create_model
from kd.config import DataConfig, ModelConfig
from kd.teacher_baselines import (_report, _student_run, _target_classes, load_spec, plan_study,
                                 run_study, smoke_spec, study_configs)
from kd.teacher_repair_baseline import REPAIR_NAME, TeacherRepairOptions
from kd.teacher_training import (RECIPES, TeacherRecipe, add_randaugment, mix_batch,
                                 sam_step, teacher_loss, train_teacher, update_ema)
from kd.training_provenance import bind_training_data


class TeacherLossTests(unittest.TestCase):
    def test_control_and_smoothing_match_torch(self):
        logits = torch.tensor([[1.2, -0.2, 0.5], [0.1, 0.8, -0.5]], requires_grad=True)
        labels = torch.tensor([0, 2])
        targets = F.one_hot(labels, 3).float()
        for smoothing in (0.0, 0.1):
            actual = teacher_loss(logits, targets, TeacherRecipe("test", label_smoothing=smoothing), [2, 1, 1])
            expected = F.cross_entropy(logits, labels, label_smoothing=smoothing)
            torch.testing.assert_close(actual, expected)

    def test_balanced_softmax_uses_training_counts_and_raw_logits(self):
        logits = torch.tensor([[0.3, -0.4], [-0.2, 1.0]], requires_grad=True)
        original = logits.detach().clone()
        targets = F.one_hot(torch.tensor([0, 1]), 2).float()
        actual = teacher_loss(logits, targets, RECIPES["balanced_softmax"], [10, 1])
        expected = F.cross_entropy(logits + torch.tensor([10.0, 1.0]).log(), torch.tensor([0, 1]))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(logits, original)
        with self.assertRaisesRegex(ValueError, "positive training counts"):
            teacher_loss(logits, targets, RECIPES["balanced_softmax"], [10, 0])

    def test_mse_spaces_are_distinct_and_differentiable(self):
        logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
        target = torch.tensor([[1.0, 0.0]])
        raw = teacher_loss(logits, target, RECIPES["mse_logits"], [1, 1])
        probability = teacher_loss(logits, target, RECIPES["mse_probabilities"], [1, 1])
        self.assertAlmostEqual(float(raw.detach()), 0.5)
        self.assertAlmostEqual(float(probability.detach()), (1 - float(logits.softmax(1)[0, 0].detach())) ** 2, places=6)
        probability.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_mixup_target_is_a_probability_distribution(self):
        seed_everything(42)
        images = torch.arange(4.0).view(4, 1, 1, 1).expand(4, 3, 8, 8)
        labels = torch.arange(4)
        mixed, targets = mix_batch(images, labels, 4, RECIPES["mixup"])
        torch.testing.assert_close(targets.sum(1), torch.ones(4))
        expected_pixel = targets @ torch.arange(4.0)
        torch.testing.assert_close(mixed[:, 0, 0, 0], expected_pixel)

    def test_cutmix_labels_match_realized_clipped_patch(self):
        images = torch.stack([torch.zeros(3, 8, 8), torch.ones(3, 8, 8)])
        with patch("kd.teacher_training.np.random.beta", return_value=0.0), \
                patch("kd.teacher_training.np.random.randint", return_value=0), \
                patch("kd.teacher_training.torch.randperm", return_value=torch.tensor([1, 0])):
            mixed, targets = mix_batch(images, torch.tensor([0, 1]), 2, RECIPES["cutmix"])
        self.assertAlmostEqual(float(mixed[0].mean()), 0.25)
        self.assertAlmostEqual(float(targets[0, 1]), 0.25)
        self.assertAlmostEqual(float(targets[0, 0]), 0.75)
        torch.testing.assert_close(images[0], torch.zeros_like(images[0]))

    def test_ema_averages_parameters_but_copies_buffers(self):
        average, model = torch.nn.BatchNorm1d(2), torch.nn.BatchNorm1d(2)
        with torch.no_grad():
            average.weight.fill_(0)
            model.weight.fill_(2)
            model.running_mean.fill_(3)
            model.num_batches_tracked.fill_(7)
        update_ema(average, model, 0.5)
        torch.testing.assert_close(average.weight, torch.ones(2))
        torch.testing.assert_close(average.running_mean, torch.full((2,), 3.0))
        self.assertEqual(int(average.num_batches_tracked), 7)

    def test_sam_updates_parameters_but_bn_only_once(self):
        torch.set_num_threads(2)
        model = create_model("tiny_small", 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        images, targets = torch.rand(8, 3, 16, 16), F.one_hot(torch.arange(8) % 4, 4).float()
        initial = tensor_state_fingerprint(model.state_dict())
        loss = sam_step(model, images, targets, RECIPES["sam"], [2] * 4, optimizer, torch.device("cpu"), "float32")
        self.assertTrue(torch.isfinite(loss))
        self.assertNotEqual(initial, tensor_state_fingerprint(model.state_dict()))
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                self.assertEqual(int(module.num_batches_tracked), 1)

    def test_randaugment_does_not_modify_validation_transform(self):
        spec = smoke_spec()
        data = build_data(spec.teacher.data, spec.teacher.train, include_test=False)
        before = repr(data.val.dataset.transform)
        add_randaugment(data, spec.teacher, RECIPES["randaugment"])
        self.assertIn("RandAugment", repr(data.train.dataset.transform))
        self.assertEqual(before, repr(data.val.dataset.transform))
        images, labels = next(iter(data.train))
        self.assertEqual(images.shape, (16, 3, 16, 16))
        self.assertTrue(torch.isfinite(images).all())


class TeacherBaselineStudyTests(unittest.TestCase):
    def test_protocol_hashes_student_selection_and_repair_code(self):
        protocol = plan_study(smoke_spec(), "/private/tmp/teacher-baseline-hash-test")
        for name in ("metrics.py", "calibration_metrics.py", "neuron_surgery.py",
                     "neuron_surgery_study.py", "teacher_repair_baseline.py", "catalog/v1.json"):
            self.assertEqual(protocol["source_sha256"][name], fingerprint(Path("kd") / name))

    def test_missing_manifest_entries_are_rejected_before_test_access(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            spec = replace(smoke_spec(), recipes=("control",))
            run_study(spec, root)
            directory = Path(root) / "kd_control_teacher41_seed42"
            path = directory / "baseline_completion.json"
            recorded = json.loads(path.read_text())
            recorded.pop("summary.json")
            path.write_text(json.dumps(recorded))
            with patch("kd.teacher_baselines.build_data", side_effect=AssertionError("test access")):
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    run_study(spec, root, phase="evaluate")

    def test_completed_reuse_checks_teacher_and_student_summaries(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            spec = replace(smoke_spec(), recipes=("control",))
            run_study(spec, root)
            for name in ("teacher_control_seed41", "kd_control_teacher41_seed42"):
                path = Path(root) / name / "summary.json"
                original = path.read_text()
                path.write_text("{}")
                for phase in ("train", "evaluate"):
                    with self.subTest(name=name, phase=phase), self.assertRaisesRegex(ValueError, "artifact changed"):
                        run_study(spec, root, phase=phase)
                path.write_text(original)
            self.assertEqual(run_study(spec, root)["status"], "trained")

    def test_unmanifested_student_cannot_bless_wrong_teacher_digest(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            spec = replace(smoke_spec(), recipes=("control",))
            run_study(spec, root)
            _, students, _ = study_configs(spec, root)
            config = students[0][2]
            directory = Path(root) / config.name
            (directory / "baseline_completion.json").unlink()
            summary = json.loads((directory / "summary.json").read_text())
            summary["teacher"]["checkpoint_sha256"] = "incorrect-teacher-hash"
            (directory / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "teacher digest is incorrect"):
                _student_run(config)
            self.assertFalse((directory / "baseline_completion.json").exists())

    def test_unmanifested_student_export_must_match_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            spec = replace(smoke_spec(), recipes=("control",))
            run_study(spec, root)
            _, students, _ = study_configs(spec, root)
            config = students[0][2]
            directory = Path(root) / config.name
            (directory / "baseline_completion.json").unlink()
            path = directory / "student.pt"
            state = torch.load(path, weights_only=True)
            first = next(iter(state["student"].values()))
            first.add_(0.1)
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "export differs"):
                _student_run(config)

    def test_unmanifested_student_checks_checkpoint_and_recorded_data_before_sealing(self):
        for checkpoint_name in ("best.pt", "last.pt", "data.json"):
            with self.subTest(checkpoint=checkpoint_name), tempfile.TemporaryDirectory() as root, \
                    redirect_stdout(io.StringIO()):
                spec = replace(smoke_spec(), recipes=("control",))
                run_study(spec, root)
                _, students, _ = study_configs(spec, root)
                config = students[0][2]
                directory = Path(root) / config.name
                completion = directory / "baseline_completion.json"
                completion.unlink()
                path = directory / checkpoint_name
                summary = json.loads((directory / "summary.json").read_text())
                changed_data = {**summary["data"], "fixture_identity": "another-training-dataset"}
                if checkpoint_name == "data.json":
                    write_json(path, changed_data)
                else:
                    state = torch.load(path, weights_only=True)
                    state["training_data_identity"] = bind_training_data(changed_data)
                    save_checkpoint(path, state)
                if checkpoint_name == "best.pt":
                    # Keep the selected-file digest coherent to exercise the
                    # independent best-data guard rather than only its hash.
                    last_path = directory / "last.pt"
                    last = torch.load(last_path, weights_only=True)
                    last["selected_best_checkpoint_sha256"] = fingerprint(path)
                    save_checkpoint(last_path, last)
                before = {p.name: fingerprint(p) for p in directory.iterdir() if p.is_file()}
                with patch("kd.teacher_baselines.write_manifest") as seal:
                    with self.assertRaisesRegex(ValueError, "provenance changed"):
                        _student_run(config)
                seal.assert_not_called()
                self.assertFalse(completion.exists())
                self.assertEqual(before, {
                    p.name: fingerprint(p) for p in directory.iterdir() if p.is_file()})

    def test_unmanifested_student_rejects_matching_swapped_best_and_export(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            spec = replace(smoke_spec(), recipes=("control",))
            run_study(spec, root)
            _, students, _ = study_configs(spec, root)
            config = students[0][2]
            directory = Path(root) / config.name
            completion = directory / "baseline_completion.json"
            completion.unlink()
            best_path, export_path = directory / "best.pt", directory / "student.pt"
            best = torch.load(best_path, weights_only=True)
            exported = torch.load(export_path, weights_only=True)
            name = next(name for name, value in best["student"].items()
                        if value.is_floating_point())
            best["student"][name].add_(0.1)
            exported["student"][name] = best["student"][name].clone()
            save_checkpoint(best_path, best)
            save_checkpoint(export_path, exported)
            with patch("kd.teacher_baselines.write_manifest") as seal:
                with self.assertRaisesRegex(ValueError, "selected.*checkpoint|Selected.*checkpoint"):
                    _student_run(config)
            seal.assert_not_called()
            self.assertFalse(completion.exists())

    def test_unmanifested_student_requires_committed_selected_best_binding(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            spec = replace(smoke_spec(), recipes=("control",))
            run_study(spec, root)
            _, students, _ = study_configs(spec, root)
            config = students[0][2]
            directory = Path(root) / config.name
            completion = directory / "baseline_completion.json"
            completion.unlink()
            last_path = directory / "last.pt"
            last = torch.load(last_path, weights_only=True)
            last.pop("selected_best_checkpoint_sha256", None)
            save_checkpoint(last_path, last)
            with patch("kd.teacher_baselines.write_manifest") as seal:
                with self.assertRaisesRegex(ValueError, "selected.*checkpoint|Selected.*checkpoint"):
                    _student_run(config)
            seal.assert_not_called()
            self.assertFalse(completion.exists())

    def test_manifest_verified_legacy_student_is_readable_without_resealing(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            spec = replace(smoke_spec(), recipes=("control",))
            run_study(spec, root)
            _, students, _ = study_configs(spec, root)
            config = students[0][2]
            directory = Path(root) / config.name
            # A fixture for the prior completed schema, not migration of a real
            # experiment: its existing manifest still authenticates every byte.
            for checkpoint_name in ("best.pt", "last.pt"):
                checkpoint = directory / checkpoint_name
                state = torch.load(checkpoint, weights_only=True)
                state.pop("selected_best_checkpoint_sha256", None)
                state.pop("training_data_identity", None)
                save_checkpoint(checkpoint, state)
            write_manifest(directory, "baseline_completion.json", STUDENT_FILES)
            before = fingerprint(directory / "baseline_completion.json")
            with patch("kd.teacher_baselines.write_manifest") as seal:
                summary = _student_run(config, verify_only=True)
            seal.assert_not_called()
            self.assertEqual(Path(summary["checkpoint"]).resolve(), (directory / "student.pt").resolve())
            self.assertEqual(before, fingerprint(directory / "baseline_completion.json"))

    def test_selection_export_crash_can_be_recovered(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            spec = replace(smoke_spec(), recipes=("control",))
            run_study(spec, root)
            before = fingerprint(Path(root) / "selection.json")
            (Path(root) / "study_completion.json").unlink()
            with patch("kd.teacher_baselines.run_experiment", side_effect=AssertionError("retraining")):
                self.assertEqual(run_study(spec, root)["status"], "trained")
            self.assertEqual(before, fingerprint(Path(root) / "selection.json"))
            self.assertTrue((Path(root) / "study_completion.json").is_file())

    def test_generic_training_counts_select_lowest_frequency_half(self):
        self.assertEqual(_target_classes([20, 10, 3, 1], long_tail=True), (3, 2))
        self.assertEqual(_target_classes([10, 10, 10, 10]), (0, 1, 2, 3))

    def test_long_tail_targets_round_up_and_break_ties_by_label(self):
        self.assertEqual(_target_classes([100, 20, 20, 5, 5], long_tail=True), (3, 4, 1))
        # Tiny long-tail splits can realize equal counts without becoming balanced.
        self.assertEqual(_target_classes([1] * 5, long_tail=True), (0, 1, 2))

    def test_balanced_targets_include_naturally_unequal_classes(self):
        self.assertEqual(_target_classes([8, 6, 4, 2]), (0, 1, 2, 3))
        for counts in ([], [1, 0], [1, -1]):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, "positive"):
                _target_classes(counts)

    def test_generic_report_targets_follow_declared_and_legacy_profiles(self):
        unequal = [100, 20, 20, 5, 5, 2, 1]
        cases = (("balanced", 1, unequal, list(range(7))),
                 ("lt-if10", 1, unequal, [6, 5, 3, 4]),
                 ("lt-if100", 1, [1] * 7, [0, 1, 2, 3]),
                 ("balanced", 100, unequal, [6, 5, 3, 4]))
        metrics = {name: 0.5 for name in ("accuracy", "balanced_accuracy", "target_macro_recall",
                                         "nll", "brier_score", "ece_15_bins")}
        for profile, factor, counts, targets in cases:
            with self.subTest(profile=profile, factor=factor), tempfile.TemporaryDirectory() as root:
                base = smoke_spec()
                data = replace(base.teacher.data, source="dermamnist", num_classes=7, image_size=32,
                               dataset_profile=profile, imbalance_factor=factor)
                spec = replace(base, recipes=("control",), teacher=replace(base.teacher, data=data))
                teachers, students, supervised = study_configs(spec, root)
                control = teachers[0][1]
                (Path(root) / control.name).mkdir()
                write_json(Path(root) / control.name / "summary.json", {"class_counts": counts})
                with patch("kd.teacher_baselines._validate_checkpoint", return_value=metrics) as validate, \
                        patch("kd.teacher_baselines._seconds", return_value=0):
                    report = _report(spec, teachers, students, supervised, split="val")
                self.assertEqual(report["target_classes"], targets)
                self.assertTrue(all(list(call.kwargs["targets"]) == targets for call in validate.call_args_list))

    def test_imagefolder_report_uses_actual_training_counts(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            images = Path(root) / "images"
            for split in ("train", "val", "test"):
                for label, count in enumerate((8, 6, 4, 2) if split == "train" else (2, 2, 2, 2)):
                    directory = images / split / f"class{label}"
                    directory.mkdir(parents=True)
                    for i in range(count):
                        Image.new("RGB", (16, 16), (label * 50, 100, 100)).save(directory / f"{i}.png")
            base = smoke_spec()
            spec = replace(base, recipes=("control",), teacher=replace(base.teacher,
                           data=DataConfig(source="imagefolder", root=str(images), image_size=16)))
            run_study(spec, Path(root) / "study")
            report = json.loads((Path(root) / "study/validation.json").read_text())
            self.assertEqual(report["train_per_class"], [8, 6, 4, 2])
            self.assertEqual(report["target_classes"], [0, 1, 2, 3])
            self.assertTrue(all(row["target_classes"] == [0, 1, 2, 3]
                                for row in report["per_run"].values()))

    def test_invalid_recipes_seeds_and_interventions_fail(self):
        spec = smoke_spec()
        for candidate in (replace(spec, recipes=("mixup",)),
                          replace(spec, recipes=("control", "unknown")),
                          replace(spec, student_seeds=(42, 42)),
                          replace(spec, teacher_seeds=(True,)),
                          replace(spec, teacher_selection_metric="test_accuracy")):
            with self.assertRaises(ValueError):
                candidate.validate()
        with self.assertRaises(ValueError):
            TeacherRecipe("bad", mixup_alpha=0.2, cutmix_alpha=1.0).validate()

    def test_plan_counts_and_student_recipe_stays_fixed(self):
        spec = replace(smoke_spec(), recipes=("control", "cutmix"), student_seeds=(42, 43), teacher_seeds=(41, 141))
        plan = plan_study(spec, "/private/tmp/teacher-baseline-plan-only")
        self.assertEqual(plan["counts"], {"teachers": 4, "kd_students": 8, "supervised_students": 2, "total_fits": 14})
        _, students, _ = study_configs(spec, "/private/tmp/teacher-baseline-plan-only")
        self.assertTrue(all(config.data.augmentation == "basic" for _, _, config in students))
        self.assertTrue(all(config.distillation.method == "kd" for _, _, config in students))
        self.assertEqual({config.distillation.temperature for _, _, config in students}, {4.0})

    def test_dry_run_never_builds_data_or_writes(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch("kd.teacher_baselines.build_data", side_effect=AssertionError("data access")), \
                    patch("kd.teacher_training.build_data", side_effect=AssertionError("data access")):
                result = run_study(smoke_spec(), output, dry_run=True)
            self.assertTrue(result["dry_run"])
            self.assertFalse(output.exists())

    def test_teacher_resume_restores_ema_and_completed_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            spec = smoke_spec()
            config = replace(spec.teacher, output_dir=root, name="ema",
                             train=replace(spec.teacher.train, epochs=2))
            import kd.teacher_training as training
            original = training._validation
            calls = 0

            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected interruption")
                return original(*args, **kwargs)

            with patch.object(training, "_validation", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                    train_teacher(config, RECIPES["ema"])
            resumed = train_teacher(config, RECIPES["ema"])
            clean = train_teacher(replace(config, name="clean"), RECIPES["ema"])
            resumed_state = torch.load(resumed["checkpoint"], weights_only=True)
            clean_state = torch.load(clean["checkpoint"], weights_only=True)
            self.assertEqual(tensor_state_fingerprint(resumed_state["student"]),
                             tensor_state_fingerprint(clean_state["student"]))
            self.assertEqual(train_teacher(config, RECIPES["ema"]), resumed)
            Path(resumed["checkpoint"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                train_teacher(config, RECIPES["ema"])

    def test_matched_end_to_end_validation_gate_and_test_evaluation(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            spec = replace(smoke_spec(), recipes=("control", "cutmix"))
            with self.assertRaisesRegex(ValueError, "frozen validation selection"):
                run_study(spec, root, phase="evaluate")
            import kd.teacher_training as training
            import kd.teacher_baselines as studies
            original = build_data

            def no_test(*args, **kwargs):
                self.assertFalse(kwargs.get("include_test", True))
                return original(*args, **kwargs)

            with patch.object(training, "build_data", side_effect=no_test), \
                    patch.object(studies, "build_data", side_effect=no_test):
                result = run_study(spec, root)
            self.assertEqual(result["status"], "trained")
            selection = json.loads((Path(root) / "selection.json").read_text())
            self.assertEqual(selection["selected_on"], "validation only")
            self.assertEqual(len(selection["checkpoints"]), 5)
            selection_hash = fingerprint(Path(root) / "selection.json")
            evaluated = run_study(spec, root, phase="evaluate")
            self.assertEqual(evaluated["status"], "evaluated")
            self.assertEqual(fingerprint(Path(root) / "selection.json"), selection_hash)
            report = json.loads((Path(root) / "comparison.json").read_text())
            control = next(row for row in report["rows"] if row["recipe"] == "control")
            self.assertEqual(control["paired_student_delta_vs_control"]["accuracy"]["mean"], 0)
            self.assertEqual(run_study(spec, root)["status"], "trained")
            (Path(root) / "validation.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact changed.*validation.json"):
                run_study(spec, root, phase="evaluate")


class _TinyCifar:
    """Offline archive adapter; exercises the unchanged catalog/repair pipeline."""
    def __init__(self, root, train, download=False, transform=None):
        self.targets = [i % 10 for i in range(60 if train else 20)]
        self.classes = [f"class_{i}" for i in range(10)]
        self.transform = transform
        self.images = SyntheticImages(len(self.targets), 10, 32, 2026 if train else 3026,
                                      lambda image: image)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image, label = self.images[index]
        return self.transform(image), label


class TeacherRepairIntegrationTests(unittest.TestCase):
    def spec(self, root, *, teacher_model="cifar_teacher", student_model="cifar_student"):
        base = smoke_spec()
        return replace(base, teacher=replace(base.teacher,
                       data=DataConfig(source="cifar10", root=root, num_classes=10, image_size=32),
                       student=ModelConfig(teacher_model)),
                       student_model=student_model, recipes=("control", REPAIR_NAME),
                       repair=TeacherRepairOptions(channel_budgets=(1,), learning_rates=(0.001,),
                           repair_epochs=1, repair_samples_per_epoch=8, repair_batch_size=8,
                           companion_count=1, allow_sparse_class_fallback=True))

    def fake_repair(self, selected):
        def compute(baseline, output, device, seeds, *, spec):
            self.assertFalse(spec.downstream_kd)
            self.assertEqual(spec.teacher_run, "teacher_control_seed41")
            self.assertEqual(spec.target_class_policy, "all")
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "protocol.json", {"test_fixture": True, "spec": spec.to_dict()})
            result = {"status": "repair_only_complete" if selected else "no_repair_selected",
                      "reason": "controlled integration fixture"}
            write_json(output / "comparison.json", result)
            (output / "report.md").write_text("Fixture repair selection\n")
            names = ["comparison.json", "report.md"]
            if selected:
                state = torch.load(Path(baseline) / spec.teacher_run / "student.pt", weights_only=True)
                state["student"]["stages.stage3.3.weight"][0].add_(0.001)
                state["repair"] = {"base_checkpoint_sha256": fingerprint(
                    Path(baseline) / spec.teacher_run / "student.pt")}
                torch.save(state, output / "teacher_repaired.pt")
                names.append("teacher_repaired.pt")
            write_json(output / "report_manifest.json", {
                "protocol_sha256": fingerprint(output / "protocol.json"),
                "files": {name: fingerprint(output / name) for name in names}})
            return result
        return compute

    def test_presets_plan_the_original_repair_condition_without_dataset_access(self):
        for filename in ("cifar10_lt100.json", "cifar100_balanced.json"):
            spec = load_spec(Path("configs/teacher_baselines") / filename)
            self.assertEqual(spec.teacher.train.precision, "bfloat16")
            plan = plan_study(spec, "/private/tmp/teacher-repair-plan-only")
            self.assertIn(REPAIR_NAME, spec.recipes)
            self.assertFalse(plan["repair_protocols"][0]["downstream_kd"])
            self.assertEqual(plan["counts"]["repair_candidate_fits_max"], 6)
            self.assertEqual(plan["counts"]["total_fits_max"], 56 if filename.startswith("cifar10_") else 52)

    def test_unsupported_repair_architecture_and_dataset_fail_during_planning(self):
        with tempfile.TemporaryDirectory() as root:
            spec = self.spec(root)
            with self.assertRaisesRegex(ValueError, "CifarCNN or torchvision ResNet"):
                replace(spec, teacher=replace(spec.teacher, student=ModelConfig("tiny_medium"))).validate()
            with self.assertRaisesRegex(ValueError, "supported 32x32"):
                base = smoke_spec()
                replace(base, teacher=replace(base.teacher, student=ModelConfig("cifar_teacher")),
                        recipes=("control", REPAIR_NAME), repair=TeacherRepairOptions()).validate()

    def test_neutral_teacher_aliases_keep_explicit_repair_stages_and_model_identity(self):
        with tempfile.TemporaryDirectory() as root:
            for name, stages in (
                ("convnet6_w32", ["stage2", "stage3"]),
                ("convnet6_w16", ["stage2", "stage3"]),
                ("resnet18_smallstem", ["stage3", "stage4"]),
                ("resnet34_smallstem", ["stage3", "stage4"]),
                ("resnet50_smallstem", ["stage3", "stage4"]),
            ):
                with self.subTest(model=name):
                    spec = self.spec(root, teacher_model=name, student_model="convnet6_w16")
                    plan = plan_study(spec, Path(root) / name)
                    repair = plan["repair_protocols"][0]
                    self.assertEqual(repair["stages"], stages)
                    self.assertEqual(repair["teacher_model"], name)
                    self.assertEqual(repair["student_model"], "convnet6_w16")
                    self.assertEqual(plan["student_runs"][0]["config"]["teacher"]["name"], name)

    def test_neutral_alias_runs_actual_repair_and_frozen_teacher_kd(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()), \
                patch("kd.data.datasets.CIFAR10", _TinyCifar):
            spec = self.spec(root, teacher_model="convnet6_w32", student_model="convnet6_w16")
            output = Path(root) / "neutral-study"
            self.assertEqual(run_study(spec, output)["status"], "trained")
            teacher = torch.load(output / "teacher_control_seed41/student.pt", weights_only=True)
            student = torch.load(output / "kd_control_teacher41_seed42/student.pt", weights_only=True)
            self.assertEqual(teacher["model_name"], "convnet6_w32")
            self.assertEqual(student["model_name"], "convnet6_w16")
            self.assertTrue((output / "teacher_repair_seed41/repair/localization.json").is_file())
            self.assertEqual(run_study(spec, output, phase="evaluate")["status"], "evaluated")

    def test_rejected_repair_reuses_control_and_has_exactly_zero_effect(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()), \
                patch("kd.data.datasets.CIFAR10", _TinyCifar), \
                patch("kd.teacher_repair_baseline.run_neuron_surgery_study", side_effect=self.fake_repair(False)):
            spec = self.spec(root)
            output = Path(root) / "study"
            result = run_study(spec, output)
            report = json.loads((output / "validation.json").read_text())
            repair = next(row for row in report["rows"] if row["recipe"] == REPAIR_NAME)
            self.assertEqual(repair["repair_outcomes"]["41"]["status"], "no_change")
            self.assertEqual(repair["paired_student_delta_vs_control"]["accuracy"]["mean"], 0)
            self.assertEqual(repair["student_seconds"]["mean"], 0)
            self.assertEqual(fingerprint(output / "teacher_repair_seed41/student.pt"),
                             fingerprint(output / "teacher_control_seed41/student.pt"))
            alias = output / "kd_teacher_repair_teacher41_seed42"
            self.assertTrue((alias / "baseline_alias.json").is_file())
            last = torch.load(alias / "last.pt", weights_only=True)
            self.assertEqual(last["selected_best_checkpoint_sha256"], fingerprint(alias / "best.pt"))
            with patch("kd.teacher_baselines.run_experiment", side_effect=AssertionError("retraining")):
                self.assertEqual(run_study(spec, output), result)
                self.assertEqual(run_study(spec, output, phase="evaluate")["status"], "evaluated")

    def test_selected_repair_distills_same_initialized_student_and_keeps_test_gate(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()), \
                patch("kd.data.datasets.CIFAR10", _TinyCifar), \
                patch("kd.teacher_repair_baseline.run_neuron_surgery_study", side_effect=self.fake_repair(True)):
            spec = self.spec(root)
            output = Path(root) / "study"
            original_build = build_data

            def forbid_test(*args, **kwargs):
                self.assertFalse(kwargs.get("include_test", True))
                return original_build(*args, **kwargs)

            with patch("kd.teacher_baselines.build_data", side_effect=forbid_test), \
                    patch("kd.teacher_training.build_data", side_effect=forbid_test), \
                    patch("kd.engine.build_data", side_effect=forbid_test):
                run_study(spec, output)
            control = json.loads((output / "kd_control_teacher41_seed42/summary.json").read_text())
            candidate = json.loads((output / "kd_teacher_repair_teacher41_seed42/summary.json").read_text())
            self.assertEqual(control["initial_student_sha256"], candidate["initial_student_sha256"])
            self.assertNotEqual(control["teacher"]["checkpoint_sha256"], candidate["teacher"]["checkpoint_sha256"])
            self.assertFalse((output / "kd_teacher_repair_teacher41_seed42/baseline_alias.json").exists())
            selected_hash = fingerprint(output / "selection.json")
            self.assertEqual(run_study(spec, output, phase="evaluate")["status"], "evaluated")
            self.assertEqual(selected_hash, fingerprint(output / "selection.json"))
            artifact = output / "teacher_repair_seed41/repair/teacher_repaired.pt"
            state = torch.load(artifact, weights_only=True)
            state["student"]["stages.stage3.3.weight"][0].add_(0.01)
            torch.save(state, artifact)
            with patch("kd.teacher_baselines.build_data", side_effect=AssertionError("test access")):
                with self.assertRaisesRegex(ValueError, "Repair export differs|Repair artifact changed"):
                    run_study(spec, output, phase="evaluate")

    def test_actual_repair_runner_operates_validation_only_without_core_edits(self):
        source = Path("kd/neuron_surgery_study.py")
        before = fingerprint(source)
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()), \
                patch("kd.data.datasets.CIFAR10", _TinyCifar):
            spec = self.spec(root)
            output = Path(root) / "study"
            original_build = build_data

            def forbid_test(*args, **kwargs):
                self.assertFalse(kwargs.get("include_test", True))
                return original_build(*args, **kwargs)

            with patch("kd.neuron_surgery_study.build_data", side_effect=forbid_test):
                run_study(spec, output)
            decision = json.loads((output / "teacher_repair_seed41/repair_decision.json").read_text())
            self.assertIn(decision["status"], ("selected", "no_change"))
            self.assertTrue((output / "teacher_repair_seed41/repair/localization.json").is_file())
            self.assertFalse((output / "teacher_repair_seed41/repair/teacher_evaluation").exists())
        self.assertEqual(before, fingerprint(source))


if __name__ == "__main__":
    unittest.main()
