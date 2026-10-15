"""Resume guards bind recorded data identity before adopting saved training state."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from kd.checkpoints import fingerprint, save_checkpoint, tensor_state_fingerprint
from kd.config import (BenchmarkConfig, DataConfig, DistillationConfig,
                       ExperimentConfig, ModelConfig, TrainConfig)
from kd.engine import Trainer, run_experiment
from kd.teacher_training import TeacherRecipe, train_teacher
from kd.training_provenance import (bind_training_data, training_provenance,
                                    validate_training_data)


class TrainingIdentityTests(unittest.TestCase):
    def setUp(self):
        self.provenance = {"source": "cifar10", "split_sizes": {"train": 16, "val": 8},
                           "train_index_sha256": "train", "val_index_sha256": "val",
                           "artifact_sha256": {"archive": "original"}}

    def test_evaluation_only_loader_sizes_are_ignored_without_mutating_input(self):
        expanded = deepcopy(self.provenance)
        expanded["split_sizes"].update(test=20, confirmation=4)
        self.assertEqual(training_provenance(expanded), self.provenance)
        self.assertEqual(expanded["split_sizes"]["test"], 20)
        validate_training_data({"training_data_identity": bind_training_data(self.provenance)}, expanded)

    def test_each_recorded_identity_change_is_rejected(self):
        state = {"training_data_identity": bind_training_data(self.provenance)}
        for key, value in (("train_index_sha256", "different"), ("val_index_sha256", "different"),
                           ("artifact_sha256", {"archive": "different"}),
                           ("split_sizes", {"train": 16, "val": 9})):
            with self.subTest(key=key):
                changed = deepcopy(self.provenance)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, "provenance changed"):
                    validate_training_data(state, changed)

    def test_missing_binding_and_corrupt_binding_fail_closed(self):
        for state in ({}, {"data": self.provenance}, {"training_data_identity": None}):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "checkpoint-bound"):
                validate_training_data(state, self.provenance)
        state = {"training_data_identity": bind_training_data(self.provenance)}
        state["training_data_identity"]["provenance"]["val_index_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_training_data(state, self.provenance)

    def test_missing_training_identity_is_not_silently_bound(self):
        for value in ({}, {"source": "synthetic", "split_sizes": {"test": 4}},
                      {"source": "synthetic", "split_sizes": {"train": True, "val": 4}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bind_training_data(value)


class ResumeDataGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        for split in ("train", "val"):
            for label, color in (("left", "red"), ("right", "blue")):
                directory = self.images / split / label
                directory.mkdir(parents=True)
                for index in range(2):
                    Image.new("RGB", (16, 16), color).save(directory / f"{index}.png")
        self.config = ExperimentConfig(
            name="interrupted", output_dir=str(self.root / "runs"),
            data=DataConfig(source="imagefolder", root=str(self.images), num_classes=2, image_size=16),
            student=ModelConfig("tiny_small"), distillation=DistillationConfig(method="supervised"),
            train=TrainConfig(epochs=2, batch_size=4, threads=1, device="cpu", workers=0),
            benchmark=BenchmarkConfig(warmup=0, iterations=1))
        self.output = io.StringIO()

    @property
    def directory(self):
        return Path(self.config.output_dir) / self.config.name

    def snapshot(self):
        return {path.name: (fingerprint(path), path.stat().st_mtime_ns)
                for path in self.directory.iterdir() if path.is_file()}

    def interrupt_engine(self):
        original = Trainer.train_epoch

        def interrupt(trainer, loader, epoch):
            if epoch == 1:
                raise InterruptedError("epoch-boundary interruption")
            return original(trainer, loader, epoch)

        with patch.object(Trainer, "train_epoch", interrupt), redirect_stdout(self.output):
            with self.assertRaises(InterruptedError):
                run_experiment(self.config)

    def interrupt_teacher(self, recipe):
        import kd.teacher_training as training
        original = training._validation
        calls = 0

        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise InterruptedError("epoch-boundary interruption")
            return original(*args, **kwargs)

        with patch.object(training, "_validation", interrupt), redirect_stdout(self.output):
            with self.assertRaises(InterruptedError):
                train_teacher(self.config, recipe)

    def test_engine_changed_actual_training_membership_rejected_before_adoption_or_writes(self):
        self.interrupt_engine()
        original_artifacts = self.snapshot()
        Image.new("RGB", (16, 16), "red").save(self.images / "train" / "left" / "added.png")
        with patch("torch.nn.Module.load_state_dict") as adopt, \
                patch("kd.engine.Trainer.train_epoch") as fit, \
                patch("kd.engine.write_json") as write, redirect_stdout(self.output):
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                run_experiment(self.config, resume=str(self.directory / "last.pt"))
        adopt.assert_not_called()
        fit.assert_not_called()
        write.assert_not_called()
        self.assertEqual(original_artifacts, self.snapshot())

    def test_teacher_changed_actual_validation_membership_rejected_before_adoption_or_writes(self):
        recipe = TeacherRecipe("ema", ema_decay=0.9)
        self.interrupt_teacher(recipe)
        original_artifacts = self.snapshot()
        Image.new("RGB", (16, 16), "blue").save(self.images / "val" / "right" / "added.png")
        with patch("torch.nn.Module.load_state_dict") as adopt, \
                patch("kd.teacher_training.teacher_loss") as fit, \
                patch("kd.teacher_training.write_json") as write, redirect_stdout(self.output):
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                train_teacher(self.config, recipe)
        adopt.assert_not_called()
        fit.assert_not_called()
        write.assert_not_called()
        self.assertEqual(original_artifacts, self.snapshot())

    def test_unchanged_engine_resume_matches_clean_final_state(self):
        self.interrupt_engine()
        with redirect_stdout(self.output):
            run_experiment(self.config, resume=str(self.directory / "last.pt"))
            clean = run_experiment(replace(self.config, name="clean"))
        resumed = torch.load(self.directory / "last.pt", weights_only=True)
        expected = torch.load(Path(clean["checkpoint"]).with_name("last.pt"), weights_only=True)
        self.assertEqual(tensor_state_fingerprint(resumed["student"]),
                         tensor_state_fingerprint(expected["student"]))
        self.assertEqual(resumed["training_data_identity"], expected["training_data_identity"])

    def test_unchanged_teacher_resume_matches_clean_model_ema_and_rng(self):
        recipe = TeacherRecipe("ema", ema_decay=0.9)
        self.interrupt_teacher(recipe)
        with redirect_stdout(self.output):
            train_teacher(self.config, recipe)
            clean = train_teacher(replace(self.config, name="clean"), recipe)
        resumed = torch.load(self.directory / "last.pt", weights_only=True)
        expected = torch.load(Path(clean["checkpoint"]).with_name("last.pt"), weights_only=True)
        for section in ("student", "ema"):
            self.assertEqual(tensor_state_fingerprint(resumed[section]),
                             tensor_state_fingerprint(expected[section]))
        torch.testing.assert_close(resumed["rng"]["torch"], expected["rng"]["torch"], rtol=0, atol=0)
        self.assertEqual(resumed["scheduler"], expected["scheduler"])

    def test_legacy_engine_checkpoint_rejected_even_with_adjacent_data_json(self):
        self.interrupt_engine()
        path = self.directory / "last.pt"
        state = torch.load(path, weights_only=True)
        state.pop("training_data_identity")
        save_checkpoint(path, state)
        self.assertTrue((self.directory / "data.json").exists())
        original_artifacts = self.snapshot()
        with redirect_stdout(self.output), self.assertRaisesRegex(ValueError, "unbound adjacent data.json"):
            run_experiment(self.config, resume=str(path))
        self.assertEqual(original_artifacts, self.snapshot())

    def test_legacy_teacher_checkpoint_rejected_without_bound_identity(self):
        recipe = TeacherRecipe("ce")
        self.interrupt_teacher(recipe)
        path = self.directory / "last.pt"
        state = torch.load(path, weights_only=True)
        state.pop("training_data_identity")
        save_checkpoint(path, state)
        original_artifacts = self.snapshot()
        with redirect_stdout(self.output), self.assertRaisesRegex(ValueError, "checkpoint-bound"):
            train_teacher(self.config, recipe)
        self.assertEqual(original_artifacts, self.snapshot())

    def change_selected_artifact(self, mutation, *, teacher=False):
        best_path, last_path = self.directory / "best.pt", self.directory / "last.pt"
        best = torch.load(best_path, weights_only=True)
        last = torch.load(last_path, weights_only=True)
        if mutation == "missing_best":
            best_path.unlink()
            return
        if mutation == "missing_anchor":
            last_path.unlink()
            return
        if mutation == "missing_selection_binding":
            last.pop("selected_best_checkpoint_sha256")
            save_checkpoint(last_path, last)
            return
        if mutation == "corrupt_selection_binding":
            last["selected_best_checkpoint_sha256"] = "not-a-sha256"
            save_checkpoint(last_path, last)
            return
        if mutation == "data":
            provenance = deepcopy(best["training_data_identity"]["provenance"])
            provenance["foreign_fixture"] = True
            best["training_data_identity"] = bind_training_data(provenance)
        elif mutation == "missing_data_binding":
            best.pop("training_data_identity")
        elif mutation == "weights":
            name = next(name for name, tensor in best["student"].items() if tensor.is_floating_point())
            best["student"][name] = best["student"][name] + 1
        elif mutation == "config":
            config = best["identity"]["config"] if teacher else best["config"]
            config["train"]["learning_rate"] += 0.001
        elif mutation == "teacher":
            best["teacher_sha256"] = "foreign-teacher"
        elif mutation == "surgery":
            best["surgery_plan_sha256"] = "foreign-surgery"
        elif mutation == "initialization":
            best["initial_sha256" if teacher else "initial_student_sha256"] = "0" * 64
        elif mutation == "recipe":
            best["teacher_recipe"]["ema_decay"] = 0.5
        else:
            raise AssertionError(f"Unknown fixture mutation: {mutation}")
        save_checkpoint(best_path, best)
        if mutation in {"config", "teacher", "surgery", "initialization", "recipe"}:
            # Exercise semantic checks independently of the physical hash guard.
            last["selected_best_checkpoint_sha256"] = fingerprint(best_path)
            save_checkpoint(last_path, last)

    def test_engine_selected_best_guards_run_before_adoption_or_output_writes(self):
        cases = {"data": "provenance changed", "missing_data_binding": "checkpoint-bound",
                 "weights": "differs from committed last.pt", "config": "configuration differs",
                 "teacher": "Teacher checkpoint changed", "surgery": "Surgery plan changed",
                 "initialization": "initialization", "missing_best": "missing",
                 "missing_anchor": "committed last.pt", "missing_selection_binding": "committed last.pt",
                 "corrupt_selection_binding": "committed last.pt"}
        for mutation, message in cases.items():
            with self.subTest(mutation=mutation):
                self.config = replace(self.config, name=f"engine_selected_{mutation}")
                self.interrupt_engine()
                self.change_selected_artifact(mutation)
                artifacts = self.snapshot()
                with patch("torch.nn.Module.load_state_dict") as adopt, \
                        patch("kd.engine.Trainer.train_epoch") as fit, \
                        patch("kd.engine.write_json") as write, redirect_stdout(self.output):
                    with self.assertRaisesRegex(ValueError, message):
                        run_experiment(self.config, resume=str(self.directory / "last.pt"))
                adopt.assert_not_called()
                fit.assert_not_called()
                write.assert_not_called()
                self.assertEqual(artifacts, self.snapshot())

    def test_teacher_selected_best_guards_run_before_adoption_or_output_writes(self):
        recipe = TeacherRecipe("ema", ema_decay=0.9)
        cases = {"data": "provenance changed", "missing_data_binding": "checkpoint-bound",
                 "weights": "differs from committed last.pt", "config": "recipe, protocol",
                 "recipe": "recipe changed", "initialization": "initialization",
                 "missing_best": "missing", "missing_anchor": "no committed last.pt",
                 "missing_selection_binding": "committed last.pt",
                 "corrupt_selection_binding": "committed last.pt"}
        for mutation, message in cases.items():
            with self.subTest(mutation=mutation):
                self.config = replace(self.config, name=f"teacher_selected_{mutation}")
                self.interrupt_teacher(recipe)
                self.change_selected_artifact(mutation, teacher=True)
                artifacts = self.snapshot()
                with patch("torch.nn.Module.load_state_dict") as adopt, \
                        patch("kd.teacher_training.teacher_loss") as fit, \
                        patch("kd.teacher_training.write_json") as write, redirect_stdout(self.output):
                    with self.assertRaisesRegex(ValueError, message):
                        train_teacher(self.config, recipe)
                adopt.assert_not_called()
                fit.assert_not_called()
                write.assert_not_called()
                self.assertEqual(artifacts, self.snapshot())

    def test_engine_resume_from_best_uses_the_committed_last_anchor(self):
        self.interrupt_engine()
        best_path = self.directory / "best.pt"
        with redirect_stdout(self.output):
            run_experiment(self.config, resume=str(best_path))
            clean = run_experiment(replace(self.config, name="clean"))
        resumed = torch.load(self.directory / "last.pt", weights_only=True)
        expected = torch.load(Path(clean["checkpoint"]).with_name("last.pt"), weights_only=True)
        self.assertEqual(tensor_state_fingerprint(resumed["student"]),
                         tensor_state_fingerprint(expected["student"]))
        self.assertEqual(resumed["selected_best_checkpoint_sha256"], fingerprint(best_path))
        self.change_selected_artifact("weights")
        artifacts = self.snapshot()
        with patch("torch.nn.Module.load_state_dict") as adopt, \
                patch("kd.engine.write_json") as write, redirect_stdout(self.output):
            with self.assertRaisesRegex(ValueError, "differs from committed last.pt"):
                run_experiment(self.config, resume=str(best_path))
        adopt.assert_not_called()
        write.assert_not_called()
        self.assertEqual(artifacts, self.snapshot())

    def test_save_best_before_last_interruption_fails_closed_without_rebinding(self):
        for teacher in (False, True):
            with self.subTest(teacher=teacher):
                self.config = replace(self.config, name=f"between_commits_{teacher}")
                recipe = TeacherRecipe("control")
                if teacher:
                    self.interrupt_teacher(recipe)
                else:
                    self.interrupt_engine()
                best_path = self.directory / "best.pt"
                best = torch.load(best_path, weights_only=True)
                best["epoch"] += 1
                save_checkpoint(best_path, best)
                artifacts = self.snapshot()
                with patch("torch.nn.Module.load_state_dict") as adopt, redirect_stdout(self.output):
                    with self.assertRaisesRegex(ValueError, "interrupted between commits"):
                        if teacher:
                            train_teacher(self.config, recipe)
                        else:
                            run_experiment(self.config, resume=str(self.directory / "last.pt"))
                adopt.assert_not_called()
                self.assertEqual(artifacts, self.snapshot())

    def test_final_export_rechecks_the_committed_selected_artifact(self):
        from kd.checkpoints import write_json
        for teacher in (False, True):
            with self.subTest(teacher=teacher):
                self.config = replace(self.config, name=f"late_change_{teacher}",
                                      train=replace(self.config.train, epochs=1))
                module = "kd.teacher_training" if teacher else "kd.engine"

                def alter_after_epoch(path, value):
                    write_json(path, value)
                    if path.name == "history.json":
                        self.change_selected_artifact("weights", teacher=teacher)

                with patch(f"{module}.write_json", side_effect=alter_after_epoch), redirect_stdout(self.output):
                    with self.assertRaisesRegex(ValueError, "differs from committed last.pt"):
                        if teacher:
                            train_teacher(self.config, TeacherRecipe("control"))
                        else:
                            run_experiment(self.config)
                self.assertFalse((self.directory / "student.pt").exists())
                self.assertFalse((self.directory / "summary.json").exists())
                self.assertFalse((self.directory / "completion.json").exists())


if __name__ == "__main__":
    unittest.main()
