"""Compact-protocol regression checks; transfer defaults are tested separately."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kd.neuron_surgery_matrix import DATASETS, PROFILES
from kd.teacher_baselines import load_spec, main as study_main, plan_study, run_study, study_configs
from kd.teacher_baseline_matrix import (DEFAULT_EPOCHS, PAIRINGS, main, materialize_matrix_plan,
                                        matrix_jobs, plan_matrix)
from kd.matrix_binding import (MATRIX_SOURCE_FILES, STUDY_SOURCE_FILES, matrix_payload_digest)
from kd.teacher_training import RECIPES


# Keep the original preset's regression contracts explicit, independent of the
# new pretrained default. These wrappers still exercise the production planner.
COMPACT_PAIRINGS = ("convnet6", "resnet34_resnet18")
_matrix_jobs, _plan_matrix, _materialize_matrix_plan = matrix_jobs, plan_matrix, materialize_matrix_plan


def matrix_jobs(*args, **kwargs):
    return _matrix_jobs(*args, **{"pairs": ("convnet6",), **kwargs})


def plan_matrix(*args, **kwargs):
    return _plan_matrix(*args, **{"pairs": ("convnet6",), **kwargs})


def materialize_matrix_plan(*args, **kwargs):
    return _materialize_matrix_plan(*args, **{"pairs": ("convnet6",), **kwargs})


class TeacherBaselineMatrixTests(unittest.TestCase):
    def test_all_52_specs_roundtrip_and_follow_main_dataset_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = matrix_jobs(datasets=tuple(DATASETS), root=root / "data")
            self.assertEqual(len(jobs), 52)
            self.assertEqual(len({job["id"] for job in jobs}), 52)
            self.assertEqual({job["dataset"] for job in jobs}, set(DATASETS))
            self.assertEqual({job["profile"] for job in jobs}, set(PROFILES))
            for index, job in enumerate(jobs):
                with self.subTest(job=job["id"]):
                    config = root / f"{index}.json"
                    config.write_text(json.dumps(job["study"]))
                    spec = load_spec(config)
                    self.assertEqual(spec.to_dict(), job["study"])
                    data, train = spec.teacher.data, spec.teacher.train
                    self.assertEqual(data.source, job["dataset"])
                    self.assertEqual(data.dataset_version, DATASETS[data.source]["version"])
                    self.assertEqual(data.num_classes, DATASETS[data.source]["classes"])
                    self.assertEqual(data.image_size, DATASETS[data.source]["image_size"])
                    self.assertEqual(data.dataset_profile, job["profile"])
                    self.assertEqual(data.horizontal_flip, data.source in {"cifar10", "cifar100", "cinic10"})
                    self.assertEqual(data.root, str((root / "data").resolve()))
                    self.assertEqual(data.validation_fraction, 0.1)
                    self.assertEqual(data.confirmation_fraction, 0)
                    self.assertEqual(data.split_seed, 2026)
                    self.assertEqual(train.epochs, DEFAULT_EPOCHS)
                    self.assertEqual(train.workers, 0)
                    self.assertEqual(train.precision, "float32")
                    self.assertEqual(train.device, "auto")
                    self.assertEqual(spec.teacher.student.name, "convnet6_w32")
                    self.assertEqual(spec.student_model, "convnet6_w16")
                    self.assertEqual(spec.teacher_seeds, (41,))
                    self.assertEqual(spec.student_seeds, (42, 43, 44))
                    self.assertEqual(spec.student_selection_metric,
                                     "accuracy" if job["profile"] == "balanced" else "balanced_accuracy")
                    self.assertTrue(spec.repair.allow_sparse_class_fallback)
                    self.assertEqual(spec.repair.channel_budgets, (2, 4, 8))
                    self.assertEqual(spec.repair.learning_rates, (0.001, 0.005))
                    plan = plan_study(spec, root / "study" / job["id"])
                    self.assertEqual(plan["repair_protocols"][0]["imbalance_factor"],
                                     PROFILES[job["profile"]]["factor"])
                    self.assertEqual(plan["repair_protocols"][0]["target_class_policy"],
                                     PROFILES[job["profile"]]["target_policy"])
                    self.assertEqual(plan["repair_protocols"][0]["stages"], ["stage2", "stage3"])

    def test_optional_two_pairings_have_104_valid_settings(self):
        jobs = matrix_jobs(datasets=tuple(DATASETS), pairs=COMPACT_PAIRINGS)
        self.assertEqual(len(jobs), 104)
        self.assertEqual(len({job["id"] for job in jobs}), 104)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, job in enumerate(jobs):
                with self.subTest(job=job["id"]):
                    config = root / f"{index}.json"
                    config.write_text(json.dumps(job["study"]))
                    spec = load_spec(config)
                    self.assertEqual(spec.teacher.student.name, PAIRINGS[job["pair"]]["teacher"])
                    self.assertEqual(spec.student_model, PAIRINGS[job["pair"]]["student"])
                    protocol = plan_study(spec, root / "runs" / job["id"])
                    stages = ["stage2", "stage3"] if job["pair"] == "convnet6" else ["stage3", "stage4"]
                    self.assertEqual(protocol["repair_protocols"][0]["stages"], stages)

    def test_balanced_softmax_omitted_only_for_equal_count_cifar_controls(self):
        for job in matrix_jobs():
            recipes = job["study"]["recipes"]
            omitted = job["dataset"] in {"cifar10", "cifar100"} and job["profile"] == "balanced"
            self.assertEqual("balanced_softmax" not in recipes, omitted)
            self.assertEqual(set(recipes) - {"teacher_repair"},
                             set(RECIPES) - ({"balanced_softmax"} if omitted else set()))

    def test_aggregate_fit_counts_are_upper_bounds_not_job_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = plan_matrix(Path(temporary) / "plan", datasets=tuple(DATASETS))
            self.assertEqual(plan["job_count"], 52)
            self.assertEqual(plan["counts"], {
                "teacher_conditions": 622, "teacher_training_fits": 570,
                "repair_candidate_fits_max": 312, "kd_student_fits_max": 1866,
                "supervised_student_fits": 156, "total_fits_max": 2904})
            self.assertIn("upper bounds", plan["counts_note"])
            doubled = plan_matrix(Path(temporary) / "two-pairs", datasets=tuple(DATASETS), pairs=COMPACT_PAIRINGS)
            self.assertEqual(doubled["job_count"], 104)
            self.assertEqual(doubled["counts"], {key: value * 2 for key, value in plan["counts"].items()})

    def test_dry_run_and_cli_create_no_paths_and_do_not_access_datasets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, data, runs = root / "parent" / "plan", root / "data", root / "runs"
            with patch("kd.teacher_baselines.build_data", side_effect=AssertionError("dataset access")), \
                    patch("kd.teacher_baselines.run_experiment", side_effect=AssertionError("training")):
                plan = materialize_matrix_plan(output, datasets=("cifar10",), profiles=("lt-if100",),
                                               root=data, run_root=runs, dry_run=True)
                self.assertEqual(plan["job_count"], 1)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    main(["plan", "--output", str(output), "--root", str(data), "--run-root", str(runs),
                          "--datasets", "cifar10", "--profiles", "lt-if100", "--pairs", "convnet6", "--dry-run"])
                self.assertEqual(json.loads(stdout.getvalue()), plan)
            self.assertEqual(list(root.iterdir()), [])

    def test_materialization_is_idempotent_and_keeps_runs_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "plan"
            arguments = {"datasets": ("cifar10", "svhn"), "profiles": ("balanced", "lt-if10"),
                         "root": root / "data", "run_root": root / "study-runs"}
            result = materialize_matrix_plan(output, **arguments)
            self.assertEqual(result["job_count"], 4)
            before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in output.rglob("*") if path.is_file()}
            self.assertEqual(materialize_matrix_plan(output, **arguments), result)
            after = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                     for path in output.rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            protocol = json.loads((output / "protocol.json").read_text())
            for job in protocol["jobs"]:
                self.assertEqual(load_spec(job["config"]).to_dict(), job["study"])
                self.assertEqual(job["study"]["teacher"]["output_dir"], job["output"])
                self.assertFalse(Path(job["output"]).is_relative_to(output))
                self.assertFalse(Path(job["output"]).exists())
                self.assertEqual(job["study_policy"]["student_checkpoint_rule"],
                                 "best validation accuracy (existing engine)")
                self.assertIn("repair_protocols", job["study_policy"])
            self.assertEqual(len((output / "jobs.tsv").read_text().splitlines()), 5)
            for source in ("teacher_baseline_matrix.py", "neuron_surgery_matrix.py",
                           "neuron_surgery_baselines.py", "catalog/v1.json", "models.py"):
                self.assertIn(source, protocol["source_sha256"])

    def test_changed_settings_and_changed_source_reject_existing_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plan"
            settings = {"datasets": ("svhn",), "profiles": ("lt-if50",)}
            materialize_matrix_plan(output, **settings)
            with self.assertRaisesRegex(ValueError, "changed"):
                materialize_matrix_plan(output, **settings, workers=2)
            protocol_path = output / "protocol.json"
            original = protocol_path.read_bytes()
            with patch("kd.teacher_baseline_matrix.fingerprint", return_value="changed-source"):
                with self.assertRaisesRegex(ValueError, "changed"):
                    materialize_matrix_plan(output, **settings)
            self.assertEqual(protocol_path.read_bytes(), original)

    def test_conflicting_config_or_queue_is_never_overwritten(self):
        for target in ("config", "queue"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "plan"
                settings = {"datasets": ("svhn",), "profiles": ("balanced",)}
                materialize_matrix_plan(output, **settings)
                config = output / "configs" / "svhn-balanced-convnet6.json"
                damaged = config if target == "config" else output / "jobs.tsv"
                damaged.write_text("{}" if target == "config" else "changed queue\n")
                snapshot = damaged.read_bytes()
                # A missing earlier artifact must not be recreated before the
                # planner notices a conflict in a later artifact.
                if target == "queue":
                    config.unlink()
                with self.assertRaisesRegex(ValueError, "changed"):
                    materialize_matrix_plan(output, **settings)
                self.assertEqual(damaged.read_bytes(), snapshot)
                if target == "queue":
                    self.assertFalse(config.exists())

    def test_nonempty_unbound_directory_and_symlink_config_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plan"
            output.mkdir()
            unrelated = output / "user.txt"
            unrelated.write_text("preserve me")
            settings = {"datasets": ("cifar10",), "profiles": ("balanced",)}
            with self.assertRaises(FileExistsError):
                materialize_matrix_plan(output, **settings)
            self.assertEqual(unrelated.read_text(), "preserve me")
            fresh = Path(temporary) / "fresh"
            materialize_matrix_plan(fresh, **settings)
            config = fresh / "configs" / "cifar10-balanced-convnet6.json"
            moved = Path(temporary) / "external.json"
            config.rename(moved)
            config.symlink_to(moved)
            with self.assertRaisesRegex(ValueError, "symlink"):
                materialize_matrix_plan(fresh, **settings)

    def test_invalid_selections_seeds_runtime_and_paths_are_rejected(self):
        invalid = ({"datasets": ("../cifar10",)}, {"profiles": ("../balanced",)},
                   {"pairs": ("../convnet6",)}, {"datasets": ()}, {"profiles": ()}, {"pairs": ()},
                   {"datasets": ("cifar10", "cifar10")}, {"pairs": ("convnet6", "convnet6")},
                   {"teacher_seeds": ()}, {"student_seeds": ()}, {"student_seeds": (42, 42)},
                   {"teacher_seeds": (True,)}, {"teacher_seeds": (-1,)}, {"workers": True},
                   {"epochs": True}, {"epochs": 0}, {"epochs": -1}, {"epochs": 1.5},
                   {"epochs": "100"}, {"epochs": None},
                   {"workers": -1}, {"device": "cuda:bad"}, {"precision": "fp32"},
                   {"precision": "float16", "device": "cuda:0"},
                   {"root": ""}, {"root": "data\nother"}, {"root": "data\x7fother"}, {"root": None})
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                matrix_jobs(**options)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for output, run_root in ((root / "plan", root / "plan"),
                                     (root / "plan", root / "plan" / "studies"),
                                     (root / "plan", root)):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    plan_matrix(output, datasets=("svhn",), profiles=("balanced",), run_root=run_root)
            for output in ("", "plan\tbad", None):
                with self.assertRaises(ValueError):
                    plan_matrix(output, datasets=("svhn",), profiles=("balanced",))

    def test_unknown_plan_entries_fail_before_recreating_missing_files(self):
        for kind in ("file", "directory", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                output = root / "plan"
                settings = {"datasets": ("svhn",), "profiles": ("balanced",)}
                materialize_matrix_plan(output, **settings)
                config = output / "configs" / "svhn-balanced-convnet6.json"
                config.unlink()
                unknown = output / "unknown"
                if kind == "directory":
                    unknown.mkdir()
                elif kind == "symlink":
                    unknown.symlink_to(root / "missing-target")
                else:
                    unknown.write_text("do not modify")
                before = (output / "protocol.json").read_bytes()
                with self.assertRaisesRegex(ValueError, "Unknown|symlink"):
                    materialize_matrix_plan(output, **settings)
                self.assertFalse(config.exists())
                self.assertEqual((output / "protocol.json").read_bytes(), before)
                self.assertTrue(unknown.exists() or unknown.is_symlink())

    def test_data_root_and_effective_study_paths_cannot_overlap_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output, runs, data = root / "plan", root / "runs", root / "data"
            settings = {"datasets": ("svhn",), "profiles": ("balanced",)}
            for data_root in (output, output / "data", root, runs, runs / "data"):
                with self.subTest(data_root=data_root), self.assertRaisesRegex(ValueError, "overlap"):
                    plan_matrix(output, **settings, run_root=runs, root=data_root)
            data.mkdir()
            runs.mkdir()
            (runs / "convnet6").symlink_to(data, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Effective study output"):
                plan_matrix(output, **settings, run_root=runs, root=data)
            self.assertFalse(output.exists())
            self.assertEqual(list(data.iterdir()), [])

    def test_effective_study_path_aliases_and_file_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output, runs = root / "plan", root / "runs"
            base = runs / "convnet6" / "svhn"
            base.mkdir(parents=True)
            (base / "lt-if10").symlink_to(base / "balanced", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "unique"):
                plan_matrix(output, datasets=("svhn",), profiles=("balanced", "lt-if10"),
                            run_root=runs, root=root / "data")
            existing_file = root / "existing-file"
            existing_file.write_text("preserve me")
            for options in ({"output": existing_file}, {"run_root": existing_file}, {"root": existing_file}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "directory"):
                    plan_matrix(**{"output": output, "run_root": runs, "root": root / "data",
                                   "datasets": ("cifar10",), "profiles": ("balanced",), **options})
            self.assertEqual(existing_file.read_text(), "preserve me")

    def test_matrix_records_added_per_study_policy_metadata(self):
        def extended_plan(*args, **kwargs):
            return {**plan_study(*args, **kwargs), "teacher_augmentation_policy": {"safe": True}}

        with tempfile.TemporaryDirectory() as temporary, \
                patch("kd.teacher_baseline_matrix.plan_study", side_effect=extended_plan):
            plan = plan_matrix(Path(temporary) / "plan", datasets=("svhn",), profiles=("balanced",))
            self.assertEqual(plan["jobs"][0]["study_policy"]["teacher_augmentation_policy"], {"safe": True})

    def test_dataset_specific_randaugment_policies_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = plan_matrix(Path(temporary) / "plan", profiles=("balanced",))
            for job in plan["jobs"]:
                with self.subTest(dataset=job["dataset"]):
                    policies = job["study_policy"]["teacher_augmentation_policies"]
                    self.assertEqual(set(policies), {"randaugment", "randaugment_cutmix_ema"})
                    self.assertEqual(policies["randaugment"], policies["randaugment_cutmix_ema"])
                    policy = policies["randaugment"]
                    if job["dataset"] == "colored_mnist":
                        self.assertEqual(policy["name"], "restricted_colored_digit_photometric")
                        self.assertEqual(policy["operations"], ["Identity", "Brightness", "Contrast", "Sharpness"])
                    elif job["dataset"] in {"svhn", "gtsrb"}:
                        self.assertEqual(policy["name"], "restricted_digit_sign_photometric")
                        self.assertEqual(policy["operations"],
                                         ["Identity", "Brightness", "Contrast", "Sharpness", "AutoContrast", "Equalize"])
                    elif job["dataset"] in {"pathmnist", "organamnist", "bloodmnist", "dermamnist"}:
                        self.assertEqual(policy["name"], "restricted_medical_photometric")
                        self.assertEqual(policy["operations"], ["Identity", "Brightness", "Contrast", "Sharpness"])
                    else:
                        self.assertEqual(policy["name"], "torchvision_standard")
                        self.assertEqual(policy["operations"],
                                         ["Identity", "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate",
                                          "Brightness", "Color", "Contrast", "Sharpness", "Posterize", "Solarize",
                                          "AutoContrast", "Equalize"])

    def test_explicit_runtime_and_seed_overrides_are_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plan"
            plan = plan_matrix(output, datasets=("organamnist",), profiles=("lt-if10",),
                               device="cuda:3", precision="bfloat16", workers=3,
                               teacher_seeds=(7, 8), student_seeds=(9, 10))
            job = plan["jobs"][0]
            train = job["study"]["teacher"]["train"]
            self.assertEqual((train["device"], train["precision"], train["workers"]),
                             ("cuda:3", "bfloat16", 3))
            self.assertEqual(job["study"]["teacher_seeds"], [7, 8])
            self.assertEqual(job["study"]["student_seeds"], [9, 10])
            self.assertEqual(job["counts"]["teachers"], 24)
            self.assertEqual(job["counts"]["repair_candidate_fits_max"], 12)
            self.assertEqual(job["counts"]["kd_students"], 48)
            self.assertEqual(job["counts"]["total_fits_max"], 84)

    def test_uniform_epoch_budget_is_explicit_without_changing_historical_profiles(self):
        self.assertEqual({name: value["epochs"] for name, value in PROFILES.items()},
                         {"balanced": 81, "lt-if10": 49, "lt-if50": 71, "lt-if100": 81})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = plan_matrix(root / "plan", pairs=COMPACT_PAIRINGS, epochs=120)
            self.assertEqual(plan["version"], 2)
            self.assertEqual(plan["epoch_policy"], {
                "teacher_student_epochs": 120,
                "scope": "uniform across all selected datasets, profiles and architecture pairs",
                "historical_profile_epochs_unchanged": True, "repair_epochs": 20})
            self.assertEqual({job["study"]["teacher"]["train"]["epochs"] for job in plan["jobs"]}, {120})
            materialize_matrix_plan(root / "one", datasets=("cifar10",), profiles=("balanced",),
                                    root=root / "data", epochs=120)
            spec = load_spec(root / "one/configs/cifar10-balanced-convnet6.json")
            teachers, students, supervised = study_configs(spec, spec.teacher.output_dir)
            self.assertEqual({config.train.epochs for config in
                              [*(row[1] for row in teachers), *(row[2] for row in students), *supervised]}, {120})
            self.assertEqual(spec.repair.repair_epochs, 20)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(["plan", "--output", str(root / "cli"), "--datasets", "svhn",
                      "--profiles", "lt-if10", "--epochs", "120", "--dry-run"])
            self.assertEqual(json.loads(stdout.getvalue())["epoch_policy"]["teacher_student_epochs"], 120)
            self.assertFalse((root / "cli").exists())

    def test_generated_execution_binding_covers_parent_config_sources_and_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize_matrix_plan(root / "plan", datasets=("svhn",), profiles=("balanced",),
                                    root=root / "data")
            protocol = json.loads((root / "plan/protocol.json").read_text())
            job = protocol["jobs"][0]
            binding = job["study"]["matrix_binding"]
            self.assertEqual(binding["parent_payload_sha256"], matrix_payload_digest(protocol))
            self.assertEqual(set(protocol["source_sha256"]), set(STUDY_SOURCE_FILES + MATRIX_SOURCE_FILES))
            spec = load_spec(job["config"])
            snapshot = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                        for path in root.rglob("*") if path.is_file()}
            for phase in ("plan", "train", "evaluate"):
                with self.subTest(phase=phase):
                    report = run_study(spec, job["output"], phase=phase, dry_run=True)
                    self.assertTrue(report["dry_run"])
            self.assertEqual(snapshot, {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                                       for path in root.rglob("*") if path.is_file()})
            self.assertFalse(Path(job["output"]).exists())

    def test_execution_binding_rejects_config_and_in_memory_spec_changes_before_writes(self):
        for mutation in ("recipe", "epochs", "model", "root", "output"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                materialize_matrix_plan(root / "plan", datasets=("svhn",), profiles=("balanced",),
                                        root=root / "data")
                config = root / "plan/configs/svhn-balanced-convnet6.json"
                spec = load_spec(config)
                raw = spec.to_dict()
                if mutation == "recipe":
                    modified = replace(spec, recipes=("control",), repair=None)
                    raw["recipes"], raw["repair"] = ["control"], None
                elif mutation == "epochs":
                    modified = replace(spec, teacher=replace(spec.teacher,
                        train=replace(spec.teacher.train, epochs=3)))
                    raw["teacher"]["train"]["epochs"] = 3
                elif mutation == "model":
                    modified = replace(spec, student_model="resnet18_smallstem")
                    raw["student_model"] = "resnet18_smallstem"
                elif mutation == "root":
                    modified = replace(spec, teacher=replace(spec.teacher,
                        data=replace(spec.teacher.data, root=str(root / "other-data"))))
                    raw["teacher"]["data"]["root"] = str(root / "other-data")
                else:
                    modified = replace(spec, teacher=replace(spec.teacher, output_dir=str(root / "other-output")))
                    raw["teacher"]["output_dir"] = str(root / "other-output")
                with patch("kd.teacher_baselines.build_data", side_effect=AssertionError("dataset access")), \
                        patch("kd.teacher_baselines.resolve_device", side_effect=AssertionError("device access")):
                    for phase in ("train", "evaluate"):
                        with self.assertRaisesRegex(ValueError, "binding"):
                            run_study(modified, spec.teacher.output_dir, phase=phase)
                    config.write_text(json.dumps(raw))
                    with self.assertRaisesRegex(ValueError, "binding"):
                        load_spec(config)
                self.assertFalse(Path(spec.teacher.output_dir).exists())
                self.assertFalse((root / "data").exists())
                self.assertFalse((root / "other-output").exists())

    def test_execution_binding_rejects_parent_source_queue_and_source_coverage_changes(self):
        for mutation in ("parent", "missing-parent", "source", "missing-source", "extra-source", "queue"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                materialize_matrix_plan(root / "plan", datasets=("svhn",), profiles=("balanced",),
                                        root=root / "data")
                config = root / "plan/configs/svhn-balanced-convnet6.json"
                spec = load_spec(config)
                protocol_path = root / "plan/protocol.json"
                protocol = json.loads(protocol_path.read_text())
                if mutation == "parent":
                    protocol["epoch_policy"]["teacher_student_epochs"] = 999
                    protocol_path.write_text(json.dumps(protocol))
                elif mutation == "missing-parent":
                    protocol_path.unlink()
                elif mutation in {"missing-source", "extra-source"}:
                    if mutation == "missing-source":
                        del protocol["source_sha256"]["engine.py"]
                    else:
                        protocol["source_sha256"]["unexpected.py"] = "0" * 64
                    # Even a consistent re-digest cannot make incomplete source
                    # coverage acceptable to the current execution guard.
                    binding = protocol["jobs"][0]["study"]["matrix_binding"]
                    binding["parent_payload_sha256"] = matrix_payload_digest(protocol)
                    protocol_path.write_text(json.dumps(protocol))
                    config.write_text(json.dumps(protocol["jobs"][0]["study"]))
                    spec = replace(spec, matrix_binding=binding)
                elif mutation == "queue":
                    (root / "plan/jobs.tsv").write_text("changed queue\n")
                source_patch = (patch("kd.matrix_binding.fingerprint", return_value="changed-source")
                                if mutation == "source" else contextlib.nullcontext())
                with source_patch, patch("kd.teacher_baselines.build_data", side_effect=AssertionError("dataset access")):
                    with self.assertRaisesRegex(ValueError, "binding"):
                        load_spec(config)
                    for phase in ("train", "evaluate"):
                        with self.assertRaisesRegex(ValueError, "binding"):
                            run_study(spec, spec.teacher.output_dir, phase=phase)
                self.assertFalse(Path(spec.teacher.output_dir).exists())
                self.assertFalse((root / "data").exists())

    def test_cli_output_and_recipe_overrides_and_relocated_config_fail_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize_matrix_plan(root / "plan", datasets=("svhn",), profiles=("balanced",),
                                    root=root / "data")
            config = root / "plan/configs/svhn-balanced-convnet6.json"
            spec = load_spec(config)
            for phase in ("train", "evaluate"):
                for overrides in (["--output", str(root / "other-output")],
                                  ["--output", spec.teacher.output_dir, "--recipes", "control"],
                                  ["--output", spec.teacher.output_dir, "--device", "cpu"]):
                    with self.subTest(phase=phase, overrides=overrides), \
                            patch("kd.teacher_baselines.build_data", side_effect=AssertionError("dataset access")), \
                            self.assertRaisesRegex(ValueError, "binding"):
                        study_main([phase, "--config", str(config), *overrides])
            copy = root / "copied-bound.json"
            copy.write_bytes(config.read_bytes())
            with self.assertRaisesRegex(ValueError, "config path"):
                load_spec(copy)
            self.assertFalse((root / "other-output").exists())
            self.assertFalse(Path(spec.teacher.output_dir).exists())

    def test_unbound_legacy_and_stripped_generated_configs_require_a_fresh_plan(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                materialize_matrix_plan(root / "plan", datasets=("svhn",), profiles=("balanced",),
                                        root=root / "data")
                config = root / "plan/configs/svhn-balanced-convnet6.json"
                raw = json.loads(config.read_text())
                del raw["matrix_binding"]
                config.write_text(json.dumps(raw))
                parent = root / "plan/protocol.json"
                protocol = json.loads(parent.read_text())
                protocol["version"] = version
                parent.write_text(json.dumps(protocol))
                with self.assertRaisesRegex(ValueError, "fresh matrix plan"):
                    load_spec(config)
                standalone = root / "explicit-standalone.json"
                standalone.write_text(json.dumps(raw))
                self.assertIsNone(load_spec(standalone).matrix_binding)


if __name__ == "__main__":
    unittest.main()
