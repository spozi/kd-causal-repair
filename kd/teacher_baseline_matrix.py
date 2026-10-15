"""Plan the all-dataset teacher-training/repair and matched-KD benchmark.

This module only generates immutable configurations. It never fetches datasets,
constructs a model, or launches training. Run each generated config with
``python -m kd.teacher_baselines train --config ... --output ...`` afterwards.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .checkpoints import fingerprint
from .config import (BenchmarkConfig, DataConfig, DistillationConfig, ExperimentConfig,
                     ModelConfig, TrainConfig, from_dict)
from .matrix_binding import (BINDING_VERSION, MATRIX_SOURCE_FILES, canonical_digest,
                             matrix_payload_digest)
from .neuron_surgery_baselines import baseline_configs
from .neuron_surgery_matrix import DATASETS as CLASS_DATASETS, PROFILES, matrix_spec
from .shortcut_data import SHORTCUT_DATASETS
from .shortcut_baselines import DEFAULT_GROUP_POLICY
from .runtime import PRECISIONS, valid_device_spec
from .teacher_baselines import TeacherBaselineSpec, plan_study
from .teacher_repair_baseline import REPAIR_NAME, TeacherRepairOptions, repair_options
from .teacher_training import RECIPES


MATRIX_VERSION = 2
DEFAULT_EPOCHS = 100
PAIRINGS = {
    "convnet6": {"teacher": "convnet6_w32", "student": "convnet6_w16"},
    "resnet34_resnet18": {"teacher": "resnet34_smallstem", "student": "resnet18_smallstem"},
}
DEFAULT_PAIRS = ("convnet6",)
DATASETS = {**CLASS_DATASETS, **SHORTCUT_DATASETS}


def _shortcut_job(dataset, *, root, device, precision, workers, epochs, teacher_seeds, student_seeds):
    info = SHORTCUT_DATASETS[dataset]
    small = dataset == "colored_mnist"
    teacher_model, student_model = (("convnet6_w32", "convnet6_w16") if small else ("resnet34", "resnet18"))
    weights = None if small else "IMAGENET1K_V1"
    teacher = ExperimentConfig(name="teacher", output_dir="runs/teacher-baselines-v5",
        data=DataConfig(source=dataset, root=str(root), num_classes=2, image_size=info["image_size"],
                        dataset_version=info["version"], dataset_profile="native-shortcut",
                        horizontal_flip=dataset == "urbancars"),
        student=ModelConfig(teacher_model, weights=weights), distillation=DistillationConfig(method="supervised"),
        train=TrainConfig(epochs=epochs, batch_size=128 if small else 32,
            learning_rate=0.05 if small else 0.001, momentum=0.9, weight_decay=0.0005,
            workers=workers, seed=teacher_seeds[0], device=device, precision=precision, threads=4,
            selection_metric="worst_group_accuracy"), benchmark=BenchmarkConfig(warmup=20, iterations=100))
    spec = TeacherBaselineSpec(teacher=teacher, student_model=student_model, student_weights=weights,
        teacher_seeds=teacher_seeds, student_seeds=student_seeds, recipes=(*RECIPES, REPAIR_NAME),
        teacher_selection_metric="worst_group_accuracy", student_selection_metric="worst_group_accuracy",
        repair=TeacherRepairOptions(repair_batch_size=64 if small else 16), group_policy=dict(DEFAULT_GROUP_POLICY))
    spec.validate()
    pair = "convnet6" if small else "resnet34_resnet18_imagenet"
    return {"id": f"{dataset}-native-shortcut-{pair}", "dataset": dataset, "version": info["version"],
            "profile": "native-shortcut", "pair": pair, "evaluation_family": "shortcut", "study": spec.to_dict()}


def _selection(values, choices, label):
    values = tuple(choices if values is None else values)
    if (not values or any(not isinstance(value, str) or value not in choices for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{label} must be nonempty, unique selections from {tuple(choices)}")
    return values


def _path(value, label):
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{label} must be a string or Path")
    raw = str(value)
    if not raw.strip() or any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise ValueError(f"{label} must be a nonempty path without control characters")
    return Path(raw).expanduser().resolve()


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _overlap(first, second):
    return first.is_relative_to(second) or second.is_relative_to(first)


def _spec_digest(value):
    return canonical_digest(value)


def matrix_jobs(datasets=None, profiles=None, pairs=None, *, root="data", device="auto",
                precision="float32", workers=0, epochs=DEFAULT_EPOCHS,
                teacher_seeds=(41,), student_seeds=(42, 43, 44)):
    """Return validated dataset/profile/pair settings, without dataset access.

    Dataset sizes, flip policy and split identity come from the historical main
    matrix. New teacher/student runs explicitly override its profile-dependent
    epoch budgets with one common budget; repair retains its separate budget.
    """
    datasets = _selection(datasets, DATASETS, "datasets")
    profiles = _selection(profiles, PROFILES, "profiles")
    pairs = _selection(DEFAULT_PAIRS if pairs is None else pairs, PAIRINGS, "pairs")
    root = _path(root, "root")
    if root.exists() and not root.is_dir():
        raise ValueError(f"Dataset root must be a directory: {root}")
    if not isinstance(device, str) or not valid_device_spec(device):
        raise ValueError("Invalid device specification")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    teacher_seeds, student_seeds = tuple(teacher_seeds), tuple(student_seeds)
    # Validate seeds before indexing teacher_seeds below.
    for seeds in (teacher_seeds, student_seeds):
        if (not seeds or len(set(seeds)) != len(seeds)
                or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)):
            raise ValueError("Seeds must be nonempty, unique nonnegative integers")

    jobs = []
    for dataset in datasets:
        if dataset in SHORTCUT_DATASETS:
            jobs.append(_shortcut_job(dataset, root=root, device=device, precision=precision,
                workers=workers, epochs=epochs, teacher_seeds=teacher_seeds, student_seeds=student_seeds))
            continue
        for profile in profiles:
            for pair in pairs:
                models = PAIRINGS[pair]
                original = replace(matrix_spec(dataset, profile),
                                   teacher_model=models["teacher"], student_model=models["student"],
                                   teacher_seed=teacher_seeds[0], student_seeds=student_seeds)
                # output_dir is a placeholder; plan_study/run_study bind the
                # actual study output for every teacher/student run.
                teacher, _ = baseline_configs("runs/teacher-baselines-v3", root, device, original)
                teacher = replace(teacher, name="teacher",
                                  train=replace(teacher.train, workers=workers, precision=precision,
                                                epochs=epochs))
                recipes = tuple(RECIPES)
                # These two balanced training splits have exactly equal counts,
                # so Balanced Softmax would duplicate the CE control. Other
                # naturally uneven datasets retain it even in balanced profile.
                if dataset in {"cifar10", "cifar100"} and profile == "balanced":
                    recipes = tuple(recipe for recipe in recipes if recipe != "balanced_softmax")
                spec = TeacherBaselineSpec(
                    teacher=teacher, student_model=models["student"], teacher_seeds=teacher_seeds,
                    student_seeds=student_seeds, recipes=(*recipes, REPAIR_NAME),
                    teacher_selection_metric="accuracy",
                    student_selection_metric="accuracy" if profile == "balanced" else "balanced_accuracy",
                    repair=TeacherRepairOptions(allow_sparse_class_fallback=True))
                spec.validate()
                jobs.append({"id": f"{dataset}-{profile}-{pair}", "dataset": dataset,
                             "version": DATASETS[dataset]["version"], "profile": profile,
                             "pair": pair, "study": spec.to_dict()})
    return jobs


def plan_matrix(output, datasets=None, profiles=None, pairs=None, *, run_root=None, **settings):
    """Build a read-only, source-bound matrix protocol and fit-count upper bounds."""
    output = _path(output, "output")
    run_root = _path(output.parent / "studies" if run_root is None else run_root, "run_root")
    if _overlap(output, run_root):
        raise ValueError("Matrix plan directory and study run_root must not overlap")
    data_root = _path(settings.get("root", "data"), "root")
    if _overlap(data_root, output) or _overlap(data_root, run_root):
        raise ValueError("Dataset root must not overlap the matrix plan or study run_root")
    for path, label in ((output, "Matrix output"), (run_root, "Study run_root")):
        if path.exists() and not path.is_dir():
            raise ValueError(f"{label} must be a directory: {path}")
    jobs = matrix_jobs(datasets, profiles, pairs, **settings)
    totals = {"teacher_conditions": 0, "teacher_training_fits": 0,
              "repair_candidate_fits_max": 0, "kd_student_fits_max": 0,
              "supervised_student_fits": 0, "total_fits_max": 0}
    sources = None
    study_paths = set()
    for job in jobs:
        study = job["study"]
        # Use the same validator/converter as the per-study loader, without
        # requiring a temporary JSON file or any filesystem writes.
        spec = TeacherBaselineSpec(
            **{**study, "teacher": from_dict(study["teacher"]),
               "teacher_seeds": tuple(study["teacher_seeds"]),
               "student_seeds": tuple(study["student_seeds"]), "recipes": tuple(study["recipes"]),
               "repair": repair_options(study["repair"]) if study["repair"] is not None else None})
        study_output = (run_root / job["pair"] / job["dataset"] / job["profile"]).resolve()
        if (not study_output.is_relative_to(run_root) or _overlap(study_output, data_root)
                or _overlap(study_output, output) or study_output in study_paths):
            raise ValueError("Effective study output paths must be unique, contained in run_root, and separate from plan/data roots")
        if study_output.exists() and not study_output.is_dir():
            raise ValueError(f"Study output must be a directory: {study_output}")
        study_paths.add(study_output)
        # Materialized configs should advertise the same output as the queue,
        # even though the per-study runner explicitly binds its --output again.
        spec = replace(spec, teacher=replace(spec.teacher, output_dir=str(study_output)))
        study = spec.to_dict()
        job["study"] = study
        plan = plan_study(spec, study_output)
        if sources is not None and sources != plan["source_sha256"]:
            raise ValueError("Source files changed while planning the matrix; retry with stable code")
        sources = plan["source_sha256"]
        counts = plan["counts"]
        excluded = {"version", "purpose", "spec", "output", "teacher_runs", "student_runs",
                    "supervised_student_runs", "counts", "counts_note", "source_sha256"}
        job.update(config=str(output / "configs" / f"{job['id']}.json"),
                   output=str(study_output), spec_sha256=_spec_digest(study), counts=counts,
                   study_policy={key: value for key, value in plan.items() if key not in excluded})
        for aggregate, key in (("teacher_conditions", "teachers"),
                               ("teacher_training_fits", "teacher_training_fits"),
                               ("repair_candidate_fits_max", "repair_candidate_fits_max"),
                               ("kd_student_fits_max", "kd_students"),
                               ("supervised_student_fits", "supervised_students"),
                               ("total_fits_max", "total_fits_max")):
            totals[aggregate] += counts[key]
    sources = {**sources, **{name: fingerprint(Path(__file__).parent / name)
                            for name in MATRIX_SOURCE_FILES}}
    protocol = _json({"version": MATRIX_VERSION,
                  "purpose": "All-dataset teacher performance and downstream distillability benchmark",
                  "datasets": list(dict.fromkeys(job["dataset"] for job in jobs)),
                  "profiles": list(dict.fromkeys(job["profile"] for job in jobs)),
                  "pairs": list(dict.fromkeys(job["pair"] for job in jobs)),
                  "output": str(output), "run_root": str(run_root), "job_count": len(jobs),
                  "counts": totals,
                  "counts_note": "Jobs count dataset/profile/pair settings, not training fits. Fit counts are upper bounds; rejected repairs reuse control KD students.",
                  "training_policy": "This planner performs no training or dataset access; run each generated config separately with kd.teacher_baselines.",
                  "epoch_policy": {"teacher_student_epochs": settings.get("epochs", DEFAULT_EPOCHS),
                                   "scope": "uniform across all selected datasets, profiles and architecture pairs",
                                   "historical_profile_epochs_unchanged": True,
                                   "repair_epochs": TeacherRepairOptions().repair_epochs},
                  "source_sha256": sources, "jobs": jobs})
    parent_digest = matrix_payload_digest(protocol)
    for job in protocol["jobs"]:
        job["study"]["matrix_binding"] = {
            "version": BINDING_VERSION, "parent_protocol": str(output / "protocol.json"),
            "parent_payload_sha256": parent_digest, "job_id": job["id"],
            "config": job["config"], "output": job["output"]}
    return protocol


def _queue(protocol):
    columns = ("id", "dataset", "version", "profile", "pair", "config", "output",
               "epochs", "total_fits_max")
    rows = ["\t".join(columns)]
    for job in protocol["jobs"]:
        rows.append("\t".join(str(value) for value in
                              (job["id"], job["dataset"], job["version"], job["profile"],
                               job["pair"], job["config"], job["output"],
                               job["study"]["teacher"]["train"]["epochs"],
                               job["counts"]["total_fits_max"])))
    return "\n".join(rows) + "\n"


def _check_existing(path, expected, *, as_json=False):
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink in immutable matrix plan: {path}")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Expected an immutable matrix file: {path}")
        try:
            actual = json.loads(path.read_text()) if as_json else path.read_text()
        except (OSError, ValueError) as error:
            raise ValueError(f"Unreadable immutable matrix file: {path}") from error
        if actual != expected:
            raise ValueError(f"Matrix protocol/config/queue changed: {path}; use a new plan directory")


def materialize_matrix_plan(output, datasets=None, profiles=None, pairs=None, *, dry_run=False,
                            run_root=None, **settings):
    """Write or verify immutable files; dry_run returns the plan without writes."""
    protocol = plan_matrix(output, datasets, profiles, pairs, run_root=run_root, **settings)
    if dry_run:
        return protocol
    directory = Path(protocol["output"])
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Matrix output must be a directory: {directory}")
    protocol_path = directory / "protocol.json"
    if directory.exists() and any(directory.iterdir()) and not protocol_path.exists():
        raise FileExistsError(f"Nonempty matrix directory has no compatible protocol: {directory}")
    configs = directory / "configs"
    if configs.is_symlink() or configs.exists() and not configs.is_dir():
        raise ValueError(f"Matrix configs must be a real directory: {configs}")
    queue = _queue(protocol)
    files = [(protocol_path, protocol, True),
             *( (Path(job["config"]), job["study"], True) for job in protocol["jobs"] ),
             (directory / "jobs.tsv", queue, False)]
    expected_paths = {configs, *(path for path, _, _ in files)}
    if directory.exists():
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Refusing a symlink in immutable matrix plan: {path}")
            if path not in expected_paths:
                raise ValueError(f"Unknown file/directory in immutable matrix plan: {path}; use a clean plan directory")
    # Check ALL existing artifacts before making any writes, so a conflict in a
    # later job cannot leave an incompatible partially updated plan.
    for path, expected, as_json in files:
        _check_existing(path, expected, as_json=as_json)
    configs.mkdir(parents=True, exist_ok=True)
    for path, value, as_json in files:
        if not path.exists():
            body = json.dumps(value, indent=2, allow_nan=False) + "\n" if as_json else value
            with path.open("x") as handle:
                handle.write(body)
    return {"status": "ready", "job_count": protocol["job_count"], "counts": protocol["counts"],
            "output": str(directory), "run_root": protocol["run_root"],
            "protocol": str(protocol_path), "jobs": str(directory / "jobs.tsv")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan",))
    parser.add_argument("--output", required=True, help="Immutable matrix-plan directory")
    parser.add_argument("--run-root", help="Separate parent for per-job study outputs")
    parser.add_argument("--root", default="data", help="Existing/downloadable dataset cache root")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS))
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES))
    parser.add_argument("--pairs", nargs="+", choices=tuple(PAIRINGS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=PRECISIONS, default="float32")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Common teacher/student epoch budget for every new matrix setting (default: 100)")
    parser.add_argument("--teacher-seeds", type=int, nargs="+", default=(41,))
    parser.add_argument("--student-seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--dry-run", action="store_true", help="Print protocol; no writes, downloads, or training")
    args = parser.parse_args(argv)
    report = materialize_matrix_plan(
        args.output, args.datasets, args.profiles, args.pairs, dry_run=args.dry_run,
        run_root=args.run_root, root=args.root, device=args.device, precision=args.precision,
        workers=args.workers, epochs=args.epochs,
        teacher_seeds=args.teacher_seeds, student_seeds=args.student_seeds)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
