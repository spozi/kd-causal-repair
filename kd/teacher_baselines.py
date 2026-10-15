"""Matched teacher-training and downstream KD benchmark.

Run ``python -m kd.teacher_baselines --help``. Training never loads test data;
the separate evaluate phase requires a frozen validation-based selection.
"""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
import statistics

import torch

from .baseline_artifacts import (STUDENT_FILES, TEACHER_FILES, read_json,
                                 verify_manifest, write_manifest)
from .checkpoints import (fingerprint, load_model_checkpoint, metadata,
                          save_checkpoint, tensor_state_fingerprint, write_json)
from .config import (BenchmarkConfig, DataConfig, DistillationConfig, ExperimentConfig,
                     ModelConfig, TrainConfig, from_dict)
from .data import _effective_factor, build_data
from .engine import run_experiment
from .evaluation import collect_predictions, prediction_metrics, save_prediction_report
from .models import MODEL_NAMES, create_model
from .matrix_binding import STUDY_SOURCE_FILES, verify_matrix_binding
from .runtime import configure_accelerator, prepare_model, resolve_device, resolve_precision
from .teacher_repair_baseline import (REPAIR_FILES, REPAIR_NAME, TeacherRepairOptions,
                                     repair_options, repair_spec, repair_targets, train_repair_teacher,
                                     verify_repair_details)
from .teacher_training import RECIPES, randaugment_policy, train_teacher
from .training_provenance import load_selected_checkpoint, validate_training_data


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


@dataclass(frozen=True)
class TeacherBaselineSpec:
    teacher: ExperimentConfig
    student_model: str = "cifar_student"
    teacher_seeds: tuple[int, ...] = (41,)
    student_seeds: tuple[int, ...] = (42, 43, 44)
    recipes: tuple[str, ...] = tuple(RECIPES)
    teacher_selection_metric: str = "accuracy"
    student_selection_metric: str = "balanced_accuracy"
    temperature: float = 4.0
    kd_weight: float = 0.5
    warmup_epochs: int = 5
    repair: TeacherRepairOptions | None = None
    matrix_binding: dict | None = None
    student_weights: str | None = None
    group_policy: dict | None = None

    def validate(self):
        self.teacher.validate()
        from .shortcut_data import SHORTCUT_DATASETS
        shortcut = self.teacher.data.source in SHORTCUT_DATASETS
        if self.teacher.distillation.method != "supervised":
            raise ValueError("Base teacher configuration must be supervised")
        if (self.teacher.calibration.enabled or self.teacher.cpc.enabled
                or self.teacher.supervised_loss.method != "cross_entropy"
                or self.teacher.surgery.action != "none"):
            raise ValueError("Teacher baselines require an unmodified supervised base recipe")
        if self.teacher.train.persistent_workers:
            raise ValueError("Epoch-boundary baseline resume requires persistent_workers=false")
        if self.teacher.student.name not in MODEL_NAMES or self.student_model not in MODEL_NAMES:
            raise ValueError("Unknown teacher/student architecture")
        if not self.recipes or len(set(self.recipes)) != len(self.recipes) or "control" not in self.recipes:
            raise ValueError("Recipes must be unique and include control")
        for name in self.recipes:
            if name == REPAIR_NAME:
                if self.repair is None:
                    raise ValueError("teacher_repair requires explicit repair options")
                continue
            if name not in RECIPES:
                raise ValueError(f"Unknown teacher recipe: {name}")
            RECIPES[name].validate()
            if RECIPES[name].sam_rho and self.teacher.train.precision == "float16":
                raise ValueError("SAM supports float32/bfloat16; set teacher precision explicitly")
        for seeds in (self.teacher_seeds, self.student_seeds):
            if (not seeds or len(set(seeds)) != len(seeds)
                    or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)):
                raise ValueError("Seeds must be unique nonnegative integers")
        if self.repair is not None:
            if REPAIR_NAME not in self.recipes:
                raise ValueError("Repair options require the teacher_repair condition")
            if shortcut:
                self.repair.validate()
            else:
                for seed in self.teacher_seeds:
                    repair_spec(self, seed)
        for metric in (self.teacher_selection_metric, self.student_selection_metric):
            if metric not in {"accuracy", "balanced_accuracy", "worst_group_accuracy"}:
                raise ValueError("Unknown teacher/student selection metric")
        if shortcut:
            from .shortcut_baselines import validate_shortcut_spec
            validate_shortcut_spec(self)
        elif self.group_policy is not None or "worst_group_accuracy" in (self.teacher_selection_metric, self.student_selection_metric):
            raise ValueError("Group policy/selection requires a shortcut dataset")
        if self.student_weights is not None:
            replace(self.teacher, student=ModelConfig(self.student_model, weights=self.student_weights)).validate(require_teacher=False)
        # Reuse existing validation of KD hyperparameters without needing a checkpoint yet.
        replace(self.teacher, distillation=DistillationConfig(
            method="kd", temperature=self.temperature, weight=self.kd_weight,
            warmup_epochs=self.warmup_epochs)).validate(require_teacher=False)

    def to_dict(self):
        value = {"teacher": self.teacher.to_dict(), "student_model": self.student_model,
                      "teacher_seeds": self.teacher_seeds, "student_seeds": self.student_seeds,
                      "recipes": self.recipes, "teacher_selection_metric": self.teacher_selection_metric,
                      "student_selection_metric": self.student_selection_metric,
                      "temperature": self.temperature, "kd_weight": self.kd_weight,
                      "warmup_epochs": self.warmup_epochs,
                      "repair": (self.repair.__dict__ if self.repair is not None else None)}
        if self.matrix_binding is not None:
            value["matrix_binding"] = self.matrix_binding
        if self.student_weights is not None:
            value["student_weights"] = self.student_weights
        if self.group_policy is not None:
            value["group_policy"] = self.group_policy
        return _json(value)


def load_spec(path):
    raw = json.loads(Path(path).read_text())
    raw["teacher"] = from_dict(raw["teacher"])
    if raw.get("repair") is not None:
        raw["repair"] = repair_options(raw["repair"])
    for name in ("teacher_seeds", "student_seeds", "recipes"):
        if name in raw:
            raw[name] = tuple(raw[name])
    spec = TeacherBaselineSpec(**raw)
    spec.validate()
    verify_matrix_binding(spec, config_path=path)
    return spec


def study_configs(spec: TeacherBaselineSpec, output):
    spec.validate()
    output = str(Path(output).resolve())
    teachers, students, supervised = [], [], []
    from .shortcut_baselines import mechanism_conditions
    for recipe in (*spec.recipes, *mechanism_conditions(spec)):
        for teacher_seed in spec.teacher_seeds:
            name = (f"teacher_repair_seed{teacher_seed}" if recipe == REPAIR_NAME
                    else f"teacher_{recipe}_seed{teacher_seed}")
            teacher = replace(spec.teacher, name=name, output_dir=output,
                              train=replace(spec.teacher.train, seed=teacher_seed))
            teachers.append((recipe, teacher))
            for student_seed in spec.student_seeds:
                student = replace(spec.teacher,
                    name=f"kd_{recipe}_teacher{teacher_seed}_seed{student_seed}", output_dir=output,
                    student=ModelConfig(spec.student_model, weights=spec.student_weights),
                    teacher=ModelConfig(spec.teacher.student.name, str(Path(output) / name / "student.pt")),
                    distillation=DistillationConfig(method="kd", temperature=spec.temperature,
                                                   weight=spec.kd_weight, warmup_epochs=spec.warmup_epochs),
                    train=replace(spec.teacher.train, seed=student_seed))
                student.validate()
                students.append((recipe, teacher_seed, student))
    for seed in spec.student_seeds:
        supervised.append(replace(spec.teacher, name=f"supervised_student_seed{seed}", output_dir=output,
                                  student=ModelConfig(spec.student_model, weights=spec.student_weights),
                                  train=replace(spec.teacher.train, seed=seed)))
    return teachers, students, supervised


def plan_study(spec, output):
    verify_matrix_binding(spec, output)
    teachers, students, supervised = study_configs(spec, output)
    repair_candidates = (len(spec.teacher_seeds) * len(spec.repair.channel_budgets)
                         * len(spec.repair.learning_rates) if spec.repair is not None else 0)
    counts = {"teachers": len(teachers), "kd_students": len(students),
              "supervised_students": len(supervised),
              "total_fits": len(teachers) + len(students) + len(supervised)}
    if spec.repair is not None:
        maximum = counts["total_fits"] - len(spec.teacher_seeds) + repair_candidates
        counts.update(teacher_training_fits=len(teachers) - len(spec.teacher_seeds),
                      repair_candidate_fits_max=repair_candidates,
                      total_fits=maximum, total_fits_max=maximum)
    from .shortcut_data import SHORTCUT_DATASETS
    shortcut = spec.teacher.data.source in SHORTCUT_DATASETS
    if shortcut:
        from .shortcut_baselines import shortcut_plan_fields
        return _json(shortcut_plan_fields(spec, output, teachers, students, supervised))
    return _json({"version": 2, "purpose": "Teacher performance and downstream distillability",
                  "spec": spec.to_dict(), "output": str(Path(output).resolve()),
                  "teacher_runs": [{"recipe": recipe, "config": config.to_dict()}
                                   for recipe, config in teachers],
                  "student_runs": [{"recipe": recipe, "teacher_seed": seed, "config": config.to_dict()}
                                   for recipe, seed, config in students],
                  "supervised_student_runs": [config.to_dict() for config in supervised],
                  "counts": counts,
                  "teacher_augmentation_policies": {
                      name: randaugment_policy(spec.teacher.data.source)
                      for name in spec.recipes if name != REPAIR_NAME and RECIPES[name].randaugment},
                  "repair_protocols": ([repair_spec(spec, seed).to_dict() for seed in spec.teacher_seeds]
                                       if spec.repair is not None else []),
                  "repair_rejection": "reuse control teacher and matched KD students; zero intervention effect",
                  "counts_note": "With repair enabled, teachers counts conditions; total_fits is an upper bound including all repair candidates. Rejected repairs reuse KD controls.",
                  "student_checkpoint_rule": "best validation accuracy (existing engine)",
                  "test_access": "explicit evaluate only, after all checkpoints and selection are frozen",
                  "source_sha256": {name: fingerprint(Path(__file__).parent / name)
                                    for name in STUDY_SOURCE_FILES}})


def _freeze_protocol(directory, protocol):
    path = directory / "protocol.json"
    if directory.exists() and any(directory.iterdir()) and not path.exists():
        raise FileExistsError(f"Nonempty output has no teacher-baseline protocol: {directory}")
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Study protocol changed; use a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_json(path, protocol)


def _check_student_summary(config, summary, *, require_bound_selection=False):
    """Validate teacher binding and the selected export before adopting a run."""
    directory = Path(config.output_dir) / config.name
    if read_json(directory / "config.json") != _json(config.to_dict()):
        raise ValueError(f"Student configuration changed: {directory}")
    if Path(summary["checkpoint"]).resolve() != (directory / "student.pt").resolve():
        raise ValueError("Student summary refers to a different checkpoint")
    expected_teacher = fingerprint(config.teacher.checkpoint) if config.distillation.method == "kd" else None
    if (summary.get("teacher") or {}).get("checkpoint_sha256") != expected_teacher:
        raise ValueError("Teacher checkpoint changed or student teacher digest is incorrect")
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    exported = torch.load(directory / "student.pt", map_location="cpu", weights_only=True)
    for state in (best, last):
        if (state.get("teacher_sha256") != expected_teacher
                or _json(state["config"]) != _json(config.to_dict())):
            raise ValueError("Student training checkpoint belongs to another configuration/teacher")
    if require_bound_selection or "selected_best_checkpoint_sha256" in last:
        # A verified historical completion manifest remains evidence for its
        # saved schema. Unsealed recovery must not bless legacy or foreign best
        # weights merely because their config and exported tensors agree.
        validate_training_data(last, summary.get("data"))
        validate_training_data(last, read_json(directory / "data.json"))
        initial = summary.get("initial_student_sha256")
        if (not isinstance(initial, str) or len(initial) != 64
                or any(character not in "0123456789abcdef" for character in initial)
                or any(state.get("initial_student_sha256") != initial for state in (best, last))):
            raise ValueError("Student initialization differs from its training checkpoints")
        best = load_selected_checkpoint(
            directory / "best.pt",
            metadata(config.student.name, last.get("classes", []),
                     config.data.image_size, config.data.source),
            summary.get("data"), last)
    if (exported.get("epoch") != best["epoch"] or summary["best_epoch"] != best["epoch"] + 1
            or tensor_state_fingerprint(exported["student"]) != tensor_state_fingerprint(best["student"])):
        raise ValueError("Student export differs from its selected best checkpoint")
    for key in ("format_version", "model_name", "classes", "preprocessing"):
        if exported.get(key) != best.get(key):
            raise ValueError("Student export metadata differs from best checkpoint")
    if exported["model_name"] != config.student.name:
        raise ValueError("Student export has the wrong architecture")


def _student_run(config, *, verify_only=False):
    directory = Path(config.output_dir) / config.name
    completion = directory / "baseline_completion.json"
    if completion.exists():
        required = (*STUDENT_FILES, "baseline_alias.json") if (directory / "baseline_alias.json").exists() else STUDENT_FILES
        verify_manifest(directory, "baseline_completion.json", required)
        summary = read_json(directory / "summary.json")
        _check_student_summary(config, summary)
        return summary
    if verify_only:
        raise ValueError(f"Missing student completion manifest: {completion}")
    summary_path = directory / "summary.json"
    if summary_path.exists():
        if json.loads((directory / "config.json").read_text()) != _json(config.to_dict()):
            raise ValueError("Unmanifested student configuration changed")
        summary = json.loads(summary_path.read_text())
    else:
        last = directory / "last.pt"
        summary = run_experiment(config, resume=str(last) if last.exists() else None)
    _check_student_summary(config, summary, require_bound_selection=True)
    required = (*STUDENT_FILES, "baseline_alias.json") if (directory / "baseline_alias.json").exists() else STUDENT_FILES
    write_manifest(directory, "baseline_completion.json", required)
    return summary


def _reuse_control_student(config, control_config):
    """Record an exact no-change arm without fitting a redundant student."""
    source = Path(control_config.output_dir) / control_config.name
    directory = Path(config.output_dir) / config.name
    original = _student_run(control_config, verify_only=True)
    equivalent = replace(control_config, name=config.name, output_dir=config.output_dir, teacher=config.teacher)
    if _json(equivalent.to_dict()) != _json(config.to_dict()):
        raise ValueError("No-change student differs from its matched control")
    if fingerprint(config.teacher.checkpoint) != fingerprint(control_config.teacher.checkpoint):
        raise ValueError("No-change repair does not retain the original teacher")
    identity = {"source_run": control_config.name,
                "source_completion_sha256": fingerprint(source / "baseline_completion.json")}
    if directory.exists() and any(directory.iterdir()):
        if not (directory / "baseline_alias.json").exists():
            raise FileExistsError(f"No-change output is not an alias run: {directory}")
        if (directory / "baseline_alias.json").exists() and read_json(directory / "baseline_alias.json") != identity:
            raise ValueError("No-change alias source changed")
        if (directory / "baseline_completion.json").exists():
            return _student_run(config, verify_only=True)
    # Existing manifest-verified legacy aliases remain readable, but creating a
    # new alias must not manufacture a missing selected-checkpoint binding.
    _check_student_summary(control_config, original, require_bound_selection=True)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "baseline_alias.json", identity)
    for name in ("student.pt", "data.json", "history.json"):
        shutil.copyfile(source / name, directory / name)
    # Rebase only location/name metadata; model, optimizer and RNG tensors remain identical.
    for name in ("best.pt", "last.pt"):
        state = torch.load(source / name, map_location="cpu", weights_only=True)
        state["config"] = config.to_dict()
        if name == "last.pt":
            # The verified best tensors are unchanged, but rebasing config
            # metadata changes the serialized file authenticated by last.pt.
            state["selected_best_checkpoint_sha256"] = fingerprint(directory / "best.pt")
        save_checkpoint(directory / name, state)
    write_json(directory / "config.json", config.to_dict())
    summary = {**original, "name": config.name, "checkpoint": str(directory / "student.pt"),
               "reused_from": control_config.name,
               "training": {**original["training"], "total_epoch_seconds": 0.0,
                            "mean_epoch_seconds": 0.0, "timing_scope": "no new training; exact matched control reuse"}}
    write_json(directory / "summary.json", summary)
    return _student_run(config)


def _target_classes(counts, *, long_tail=False):
    """Follow the declared profile, not the realized equality of class counts."""
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("Target selection requires positive training class counts")
    if not long_tail:
        return tuple(range(len(counts)))
    count = (len(counts) + 1) // 2
    return tuple(sorted(range(len(counts)), key=lambda label: (counts[label], label))[:count])


def _training_data_identity(provenance):
    """Compare rebuilt training/validation data without final-evaluation additions."""
    identity = _json(provenance)
    sizes = identity.get("split_sizes", {})
    sizes.pop("test", None)
    sizes.pop("confirmation", None)
    for key in ("test_index_sha256", "test_per_class", "test_policy",
                "confirmation_index_sha256", "confirmation_per_class", "confirmation_policy"):
        identity.pop(key, None)
    return identity


def _validate_checkpoint(config, split="val", output=None, *, targets=None):
    device = resolve_device(config.train.device)
    configure_accelerator(device, config.train.cuda_mode)
    data = build_data(config.data, replace(config.train, workers=0),
                      include_test=split == "test", diagnostic=True)
    recorded = read_json(Path(config.output_dir) / config.name / "summary.json")["data"]
    if _training_data_identity(data.provenance) != _training_data_identity(recorded):
        raise ValueError("Evaluation data differs from the frozen training/validation provenance")
    model = create_model(config.student.name, config.data.num_classes)
    load_model_checkpoint(model, Path(config.output_dir) / config.name / "student.pt",
                          metadata(config.student.name, data.classes, config.data.image_size, config.data.source))
    prepare_model(model, device, config.train.channels_last)
    labels, probabilities = collect_predictions(model, data.val if split == "val" else data.test,
                                               device, precision=config.train.precision,
                                               channels_last=config.train.channels_last)
    metrics = (save_prediction_report(Path(output), labels, probabilities, data.classes)
               if output is not None else prediction_metrics(labels, probabilities, data.classes))
    if targets is None:
        counts = data.provenance.get("train_per_class")
        if counts is None:
            counts = read_json(Path(config.output_dir) / config.name / "summary.json").get("class_counts")
        if counts is None:
            counts = [0] * config.data.num_classes
            for _, labels in data.train:
                for label, count in zip(*torch.unique(labels, return_counts=True)):
                    counts[int(label)] += int(count)
        targets = _target_classes(counts, long_tail=_effective_factor(config.data) > 1)
    metrics["target_classes"] = list(targets)
    metrics["target_macro_recall"] = statistics.mean(metrics["per_class"][i]["recall"] for i in targets)
    return metrics


def _seconds(config):
    summary = read_json(Path(config.output_dir) / config.name / "summary.json")
    return summary.get("training_seconds", (summary.get("training") or {}).get("total_epoch_seconds", 0.0))


def _aggregate(values):
    return {"values": list(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def _report(spec, teachers, students, supervised, *, split, output=None):
    rows, metrics = [], {}
    control = next(config for recipe, config in teachers if recipe == "control")
    counts = read_json(Path(control.output_dir) / control.name / "summary.json")["class_counts"]
    if len(counts) != spec.teacher.data.num_classes:
        raise ValueError("Training class counts do not match the class vocabulary")
    targets = (repair_targets(spec, counts) if spec.repair is not None else
               _target_classes(counts, long_tail=_effective_factor(spec.teacher.data) > 1))
    for recipe, config in teachers:
        metrics[config.name] = _validate_checkpoint(config, split, Path(output) / config.name if output else None,
                                                   targets=targets)
    for _, _, config in students:
        metrics[config.name] = _validate_checkpoint(config, split, Path(output) / config.name if output else None,
                                                   targets=targets)
    for config in supervised:
        metrics[config.name] = _validate_checkpoint(config, split, Path(output) / config.name if output else None,
                                                   targets=targets)
    supervised_metrics = {key: _aggregate([metrics[config.name][key] for config in supervised])
                          for key in ("accuracy", "balanced_accuracy", "target_macro_recall")}
    for recipe in spec.recipes:
        teacher_names = [config.name for name, config in teachers if name == recipe]
        student_names = [config.name for name, _, config in students if name == recipe]
        control_names = [f"kd_control_teacher{teacher_seed}_seed{config.train.seed}"
                         for name, teacher_seed, config in students if name == recipe]
        row = {"recipe": recipe,
               "teacher_seconds": _aggregate([_seconds(config) for name, config in teachers if name == recipe]),
               "student_seconds": _aggregate([_seconds(config) for name, _, config in students if name == recipe]),
               "teacher": {key: _aggregate([metrics[name][key] for name in teacher_names]) for key in
                           ("accuracy", "balanced_accuracy", "target_macro_recall", "nll", "brier_score", "ece_15_bins")},
               "student": {key: _aggregate([metrics[name][key] for name in student_names]) for key in
                           ("accuracy", "balanced_accuracy", "target_macro_recall", "nll", "brier_score", "ece_15_bins")},
               "paired_student_delta_vs_control": {key: _aggregate([
                   metrics[name][key] - metrics[control][key] for name, control in zip(student_names, control_names, strict=True)])
                   for key in ("accuracy", "balanced_accuracy", "target_macro_recall")}}
        if recipe == REPAIR_NAME:
            row["repair_outcomes"] = {
                str(config.train.seed): read_json(Path(config.output_dir) / config.name / "repair_decision.json")
                for name, config in teachers if name == recipe}
        # A crossed teacher/student grid is not a set of independent student seeds.
        row["student_by_teacher_seed"] = {
            str(seed): {key: _aggregate([metrics[config.name][key]
                for name, teacher_seed, config in students if name == recipe and teacher_seed == seed])
                for key in ("accuracy", "balanced_accuracy", "target_macro_recall")}
            for seed in spec.teacher_seeds}
        rows.append(row)
    return {"split": split, "rows": rows, "supervised_student": supervised_metrics,
            "target_classes": list(targets), "train_per_class": counts,
            "per_run": metrics, "synthetic_smoke_only": spec.teacher.data.source == "synthetic",
            "uncertainty_note": "Descriptive SD; crossed fits share teacher/student seeds. No significance claim."}


def _markdown(report):
    lines = ["# Teacher training and downstream KD", "", f"Evaluation split: {report['split']}.", "",
             "| Recipe | Teacher acc. | Teacher balanced acc. | KD acc. | KD balanced acc. | KD target recall | Paired KD acc. delta | Teacher seconds | KD seconds/fit |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        def display(group, key):
            value = row[group][key]
            return f"{100 * value['mean']:.2f} ± {100 * value['std']:.2f}"
        lines.append(f"| {row['recipe']} | {display('teacher', 'accuracy')} | "
                     f"{display('teacher', 'balanced_accuracy')} | {display('student', 'accuracy')} | "
                     f"{display('student', 'balanced_accuracy')} | {display('student', 'target_macro_recall')} | "
                     f"{100 * row['paired_student_delta_vs_control']['accuracy']['mean']:+.2f} pp | "
                     f"{row['teacher_seconds']['mean']:.1f} | {row['student_seconds']['mean']:.1f} |")
    lines.extend(["", "All values are percentages; variation is descriptive across matched fits.",
                  report["uncertainty_note"], ""])
    if report["synthetic_smoke_only"]:
        lines.append("Synthetic execution check only; these numbers do not establish a winning method.\n")
    return "\n".join(lines)


def _verify_runs(spec, teachers, students, supervised):
    """Verify every completed run without starting training or writing artifacts."""
    for recipe, config in teachers:
        directory = Path(config.output_dir) / config.name
        if recipe == REPAIR_NAME:
            verify_repair_details(directory)
            control = Path(config.output_dir) / f"teacher_control_seed{config.train.seed}"
            identity = read_json(directory / "teacher_protocol.json")
            if (identity["base_teacher_sha256"] != fingerprint(control / "student.pt")
                    or identity["base_completion_sha256"] != fingerprint(control / "completion.json")):
                raise ValueError("Repair arm no longer refers to its frozen control teacher")
        else:
            verify_manifest(directory, "completion.json", TEACHER_FILES)
        if read_json(directory / "config.json") != _json(config.to_dict()):
            raise ValueError("Completed teacher configuration changed")
        summary = read_json(directory / "summary.json")
        if (Path(summary["checkpoint"]).resolve() != (directory / "student.pt").resolve()
                or summary["checkpoint_sha256"] != fingerprint(directory / "student.pt")):
            raise ValueError("Completed teacher summary checkpoint changed")
    for _, _, config in students:
        _student_run(config, verify_only=True)
        directory = Path(config.output_dir) / config.name
        if (directory / "baseline_alias.json").exists():
            alias = read_json(directory / "baseline_alias.json")
            source = Path(config.output_dir) / alias["source_run"]
            if alias["source_completion_sha256"] != fingerprint(source / "baseline_completion.json"):
                raise ValueError("No-change student source manifest changed")
    for config in supervised:
        _student_run(config, verify_only=True)


STUDY_FILES = ("protocol.json", "validation.json", "validation.md", "selection.json")


def _study_files(teachers, students, supervised):
    return (*STUDY_FILES, *(f"{config.name}/completion.json" for _, config in teachers),
            *(f"{config.name}/baseline_completion.json" for _, _, config in students),
            *(f"{config.name}/baseline_completion.json" for config in supervised))


def _verify_selection(spec, directory, teachers, students, supervised):
    verify_manifest(directory, "study_completion.json", _study_files(teachers, students, supervised))
    selection = read_json(directory / "selection.json")
    if selection["protocol_sha256"] != fingerprint(directory / "protocol.json"):
        raise ValueError("Selection protocol changed")
    if selection["validation_sha256"] != fingerprint(directory / "validation.json"):
        raise ValueError("Validation results changed after selection")
    _check_selection_rule(spec, selection, read_json(directory / "validation.json"))
    expected = {config.name for config in [*(c for _, c in teachers), *(c for _, _, c in students), *supervised]}
    if set(selection["checkpoints"]) != expected:
        raise ValueError("Selection omits expected teacher/student checkpoints")
    for name, digest in selection["checkpoints"].items():
        if fingerprint(directory / name / "student.pt") != digest:
            raise ValueError("Checkpoint changed after validation selection")
    _verify_runs(spec, teachers, students, supervised)
    return selection


def _check_selection_rule(spec, selection, report):
    if (report["split"] != "val" or {row["recipe"] for row in report["rows"]} != set(spec.recipes)
            or len(report["rows"]) != len(spec.recipes)):
        raise ValueError("Validation report does not cover the prespecified conditions")
    # Apply the prespecified tie order, independently of a JSON file's row order.
    rows = {row["recipe"]: row for row in report["rows"]}
    selected = max(spec.recipes, key=lambda name: rows[name]["student"][spec.student_selection_metric]["mean"])
    if (selection["recipe"] != selected or selection["metric"] != spec.student_selection_metric
            or selection["selected_on"] != "validation only"):
        raise ValueError("Selection differs from the prespecified validation rule")


def run_study(spec, output, *, phase="train", dry_run=False):
    from .shortcut_data import SHORTCUT_DATASETS
    if spec.teacher.data.source in SHORTCUT_DATASETS:
        from .shortcut_baselines import run_shortcut_study
        return run_shortcut_study(spec, output, phase=phase, dry_run=dry_run)
    if phase not in {"plan", "train", "evaluate"}:
        raise ValueError("Unknown study phase")
    protocol = plan_study(spec, output)
    if dry_run:
        return {"dry_run": True, "phase": phase, "protocol": protocol}
    if phase == "train":
        # Resolve auto precision before writing a protocol or fitting earlier arms.
        device = resolve_device(spec.teacher.train.device)
        precision = resolve_precision(spec.teacher.train.precision, device)
        if precision == "float16" and any(
                name != REPAIR_NAME and RECIPES[name].sam_rho for name in spec.recipes):
            raise ValueError("SAM supports float32/bfloat16; set teacher precision explicitly")
    directory = Path(output).resolve()
    _freeze_protocol(directory, protocol)
    if phase == "plan":
        return {"status": "planned", "counts": protocol["counts"], "output": str(directory)}
    teachers, students, supervised = study_configs(spec, output)
    selection_path = directory / "selection.json"
    if phase == "evaluate":
        if not selection_path.exists():
            raise ValueError("Test access requires completed training and a frozen validation selection")
        selection = _verify_selection(spec, directory, teachers, students, supervised)
        report = _report(spec, teachers, students, supervised, split="test", output=directory / "test")
        report["validation_selected_recipe"] = selection["recipe"]
        report["interpretation"] = "All prespecified arms are reported; test results do not reselect the recipe."
        write_json(directory / "comparison.json", report)
        (directory / "report.md").write_text(_markdown(report))
        evaluated_files = tuple(path.relative_to(directory).as_posix() for path in (directory / "test").rglob("*") if path.is_file())
        write_manifest(directory, "evaluation_completion.json", ("comparison.json", "report.md", "study_completion.json", *evaluated_files))
        return {"status": "evaluated", "selection": selection["recipe"], "report": str(directory / "report.md")}
    if selection_path.exists():
        # A crash between atomic selection export and final manifest is recoverable:
        # verify all runs and the validation/checkpoint bindings before sealing it.
        if not (directory / "study_completion.json").exists():
            _verify_runs(spec, teachers, students, supervised)
            selection = read_json(selection_path)
            if (selection["protocol_sha256"] != fingerprint(directory / "protocol.json")
                    or selection["validation_sha256"] != fingerprint(directory / "validation.json")):
                raise ValueError("Unsealed selection identity changed")
            expected = {config.name: fingerprint(directory / config.name / "student.pt")
                        for config in [*(c for _, c in teachers), *(c for _, _, c in students), *supervised]}
            if selection["checkpoints"] != expected:
                raise ValueError("Unsealed selection checkpoint map changed")
            _check_selection_rule(spec, selection, read_json(directory / "validation.json"))
            write_manifest(directory, "study_completion.json", _study_files(teachers, students, supervised))
        selection = _verify_selection(spec, directory, teachers, students, supervised)
        return {"status": "trained", "selection": selection["recipe"], "output": str(directory)}
    teacher_initials, student_initials, data_identities = {}, {}, []
    for recipe, config in sorted(teachers, key=lambda row: row[0] == REPAIR_NAME):
        if recipe == REPAIR_NAME:
            summary = train_repair_teacher(spec, config)
        else:
            summary = train_teacher(config, RECIPES[recipe], selection_metric=spec.teacher_selection_metric)
        digest = summary["initial_student_sha256"]
        if teacher_initials.setdefault(config.train.seed, digest) != digest:
            raise ValueError("Teacher initialization differs across recipes")
        data_identities.append(summary["data"])
    for config in supervised:
        summary = _student_run(config)
        student_initials[config.train.seed] = summary["initial_student_sha256"]
        data_identities.append(summary["data"])
    # Run controls first, even when a user specifies a different recipe order.
    for recipe, teacher_seed, config in sorted(students, key=lambda row: row[0] != "control"):
        repair_directory = directory / f"teacher_repair_seed{teacher_seed}"
        no_change = (recipe == REPAIR_NAME
                     and read_json(repair_directory / "repair_decision.json")["status"] == "no_change")
        if no_change:
            control_config = next(c for name, seed, c in students if name == "control"
                                  and seed == teacher_seed and c.train.seed == config.train.seed)
            summary = _reuse_control_student(config, control_config)
        else:
            summary = _student_run(config)
        if summary["initial_student_sha256"] != student_initials[config.train.seed]:
            raise ValueError("Student initialization differs across arms")
        data_identities.append(summary["data"])
    for key in ("source", "train_index_sha256", "val_index_sha256", "train_per_class", "val_per_class", "recipe_sha256"):
        if any(row.get(key) != data_identities[0].get(key) for row in data_identities):
            raise ValueError(f"Dataset/split identity differs across arms: {key}")
    report = _report(spec, teachers, students, supervised, split="val")
    write_json(directory / "validation.json", report)
    (directory / "validation.md").write_text(_markdown(report))
    # Stable ties retain the earlier recipe; no teacher-quality prefilter hides transfer failures.
    selected = max(report["rows"], key=lambda row: row["student"][spec.student_selection_metric]["mean"])
    selection = {"recipe": selected["recipe"], "metric": spec.student_selection_metric,
                 "selected_on": "validation only", "protocol_sha256": fingerprint(directory / "protocol.json"),
                 "validation_sha256": fingerprint(directory / "validation.json"),
                 "checkpoints": {config.name: fingerprint(directory / config.name / "student.pt")
                                 for config in [*(config for _, config in teachers),
                                                *(config for _, _, config in students), *supervised]}}
    write_json(selection_path, selection)
    _verify_runs(spec, teachers, students, supervised)
    write_manifest(directory, "study_completion.json", _study_files(teachers, students, supervised))
    return {"status": "trained", "selection": selection["recipe"], "output": str(directory)}


def smoke_spec():
    return TeacherBaselineSpec(teacher=ExperimentConfig(
        name="teacher", data=DataConfig(image_size=16, train_samples=32, val_samples=16, test_samples=16),
        student=ModelConfig("tiny_medium"), distillation=DistillationConfig(method="supervised"),
        train=TrainConfig(epochs=1, batch_size=16, device="cpu", threads=2, precision="float32"),
        benchmark=BenchmarkConfig(warmup=0, iterations=1)), student_model="tiny_small", student_seeds=(42,))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "train", "evaluate", "smoke"))
    parser.add_argument("--config", help="Teacher-baseline study JSON")
    parser.add_argument("--output", required=True, help="Fresh study root (compatible runs resume)")
    parser.add_argument("--device", help="Override the base training device")
    parser.add_argument("--recipes", nargs="+", choices=(*RECIPES, REPAIR_NAME))
    parser.add_argument("--dry-run", action="store_true", help="Print the complete protocol without writes or dataset access")
    args = parser.parse_args(argv)
    if args.phase == "smoke":
        spec = smoke_spec()
    else:
        if not args.config:
            parser.error("--config is required outside smoke")
        spec = load_spec(args.config)
    if args.device:
        spec = replace(spec, teacher=replace(spec.teacher, train=replace(spec.teacher.train, device=args.device)))
    if args.recipes:
        spec = replace(spec, recipes=tuple(args.recipes),
                       repair=spec.repair if REPAIR_NAME in args.recipes else None)
    report = run_study(spec, args.output, phase="train" if args.phase == "smoke" else args.phase,
                       dry_run=args.dry_run)
    if args.phase == "smoke" and not args.dry_run:
        report = run_study(spec, args.output, phase="evaluate")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
