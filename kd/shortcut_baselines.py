"""Group-aware execution of the unified teacher baseline study.

Ordinary training uses the common engine/teacher recipes. Group repair and its
mechanism controls have isolated artifacts, without rewriting historical studies.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .baseline_artifacts import TEACHER_FILES, read_json, verify_manifest, write_manifest
from .checkpoints import fingerprint, load_model_checkpoint, metadata, write_json
from .data import build_data
from .evaluation import collect_predictions, prediction_metrics
from .group_metrics import add_group_metrics, dataset_metadata, selection_key
from .models import create_model
from .runtime import configure_accelerator, prepare_model, resolve_device, resolve_precision
from .training_provenance import training_provenance


MECHANISM_CONDITIONS = ("repair_ce_continuation", "repair_random_channels", "repair_attribution_only")
DEFAULT_GROUP_POLICY = {"version": 1, "seed": 2026,
    "overall_accuracy_guardrail": 0.005, "group_accuracy_guardrail": 0.005,
    "bootstrap_repetitions": 20, "stability_threshold": 0.7, "top_fraction": 0.25,
    "screen_limit": 32, "preservation_per_group": 64, "localization_batch_size": 2,
    "mechanism_controls": True}
DISCORDANT_GROUPS = {"colored_mnist": (1, 2), "waterbirds": (1, 2),
                     "multiceleba": (0, 2, 3, 4, 5, 7), "urbancars": (1, 2, 3, 4, 5, 6)}


def mechanism_conditions(spec):
    return MECHANISM_CONDITIONS if spec.group_policy and spec.group_policy["mechanism_controls"] else ()


def validate_shortcut_spec(spec):
    if spec.teacher.data.source == "multiceleba" and (spec.repair is not None or "teacher_repair" in spec.recipes):
        raise ValueError("MultiCelebA currently supports baseline/KD only; facial-attribute repair is not validated")
    policy = spec.group_policy
    if (not isinstance(policy, dict) or set(policy) != set(DEFAULT_GROUP_POLICY)
            or type(policy.get("version")) is not int or policy["version"] != 1):
        raise ValueError("Shortcut studies require a complete version-1 frozen group_policy")
    for key in ("seed", "bootstrap_repetitions", "screen_limit", "preservation_per_group", "localization_batch_size"):
        if type(policy[key]) is not int or policy[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid group_policy.{key}")
    for key in ("overall_accuracy_guardrail", "group_accuracy_guardrail", "stability_threshold", "top_fraction"):
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid group_policy.{key}")
    if not policy["top_fraction"] or type(policy["mechanism_controls"]) is not bool:
        raise ValueError("Invalid group policy fraction or controls flag")
    if policy["mechanism_controls"] and spec.repair is None:
        raise ValueError("Mechanism controls require explicit repair options")
    if (spec.teacher_selection_metric != "worst_group_accuracy"
            or spec.student_selection_metric != "worst_group_accuracy"
            or spec.teacher.train.selection_metric != "worst_group_accuracy"):
        raise ValueError("Shortcut teacher/student selection must use validation worst-group accuracy")
    if spec.teacher.student.name not in {"convnet6_w32", "convnet6_w16", "cifar_teacher", "cifar_student", "resnet18", "resnet34", "resnet50"}:
        raise ValueError("Shortcut teachers require supported ordinary-stem/ConvNet repair maps")


def shortcut_plan_fields(spec, output, teachers, students, supervised):
    from .matrix_binding import STUDY_SOURCE_FILES
    from .teacher_training import RECIPES, randaugment_policy
    repair_conditions = 1 + len(mechanism_conditions(spec)) if spec.repair else 0
    candidates = (repair_conditions * len(spec.teacher_seeds) * len(spec.repair.channel_budgets)
                  * len(spec.repair.learning_rates)) if spec.repair else 0
    ordinary = len([1 for condition, _ in teachers if condition in spec.recipes and condition != "teacher_repair"])
    total = ordinary + candidates + len(students) + len(supervised)
    return {"version": 2, "purpose": "Group-aware teacher performance and downstream distillability",
        "spec": spec.to_dict(), "output": str(Path(output).resolve()), "evaluation_family": "shortcut",
        "teacher_runs": [{"recipe": name, "config": config.to_dict()} for name, config in teachers],
        "student_runs": [{"recipe": name, "teacher_seed": seed, "config": config.to_dict()} for name, seed, config in students],
        "supervised_student_runs": [config.to_dict() for config in supervised],
        "counts": {"teachers": len(teachers), "teacher_training_fits": ordinary,
                   "repair_candidate_fits_max": candidates, "kd_students": len(students),
                   "supervised_students": len(supervised), "total_fits": total, "total_fits_max": total},
        "group_policy": spec.group_policy, "target_groups": list(DISCORDANT_GROUPS[spec.teacher.data.source]),
        "mechanism_conditions": list(mechanism_conditions(spec)),
        "teacher_augmentation_policies": {name: randaugment_policy(spec.teacher.data.source)
            for name in spec.recipes if name != "teacher_repair" and RECIPES[name].randaugment},
        "eligibility_rule": "Omit Balanced Softmax before fitting when actual training class counts are exactly equal",
        "student_checkpoint_rule": "best validation WGA, then raw accuracy; earliest exact tie",
        "test_endpoints": ["reversed", "unbiased"] if spec.teacher.data.source == "colored_mnist" else ["released"],
        "repair_rejection": "verified control-equivalent aliases, not independent fits",
        "counts_note": "Upper bounds; conditional recipe omissions and rejected/infeasible repairs reduce fitting. Mechanism controls are not winner candidates.",
        "test_access": "explicit evaluate after checkpoints, eligibility and validation selection are frozen",
        "source_sha256": {name: fingerprint(Path(__file__).parent / name) for name in STUDY_SOURCE_FILES}}


def _aggregate(values):
    values = list(values)
    if any(value is None for value in values):
        return {"values": values, "mean": None, "std": None, "available": False,
                "reason": "Required group endpoint unavailable in one or more runs"}
    return {"values": values, "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0, "available": True}


def _evaluate_run(config, split, endpoint=None, output=None):
    device = resolve_device(config.train.device)
    configure_accelerator(device, config.train.cuda_mode)
    data = build_data(config.data, replace(config.train, workers=0), include_test=split == "test", diagnostic=True)
    summary = read_json(Path(config.output_dir) / config.name / "summary.json")
    if training_provenance(data.provenance) != training_provenance(summary["data"]):
        raise ValueError("Shortcut evaluation data differs from frozen training/validation identity")
    loader = data.val if split == "val" else data.test_endpoints[endpoint]
    model = create_model(config.student.name, config.data.num_classes)
    load_model_checkpoint(model, Path(config.output_dir) / config.name / "student.pt",
                          metadata(config.student.name, data.classes, config.data.image_size, config.data.source))
    prepare_model(model, device, config.train.channels_last)
    labels, probabilities = collect_predictions(model, loader, device, precision=config.train.precision,
                                                channels_last=config.train.channels_last)
    result = add_group_metrics(prediction_metrics(labels, probabilities, data.classes), labels, probabilities, loader)
    clean = dataset_metadata(loader.dataset, "clean_targets")
    if clean is not None:
        result["clean_digit_category_accuracy"] = float(np.mean(probabilities.argmax(1) == clean))
    if config.data.source == "urbancars":
        from .urban_metrics import urban_metrics
        result.update(urban_metrics(labels, probabilities.argmax(1), dataset_metadata(loader.dataset, "nuisance_labels"),
                                    data.provenance["train_per_group"]))
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "predictions.npz", labels=labels, probabilities=probabilities,
            groups=dataset_metadata(loader.dataset, "group_ids"),
            nuisance=dataset_metadata(loader.dataset, "nuisance_labels"),
            source_ids=dataset_metadata(loader.dataset, "source_ids"), classes=np.asarray(data.classes))
        write_json(output / "metrics.json", result)
    return result


def _report(spec, teachers, students, supervised, eligibility, *, split, endpoint=None, output=None):
    conditions = (*eligibility["recipes"], *mechanism_conditions(spec))
    metrics = {config.name: _evaluate_run(config, split, endpoint,
        Path(output) / config.name if output is not None else None)
        for config in [*(config for _, config in teachers), *(config for _, _, config in students), *supervised]}
    keys = ("accuracy", "balanced_accuracy", "worst_group_accuracy", "group_balanced_accuracy", "nll", "brier_score", "ece_15_bins")
    rows = []
    for condition in conditions:
        teacher_names = [config.name for name, config in teachers if name == condition]
        selected_students = [(seed, config) for name, seed, config in students if name == condition]
        row = {"recipe": condition, "role": "mechanism_control" if condition in MECHANISM_CONDITIONS else "recipe",
            "teacher": {key: _aggregate(metrics[name][key] for name in teacher_names) for key in keys},
            "student": {key: _aggregate(metrics[config.name][key] for _, config in selected_students) for key in keys},
            "paired_student_delta_vs_control": {key: _aggregate(
                None if metrics[config.name][key] is None or metrics[f"kd_control_teacher{seed}_seed{config.train.seed}"][key] is None else
                metrics[config.name][key] - metrics[f"kd_control_teacher{seed}_seed{config.train.seed}"][key]
                for seed, config in selected_students) for key in ("accuracy", "worst_group_accuracy", "group_balanced_accuracy")},
            "student_by_teacher_seed": {str(seed): {key: _aggregate(metrics[config.name][key]
                for teacher_seed, config in selected_students if teacher_seed == seed) for key in keys} for seed in spec.teacher_seeds}}
        if condition == "teacher_repair" or condition in MECHANISM_CONDITIONS:
            row["repair_outcomes"] = {str(config.train.seed): read_json(Path(config.output_dir) / config.name / "repair_decision.json")
                                       for name, config in teachers if name == condition}
        rows.append(row)
    return {"split": split, "endpoint": endpoint, "evaluation_family": "shortcut", "primary_metric": "worst_group_accuracy",
            "rows": rows, "per_run": metrics, "omitted_recipes": eligibility["omitted"],
            "supervised_student": {key: _aggregate(metrics[config.name][key] for config in supervised) for key in keys},
            "uncertainty_note": "Descriptive paired student SD conditional on each teacher; no significance/teacher-seed uncertainty claim."}


def _markdown(report):
    lines = ["# Shortcut teacher quality and downstream KD", "", f"Split: {report['split']}; endpoint: {report['endpoint']}.", "",
             "| Condition | Teacher raw Acc | Teacher WGA | Student raw Acc | Student WGA | Paired WGA delta (pp) |",
             "|---|---:|---:|---:|---:|---:|"]
    def value(group):
        return "Unavailable" if group["mean"] is None else f"{100 * group['mean']:.2f} ± {100 * group['std']:.2f}"
    for row in report["rows"]:
        lines.append(f"| {row['recipe']} | {value(row['teacher']['accuracy'])} | {value(row['teacher']['worst_group_accuracy'])} | "
                     f"{value(row['student']['accuracy'])} | {value(row['student']['worst_group_accuracy'])} | {value(row['paired_student_delta_vs_control']['worst_group_accuracy'])} |")
    lines.extend(["", "Scores are percentages; deltas are percentage points.", report["uncertainty_note"],
                  "Omitted conditions: " + json.dumps(report["omitted_recipes"]), ""])
    return "\n".join(lines)


def _files(teachers, students, supervised):
    return ("protocol.json", "eligibility.json", "validation.json", "validation.md", "selection.json",
        *(f"{config.name}/completion.json" for _, config in teachers),
        *(f"{config.name}/baseline_completion.json" for _, _, config in students),
        *(f"{config.name}/baseline_completion.json" for config in supervised))


def _verify_runs(teachers, students, supervised):
    from .teacher_baselines import _student_run
    from .shortcut_repair import GROUP_REPAIR_FILES
    for condition, config in teachers:
        directory = Path(config.output_dir) / config.name
        repair = condition == "teacher_repair" or condition in MECHANISM_CONDITIONS
        verify_manifest(directory, "completion.json", GROUP_REPAIR_FILES if repair else TEACHER_FILES)
        if repair:
            verify_manifest(directory / "repair", "report_manifest.json", ("protocol.json", "localization.json"))
        if read_json(directory / "config.json") != json.loads(json.dumps(config.to_dict())):
            raise ValueError("Shortcut teacher config changed")
        summary = read_json(directory / "summary.json")
        if (Path(summary["checkpoint"]).resolve() != (directory / "student.pt").resolve()
                or summary["checkpoint_sha256"] != fingerprint(directory / "student.pt")):
            raise ValueError("Shortcut teacher export changed")
        if repair:
            control = directory.parent / f"teacher_control_seed{config.train.seed}"
            identity = read_json(directory / "teacher_protocol.json")
            if (identity["base_teacher_sha256"] != fingerprint(control / "student.pt")
                    or identity["base_completion_sha256"] != fingerprint(control / "completion.json")):
                raise ValueError("Shortcut repair control identity changed")
    for config in [*(config for _, _, config in students), *supervised]:
        _student_run(config, verify_only=True)
        alias = Path(config.output_dir) / config.name / "baseline_alias.json"
        if alias.exists():
            recorded = read_json(alias)
            if recorded["source_completion_sha256"] != fingerprint(Path(config.output_dir) / recorded["source_run"] / "baseline_completion.json"):
                raise ValueError("Shortcut student control alias changed")


def _verify_selection(spec, directory, teachers, students, supervised, eligibility):
    verify_manifest(directory, "study_completion.json", _files(teachers, students, supervised))
    return _check_selection(spec, directory, teachers, students, supervised, eligibility)


def _check_selection(spec, directory, teachers, students, supervised, eligibility):
    selection = read_json(directory / "selection.json")
    for field, filename in (("protocol_sha256", "protocol.json"), ("validation_sha256", "validation.json"), ("eligibility_sha256", "eligibility.json")):
        if selection[field] != fingerprint(directory / filename):
            raise ValueError("Shortcut frozen selection/protocol changed")
    report = read_json(directory / "validation.json")
    rows = {row["recipe"]: row for row in report["rows"] if row["role"] == "recipe"}
    if (set(rows) != set(eligibility["recipes"]) or report["split"] != "val"
            or len(report["rows"]) != len(eligibility["recipes"]) + len(mechanism_conditions(spec))):
        raise ValueError("Shortcut selection lacks complete validation recipe coverage")
    winner = max(eligibility["recipes"], key=lambda name: rows[name]["student"]["worst_group_accuracy"]["mean"])
    expected = {config.name: fingerprint(directory / config.name / "student.pt")
                for config in [*(config for _, config in teachers), *(config for _, _, config in students), *supervised]}
    if (selection["recipe"] != winner or selection["checkpoints"] != expected
            or selection["metric"] != "worst_group_accuracy" or selection["selected_on"] != "validation only"):
        raise ValueError("Shortcut checkpoint map or validation-only winner changed")
    _verify_runs(teachers, students, supervised)
    return selection


def run_shortcut_study(spec, output, *, phase="train", dry_run=False):
    from .teacher_baselines import (_freeze_protocol, _reuse_control_student, _student_run,
                                   plan_study, study_configs)
    from .teacher_training import RECIPES, train_teacher
    from .shortcut_repair import train_shortcut_repair
    if phase not in {"plan", "train", "evaluate"}:
        raise ValueError("Unknown shortcut study phase")
    protocol = plan_study(spec, output)
    if dry_run:
        return {"dry_run": True, "phase": phase, "protocol": protocol}
    if phase == "train" and resolve_precision(spec.teacher.train.precision, resolve_device(spec.teacher.train.device)) == "float16" and "sam" in spec.recipes:
        raise ValueError("SAM requires float32/bfloat16")
    directory = Path(output).resolve()
    if phase == "evaluate" and not (directory / "study_completion.json").exists():
        raise ValueError("Test access requires a completed group study and frozen validation selection")
    _freeze_protocol(directory, protocol)
    if phase == "plan":
        return {"status": "planned", "counts": protocol["counts"], "output": str(directory)}
    data = build_data(spec.teacher.data, replace(spec.teacher.train, workers=0), include_test=False, diagnostic=True)
    counts = data.provenance["train_per_class"]
    omitted = {"balanced_softmax": "Exact equal training class counts: CE-equivalent, not independent evidence"} if "balanced_softmax" in spec.recipes and len(set(counts)) == 1 else {}
    eligibility = {"recipes": [name for name in spec.recipes if name not in omitted], "omitted": omitted,
                   "train_per_class": counts, "training_provenance": training_provenance(data.provenance)}
    path = directory / "eligibility.json"
    if path.exists() and read_json(path) != eligibility:
        raise ValueError("Shortcut training data or pre-fit recipe eligibility changed")
    if not path.exists():
        if phase == "evaluate":
            raise ValueError("Missing frozen pre-fit recipe eligibility")
        write_json(path, eligibility)
    teachers, students, supervised = study_configs(spec, output)
    teachers = [(name, config) for name, config in teachers if name not in omitted]
    students = [(name, seed, config) for name, seed, config in students if name not in omitted]
    if phase == "evaluate":
        selection = _verify_selection(spec, directory, teachers, students, supervised, eligibility)
        endpoints = protocol["test_endpoints"]
        reports = {endpoint: _report(spec, teachers, students, supervised, eligibility,
            split="test", endpoint=endpoint, output=directory / "test" / endpoint) for endpoint in endpoints}
        for endpoint, report in reports.items():
            report["validation_selected_recipe"] = selection["recipe"]
            (directory / "test" / endpoint / "report.md").write_text(_markdown(report))
        comparison = {**reports[endpoints[0]], "endpoint_reports": reports,
                      "validation_selected_recipe": selection["recipe"]}
        if spec.teacher.data.source == "colored_mnist":
            comparison["paired_nuisance_checks"] = _paired_color_checks(directory, reports)
        write_json(directory / "comparison.json", comparison)
        (directory / "report.md").write_text("\n".join(_markdown(report) for report in reports.values()))
        evaluated_files = tuple(path.relative_to(directory).as_posix() for path in (directory / "test").rglob("*") if path.is_file())
        write_manifest(directory, "evaluation_completion.json", ("comparison.json", "report.md", "study_completion.json", *evaluated_files))
        return {"status": "evaluated", "selection": selection["recipe"], "report": str(directory / "report.md")}
    if (directory / "selection.json").exists():
        if not (directory / "study_completion.json").exists():
            _verify_runs(teachers, students, supervised)
            selection = read_json(directory / "selection.json")
            if any(selection[field] != fingerprint(directory / filename) for field, filename in (
                    ("protocol_sha256", "protocol.json"), ("validation_sha256", "validation.json"), ("eligibility_sha256", "eligibility.json"))):
                raise ValueError("Unsealed shortcut selection identity changed")
            expected = {config.name: fingerprint(directory / config.name / "student.pt") for config in [*(c for _, c in teachers), *(c for _, _, c in students), *supervised]}
            if selection["checkpoints"] != expected:
                raise ValueError("Unsealed shortcut selected checkpoint map changed")
            _check_selection(spec, directory, teachers, students, supervised, eligibility)
            write_manifest(directory, "study_completion.json", _files(teachers, students, supervised))
        selection = _verify_selection(spec, directory, teachers, students, supervised, eligibility)
        return {"status": "trained", "selection": selection["recipe"], "output": str(directory)}
    initial_teachers, initial_students = {}, {}
    for condition, config in sorted(teachers, key=lambda row: row[0] not in RECIPES):
        summary = (train_teacher(config, RECIPES[condition], selection_metric="worst_group_accuracy")
                   if condition in RECIPES else train_shortcut_repair(spec, condition, config))
        if initial_teachers.setdefault(config.train.seed, summary["initial_student_sha256"]) != summary["initial_student_sha256"]:
            raise ValueError("Shortcut teacher initialization differs across conditions")
        if training_provenance(summary["data"]) != eligibility["training_provenance"]:
            raise ValueError("Shortcut teacher data differs across conditions")
    for config in supervised:
        summary = _student_run(config)
        initial_students[config.train.seed] = summary["initial_student_sha256"]
    for condition, seed, config in sorted(students, key=lambda row: row[0] != "control"):
        teacher_directory = directory / next(c.name for name, c in teachers if name == condition and c.train.seed == seed)
        no_change = (condition == "teacher_repair" or condition in MECHANISM_CONDITIONS) and read_json(teacher_directory / "repair_decision.json")["status"] == "no_change"
        control = next(c for name, teacher_seed, c in students if name == "control" and teacher_seed == seed and c.train.seed == config.train.seed)
        summary = _reuse_control_student(config, control) if no_change else _student_run(config)
        if summary["initial_student_sha256"] != initial_students[config.train.seed] or training_provenance(summary["data"]) != eligibility["training_provenance"]:
            raise ValueError("Shortcut student initialization/data differs across arms")
    report = _report(spec, teachers, students, supervised, eligibility, split="val")
    write_json(directory / "validation.json", report)
    (directory / "validation.md").write_text(_markdown(report))
    rows = {row["recipe"]: row for row in report["rows"] if row["role"] == "recipe"}
    winner = max(eligibility["recipes"], key=lambda name: rows[name]["student"]["worst_group_accuracy"]["mean"])
    selection = {"recipe": winner, "metric": "worst_group_accuracy", "selected_on": "validation only",
        "protocol_sha256": fingerprint(directory / "protocol.json"), "validation_sha256": fingerprint(directory / "validation.json"),
        "eligibility_sha256": fingerprint(path), "checkpoints": {config.name: fingerprint(directory / config.name / "student.pt")
            for config in [*(config for _, config in teachers), *(config for _, _, config in students), *supervised]}}
    write_json(directory / "selection.json", selection)
    _verify_runs(teachers, students, supervised)
    write_manifest(directory, "study_completion.json", _files(teachers, students, supervised))
    _verify_selection(spec, directory, teachers, students, supervised, eligibility)
    return {"status": "trained", "selection": winner, "output": str(directory)}


def _paired_color_checks(directory, reports):
    result = {}
    for name in reports["reversed"]["per_run"]:
        with np.load(directory / "test" / "reversed" / name / "predictions.npz", allow_pickle=False) as left, \
             np.load(directory / "test" / "unbiased" / name / "predictions.npz", allow_pickle=False) as right:
            if not np.array_equal(left["source_ids"], right["source_ids"]) or not np.array_equal(left["labels"], right["labels"]):
                raise ValueError("Colored MNIST nuisance endpoints lost foreground/label pairing")
            changed = left["nuisance"][:, 0] != right["nuisance"][:, 0]
            pred_left, pred_right = left["probabilities"].argmax(1), right["probabilities"].argmax(1)
            result[name] = {"paired_images": len(changed), "nuisance_changed": int(changed.sum()),
                "flip_rate_on_changed": float(np.mean(pred_left[changed] != pred_right[changed])) if changed.any() else None,
                "reversed_accuracy_on_changed": float(np.mean(pred_left[changed] == left["labels"][changed])) if changed.any() else None,
                "unbiased_accuracy_on_changed": float(np.mean(pred_right[changed] == right["labels"][changed])) if changed.any() else None,
                "interpretation": "Same digits/observed labels; correlated endpoint renderings, not independent replications"}
    return result
