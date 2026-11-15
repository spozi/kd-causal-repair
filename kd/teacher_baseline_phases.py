"""Prespecified class-data baselines followed by repair of their exact CE teacher.

This opt-in workflow leaves ``teacher_baselines.run_study`` unchanged. It keeps
the baseline's frozen protocol and final evaluation intact, and writes the
combined baseline/repair comparison to a separate directory.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path

from .baseline_artifacts import read_json, verify_manifest, write_manifest
from .checkpoints import fingerprint, write_json
from .teacher_baselines import (
    _check_selection_rule, _markdown, _report, _reuse_control_student,
    _student_run, _training_data_identity, _verify_runs, _verify_selection,
    load_spec, plan_study, run_study, study_configs,
)
from .teacher_repair_baseline import REPAIR_NAME, train_repair_teacher


def _freeze(path, value):
    """Write an immutable JSON record, or verify the existing record."""
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen phased baseline record changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)


def _overlap(first, second):
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _plan(spec, output, repair_output):
    from .shortcut_data import SHORTCUT_DATASETS

    if spec.matrix_binding is not None:
        raise ValueError("Phased repair does not support matrix-bound specs; use a new standalone configuration")
    if spec.teacher.data.source in SHORTCUT_DATASETS:
        raise ValueError("Phased repair currently supports class datasets, not shortcut datasets")
    spec.validate()
    if spec.repair is None or REPAIR_NAME not in spec.recipes:
        raise ValueError("Phased repair requires explicit teacher_repair options and condition")
    output, repair_output = Path(output).resolve(), Path(repair_output).resolve()
    data_root = Path(spec.teacher.data.root).resolve()
    if _overlap(output, repair_output):
        raise ValueError("Baseline and repair output paths must not overlap")
    if any(_overlap(path, data_root) for path in (output, repair_output)):
        raise ValueError("Output paths must not overlap the dataset root")
    baseline = replace(spec, recipes=tuple(name for name in spec.recipes if name != REPAIR_NAME), repair=None)
    baseline_plan, combined_plan = plan_study(baseline, output), plan_study(spec, output)
    baseline_fits = baseline_plan["counts"]["total_fits"]
    maximum = combined_plan["counts"]["total_fits_max"]
    workflow = {
        "version": 1, "purpose": "Complete baselines before matched CE-teacher repair",
        "spec": spec.to_dict(), "baseline_output": str(output), "repair_output": str(repair_output),
        "workflow": ["baseline_train", "baseline_evaluate", "repair_and_matched_kd", "combined_evaluate"],
        "baseline_protocol": baseline_plan, "combined_plan": combined_plan,
        "source_sha256": {"teacher_baseline_phases.py": fingerprint(Path(__file__)),
                          **combined_plan["source_sha256"]},
        "counts": {"baseline_fits": baseline_fits, "repair_additional_fits_max": maximum - baseline_fits,
                   "total_fits_max": maximum},
        "selection": "Validation only; frozen repair settings are not changed after baseline test access",
        "reuse": "All baseline teachers, KD students and supervised students are reused without refitting",
    }
    return baseline, output, repair_output, workflow


def _verify_parent(baseline, output, expected_protocol):
    if read_json(output / "protocol.json") != expected_protocol:
        raise ValueError("Completed baseline protocol differs from the prespecified phased workflow")
    teachers, students, supervised = study_configs(baseline, output)
    _verify_selection(baseline, output, teachers, students, supervised)
    verify_manifest(output, "evaluation_completion.json", ("comparison.json", "report.md", "study_completion.json"))


def _check_matched_students(students, supervised, output):
    references = {config.train.seed: read_json(output / config.name / "summary.json")["initial_student_sha256"]
                  for config in supervised}
    identities = []
    for config in [*(config for _, _, config in students), *supervised]:
        summary = read_json(output / config.name / "summary.json")
        if summary["initial_student_sha256"] != references[config.train.seed]:
            raise ValueError("Student initialization differs across baseline/repair arms")
        identities.append(_training_data_identity(summary["data"]))
    if any(identity != identities[0] for identity in identities):
        raise ValueError("Baseline and repair training-data identities differ")


def _verify_extension_selection(spec, output, repair_output, configs):
    selection = read_json(repair_output / "selection.json")
    if (selection["protocol_sha256"] != fingerprint(repair_output / "protocol.json")
            or selection["validation_sha256"] != fingerprint(repair_output / "validation.json")):
        raise ValueError("Repair extension selection bindings changed")
    _check_selection_rule(spec, selection, read_json(repair_output / "validation.json"))
    if set(selection["checkpoints"]) != {config.name for config in configs}:
        raise ValueError("Repair extension selection omits expected checkpoints")
    for config in configs:
        if fingerprint(output / config.name / "student.pt") != selection["checkpoints"][config.name]:
            raise ValueError("Selected checkpoint changed after validation")
    return selection


def _extend(spec, baseline, output, repair_output, workflow, *, parent_directory=None, parent_verifier=None):
    parent_directory = output if parent_directory is None else Path(parent_directory)
    verifier = _verify_parent if parent_verifier is None else parent_verifier
    verifier(baseline, output, workflow["baseline_protocol"])
    parent_hash = fingerprint(parent_directory / "protocol.json")
    _freeze(repair_output / "protocol.json", {
        "version": 1, "workflow_sha256": fingerprint(repair_output / "workflow.json"),
        "baseline_protocol_sha256": parent_hash,
        "baseline_evaluation_sha256": fingerprint(parent_directory / "evaluation_completion.json"),
        "spec": spec.to_dict(), "selection": workflow["selection"], "reuse": workflow["reuse"],
    })
    teachers, students, supervised = study_configs(spec, output)
    configs = [*(config for _, config in teachers), *(config for _, _, config in students), *supervised]
    sealed = ("workflow.json", "protocol.json", "validation.json", "validation.md", "selection.json")
    if not (repair_output / "training_completion.json").exists():
        for recipe, config in teachers:
            if recipe == REPAIR_NAME:
                train_repair_teacher(spec, config)
        for recipe, teacher_seed, config in students:
            if recipe != REPAIR_NAME:
                _student_run(config, verify_only=True)
                continue
            decision = read_json(output / f"teacher_repair_seed{teacher_seed}" / "repair_decision.json")
            control = next(other for name, seed, other in students if name == "control"
                           and seed == teacher_seed and other.train.seed == config.train.seed)
            if decision["status"] == "no_change":
                _reuse_control_student(config, control)
            else:
                _student_run(config)
        _verify_runs(spec, teachers, students, supervised)
        _check_matched_students(students, supervised, output)
        validation = _report(spec, teachers, students, supervised, split="val")
        _freeze(repair_output / "validation.json", validation)
        markdown = _markdown(validation)
        markdown_path = repair_output / "validation.md"
        if markdown_path.exists() and markdown_path.read_text() != markdown:
            raise ValueError("Frozen validation markdown changed")
        if not markdown_path.exists():
            markdown_path.write_text(markdown)
        rows = {row["recipe"]: row for row in validation["rows"]}
        chosen = max(spec.recipes, key=lambda name: rows[name]["student"][spec.student_selection_metric]["mean"])
        _freeze(repair_output / "selection.json", {
            "recipe": chosen, "selected_on": "validation only", "metric": spec.student_selection_metric,
            "protocol_sha256": fingerprint(repair_output / "protocol.json"),
            "validation_sha256": fingerprint(repair_output / "validation.json"),
            "checkpoints": {config.name: fingerprint(output / config.name / "student.pt") for config in configs},
        })
        _verify_extension_selection(spec, output, repair_output, configs)
        write_manifest(repair_output, "training_completion.json", sealed)
    verify_manifest(repair_output, "training_completion.json", sealed)
    _verify_runs(spec, teachers, students, supervised)
    _check_matched_students(students, supervised, output)
    selection = _verify_extension_selection(spec, output, repair_output, configs)
    if fingerprint(parent_directory / "protocol.json") != parent_hash:
        raise ValueError("Baseline protocol changed during repair extension")
    if (repair_output / "evaluation_completion.json").exists():
        verify_manifest(repair_output, "evaluation_completion.json", ("comparison.json", "report.md", "training_completion.json"))
        return read_json(repair_output / "comparison.json")
    report = _report(spec, teachers, students, supervised, split="test", output=repair_output / "test")
    report["validation_selected_recipe"] = selection["recipe"]
    report["interpretation"] = workflow["reuse"] + "; " + workflow["selection"]
    write_json(repair_output / "comparison.json", report)
    (repair_output / "report.md").write_text(_markdown(report))
    test_files = [path.relative_to(repair_output).as_posix() for path in (repair_output / "test").rglob("*") if path.is_file()]
    write_manifest(repair_output, "evaluation_completion.json", ("comparison.json", "report.md", "training_completion.json", *test_files))
    return report


def run_baseline_then_repair(spec, output, *, repair_output, callback=None, dry_run=False):
    """Run class baselines, evaluate them, then fit only repair and its new KD arm.

    ``spec`` contains every baseline recipe and ``teacher_repair`` together.
    The full settings and implementation identity are frozen before any fit.
    Results/predictions remain separate; selected models share ``output`` so
    repair can bind to the exact baseline CE checkpoint. Completed evaluation
    is verified rather than regenerated on resume. No checkpoint pruning is
    performed, and previously pruned studies cannot be resumed here.
    """
    baseline, output, repair_output, workflow = _plan(spec, output, repair_output)
    if dry_run:
        return {"dry_run": True, "workflow": workflow}
    # Reject incompatible/non-protocol output directories before creating records.
    if output.exists() and any(output.iterdir()):
        if not (output / "protocol.json").exists():
            raise FileExistsError(f"Nonempty baseline output has no protocol: {output}")
        if read_json(output / "protocol.json") != workflow["baseline_protocol"]:
            raise ValueError("Baseline protocol differs from the prespecified phased workflow")
        if not (repair_output / "workflow.json").exists():
            raise ValueError("Cannot adopt an existing baseline without a prespecified phased workflow")
    if repair_output.exists() and any(repair_output.iterdir()) and not (repair_output / "workflow.json").exists():
        raise FileExistsError(f"Nonempty repair output has no phased workflow: {repair_output}")
    _freeze(repair_output / "workflow.json", workflow)
    notify = callback if callback is not None else lambda phase: None
    if (output / "evaluation_completion.json").exists():
        _verify_parent(baseline, output, workflow["baseline_protocol"])
        notify("baseline_reuse_verified")
    else:
        for phase in ("train", "evaluate"):
            notify("baseline_" + phase)
            run_study(baseline, output, phase=phase)
    notify("repair_and_matched_kd")
    return _extend(spec, baseline, output, repair_output, workflow)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repair-output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_baseline_then_repair(load_spec(args.config), args.output,
        repair_output=args.repair_output, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
