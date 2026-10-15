"""Preflight, execute/resume, and summarize a frozen 16-dataset matrix plan."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from .baseline_artifacts import read_json, verify_manifest
from .checkpoints import fingerprint, write_json
from .data import build_data
from .teacher_baselines import load_spec, plan_study, run_study, study_configs


def load_jobs(plan):
    directory = Path(plan).expanduser().resolve()
    protocol = read_json(directory / "protocol.json")
    if protocol.get("version") != 2 or protocol.get("output") != str(directory):
        raise ValueError("Full benchmark requires a source-bound version-2 matrix plan")
    jobs = [(job, load_spec(job["config"])) for job in protocol["jobs"]]
    if len(jobs) != protocol["job_count"] or len({job["id"] for job, _ in jobs}) != len(jobs):
        raise ValueError("Matrix plan job coverage is inconsistent")
    return protocol, jobs


def preflight(plan):
    """Validate every local dataset before any model/weights/training are started.

    Existing class adapters may create/verify their deterministic split cache;
    preflight does not download data or pretrained weights or write study outputs.
    """
    protocol, jobs = load_jobs(plan)
    checked, failures, seen, warnings = {}, [], set(), []
    for job, spec in jobs:
        source = spec.teacher.data.source
        contract = (source, spec.teacher.data.preprocessing, spec.teacher.data.image_size)
        if contract in seen:
            continue
        seen.add(contract)
        disclosure = job.get("study_policy", {}).get("pretraining_overlap")
        if disclosure and disclosure["imagenet_derived_dataset"]:
            warnings.append({"dataset": source, **disclosure})
        try:
            data = build_data(replace(spec.teacher.data, download=False),
                              replace(spec.teacher.train, workers=0, device="cpu"),
                              include_test=False, diagnostic=True)
            image, label = data.train.dataset[0]
            if tuple(image.shape) != (3, spec.teacher.data.image_size, spec.teacher.data.image_size):
                raise ValueError("Adapter returned a different input shape than its frozen model contract")
            outcome = {"status": "ready", "split_sizes": data.provenance["split_sizes"],
                               "train_per_class": data.provenance.get("train_per_class"),
                               "train_per_group": data.provenance.get("train_per_group"),
                               "input_shape": list(image.shape), "label": int(label)}
        except (ValueError, RuntimeError, OSError) as error:
            outcome = {"status": "not_ready", "reason": str(error)}
            if source not in failures:
                failures.append(source)
        outcome["preprocessing"] = spec.teacher.data.preprocessing
        if source not in checked:
            checked[source] = {**outcome, "input_contracts": []}
        checked[source]["input_contracts"].append(outcome)
        if outcome["status"] == "not_ready":
            checked[source].update(status="not_ready", reason=outcome["reason"])
    return {"status": "ready" if not failures else "not_ready", "datasets": checked,
            "failed_datasets": failures, "job_count": protocol["job_count"], "warnings": warnings,
            "weight_policy": "ImageNet weights are verified/retrieved by model initialization during explicit training, not preflight"}


def execute(plan, *, phase="train", job_ids=None):
    if phase not in {"train", "evaluate"}:
        raise ValueError("Execution phase must be train or evaluate")
    protocol, jobs = load_jobs(plan)
    if job_ids is not None:
        requested = set(job_ids)
        if len(requested) != len(job_ids) or not requested.issubset({job["id"] for job, _ in jobs}):
            raise ValueError("Job selection must name unique jobs in the frozen plan")
        jobs = [(job, spec) for job, spec in jobs if job["id"] in requested]
    if phase == "train":
        readiness = preflight(plan)
        if readiness["status"] != "ready":
            raise ValueError("Dataset preflight failed before training: " + json.dumps(readiness["datasets"]))
    outcomes = []
    for job, spec in jobs:
        print(f"{phase}: {job['id']}", flush=True)
        outcomes.append({"id": job["id"], **run_study(spec, job["output"], phase=phase)})
    return {"status": "complete", "phase": phase, "selected_jobs": len(jobs),
            "planned_jobs": protocol["job_count"], "outcomes": outcomes}


def _report_rows(comparison, endpoints, recipes, mechanisms, metric, winner):
    """Require explicit endpoint/arm coverage; never fall back to another endpoint."""
    endpoint_reports = comparison.get("endpoint_reports")
    if endpoint_reports is None:
        if endpoints != ["test"]:
            raise ValueError("Comparison lacks the prespecified endpoint reports")
        endpoint_reports = {"test": comparison}
    if set(endpoint_reports) != set(endpoints):
        raise ValueError("Comparison endpoint coverage differs from its frozen plan")
    result = {}
    for endpoint, report in endpoint_reports.items():
        if (report["split"] != "test" or report["validation_selected_recipe"] != winner
                or (endpoint != "test" and report["endpoint"] != endpoint)):
            raise ValueError("Endpoint report changed its split, identity or frozen winner")
        rows = report["rows"]
        methods = {item["recipe"]: item for item in rows if item.get("role", "recipe") == "recipe"}
        controls = {item["recipe"] for item in rows if item.get("role") == "mechanism_control"}
        if set(methods) != set(recipes) or controls != set(mechanisms) or len(rows) != len(recipes) + len(mechanisms):
            raise ValueError("Comparison recipe/mechanism coverage differs from its frozen plan")
        for item in methods.values():
            for value in (item["student"][metric], item["paired_student_delta_vs_control"][metric]):
                mean, std = value["mean"], value["std"]
                if mean is None and std is None:
                    continue  # Missing required groups remain explicitly unavailable.
                if (not isinstance(mean, (int, float)) or not isinstance(std, (int, float))
                        or not math.isfinite(mean) or not math.isfinite(std) or std < 0):
                    raise ValueError("Comparison contains invalid aggregate measurements")
        result[endpoint] = methods
    return result


def summarize(plan, output):
    protocol, jobs = load_jobs(plan)
    rows = []
    for job, spec in jobs:
        directory = Path(job["output"])
        endpoints = job.get("study_policy", {}).get("test_endpoints", ["test"])
        comparison_path = directory / "comparison.json"
        comparison = None
        method_reports = {}
        status, reason = "pending", "No final evaluation report"
        if comparison_path.exists():
            try:
                # Final reports are sealed separately from pre-test selections.
                verify_manifest(directory, "evaluation_completion.json", ("comparison.json", "report.md", "study_completion.json"))
                if read_json(directory / "protocol.json") != plan_study(spec, directory):
                    raise ValueError("Evaluated study does not match its frozen matrix job")
                verify_manifest(directory, "study_completion.json", ("protocol.json", "selection.json", "validation.json"))
                teachers, students, supervised = study_configs(spec, directory)
                if spec.group_policy is not None:
                    from .shortcut_baselines import _verify_selection
                    eligibility = read_json(directory / "eligibility.json")
                    omitted = set(eligibility["omitted"])
                    teachers = [(name, config) for name, config in teachers if name not in omitted]
                    students = [(name, seed, config) for name, seed, config in students if name not in omitted]
                    selection = _verify_selection(spec, directory, teachers, students, supervised, eligibility)
                else:
                    from .teacher_baselines import _verify_selection
                    selection = _verify_selection(spec, directory, teachers, students, supervised)
                comparison = read_json(comparison_path)
                if comparison.get("split") != "test" or comparison.get("validation_selected_recipe") != selection["recipe"]:
                    raise ValueError("Final comparison is not a test report with its validation-selected winner")
                recipes = list(dict.fromkeys(name for name, _ in teachers if name in spec.recipes))
                mechanisms = list(dict.fromkeys(name for name, _ in teachers if name not in spec.recipes))
                method_reports = _report_rows(comparison, endpoints, recipes, mechanisms,
                                               spec.student_selection_metric, selection["recipe"])
                status, reason = "complete", None
            except (ValueError, OSError, KeyError, TypeError) as error:
                comparison = None
                status, reason = "invalid", str(error)
        for endpoint in endpoints:
            metric = spec.student_selection_metric
            row = {"job_id": job["id"], "dataset": job["dataset"], "profile": job["profile"],
                   "pair": job["pair"], "endpoint": endpoint, "primary_metric": metric,
                   "status": status, "reason": reason, "control_kd": None, "repaired_kd": None,
                   "validation_selected_recipe": None, "selected_kd": None,
                   "paired_repair_delta": None, "paired_selected_delta": None,
                   "recipe_coverage": {"completed": 0, "eligible": len(spec.recipes)}, "mechanism_coverage": None}
            row["input_protocol"] = job.get("study_policy", {}).get("input_protocol")
            row["pretraining_overlap"] = job.get("study_policy", {}).get("pretraining_overlap")
            if comparison is not None:
                method_rows = method_reports[endpoint]
                chosen = comparison["validation_selected_recipe"]
                if chosen not in method_rows or "control" not in method_rows:
                    raise ValueError("Comparison lacks control or validation-selected recipe")
                row.update(control_kd=method_rows["control"]["student"][metric],
                    repaired_kd=method_rows.get("teacher_repair", {}).get("student", {}).get(metric),
                    validation_selected_recipe=chosen, selected_kd=method_rows[chosen]["student"][metric],
                    paired_repair_delta=method_rows.get("teacher_repair", {}).get("paired_student_delta_vs_control", {}).get(metric),
                    paired_selected_delta=method_rows[chosen]["paired_student_delta_vs_control"][metric],
                    recipe_coverage={"completed": len(method_rows), "eligible": len(recipes)},
                    mechanism_coverage={"completed": len(mechanisms), "eligible": len(mechanisms)})
                if row["selected_kd"]["mean"] is None:
                    row.update(status="unavailable", reason="Required group endpoint unavailable")
            rows.append(row)
    result = {"protocol_sha256": fingerprint(Path(protocol["output"]) / "protocol.json"),
              "datasets": protocol["datasets"], "job_count": protocol["job_count"], "endpoint_rows": len(rows),
              "rows": rows, "aggregate_policy": "Dataset/profile/pair/metric-specific; no mixed Acc/BAcc/WGA pooled average"}
    target = Path(output).expanduser().resolve()
    if target.is_relative_to(Path(protocol["output"])) or any(target == Path(job["output"]) or target.is_relative_to(Path(job["output"])) for job, _ in jobs):
        raise ValueError("Summary output must be separate from the immutable plan and individual studies")
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "overview.json", result)
    lines = ["# Full benchmark overview", "", "| Dataset / profile / endpoint | Pair | Metric | Control KD | Repaired KD | Selected recipe | Selected KD | Paired selected delta (pp) | Coverage / status |",
             "|---|---|---|---:|---:|---|---:|---:|---|"]
    def display(value):
        return "—" if value is None or value.get("mean") is None else f"{100 * value['mean']:.2f} ± {100 * value['std']:.2f}"
    for row in rows:
        coverage = row["recipe_coverage"]
        lines.append(f"| {row['dataset']} / {row['profile']} / {row['endpoint']} | {row['pair']} | {row['primary_metric']} | "
            f"{display(row['control_kd'])} | {display(row['repaired_kd'])} | {row['validation_selected_recipe'] or '—'} | "
            f"{display(row['selected_kd'])} | {display(row['paired_selected_delta'])} | {coverage['completed']}/{coverage['eligible']} / {row['status']} |")
    lines.extend(["", result["aggregate_policy"], "Missing/invalid/unavailable outcomes are not zero gains.", ""])
    flagged = sorted({row["dataset"] for row in rows if row["pretraining_overlap"] and row["pretraining_overlap"]["imagenet_derived_dataset"]})
    if flagged:
        lines.extend(["Pretraining-overlap warning: " + ", ".join(flagged)
                      + " are ImageNet-derived; overlap with ImageNet-1K pretraining has not been audited. Do not present these endpoints as pretraining-disjoint.", ""])
    (target / "overview.md").write_text("\n".join(lines))
    return {"status": "reported", "endpoint_rows": len(rows), "output": str(target)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "train", "evaluate", "report"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--output", help="Separate report output directory")
    args = parser.parse_args(argv)
    if args.phase == "preflight":
        result = preflight(args.plan)
    elif args.phase == "report":
        if not args.output:
            parser.error("report requires --output")
        result = summarize(args.plan, args.output)
    else:
        result = execute(args.plan, phase=args.phase, job_ids=args.jobs)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("status") == "not_ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
