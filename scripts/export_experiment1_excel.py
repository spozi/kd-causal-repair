#!/usr/bin/env python3
"""Export the Experiment 1 metadata bundle as a filterable Excel workbook."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side


DATASET_ORDER = (
    "cinic10", "pathmnist", "svhn", "fashionmnist", "organamnist", "cifar100",
    "cifar10", "gtsrb", "eurosat", "bloodmnist", "caltech101", "dermamnist", "stl10",
)
PROFILE_ORDER = ("balanced", "lt-if10", "lt-if50", "lt-if100")
CORE_METRICS = (
    "samples", "accuracy", "macro_f1", "balanced_accuracy", "nll", "brier_score",
    "ece_15_bins", "mean_confidence", "target_macro_recall", "target_balanced_nll",
    "tail_macro_recall", "tail_balanced_nll",
)
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)
THIN_GRAY = Side(style="thin", color="D9E2F3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True,
                        help="Extracted runs/experiment1-multidataset directory")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_cell(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (list, dict, tuple)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    value = str(value)
    return value if len(value) <= 32767 else value[:32740] + "...[truncated]"


def metric_values(metrics: dict | None, prefix: str = "") -> dict:
    metrics = metrics or {}
    values = {f"{prefix}{name}": metrics.get(name) for name in CORE_METRICS}
    extended = metrics.get("extended") or {}
    for name in ("aurc", "ece_l2_15_bins", "classwise_ece_15_bins",
                 "ece_equal_mass_15_bins", "selective_accuracy_80", "selective_accuracy_90"):
        values[f"{prefix}{name}"] = extended.get(name)
    return values


def study_identity(path: Path, root: Path) -> tuple[str | None, str | None]:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None, None
    if len(parts) >= 2 and parts[0] in DATASET_ORDER and parts[1] in PROFILE_ORDER:
        return parts[0], parts[1]
    return None, None


def add_table_sheet(workbook: Workbook, title: str, rows: list[dict], *, freeze: str = "A2"):
    sheet = workbook.create_sheet(title)
    headers = list(rows[0]) if rows else ["message"]
    if not rows:
        rows = [{"message": "No records"}]
    sheet.append(headers)
    for row in rows:
        sheet.append([safe_cell(row.get(header)) for header in headers])
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=THIN_GRAY)
    sheet.freeze_panes = freeze
    sheet.auto_filter.ref = sheet.dimensions
    for column in sheet.columns:
        letter = column[0].column_letter
        width = max(len(str(cell.value or "")) for cell in column[: min(len(column), 250)]) + 2
        sheet.column_dimensions[letter].width = min(max(width, 10), 42)
        for cell in column[1:]:
            if isinstance(cell.value, float):
                cell.number_format = "0.0000"
    return sheet


def add_compatible_sheets(workbook: Workbook, title: str, rows: list[dict],
                          *, maximum_rows: int = 100_000) -> None:
    """Keep worksheets modest and avoid formal table XML for broad Excel compatibility."""
    if len(rows) <= maximum_rows:
        add_table_sheet(workbook, title, rows)
        return
    for index, start in enumerate(range(0, len(rows), maximum_rows), start=1):
        chunk_title = f"{title[:27]} {index}"
        add_table_sheet(workbook, chunk_title, rows[start:start + maximum_rows])


def comparison_rows(summary: dict) -> list[dict]:
    rows = []
    for job in summary["jobs"]:
        rows.append({
            "job_id": job["id"], "dataset": job["dataset"], "profile": job["profile"],
            "status": job["status"], "terminal": job["status"] != "missing",
            "transfer_success": job.get("success"),
            "mean_accuracy_delta": job.get("mean_accuracy_delta"),
            "mean_target_recall_delta": job.get("mean_target_recall_delta"),
            "reason": job.get("reason"),
        })
    return rows


def rollup(rows: list[dict], field: str, order: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[field]].append(row)
    result = []
    for key in order:
        group = grouped[key]
        complete = [row for row in group if row["status"] == "complete"]
        acc = [row["mean_accuracy_delta"] for row in complete]
        recall = [row["mean_target_recall_delta"] for row in complete]
        result.append({
            field: key, "planned_jobs": len(group),
            "terminal_jobs": sum(row["terminal"] for row in group),
            "complete_comparisons": len(complete),
            "no_repair_selected": sum(row["status"] == "no_repair_selected" for row in group),
            "successful_transfers": sum(row["transfer_success"] is True for row in group),
            "mean_accuracy_delta": statistics.fmean(acc) if acc else None,
            "mean_target_recall_delta": statistics.fmean(recall) if recall else None,
        })
    return result


def active_paths(root: Path, name: str) -> list[Path]:
    return sorted(path for path in root.rglob(name)
                  if "_archive" not in path.parts
                  and not any(part.startswith("matrix-plan-") for part in path.parts))


def model_context(path: Path, root: Path) -> dict:
    dataset, profile = study_identity(path, root)
    parts = path.relative_to(root).parts
    role = variant = seed = None
    if "teacher_evaluation" in parts:
        role = "teacher"
        index = parts.index("teacher_evaluation")
        variant = parts[index + 1]
    elif "student_evaluation" in parts:
        role = "student"
        index = parts.index("student_evaluation")
        seed = int(parts[index + 1].removeprefix("seed"))
        variant = parts[index + 2]
    return {"dataset": dataset, "profile": profile, "role": role,
            "seed": seed, "variant": variant, "source_path": str(path.relative_to(root))}


def workbook_data(root: Path, archive: Path) -> dict[str, list[dict]]:
    summary = read_json(root / "matrix_summary.json")
    matrix = comparison_rows(summary)
    result: dict[str, list[dict]] = {
        "Matrix Results": matrix,
        "Dataset Rollup": rollup(matrix, "dataset", DATASET_ORDER),
        "Profile Rollup": rollup(matrix, "profile", PROFILE_ORDER),
        "Successful Transfers": [row for row in matrix if row["transfer_success"] is True],
        "No Repair": [row for row in matrix if row["status"] == "no_repair_selected"],
    }

    protocols = []
    for job in summary["jobs"]:
        config = read_json(root / "matrix-plan" / "configs" / f"{job['id']}.json")
        row = {"job_id": job["id"], "dataset": job["dataset"], "profile": job["profile"]}
        for key, value in config.items():
            row[key] = value
        protocols.append(row)
    result["Study Protocols"] = protocols

    teachers, students = [], []
    for path in active_paths(root, "comparison.json"):
        dataset, profile = study_identity(path, root)
        comparison = read_json(path)
        if comparison.get("status") != "complete":
            continue
        teacher = comparison["teacher"]
        teacher_row = {"job_id": comparison["study"]["name"], "dataset": dataset,
                       "profile": profile, "evaluation_split": comparison["evaluation_split"]}
        teacher_row.update(metric_values(teacher["baseline"], "baseline_"))
        teacher_row.update(metric_values(teacher["candidate"], "candidate_"))
        teacher_row.update({
            "accuracy_delta": teacher["accuracy"].get("accuracy_delta"),
            "accuracy_ci_low": (teacher["accuracy"].get("paired_bootstrap_95") or [None, None])[0],
            "accuracy_ci_high": (teacher["accuracy"].get("paired_bootstrap_95") or [None, None])[1],
            "mcnemar_p": teacher["accuracy"].get("mcnemar_exact_two_sided_p"),
            "ece_delta": teacher["calibration"].get("ece_delta"),
            "nll_delta": teacher["calibration"].get("nll_delta"),
            "brier_delta": teacher["calibration"].get("brier_delta"),
        })
        teachers.append(teacher_row)
        for student in comparison["students"]:
            row = {"job_id": comparison["study"]["name"], "dataset": dataset,
                   "profile": profile, "seed": student["seed"]}
            row.update(metric_values(student["baseline"], "baseline_"))
            row.update(metric_values(student["candidate"], "candidate_"))
            row.update({
                "accuracy_delta": student.get("accuracy_delta"),
                "target_recall_delta": student.get("target_recall_delta"),
                "tail_recall_delta": student.get("tail_recall_delta"),
                "accuracy_ci_low": (student["paired_accuracy"].get("paired_bootstrap_95")
                                    or [None, None])[0],
                "accuracy_ci_high": (student["paired_accuracy"].get("paired_bootstrap_95")
                                     or [None, None])[1],
                "mcnemar_p": student["paired_accuracy"].get("mcnemar_exact_two_sided_p"),
                "ece_delta": student["paired_calibration"].get("ece_delta"),
                "nll_delta": student["paired_calibration"].get("nll_delta"),
                "brier_delta": student["paired_calibration"].get("brier_delta"),
            })
            students.append(row)
    result["Teacher Comparisons"] = teachers
    result["Student Comparisons"] = students

    evaluations, per_class, target_labels, calibration_bins = [], [], [], []
    confusions = []
    for path in active_paths(root, "metrics.json"):
        context = model_context(path, root)
        metrics = read_json(path)
        evaluations.append({**context, **metric_values(metrics)})
        for row in metrics.get("per_class", []):
            per_class.append({**context, **row})
        labels = [row.get("class") for row in metrics.get("per_class", [])]
        for true_index, counts in enumerate(metrics.get("confusion_matrix", [])):
            for predicted_index, count in enumerate(counts):
                if not count:
                    continue
                confusions.append({**context, "true_class": labels[true_index],
                                   "predicted_class": labels[predicted_index],
                                   "count": count, "correct": true_index == predicted_index})
        recalls = metrics.get("target_recall_by_label", {})
        nlls = metrics.get("target_nll_by_label", {})
        for label in sorted(set(recalls) | set(nlls), key=lambda value: int(value)):
            target_labels.append({**context, "class_label": int(label),
                                  "recall": recalls.get(label), "nll": nlls.get(label)})
        extended = metrics.get("extended") or {}
        for kind, key in (("equal_width", "equal_width_bins"),
                          ("equal_mass", "equal_mass_bins")):
            for row in extended.get(key, []):
                calibration_bins.append({**context, "binning": kind, **row})
    result["Model Evaluations"] = evaluations
    result["Per Class Metrics"] = per_class
    result["Confusion Matrix"] = confusions
    result["Target Label Metrics"] = target_labels
    result["Calibration Bins"] = calibration_bins

    selections, selected_channels, candidates = [], [], []
    for path in active_paths(root, "teacher_selection.json"):
        dataset, profile = study_identity(path, root)
        data = read_json(path)
        selected = data.get("selected") or {}
        baseline = data.get("baseline_validation") or {}
        chosen = selected.get("validation") or {}
        row = {"job_id": f"{dataset}-{profile}", "dataset": dataset, "profile": profile,
               "eligible_count": data.get("eligible_count"), "selected": bool(selected),
               "budget": selected.get("budget"), "learning_rate": selected.get("learning_rate"),
               "baseline_accuracy": baseline.get("accuracy"),
               "selected_accuracy": chosen.get("accuracy"),
               "baseline_target_recall": baseline.get("target_macro_recall"),
               "selected_target_recall": chosen.get("target_macro_recall"),
               "checkpoint_sha256": selected.get("checkpoint_sha256")}
        selections.append(row)
        for channel in selected.get("channels", []):
            selected_channels.append({"job_id": f"{dataset}-{profile}", "dataset": dataset,
                                      "profile": profile, **channel})
    for path in active_paths(root, "candidate_results.json"):
        dataset, profile = study_identity(path, root)
        for index, candidate in enumerate(read_json(path), start=1):
            validation = candidate.get("validation") or {}
            candidates.append({
                "job_id": f"{dataset}-{profile}", "dataset": dataset, "profile": profile,
                "candidate_index": index, "budget": candidate.get("budget"),
                "learning_rate": candidate.get("learning_rate"),
                "channels": candidate.get("channels"), "accuracy": validation.get("accuracy"),
                "target_macro_recall": validation.get("target_macro_recall"),
                "target_balanced_nll": validation.get("target_balanced_nll"),
                "ece_15_bins": validation.get("ece_15_bins"),
                "checkpoint_sha256": candidate.get("checkpoint_sha256"),
            })
    result["Repair Selections"] = selections
    result["Selected Channels"] = selected_channels
    result["Repair Candidates"] = candidates

    localization, causal, consensus = [], [], []
    targets, companions = [], []
    for path in active_paths(root, "localization.json"):
        dataset, profile = study_identity(path, root)
        for rank, channel in enumerate(read_json(path)["payload"].get("ranking", []), start=1):
            localization.append({"job_id": f"{dataset}-{profile}", "dataset": dataset,
                                 "profile": profile, "rank": rank, **channel})
    for path in active_paths(root, "causal_ablation.json"):
        dataset, profile = study_identity(path, root)
        payload = read_json(path)["payload"]
        for channel in payload.get("channels", []):
            causal.append({"job_id": f"{dataset}-{profile}", "dataset": dataset,
                           "profile": profile, **channel})
    for path in active_paths(root, "consensus_scores.json"):
        dataset, profile = study_identity(path, root)
        payload = read_json(path)["payload"]
        for channel in payload.get("channels", []):
            consensus.append({
                "job_id": f"{dataset}-{profile}", "dataset": dataset, "profile": profile,
                "stage": channel.get("stage"), "channel": channel.get("channel"),
                "consensus_score": channel.get("consensus_score"),
                "bootstrap_stability": channel.get("bootstrap_stability"),
                "eligible": channel.get("eligible"),
            })
    for path in active_paths(root, "targets.json"):
        dataset, profile = study_identity(path, root)
        payload = read_json(path)["payload"]
        for target in payload.get("targets", []):
            targets.append({"job_id": f"{dataset}-{profile}", "dataset": dataset,
                            "profile": profile, **target})
    for path in active_paths(root, "companions.json"):
        dataset, profile = study_identity(path, root)
        for group in read_json(path)["payload"]:
            target = group["target"]
            for rank, companion in enumerate(group.get("companions", []), start=1):
                companions.append({
                    "job_id": f"{dataset}-{profile}", "dataset": dataset, "profile": profile,
                    "target_index": target.get("index"), "target_label": target.get("true_label"),
                    "target_reason": target.get("selection_reason"),
                    "policy": group.get("companion_policy"), "rank": rank, **companion,
                })
    result["Localization Ranking"] = localization
    result["Causal Channels"] = causal
    result["Consensus Channels"] = consensus
    result["Target Examples"] = targets
    result["Companions"] = companions

    runs, history, splits = [], [], {}
    for path in active_paths(root, "summary.json"):
        dataset, profile = study_identity(path, root)
        if dataset is None:
            continue
        data = read_json(path)
        validation = data.get("validation") or {}
        runtime = data.get("runtime") or {}
        inference = data.get("inference") or {}
        training = data.get("training") or {}
        runs.append({
            "dataset": dataset, "profile": profile, "name": data.get("name"),
            "method": data.get("method"), "seed": data.get("seed"),
            "best_epoch": data.get("best_epoch"), "validation_accuracy": validation.get("accuracy"),
            "validation_nll": validation.get("nll"), "validation_ece": validation.get("ece_15_bins"),
            "parameters": inference.get("parameters"), "estimated_flops": inference.get("estimated_flops"),
            "latency_median_ms": inference.get("latency_median_ms"),
            "latency_p95_ms": inference.get("latency_p95_ms"),
            "mean_epoch_seconds": training.get("mean_epoch_seconds"),
            "total_epoch_seconds": training.get("total_epoch_seconds"),
            "gpu": runtime.get("gpu_name"), "precision": runtime.get("precision"),
            "source_path": str(path.relative_to(root)),
        })
        provenance = data.get("data") or {}
        splits.setdefault((dataset, profile), {
            "dataset": dataset, "profile": profile,
            "dataset_version": provenance.get("dataset_version"),
            "split_seed": provenance.get("split_seed"),
            "split_sizes": provenance.get("split_sizes"),
            "train_per_class": provenance.get("train_per_class"),
            "val_per_class": provenance.get("val_per_class"),
            "test_per_class": provenance.get("test_per_class"),
            "train_index_sha256": provenance.get("train_index_sha256"),
            "val_index_sha256": provenance.get("val_index_sha256"),
            "test_index_sha256": provenance.get("test_index_sha256"),
        })
    for path in active_paths(root, "history.json"):
        dataset, profile = study_identity(path, root)
        if dataset is None:
            continue
        records = read_json(path)
        if not isinstance(records, list):
            continue
        run_name = path.parent.name
        for record in records:
            train = record.get("train") or {}
            val = record.get("val") or {}
            history.append({
                "dataset": dataset, "profile": profile, "run": run_name,
                "epoch": record.get("epoch"), "learning_rate": record.get("learning_rate"),
                "train_total": train.get("total", record.get("total")),
                "train_ce": train.get("ce", record.get("ce")),
                "train_supervised": train.get("supervised"),
                "train_response": train.get("response"),
                "train_features": train.get("features", record.get("features")),
                "preservation_ce": record.get("preservation_ce"), "kd": record.get("kd"),
                "train_accuracy": train.get("accuracy"), "train_samples": train.get("samples"),
                "epoch_seconds": train.get("seconds"),
                "val_accuracy": val.get("accuracy"), "val_nll": val.get("nll"),
                "val_ece": val.get("ece_15_bins"),
                "source_path": str(path.relative_to(root)),
            })
    result["Run Summaries"] = runs
    result["Training History"] = history
    result["Dataset Splits"] = list(splits.values())

    environments = []
    for path in active_paths(root, "environment.json"):
        dataset, profile = study_identity(path, root)
        environments.append({"dataset": dataset, "profile": profile, **read_json(path)})
    result["Environments"] = environments

    listing = subprocess.run(["tar", "--zstd", "-tf", str(archive)], check=True,
                             capture_output=True, text=True).stdout.splitlines()
    artifacts = []
    prefix = "runs/experiment1-multidataset/"
    for item in listing:
        if not item or item.endswith("/"):
            continue
        parts = item.split("/")
        dataset = profile = None
        if item.startswith(prefix):
            relative = item[len(prefix):].split("/")
            if len(relative) >= 2 and relative[0] in DATASET_ORDER and relative[1] in PROFILE_ORDER:
                dataset, profile = relative[:2]
        name = Path(item).name
        artifacts.append({
            "path": item, "dataset": dataset, "profile": profile,
            "archived_provenance": "/_archive/" in item,
            "extension": Path(name).suffix.lower(), "filename": name,
        })
    result["Artifact Index"] = artifacts
    return result


def add_overview(workbook: Workbook, root: Path, archive: Path, data: dict[str, list[dict]]) -> None:
    summary = read_json(root / "matrix_summary.json")
    sheet = workbook.active
    sheet.title = "Overview"
    sheet["A1"] = "Experiment 1: All 52 Results"
    sheet["A1"].font = Font(size=20, bold=True, color="1F4E78")
    details = [
        ("Generated UTC", datetime.now(timezone.utc).isoformat()),
        ("Source archive", archive.name),
        ("Archive SHA-256", sha256(archive)),
        # Rewritten 2026-10-02 to drop co-author trailers; the archive records the old
        # hash, which docs/history-rewrite.md maps to this one with the same tree.
        ("Source revision", "059d6296434466615b2abe7d44a82e095592696c"),
        ("Source revision as recorded", "89992ca1495ab357283613db30556fea44b2622d"),
        ("Matrix protocol SHA-256", summary["matrix_protocol_sha256"]),
        ("Planned jobs", summary["job_count"]),
        ("Terminal jobs", sum(row["terminal"] for row in data["Matrix Results"])),
        ("Complete comparisons", summary["complete_count"]),
        ("No repair selected", len(data["No Repair"])),
        ("Successful transfers", summary["successful_transfer_count"]),
        ("All jobs finished", summary["all_jobs_finished"]),
        ("Artifact files indexed", len(data["Artifact Index"])),
    ]
    for row, (label, value) in enumerate(details, start=3):
        sheet.cell(row, 1, label).font = Font(bold=True)
        sheet.cell(row, 2, safe_cell(value))
    guide_row = len(details) + 5
    sheet.cell(guide_row, 1, "Workbook sheet")
    sheet.cell(guide_row, 2, "Rows")
    sheet.cell(guide_row, 3, "Purpose")
    for cell in sheet[guide_row]:
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
    descriptions = {
        "Matrix Results": "All 52 terminal outcomes",
        "Dataset Rollup": "Outcome counts and mean effects by dataset",
        "Profile Rollup": "Outcome counts and mean effects by imbalance profile",
        "Model Evaluations": "Every original/repaired teacher and student evaluation",
        "Per Class Metrics": "Precision, recall and F1 by evaluated class",
        "Confusion Matrix": "Non-zero true/predicted pairs; rows=true, columns=predicted",
        "Target Label Metrics": "Recall and NLL for declared target classes",
        "Training History": "Epoch-level baseline and repair-candidate histories",
        "Artifact Index": "Every file retained in the compressed results archive",
    }
    for offset, (name, rows) in enumerate(data.items(), start=1):
        sheet.cell(guide_row + offset, 1, name)
        sheet.cell(guide_row + offset, 2, len(rows))
        sheet.cell(guide_row + offset, 3, descriptions.get(name, "Detailed experiment table"))
    sheet.column_dimensions["A"].width = 30
    sheet.column_dimensions["B"].width = 72
    sheet.column_dimensions["C"].width = 64
    sheet.freeze_panes = "A3"


def build_workbook(root: Path, archive: Path, output: Path) -> None:
    data = workbook_data(root, archive)
    workbook = Workbook()
    add_overview(workbook, root, archive, data)
    for title, rows in data.items():
        add_compatible_sheets(workbook, title, rows)
    workbook.properties.title = "Experiment 1 - All 52 Results"
    workbook.properties.subject = "Multi-dataset neuron repair and knowledge distillation results"
    workbook.properties.creator = "kd-repair export_experiment1_excel.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)

    checked = load_workbook(output, read_only=True, data_only=False)
    if "Matrix Results" not in checked.sheetnames or checked["Matrix Results"].max_row != 53:
        raise RuntimeError("Workbook validation failed: expected 52 matrix result rows")
    if len(checked.sheetnames) < 20:
        raise RuntimeError("Workbook validation failed: missing detail sheets")
    checked.close()


def main() -> None:
    args = parse_args()
    build_workbook(args.metadata_root.resolve(), args.archive.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
