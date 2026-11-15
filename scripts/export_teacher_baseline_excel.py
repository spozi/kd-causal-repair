#!/usr/bin/env python3
"""Export the finalized teacher-training baseline block to a native XLSX workbook.

The input is the small, results-only package copied from the completed ascending
baseline queue.  It deliberately excludes checkpoints and prediction arrays.  The
workbook contains setting summaries, fitted-run metrics, per-class precision/recall/F1,
aggregates, timing, protocol, significance, coverage, and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs/experiment-results/teacher-baseline-final-20261102/source"
DEFAULT_OUTPUT = (ROOT / "runs/experiment-results/teacher-baseline-final-20261102"
                  / "teacher-training-baselines-final-20261102.xlsx")
DATASET_NAMES = {
    "eurosat": "EuroSAT", "gtsrb": "GTSRB", "organamnist": "OrganAMNIST",
    "fashionmnist": "Fashion-MNIST", "svhn": "SVHN", "pathmnist": "PathMNIST",
    "cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "cinic10": "CINIC-10",
    "bloodmnist": "BloodMNIST", "dermamnist": "DermaMNIST",
    "caltech101": "Caltech-101", "stl10": "STL-10",
}
PROFILE_ORDER = {"balanced": 0, "lt-if10": 1, "lt-if50": 2, "lt-if100": 3}
RECIPE_ORDER = {
    "control": 0, "randaugment": 1, "label_smoothing": 2,
    "balanced_softmax": 3, "mse_logits": 4, "mse_probabilities": 5,
    "sam": 6, "ema": 7,
}
RECIPE_NAMES = {
    "control": "CE control", "randaugment": "RandAugment",
    "label_smoothing": "Label smoothing", "balanced_softmax": "Balanced Softmax",
    "mse_logits": "MSE logits", "mse_probabilities": "MSE probabilities",
    "sam": "SAM", "ema": "EMA",
}
INCLUDED_DATASETS = ("eurosat", "gtsrb", "organamnist", "fashionmnist", "svhn", "pathmnist")
HISTORICAL_DATASETS = ("cifar10", "cifar100", "cinic10", "bloodmnist",
                       "dermamnist", "caltech101", "stl10")
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TITLE_FILL = PatternFill("solid", fgColor="D9EAF7")
POSITIVE_FILL = PatternFill("solid", fgColor="E2F0D9")
NEGATIVE_FILL = PatternFill("solid", fgColor="FCE4D6")
PP_FORMAT = "+0.000;-0.000;0.000"


def read(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_setting(setting: str) -> tuple[str, str]:
    for profile in ("lt-if100", "lt-if50", "lt-if10", "balanced"):
        suffix = "-" + profile
        if setting.endswith(suffix):
            return setting[:-len(suffix)], profile
    raise ValueError(f"Unknown setting profile: {setting}")


def safe_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running, count = {}, 0.0, len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def percent_fields(metrics: dict) -> dict:
    interval = metrics.get("accuracy_wilson_95") or (None, None)
    extended = metrics.get("extended") or {}
    return {
        "Samples": metrics.get("samples"),
        "Accuracy (%)": 100 * metrics["accuracy"],
        "Accuracy CI low (%)": 100 * interval[0] if interval[0] is not None else None,
        "Accuracy CI high (%)": 100 * interval[1] if interval[1] is not None else None,
        "Balanced accuracy (%)": 100 * metrics.get("balanced_accuracy", 0),
        "Macro F1 (%)": 100 * metrics.get("macro_f1", 0),
        "NLL": metrics.get("nll"),
        "Brier score": metrics.get("brier_score"),
        "ECE 15 bins (%)": 100 * metrics.get("ece_15_bins", 0),
        "Equal-mass ECE (%)": 100 * extended.get("ece_equal_mass_15_bins", 0),
        "Classwise ECE (%)": 100 * extended.get("classwise_ece_15_bins", 0),
        "Mean confidence (%)": 100 * metrics.get("mean_confidence", 0),
        "AURC": extended.get("aurc"),
        "Selective accuracy 80% (%)": 100 * extended.get("selective_accuracy_80", 0),
        "Selective accuracy 90% (%)": 100 * extended.get("selective_accuracy_90", 0),
        "Target macro recall (%)": 100 * metrics.get("target_macro_recall", 0),
        "Tail macro recall (%)": 100 * metrics.get("tail_macro_recall", 0),
        "Target balanced NLL": metrics.get("target_balanced_nll"),
        "Tail balanced NLL": metrics.get("tail_balanced_nll"),
    }


def load_results(source: Path):
    studies = source / "studies"
    setting_files = sorted(studies.glob("*/comparison.json"))
    if len(setting_files) != 24:
        raise ValueError(f"Expected 24 setting comparisons, found {len(setting_files)}")
    settings, method_reports = {}, {}
    for path in setting_files:
        setting = path.parent.name
        comparison = read(path)
        if comparison.get("status") != "complete" or len(comparison.get("rows", [])) != 8:
            raise ValueError(f"Incomplete setting comparison: {setting}")
        settings[setting] = comparison
        reports = {}
        for report in sorted((path.parent / "reports").glob("*/comparison.json")):
            payload = read(report)
            if payload.get("status") != "complete" or len(payload.get("runs", [])) != 4:
                raise ValueError(f"Incomplete method report: {report}")
            reports[payload["recipe"]] = payload
        if set(reports) != set(RECIPE_ORDER):
            raise ValueError(f"Setting {setting} has methods {sorted(reports)}")
        method_reports[setting] = reports
    status = read(source / "status.json")
    if status.get("phase") != "complete" or len(status.get("completed", [])) != 24:
        raise ValueError("Queue status is not a complete 24-setting ledger")
    return settings, method_reports, status


def build_rows(source: Path, settings: dict, reports: dict):
    setting_rows, run_rows, class_rows = [], [], []
    timing_rows, protocol_rows = [], []
    for setting in sorted(settings, key=lambda x: (INCLUDED_DATASETS.index(split_setting(x)[0]),
                                                    PROFILE_ORDER[split_setting(x)[1]])):
        dataset, profile = split_setting(setting)
        comparison = settings[setting]
        top = max(row["student_accuracy_mean"] for row in comparison["rows"])
        for row in sorted(comparison["rows"], key=lambda x: RECIPE_ORDER[x["recipe"]]):
            setting_rows.append({
                "Setting": setting, "Dataset": DATASET_NAMES[dataset], "Profile": profile,
                "Method": RECIPE_NAMES[row["recipe"]], "Method key": row["recipe"],
                "Teacher accuracy (%)": 100 * row["teacher_accuracy"],
                "KD student accuracy mean (%)": 100 * row["student_accuracy_mean"],
                "KD student accuracy SD (pp)": 100 * row["student_accuracy_sd"],
                "KD student target recall mean (%)": 100 * row["student_target_recall_mean"],
                "KD accuracy delta vs CE (pp)": 100 * row["accuracy_delta_vs_ce"],
                "Target recall delta vs CE (pp)": 100 * row["target_recall_delta_vs_ce"],
                "Best KD mean in setting": abs(row["student_accuracy_mean"] - top) < 1e-12,
            })
        for recipe in sorted(reports[setting], key=RECIPE_ORDER.get):
            report = reports[setting][recipe]
            targets = set(report.get("target_classes", []))
            for run in report["runs"]:
                metrics = run["metrics"]
                run_rows.append({
                    "Setting": setting, "Dataset": DATASET_NAMES[dataset], "Profile": profile,
                    "Method": RECIPE_NAMES[recipe], "Method key": recipe,
                    "Role": run["role"], "Seed": run["seed"], "Run": run["run"],
                    **percent_fields(metrics),
                    "Accuracy delta vs paired CE (pp)": 100 * run["accuracy_delta_vs_ce"],
                    "Target recall delta vs paired CE (pp)": 100 * run["target_recall_delta_vs_ce"],
                    "Evaluation split": report["evaluation_split"],
                })
                for index, item in enumerate(metrics.get("per_class", [])):
                    class_rows.append({
                        "Setting": setting, "Dataset": DATASET_NAMES[dataset], "Profile": profile,
                        "Method": RECIPE_NAMES[recipe], "Role": run["role"], "Seed": run["seed"],
                        "Class index": index, "Class": item["class"], "Is target class": index in targets,
                        "Support": item["support"], "Precision (%)": 100 * item["precision"],
                        "Recall (%)": 100 * item["recall"], "F1 (%)": 100 * item["f1"],
                    })
        study = source / "studies" / setting
        protocol = read(study / "protocol.json")
        condition = protocol["conditions"][0]
        teacher = condition["teacher"]
        student = condition["students"][0]
        expected = protocol["case"]["expected_data"]
        start, finish = (study / "protocol.json").stat().st_mtime, (study / "completion.json").stat().st_mtime
        timing_rows.append({
            "Setting": setting, "Dataset": DATASET_NAMES[dataset], "Profile": profile,
            "Training examples": sum(expected.get("train_per_class", [])),
            "Validation examples": sum(expected.get("val_per_class", [])),
            "Started UTC": datetime.fromtimestamp(start, timezone.utc).isoformat(timespec="seconds"),
            "Completed UTC": datetime.fromtimestamp(finish, timezone.utc).isoformat(timespec="seconds"),
            "Wall-clock hours": (finish - start) / 3600,
            "New teacher fits": 7, "New KD student fits": 21, "Total new fits": 28,
        })
        protocol_rows.append({
            "Setting": setting, "Dataset": DATASET_NAMES[dataset], "Profile": profile,
            "Training examples": sum(expected.get("train_per_class", [])),
            "Validation examples": sum(expected.get("val_per_class", [])),
            "Classes": teacher["data"]["num_classes"], "Image size": teacher["data"]["image_size"],
            "Preprocessing": teacher["data"]["preprocessing"],
            "Teacher": teacher["student"]["name"], "Teacher weights": teacher["student"]["weights"],
            "Student": student["student"]["name"], "Student weights": student["student"]["weights"],
            "Epochs": teacher["train"]["epochs"], "Batch size": teacher["train"]["batch_size"],
            "Learning rate": teacher["train"]["learning_rate"], "Momentum": teacher["train"]["momentum"],
            "Weight decay": teacher["train"]["weight_decay"], "Precision": teacher["train"]["precision"],
            "Teacher seed": teacher["train"]["seed"], "Student seeds": "42, 43, 44",
            "KD temperature": student["distillation"]["temperature"],
            "KD weight": student["distillation"]["weight"],
            "KD warmup epochs": student["distillation"]["warmup_epochs"],
            "Evaluation": protocol["evaluation"], "Protocol SHA-256": comparison["protocol_sha256"],
        })
    return setting_rows, run_rows, class_rows, timing_rows, protocol_rows


def aggregate_rows(setting_rows: list[dict], key: str):
    grouped = defaultdict(list)
    for row in setting_rows:
        grouped[(row[key], row["Method key"])].append(row)
    out = []
    order = INCLUDED_DATASETS if key == "Dataset" else tuple(PROFILE_ORDER)
    display_order = {DATASET_NAMES[x]: i for i, x in enumerate(order)} if key == "Dataset" else PROFILE_ORDER
    for (group, method), rows in sorted(grouped.items(), key=lambda x: (display_order[x[0][0]], RECIPE_ORDER[x[0][1]])):
        deltas = [r["KD accuracy delta vs CE (pp)"] for r in rows]
        out.append({
            key: group, "Method": RECIPE_NAMES[method], "Settings": len(rows),
            "Teacher accuracy mean (%)": statistics.mean(r["Teacher accuracy (%)"] for r in rows),
            "KD accuracy mean (%)": statistics.mean(r["KD student accuracy mean (%)"] for r in rows),
            "KD accuracy delta mean (pp)": statistics.mean(deltas),
            "Target recall delta mean (pp)": statistics.mean(r["Target recall delta vs CE (pp)"] for r in rows),
            "Settings improved": sum(d > 1e-12 for d in deltas),
            "Settings tied": sum(abs(d) <= 1e-12 for d in deltas),
            "Settings worse": sum(d < -1e-12 for d in deltas),
        })
    return out


def method_aggregate(setting_rows: list[dict]):
    grouped = defaultdict(list)
    for row in setting_rows:
        grouped[row["Method key"]].append(row)
    output = []
    for method in sorted(grouped, key=RECIPE_ORDER.get):
        rows = grouped[method]
        delta = [r["KD accuracy delta vs CE (pp)"] for r in rows]
        output.append({
            "Method": RECIPE_NAMES[method], "Method key": method, "Settings": len(rows),
            "Teacher accuracy mean (%)": statistics.mean(r["Teacher accuracy (%)"] for r in rows),
            "KD accuracy mean (%)": statistics.mean(r["KD student accuracy mean (%)"] for r in rows),
            "KD accuracy delta mean (pp)": statistics.mean(delta),
            "KD accuracy delta median (pp)": statistics.median(delta),
            "Target recall delta mean (pp)": statistics.mean(r["Target recall delta vs CE (pp)"] for r in rows),
            "Settings improved": sum(d > 1e-12 for d in delta),
            "Settings tied": sum(abs(d) <= 1e-12 for d in delta),
            "Settings worse": sum(d < -1e-12 for d in delta),
            "Best placements": sum(r["Best KD mean in setting"] for r in rows),
        })
    return output


def significance_rows(setting_rows: list[dict]):
    grouped = defaultdict(list)
    for row in setting_rows:
        if row["Method key"] != "control":
            grouped[row["Method key"]].append(row["KD accuracy delta vs CE (pp)"])
    raw = {}
    for method, values in grouped.items():
        nonzero = [round(value, 12) for value in values if abs(value) > 1e-12]
        raw[method] = float(wilcoxon(nonzero, alternative="two-sided").pvalue) if len(nonzero) > 1 else 1.0
    adjusted = holm(raw)
    rows = []
    for method in sorted(grouped, key=RECIPE_ORDER.get):
        values = grouped[method]
        rows.append({
            "Method": RECIPE_NAMES[method], "Settings": len(values),
            "Mean paired delta (pp)": statistics.mean(values),
            "Median paired delta (pp)": statistics.median(values),
            "Positive / zero / negative": f"{sum(v > 1e-12 for v in values)} / {sum(abs(v) <= 1e-12 for v in values)} / {sum(v < -1e-12 for v in values)}",
            "Wilcoxon p (two-sided)": raw[method], "Holm-adjusted p": adjusted[method],
            "Holm significant at 0.05": adjusted[method] < 0.05,
            "Interpretation scope": "Across 24 setting-level mean deltas; descriptive, conditional on one teacher seed",
        })
    return rows


def apply_number_format(cell, header: str):
    if "(pp)" in header:
        cell.number_format = PP_FORMAT
    elif "(%)" in header or header.endswith("hours"):
        cell.number_format = "0.000"
    elif " p" in header.lower() or header.lower().endswith("p"):
        cell.number_format = "0.000E+00"
    elif isinstance(cell.value, float):
        cell.number_format = "0.000000"


def write_table_sheet(workbook: Workbook, title: str, rows: list[dict], table_name: str):
    sheet = workbook.create_sheet(title)
    if not rows:
        sheet.append(["No rows"])
        return sheet
    headers = list(rows[0])
    sheet.append(headers)
    for row in rows:
        sheet.append([safe_value(row.get(header)) for header in headers])
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    table = Table(displayName=table_name, ref=sheet.dimensions)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True,
                                          showFirstColumn=False, showLastColumn=False)
    sheet.add_table(table)
    for index, header in enumerate(headers, 1):
        max_len = max(len(str(header)), *(len(str(sheet.cell(row, index).value or "")) for row in range(2, min(sheet.max_row, 500) + 1)))
        sheet.column_dimensions[sheet.cell(1, index).column_letter].width = min(48, max(10, max_len + 2))
        for row in range(2, sheet.max_row + 1):
            apply_number_format(sheet.cell(row, index), header)
    delta_columns = [i for i, h in enumerate(headers, 1) if "delta" in h.lower() or "change" in h.lower()]
    for index in delta_columns:
        letter = sheet.cell(1, index).column_letter
        region = f"{letter}2:{letter}{sheet.max_row}"
        sheet.conditional_formatting.add(region, CellIsRule(operator="greaterThan", formula=["0"], fill=POSITIVE_FILL))
        sheet.conditional_formatting.add(region, CellIsRule(operator="lessThan", formula=["0"], fill=NEGATIVE_FILL))
    return sheet


def summary_sheet(workbook: Workbook, aggregate: list[dict], status: dict, generated: str):
    sheet = workbook.active
    sheet.title = "Summary"
    sheet.sheet_view.showGridLines = False
    sheet.merge_cells("A1:H1")
    sheet["A1"] = "Teacher-Training Baseline Results"
    sheet["A1"].font = Font(size=18, bold=True, color="1F4E78")
    sheet["A3"] = "Scope"
    sheet["B3"] = "Finalized six-dataset completion block: 24 settings, eight methods, ResNet-34 → ResNet-18"
    sheet["A4"] = "Queue status"
    sheet["B4"] = status["phase"]
    sheet["A5"] = "Completed settings"
    sheet["B5"] = len(status["completed"])
    sheet["A6"] = "New fits"
    sheet["B6"] = 672
    sheet["A7"] = "Generated UTC"
    sheet["B7"] = generated
    sheet["A8"] = "Important scope note"
    sheet["B8"] = ("Project coverage is 52/52 settings across 13 datasets. This workbook embeds the 24 newly finalized "
                   "settings only; the seven historical suites are identified on Coverage and are not silently mixed.")
    for cell in (sheet["A3"], sheet["A4"], sheet["A5"], sheet["A6"], sheet["A7"], sheet["A8"]):
        cell.font = Font(bold=True)
        cell.fill = TITLE_FILL
    sheet["B8"].alignment = Alignment(wrap_text=True)
    headers = ["Method", "Teacher accuracy mean (%)", "KD accuracy mean (%)", "KD accuracy delta mean (pp)",
               "Target recall delta mean (pp)", "Improved / tied / worse", "Best placements"]
    sheet.append([])
    sheet.append(headers)
    start = sheet.max_row
    for item in aggregate:
        sheet.append([item["Method"], item["Teacher accuracy mean (%)"], item["KD accuracy mean (%)"],
                      item["KD accuracy delta mean (pp)"], item["Target recall delta mean (pp)"],
                      f"{item['Settings improved']} / {item['Settings tied']} / {item['Settings worse']}",
                      item["Best placements"]])
    for cell in sheet[start]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True)
    for row in range(start + 1, start + 1 + len(aggregate)):
        for column in (2, 3):
            sheet.cell(row, column).number_format = "0.000"
        for column in (4, 5):
            sheet.cell(row, column).number_format = PP_FORMAT
    findings_row = start + len(aggregate) + 3
    findings = [
        "Balanced Softmax is strongest overall: +0.947 pp mean KD accuracy and +3.260 pp target recall.",
        "Excluding Balanced Softmax, RandAugment is strongest: +0.433 pp and improved 17/24 settings.",
        "EMA gives a small positive mean; SAM has a slightly positive mean but negative median and higher compute.",
        "Label smoothing and both MSE objectives generally reduce student transfer performance.",
        "Teacher top-1 accuracy alone is insufficient: SAM has the highest teacher mean but weak KD gain.",
        "Inference is conditional on one teacher seed and three paired student seeds; see Significance for scope.",
    ]
    sheet.cell(findings_row, 1, "Key findings").font = Font(bold=True, size=13, color="1F4E78")
    for offset, finding in enumerate(findings, 1):
        sheet.cell(findings_row + offset, 1, f"• {finding}")
        sheet.merge_cells(start_row=findings_row + offset, start_column=1,
                          end_row=findings_row + offset, end_column=8)
    sheet.column_dimensions["A"].width = 33
    for letter in "BCDEFGH":
        sheet.column_dimensions[letter].width = 23
    sheet.freeze_panes = "A11"
    return sheet


def coverage_rows():
    rows = []
    for dataset in INCLUDED_DATASETS:
        rows.append({"Dataset": DATASET_NAMES[dataset], "Settings": 4,
                     "Coverage": "Included in this workbook", "Source block": "Ascending baseline queue (finalized 2026-11-02)"})
    for dataset in HISTORICAL_DATASETS:
        rows.append({"Dataset": DATASET_NAMES[dataset], "Settings": 4,
                     "Coverage": "Historical full suite; not embedded here",
                     "Source block": "Earlier validated experiment archives"})
    return rows


def provenance_rows(source: Path, archive: Path, settings: dict, reports: dict, status: dict, generated: str):
    completion_files = list((source / "studies").glob("*/completion.json"))
    report_manifests = list((source / "studies").glob("*/reports/*/report_manifest.json"))
    receipts = [receipt for path in completion_files for receipt in read(path).get("archives", [])]
    return [
        {"Item": "Generated UTC", "Value": generated},
        {"Item": "Repository commit", "Value": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()},
        {"Item": "Source archive", "Value": str(archive.relative_to(ROOT))},
        {"Item": "Source archive SHA-256", "Value": sha256(archive)},
        {"Item": "Source files extracted", "Value": sum(1 for p in source.rglob("*") if p.is_file())},
        {"Item": "Queue phase", "Value": status["phase"]},
        {"Item": "Completed setting ledgers", "Value": len(completion_files)},
        {"Item": "Setting comparisons", "Value": len(settings)},
        {"Item": "Method comparisons", "Value": sum(len(x) for x in reports.values())},
        {"Item": "Report manifests", "Value": len(report_manifests)},
        {"Item": "Verified archive receipts recorded", "Value": len(receipts)},
        {"Item": "Receipt statuses", "Value": json.dumps(dict(Counter(x.get("status") for x in receipts)), sort_keys=True)},
        {"Item": "Prediction panels recorded", "Value": sum(x.get("prediction_panels", 0) for x in receipts)},
        {"Item": "Checkpoints included in source package", "Value": False},
        {"Item": "Prediction arrays included in source package", "Value": False},
        {"Item": "Metric source", "Value": "Canonical CPU FP32 test evaluation recorded in each method comparison.json"},
        {"Item": "Uncertainty limitation", "Value": "One teacher seed; three paired student seeds; setting-level tests are conditional/descriptive"},
    ]


def validate_workbook(path: Path, expected_rows: dict) -> dict:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Corrupt XLSX member: {bad_member}")
        members = len(archive.namelist())
    workbook = load_workbook(path, read_only=True, data_only=False)
    actual = {sheet: workbook[sheet].max_row - 1 for sheet in expected_rows}
    missing = sorted(set(expected_rows) - set(workbook.sheetnames))
    mismatched = {key: (expected_rows[key], actual.get(key)) for key in expected_rows if actual.get(key) != expected_rows[key]}
    if missing or mismatched:
        raise ValueError(f"Workbook validation failed: missing={missing}, mismatched={mismatched}")
    workbook.close()
    return {"status": "valid", "xlsx_sha256": sha256(path), "bytes": path.stat().st_size,
            "zip_members": members, "sheets": actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    archive = source.parent / "baseline-results-source-20261102.tar.gz"
    if not source.is_dir() or not archive.is_file():
        raise FileNotFoundError("Final result source or source archive is missing")
    settings, reports, status = load_results(source)
    setting_rows, run_rows, class_rows, timing_rows, protocol_rows = build_rows(source, settings, reports)
    method_rows = method_aggregate(setting_rows)
    dataset_rows = aggregate_rows(setting_rows, "Dataset")
    profile_rows = aggregate_rows(setting_rows, "Profile")
    sig_rows = significance_rows(setting_rows)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    workbook = Workbook()
    workbook.properties.title = "Teacher-training baseline results"
    workbook.properties.subject = "Finalized six-dataset ResNet-34 to ResNet-18 KD baseline block"
    workbook.properties.creator = "kd-autonomous-car experiment exporter"
    workbook.properties.description = "Native XLSX; numeric metrics are stored as numeric cells, not formatted strings."
    summary_sheet(workbook, method_rows, status, generated)
    write_table_sheet(workbook, "Method Aggregate", method_rows, "MethodAggregate")
    write_table_sheet(workbook, "Setting Summary", setting_rows, "SettingSummary")
    write_table_sheet(workbook, "Run Metrics", run_rows, "RunMetrics")
    write_table_sheet(workbook, "Per-Class Metrics", class_rows, "PerClassMetrics")
    write_table_sheet(workbook, "Dataset Aggregate", dataset_rows, "DatasetAggregate")
    write_table_sheet(workbook, "Profile Aggregate", profile_rows, "ProfileAggregate")
    write_table_sheet(workbook, "Significance", sig_rows, "SignificanceTests")
    write_table_sheet(workbook, "Setting Timing", timing_rows, "SettingTiming")
    write_table_sheet(workbook, "Protocol", protocol_rows, "ProtocolSettings")
    write_table_sheet(workbook, "Coverage", coverage_rows(), "CoverageTable")
    provenance = provenance_rows(source, archive, settings, reports, status, generated)
    write_table_sheet(workbook, "Provenance", provenance, "ProvenanceTable")

    workbook._sheets = [workbook[title] for title in (
        "Summary", "Method Aggregate", "Setting Summary", "Run Metrics", "Per-Class Metrics",
        "Dataset Aggregate", "Profile Aggregate", "Significance", "Setting Timing", "Protocol",
        "Coverage", "Provenance")]
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    expected = {
        "Method Aggregate": len(method_rows), "Setting Summary": len(setting_rows),
        "Run Metrics": len(run_rows), "Per-Class Metrics": len(class_rows),
        "Dataset Aggregate": len(dataset_rows), "Profile Aggregate": len(profile_rows),
        "Significance": len(sig_rows), "Setting Timing": len(timing_rows),
        "Protocol": len(protocol_rows), "Coverage": 13, "Provenance": len(provenance),
    }
    validation = validate_workbook(output, expected)
    validation["workbook"] = str(output)
    validation["scope"] = "24 finalized settings across six datasets; historical seven-dataset suites indexed only"
    validation_path = output.with_suffix(".validation.json")
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
