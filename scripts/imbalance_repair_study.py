"""Report direct supervised classifier comparisons after CE-only repair.

    python3 scripts/imbalance_repair_study.py \
        --vanilla-root runs/vanilla-imbalance-repair \
        --output docs/vanilla-imbalance-repair-results.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PROFILES = ("lt-if10", "lt-if50", "lt-if100")
LABELS = {"lt-if10": "IF10", "lt-if50": "IF50", "lt-if100": "IF100"}
METRICS = ("accuracy", "macro_f1", "target_recall", "head_recall")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def _pp(value: float) -> str:
    return f"{100 * (0.0 if abs(value) < 0.00005 else value):+.2f}"


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _group_recall(metrics: dict, indices: set[int]) -> float | None:
    values = [metrics["per_class"][index]["recall"] for index in sorted(indices)
              if metrics["per_class"][index]["support"] > 0]
    return _mean(values) if values else None


def _paired_scores(original: dict, repaired: dict, target: set[int]) -> dict:
    left, right = original["per_class"], repaired["per_class"]
    if (original["samples"] != repaired["samples"] or len(left) != len(right)
            or [(c["class"], c["support"]) for c in left]
            != [(c["class"], c["support"]) for c in right]):
        raise ValueError("Original and repaired models have different test class vocabulary or support")
    classes = len(left)
    if not target or len(target) == classes or any(index < 0 or index >= classes for index in target):
        raise ValueError("Target and head classes must both be present")
    def scores(metrics: dict) -> dict:
        return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "target_recall": _group_recall(metrics, target),
                "head_recall": _group_recall(metrics, set(range(classes)) - target)}
    before, after = scores(original), scores(repaired)
    if before["target_recall"] is None or before["head_recall"] is None:
        raise ValueError("Target and head classes need test support")
    delta = {metric: after[metric] - before[metric] for metric in METRICS}
    return {"original": before, "repaired": after, "delta": delta}


def load(matrix: Path, vanilla_root: Path) -> tuple[list[dict], list[dict]]:
    """Require a finalized CE-only outcome for every long-tail setting."""
    summary = _read(matrix / "matrix_summary.json")
    jobs = [job for job in summary["jobs"] if job["profile"] in PROFILES]
    if not summary["all_jobs_finished"] or len(jobs) != 3 * len({j["dataset"] for j in jobs}):
        raise ValueError("Source matrix lacks a finalized three-profile long-tail plan")
    if len({(job["dataset"], job["profile"]) for job in jobs}) != len(jobs):
        raise ValueError("Duplicate dataset/profile setting")
    rows, pending = [], []
    for job in jobs:
        path = vanilla_root / job["dataset"] / job["profile"] / "test_comparison.json"
        if not path.is_file():
            pending.append(str(path))
            continue
        comparison = _read(path)
        spec = comparison["study"]
        if (spec["name"] != f"{job['id']}-vanilla-ce" or spec["teacher_model"] != "cifar_teacher"
                or (spec["kd_weight"], spec["feature_weight"], spec["preservation_ce_weight"],
                    spec["downstream_kd"]) != (0.0, 0.0, 1.0, False)):
            raise ValueError(f"Comparison is not from the frozen supervised-only protocol: {path}")
        if comparison["status"] == "no_repair_selected":
            continue
        if comparison["status"] != "complete" or comparison["evaluation_split"] != "test":
            raise ValueError(f"Not a complete test comparison: {path}")
        baseline, candidate = comparison["teacher"]["baseline"], comparison["teacher"]["candidate"]
        target_classes = comparison["target_classes"]
        target = set(target_classes)
        if len(target) != len(target_classes) or any(type(index) is not int for index in target):
            raise ValueError(f"Invalid target classes: {path}")
        pair = _paired_scores(baseline, candidate, target)
        if abs(pair["original"]["target_recall"] - baseline["target_macro_recall"]) > 1e-9:
            raise ValueError(f"Target recall disagrees with saved metric: {path}")
        changes = [{"class": baseline["per_class"][index]["class"],
                    "support": baseline["per_class"][index]["support"],
                    "delta": candidate["per_class"][index]["recall"]
                             - baseline["per_class"][index]["recall"]}
                   for index in sorted(target)]
        rows.append({"dataset": job["dataset"], "profile": job["profile"], "model": pair,
                     "target_classes_improved": sum(c["delta"] > 0 for c in changes),
                     "target_classes_declined": sum(c["delta"] < 0 for c in changes),
                     "worst_target_class": min(changes, key=lambda c: c["delta"])})
    if pending:
        raise FileNotFoundError(f"{len(pending)} supervised-only settings remain pending; first: {pending[0]}")
    return jobs, rows


def render(jobs: list[dict], rows: list[dict]) -> str:
    datasets = {job["dataset"] for job in jobs}
    lines = ["# Supervised-only repair on imbalanced datasets", "",
             "Generated by `scripts/imbalance_repair_study.py` from the CE-only repair arm. "
             "Each comparison is the original supervised classifier versus its validation-selected "
             "repaired checkpoint. Repair used CE(target) + CE(preservation), with zero logits KD "
             "and feature preservation weights. No student model was trained.", "",
             "## Research question and endpoints", "",
             "Does supervised-only channel repair improve minority-class recall while preserving "
             "head-class recall, accuracy, and macro-F1?", "",
             f"The {len(datasets)} datasets each have IF10, IF50, and IF100 profiles: "
             f"{len(jobs)} long-tail settings. Train and validation data are deterministically "
             "downsampled; the test split is unresampled. Minority classes are each setting's "
             "frozen `target_classes`; head classes are their complement. Group recall is the "
             "unweighted mean of per-class recall where test support is nonzero. One setting is "
             "the unit of analysis. Differences are repaired minus original, in percentage points.", "",
             "The primary descriptive endpoint is minority recall change across all settings "
             "under the validation-gated no-change policy. Head recall, accuracy, macro-F1, "
             "and per-class regressions are secondary endpoints.", "",
             "## Coverage", "",
             "| Profile | Planned | Repair selected | No repair selected |",
             "|---|---:|---:|---:|"]
    for profile in PROFILES:
        group = [job for job in jobs if job["profile"] == profile]
        selected = sum(row["profile"] == profile for row in rows)
        lines.append(f"| {LABELS[profile]} | {len(group)} | {selected} | {len(group) - selected} |")
    lines += [f"| **All long-tail** | {len(jobs)} | {len(rows)} | {len(jobs) - len(rows)} |", "",
              "No selected repair means the original classifier is retained, with zero change "
              "by policy. No test score is imputed for those settings.", "",
              "## Test performance where repair was selected", "",
              "Each mean gives equal weight to selected settings in that profile.", "",
              "| Profile | n | Target recall, original → repaired | Δ head recall | Δ accuracy | Δ macro-F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for profile in PROFILES:
        group = [row for row in rows if row["profile"] == profile]
        if not group:
            continue
        before = _mean([row["model"]["original"]["target_recall"] for row in group])
        after = _mean([row["model"]["repaired"]["target_recall"] for row in group])
        deltas = {metric: _mean([row["model"]["delta"][metric] for row in group])
                  for metric in ("head_recall", "accuracy", "macro_f1")}
        lines.append(f"| {LABELS[profile]} | {len(group)} | {_pct(before)} → {_pct(after)} | "
                     f"{_pp(deltas['head_recall'])} | {_pp(deltas['accuracy'])} | "
                     f"{_pp(deltas['macro_f1'])} |")
    lines += ["", "## Full no-change policy", "",
              "Selected-repair deltas are divided by every planned setting in that profile; "
              "these means do not estimate a forced repair.", "",
              "| Profile | Δ target recall | Δ head recall | Δ accuracy |",
              "|---|---:|---:|---:|"]
    for profile in PROFILES:
        group = [row for row in rows if row["profile"] == profile]
        planned = sum(job["profile"] == profile for job in jobs)
        lines.append(f"| {LABELS[profile]} | "
                     f"{_pp(sum(row['model']['delta']['target_recall'] for row in group) / planned)} | "
                     f"{_pp(sum(row['model']['delta']['head_recall'] for row in group) / planned)} | "
                     f"{_pp(sum(row['model']['delta']['accuracy'] for row in group) / planned)} |")
    lines += [f"| **All long-tail** | "
              f"{_pp(sum(row['model']['delta']['target_recall'] for row in rows) / len(jobs))} | "
              f"{_pp(sum(row['model']['delta']['head_recall'] for row in rows) / len(jobs))} | "
              f"{_pp(sum(row['model']['delta']['accuracy'] for row in rows) / len(jobs))} |", "",
              "## Selected settings", "",
              "Target classes +/− counts classes with higher/lower recall. The worst-class "
              "column includes its test support.", "",
              "| Dataset | Profile | Δ target recall | Δ head recall | Δ accuracy | Δ macro-F1 | "
              "Target classes +/− | Worst target class Δ |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in sorted(rows, key=lambda item: (item["dataset"], PROFILES.index(item["profile"]))):
        delta, worst = row["model"]["delta"], row["worst_target_class"]
        lines.append(f"| {row['dataset']} | {LABELS[row['profile']]} | {_pp(delta['target_recall'])} | "
                     f"{_pp(delta['head_recall'])} | {_pp(delta['accuracy'])} | {_pp(delta['macro_f1'])} | "
                     f"{row['target_classes_improved']}/{row['target_classes_declined']} | "
                     f"{worst['class']} {_pp(worst['delta'])} (n={worst['support']}) |")
    lines += ["", "## Interpretation limits", "",
              "The test sets were examined in Experiment 1, so these results are exploratory. "
              "The validation gate may select different datasets at each imbalance level, "
              "and the three profiles from one dataset are related. Profile means do not "
              "establish a monotonic imbalance effect or statistical significance. "
              "Small per-class test support can make individual changes noisy.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, default=Path("runs/experiment1-multidataset"))
    parser.add_argument("--vanilla-root", type=Path, default=Path("runs/vanilla-imbalance-repair"))
    parser.add_argument("--output", type=Path, help="Write Markdown here instead of stdout")
    args = parser.parse_args()
    try:
        jobs, rows = load(args.matrix.resolve(), args.vanilla_root.resolve())
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    report = render(jobs, rows)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
