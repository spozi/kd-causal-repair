#!/usr/bin/env python3
"""Export every full-matrix result as one filterable Excel workbook.

The full matrix is 5 teacher-student pairings x 52 dataset settings, plus the scaled
channel-budget arm for the two ResNet teachers. The workbook reads the merged runs/ tree
and holds summary sheets (with Wilcoxon signed-rank tests across settings), one row per
setting for every decision and comparison, one row per seed, the lower-response weights,
the earlier loss-balance ablation, an index of every trained model, and a provenance sheet
that records the cross-checks run while building it.

    python scripts/export_full_matrix_excel.py \\
        --output compiled-results/full-matrix-20260928/full-matrix-results-20260928.xlsx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import alternative_losses  # noqa: E402
from pairing_cells import cells  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
MATRIX = RUNS / "experiment1-multidataset"
SEEDS = ("42", "43", "44")
DATASETS = {"cinic10": "CINIC-10", "pathmnist": "PathMNIST", "svhn": "SVHN", "fashionmnist": "FashionMNIST",
            "organamnist": "OrganAMNIST", "cifar100": "CIFAR-100", "cifar10": "CIFAR-10", "gtsrb": "GTSRB",
            "eurosat": "EuroSAT", "bloodmnist": "BloodMNIST", "caltech101": "Caltech-101",
            "dermamnist": "DermaMNIST", "stl10": "STL-10"}
PROFILES = {"balanced": 1, "lt-if10": 10, "lt-if50": 50, "lt-if100": 100}
# Teacher arms: key, teacher, channel budgets, channels searched, study root for a setting.
ARMS = {
    "cnn": ("CNN-T", "2/4/8", 192, lambda j: MATRIX / j["dataset"] / j["profile"]),
    "r18": ("ResNet-18", "2/4/8", 768, lambda j: RUNS / "teacher-architecture" / f"{j['id']}-r18"),
    "r34": ("ResNet-34", "2/4/8", 768, lambda j: RUNS / "teacher-architecture" / f"{j['id']}-r34"),
    "r18s": ("ResNet-18", "8/16/32", 768, lambda j: RUNS / "teacher-architecture-scaled" / f"{j['id']}-r18"),
    "r34s": ("ResNet-34", "8/16/32", 768, lambda j: RUNS / "teacher-architecture-scaled" / f"{j['id']}-r34"),
}
# Pairings: key, student label, teacher arm, alternative-loss report used for controls and losses.
PAIRINGS = {
    "cnn-cnn": ("CNN-S", "cnn", "cnn-cnn"),
    "cnn-resnet8": ("ResNet-8", "cnn", "cnn-resnet8"),
    "cnn-resnet20": ("ResNet-20", "cnn", "cnn-resnet20"),
    "r18-cnn": ("CNN-S", "r18", "r18-cnn"),
    "r34-cnn": ("CNN-S", "r34", "r34-cnn"),
    "r18-cnn-scaled": ("CNN-S", "r18s", "r18-cnn"),
    "r34-cnn-scaled": ("CNN-S", "r34s", "r34-cnn"),
}
LOSS_PAIRINGS = ("cnn-cnn", "cnn-resnet8", "cnn-resnet20", "r18-cnn", "r34-cnn")
METHODS = {"kd": "Classical KD", "repair": "Repair arm", "loca": "LoCa", "dkd": "DKD (published)",
           "rld": "RLD (published)", "dkd_matched": "DKD (lower weight)", "rld_matched": "RLD (lower weight)"}
ABLATION = {"kd": "Classical KD (weight 0.5)", "dkd": "DKD published (weight 1, beta 8)",
            "dkd_w0.15": "DKD weight 0.15, beta 8", "dkd_beta1": "DKD weight 1, beta 1",
            "rld": "RLD published (weight 1, beta 8)", "rld_w0.2": "RLD weight 0.2, beta 8"}
RUN_GLOBS = (
    ("Experiment 1", "CNN-T teacher", "experiment1-multidataset/*/*/baselines/confirmatory/teacher_*"),
    ("Experiment 1", "Classical-KD student (CNN-T teacher)", "experiment1-multidataset/*/*/baselines/confirmatory/kd_*"),
    ("Experiment 1", "Repaired-teacher student", "experiment1-multidataset/*/*/studies/confirmatory/kd_repaired_*"),
    ("Output-level baselines", "Published LoCa/DKD/RLD student", "experiment1-multidataset/*/*/baselines/distillation/*/*_seed*"),
    ("Matched-balance rerun", "Lower-weight DKD/RLD student", "matched-balance/*/*/*_seed*"),
    ("Loss-balance ablation", "Ablation student", "distillation-ablation/*/*/*_seed*"),
    ("Student architectures", "ResNet-8/20 student", "student-architecture/*/*/*_seed*"),
    ("Teacher architectures", "ResNet teacher", "teacher-architecture/*/baselines/confirmatory/teacher_*"),
    ("Teacher architectures", "Classical-KD student (ResNet teacher)", "teacher-architecture/*/baselines/confirmatory/kd_*"),
    ("Teacher architectures", "Repaired-teacher student (2/4/8)", "teacher-architecture/*/studies/confirmatory/kd_repaired_*"),
    ("Scaled budget arm", "Repaired-teacher student (8/16/32)", "teacher-architecture-scaled/*/studies/confirmatory/kd_repaired_*"),
    ("Alternative losses", "Loss-comparison student", "alternative-losses/*/*/*_seed*"),
)
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)
THIN_GRAY = Side(style="thin", color="D9E2F3")
PP = "+0.00;-0.00;0.00"


def read(path: Path):
    return json.loads(Path(path).read_text())


def mean(values):
    return sum(values) / len(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sig(value, digits=3):
    return None if value is None else float(f"{value:.{digits}g}")


def test(diffs: list[float]) -> dict:
    """Mean and median in pp, settings higher/lower, two-sided Wilcoxon p with exact zeros dropped.

    Differences are rounded to 1e-10 first: they are multiples of 1/(3 x test-set size), so this
    changes no real value but stops floating-point noise from hiding genuine ties, which would
    switch scipy between its exact and tie-corrected methods depending on how a mean was computed.
    """
    diffs = [round(d, 10) for d in diffs]
    nonzero = [d for d in diffs if d != 0]
    return {"mean": mean(diffs) * 100, "median": statistics.median(diffs) * 100,
            "higher": sum(d > 1e-12 for d in diffs), "lower": sum(d < -1e-12 for d in diffs),
            "p": sig(wilcoxon(nonzero).pvalue) if len(nonzero) > 1 else None}


class Results:
    """Every per-setting and per-seed quantity, loaded once from runs/."""

    def __init__(self):
        summary = read(MATRIX / "matrix_summary.json")
        order = {d: i for i, d in enumerate(DATASETS)}
        self.jobs = sorted(summary["jobs"], key=lambda j: (order[j["dataset"]], PROFILES[j["profile"]]))
        self.spec = {j["id"]: read(MATRIX / "matrix-plan" / "configs" / f"{j['id']}.json") for j in self.jobs}
        self.comparison = {arm: {j["id"]: read(root(j) / "studies" / "confirmatory" / "comparison.json")
                                 for j in self.jobs} for arm, (*_, root) in ARMS.items()}
        self.alt = {p: read(RUNS / "alternative-losses" / p / "report.json") for p in LOSS_PAIRINGS}
        self.study_b = defaultdict(dict)
        for path in sorted(RUNS.glob("student-architecture/*/report.json")):
            for job, entry in read(path).items():
                for student, pair in entry.items():
                    self.study_b[job].setdefault(student, {}).update(pair)
        self.matched = defaultdict(dict)
        for path in sorted(RUNS.glob("matched-balance/*/report.json")):
            for job, entry in read(path).items():
                self.matched[job].update(entry)
        self.checks: list[dict] = []

    def check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"Check": name, "Passed": passed, "Detail": detail})

    # ---- per-seed students ------------------------------------------------------------
    def students(self, pairing: str, job: str) -> dict:
        """{seed: {"original": (acc, recall), "repaired": (acc, recall) | None}} for one pairing and setting."""
        student, arm, report = PAIRINGS[pairing]
        out = {}
        if pairing in ("cnn-resnet8", "cnn-resnet20"):
            pair = self.study_b[job][student.lower().replace("-", "")]
            for s in SEEDS:
                o = pair["original"][s]
                r = pair.get("repaired", {}).get(s)
                out[s] = {"original": (o["accuracy"], o["target_recall"]),
                          "repaired": (r["accuracy"], r["target_recall"]) if r else None}
            return out
        c = self.comparison[arm][job]
        if c["status"] == "complete":
            for row in c["students"]:
                b, k = row["baseline"], row["candidate"]
                out[str(row["seed"])] = {"original": (b["accuracy"], b["target_macro_recall"]),
                                         "repaired": (k["accuracy"], k["target_macro_recall"])}
            return out
        kd = self.alt[report][job]["kd"]
        return {s: {"original": (kd[s]["accuracy"], kd[s]["target_recall"]), "repaired": None} for s in SEEDS}

    def effect(self, pairing: str, job: str) -> dict:
        seeds = self.students(pairing, job)
        repaired = all(v["repaired"] for v in seeds.values())
        da = [v["repaired"][0] - v["original"][0] if repaired else 0.0 for v in seeds.values()]
        dr = [v["repaired"][1] - v["original"][1] if repaired else 0.0 for v in seeds.values()]
        arm = PAIRINGS[pairing][1]
        c = self.comparison[arm][job]
        if not repaired:
            passed = False
        elif pairing in ("cnn-resnet8", "cnn-resnet20"):
            # Experiment 1's rule: mean target recall +1 pp, two positive seeds, accuracy guardrail.
            passed = mean(dr) >= 0.01 and sum(d > 0 for d in dr) >= 2 and mean(da) >= -0.005
        else:
            passed = c["success"]["passed"]
        return {"repaired": repaired, "seeds": seeds, "da": da, "dr": dr, "passed": passed}

    # ---- losses -----------------------------------------------------------------------
    def losses(self, pairing: str, job: str) -> dict:
        """{method: {"weight": w, "seeds": {seed: (acc, recall)}, "source": str}} for one loss cell."""
        out = {}
        if job in self.alt[pairing]:
            for method, runs in self.alt[pairing][job].items():
                if method == "repair":
                    continue
                out[method] = {"weight": runs[SEEDS[0]].get("response_weight"),
                               "seeds": {s: (runs[s]["accuracy"], runs[s]["target_recall"]) for s in SEEDS},
                               "source": f"runs/alternative-losses/{pairing}/report.json"}
        else:  # CNN-T -> CNN-S settings with a selected repair: the earlier studies scored these
            j = next(x for x in self.jobs if x["id"] == job)
            profile = MATRIX / j["dataset"] / j["profile"]
            control = read(profile / "baselines" / "confirmatory" / f"kd_{job}_seed42" / "config.json")
            eff = self.students(pairing, job)
            out["kd"] = {"weight": control["distillation"]["weight"],
                         "seeds": {s: eff[s]["original"] for s in SEEDS},
                         "source": "Experiment 1 comparison.json"}
            for method in ("loca", "dkd", "rld"):
                c = read(profile / "baselines" / "distillation" / method / "comparison.json")
                out[method] = {"weight": c["response_weight"],
                               "seeds": {str(r["seed"]): (r["metrics"]["accuracy"], r["metrics"]["target_macro_recall"])
                                         for r in c["seeds"]},
                               "source": f"baselines/distillation/{method}/comparison.json"}
            for method, variant, weight in (("dkd_matched", "dkd_w0.15", 0.15), ("rld_matched", "rld_w0.2", 0.2)):
                runs = self.matched[job][variant]
                out[method] = {"weight": weight,
                               "seeds": {s: (runs[s]["accuracy"], runs[s]["target_recall"]) for s in SEEDS},
                               "source": "runs/matched-balance report.json"}
        eff = self.effect(pairing, job)
        out["repair"] = {"weight": out["kd"]["weight"],
                         "seeds": {s: (eff["seeds"][s]["repaired"] or eff["seeds"][s]["original"]) for s in SEEDS},
                         "source": "repaired-teacher student" if eff["repaired"] else "classical-KD student (no repair)"}
        return out


# ---- sheets --------------------------------------------------------------------------------
def setting_columns(results: Results, job: dict) -> dict:
    return {"Setting": job["id"], "Dataset": DATASETS[job["dataset"]], "Imbalance factor (rho)": PROFILES[job["profile"]]}


def settings_rows(r: Results) -> list[dict]:
    rows = []
    for j in r.jobs:
        spec = r.spec[j["id"]]
        targets = r.comparison["cnn"][j["id"]]["target_classes"]
        rows.append({**setting_columns(r, j), "Profile": j["profile"], "Classes": spec["dataset_num_classes"],
                     "Target classes (n)": len(targets), "Target class ids": json.dumps(targets),
                     "Evaluation split": spec["evaluation_split"], "Training epochs": spec["training_epochs"],
                     "Split seed": spec["split_seed"], "Student seeds": "42, 43, 44",
                     "CNN-T decision": stop(r.comparison["cnn"][j["id"]])})
    return rows


def stop(c: dict) -> str:
    if c["status"] == "complete":
        return "repair selected"
    return "stopped at localization" if "localization" in c.get("reason", "") else "stopped at development selection"


def teacher_rows(r: Results) -> list[dict]:
    rows = []
    for arm, (teacher, budgets, searched, root) in ARMS.items():
        for j in r.jobs:
            c = r.comparison[arm][j["id"]]
            study = root(j) / "studies" / "confirmatory"
            loc = read(study / "localization.json")["payload"] if (study / "localization.json").is_file() else {}
            selection = c.get("teacher_selection", {})
            chosen = selection.get("selected") or {}
            teacher_run = r.spec[j["id"]]["teacher_run"] if arm == "cnn" else f"teacher_{root(j).name}"
            original = read(root(j) / "baselines" / "confirmatory" / teacher_run / "summary.json")
            row = {**setting_columns(r, j), "Teacher": teacher, "Channel budgets": budgets,
                   "Channels searched": searched, "Decision": stop(c),
                   "Channels kept after causal validation": len(loc.get("ranking", [])) if loc else None,
                   "Feasible budgets": ", ".join(map(str, loc.get("available_channel_budgets", []))) if loc else None,
                   "Eligible repair candidates": selection.get("eligible_count"),
                   "Selected budget (channels)": chosen.get("budget"), "Selected learning rate": chosen.get("learning_rate"),
                   "Share of searched channels edited (%)": chosen["budget"] / searched * 100 if chosen else None,
                   "Teacher validation accuracy (%)": original["validation"]["accuracy"] * 100,
                   "Teacher test accuracy, original (%)": None, "Teacher test accuracy, repaired (%)": None,
                   "Teacher accuracy change (pp)": None,
                   "Teacher target recall, original (%)": None, "Teacher target recall, repaired (%)": None,
                   "Teacher target recall change (pp)": None,
                   "Reason (no repair)": c.get("reason") if c["status"] != "complete" else None}
            if c["status"] == "complete":
                b, k = c["teacher"]["baseline"], c["teacher"]["candidate"]
                row.update({"Teacher test accuracy, original (%)": b["accuracy"] * 100,
                            "Teacher test accuracy, repaired (%)": k["accuracy"] * 100,
                            "Teacher accuracy change (pp)": (k["accuracy"] - b["accuracy"]) * 100,
                            "Teacher target recall, original (%)": b["target_macro_recall"] * 100,
                            "Teacher target recall, repaired (%)": k["target_macro_recall"] * 100,
                            "Teacher target recall change (pp)": (k["target_macro_recall"] - b["target_macro_recall"]) * 100})
            rows.append(row)
    return rows


def pairing_label(pairing: str) -> tuple[str, str, str]:
    student, arm, _ = PAIRINGS[pairing]
    teacher, budgets, *_ = ARMS[arm]
    return teacher, student, budgets


def effect_rows(r: Results) -> list[dict]:
    rows = []
    for pairing in PAIRINGS:
        teacher, student, budgets = pairing_label(pairing)
        for j in r.jobs:
            e = r.effect(pairing, j["id"])
            s = e["seeds"]
            row = {"Teacher": teacher, "Student": student, "Channel budgets": budgets, **setting_columns(r, j),
                   "Repair selected": e["repaired"]}
            for i, seed in enumerate(SEEDS):
                row[f"Accuracy change, seed {seed} (pp)"] = e["da"][i] * 100
            for i, seed in enumerate(SEEDS):
                row[f"Target recall change, seed {seed} (pp)"] = e["dr"][i] * 100
            row.update({
                "Accuracy change, mean (pp)": mean(e["da"]) * 100,
                "Target recall change, mean (pp)": mean(e["dr"]) * 100,
                "Seeds with higher target recall": sum(d > 0 for d in e["dr"]) if e["repaired"] else None,
                "Transfer rule passed": e["passed"],
                "Original-teacher student accuracy (%)": mean([v["original"][0] for v in s.values()]) * 100,
                "Original-teacher student target recall (%)": mean([v["original"][1] for v in s.values()]) * 100,
                "Repaired-teacher student accuracy (%)": mean([v["repaired"][0] for v in s.values()]) * 100 if e["repaired"] else None,
                "Repaired-teacher student target recall (%)": mean([v["repaired"][1] for v in s.values()]) * 100 if e["repaired"] else None,
            })
            rows.append(row)
    return rows


def student_seed_rows(r: Results) -> list[dict]:
    rows = []
    for pairing in PAIRINGS:
        teacher, student, budgets = pairing_label(pairing)
        for j in r.jobs:
            for seed, v in r.students(pairing, j["id"]).items():
                for condition in ("original", "repaired"):
                    if v[condition] is None:
                        continue
                    rows.append({"Teacher": teacher, "Student": student, "Channel budgets": budgets,
                                 **setting_columns(r, j), "Seed": int(seed),
                                 "Teacher condition": f"{condition} teacher",
                                 "Accuracy (%)": v[condition][0] * 100, "Target recall (%)": v[condition][1] * 100})
    return rows


def loss_setting_rows(r: Results) -> list[dict]:
    rows = []
    for pairing in LOSS_PAIRINGS:
        teacher, student, _ = pairing_label(pairing)
        for j in r.jobs:
            cell = r.losses(pairing, j["id"])
            row = {"Teacher": teacher, "Student": student, **setting_columns(r, j),
                   "Repair selected": cell["repair"]["source"] == "repaired-teacher student"}
            for method, label in METHODS.items():
                row[f"{label} accuracy (%)"] = mean([v[0] for v in cell[method]["seeds"].values()]) * 100
            for method, label in METHODS.items():
                row[f"{label} target recall (%)"] = mean([v[1] for v in cell[method]["seeds"].values()]) * 100
            for method, label in METHODS.items():
                if method != "repair":
                    row[f"Repair arm minus {label}, recall (pp)"] = row[f"Repair arm target recall (%)"] - row[f"{label} target recall (%)"]
            rows.append(row)
    return rows


def loss_seed_rows(r: Results) -> list[dict]:
    rows = []
    for pairing in LOSS_PAIRINGS:
        teacher, student, _ = pairing_label(pairing)
        for j in r.jobs:
            for method, cell in r.losses(pairing, j["id"]).items():
                for seed in SEEDS:
                    acc, rec = cell["seeds"][seed]
                    rows.append({"Teacher": teacher, "Student": student, **setting_columns(r, j),
                                 "Method": METHODS[method], "Response weight": cell["weight"], "Seed": int(seed),
                                 "Accuracy (%)": acc * 100, "Target recall (%)": rec * 100, "Source": cell["source"]})
    return rows


def summary_repair_rows(r: Results) -> list[dict]:
    rows = []
    for pairing in PAIRINGS:
        teacher, student, budgets = pairing_label(pairing)
        arm = PAIRINGS[pairing][1]
        effects = {j["id"]: r.effect(pairing, j["id"]) for j in r.jobs}
        stops = Counter(stop(r.comparison[arm][j]) for j in effects)
        repaired = [j for j, e in effects.items() if e["repaired"]]
        acc = test([mean(e["da"]) for e in effects.values()])
        rec = test([mean(e["dr"]) for e in effects.values()])
        teacher_gain = [r.comparison[arm][j]["teacher"]["candidate"]["target_macro_recall"]
                        - r.comparison[arm][j]["teacher"]["baseline"]["target_macro_recall"] for j in repaired]
        teacher_acc = [r.comparison[arm][j]["teacher"]["candidate"]["accuracy"]
                       - r.comparison[arm][j]["teacher"]["baseline"]["accuracy"] for j in repaired]
        student_rec = mean([mean(effects[j]["dr"]) for j in repaired])
        rows.append({
            "Teacher": teacher, "Student": student, "Channel budgets": budgets, "Settings": len(effects),
            "Repair selected": len(repaired), "Stopped at localization": stops["stopped at localization"],
            "Stopped at development selection": stops["stopped at development selection"],
            "Teacher target recall change, repaired settings (pp)": mean(teacher_gain) * 100,
            "Teacher accuracy change, repaired settings (pp)": mean(teacher_acc) * 100,
            "Student accuracy change, all settings (pp)": acc["mean"],
            "Accuracy: settings higher": acc["higher"], "Accuracy: settings lower": acc["lower"], "Accuracy: p": acc["p"],
            "Student target recall change, all settings (pp)": rec["mean"],
            "Recall: settings higher": rec["higher"], "Recall: settings lower": rec["lower"], "Recall: p": rec["p"],
            "Student accuracy change, repaired settings (pp)": mean([mean(effects[j]["da"]) for j in repaired]) * 100,
            "Student target recall change, repaired settings (pp)": student_rec * 100,
            "Share of teacher recall gain reaching student": student_rec / mean(teacher_gain),
            "Transfer rule passed": sum(effects[j]["passed"] for j in repaired),
            "Selected budgets (channels: settings)": ", ".join(
                f"{b}: {n}" for b, n in sorted(Counter(r.comparison[arm][j]["teacher_selection"]["selected"]["budget"]
                                                       for j in repaired).items())),
        })
    return rows


def summary_loss_rows(r: Results) -> list[dict]:
    rows = []
    for pairing in LOSS_PAIRINGS:
        teacher, student, _ = pairing_label(pairing)
        cells_ = {j["id"]: r.losses(pairing, j["id"]) for j in r.jobs}
        repaired = [j for j, c in cells_.items() if c["repair"]["source"] == "repaired-teacher student"]
        for method, label in METHODS.items():
            if method == "repair":
                continue
            diff = lambda jobs, k: [mean([c["repair"]["seeds"][s][k] for s in SEEDS]) - mean([c[method]["seeds"][s][k] for s in SEEDS])
                                    for c in (cells_[j] for j in jobs)]
            acc, rec = test(diff(cells_, 0)), test(diff(cells_, 1))
            racc, rrec = test(diff(repaired, 0)), test(diff(repaired, 1))
            weights = sorted({cells_[j][method]["weight"] for j in cells_})
            rows.append({"Teacher": teacher, "Student": student, "Compared with": label,
                         "Response weight": ", ".join(f"{w:g}" for w in weights),
                         "Accuracy: repair arm minus method, all 52 (pp)": acc["mean"],
                         "Accuracy: settings higher": acc["higher"], "Accuracy: settings lower": acc["lower"],
                         "Accuracy: p": acc["p"],
                         "Recall: repair arm minus method, all 52 (pp)": rec["mean"],
                         "Recall: settings higher": rec["higher"], "Recall: settings lower": rec["lower"],
                         "Recall: p": rec["p"],
                         "Repaired settings (n)": len(repaired),
                         "Accuracy: repaired settings only (pp)": racc["mean"], "Accuracy: repaired only p": racc["p"],
                         "Recall: repaired settings only (pp)": rrec["mean"], "Recall: repaired only p": rrec["p"]})
    return rows


def summary_level_rows(r: Results) -> list[dict]:
    rows = []
    for pairing in LOSS_PAIRINGS:
        teacher, student, _ = pairing_label(pairing)
        cells_ = {j["id"]: r.losses(pairing, j["id"]) for j in r.jobs}
        repaired = [j for j, c in cells_.items() if c["repair"]["source"] == "repaired-teacher student"]
        for method, label in METHODS.items():
            level = lambda jobs, k: mean([mean([cells_[j][method]["seeds"][s][k] for s in SEEDS]) for j in jobs]) * 100
            rows.append({"Teacher": teacher, "Student": student, "Method": label,
                         "Accuracy, all 52 settings (%)": level(cells_, 0),
                         "Target recall, all 52 settings (%)": level(cells_, 1),
                         "Accuracy, repaired settings (%)": level(repaired, 0),
                         "Target recall, repaired settings (%)": level(repaired, 1)})
    return rows


def scaled_rows(r: Results) -> tuple[list[dict], list[dict]]:
    by_setting, summary = [], []
    for base, scaled, teacher in (("r18-cnn", "r18-cnn-scaled", "ResNet-18"), ("r34-cnn", "r34-cnn-scaled", "ResNet-34")):
        fa, sa = PAIRINGS[base][1], PAIRINGS[scaled][1]
        dteach, dstud, dacc, transitions = [], [], [], Counter()
        for j in r.jobs:
            f, s = r.effect(base, j["id"]), r.effect(scaled, j["id"])
            cf, cs = r.comparison[fa][j["id"]], r.comparison[sa][j["id"]]
            gain = lambda c: (c["teacher"]["candidate"]["target_macro_recall"] - c["teacher"]["baseline"]["target_macro_recall"]) if c["status"] == "complete" else 0.0
            dteach.append(gain(cs) - gain(cf)); dstud.append(mean(s["dr"]) - mean(f["dr"])); dacc.append(mean(s["da"]) - mean(f["da"]))
            transitions[("repair" if f["repaired"] else "none", "repair" if s["repaired"] else "none")] += 1
            by_setting.append({"Teacher": teacher, **setting_columns(r, j),
                               "2/4/8 decision": stop(cf), "8/16/32 decision": stop(cs),
                               "2/4/8 selected budget": (cf.get("teacher_selection", {}).get("selected") or {}).get("budget"),
                               "8/16/32 selected budget": (cs.get("teacher_selection", {}).get("selected") or {}).get("budget"),
                               "2/4/8 teacher recall change (pp)": gain(cf) * 100, "8/16/32 teacher recall change (pp)": gain(cs) * 100,
                               "2/4/8 student recall change (pp)": mean(f["dr"]) * 100, "8/16/32 student recall change (pp)": mean(s["dr"]) * 100,
                               "2/4/8 student accuracy change (pp)": mean(f["da"]) * 100, "8/16/32 student accuracy change (pp)": mean(s["da"]) * 100,
                               "2/4/8 transfer rule passed": f["passed"], "8/16/32 transfer rule passed": s["passed"],
                               "Scaled minus fixed, student recall (pp)": dstud[-1] * 100})
        for label, values in (("Teacher target recall gain", dteach), ("Student target recall change", dstud),
                              ("Student accuracy change", dacc)):
            t = test(values)
            summary.append({"Teacher": teacher, "Quantity (8/16/32 minus 2/4/8, per setting)": label,
                            "Mean difference (pp)": t["mean"], "Median difference (pp)": t["median"],
                            "Settings higher with 8/16/32": t["higher"], "Settings lower with 8/16/32": t["lower"], "p": t["p"],
                            "Repair in both arms": transitions[("repair", "repair")],
                            "Repair only with 2/4/8": transitions[("repair", "none")],
                            "Repair only with 8/16/32": transitions[("none", "repair")],
                            "No repair in either": transitions[("none", "none")]})
    return by_setting, summary


def weight_rows(r: Results) -> list[dict]:
    rows = []
    seeds = [int(s) for s in SEEDS]
    teachers, students = (RUNS / "teacher-architecture").resolve(), (RUNS / "student-architecture").resolve()
    used = {"cnn-cnn": (0.15, 0.2)}
    for pairing in LOSS_PAIRINGS:
        pc = cells(pairing, MATRIX.resolve(), teachers=teachers, students=students, verify_repaired=False)
        output = (RUNS / "alternative-losses" / pairing).resolve()
        scopes = [("all 52 settings", pc)]
        if pairing == "cnn-cnn":
            scopes = [("23 settings with a selected repair (fixed before the full matrix)", [c for c in pc if c.repaired]),
                      ("all 52 settings (reference only)", pc)]
        for scope, subset in scopes:
            rule = alternative_losses.suggest_weights(subset, output, seeds)
            teacher, student, _ = pairing_label(pairing)
            dkd_used = used.get(pairing, (rule["dkd"], rule["rld"]))[0]
            rld_used = used.get(pairing, (rule["dkd"], rule["rld"]))[1]
            rows.append({"Teacher": teacher, "Student": student, "Rule computed over": scope, "Settings": len(subset),
                         "Seeds": 3, "Warm-up epochs excluded": alternative_losses.WARMUP,
                         "Classical KD median teacher-term/CE ratio": rule["kd_balance"],
                         "Published DKD median ratio": rule["dkd_balance"], "Published RLD median ratio": rule["rld_balance"],
                         "Rule output: DKD weight": rule["dkd"], "Rule output: RLD weight": rule["rld"],
                         "DKD weight used": dkd_used, "RLD weight used": rld_used})
    return rows


def ablation_rows() -> list[dict]:
    rows = []
    report = read(RUNS / "distillation-ablation" / "report.json")
    for setting, variants in report.items():
        for variant, seeds in variants.items():
            vals = list(seeds.values())
            row = {"Setting": setting, "Variant": variant, "Description": ABLATION.get(variant, variant)}
            for seed, v in seeds.items():
                row[f"Accuracy, seed {seed} (%)"] = v["accuracy"] * 100
            for seed, v in seeds.items():
                row[f"Target recall, seed {seed} (%)"] = v["target_recall"] * 100
            row.update({"Accuracy, mean (%)": mean([v["accuracy"] for v in vals]) * 100,
                        "Target recall, mean (%)": mean([v["target_recall"] for v in vals]) * 100,
                        "Teacher-term/CE ratio, mean": mean([v["ratio"] for v in vals]),
                        "Epoch-to-epoch accuracy swing, mean (pp)": mean([v["swing"] for v in vals]) * 100})
            rows.append(row)
    return rows


def run_rows(r: Results) -> list[dict]:
    ids = sorted((j["id"] for j in r.jobs), key=len, reverse=True)
    rows = []
    for study, role, pattern in RUN_GLOBS:
        for directory in sorted(RUNS.glob(pattern)):
            if directory.is_symlink() or not (directory / "summary.json").is_file():
                continue
            summary, config = read(directory / "summary.json"), read(directory / "config.json")
            rel = directory.relative_to(ROOT).as_posix()
            runtime, teacher = summary.get("runtime", {}), summary.get("teacher") or {}
            rows.append({"Study": study, "Role": role,
                         "Setting": next((i for i in ids if f"/{i}" in rel or f"_{i}_" in rel or rel.endswith(i)), None),
                         "Pairing": rel.split("/")[2] if study == "Alternative losses" else None,
                         "Run": summary["name"], "Model": config["student"]["name"],
                         "Teacher": config["teacher"]["name"] if config["teacher"].get("checkpoint") else None,
                         "Method": config["distillation"]["method"] if config["teacher"].get("checkpoint") else "supervised",
                         "Response weight": config["distillation"]["weight"] if config["teacher"].get("checkpoint") else None,
                         "Seed": summary.get("seed"), "Epochs": config["train"]["epochs"], "Best epoch": summary.get("best_epoch"),
                         "Validation accuracy (%)": summary.get("validation", {}).get("accuracy", 0) * 100,
                         "Training minutes": summary.get("training", {}).get("total_epoch_seconds", 0) / 60,
                         "GPU": runtime.get("gpu_name"), "Precision": runtime.get("precision"),
                         "Parameters": summary.get("inference", {}).get("parameters"),
                         "Teacher parameters": teacher.get("parameters"),
                         "Initial weights SHA-256 (first 12)": (summary.get("initial_student_sha256") or "")[:12],
                         "Teacher SHA-256 (first 12)": (teacher.get("checkpoint_sha256") or "")[:12],
                         "Path": rel})
    return rows


def cross_checks(r: Results):
    """Record that sources which must agree do agree."""
    mismatches = 0
    compared = 0
    for pairing in ("cnn-cnn", "r18-cnn", "r34-cnn", "r18-cnn-scaled", "r34-cnn-scaled"):
        arm, report = PAIRINGS[pairing][1], PAIRINGS[pairing][2]
        for j in r.jobs:
            c = r.comparison[arm][j["id"]]
            if c["status"] != "complete" or j["id"] not in r.alt[report]:
                continue
            for row in c["students"]:
                kd = r.alt[report][j["id"]]["kd"][str(row["seed"])]
                compared += 1
                if abs(kd["accuracy"] - row["baseline"]["accuracy"]) > 1e-9 or abs(kd["target_recall"] - row["baseline"]["target_macro_recall"]) > 1e-9:
                    mismatches += 1
    r.check("Classical-KD student scores agree between the teacher studies and the loss reports",
            mismatches == 0, f"{compared} seed-level pairs compared, {mismatches} differ")
    for pairing in ("cnn-resnet8", "cnn-resnet20"):
        compared = mismatches = 0
        for j in r.jobs:
            e = r.effect(pairing, j["id"])
            kd = r.alt[pairing][j["id"]]["kd"]
            for s in SEEDS:
                compared += 1
                if abs(kd[s]["accuracy"] - e["seeds"][s]["original"][0]) > 1e-9 or abs(kd[s]["target_recall"] - e["seeds"][s]["original"][1]) > 1e-9:
                    mismatches += 1
        r.check(f"{PAIRINGS[pairing][0]} original-teacher scores agree between Study B and the loss reports",
                mismatches == 0, f"{compared} seed-level pairs compared, {mismatches} differ")
    same_targets = all(r.comparison[a][j["id"]]["target_classes"] == r.comparison["cnn"][j["id"]]["target_classes"]
                       for a in ARMS for j in r.jobs)
    r.check("Every teacher arm uses the same target classes per setting", same_targets)
    for arm, (teacher, budgets, *_rest) in ARMS.items():
        statuses = Counter(r.comparison[arm][j["id"]]["status"] for j in r.jobs)
        r.check(f"{teacher} {budgets}: all 52 settings decided", set(statuses) <= {"complete", "no_repair_selected"},
                ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items())))
    for pairing in LOSS_PAIRINGS:
        complete = all(set(r.losses(pairing, j["id"])) >= set(METHODS) for j in r.jobs)
        r.check(f"{pairing}: every method present in all 52 settings", complete)


def provenance_rows(r: Results) -> list[dict]:
    archives = ROOT / "compiled-results" / "full-matrix-20260928"
    rows = [{"Item": "Generated (UTC)", "Value": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            {"Item": "Repository commit when exported", "Value": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()},
            {"Item": "Code deployed to the servers", "Value": "commit 7482848; scripts/teacher_architecture_study.py replaced by commit d544c0f on 2026-09-28 (scaled-arm baseline re-confirmation uses the recorded device)"},
            {"Item": "Hardware and precision", "Value": "NVIDIA GeForce RTX 5060 Ti, bfloat16, PyTorch 2.14.0+cu132, CUDA 13.2 (all runs, all studies)"},
            {"Item": "Servers (full matrix, 27-28 Sep 2026)", "Value": "A: 4 GPUs, 15.4 CPU cores; B: 2 GPUs, 30.7 cores; C: 1 GPU, 15.4 cores (Vast.ai containers)"},
            {"Item": "Seeds", "Value": "Teacher seed 41; student seeds 42, 43, 44; split seed 2026"}]
    for name in ("results-A1.tar.zst", "results-B.tar.zst", "results-C.tar.zst"):
        path = archives / name
        if path.is_file():
            recorded = (archives / name.replace(".tar.zst", ".sha256")).read_text().split()[0]
            rows.append({"Item": f"Archive {name}", "Value": f"{path.stat().st_size / 1e9:.2f} GB; SHA-256 {recorded}; matches server: {sha256(path) == recorded}"})
    a2 = archives / "results-A2.sha256"
    if a2.is_file():
        rows.append({"Item": "Archive results-A2.tar.zst (localization intermediates, server A)",
                     "Value": f"Not downloaded; kept in the owner's Google Drive. SHA-256 {a2.read_text().split()[0]}"})
    rows += [{"Item": f"Check: {c['Check']}", "Value": f"{'PASSED' if c['Passed'] else 'FAILED'}{'; ' + c['Detail'] if c['Detail'] else ''}"}
             for c in r.checks]
    return rows


# ---- workbook ------------------------------------------------------------------------------
def number_format(header: str) -> str | None:
    if "(pp)" in header:
        return PP
    if "(%)" in header:
        return "0.00"
    if header == "p" or header.endswith(": p") or header.endswith(" only p"):
        return "General"
    if "ratio" in header.lower() or "weight" in header.lower() or header.startswith("Share"):
        return "0.000"
    if header == "Training minutes":
        return "0.0"
    return None


def add_table_sheet(workbook: Workbook, title: str, rows: list[dict], *, freeze: str = "A2") -> None:
    sheet = workbook.create_sheet(title)
    headers = list(rows[0]) if rows else ["message"]
    rows = rows or [{"message": "No records"}]
    sheet.append(headers)
    for row in rows:
        sheet.append([row.get(h) for h in headers])
    for cell in sheet[1]:
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=THIN_GRAY)
    sheet.row_dimensions[1].height = 45
    sheet.freeze_panes = freeze
    sheet.auto_filter.ref = sheet.dimensions
    for column in sheet.columns:
        header = str(column[0].value)
        width = max([len(str(c.value if c.value is not None else "")) for c in column[1:min(len(column), 300)]] + [10])
        sheet.column_dimensions[column[0].column_letter].width = min(max(width + 2, 12, min(len(header), 28) * 0.8), 44)
        fmt = number_format(header)
        for cell in column[1:]:
            if isinstance(cell.value, float):
                cell.number_format = fmt or "0.0000"


GUIDE = {
    "Summary - repair": "Repair effect on the student per pairing and budget arm: decisions, teacher gain, student change over 52 settings with Wilcoxon tests",
    "Summary - vs losses": "Repair arm against classical KD, LoCa, DKD and RLD (published and lower weight), per pairing",
    "Summary - method means": "Mean accuracy and target recall of every method per pairing",
    "Summary - scaled vs fixed": "8/16/32 against 2/4/8 channel budgets for the ResNet teachers, paired per setting",
    "Loss weights": "Lower-response DKD/RLD weights: rule inputs (training histories only) and the weights used",
    "Settings": "The 52 dataset settings: classes, target classes, split and schedule",
    "Teacher decisions": "Every teacher repair decision (5 teacher arms x 52 settings) with localization, selection and teacher test metrics",
    "Student repair effects": "Repaired minus original teacher, per student, per setting and seed (7 pairing/arm combinations x 52)",
    "Student seeds": "Every original- and repaired-teacher student score, one row per seed",
    "Losses by setting": "All seven methods per pairing and setting (seed means), with repair-arm differences",
    "Losses by seed": "Every loss-comparison student score, one row per seed, with its source",
    "Scaled vs fixed by setting": "Per-setting decisions and effects for 2/4/8 and 8/16/32 channel budgets",
    "Loss-balance ablation": "Earlier DKD/RLD balance ablation (BloodMNIST rho=100, OrganAMNIST rho=1)",
    "Run index": "Every trained model with its GPU, precision, epochs, timing and identity hashes",
    "Provenance and checks": "Code, hardware, archives with checksums, and cross-checks run while building this workbook",
}
DEFINITIONS = [
    ("Setting", "One dataset at one imbalance factor rho in {1, 10, 50, 100}: 13 datasets x 4 = 52 settings."),
    ("Target recall", "Macro-averaged recall over the setting's target (rare) classes on its sealed test split."),
    ("Repair effect", "Repaired-teacher student minus original-teacher (classical-KD) student, same seed, initial weights and data; per setting, the mean of seeds 42, 43, 44."),
    ("No repair selected", "The repair effect is exactly 0 by convention; only original-teacher students exist."),
    ("Repair arm", "In loss comparisons: the repaired-teacher student where a repair was selected, else the classical-KD student."),
    ("Transfer rule", "Mean student target recall +1 pp or more, at least 2 of 3 seeds higher, and mean accuracy change of -0.5 pp or more."),
    ("Lower weight", "DKD/RLD response weight set so the median teacher-term/cross-entropy ratio (after 5 warm-up epochs) equals classical KD's; training histories only, never test results."),
    ("Statistics", "Two-sided Wilcoxon signed-rank across settings (each setting = mean of 3 seeds); exact zeros are dropped (scipy default); means include all 52 settings."),
    ("Units", "(%) = percent; (pp) = percentage points. Positive differences favour the first-named arm."),
    ("Status", "Private working results: not for publication or sharing outside the author group."),
]


def add_overview(workbook: Workbook, sheets: dict) -> None:
    sheet = workbook.active
    sheet.title = "Overview"
    sheet["A1"] = "Full matrix: teacher repair before knowledge distillation"
    sheet["A1"].font = Font(size=18, bold=True, color="1F4E78")
    sheet["A2"] = ("5 teacher-student pairings x 52 settings, plus the 8/16/32 channel-budget arm for the ResNet teachers. "
                   "Private working results.")
    row = 4
    for label, value in (("Pairings", "CNN-T -> CNN-S, CNN-T -> ResNet-8, CNN-T -> ResNet-20, ResNet-18 -> CNN-S, ResNet-34 -> CNN-S"),
                         ("Teacher arms", "CNN-T, ResNet-18 and ResNet-34 with 2/4/8 channels; ResNet-18 and ResNet-34 with 8/16/32 channels"),
                         ("Loss comparisons", "Classical KD, LoCa, DKD and RLD (published settings), DKD and RLD (lower weight)"),
                         ("Generated (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds"))):
        sheet.cell(row, 1, label).font = Font(bold=True)
        sheet.cell(row, 2, value)
        row += 1
    row += 1
    for col, head in enumerate(("Sheet", "Rows", "Contents"), start=1):
        cell = sheet.cell(row, col, head)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
    for name, rows in sheets.items():
        row += 1
        sheet.cell(row, 1, name)
        sheet.cell(row, 2, len(rows))
        sheet.cell(row, 3, GUIDE[name])
    row += 2
    for col, head in enumerate(("Term", "Definition"), start=1):
        cell = sheet.cell(row, col, head)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
    for term, text in DEFINITIONS:
        row += 1
        sheet.cell(row, 1, term).font = Font(bold=True)
        sheet.cell(row, 2, text).alignment = Alignment(wrap_text=True, vertical="top")
    sheet.column_dimensions["A"].width = 30
    sheet.column_dimensions["B"].width = 110
    sheet.column_dimensions["C"].width = 110


def build(output: Path) -> None:
    r = Results()
    cross_checks(r)
    scaled_setting, scaled_summary = scaled_rows(r)
    sheets = {
        "Summary - repair": summary_repair_rows(r),
        "Summary - vs losses": summary_loss_rows(r),
        "Summary - method means": summary_level_rows(r),
        "Summary - scaled vs fixed": scaled_summary,
        "Loss weights": weight_rows(r),
        "Settings": settings_rows(r),
        "Teacher decisions": teacher_rows(r),
        "Student repair effects": effect_rows(r),
        "Student seeds": student_seed_rows(r),
        "Losses by setting": loss_setting_rows(r),
        "Losses by seed": loss_seed_rows(r),
        "Scaled vs fixed by setting": scaled_setting,
        "Loss-balance ablation": ablation_rows(),
        "Run index": run_rows(r),
    }
    sheets["Provenance and checks"] = provenance_rows(r)
    workbook = Workbook()
    add_overview(workbook, sheets)
    for title, rows in sheets.items():
        add_table_sheet(workbook, title, rows)
    workbook.properties.title = "Full matrix results"
    workbook.properties.subject = "Teacher repair before knowledge distillation: 5 pairings x 52 settings"
    workbook.properties.creator = "kd-repair export_full_matrix_excel.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)

    checked = load_workbook(output, read_only=True)
    expected = {"Settings": 52, "Teacher decisions": 260, "Student repair effects": 364, "Losses by setting": 260,
                "Scaled vs fixed by setting": 104, "Summary - repair": 7, "Summary - vs losses": 30}
    for name, n in expected.items():
        if checked[name].max_row != n + 1:
            raise RuntimeError(f"Workbook validation failed: {name} has {checked[name].max_row - 1} rows, expected {n}")
    checked.close()
    failed = [c for c in r.checks if not c["Passed"]]
    for c in r.checks:
        print(("PASS " if c["Passed"] else "FAIL ") + c["Check"] + (f" ({c['Detail']})" if c["Detail"] else ""))
    print({name: len(rows) for name, rows in sheets.items()})
    if failed:
        raise SystemExit(f"{len(failed)} cross-checks failed; see the Provenance and checks sheet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
