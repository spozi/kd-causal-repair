"""Validation-only pilot: repair the ResNet teachers from held-out errors.

The ResNet teachers fit their training images almost perfectly, so the hard images the
original repair learns from carry little error signal and most candidates leave the
teacher unchanged. This pilot reruns each ResNet study with ``hard_source="validation"``:
hard images come from one half of the validation set and candidates are selected on the
other. It stops before student distillation and never builds the test set.

The comparison uses the original study's stored candidate results, which were scored
on the full validation set; their checkpoints were not kept, so they cannot be re-scored
on the selection half. Held-out candidates are scored on the selection half only.

    python3 scripts/held_out_repair_pilot.py --plan
    python3 scripts/held_out_repair_pilot.py --only cifar10/lt-if10 --device mps
    python3 scripts/held_out_repair_pilot.py --device mps --keep-going
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kd.checkpoints import write_json
from kd.neuron_surgery_study import neuron_surgery_spec_from_dict, run_neuron_surgery_study

ARCHITECTURES = ("r18", "r34")
GUARDRAIL = 0.005


def payload(path: Path):
    data = json.loads(path.read_text())
    return data.get("payload", data) if isinstance(data, dict) else data


def settings(source_root: Path, only: list[str], data_root: Path) -> list[dict]:
    rows = []
    for protocol in sorted(source_root.glob("*/studies/confirmatory/protocol.json")):
        run = protocol.parents[2].name
        setting, arch = run.rsplit("-", 1)
        spec = payload(protocol)["study"]
        dataset, profile = spec["dataset_source"], spec["dataset_profile"] or "balanced"
        name = f"{dataset}/{profile}"
        if arch not in ARCHITECTURES or (only and not any(t in (dataset, name) for t in only)):
            continue
        if not (data_root / dataset).exists():
            continue
        # Settings whose teacher weights were not kept cannot be repaired.
        teacher = protocol.parents[2] / "baselines" / "confirmatory" / spec["teacher_run"] / "student.pt"
        if not teacher.is_file():
            continue
        rows.append({"run": run, "dataset": dataset, "profile": profile, "arch": arch,
                     "source": protocol.parent, "baseline": protocol.parents[2] / "baselines" / "confirmatory",
                     "spec": spec})
    return rows


def run_held_out(row: dict, output_root: Path, device: str, data_root: Path,
                 learning_rates: tuple[float, ...]) -> Path:
    spec = replace(neuron_surgery_spec_from_dict(row["spec"]), name=f"{row['run']}-heldout",
                   hard_source="validation", downstream_kd=False,
                   learning_rates=learning_rates)
    output = output_root / row["run"]
    run_neuron_surgery_study(baseline=str(row["baseline"]), output=str(output), device=device,
                             seeds=spec.student_seeds, spec=spec, data_root=data_root)
    return output


def candidate_deltas(baseline: dict, candidates: list[dict]) -> list[dict]:
    return [{"budget": c["budget"], "learning_rate": c["learning_rate"],
             "recall": (c["validation"]["target_macro_recall"] - baseline["target_macro_recall"]) * 100,
             "accuracy": (c["validation"]["accuracy"] - baseline["accuracy"]) * 100,
             "nll": c["validation"]["target_balanced_nll"]} for c in candidates]


def selected(deltas: list[dict]) -> dict | None:
    """The study's rule: strictly higher target recall within the accuracy guardrail."""
    eligible = [d for d in deltas if d["recall"] > 0 and d["accuracy"] >= -GUARDRAIL * 100]
    eligible.sort(key=lambda d: (-d["recall"], d["nll"], d["budget"], d["learning_rate"]))
    return eligible[0] if eligible else None


def summarize(deltas: list[dict]) -> dict:
    choice = selected(deltas)
    return {"candidates": len(deltas),
            "recall_up": sum(d["recall"] > 0 for d in deltas),
            "median_recall": statistics.median(d["recall"] for d in deltas) if deltas else None,
            "selected": choice is not None,
            "selected_recall": choice["recall"] if choice else 0.0,
            "selected_accuracy": choice["accuracy"] if choice else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=Path("runs/teacher-architecture"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/held-out-repair-pilot"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--only", nargs="+", default=[], metavar="DATASET[/PROFILE]")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.001, 0.005])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plan", action="store_true", help="List the settings and stop")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    jobs = settings(args.source_root, args.only, args.data_root)
    print(f"{len(jobs)} ResNet settings with local data")
    for row in jobs:
        print(f"  {row['run']}")
    if args.plan or not jobs:
        return 0 if jobs else 1

    rates = tuple(sorted(args.learning_rates))
    output_root = args.output_root / ("lr-" + "-".join(f"{rate:g}" for rate in rates))
    rows, failures = [], []
    for index, row in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {row['run']}", flush=True)
        try:
            held_out_dir = run_held_out(row, output_root, args.device, args.data_root, rates)
            new_base = payload(held_out_dir / "teacher_validation_baseline.json")
            results = held_out_dir / "candidate_results.json"
            new = summarize(candidate_deltas(new_base, payload(results)) if results.exists() else [])
            old = summarize(candidate_deltas(payload(row["source"] / "teacher_validation_baseline.json"),
                                             payload(row["source"] / "candidate_results.json")))
            rows.append({"run": row["run"], "arch": row["arch"], "dataset": row["dataset"],
                         "profile": row["profile"], "train_images": old, "held_out": new})
            print(f"    training images: {old['recall_up']}/{old['candidates']} candidates raise recall, "
                  f"selected {old['selected_recall']:+.2f} pp | held-out: "
                  f"{new['recall_up']}/{new['candidates']}, selected {new['selected_recall']:+.2f} pp",
                  flush=True)
        except Exception as error:
            failures.append({"run": row["run"], "error": f"{type(error).__name__}: {error}"})
            print(f"    FAILED {type(error).__name__}: {error}", file=sys.stderr)
            if not args.keep_going:
                traceback.print_exc()
                break

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "pilot_summary.json",
               {"learning_rates": list(rates), "rows": rows, "failures": failures})
    for arch in ARCHITECTURES:
        sub = [r for r in rows if r["arch"] == arch]
        if not sub:
            continue
        print(f"\n{arch}: {len(sub)} settings")
        for method in ("train_images", "held_out"):
            picked = [r[method] for r in sub]
            print(f"  {method:13} repairs selected {sum(p['selected'] for p in picked)}/{len(sub)}, "
                  f"mean selected recall gain {statistics.mean(p['selected_recall'] for p in picked):+.2f} pp, "
                  f"mean accuracy change {statistics.mean(p['selected_accuracy'] for p in picked):+.2f} pp")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
