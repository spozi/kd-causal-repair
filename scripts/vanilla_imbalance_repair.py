"""Run a supervised-only channel repair on the frozen long-tail matrix.

The original classifier is reused. Repair uses CE(target) + CE(preservation),
with both logits KD and feature preservation disabled. No student is trained.
Each selected repaired classifier is evaluated on the original test split.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

PROFILES = ("lt-if10", "lt-if50", "lt-if100")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_once(path: Path, payload: dict) -> None:
    normalized = json.loads(json.dumps(payload, allow_nan=False))
    if path.exists():
        if _read(path) != normalized:
            raise ValueError(f"Existing artifact differs; choose a new output root: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")


def plan(matrix: Path, dataset: str | None, profile: str | None, all_jobs: bool) -> list[dict]:
    if not all_jobs and (dataset is None or profile is None):
        raise ValueError("Supply both --dataset and --profile, or use --all")
    if all_jobs and (dataset is not None or profile is not None):
        raise ValueError("--all cannot be combined with --dataset or --profile")
    if profile is not None and profile not in PROFILES:
        raise ValueError(f"Only long-tail profiles are supported: {PROFILES}")
    summary = _read(matrix / "matrix_summary.json")
    jobs = [job for job in summary["jobs"] if job["profile"] in PROFILES]
    if not summary["all_jobs_finished"]:
        raise ValueError("Source matrix must be finalized")
    if not all_jobs:
        jobs = [job for job in jobs if job["dataset"] == dataset and job["profile"] == profile]
    if not jobs:
        raise ValueError("Requested dataset/profile is absent from the matrix")
    planned = []
    for job in jobs:
        raw = _read(matrix / "matrix-plan" / "configs" / f"{job['id']}.json")
        raw.update(name=f"{job['id']}-vanilla-ce", kd_weight=0.0, feature_weight=0.0,
                   preservation_ce_weight=1.0, downstream_kd=False, repair_subject="teacher")
        baseline = matrix / job["dataset"] / job["profile"] / "baselines" / "confirmatory"
        checkpoint = baseline / raw["teacher_run"] / "student.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing supervised classifier checkpoint: {checkpoint}")
        planned.append({"id": job["id"], "dataset": job["dataset"], "profile": job["profile"],
                        "baseline": str(baseline), "study": raw})
    return planned


def _relocated_baseline(item: dict, output_root: Path, data_root: Path) -> Path:
    source = Path(item["baseline"])
    study = item["study"]
    staged = output_root / "_inputs" / item["dataset"] / item["profile"]
    destination = staged / study["teacher_run"]
    source_checkpoint = source / study["teacher_run"] / "student.pt"
    staged_checkpoint = destination / "student.pt"
    destination.mkdir(parents=True, exist_ok=True)
    if staged_checkpoint.exists():
        from kd.checkpoints import fingerprint
        if fingerprint(staged_checkpoint) != fingerprint(source_checkpoint):
            raise ValueError(f"Staged classifier differs from source: {staged_checkpoint}")
    else:
        shutil.copy2(source_checkpoint, staged_checkpoint)
    config = _read(source / study["teacher_run"] / "config.json")
    config["data"]["root"] = str(data_root.resolve())
    config["output_dir"] = str(staged)
    _write_once(destination / "config.json", config)
    return staged


def run_job(item: dict, output_root: Path, device: str,
            data_root: Path | None = None) -> dict:
    import numpy as np

    from kd.checkpoints import fingerprint, load_model_checkpoint, metadata
    from kd.config import from_dict
    from kd.data import build_data
    from kd.evaluation import collect_predictions, prediction_metrics
    from kd.models import create_model
    from kd.neuron_surgery_study import neuron_surgery_spec_from_dict, run_neuron_surgery_study
    from kd.runtime import prepare_model, resolve_device

    spec = neuron_surgery_spec_from_dict(item["study"])
    if (spec.kd_weight, spec.feature_weight, spec.preservation_ce_weight,
            spec.downstream_kd) != (0.0, 0.0, 1.0, False):
        raise ValueError("This runner requires CE-only repair without downstream KD")
    output = output_root / item["dataset"] / item["profile"]
    result_path = output / "test_comparison.json"
    baseline = (_relocated_baseline(item, output_root, data_root) if data_root is not None
                else Path(item["baseline"]))
    original_checkpoint = baseline / spec.teacher_run / "student.pt"
    if result_path.is_file():
        saved = _read(result_path)
        if (saved["study"] != spec.to_dict()
                or saved["original_checkpoint_sha256"] != fingerprint(original_checkpoint)):
            raise ValueError(f"Saved comparison belongs to a different study: {result_path}")
        if saved.get("repaired_checkpoint_sha256"):
            repaired_path = output / "teacher_repaired.pt"
            if saved["repaired_checkpoint_sha256"] != fingerprint(repaired_path):
                raise ValueError(f"Repaired checkpoint changed: {repaired_path}")
        return saved

    repair = run_neuron_surgery_study(baseline, output, device, spec.student_seeds, spec=spec)
    common = {"study": spec.to_dict(), "target_classes": repair["target_classes"],
              "original_checkpoint_sha256": fingerprint(original_checkpoint)}
    if repair["status"] == "no_repair_selected":
        comparison = {**common, "status": "no_repair_selected", "reason": repair["reason"]}
        _write_once(result_path, comparison)
        return comparison
    if repair["status"] != "repair_only_complete":
        raise ValueError(f"Unexpected repair status: {repair['status']}")

    repaired_checkpoint = output / "teacher_repaired.pt"
    config = from_dict(_read(baseline / spec.teacher_run / "config.json"))
    resolved_device = resolve_device(device)
    data = build_data(config.data, replace(config.train, device=str(resolved_device), workers=0),
                      include_test=True, diagnostic=True)
    if data.test is None:
        raise ValueError("Dataset has no test split")
    original_metrics = repaired_metrics = None
    labels_reference = None
    for checkpoint, label in ((original_checkpoint, "original"),
                              (repaired_checkpoint, "repaired")):
        model = create_model(spec.teacher_model, config.data.num_classes)
        load_model_checkpoint(model, checkpoint, metadata(
            spec.teacher_model, data.classes, config.data.image_size, config.data.source))
        prepare_model(model, resolved_device, config.train.channels_last).eval()
        labels, probabilities = collect_predictions(
            model, data.test, resolved_device, precision=config.train.precision,
            channels_last=config.train.channels_last)
        if labels_reference is not None and not np.array_equal(labels_reference, labels):
            raise ValueError("Original and repaired models saw different test examples")
        labels_reference = labels
        quality = prediction_metrics(labels, probabilities, data.classes)
        target_recalls = [quality["per_class"][index]["recall"]
                          for index in repair["target_classes"]
                          if quality["per_class"][index]["support"] > 0]
        if not target_recalls:
            raise ValueError("No target class has test support")
        quality["target_macro_recall"] = sum(target_recalls) / len(target_recalls)
        if label == "original":
            original_metrics = quality
        else:
            repaired_metrics = quality
    comparison = {**common, "status": "complete", "evaluation_split": "test",
                  "repaired_checkpoint_sha256": fingerprint(repaired_checkpoint),
                  "teacher": {"baseline": original_metrics, "candidate": repaired_metrics},
                  "test_provenance": data.provenance}
    _write_once(result_path, comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, default=Path("runs/experiment1-multidataset"))
    parser.add_argument("--output", type=Path, default=Path("runs/vanilla-imbalance-repair"))
    parser.add_argument("--dataset")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--all", action="store_true", help="Run every long-tail setting sequentially")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path,
                        help="Relocate frozen dataset paths to this local root")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without training")
    args = parser.parse_args()
    items = plan(args.matrix.resolve(), args.dataset, args.profile, args.all)
    if args.dry_run:
        print(json.dumps({"job_count": len(items), "loss": "CE(target)+CE(preservation)",
                          "no_kd": True, "jobs": [{"id": item["id"],
                                                   "baseline": item["baseline"]} for item in items]},
                         indent=2))
        return
    for item in items:
        result = run_job(item, args.output.resolve(), args.device,
                         args.data_root.resolve() if args.data_root else None)
        print(f"{item['id']}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
