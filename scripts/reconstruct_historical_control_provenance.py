"""Reconstruct explicit provenance records for legacy CE/KD control folders.

The legacy campaign predates baseline-level ``protocol.json`` files. This tool
does not pretend those files existed at training time: it creates a deterministic
record from immutable campaign, study, run-completion, summary, config, and
checkpoint evidence, and labels the record as reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUNS = ("teacher_control_seed41", "kd_control_teacher41_seed42",
        "kd_control_teacher41_seed43", "kd_control_teacher41_seed44")


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def canonical(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def completion_files(completion: dict) -> dict[str, str]:
    files = completion.get("files", completion)
    return {name: digest for name, digest in files.items()
            if name != "protocol_sha256" and isinstance(digest, str)}


def reconstruct_case(campaign: Path, case: str) -> tuple[dict, dict]:
    study = campaign / "studies" / case
    baseline = study / "baseline"
    study_protocol_path = study / "protocol.json"
    campaign_manifest_path = campaign / "manifest.json"
    receipt_path = campaign / f"{case}-receipt.json"
    study_protocol = read(study_protocol_path)
    campaign_manifest = read(campaign_manifest_path)
    receipt = read(receipt_path)

    if study_protocol.get("manifest") != campaign_manifest:
        raise ValueError(f"{case}: study and campaign source manifests differ")
    for name, digest in campaign_manifest.items():
        path = campaign / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"{case}: frozen campaign source changed: {name}")
    teacher_spec = study_protocol["spec"]["teacher"]
    if Path(teacher_spec["output_dir"]).resolve() != baseline.resolve():
        raise ValueError(f"{case}: study protocol points to another baseline")
    dataset, profile = case.split("-", 1)
    study_profile = teacher_spec["data"].get("dataset_profile", "balanced")
    if (teacher_spec["data"]["source"] != dataset or study_profile != profile):
        raise ValueError(f"{case}: study dataset/profile identity differs")

    run_records = {}
    data_identity = None
    teacher_hash = None
    for run in RUNS:
        directory = baseline / run
        config_path = directory / "config.json"
        summary_path = directory / "summary.json"
        completion_path = directory / "completion.json"
        checkpoint = directory / "student.pt"
        config, summary, completion = read(config_path), read(summary_path), read(completion_path)
        listed = completion_files(completion)
        required = {"config.json", "summary.json", "student.pt"}
        if not required.issubset(listed):
            raise ValueError(f"{case}/{run}: completion omits required files")
        for name, digest in listed.items():
            path = directory / name
            if not path.is_file() or sha(path) != digest:
                raise ValueError(f"{case}/{run}: completed artifact changed: {name}")
        if config["name"] != run or config["output_dir"] != str(baseline):
            raise ValueError(f"{case}/{run}: run identity differs")
        if (config["data"]["source"] != dataset
                or config["data"].get("dataset_profile", "balanced") != profile
                or config["train"]["epochs"] != 100):
            raise ValueError(f"{case}/{run}: data or schedule differs")
        if summary["data"] != data_identity and data_identity is not None:
            raise ValueError(f"{case}/{run}: data provenance differs within control family")
        data_identity = summary["data"] if data_identity is None else data_identity
        checkpoint_hash = sha(checkpoint)
        if run == RUNS[0]:
            teacher_hash = checkpoint_hash
            if (config["train"]["seed"] != 41
                    or config["distillation"]["method"] != "supervised"):
                raise ValueError(f"{case}: teacher is not seed-41 supervised CE")
            if receipt["baseline_teacher_sha256"] != teacher_hash:
                raise ValueError(f"{case}: case receipt identifies another teacher")
        else:
            seed = int(run.rsplit("seed", 1)[1])
            recorded = summary.get("teacher", {}).get("checkpoint_sha256")
            if (config["train"]["seed"] != seed or seed not in {42, 43, 44}
                    or config["distillation"]["method"] != "kd"
                    or recorded != teacher_hash):
                raise ValueError(f"{case}/{run}: KD/teacher identity differs")
        run_records[run] = {
            "config_sha256": sha(config_path),
            "summary_sha256": sha(summary_path),
            "completion_sha256": sha(completion_path),
            "checkpoint_sha256": checkpoint_hash,
            "initial_model_sha256": summary["initial_student_sha256"],
            "completion_files": listed,
            **({"training_teacher_sha256": summary["teacher"]["checkpoint_sha256"]}
               if run != RUNS[0] else {}),
        }

    record = {
        "schema": "historical-control-provenance-reconstruction-v1",
        "purpose": "Provenance identity for legacy CE teacher and three paired classical-KD controls",
        "provenance_status": "reconstructed_from_verified_training_artifacts",
        "training_time_file_claim": False,
        "case": case,
        "baseline": str(baseline),
        "source_evidence": {
            "campaign": campaign.name,
            "campaign_manifest_sha256": sha(campaign_manifest_path),
            "study_protocol_sha256": sha(study_protocol_path),
            "case_receipt_sha256": sha(receipt_path),
            "case_archive_sha256": receipt["archive_sha256"],
        },
        "training_source_manifest": campaign_manifest,
        "data_identity": data_identity,
        "runs": run_records,
        "validation": {
            "all_campaign_source_hashes_match": True,
            "study_manifest_matches_campaign_manifest": True,
            "all_completion_listed_file_hashes_match": True,
            "all_four_runs_share_data_identity": True,
            "all_three_students_record_same_teacher_hash": True,
            "teacher_hash_matches_case_receipt": True,
        },
    }
    protocol_hash = hashlib.sha256(canonical(record)).hexdigest()
    manifest = {
        "schema": "historical-control-provenance-manifest-v1",
        "case": case,
        "protocol_sha256": protocol_hash,
        "checkpoints": {name: row["checkpoint_sha256"] for name, row in run_records.items()},
        "configs": {name: row["config_sha256"] for name, row in run_records.items()},
        "summaries": {name: row["summary_sha256"] for name, row in run_records.items()},
        "initial_model_sha256": {name: row["initial_model_sha256"]
                                 for name, row in run_records.items()},
    }
    return record, manifest


def freeze(path: Path, value: dict, write: bool) -> None:
    raw = canonical(value)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"existing provenance differs: {path}")
        return
    if not write:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def run(campaign: Path, cases: list[str], write: bool) -> dict:
    results = []
    for case in cases:
        record, manifest = reconstruct_case(campaign, case)
        baseline = campaign / "studies" / case / "baseline"
        freeze(baseline / "protocol.json", record, write)
        freeze(baseline / "baseline_manifest.json", manifest, write)
        results.append({"case": case, "protocol_sha256": hashlib.sha256(canonical(record)).hexdigest(),
                        "teacher_sha256": record["runs"][RUNS[0]]["checkpoint_sha256"],
                        "written": write})
    return {"status": "validated_and_written" if write else "validated_dry_run",
            "campaign": str(campaign), "cases": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.campaign.resolve(), args.cases, args.write), indent=2))


if __name__ == "__main__":
    main()
