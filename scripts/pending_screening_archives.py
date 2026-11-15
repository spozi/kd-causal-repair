"""Verified results/predictions archives for the pending-dataset campaign."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parent
RESULT_SUFFIXES = {".json", ".md", ".csv", ".npz"}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _relative(name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Invalid result path: {name}")
    return relative


def _members(path):
    members = {}
    prediction_panels = 0
    with tarfile.open(path) as archive:
        for member in archive:
            relative = _relative(member.name)
            if (not member.isfile() or member.name in members
                    or relative.suffix not in RESULT_SUFFIXES):
                raise ValueError("Invalid, duplicate, or non-result archived member")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("Missing archived member bytes")
            raw = extracted.read()
            members[member.name] = hashlib.sha256(raw).hexdigest()
            if member.name.endswith(".npz"):
                with zipfile.ZipFile(io.BytesIO(raw)) as arrays:
                    if arrays.testzip() is not None:
                        raise ValueError("Invalid NPZ CRC in result archive")
            if relative.name == "predictions.npz":
                prediction_panels += 1
    return members, prediction_panels


def archive_result(job, expected_prediction_panels=8):
    """Archive immutable reports, using four panels for controls or eight for repair."""
    if expected_prediction_panels not in {4, 8}:
        raise ValueError("Expected four baseline or eight paired prediction panels")
    job_id = job["id"]
    if (not isinstance(job_id, str) or not job_id or job_id in {".", ".."}
            or Path(job_id).name != job_id):
        raise ValueError("Archive ID must be a single path component")
    output = Path(job["output"]).resolve()
    report = json.loads((output / "comparison.json").read_text())
    if report.get("status") not in {"complete", "no_repair_selected"}:
        raise ValueError("Study did not complete teacher selection/downstream comparison")
    if expected_prediction_panels == 4 and report["status"] != "complete":
        raise ValueError("A CE-control baseline must complete its four prediction panels")
    manifest = json.loads((output / "report_manifest.json").read_text())
    if manifest.get("protocol_sha256") != sha(output / "protocol.json"):
        raise ValueError("Report protocol changed")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Final report manifest has no files")
    for name, digest in manifest["files"].items():
        relative = _relative(name)
        path = output / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(output)
                or sha(path) != digest):
            raise ValueError(f"Final report integrity check failed: {name}")
    entries = list(output.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Result tree contains a symlink")
    paths = sorted(path for path in entries if path.is_file()
                   and path.suffix in RESULT_SUFFIXES
                   and "failures" not in path.relative_to(output).parts)
    if not paths:
        raise ValueError("Invalid archive file list")
    expected = {str(path.relative_to(output)): sha(path) for path in paths}
    expected_panels = expected_prediction_panels if report["status"] == "complete" else 0
    directory = ROOT / "archives"
    directory.mkdir(exist_ok=True)
    destination = directory / (job_id + "-results-with-predictions.tar.gz")
    receipt_path = directory / (job_id + "-verification.json")
    # Resume verifies existing archives rather than replacing a verified backup.
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("archive") != str(destination)
                or receipt.get("status") != report["status"]
                or receipt.get("bytes") != destination.stat().st_size
                or receipt.get("sha256") != sha(destination)):
            raise ValueError("Existing result archive changed")
        members, panels = _members(destination)
        if (members != expected or panels != expected_panels
                or receipt.get("members") != len(members)
                or receipt.get("prediction_panels") != panels
                or receipt.get("checkpoints_included") is not False
                or receipt.get("models_deleted") is not False):
            raise ValueError("Existing result archive differs from final report")
        return receipt
    if destination.exists():
        raise FileExistsError("Unverified result archive exists; inspect before resuming")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", dereference=True) as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(output)), recursive=False)
    members, panels = _members(temporary)
    if members != expected:
        raise ValueError("Archived bytes differ from source results")
    if panels != expected_panels:
        raise ValueError(f"Expected {expected_panels} prediction panels; found {panels}")
    temporary.replace(destination)
    receipt = dict(archive=str(destination), sha256=sha(destination),
                   bytes=destination.stat().st_size, status=report["status"],
                   members=len(members), prediction_panels=panels,
                   checkpoints_included=False, models_deleted=False)
    receipt_temporary = receipt_path.with_suffix(".json.tmp")
    receipt_temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt_temporary.replace(receipt_path)
    return receipt
