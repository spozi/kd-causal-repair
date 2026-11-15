"""Offline checks for CE-control creation and paired repair orchestration."""

from __future__ import annotations

from collections import namedtuple
import fcntl
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

import pytest

from scripts import pending_screening_archives as archives
from scripts import pending_screening_queue as queue


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def result_tree(path, panels=8, status="complete"):
    path.mkdir(parents=True)
    write_json(path / "protocol.json", {"setting": path.name})
    write_json(path / "comparison.json", {"status": status})
    (path / "report.md").write_text("Final evaluation\n")
    for i in range(panels):
        target = path / "evaluation" / str(i) / "predictions.npz"
        target.parent.mkdir(parents=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("labels.npy", b"checked payload")
    (path / "student.pt").write_bytes(b"selected checkpoint must remain outside archive")
    (path / "last.pt").write_bytes(b"resume checkpoint must remain outside archive")
    (path / "failures").mkdir()
    write_json(path / "failures" / "debug.json", {"excluded": True})
    seal(path)
    return path


def seal(path):
    files = {str(f.relative_to(path)): archives.sha(f) for f in path.rglob("*")
             if f.is_file() and f.name != "report_manifest.json"
             and f.suffix in {".json", ".md", ".csv", ".npz"}
             and "failures" not in f.relative_to(path).parts}
    write_json(path / "report_manifest.json",
               {"protocol_sha256": archives.sha(path / "protocol.json"), "files": files})


@pytest.mark.parametrize("panels", [4, 8])
def test_archive_contains_verified_result_bytes_and_predictions_but_preserves_models(tmp_path, monkeypatch, panels):
    monkeypatch.setattr(archives, "ROOT", tmp_path)
    source = result_tree(tmp_path / "source", panels=panels)
    receipt = archives.archive_result({"id": "case", "output": str(source)}, panels)
    assert receipt["prediction_panels"] == panels
    assert receipt["checkpoints_included"] is False
    assert receipt["models_deleted"] is False
    with tarfile.open(receipt["archive"]) as archive:
        names = archive.getnames()
        assert len([n for n in names if n.endswith("predictions.npz")]) == panels
        assert all(not n.endswith(".pt") and not n.startswith("failures/") for n in names)
        assert archive.extractfile("report.md").read() == (source / "report.md").read_bytes()
    assert (source / "student.pt").exists() and (source / "last.pt").exists()
    assert json.loads((tmp_path / "archives/case-verification.json").read_text()) == receipt
    assert receipt["sha256"] == archives.sha(receipt["archive"])
    # A resume verifies the backup without rewriting its compressed bytes.
    original_mtime = Path(receipt["archive"]).stat().st_mtime_ns
    assert archives.archive_result({"id": "case", "output": str(source)}, panels) == receipt
    assert Path(receipt["archive"]).stat().st_mtime_ns == original_mtime


def test_rejected_repair_archive_requires_no_fabricated_prediction_panels(tmp_path, monkeypatch):
    monkeypatch.setattr(archives, "ROOT", tmp_path)
    source = result_tree(tmp_path / "source", panels=0, status="no_repair_selected")
    receipt = archives.archive_result({"id": "rejected", "output": str(source)})
    assert receipt["prediction_panels"] == 0
    with pytest.raises(ValueError, match="CE-control baseline"):
        archives.archive_result({"id": "baseline", "output": str(source)}, 4)


@pytest.mark.parametrize("change", ["source", "protocol", "archive", "receipt", "symlink", "path", "npz"])
def test_archive_rejects_changed_or_unsafe_inputs(tmp_path, monkeypatch, change):
    monkeypatch.setattr(archives, "ROOT", tmp_path)
    source = result_tree(tmp_path / "source", panels=4)
    job = {"id": "case", "output": str(source)}
    if change in {"archive", "receipt"}:
        receipt = archives.archive_result(job, 4)
        if change == "archive":
            with Path(receipt["archive"]).open("ab") as handle:
                handle.write(b"changed")
        else:
            receipt["prediction_panels"] = 8
            write_json(tmp_path / "archives/case-verification.json", receipt)
    elif change == "source":
        (source / "report.md").write_text("changed")
    elif change == "protocol":
        write_json(source / "protocol.json", {"changed": True})
    elif change == "symlink":
        (source / "other.json").symlink_to(source / "comparison.json")
    elif change == "path":
        manifest = json.loads((source / "report_manifest.json").read_text())
        manifest["files"]["../outside.json"] = "a" * 64
        write_json(source / "report_manifest.json", manifest)
    elif change == "npz":
        (source / "evaluation/0/predictions.npz").write_bytes(b"not a zip archive")
        seal(source)
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        archives.archive_result(job, 4)
    assert (source / "student.pt").exists()


def test_wrong_panel_count_is_not_committed_as_verified_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(archives, "ROOT", tmp_path)
    source = result_tree(tmp_path / "source", panels=4)
    with pytest.raises(ValueError, match="Expected 8 prediction panels"):
        archives.archive_result({"id": "case", "output": str(source)})
    assert not (tmp_path / "archives/case-verification.json").exists()
    assert not (tmp_path / "archives/case-results-with-predictions.tar.gz").exists()


def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(archives, "ROOT", tmp_path)
    cases = []
    jobs = []
    for dataset in ("gtsrb", "fashionmnist"):
        case_id = dataset + "-lt-if50"
        baseline = str(tmp_path / "baselines" / case_id)
        cases.append({"id": case_id, "dataset": dataset, "profile": "lt-if50",
                      "baseline": baseline, "baseline_config": f"configs/{case_id}-baseline.json"})
        for policy in ("strict", "screened_then_ranked"):
            job_id = case_id + "-" + policy
            jobs.append({"id": job_id, "case": case_id, "baseline": baseline,
                         "config": f"configs/{job_id}.json",
                         "output": str(tmp_path / "studies" / job_id)})
    plan = {"remote_root": str(tmp_path), "device": "cuda:0", "disk_free_minimum_gib": 12,
            "cases": cases, "jobs": jobs}
    write_json(tmp_path / "experiment.json", plan)
    write_json(tmp_path / "deployment.json", {"controller_sha256": queue.sha(queue.__file__)})
    usage = namedtuple("usage", "total used free")(100 * 1024**3, 0, 100 * 1024**3)
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda _: usage)
    return plan


def test_queue_trains_shared_controls_once_then_runs_strict_and_expanded_in_order(tmp_path, monkeypatch):
    plan = campaign(tmp_path, monkeypatch)
    commands = []
    archived = []

    def fake_run(command, **kwargs):
        commands.append(command[1:])
        assert kwargs["check"] is True
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"]["PYTHONPATH"] == str(tmp_path / "code")
        assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
        assert json.loads((tmp_path / "status.json").read_text())["phase"] != "complete"

    def fake_archive(job, expected_prediction_panels=8):
        state = json.loads((tmp_path / "status.json").read_text())
        assert job["id"] not in state["completed"]
        if job["id"].endswith("-baseline"):
            assert job["id"][:-9] not in state["completed_baselines"]
        archived.append((job["id"], expected_prediction_panels))
        return {"status": "complete"}

    monkeypatch.setattr(queue.subprocess, "run", fake_run)
    monkeypatch.setattr(archives, "archive_result", fake_archive)
    queue.main()
    expected = [["preflight.py", "--device", "cuda:0"]]
    for case in plan["cases"]:
        expected += [["preflight.py", "--device", "cuda:0", "--baseline", case["id"]],
                     ["baseline_runner.py", "--case", case["id"]]]
        for job in [j for j in plan["jobs"] if j["case"] == case["id"]]:
            expected += [["preflight.py", "--device", "cuda:0", "--job", job["id"]],
                         ["-m", "kd", "neuron-surgery-study", "--baseline", job["baseline"],
                          "--output", job["output"], "--study-config", job["config"], "--device", "cuda:0"]]
    assert commands == expected
    assert archived == [(case["id"] + suffix, panels) for case in plan["cases"]
                        for suffix, panels in [("-baseline", 4), ("-strict", 8), ("-screened_then_ranked", 8)]]
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["phase"] == "complete" and state["current_job"] is None
    assert state["completed"] == [j["id"] for j in plan["jobs"]]
    assert state["completed_baselines"] == [c["id"] for c in plan["cases"]]


@pytest.mark.parametrize("failure", ["child", "archive", "disk"])
def test_failed_step_stops_before_next_arm_and_never_counts_it_complete(tmp_path, monkeypatch, failure):
    plan = campaign(tmp_path, monkeypatch)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command[1:])
        if failure == "child" and command[1] == "baseline_runner.py":
            raise subprocess.CalledProcessError(1, command)

    def fake_archive(job, expected_prediction_panels=8):
        if failure == "archive":
            raise ValueError("archive verification failed")
        return {"status": "complete"}

    monkeypatch.setattr(queue.subprocess, "run", fake_run)
    monkeypatch.setattr(archives, "archive_result", fake_archive)
    if failure == "disk":
        usage = namedtuple("usage", "total used free")(100 * 1024**3, 0, 11 * 1024**3)
        monkeypatch.setattr(queue.shutil, "disk_usage", lambda _: usage)
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
        queue.main()
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["phase"] == "failed"
    assert state["completed"] == state["completed_baselines"] == []
    assert not any("neuron-surgery-study" in cmd for cmd in commands)
    if failure == "disk":
        assert commands == []


def test_exclusive_lock_prevents_second_controller(tmp_path, monkeypatch):
    campaign(tmp_path, monkeypatch)
    with (tmp_path / "queue.lock").open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            queue.main()
    assert not (tmp_path / "status.json").exists()


def test_changed_controller_fails_before_any_child_command(tmp_path, monkeypatch):
    campaign(tmp_path, monkeypatch)
    write_json(tmp_path / "deployment.json", {"controller_sha256": "a" * 64})
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: pytest.fail("no child authorized"))
    with pytest.raises(ValueError, match="controller differs"):
        queue.main()
    assert json.loads((tmp_path / "status.json").read_text())["phase"] == "failed"


@pytest.mark.parametrize("change", ["guard", "pair", "shared", "output", "config"])
def test_plan_cannot_silently_change_pairing_or_write_previous_campaign(tmp_path, monkeypatch, change):
    plan = campaign(tmp_path, monkeypatch)
    if change == "guard":
        plan["disk_free_minimum_gib"] = 2
    elif change == "pair":
        plan["jobs"][0], plan["jobs"][1] = plan["jobs"][1], plan["jobs"][0]
    elif change == "shared":
        plan["jobs"][0]["baseline"] = str(tmp_path / "baselines" / "other")
    elif change == "output":
        plan["jobs"][0]["output"] = str(tmp_path.parent / "previous")
    elif change == "config":
        plan["jobs"][0]["config"] = "../unfrozen.json"
    write_json(tmp_path / "experiment.json", plan)
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: pytest.fail("no child authorized"))
    with pytest.raises(ValueError):
        queue.main()
