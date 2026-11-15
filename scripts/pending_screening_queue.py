"""Serial CE-control and paired-screening controller; never changes frozen inputs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

try:
    from . import pending_screening_archives as archives
except ImportError:  # Direct script execution in the prepared remote package.
    import pending_screening_archives as archives


ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def status(phase, completed, completed_baselines, job=None, **extra):
    value = dict(phase=phase, completed=list(completed),
                 completed_baselines=list(completed_baselines), current_job=job,
                 updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = ROOT / "status.json.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(ROOT / "status.json")


def _admit_storage(plan):
    minimum = plan["disk_free_minimum_gib"]
    if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
            or not math.isfinite(minimum) or minimum < 12):
        raise ValueError("Retained-model campaign requires a disk guard of at least 12 GiB")
    if shutil.disk_usage(ROOT).free < minimum * 1024**3:
        raise RuntimeError("Insufficient free disk for retained models; no files were deleted")


def _validate_plan(plan):
    if Path(plan["remote_root"]).resolve() != ROOT.resolve():
        raise ValueError("Queue was not launched at its frozen remote root")
    if not isinstance(plan.get("device"), str) or not plan["device"].startswith("cuda:"):
        raise ValueError("Frozen BF16 campaign requires an explicit CUDA device")
    cases, jobs = plan["cases"], plan["jobs"]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and unique")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Repair job IDs must be unique")
    expected = [case["id"] + "-" + policy for case in cases
                for policy in ("strict", "screened_then_ranked")]
    if [job["id"] for job in jobs] != expected:
        raise ValueError("Each case must run strict then expanded with no missing or extra jobs")
    for case in cases:
        name = case["id"]
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError("Case ID must be a single path component")
        baseline = Path(case["baseline"]).resolve()
        if baseline == ROOT.resolve() or not baseline.is_relative_to(ROOT.resolve()):
            raise ValueError("Baseline output must belong to the new campaign")
    by_case = {case["id"]: case for case in cases}
    for job in jobs:
        if (job["case"] not in by_case
                or Path(job["baseline"]).resolve() != Path(by_case[job["case"]]["baseline"]).resolve()):
            raise ValueError("Repair policies must share the case's exact CE/KD controls")
        output = Path(job["output"]).resolve()
        if output == ROOT.resolve() or not output.is_relative_to(ROOT.resolve()):
            raise ValueError("Repair output must belong to the new campaign")
        config = Path(job["config"])
        if config.is_absolute() or ".." in config.parts:
            raise ValueError("Repair config must be a frozen relative input")
    outputs = [str(Path(case["baseline"]).resolve()) for case in cases]
    outputs += [str(Path(job["output"]).resolve()) for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Study and baseline outputs must be distinct")


def _run(arguments, log, env):
    with (ROOT / log).open("a") as handle:
        subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    import fcntl

    # Keep the handle alive for the entire queue; a disconnected SSH does not own it.
    with (ROOT / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed, completed_baselines, current = [], [], None
        try:
            deployment = json.loads((ROOT / "deployment.json").read_text())
            if sha(__file__) != deployment["controller_sha256"]:
                raise ValueError("Deployment controller differs from frozen deployment identity")
            plan = json.loads((ROOT / "experiment.json").read_text())
            _validate_plan(plan)
            archives.ROOT = ROOT
            env = dict(os.environ, PYTHONPATH=str(ROOT / "code"), PYTHONUNBUFFERED="1")
            _admit_storage(plan)
            status("preflight", completed, completed_baselines)
            _run(["preflight.py", "--device", plan["device"]], "queue-preflight.log", env)
            for case in plan["cases"]:
                current = case["id"] + "-baseline"
                _admit_storage(plan)
                status("baseline_admission", completed, completed_baselines, current)
                _run(["preflight.py", "--device", plan["device"], "--baseline", case["id"]],
                     current + ".log", env)
                status("baseline_training", completed, completed_baselines, current)
                _run(["baseline_runner.py", "--case", case["id"]], current + ".log", env)
                status("baseline_archiving", completed, completed_baselines, current)
                receipt = archives.archive_result(
                    {"id": current, "output": case["baseline"]}, expected_prediction_panels=4)
                completed_baselines.append(case["id"])
                status("queued", completed, completed_baselines, archive=receipt)
                for job in (job for job in plan["jobs"] if job["case"] == case["id"]):
                    current = job["id"]
                    _admit_storage(plan)
                    status("admission", completed, completed_baselines, current)
                    _run(["preflight.py", "--device", plan["device"], "--job", current],
                         current + ".log", env)
                    status("running", completed, completed_baselines, current)
                    _run(["-m", "kd", "neuron-surgery-study", "--baseline", job["baseline"],
                          "--output", job["output"], "--study-config", job["config"],
                          "--device", plan["device"]], current + ".log", env)
                    status("archiving", completed, completed_baselines, current)
                    receipt = archives.archive_result(job)
                    completed.append(current)
                    status("queued", completed, completed_baselines, archive=receipt)
            status("complete", completed, completed_baselines)
        except BaseException as error:
            status("failed", completed, completed_baselines, current, error=repr(error))
            raise


if __name__ == "__main__":
    main()
