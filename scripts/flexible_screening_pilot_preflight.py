"""Verify a prepared pilot on its target host. Never trains or starts a queue."""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlparse

sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_snapshot(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest.items():
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(root)):
            raise ValueError(f"Invalid snapshot path: {name}")
        if sha(path) != digest:
            raise ValueError(f"Frozen snapshot changed: {name}")
    required = {"experiment.json", "preflight.py", "code/kd/repair_screening.py",
                "code/kd/neuron_surgery_study.py", "code/kd/repair_budget.py"}
    if not required <= set(manifest):
        raise ValueError("Snapshot manifest omits required runtime files")
    return manifest


def inspect(root, device="cuda:0", job_id=None):
    """Dry-run adoption plus pinned checkpoints/data, GPU and storage checks."""
    root = Path(root).resolve()
    manifest = verify_snapshot(root)
    sys.path.insert(0, str(root / "code"))
    import torch
    import kd
    from kd.data import build_data
    from kd.neuron_surgery_study import _load_inputs, load_neuron_surgery_spec, run_neuron_surgery_study
    from torchvision.models import ResNet18_Weights

    if not Path(kd.__file__).resolve().is_relative_to(root / "code"):
        raise ValueError("Preflight imported a different project checkout")
    plan = json.loads((root / "experiment.json").read_text())
    remote = Path(plan["remote_root"])
    if root != remote.resolve():
        raise ValueError(f"Extract the prepared bundle at its frozen remote root: {remote}")
    jobs = [job for job in plan["jobs"] if job_id is None or job["id"] == job_id]
    if not jobs:
        raise ValueError(f"Unknown pilot job: {job_id}")
    free = shutil.disk_usage(root).free
    if free < plan["disk_free_minimum_gib"] * 1024**3:
        raise RuntimeError("Insufficient free disk for retained models; do not delete automatically")
    # Check for other compute processes before this preflight creates a CUDA context.
    if not device.startswith("cuda:"):
        raise ValueError("This frozen BF16 pilot requires an explicit CUDA device")
    gpu_index = int(device.split(":", 1)[1])
    active = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu_index), "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ], text=True).strip().splitlines()
    if any(row.strip().isdigit() and int(row.strip()) != os.getpid() for row in active):
        raise RuntimeError("GPU has another compute process; do not launch a competing pilot")
    if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
        raise RuntimeError("Requested CUDA device is unavailable")
    with torch.cuda.device(gpu_index):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested GPU does not support the inherited BF16 protocol")
    weights = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(ResNet18_Weights.IMAGENET1K_V1.url).path).name
    if not weights.is_file():
        raise FileNotFoundError("Restore the cached ResNet-18 initialization before launch; no implicit downloads")
    cases = {case["id"]: case for case in plan["cases"]}
    verified_cases = set()
    reports = []
    for job in jobs:
        if job["config"] not in manifest:
            raise ValueError("Job configuration is not frozen in the source manifest")
        spec = load_neuron_surgery_spec(root / job["config"])
        case = cases[job["case"]]
        if case["id"] not in verified_cases:
            for run, digest in case["expected_model_sha256"].items():
                path = Path(case["baseline"]) / run / "student.pt"
                if sha(path) != digest:
                    raise ValueError(f"Original model checkpoint changed: {path}")
            verified_cases.add(case["id"])
        dry = run_neuron_surgery_study(job["baseline"], job["output"], device,
                                       spec.student_seeds, dry_run=True, spec=spec)
        _, teacher, _, students = _load_inputs(job["baseline"], spec.student_seeds, spec)
        if teacher.data.download:
            raise ValueError("Frozen pilot must not implicitly download datasets")
        configurations = [teacher, *(row["config"] for row in students.values())]
        if any(config.train.precision != "bfloat16" or config.train.epochs != 100
               or config.train.batch_size != 128 or config.train.learning_rate != 0.001
               for config in configurations):
            raise ValueError("Available baselines differ from the frozen BF16/100-epoch recipe")
        data = build_data(teacher.data, replace(teacher.train, workers=0, device=device),
                          include_test=False, diagnostic=True)
        for key, expected in case["expected_data"].items():
            if data.provenance.get(key) != expected:
                raise ValueError(f"Baseline data identity changed: {case['id']} / {key}")
        directory = Path(job["output"])
        if directory.exists() and any(directory.iterdir()):
            recorded = json.loads((directory / "protocol.json").read_text())
            if recorded != dry["protocol"]:
                raise ValueError("Existing output belongs to a different protocol; use a fresh pilot root")
        reports.append({"id": job["id"], "policy": spec.screening_policy,
                        "baseline_model_hashes_verified": True, "data_verified": True,
                        "dry_run": dry})
    return {"ready": True, "training_started": False, "free_gib": free / 1024**3,
            "gpu": torch.cuda.get_device_name(gpu_index), "jobs": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--job", default=None)
    args = parser.parse_args()
    print(json.dumps(inspect(Path(__file__).resolve().parent, args.device, args.job), indent=2))


if __name__ == "__main__":
    main()
