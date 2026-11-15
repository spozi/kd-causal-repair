"""Prepare an immutable local pilot bundle. Never connects to a server or trains."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import tarfile

from kd.neuron_surgery_study import load_neuron_surgery_spec


ROOT = Path(__file__).resolve().parents[1]
DEFINITION = ROOT / "configs/flexible_screening_pilot.json"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(output, definition=DEFINITION):
    """Snapshot runtime code and four paired configs; refuse existing outputs."""
    plan = json.loads(Path(definition).read_text())
    if plan["policies"] != ["strict", "screened_then_ranked"] or len(plan["cases"]) != 2:
        raise ValueError("This frozen pilot requires two cases and two screening policies")
    # Validate everything before creating the output directory.
    jobs = []
    specs = {}
    remote = Path(plan["remote_root"])
    if not remote.is_absolute() or remote == Path("/"):
        raise ValueError("Pilot remote root must be a dedicated absolute directory")
    for case in plan["cases"]:
        original = load_neuron_surgery_spec(ROOT / case["template"])
        runs = {original.teacher_run, *(original.student_run_template.format(seed=seed)
                                      for seed in original.student_seeds)}
        if (set(case["expected_model_sha256"]) != runs
                or any(len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
                       for digest in case["expected_model_sha256"].values())):
            raise ValueError("Pin the exact original teacher and three KD checkpoint SHA256s")
        if not {"train_index_sha256", "val_index_sha256", "recipe_sha256", "artifact_sha256"} <= set(case["expected_data"]):
            raise ValueError("Pin original data splits, recipe and source artifacts")
        if (original.parameter_budget_fractions != (0.004, 0.008, 0.016, 0.032)
                or original.max_preservation_kl != 0.01
                or original.student_seeds != (42, 43, 44)
                or original.learning_rates != (0.001, 0.005)
                or original.training_epochs != 100 or original.repair_epochs != 20
                or original.repair_samples_per_epoch != 640 or original.repair_batch_size != 64
                or not original.downstream_kd):
            raise ValueError("Pilot template differs from the declared grid or student pairing")
        for policy in plan["policies"]:
            job_id = f"{case['id']}-{policy}"
            spec = replace(original, name=job_id, screening_policy=policy)
            spec.validate()
            specs[job_id] = spec
            jobs.append({"id": job_id, "case": case["id"], "screening_policy": policy,
                         "config": f"configs/{job_id}.json", "baseline": case["baseline"],
                         "output": str(remote / "studies" / job_id)})
    if len({job["id"] for job in jobs}) != 4 or len({job["output"] for job in jobs}) != 4:
        raise ValueError("Pilot cases must have distinct output identities")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a prepared pilot: {output}")
    output.mkdir(parents=True)
    write_json(output / "experiment.json", {**plan, "jobs": jobs})
    for job in jobs:
        write_json(output / job["config"], specs[job["id"]].to_dict())
    runtime = sorted((ROOT / "kd").rglob("*.py")) + sorted((ROOT / "kd/catalog").glob("*.json"))
    for source in runtime:
        target = output / "code" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ("pyproject.toml", "requirements-cuda.txt"):
        shutil.copyfile(ROOT / name, output / "code" / name)
    shutil.copyfile(ROOT / "scripts/flexible_screening_pilot_preflight.py", output / "preflight.py")
    shutil.copyfile(Path(__file__), output / "prepare-source.py")
    documentation = ROOT / "docs/experiments/flexible-screening-pilot-20261026.md"
    shutil.copyfile(documentation, output / "README.md")
    commands = ["# Run serially from the extracted bundle root, using the existing CUDA environment.",
                "# PYTHON must name that environment's Python executable. Nothing here installs dependencies.",
                "set -eu", ': "${PYTHON:?Set PYTHON to the existing CUDA Python executable}"',
                'export PYTHONPATH="$PWD/code"',
                '"$PYTHON" preflight.py --device ' + shlex.quote(plan["device"])]
    for job in jobs:
        # A separate per-job check enforces the disk admission guard as storage grows.
        commands.append('"$PYTHON" preflight.py --device ' + shlex.quote(plan["device"])
                        + " --job " + shlex.quote(job["id"]))
        commands.append('"$PYTHON" -m kd neuron-surgery-study --baseline '
                        + shlex.quote(job["baseline"]) + " --output " + shlex.quote(job["output"])
                        + " --study-config " + shlex.quote(job["config"]) + " --device "
                        + shlex.quote(plan["device"]))
    (output / "run-serial.sh").write_text("\n".join(commands) + "\n")
    manifest = {str(p.relative_to(output)): sha(p)
                for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", manifest)
    with tarfile.open(output / "bundle.tar.gz", "w:gz") as archive:
        for name in [*manifest, "manifest.json"]:
            archive.add(output / name, arcname=name, recursive=False)
    receipt = {"bundle_sha256": sha(output / "bundle.tar.gz"),
               "bundle_bytes": (output / "bundle.tar.gz").stat().st_size,
               "jobs": len(jobs), "maximum_candidate_fits": 32,
               "maximum_new_student_fits": 12, "remote_contacted": False,
               "training_started": False}
    write_json(output / "preparation.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/server-deployments/flexible-screening-pilot-20261026")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
