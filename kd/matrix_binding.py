"""Read-only execution guards for explicitly generated teacher-matrix studies.

The parent digest uses canonical JSON with only each study's ``matrix_binding``
removed. This avoids a self-referential digest while binding all computational
settings, outputs, job metadata, policies and source fingerprints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .checkpoints import fingerprint


BINDING_VERSION = 1
STUDY_SOURCE_FILES = (
    "teacher_baselines.py", "teacher_training.py", "baseline_artifacts.py",
    "teacher_repair_baseline.py", "neuron_surgery.py", "neuron_surgery_study.py", "repair_failure.py",
    "engine.py", "data.py", "dataset_registry.py", "catalog/v1.json", "config.py",
    "losses.py", "models.py", "checkpoints.py", "evaluation.py", "metrics.py",
    "calibration.py", "calibration_metrics.py", "adaptive_focal.py", "cpc.py",
    "surgery.py", "runtime.py", "training_provenance.py", "matrix_binding.py",
    "shortcut_data.py", "group_metrics.py", "shortcut_baselines.py", "shortcut_repair.py", "urban_metrics.py",
)
MATRIX_SOURCE_FILES = (
    "teacher_baseline_matrix.py", "neuron_surgery_matrix.py", "neuron_surgery_baselines.py",
    "full_benchmark.py",
)


def canonical_digest(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def computational_spec(value):
    """Return the JSON spec without execution-origin metadata."""
    return {key: item for key, item in value.items() if key != "matrix_binding"}


def matrix_payload_digest(protocol):
    payload = json.loads(json.dumps(protocol, allow_nan=False))
    for job in payload["jobs"]:
        job["study"] = computational_spec(job["study"])
    return canonical_digest(payload)


def _read_json(path, label):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Matrix binding requires an existing non-symlink {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Unreadable matrix binding {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Invalid matrix binding {label}: {path}")
    return value


def reject_unbound_matrix_config(path):
    """Reject recognized generated configs, never auto-bind standalone files.

    Old version-1 plans lack an execution binding and must be regenerated in a
    fresh directory. An explicitly copied standalone config elsewhere remains a
    standalone experiment, not an execution of the historical matrix queue.
    """
    lexical_path = Path(path).absolute()
    path = lexical_path.resolve()
    parent = lexical_path.parent.parent / "protocol.json"
    if lexical_path.parent.name != "configs" or not parent.is_file():
        return
    protocol = _read_json(parent, "parent protocol")
    for job in protocol.get("jobs", []):
        if isinstance(job, dict) and Path(job.get("config", "")).resolve() == path:
            raise ValueError("Generated matrix config is missing its execution binding; generate a fresh matrix plan")


def verify_matrix_binding(spec, output=None, *, config_path=None):
    """Verify explicit parent/config/output/source identities before execution."""
    binding = spec.matrix_binding
    if binding is None:
        if config_path is not None:
            reject_unbound_matrix_config(config_path)
        return
    keys = {"version", "parent_protocol", "parent_payload_sha256", "job_id", "config", "output"}
    if (not isinstance(binding, dict) or set(binding) != keys
            or type(binding.get("version")) is not int or binding.get("version") != BINDING_VERSION
            or any(not isinstance(binding.get(key), str) or not binding[key] for key in keys - {"version"})):
        raise ValueError("Invalid matrix execution binding")
    parent_path, bound_config = Path(binding["parent_protocol"]), Path(binding["config"])
    bound_output = Path(binding["output"])
    if any(not path.is_absolute() or str(path.resolve()) != str(path)
           for path in (parent_path, bound_config, bound_output)):
        raise ValueError("Matrix binding paths must retain their canonical absolute identities")
    if config_path is not None and Path(config_path).resolve() != bound_config:
        raise ValueError("Matrix binding config path changed")
    if output is not None and Path(output).resolve() != bound_output:
        raise ValueError("Matrix binding output changed; generate a fresh matrix plan")
    protocol = _read_json(parent_path, "parent protocol")
    if (protocol.get("version") != 2 or protocol.get("output") != str(parent_path.parent)
            or not isinstance(protocol.get("jobs"), list)):
        raise ValueError("Matrix binding requires a compatible version-2 parent protocol")
    try:
        digest = matrix_payload_digest(protocol)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid matrix binding parent payload") from error
    if digest != binding["parent_payload_sha256"]:
        raise ValueError("Matrix binding parent protocol changed")
    jobs = [job for job in protocol["jobs"] if job.get("id") == binding["job_id"]]
    if len(jobs) != 1:
        raise ValueError("Matrix binding job identity is missing or duplicated")
    job, actual_spec = jobs[0], spec.to_dict()
    if (job.get("config") != str(bound_config) or job.get("output") != str(bound_output)
            or bound_config.parent != parent_path.parent / "configs"
            or job.get("study") != actual_spec
            or job.get("spec_sha256") != canonical_digest(computational_spec(actual_spec))
            or spec.teacher.output_dir != str(bound_output)):
        raise ValueError("Matrix binding study/config/output identity changed")
    if _read_json(bound_config, "generated config") != actual_spec:
        raise ValueError("Matrix binding generated config changed")
    expected_sources = set(STUDY_SOURCE_FILES + MATRIX_SOURCE_FILES)
    sources = protocol.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        raise ValueError("Matrix binding source coverage changed")
    for name in sorted(expected_sources):
        if sources[name] != fingerprint(Path(__file__).parent / name):
            raise ValueError(f"Matrix binding computational source changed: {name}; generate a fresh matrix plan")
    # The queue is an execution artifact, not a second source of settings.
    from .teacher_baseline_matrix import _queue
    queue = parent_path.parent / "jobs.tsv"
    if queue.is_symlink() or not queue.is_file() or queue.read_text() != _queue(protocol):
        raise ValueError("Matrix binding frozen queue changed")
