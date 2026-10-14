"""Adapt the existing validation-only repair study to the teacher benchmark.

The original repair implementation is used without edits. Its downstream KD and
test evaluation are disabled so the benchmark can enforce one common test gate.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import time

from .baseline_artifacts import TEACHER_FILES, read_json, verify_manifest, write_manifest
from .checkpoints import fingerprint, write_json
from .neuron_surgery_study import (NeuronSurgerySpec, resolve_target_classes,
                                   run_neuron_surgery_study)


REPAIR_NAME = "teacher_repair"
REPAIR_FILES = ("teacher_protocol.json", "config.json", "student.pt", "summary.json",
                "repair_decision.json", "repair/protocol.json", "repair/report_manifest.json")


@dataclass(frozen=True)
class TeacherRepairOptions:
    channel_budgets: tuple[int, ...] = (2, 4, 8)
    learning_rates: tuple[float, ...] = (0.001, 0.005)
    repair_epochs: int = 20
    repair_samples_per_epoch: int = 640
    repair_batch_size: int = 64
    target_fraction: float = 0.2
    companion_count: int = 5
    allow_sparse_class_fallback: bool = False

    def validate(self):
        for name in ("repair_epochs", "repair_samples_per_epoch", "repair_batch_size", "companion_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"repair.{name} must be a positive integer")
        if (not self.channel_budgets or any(not isinstance(n, int) or isinstance(n, bool) or n < 1
                                           for n in self.channel_budgets)
                or tuple(sorted(set(self.channel_budgets))) != self.channel_budgets):
            raise ValueError("Repair channel budgets must be increasing unique positive integers")
        if (not self.learning_rates or any(isinstance(n, bool) or not math.isfinite(n) or n <= 0
                                          for n in self.learning_rates)):
            raise ValueError("Repair learning rates must be finite and positive")
        if (isinstance(self.target_fraction, bool) or not math.isfinite(self.target_fraction)
                or not 0 < self.target_fraction <= 1
                or not isinstance(self.allow_sparse_class_fallback, bool)):
            raise ValueError("Invalid repair target fraction or sparse-class policy")


def repair_options(raw):
    raw = dict(raw)
    for name in ("channel_budgets", "learning_rates"):
        if name in raw:
            raw[name] = tuple(raw[name])
    result = TeacherRepairOptions(**raw)
    result.validate()
    return result


def repair_spec(study, seed):
    options = study.repair
    options.validate()
    data, train = study.teacher.data, study.teacher.train
    factor = {"lt-if10": 10.0, "lt-if50": 50.0, "lt-if100": 100.0}.get(
        data.dataset_profile, data.imbalance_factor)
    model = study.teacher.student.name
    stages = (("stage2", "stage3") if model in {"cifar_teacher", "cifar_student"}
              else ("stage3", "stage4"))
    if model not in {"cifar_teacher", "cifar_student", "resnet18", "resnet34", "resnet50",
                     "cifar_resnet18", "cifar_resnet34", "cifar_resnet50"}:
        raise ValueError("Teacher repair requires a CifarCNN or torchvision ResNet adapter")
    result = NeuronSurgerySpec(
        name=f"teacher-repair-seed{seed}", teacher_run=f"teacher_control_seed{seed}",
        student_run_template=f"kd_control_teacher{seed}_seed{{seed}}",
        teacher_model=model, student_model=study.student_model, teacher_seed=seed,
        student_seeds=study.student_seeds, split_seed=data.split_seed,
        confirmation_fraction=data.confirmation_fraction, training_epochs=train.epochs,
        target_classes=None, target_class_policy="frequency_tail" if factor > 1 else "all",
        stages=stages, downstream_kd=False, expected_train_counts=None, expected_target_counts=None,
        dataset_source=data.source, dataset_num_classes=data.num_classes,
        dataset_image_size=data.image_size, dataset_version=data.dataset_version,
        dataset_profile=(None if data.dataset_profile == "balanced" and factor > 1 else data.dataset_profile),
        imbalance_factor=factor, **asdict(options))
    result.validate()
    return result


def verify_repair_details(directory):
    directory = Path(directory)
    verify_manifest(directory, "completion.json", REPAIR_FILES)
    decision = read_json(directory / "repair_decision.json")
    exported = fingerprint(directory / "student.pt")
    if exported != decision["source_checkpoint_sha256"]:
        raise ValueError("Repair export differs from its selected source")
    if decision["status"] == "no_change" and exported != decision["base_teacher_sha256"]:
        raise ValueError("Rejected repair changed the original teacher")
    if decision["status"] == "selected" and exported != fingerprint(directory / "repair/teacher_repaired.pt"):
        raise ValueError("Repair export differs from selected repaired teacher")
    manifest = read_json(directory / "repair/report_manifest.json")
    if manifest["protocol_sha256"] != fingerprint(directory / "repair/protocol.json"):
        raise ValueError("Repair protocol digest changed")
    # The existing repair manifest uses a nested 'files' mapping.
    if not {"comparison.json", "report.md"}.issubset(manifest["files"]):
        raise ValueError("Repair report manifest omits required artifacts")
    for relative, digest in manifest["files"].items():
        file = Path(relative)
        if file.is_absolute() or ".." in file.parts:
            raise ValueError("Invalid repair artifact path")
        path = directory / "repair" / file
        if not path.is_file() or fingerprint(path) != digest:
            raise ValueError(f"Repair artifact changed or missing: {path}")


def repair_targets(study, counts):
    """Use the original repair protocol's target policy for all compared arms."""
    return resolve_target_classes(repair_spec(study, study.teacher_seeds[0]), counts)


def train_repair_teacher(study, config):
    spec = repair_spec(study, config.train.seed)
    root, directory = Path(config.output_dir), Path(config.output_dir) / config.name
    control = root / spec.teacher_run
    verify_manifest(control, "completion.json", TEACHER_FILES)
    base = read_json(control / "summary.json")
    identity = {
        "config": config.to_dict(), "repair": spec.to_dict(),
        "base_teacher_sha256": fingerprint(control / "student.pt"),
        "base_completion_sha256": fingerprint(control / "completion.json"),
    }
    # Normalize tuples to the JSON representation used for comparisons on resume.
    identity = json.loads(json.dumps(identity, allow_nan=False))
    path = directory / "teacher_protocol.json"
    if directory.exists() and any(directory.iterdir()):
        if not path.exists() or read_json(path) != identity:
            raise ValueError("Repair baseline identity changed; use a new output root")
        if (directory / "completion.json").exists():
            verify_repair_details(directory)
            return read_json(directory / "summary.json")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(path, identity)
    write_json(directory / "config.json", config.to_dict())
    started = time.perf_counter()
    result = run_neuron_surgery_study(root, directory / "repair", config.train.device,
                                     study.student_seeds, spec=spec)
    elapsed = time.perf_counter() - started
    if result["status"] not in {"repair_only_complete", "no_repair_selected"}:
        raise ValueError("Unexpected result from validation-only teacher repair")
    accepted = result["status"] == "repair_only_complete"
    source = directory / "repair/teacher_repaired.pt" if accepted else control / "student.pt"
    shutil.copyfile(source, directory / "student.pt")
    decision = {"status": "selected" if accepted else "no_change",
                "reason": result.get("reason"), "base_teacher_sha256": identity["base_teacher_sha256"],
                "source_checkpoint_sha256": fingerprint(source)}
    write_json(directory / "repair_decision.json", decision)
    summary = {"checkpoint": str(directory / "student.pt"),
               "checkpoint_sha256": fingerprint(directory / "student.pt"),
               "initial_student_sha256": base["initial_student_sha256"],
               "data": base["data"], "class_counts": base["class_counts"],
               "parameters": base["parameters"], "synthetic_smoke_only": False,
               "recipe": REPAIR_NAME, "repair": decision,
               "training_seconds": base["training_seconds"] + elapsed,
               "repair_phase_seconds": elapsed,
               "timing_scope": "base-teacher train/validation time plus successful repair invocation; excludes prior interrupted invocations"}
    write_json(directory / "summary.json", summary)
    write_manifest(directory, "completion.json", REPAIR_FILES)
    verify_repair_details(directory)
    return summary
