"""Required-file integrity checks for the isolated teacher benchmark."""

import json
from pathlib import Path

from .checkpoints import fingerprint, write_json


TEACHER_FILES = ("teacher_protocol.json", "config.json", "student.pt", "best.pt",
                 "last.pt", "summary.json", "history.json")
STUDENT_FILES = ("student.pt", "config.json", "data.json", "summary.json",
                 "best.pt", "last.pt", "history.json")


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_manifest(directory, name, required=()):
    directory = Path(directory)
    path = directory / name
    if not path.is_file():
        raise ValueError(f"Missing completion manifest: {path}")
    recorded = read_json(path)
    if not isinstance(recorded, dict) or not recorded or not set(required).issubset(recorded):
        raise ValueError(f"Completion manifest omits required artifacts: {path}")
    for relative, digest in recorded.items():
        file = Path(relative)
        if file.is_absolute() or ".." in file.parts:
            raise ValueError(f"Invalid artifact path in {path}: {relative}")
        target = directory / file
        if not target.is_file() or fingerprint(target) != digest:
            raise ValueError(f"Completed artifact changed or missing: {target}")
    return recorded


def write_manifest(directory, name, files):
    directory = Path(directory)
    write_json(directory / name, {file: fingerprint(directory / file) for file in files})
