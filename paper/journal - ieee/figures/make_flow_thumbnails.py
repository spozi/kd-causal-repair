"""Thumbnails for the repair overview in Section 3 (Steps 1 and 2 example).

Takes one CIFAR-10 hard image at imbalance factor 10, a truck that the original CNN-T teacher
labels an automobile, and its five companions, exactly as the repair procedure recorded them in
companions.json. It also takes three preservation images from the recorded causal preservation
subset, and checks that each is classified correctly and lies outside the hard set. Study indices
are mapped to the CIFAR-10 training archive through the frozen split file, and the mapping is
checked against the labels stored in diagnostics.npz.

Run from the repository root:  python paper/journal/figures/make_flow_thumbnails.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "runs/experiment1-multidataset/cifar10/lt-if10/studies/confirmatory"
DATA = ROOT / "data/cifar10/1.0"
TARGET = 8782  # study index of the hard truck; see companions.json
POOL_CLASSES = (0, 3, 8)  # airplane, cat and ship from the causal preservation subset
OUT = Path(__file__).with_name("flow")
SIZE = 128  # nearest-neighbour upscaling keeps the original 32x32 pixels visible


def payload(path):
    data = json.loads(Path(path).read_text())
    return data.get("payload", data)


def main():
    batches = [pickle.load(open(DATA / f"cifar-10-batches-py/data_batch_{i}", "rb"), encoding="latin1")
               for i in range(1, 6)]
    images = np.concatenate([b["data"] for b in batches]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray([label for b in batches for label in b["labels"]])
    train = np.load(DATA / "splits/lt-if10.npz")["train"]
    diagnostics = np.load(STUDY / "diagnostics.npz")
    if not np.array_equal(labels[train], diagnostics["labels"]):
        raise ValueError("Split indices do not reproduce the study's labels")

    record = next(r for r in payload(STUDY / "companions.json") if r["target"]["index"] == TARGET)
    target = record["target"]
    if (target["true_label"], target["predicted_label"]) != (9, 1):
        raise ValueError("Expected a truck (9) predicted as an automobile (1)")
    OUT.mkdir(exist_ok=True)
    targets = payload(STUDY / "targets.json")
    hard = {row["index"] for row in targets["targets"]}
    causal = targets["causal_preservation_indices"]
    pool = [next(i for i in causal if diagnostics["labels"][i] == c) for c in POOL_CLASSES]
    if any(i in hard or diagnostics["predictions"][i] != diagnostics["labels"][i] for i in pool):
        raise ValueError("Preservation images must be correct and outside the hard set")
    companions = [row["index"] for row in record["companions"]]
    indices = [TARGET] + companions + pool
    names = (["hard"] + [f"companion{i}" for i in range(1, len(companions) + 1)]
             + [f"pool{i}" for i in range(1, len(pool) + 1)])
    for name, index in zip(names, indices):
        Image.fromarray(images[train[index]]).resize((SIZE, SIZE), Image.NEAREST).save(OUT / f"{name}.png")
    print(f"target {TARGET}: truck predicted automobile, confidence {target['confidence']:.2f}")
    for row in record["companions"]:
        print(f"  companion {row['index']}: cosine similarity {1 - row['cosine_distance']:.3f}, correct={row['correct']}")
    print(f"preservation images {pool}: classes {POOL_CLASSES}")


if __name__ == "__main__":
    main()
