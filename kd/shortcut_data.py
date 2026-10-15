"""Versioned local shortcut adapters, including legacy UrbanCars support.

No network acquisition occurs unless Colored MNIST explicitly opts into MNIST
download. Composite datasets are user-provisioned in their released layouts.
The old dataset catalog and historical matrix identities are not changed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .checkpoints import fingerprint
from .runtime import loader_performance_kwargs


SHORTCUT_DATASETS = {
    "colored_mnist": {"classes": 2, "image_size": 32, "version": "1.0", "groups": 4},
    "waterbirds": {"classes": 2, "image_size": 224, "version": "1.0", "groups": 4},
    "multiceleba": {"classes": 2, "image_size": 224, "version": "1.0", "groups": 8},
    "urbancars": {"classes": 2, "image_size": 256, "version": "1.0", "groups": 8},
}
DEFAULT_SHORTCUT_DATASETS = ("colored_mnist", "waterbirds", "multiceleba")
# Author's two-bias recipe, indexed by (High_Cheekbones, Young, Male).
# None retains the entire majority cell; all capped cells use filename order.
MULTICELEBA_CAPS = {(1, 1, 0): None, (1, 1, 1): 2200, (1, 0, 0): 2200,
                   (1, 0, 1): 110, (0, 1, 0): 40, (0, 1, 1): 800,
                   (0, 0, 0): 800, (0, 0, 1): 16220}
CMNIST_RECIPE = {"digit_label": "digit < 5", "label_noise": 0.25,
                 "train_color_mismatch": [0.2, 0.1],
                 "test_color_mismatch": {"reversed": 0.9, "unbiased": 0.5},
                 "renderer": "RGB red/green digit, blue zero, resize 28 to 32",
                 "label_policy": "noisy observed label", "version": 1}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def shortcut_transform(config, *, training=False):
    from .data import _ensure_rgb
    if config.source == "colored_mnist":
        operations = [transforms.Resize((32, 32))]
        mean, std = (0.5,) * 3, (0.5,) * 3
    elif config.source == "waterbirds":
        # Fixed author-style basic geometry; extra augmentation is a teacher recipe.
        operations = [transforms.Resize((256, 256)), transforms.CenterCrop(224)]
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif config.source == "multiceleba":
        operations = [transforms.CenterCrop(178), transforms.Resize((224, 224))]
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        operations = []
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if training and config.horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip())
    return transforms.Compose([transforms.Lambda(_ensure_rgb), *operations,
                               transforms.ToTensor(), transforms.Normalize(mean, std)])


class ColoredMNIST(Dataset):
    classes = ["digit_5_9", "digit_0_4"]
    num_groups = 4

    def __init__(self, images, digits, indices, targets, colors, environments, transform, split):
        self.images = images
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64)[self.indices]
        self.clean_targets = (np.asarray(digits)[self.indices] < 5).astype(np.int64)
        self.colors = np.asarray(colors, dtype=np.int64)[self.indices]
        self.environment_ids = np.asarray(environments, dtype=np.int64)[self.indices]
        self.nuisance_labels = self.colors[:, None]
        self.group_ids = self.targets * 2 + self.colors
        self.source_ids = np.asarray([f"mnist/{split}/{i}" for i in self.indices])
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        gray = np.asarray(self.images[int(self.indices[position])], dtype=np.uint8)
        rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
        rgb[:, :, int(self.colors[position])] = gray
        return self.transform(Image.fromarray(rgb)), int(self.targets[position])


class CompositeImages(Dataset):
    def __init__(self, root, rows, source, transform):
        self.root, self.rows, self.source, self.transform = root, rows, source, transform
        self.classes = {"waterbirds": ["landbird", "waterbird"],
                        "multiceleba": ["no_high_cheekbones", "high_cheekbones"],
                        "urbancars": ["urban", "country"]}[source]
        self.num_groups = SHORTCUT_DATASETS[source]["groups"]
        self.targets = np.asarray([row["label"] for row in rows], dtype=np.int64)
        self.nuisance_labels = np.asarray([row["nuisance"] for row in rows], dtype=np.int64)
        multiplier = 2 ** self.nuisance_labels.shape[1]
        nuisance_code = self.nuisance_labels @ (2 ** np.arange(self.nuisance_labels.shape[1] - 1, -1, -1))
        self.group_ids = self.targets * multiplier + nuisance_code
        self.source_ids = np.asarray([row["foreground_id"] for row in rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        row = self.rows[position]
        with Image.open(self.root / row["path"]) as opened:
            image = opened.convert("RGB")
            if self.source == "urbancars" and image.size != (256, 256):
                raise ValueError("UrbanCars composites must be native 256 x 256; do not silently resize")
            if self.source == "multiceleba" and image.size != (178, 218):
                raise ValueError("MultiCelebA requires aligned CelebA images at width 178 x height 218")
            return self.transform(image), int(row["label"])


def _root(config):
    base = Path(config.root).expanduser().resolve()
    versioned = base / config.source / (config.dataset_version or "1.0")
    candidates = [versioned, base / config.source, base]
    if config.source == "waterbirds":
        for root in candidates:
            if (root / "metadata.csv").is_file():
                return root.resolve()
    elif config.source == "urbancars":
        for root in candidates:
            if (root / "bg-0.95_co_occur_obj-0.95").is_dir():
                return root.resolve()
    elif config.source == "multiceleba":
        for root in candidates:
            if any((root / name).is_file() for name in ("list_attr_celeba.csv", "list_attr_celeba.txt")):
                return root.resolve()
    else:
        return versioned if config.dataset_version else base / config.source
    raise ValueError(f"Missing local {config.source} release under {versioned}; prepare the released layout before training")


def _safe_path(root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ValueError("Dataset filenames must be relative, contained paths")
    if not (root / path).resolve().is_relative_to(root):
        raise ValueError("Dataset image/symlink escapes its release root")
    return path.as_posix()


def _waterbirds_rows(root):
    rows = {name: [] for name in ("train", "val", "test")}
    names = {0: "train", 1: "val", 2: "test"}
    seen = set()
    with (root / "metadata.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"img_filename", "y", "place", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Waterbirds metadata requires img_filename, y, place, split")
        for record in reader:
            label, place, split = int(record["y"]), int(record["place"]), int(record["split"])
            if label not in (0, 1) or place not in (0, 1) or split not in names:
                raise ValueError("Waterbirds metadata has invalid binary labels or split IDs")
            path = _safe_path(root, record["img_filename"])
            # Waterbirds' foreground filename identifies the underlying CUB bird.
            foreground = record.get("foreground_id") or path
            if path in seen:
                raise ValueError("Duplicate Waterbirds image identity, including across splits")
            seen.add(path)
            rows[names[split]].append({"path": path, "label": label, "nuisance": [place],
                                      "foreground_id": foreground,
                                      "background_id": record.get("background_id")})
    return rows, {"metadata_sha256": fingerprint(root / "metadata.csv"),
                  "constituent_audit": "foreground identity; background lineage unavailable unless supplied"}


def _urbancars_rows(root, *, include_test):
    rows = {}
    for split in ("train", "val", "test"):
        if split == "test" and not include_test:
            continue
        ratio = "0.95" if split == "train" else "0.5"
        split_root = root / f"bg-{ratio}_co_occur_obj-{ratio}" / split
        rows[split] = []
        for label, obj in enumerate(("urban", "country")):
            for bg, background in enumerate(("urban", "country")):
                for co, coobject in enumerate(("urban", "country")):
                    directory = split_root / f"obj-{obj}_bg-{background}_co_occur_obj-{coobject}"
                    if not directory.is_dir():
                        raise ValueError(f"Missing released UrbanCars group folder: {directory}")
                    for image in sorted(directory.glob("*.jpg")):
                        path = _safe_path(root, image.relative_to(root).as_posix())
                        rows[split].append({"path": path, "label": label, "nuisance": [bg, co],
                                            "foreground_id": path})
    return rows, {"generation_policy": {"train_bg_agreement": 0.95, "train_coobject_agreement": 0.95,
                                        "val_test_bg_agreement": 0.5, "val_test_coobject_agreement": 0.5},
                  "constituent_audit": "released composite IDs; generator does not export original source IDs; no paired-foreground claim"}


def _multiceleba_rows(root, *, include_test):
    """Author two-bias subset; official validation/test, never re-stratified.

    Accept the author's CSV layout or original CelebA TXT annotations. Unlike
    the author code's positional arrays, join by filename and validate splits.
    Group IDs use this project's MSB-first order: 4*y + 2*Young + Male.
    """
    root = Path(root).resolve()
    attr_path = root / "list_attr_celeba.csv"
    if attr_path.is_file():
        with attr_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"image_id", "High_Cheekbones", "Young", "Male"}.issubset(reader.fieldnames or []):
                raise ValueError("MultiCelebA attributes require image_id, High_Cheekbones, Young, Male")
            attributes = list(reader)
    else:
        attr_path = root / "list_attr_celeba.txt"
        with attr_path.open() as handle:
            declared = int(next(handle).strip())
            names = next(handle).split()
            if not {"High_Cheekbones", "Young", "Male"}.issubset(names) or len(set(names)) != len(names):
                raise ValueError("Invalid MultiCelebA attribute header")
            attributes = []
            for line in handle:
                parts = line.split()
                if len(parts) != len(names) + 1:
                    raise ValueError("Invalid MultiCelebA attribute row")
                attributes.append(dict(zip(["image_id", *names], parts)))
            if len(attributes) != declared:
                raise ValueError("MultiCelebA attribute count differs from header")
    split_path = root / "list_eval_partition.csv"
    if split_path.is_file():
        with split_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"image_id", "partition"}.issubset(reader.fieldnames or []):
                raise ValueError("MultiCelebA splits require image_id and partition")
            partitions = [(row["image_id"], int(row["partition"])) for row in reader]
    else:
        split_path = root / "list_eval_partition.txt"
        with split_path.open() as handle:
            partitions = []
            for line in handle:
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError("Invalid MultiCelebA partition row")
                partitions.append((parts[0], int(parts[1])))
    split_by_id = dict(partitions)
    if len(split_by_id) != len(partitions) or any(split not in (0, 1, 2) for _, split in partitions):
        raise ValueError("Duplicate MultiCelebA identity or invalid partition")
    image_ids = [row["image_id"] for row in attributes]
    if len(set(image_ids)) != len(image_ids) or set(image_ids) != set(split_by_id):
        raise ValueError("MultiCelebA attribute/split identities must match uniquely")
    rows = {name: [] for name in (("train", "val", "test") if include_test else ("train", "val"))}
    selected = {cell: 0 for cell in MULTICELEBA_CAPS}
    for record in sorted(attributes, key=lambda row: row["image_id"]):
        name = record["image_id"]
        # Only accept canonical aligned CelebA filenames, not arbitrary paths.
        if len(name) != 10 or not name[:6].isdigit() or not name.endswith(".jpg"):
            raise ValueError("MultiCelebA requires canonical six-digit JPG filenames")
        values = tuple(int(record[key]) for key in ("High_Cheekbones", "Young", "Male"))
        if any(value not in (-1, 1) for value in values):
            raise ValueError("MultiCelebA attributes must be -1/+1")
        label, young, male = cell = tuple(int(value == 1) for value in values)
        split = ("train", "val", "test")[split_by_id[name]]
        if split == "train":
            cap = MULTICELEBA_CAPS[cell]
            if cap is not None and selected[cell] >= cap:
                continue
            selected[cell] += 1
        if split not in rows:
            continue
        path = _safe_path(root, f"img_align_celeba/{name}")
        rows[split].append({"path": path, "label": label, "nuisance": [young, male],
                            "foreground_id": name})
    return rows, {"metadata_sha256": {"attributes": fingerprint(attr_path), "partitions": fingerprint(split_path)},
        "construction": {"target": "High_Cheekbones", "nuisance_order": ["Young", "Male"],
            "train_selection": "author two-bias caps in ascending filename order within official train",
            "train_caps": [{"cell": list(cell), "cap": cap} for cell, cap in MULTICELEBA_CAPS.items()],
            "validation_test": "official CelebA partitions, unresampled",
            "group_id": "4*High_Cheekbones + 2*Young + Male"},
        "constituent_audit": "unique image IDs across partitions; celebrity identities may span splits; no person-disjoint claim"}


def _image_identity(root, rows):
    identities = []
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            raise ValueError(f"Missing dataset image: {path}")
        identities.append({**row, "image_sha256": fingerprint(path)})
    return digest(identities)


def _audit_foregrounds(rows):
    seen = {}
    for split, records in rows.items():
        for row in records:
            foreground = row["foreground_id"]
            if foreground in seen and seen[foreground] != split:
                raise ValueError("Underlying foreground reused across fitting/selection/test splits")
            seen[foreground] = split


def build_shortcut_data(config, train, *, include_test=True, diagnostic=False):
    from .data import DataBundle, seed_worker, stratified_split
    info = SHORTCUT_DATASETS[config.source]
    if (config.num_classes != 2 or config.image_size != info["image_size"]
            or config.dataset_profile not in {"balanced", "native-shortcut"}
            or config.imbalance_factor != 1 or config.confirmation_fraction
            or config.augmentation != "basic"):
        raise ValueError("Shortcut adapter requires its frozen native dimensions, basic views, and no class resampling")
    root = _root(config)
    splits, identity = {}, {"source": config.source, "dataset_version": config.dataset_version or "1.0",
                           "dataset_profile": "native-shortcut", "adapter_version": 1,
                           "adapter_sha256": fingerprint(Path(__file__)),
                           "split_seed": config.split_seed, "group_count": SHORTCUT_DATASETS[config.source]["groups"],
                           "group_definition": "label x nuisance (binary, label most significant)",
                           "preprocessing": {"image_size": config.image_size, "augmentation": config.augmentation,
                                             "horizontal_flip": config.horizontal_flip}}
    if config.source == "colored_mnist":
        base = datasets.MNIST(str(root), train=True, download=config.download)
        digits = np.asarray(base.targets, dtype=np.int64)
        clean = (digits < 5).astype(np.int64)
        indices, heldout = stratified_split(clean, config.validation_fraction, config.split_seed)
        rng = np.random.default_rng(config.split_seed)
        observed = clean ^ (rng.random(len(clean)) < 0.25)
        environments = rng.permutation(np.arange(len(clean))) % 2
        rates = np.asarray([0.2, 0.1])[environments]
        colors = observed ^ (rng.random(len(clean)) < rates)
        for split, selected in (("train", indices), ("val", heldout)):
            splits[split] = ColoredMNIST(base.data, digits, selected, observed, colors, environments,
                                        shortcut_transform(config, training=split == "train" and not diagnostic), "train")
        h = hashlib.sha256(np.asarray(base.data, dtype=np.uint8).tobytes() + digits.astype("<i8").tobytes())
        identity.update(mnist_train_sha256=h.hexdigest(), construction=CMNIST_RECIPE,
                        construction_sha256=digest(CMNIST_RECIPE), validation_fraction=config.validation_fraction)
        if include_test:
            base_test = datasets.MNIST(str(root), train=False, download=config.download)
            digits_test = np.asarray(base_test.targets, dtype=np.int64)
            rng_test = np.random.default_rng(config.split_seed + 1_000_000)
            observed_test = (digits_test < 5).astype(np.int64) ^ (rng_test.random(len(digits_test)) < 0.25)
            draws = rng_test.random(len(digits_test))
            for environment, rate in CMNIST_RECIPE["test_color_mismatch"].items():
                splits["test_" + environment] = ColoredMNIST(
                    base_test.data, digits_test, np.arange(len(digits_test)), observed_test,
                    observed_test ^ (draws < rate), np.full(len(digits_test), -1),
                    shortcut_transform(config), "test")
            identity["evaluation_only"] = {"mnist_test_sha256": hashlib.sha256(
                np.asarray(base_test.data, dtype=np.uint8).tobytes() + digits_test.astype("<i8").tobytes()).hexdigest()}
    else:
        if config.download:
            raise ValueError("Local shortcut releases require explicit provisioning; automatic download is not supported")
        if config.source == "waterbirds":
            rows, release = _waterbirds_rows(root)
        elif config.source == "multiceleba":
            rows, release = _multiceleba_rows(root, include_test=include_test)
        else:
            rows, release = _urbancars_rows(root, include_test=include_test)
        _audit_foregrounds(rows)
        identity.update(release)
        for split, records in rows.items():
            if split == "test" and not include_test:
                continue
            splits[split] = CompositeImages(root, records, config.source,
                shortcut_transform(config, training=split == "train" and not diagnostic))
            field = "evaluation_only" if split == "test" else "image_sha256"
            identity.setdefault(field, {})[split] = _image_identity(root, records)
    for split in ("train", "val"):
        dataset = splits[split]
        counts = np.bincount(dataset.group_ids, minlength=dataset.num_groups)
        if np.any(counts == 0):
            raise ValueError(f"{split} is missing required shortcut groups: {np.flatnonzero(counts == 0).tolist()}")
        identity[split + "_per_group"] = counts.tolist()
        identity[split + "_per_class"] = np.bincount(dataset.targets, minlength=2).tolist()
        identity[split + "_identity_sha256"] = digest({"ids": dataset.source_ids.tolist(),
            "labels": dataset.targets.tolist(), "groups": dataset.group_ids.tolist(),
            "nuisance": dataset.nuisance_labels.tolist()})
    loaders = {name: DataLoader(dataset, batch_size=train.batch_size,
        shuffle=name == "train" and not diagnostic, num_workers=train.workers,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(train.seed),
        **loader_performance_kwargs(train.device, train.workers,
            persistent_workers=train.persistent_workers, prefetch_factor=train.prefetch_factor))
        for name, dataset in splits.items()}
    identity["split_sizes"] = {name: len(dataset) for name, dataset in splits.items()}
    endpoints = ({name.removeprefix("test_"): loader for name, loader in loaders.items() if name.startswith("test_")}
                 if config.source == "colored_mnist" else {"released": loaders["test"]} if "test" in loaders else {})
    selected = (config.test_environment or "reversed") if config.source == "colored_mnist" else "released"
    return DataBundle(loaders["train"], loaders["val"], splits["train"].classes,
                      endpoints.get(selected), identity, test_endpoints=endpoints)


def main(argv=None):
    import argparse
    from .config import DataConfig, TrainConfig
    parser = argparse.ArgumentParser(description="Validate locally prepared shortcut datasets; optional explicit MNIST download")
    parser.add_argument("source", choices=tuple(SHORTCUT_DATASETS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--download-mnist", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args(argv)
    if args.download_mnist and args.source != "colored_mnist":
        parser.error("--download-mnist is for Colored MNIST only; prepare composite releases locally")
    info = SHORTCUT_DATASETS[args.source]
    config = DataConfig(source=args.source, root=args.root, num_classes=2, image_size=info["image_size"],
                        dataset_version="1.0", dataset_profile="native-shortcut", download=args.download_mnist)
    data = build_shortcut_data(config, TrainConfig(device="cpu", workers=0), include_test=args.include_test, diagnostic=True)
    print(json.dumps(data.provenance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
