"""Isolated teacher-training interventions using the ordinary inference contract.

These experiments do not rewrite the original training engine or saved protocols.
Student KD continues to use ``engine.run_experiment``.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset
from torchvision import transforms

from .baseline_artifacts import TEACHER_FILES, verify_manifest, write_manifest
from .checkpoints import (capture_rng, fingerprint, load_model_checkpoint, metadata,
                          restore_rng, save_checkpoint, tensor_state_fingerprint, write_json)
from .config import ExperimentConfig
from .data import MEDMNIST_SOURCES, build_data, normalization
from .engine import seed_everything
from .evaluation import collect_predictions, prediction_metrics
from .models import create_model
from .runtime import (autocast_context, configure_accelerator, make_grad_scaler,
                      move_images, move_labels, prepare_model, resolve_device,
                      resolve_precision, runtime_metadata)


@dataclass(frozen=True)
class TeacherRecipe:
    name: str
    randaugment: bool = False
    randaugment_ops: int = 2
    randaugment_magnitude: int = 9
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    label_smoothing: float = 0.0
    loss: str = "cross_entropy"
    loss_scale: float = 1.0
    ema_decay: float = 0.0
    sam_rho: float = 0.0

    def validate(self):
        if not self.name or self.name in {".", ".."} or Path(self.name).name != self.name:
            raise ValueError("Recipe name must be a single directory component")
        if self.loss not in {"cross_entropy", "balanced_softmax", "mse_logits", "mse_probabilities"}:
            raise ValueError("Unknown teacher loss")
        for name in ("mixup_alpha", "cutmix_alpha", "sam_rho", "loss_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (self.loss_scale == 0 or isinstance(self.label_smoothing, bool)
                or isinstance(self.ema_decay, bool) or not 0 <= self.label_smoothing < 1
                or not 0 <= self.ema_decay < 1):
            raise ValueError("Invalid loss scale, smoothing, or EMA decay")
        if self.mixup_alpha and self.cutmix_alpha:
            raise ValueError("Use separate Mixup and CutMix arms")
        if self.loss != "cross_entropy" and self.label_smoothing:
            raise ValueError("Smoothing is a separate cross-entropy intervention")
        if (not isinstance(self.randaugment, bool)
                or not isinstance(self.randaugment_ops, int) or isinstance(self.randaugment_ops, bool)
                or self.randaugment_ops < 1
                or not isinstance(self.randaugment_magnitude, int)
                or isinstance(self.randaugment_magnitude, bool)
                or not 0 <= self.randaugment_magnitude <= 30):
            raise ValueError("Invalid RandAugment parameters")


RECIPES = {
    recipe.name: recipe for recipe in (
        TeacherRecipe("control"),
        TeacherRecipe("randaugment", randaugment=True),
        TeacherRecipe("mixup", mixup_alpha=0.2),
        TeacherRecipe("cutmix", cutmix_alpha=1.0),
        TeacherRecipe("label_smoothing", label_smoothing=0.1),
        TeacherRecipe("balanced_softmax", loss="balanced_softmax"),
        TeacherRecipe("mse_logits", loss="mse_logits"),
        TeacherRecipe("mse_probabilities", loss="mse_probabilities"),
        TeacherRecipe("sam", sam_rho=0.05),
        TeacherRecipe("ema", ema_decay=0.999),
        TeacherRecipe("randaugment_cutmix_ema", randaugment=True,
                      cutmix_alpha=1.0, ema_decay=0.999),
    )
}


def mix_batch(images, labels, classes: int, recipe: TeacherRecipe):
    """Mix labels with the *realized* CutMix area, including boundary clipping."""
    targets = F.one_hot(labels, classes).float()
    alpha = recipe.mixup_alpha or recipe.cutmix_alpha
    if not alpha or len(labels) < 2:
        return images, targets
    weight = float(np.random.beta(alpha, alpha))
    order = torch.randperm(len(labels), device=images.device)
    if recipe.mixup_alpha:
        mixed = weight * images + (1 - weight) * images[order]
    else:
        height, width = images.shape[-2:]
        ratio = math.sqrt(1 - weight)
        patch_h, patch_w = int(height * ratio), int(width * ratio)
        cy, cx = int(np.random.randint(height)), int(np.random.randint(width))
        y0, y1 = max(0, cy - patch_h // 2), min(height, cy + (patch_h + 1) // 2)
        x0, x1 = max(0, cx - patch_w // 2), min(width, cx + (patch_w + 1) // 2)
        mixed = images.clone()
        mixed[:, :, y0:y1, x0:x1] = images[order, :, y0:y1, x0:x1]
        weight = 1 - (y1 - y0) * (x1 - x0) / (height * width)
    return mixed, weight * targets + (1 - weight) * targets[order]


def teacher_loss(logits, targets, recipe: TeacherRecipe, class_counts):
    logits = logits.float()
    if recipe.loss == "mse_logits":
        loss = F.mse_loss(logits, targets)
    elif recipe.loss == "mse_probabilities":
        loss = F.mse_loss(logits.softmax(1), targets)
    else:
        if recipe.loss == "balanced_softmax":
            counts = torch.as_tensor(class_counts, device=logits.device, dtype=torch.float32)
            if counts.shape != (logits.shape[1],) or not bool((counts > 0).all()):
                raise ValueError("Balanced Softmax requires positive training counts for every class")
            logits = logits + counts.log()
        targets = targets * (1 - recipe.label_smoothing) + recipe.label_smoothing / logits.shape[1]
        loss = -(targets * logits.log_softmax(1)).sum(1).mean()
    return recipe.loss_scale * loss


_STANDARD_RANDAUGMENT_OPERATIONS = (
    "Identity", "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate",
    "Brightness", "Color", "Contrast", "Sharpness", "Posterize", "Solarize",
    "AutoContrast", "Equalize",
)


def randaugment_policy(source):
    """Declare operation space; restricted variants are not standard RandAugment.

    The photometric restrictions avoid geometric/colour substitutions on label-
    sensitive sources. They are conservative starting policies, not evidence of
    task-specific invariance or an optimized augmentation search.
    """
    if source in {"svhn", "gtsrb"}:
        name = "restricted_digit_sign_photometric"
        operations = ("Identity", "Brightness", "Contrast", "Sharpness", "AutoContrast", "Equalize")
    elif source in MEDMNIST_SOURCES:
        name = "restricted_medical_photometric"
        operations = ("Identity", "Brightness", "Contrast", "Sharpness")
    else:
        name = "torchvision_standard"
        operations = _STANDARD_RANDAUGMENT_OPERATIONS
    return {"name": name, "operations": list(operations)}


class PolicyRandAugment(transforms.RandAugment):
    """Use torchvision magnitudes/RNG with a recorded, bounded operation space."""

    def __init__(self, *, operations, **kwargs):
        super().__init__(**kwargs)
        self.operations = tuple(operations)
        if not self.operations or len(set(self.operations)) != len(self.operations):
            raise ValueError("RandAugment operations must be nonempty and unique")

    def _augmentation_space(self, num_bins, image_size):
        available = super()._augmentation_space(num_bins, image_size)
        missing = set(self.operations) - available.keys()
        if missing:
            raise ValueError(f"Unsupported RandAugment operations: {sorted(missing)}")
        return {name: available[name] for name in self.operations}


def add_randaugment(data, config, recipe):
    if not recipe.randaugment:
        return
    dataset = data.train.dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    operations = list(dataset.transform.transforms)
    position = next(i for i, transform in enumerate(operations) if isinstance(transform, transforms.ToTensor))
    mean, _ = normalization(config.data.source)
    policy = randaugment_policy(config.data.source)
    arguments = {"num_ops": recipe.randaugment_ops, "magnitude": recipe.randaugment_magnitude,
                 "fill": tuple(round(255 * value) for value in mean)}
    if policy["name"] == "torchvision_standard":
        augmentation = transforms.RandAugment(**arguments)
        available = augmentation._augmentation_space(augmentation.num_magnitude_bins,
                                                     (config.data.image_size,) * 2)
        if list(available) != policy["operations"]:
            raise ValueError("Torchvision RandAugment operation space changed; update the frozen policy")
    else:
        augmentation = PolicyRandAugment(operations=policy["operations"], **arguments)
    operations.insert(position, augmentation)
    dataset.transform = transforms.Compose(operations)


@torch.no_grad()
def update_ema(average, model, decay):
    """Average parameters; copy BN statistics and integer buffers from the model."""
    for average_parameter, parameter in zip(average.parameters(), model.parameters(), strict=True):
        average_parameter.lerp_(parameter, 1 - decay)
    for average_buffer, buffer in zip(average.buffers(), model.buffers(), strict=True):
        average_buffer.copy_(buffer)


def _backward(loss, model):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite teacher loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)


def sam_step(model, images, targets, recipe, counts, optimizer, device, precision):
    """Two SAM passes; update BN running statistics only on the first pass."""
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, precision):
        loss = teacher_loss(model(images).logits, targets, recipe, counts)
    _backward(loss, model)
    norm = torch.stack([parameter.grad.float().norm() for parameter in model.parameters()
                        if parameter.grad is not None]).norm()
    originals = [(parameter, parameter.detach().clone()) for parameter in model.parameters()
                 if parameter.grad is not None]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    try:
        with torch.no_grad():
            for parameter, _ in originals:
                parameter.add_(parameter.grad * (recipe.sam_rho / (norm + 1e-12)))
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            perturbed_loss = teacher_loss(model(images).logits, targets, recipe, counts)
        _backward(perturbed_loss, model)
    finally:
        with torch.no_grad():
            for parameter, original in originals:
                parameter.copy_(original)
            for buffer, original in buffers:
                buffer.copy_(original)
    optimizer.step()
    return loss.detach()


def _validation(model, data, device, config):
    labels, probabilities = collect_predictions(
        model, data.val, device, precision=config.train.precision,
        channels_last=config.train.channels_last)
    return prediction_metrics(labels, probabilities, data.classes)


def train_teacher(config: ExperimentConfig, recipe: TeacherRecipe, *, selection_metric="accuracy"):
    """Resume at epoch boundaries and export a normal ``student.pt`` teacher."""
    config.validate()
    recipe.validate()
    if config.distillation.method != "supervised":
        raise ValueError("Teacher baselines require supervised training")
    if config.train.persistent_workers:
        raise ValueError("Epoch-boundary teacher resume requires persistent_workers=false")
    if selection_metric not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("Unknown teacher checkpoint selection metric")
    device = resolve_device(config.train.device)
    precision = resolve_precision(config.train.precision, device)
    if recipe.sam_rho and precision == "float16":
        raise ValueError("SAM supports float32/bfloat16; set teacher precision explicitly")
    deterministic = config.train.cuda_mode == "deterministic"
    seed_everything(config.train.seed, deterministic=deterministic)
    configure_accelerator(device, config.train.cuda_mode)
    torch.set_num_threads(config.train.threads)
    directory = Path(config.output_dir) / config.name
    identity = {"config": config.to_dict(), "recipe": asdict(recipe),
                "selection_metric": selection_metric,
                "augmentation_policy": randaugment_policy(config.data.source) if recipe.randaugment else None,
                "initial_checkpoint_sha256": (fingerprint(config.student.checkpoint)
                                              if config.student.checkpoint else None),
                "source_sha256": {name: fingerprint(Path(__file__).parent / name)
                                  for name in ("teacher_training.py", "baseline_artifacts.py", "config.py",
                                               "checkpoints.py", "engine.py", "evaluation.py", "data.py",
                                               "models.py", "runtime.py")}}
    identity = json.loads(json.dumps(identity, allow_nan=False))
    config_path = directory / "teacher_protocol.json"
    if directory.exists() and any(directory.iterdir()):
        if not config_path.exists() or json.loads(config_path.read_text()) != identity:
            raise ValueError(f"Teacher protocol changed or directory is not a baseline run: {directory}")
        if (directory / "completion.json").exists():
            verify_manifest(directory, "completion.json", TEACHER_FILES)
            return json.loads((directory / "summary.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    write_json(config_path, identity)
    write_json(directory / "config.json", config.to_dict())
    data = build_data(config.data, config.train, include_test=False)
    add_randaugment(data, config, recipe)
    counts = data.provenance.get("train_per_class")
    if counts is None:
        # ImageFolder/synthetic adapters need a training-only count pass.
        diagnostic = build_data(config.data, replace(config.train, workers=0),
                                include_test=False, diagnostic=True)
        counts = [0] * config.data.num_classes
        for _, labels in diagnostic.train:
            for label, count in zip(*torch.unique(labels, return_counts=True)):
                counts[int(label)] += int(count)
    model = create_model(config.student.name, config.data.num_classes)
    checkpoint_metadata = metadata(config.student.name, data.classes, config.data.image_size, config.data.source)
    if config.student.checkpoint:
        load_model_checkpoint(model, config.student.checkpoint, checkpoint_metadata)
    initial = tensor_state_fingerprint(model.state_dict())
    prepare_model(model, device, config.train.channels_last)
    average = deepcopy(model).requires_grad_(False).eval() if recipe.ema_decay else None
    optimizer = torch.optim.SGD(model.parameters(), lr=config.train.learning_rate,
                                momentum=config.train.momentum, weight_decay=config.train.weight_decay,
                                fused=device.type == "cuda" and not deterministic)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)
    scaler = make_grad_scaler(device, precision)
    history, best_score, start_epoch = [], -1.0, 0
    loaders = {"train": data.train, "val": data.val}
    seed_everything(config.train.seed, deterministic=deterministic)
    last = directory / "last.pt"
    if last.exists():
        state = load_model_checkpoint(model, last, checkpoint_metadata)
        if state["identity"] != identity or state["initial_sha256"] != initial:
            raise ValueError("Teacher resume identity or initialization changed")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if average is not None:
            average.load_state_dict(state["ema"])
        history, best_score, start_epoch = state["history"], state["best_score"], state["epoch"] + 1
        restore_rng(state["rng"], loaders)
    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        total, count = 0.0, 0
        start = time.perf_counter()
        for images, labels in data.train:
            images = move_images(images, device, config.train.channels_last)
            labels = move_labels(labels, device)
            images, targets = mix_batch(images, labels, config.data.num_classes, recipe)
            stepped = True
            if recipe.sam_rho:
                loss = sam_step(model, images, targets, recipe, counts, optimizer, device, precision)
            else:
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, precision):
                    loss = teacher_loss(model(images).logits, targets, recipe, counts)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite teacher loss")
                scaler.scale(loss).backward()
                if not scaler.is_enabled():
                    nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= old_scale
            if average is not None and stepped:
                update_ema(average, model, recipe.ema_decay)
            total += float(loss.detach()) * len(labels)
            count += len(labels)
        validation = _validation(average if average is not None else model, data, device, config)
        history.append({"epoch": epoch + 1, "loss": total / count,
                        "seconds": time.perf_counter() - start, "validation": validation})
        scheduler.step()
        if validation[selection_metric] > best_score:
            best_score = validation[selection_metric]
            selected = average if average is not None else model
            save_checkpoint(directory / "best.pt", {**checkpoint_metadata, "kind": "inference",
                "epoch": epoch, "student": selected.state_dict(), "teacher_recipe": asdict(recipe)})
        save_checkpoint(last, {**checkpoint_metadata, "kind": "teacher_baseline_training", "epoch": epoch,
            "student": model.state_dict(), "ema": average.state_dict() if average is not None else None,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "rng": capture_rng(loaders), "history": history,
            "best_score": best_score, "identity": identity, "initial_sha256": initial})
        write_json(directory / "history.json", history)
        print(f"{config.name} epoch={epoch + 1}/{config.train.epochs} loss={total / count:.5f} "
              f"val_{selection_metric}={validation[selection_metric]:.4f}", flush=True)
    best = load_model_checkpoint(model, directory / "best.pt", checkpoint_metadata)
    best["student"] = {name: value.cpu() for name, value in model.state_dict().items()}
    save_checkpoint(directory / "student.pt", best)
    summary = {"checkpoint": str(directory / "student.pt"), "checkpoint_sha256": fingerprint(directory / "student.pt"),
               "initial_student_sha256": initial, "recipe": asdict(recipe), "data": data.provenance,
               "augmentation_policy": identity["augmentation_policy"],
               "class_counts": counts, "best_epoch": best["epoch"] + 1,
               "selection_metric": selection_metric, "validation": _validation(model, data, device, config),
               "training_seconds": sum(row["seconds"] for row in history),
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "synthetic_smoke_only": config.data.source == "synthetic",
               "runtime": runtime_metadata(device, config.train.precision, config.train.cuda_mode,
                                           config.train.channels_last)}
    write_json(directory / "summary.json", summary)
    write_manifest(directory, "completion.json", TEACHER_FILES)
    return summary
