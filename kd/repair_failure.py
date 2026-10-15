"""Best-effort, non-resumable forensic snapshots for failed repair steps.

These artifacts deliberately cannot masquerade as inference or completion
checkpoints. A pre-step copy and the observed failure state are kept separately;
the latter may contain partially updated parameters or non-finite values.
Failures retain a separate CPU-only ``pre_step.pt`` even when the full snapshot
succeeds, trading extra failure-artifact disk space for poisoned-device resilience.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
import math
from numbers import Integral, Real
from pathlib import Path
import tempfile
import traceback

import torch

from .checkpoints import capture_rng, fingerprint, save_checkpoint, write_json


def _cpu_copy(value):
    """Detach aliases and restrict payloads to weights-only-safe structures."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if value is None or isinstance(value, (str, bool, bytes)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key is not None and not isinstance(key, (str, int, float, bool)):
                raise TypeError(f"Unsupported diagnostic mapping key: {type(key).__name__}")
            result[key] = _cpu_copy(item)
        return result
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"
    return value


def _tensor_l2(value: torch.Tensor) -> float:
    """Use scaled float64 accumulation, not a possibly overflowing FP32 norm."""
    value = value.detach().cpu()
    if torch.isnan(value).any():
        return float("nan")
    if torch.isinf(value).any():
        return float("inf")
    magnitude = (value.to(torch.complex128).abs() if value.is_complex()
                 else value.to(torch.float64).abs())
    if torch.isinf(magnitude).any():
        return float("inf")
    scale = magnitude.max().item() if magnitude.numel() else 0.0
    return torch.linalg.vector_norm(magnitude / scale).item() * scale if scale else 0.0


def _tensor_summary(value: torch.Tensor) -> dict:
    value = value.detach().cpu()
    finite = torch.isfinite(value)
    finite_values = value[finite]
    if value.is_complex():
        # Complex tensors are unusual here; a real magnitude range is explicit.
        finite_values = finite_values.abs()
    result = {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "numel": value.numel(), "finite_count": int(finite.sum().item()),
        "nonfinite_count": int((~finite).sum().item()),
        "nan_count": int(torch.isnan(value).sum().item()),
        "posinf_count": int(torch.isposinf(value).sum().item()) if not value.is_complex() else None,
        "neginf_count": int(torch.isneginf(value).sum().item()) if not value.is_complex() else None,
        "min": _finite_json(finite_values.min().item()) if finite_values.numel() else None,
        "max": _finite_json(finite_values.max().item()) if finite_values.numel() else None,
    }
    if value.numel() == 1:
        scalar = value.item()
        result["value"] = str(scalar) if isinstance(scalar, complex) else _finite_json(scalar)
    if value.is_complex():
        result["range_kind"] = "magnitude"
    if value.is_floating_point() or value.is_complex():
        result["l2_norm"] = _finite_json(_tensor_l2(value))
    return result


def _summary(value):
    """JSON-safe summaries; never retain potentially huge feature activations."""
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value)
    if hasattr(value, "logits") and hasattr(value, "features"):
        return {"logits": _summary(value.logits), "features": _summary(value.features)}
    if isinstance(value, Mapping):
        return {str(key): _summary(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_summary(item) for item in value]
    if isinstance(value, bytes):
        return repr(value)
    return _finite_json(_cpu_copy(value))


def _state(value):
    return None if value is None else _cpu_copy(value.state_dict())


def _cpu_metadata(value):
    """Fallback metadata must not touch a potentially poisoned accelerator."""
    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        return {"omitted": "non_cpu_metadata_tensor", "device": str(value.device),
                "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {str(key): _cpu_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_cpu_metadata(item) for item in value]
    try:
        return _cpu_copy(value)
    except TypeError:
        return {"omitted_type": type(value).__name__}


def _nonfinite_names(state: dict | None) -> list[str]:
    return [str(name) for name, value in (state or {}).items()
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())]


def _momentum_names(candidate, optimizer, optimizer_state: dict | None) -> list[str]:
    if optimizer is None or optimizer_state is None:
        return []
    names = {id(parameter): name for name, parameter in candidate.named_parameters()}
    parameter_names = {}
    for live_group, stored_group in zip(optimizer.param_groups, optimizer_state["param_groups"]):
        for parameter, index in zip(live_group["params"], stored_group["params"]):
            parameter_names[index] = names.get(id(parameter), str(index))
    return [f"{parameter_names.get(index, index)}.{key}"
            for index, state in optimizer_state["state"].items()
            for key, value in state.items()
            if "momentum" in str(key).lower() and isinstance(value, torch.Tensor)
            and not bool(torch.isfinite(value).all())]


class RepairFailureRecorder:
    """Capture failing steps without changing their exception or training state.

    Step counters are one-based. Capturing a step does not write files or consume
    random draws. A successful ``save`` returns its unique artifact directory;
    a diagnostic failure is attached to the original exception as a note.
    """

    def __init__(self, directory: str | Path, *, context: dict | None = None):
        self.directory = Path(directory)
        self.context = context or {}
        self._location = {"epoch": None, "step": None, "global_step": None}
        self._pre_step = None
        self._capture_errors = []

    def begin_step(self, candidate, optimizer, scheduler, scaler, *, epoch: int,
                   step: int, global_step: int, samplers: dict | None = None):
        self._location = {"epoch": epoch, "step": step, "global_step": global_step}
        self._pre_step = None
        self._capture_errors = []
        try:
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
                   for value in self._location.values()):
                raise ValueError("Repair snapshot step counters must be positive one-based integers")
            self._pre_step = {
                "candidate": _state(candidate), "optimizer": _state(optimizer),
                "scheduler": _state(scheduler), "scaler": _state(scaler),
                "rng": _cpu_copy(capture_rng({})), "samplers": _cpu_copy(samplers),
            }
        except Exception as error:
            self._capture_errors.append(f"begin_step: {type(error).__name__}: {error}")

    def before_forward(self):
        try:
            if self._pre_step is not None:
                self._pre_step["before_forward_rng"] = _cpu_copy(capture_rng({}))
        except Exception as error:
            self._capture_errors.append(f"before_forward: {type(error).__name__}: {error}")

    def save(self, error: Exception, *, phase: str, candidate, anchor, optimizer,
             scheduler, scaler, losses: dict, batches: dict, outputs: dict | None = None,
             history: list, samplers: dict | None = None, masks: dict | None = None,
             settings: dict | None = None) -> Path | None:
        snapshot = None
        receipt = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            snapshot = Path(tempfile.mkdtemp(
                prefix=f"epoch-{self._location['epoch']}-step-{self._location['step']}-",
                dir=self.directory))
            receipt = {
                "format_version": 1, "kind": "repair_failure_snapshot",
                "status": "failed", "resumable": False, **self._location,
                "phase": phase, "created_utc": datetime.now(timezone.utc).isoformat(),
                "exception": {"type": type(error).__name__, "message": str(error),
                              "traceback": "".join(traceback.format_exception(error))},
                "diagnostic_source_sha256": fingerprint(Path(__file__)),
                "source_sha256": {name: fingerprint(Path(__file__).with_name(name))
                                  for name in ("repair_failure.py", "neuron_surgery.py")},
                "capture_errors": list(self._capture_errors),
                "capture_status": "capturing", "checkpoint": None,
                "checkpoint_sha256": None, "losses": None,
                "pre_step_checkpoint": None, "pre_step_checkpoint_sha256": None,
                "pre_step_checkpoint_scope": "pre_step_only",
                "pre_step_checkpoint_resumable": False,
            }
            # Leave a readable failure receipt even if copying a poisoned device
            # state or writing the much larger checkpoint subsequently fails.
            write_json(snapshot / "failure.json", receipt)
            if self._pre_step is not None:
                # This state was already detached to CPU before the failing step.
                # Save it before inspecting losses or any current device state.
                fallback = snapshot / "pre_step.pt"
                save_checkpoint(fallback, {
                    "format_version": 1, "kind": "repair_failure_snapshot",
                    "status": "failed", "resumable": False,
                    "snapshot_scope": "pre_step_only", **self._location,
                    "phase": phase, "exception": receipt["exception"],
                    "source_sha256": receipt["source_sha256"],
                    "diagnostic_source_sha256": receipt["diagnostic_source_sha256"],
                    "context": _cpu_metadata(self.context),
                    "settings": _cpu_metadata(settings), "pre_step": self._pre_step,
                    "capture_errors": list(self._capture_errors),
                })
                receipt.update(pre_step_checkpoint=fallback.name,
                               pre_step_checkpoint_sha256=fingerprint(fallback))
                write_json(snapshot / "failure.json", receipt)
            loss_summary = _summary(losses)
            receipt["losses"] = loss_summary
            write_json(snapshot / "failure.json", receipt)
            payload = {
                **receipt, "capture_status": "committed", "checkpoint": "state.pt",
                "context": _cpu_copy(self.context),
                "settings": _cpu_copy(settings), "pre_step": self._pre_step,
                "observed": {
                    "candidate": _state(candidate), "anchor": _state(anchor),
                    "gradients": {name: _cpu_copy(parameter.grad)
                                  for name, parameter in candidate.named_parameters()},
                    "optimizer": _state(optimizer), "scheduler": _state(scheduler),
                    "scaler": _state(scaler), "rng": _cpu_copy(capture_rng({})),
                    "samplers": _cpu_copy(samplers),
                },
                "batches": _cpu_copy(batches), "losses": _cpu_copy(losses),
                "outputs": _summary(outputs), "history": _cpu_copy(history),
                "masks": _cpu_copy(masks),
            }
            gradients = payload["observed"]["gradients"]
            gradient_l2 = 0.0
            for gradient in gradients.values():
                if gradient is not None:
                    norm = _tensor_l2(gradient)
                    if math.isnan(norm):
                        gradient_l2 = norm
                        break
                    gradient_l2 = math.hypot(gradient_l2, norm)
            diagnostics = {
                "batches": _summary(payload["batches"]), "outputs": payload["outputs"],
                "gradients": _summary(gradients),
                "global_gradient_l2": _finite_json(gradient_l2),
                "nonfinite_candidate_tensors": _nonfinite_names(payload["observed"]["candidate"]),
                "nonfinite_optimizer_momentum": _momentum_names(
                    candidate, optimizer, payload["observed"]["optimizer"]),
            }
            payload["diagnostics"] = diagnostics
            receipt["diagnostics"] = diagnostics
            # Summaries remain useful even if the full checkpoint cannot fit.
            write_json(snapshot / "failure.json", receipt)
            checkpoint = snapshot / "state.pt"
            save_checkpoint(checkpoint, payload)
            receipt.update(capture_status="committed", checkpoint=checkpoint.name,
                           checkpoint_sha256=fingerprint(checkpoint), losses=loss_summary)
            write_json(snapshot / "failure.json", receipt)
            return snapshot
        except Exception as diagnostic_error:
            if snapshot is not None and receipt is not None:
                try:
                    receipt.update(capture_status="capture_failed", capture_error={
                        "type": type(diagnostic_error).__name__, "message": str(diagnostic_error)})
                    write_json(snapshot / "failure.json", receipt)
                except Exception:
                    pass
            try:
                error.add_note("Repair failure snapshot could not be completed: "
                               f"{type(diagnostic_error).__name__}: {diagnostic_error}")
            except Exception:
                pass
            return None
