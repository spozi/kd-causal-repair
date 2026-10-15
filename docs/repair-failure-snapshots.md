# Repair failing-step snapshots

New repair studies enable failure recording for class-family candidates and all
shortcut repair/control candidates. This does not enable repair in a
baseline-only study, change the optimization hyperparameters, or retrofit an
already frozen experiment. The deployed Waterbirds baseline checkout is not
modified by this implementation.

## Artifacts and semantics

Each failed attempt has a unique directory, preserving earlier failures:

```text
<candidate>/failures/epoch-<epoch>-step-<step>-<unique>/
    failure.json
    pre_step.pt
    state.pt
```

`failure.json` identifies `kind=repair_failure_snapshot`, `status=failed`,
`resumable=false`, the failing phase and exception traceback. Epoch, batch step
within epoch and attempted global step are **one-based**. The final receipt
records `capture_status=committed` and the SHA256 of `state.pt`. Non-finite
diagnostic numbers are explicit JSON strings (`nan`, `+inf`, `-inf`), not invalid
JSON numeric literals.

`pre_step.pt` is an independently SHA-bound, CPU-only fallback saved before
reading any failing loss or observed device state. It is explicitly marked
`snapshot_scope=pre_step_only` and `resumable=false`. If a CUDA error or later
write prevents `state.pt`, inspect `pre_step_checkpoint` and
`pre_step_checkpoint_sha256` in the receipt to locate/verify any surviving
fallback. A receipt with `capture_status=capture_failed` is not a complete
observed-state capture. Both files are retained when full capture succeeds;
this intentionally duplicates pre-step data on disk for failed attempts only.

The tensor payload can be loaded on CPU with `weights_only=True`:

```python
import json
from pathlib import Path
import torch
from kd.checkpoints import fingerprint

attempt = Path("<candidate>/failures/<attempt>")
receipt = json.loads((attempt / "failure.json").read_text())
assert receipt["capture_status"] == "committed"
assert fingerprint(attempt / "state.pt") == receipt["checkpoint_sha256"]
state = torch.load(attempt / "state.pt", map_location="cpu", weights_only=True)
print(receipt["phase"], receipt["epoch"], receipt["step"], receipt["losses"])
```

The payload retains:

- `pre_step`: detached CPU copies of candidate, optimizer (including momentum),
  scheduler and gradient-scaler state, RNG before batch loading, RNG immediately
  before forward, and sampler state.
- `observed`: candidate and frozen anchor at failure, available gradients,
  optimizer/scheduler/scaler state, failure-time RNG and sampler state. An
  optimizer exception may leave a partially changed model; it is not overwritten
  with the pre-step copy.
- `batches`: actual augmented target/preservation image tensors and class labels,
  plus indices into the dataset supplied to the fitter. No examples are refetched
  to construct the snapshot. A data-loading failure may leave only indices or
  the successfully loaded stream.
- `losses`, `outputs` and `diagnostics`: available component losses and summaries
  of logits, features, batches and gradients, including non-finite counts/ranges
  and gradient norms. Large feature activations are summarized, not archived in
  full; they can be investigated using the saved model, anchor and batch.
- `history`, `settings`, `context`, `masks` and source SHA256 values: completed
  epoch history, numerical settings, candidate/base-checkpoint identity and the
  selected parameter coordinates.

Sampler state includes the sampler generator, complete current-epoch draw list,
consumed cursor and the subset-to-dataset index mapping. Python, NumPy, PyTorch
and available accelerator RNG state are captured without taking new random
draws. These fields support forensic reconstruction; this is **not** an automatic
resume format, and does not promise bitwise replay across devices or versions.

## Covered failures and safeguards

The fitter tracks setup, data iteration/loading/transfer, target/preservation/
anchor forward, loss computation/validation, backward, gradient validation,
optimizer/scaler updates, post-step checks, epoch validation and final mask/buffer
verification. Only values available before an exception can be recorded.
Record-enabled non-AMP fitting checks loss and gradient finiteness per step even
when the previous fast path deferred checks until epoch end. Post-step checks
cover candidate parameters/buffers and SGD momentum. Standard AMP skipped updates
remain valid scaler behavior, not spurious failure records.

No finite clipping threshold or smaller learning rate is introduced by this
feature. An infinite gradient-norm limit remains a non-finite-gradient guard,
not protection against large finite updates. Recording diagnoses instability;
it does not claim to fix its numerical cause.

Failure saving is best effort: a small receipt is written before large device
copies/checkpoint writes, then the checkpoint and final receipt are written
atomically. Disk exhaustion, a poisoned CUDA context, process kill, power loss,
or filesystem failure may prevent a complete snapshot. Any diagnostic error is
attached to the original exception, which is still raised. No `completion.json`,
inference checkpoint or success result is fabricated. A failed diagnostic write
does not make a candidate valid.

## Cost and direct API

Pre-step state is retained in RAM, not written on every successful step. This
requires CPU memory for a candidate and optimizer copy, device-to-host transfers
and finite-check synchronization. Full-network ResNet control recording is more
expensive than a masked small-model fit. A complete failure artifact also stores
the anchor and observed state; reserve disk space accordingly. Success creates
no failure files.

Production study callers supply the path and frozen candidate context. Direct
callers can opt in explicitly:

```python
history, verification = train_repair_candidate(
    # ... existing model/data/device arguments ...
    failure_dir=candidate_dir / "failures",
    failure_context={"candidate_id": "diagnostic-trial"},
)
```

Omitting `failure_dir` retains the previous direct-call behavior. The recorder's
source is included in matrix/study computational identities, so using the new
implementation against an incompatible frozen study fails closed. Use a fresh
study output when adopting new computational source; do not rewrite old hashes.

Offline tests use synthetic tensors and injected failures only:

```bash
python -m unittest discover -s tests -p 'test_repair_failure*.py' -v
```
