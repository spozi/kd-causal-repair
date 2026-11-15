# Flexible-screening pilot — 26 October 2026

Status: prepared locally; not uploaded, queued or running. Remote hardware/input
readiness must pass the packaged preflight before launch.

## Research question

For the same available teacher, does admitting additional localized channels
under unchanged parameter caps and preservation constraints improve downstream
KD relative to strict screening? This is an exploratory policy/capacity pilot,
not proof that zero-ablation failure implies repairability or that more capacity
is always useful. The two datasets were selected after inspecting prior results.

## Frozen queue

| Order | Dataset/profile | Policy | Existing original teacher | New paired KD students |
|---|---|---|---|---|
| 1 | STL-10 default | Strict | Reuse seed 41 | Seeds 42/43/44, only if accepted |
| 2 | STL-10 default | Screened-first expansion | Same checkpoint as run 1 | Same paired seeds, only if accepted |
| 3 | CIFAR-100 LT-IF50 | Strict | Reuse seed 41 | Seeds 42/43/44, only if accepted |
| 4 | CIFAR-100 LT-IF50 | Screened-first expansion | Same checkpoint as run 3 | Same paired seeds, only if accepted |

The CE-teacher KD students are reused and re-evaluated, not retrained. No new CE
teacher, ordinary supervised student, Balanced Softmax or other teacher-training
baseline is fitted. Earlier strict results remain historical context: both
policies are rerun with one frozen source snapshot and evaluation pipeline.

Checkpoint SHA256s, train/validation index hashes, source-artifact hashes and
data-recipe hashes are pinned in
[the experiment definition](../../configs/flexible_screening_pilot.json), taken
from the previously verified STL-10 and retained-model deployment receipts.
Missing or different inputs must stop preparation/launch, not trigger refitting.

## Settings shared by both policies

| Component | Frozen setting |
|---|---|
| Teacher / student | Existing ImageNet-initialized adapted ResNet-34 / ResNet-18, RGB 32×32 |
| Editable caps | 0.4%, 0.8%, 1.6%, 3.2% of all teacher parameter coordinates; count mask unions |
| Localization | Existing differential scores, bootstrap and causal-ablation diagnostics; stages 3/4 |
| Repair grid | LR 0.001/0.005; 20 epochs; 640 draws per stream per epoch; batch 64; at most eight unique support/LR fits per run |
| Repair loss | Existing target CE + anchor KL (T=4, weight 1) + feature preservation (weight 0.25) |
| Final teacher gates | Strict target-recall improvement; accuracy loss ≤0.5 pp; T=1 forward KL ≤0.01 on selection validation |
| KD student | 100 epochs; batch 128; SGD LR 0.001, momentum 0.9, decay 0.0005; cosine schedule |
| KD objective | Temperature 4, weight 0.5, warmup 5 epochs; no student-side changes |
| Training/selection runtime | Inherit BF16, channels-last and fast CUDA settings from the pinned baselines; record actual environment |
| Final test evaluation | Existing standalone canonical CPU FP32 path, identical for original/repaired teachers and both student arms |
| Selection / test separation | Select and freeze teacher before loading test; select students on validation, never test |
| Storage | Keep selected models, candidates and resume files; require ≥12 GiB free before admission; no automatic deletion |

Upper bounds across all four runs: **32 repair candidates and 12 new KD students**.
Counts decrease for infeasible/duplicate supports or teacher rejection. Equal
requested grids do not imply equal realized search counts; report both.
Screened support can still saturate. Record actual support and shortfall reasons;
do not call a 3.2% cap a 3.2% intervention unless the achieved mask supports it.

## Outcomes and reporting

1. Capacity: pool counts, strict/fallback composition, achieved parameter
   fractions, unique supports, deduplication and saturation.
2. Teacher: selection eligibility and reasons, validation target recall,
   accuracy, target NLL and preservation KL; frozen-model test changes.
3. Student: paired changes against original-teacher KD in target macro recall
   (primary), accuracy, balanced accuracy and macro precision/recall/F1; report
   all seeds and paired-image gains/losses. Compare expanded versus strict only
   when both produced actual selected teachers/students.
4. Practical transfer success: mean target recall gain ≥1 pp, positive target
   recall gain in at least two seeds, and mean accuracy loss ≤0.5 pp. This is not
   statistical significance. A rejected repair is not an independent zero-effect
   student experiment; any fallback deployment uses the original teacher/KD.
5. Cost: localization/screening, actual candidate fits/updates, student fits,
   runtime and storage separately. Sparse support does not imply proportional
   FLOP savings.

Stop after the two preselected settings. Do not change caps, ranking or gates in
response to test performance. Existing test sets have been inspected, and the
three student seeds condition on a single available teacher per setting. Do not
pool overlapping profiles or test images as independent teacher replications.

Localization-only and capacity-/stage-matched random controls are **not included
in this first pilot**. They require additional implementation/prespecification
before making claims about localization quality or strict-first optimality.
Held-out repair retention is also not automatically produced here: it needs a
separate assessment partition; selection validation cannot substitute for one.

## Preparation and launch boundary

From the project root and its Python environment:

```sh
python -m scripts.prepare_flexible_screening_pilot
```

This writes an immutable local bundle at
`runs/server-deployments/flexible-screening-pilot-20261026/`, including frozen
runtime source (also uncommitted relevant changes), four configs, experiment and
file manifests, preflight, launch commands and this protocol. It excludes
credentials, datasets, models, unrelated research/proposal files and results.
An existing preparation directory is never overwritten; use a new directory
and experiment identity for a revised protocol.

Target: the previously used local GPU reached through
`ssh -J root@192.168.68.103 ubuntu@10.10.10.2`.
Frozen target root: `/home/ubuntu/kd-flexible-screening-pilot-20261026`.
Preparation does **not** upload or contact it. Existing checkpoint and dataset
paths in the definition must still be available there.

After explicit deployment approval, transfer/extract the verified bundle to the
frozen target root, select the existing compatible CUDA Python environment, then
run the **non-training** preflight from that directory:

```sh
PYTHONPATH="$PWD/code" "$PYTHON" preflight.py --device cuda:0
```

It checks frozen files, exact model hashes, standalone adoption, train/validation
data provenance without test loading, BF16/GPU readiness, cached student weights,
disk headroom and existing output identities. It neither trains nor launches a
queue. Do not mistake locally validated configs for a passed remote preflight.

Only an explicitly authorized launch runs:

```sh
bash run-serial.sh
```

Run under an approved persistent job/session manager, not a fragile interactive
SSH connection. Commands are serial and recheck admission before every run;
existing same-protocol outputs use the study's normal cache/resume safeguards.
No external queue or autostart service is installed by preparation.

At completion, create and verify a results/predictions archive for each run and
back it up to the Mac, retaining configs, protocols, reports and hashes. Raw
models remain on the server under the keep-all policy. Archive verification and
Mac transfer are separate completion steps, not automatic actions of the
generated shell commands. Do not delete server files without further approval.

The final comparison table should retain the following columns even for rejected
runs (student-effect cells remain not evaluated):

| Dataset/profile | Policy | Actual support % | Teacher accepted | Original KD score | Repaired KD score | Target recall Δ | Accuracy Δ | Macro-F1 Δ | Practical success | Search/student cost |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| STL-10 default | Strict / expanded | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| CIFAR-100 LT-IF50 | Strict / expanded | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
