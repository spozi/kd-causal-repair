# Remaining three-dataset strict-versus-expanded experiment

See the [consolidated results ledger](kd-baseline-current-results.md) for the
latest saved outcomes; this document records the experiment protocol.

Authorized **27 October 2026**: create the missing CE controls for GTSRB,
Fashion-MNIST and EuroSAT, then run their 24 strict/expanded repair arms. This
authorization supersedes the earlier instruction to leave these datasets
pending because new baseline teachers were outside the checkpoint-reuse scope.

## Scope and accounting

| Dataset | Profiles, in order | Settings | Repair arms |
|---|---|---:|---:|
| GTSRB | LT-IF50, LT-IF100, LT-IF10, default (`balanced`) | 4 | 8 |
| Fashion-MNIST | LT-IF50, LT-IF100, LT-IF10, default (`balanced`) | 4 | 8 |
| EuroSAT | LT-IF50, LT-IF100, LT-IF10, default (`balanced`) | 4 | 8 |
| **Total** | | **12** | **24** |

For each dataset/profile setting, fit one CE teacher with seed 41 and three
original-teacher KD students with seeds 42, 43 and 44. Both screening policies
reuse these exact controls. This requires **12 teacher fits and 36 original KD
student fits**, each for 100 epochs. The scope is the CE controls needed for the
repair comparison; it does not add the eight-method teacher-training benchmark.

Run strict screening before `screened_then_ranked` for each setting. An accepted
repair receives three paired KD students with seeds 42, 43 and 44. A rejected
repair is recorded as a selection outcome and does not launch replacement
student fits. The maximum additional workload is **192 repair-candidate fits**
(24 arms × four caps × two learning rates) and **72 repaired-teacher KD fits**.
Infeasible or duplicate supports and repair rejection reduce these counts.

The completed 76-run checkpoint-reuse campaign and four pilot arms remain
unchanged. Completing this extension fills the remaining 24 arms of the unified
13-dataset, four-profile, two-policy ledger: **104 arms in total**. Baseline fits
are prerequisites and are not counted as additional strict/expanded arms.

## Frozen scientific protocol

Use the same **58 scientific runtime files**, byte-for-byte, as the completed
matrix/pilot snapshot. Record and verify their SHA256 hashes before execution.
New deployment/configuration files must not change the scientific runtime.

| Component | Setting |
|---|---|
| Teacher → student | Adapted ImageNet-pretrained ResNet-34 → ResNet-18 |
| Initialization | Named `IMAGENET1K_V1` weights for both architectures |
| Input | RGB 32×32, existing `imagenet-smallstem-v1` preprocessing |
| Profiles | Default, LT-IF10, LT-IF50 and LT-IF100; catalog-defined splits and imbalance construction |
| Teacher seed | 41, one fitted CE teacher per setting |
| Original and repaired KD student seeds | 42, 43, 44, paired across comparisons |
| Teacher/student training | 100 epochs; batch 128; learning rate 0.001; momentum 0.9; weight decay 0.0005 |
| Student KD | Temperature 4; KD weight 0.5; five warmup epochs |
| Runtime | BF16, existing fast-CUDA mode and channels-last settings |
| Screening policies | `strict` and `screened_then_ranked`; same original teacher, localization inputs and target plan |
| Requested repair caps | 0.4%, 0.8%, 1.6%, 3.2% of teacher parameters |
| Repair fitting | 20 epochs; 640 samples per stream per epoch; batch 64; learning rates 0.001 and 0.005 |
| Repair objective | Existing target CE, anchor KL weight 1 at temperature 4, feature preservation weight 0.25 |
| Teacher acceptance | Held-out target-recall improvement; overall accuracy loss ≤0.5 percentage points; held-out forward KL at temperature 1 ≤0.01 |
| Final evaluation | Existing canonical CPU FP32 standalone pipeline; selection frozen before test evaluation |

Each candidate starts from the original CE teacher. Record the actual unique
editable-coordinate union, achieved fraction, channel membership and realized
candidate count. Equal requested caps do not imply equal realized support or
candidate counts. Expanded screening is allowed to add localization-ranked
channels after the strict prefix, including when strict screening admits none;
the final teacher-preservation constraints remain unchanged.

## Data and admission checks

Fetch only these three public datasets through the frozen runtime's catalog
definitions. Verify the catalog identity, artifact hashes and split/profile
metadata before fitting controls. Record preprocessing, class counts, split
indices, targets and initialization provenance. Both policies must bind to the
same fitted original teacher and the same three original KD checkpoints.

Preflight must validate configurations, source hashes, input dimensions,
class counts, profile construction, device/runtime settings, and available
storage. After baseline completion, verify each selected control checkpoint and
its identity before admitting either repair arm. Missing or changed inputs stop
the queue rather than trigger an unrecorded refit. Existing identity-bound
resume/completion safeguards apply to interrupted work.

## Execution, retention and archives

Use a serial queue on the existing local RTX 4000 SFF Ada server. The deployment
receipt will record the remote root, Python environment, service name, manifest
hashes and launch verification. Keep this extension's outputs separate from
the completed checkpoint-reuse campaign.

Launched **19:59 MYT, 27 October 2026**, after all 12 data/configuration checks
passed. First teacher training epochs and GPU activity were verified.

- Remote root: `/home/ubuntu/kd-flexible-screening-pending3-20261027`.
- Service: `kd-flexible-screening-pending3-20261027.service`.
- Python: `/home/ubuntu/kd-cinic10-balanced/venv/bin/python`.
- Order: GTSRB → Fashion-MNIST → EuroSAT; IF50 → IF100 → IF10 → default.
- Per setting: CE controls and four-panel test archive → strict → expanded.
- Controller survives SSH disconnection; `Restart=no`, no reboot autostart.
- Local preparation, full preflight and launch receipts:
  `runs/server-deployments/flexible-screening-pending3-20261027/`.

Actual catalog-derived train/validation counts verified before launch:

| Dataset | LT-IF50 | LT-IF100 | LT-IF10 | Default |
|---|---:|---:|---:|---:|
| GTSRB | 8,721 / 972 | 7,770 / 869 | 12,070 / 1,341 | 23,976 / 2,664 |
| Fashion-MNIST | 15,120 / 1,680 | 13,401 / 1,489 | 22,069 / 2,452 | 54,000 / 6,000 |
| EuroSAT | 6,408 / 801 | 5,732 / 716 | 9,118 / 1,140 | 21,600 / 2,700 |

GTSRB uses the configured `GTSRB-Training_fixed.zip` artifact. Its image archives
were obtained from a public mirror with the original catalog MD5 matches; the
small ground-truth ZIP was restored from the local project cache with its exact
original checksum. All three datasets passed the frozen fetch/verify pipeline;
catalog version 1.1.0 and its scientific source remain unchanged. Catalog split
and artifact identities are pinned before fitting controls.

Retain all selected models, repair candidates and optimizer/resume files.
Require at least **12 GiB free** before admitting a new run, and retain the
engine's checkpoint-storage guards. Stop safely when space is low; do not
delete models or completed experiments automatically.

After each completed repair arm, create and verify a results/predictions-only
archive and receipt. Validate report manifests, archive-member hashes, NPZ
integrity and the expected prediction panels for accepted comparisons. Retain
baseline results, predictions and provenance for the shared controls as well.
Checkpoint files remain outside the results/predictions archives. There is no
automatic transfer to the Mac and no automatic cleanup. An SSH disconnection
must not terminate the persistent queue; any reboot recovery should inspect
and resume the same frozen protocol.

## Analysis and time estimate

Report paired student target macro recall as the primary downstream measure,
alongside accuracy, balanced accuracy, macro precision/recall/F1, per-image
corrections/regressions, preservation drift, actual editable support and cost.
Practical success requires at least 1 percentage point mean target-recall gain,
positive target-recall gain in at least two student seeds, and no more than
0.5 percentage points mean accuracy loss. This is a practical criterion, not a
statistical-significance test. Results remain conditional on one fitted teacher
per dataset/profile; overlapping profiles do not provide independent teacher
replications.

Planning estimate on the current GPU: **12–20 hours**, plus dataset download and
preparation. Allow approximately **18–26 hours** if most repairs are accepted.
These estimates use measured prior training timings and the observed repair
acceptance frequency; record actual times and revise the ETA from early runs.
