# Missing teacher-training baselines, ascending dataset size

Authorized 27 October 2026: queue the missing baseline experiments on the local
GPU server in ascending dataset size. This extends the previous CE-control and
repair-only scope for the six datasets with no historical full baseline suite.

## Queue and scope

“Dataset size” means actual default-profile training examples after the
configured validation split. All inputs use the same RGB 32×32 protocol, making
this a useful ordering for training workload. The order is frozen before fitting:

| Order | Dataset | Default training examples | Profiles | New fits |
|---|---|---:|---|---:|
| 1 | EuroSAT | 21,600 | IF50, IF100, IF10, default | 112 |
| 2 | GTSRB | 23,976 | IF50, IF100, IF10, default | 112 |
| 3 | OrganAMNIST | 34,561 | IF50, IF100, IF10, default | 112 |
| 4 | Fashion-MNIST | 54,000 | IF50, IF100, IF10, default | 112 |
| 5 | SVHN | 65,931 | IF50, IF100, IF10, default | 112 |
| 6 | PathMNIST | 89,996 | IF50, IF100, IF10, default | 112 |
| **Total** | | | **24 settings** | **672** |

Each setting reuses its exact CE teacher and three original KD students. Fit
the remaining seven teacher recipes, followed by three paired KD students each:

1. RandAugment.
2. Label smoothing.
3. Balanced Softmax.
4. MSE logits.
5. MSE probabilities.
6. SAM.
7. EMA.

Total new work is **168 teacher fits and 504 KD student fits**. There are no new
CE fits, supervised-student reference fits, teacher repairs, Mixup, CutMix, or
combined RandAugment/CutMix/EMA conditions. Balanced Softmax remains an explicitly
retained condition even where equal training counts make its correction redundant;
those rows must not be interpreted as independent evidence for a distinct method.

The twelve refitted-teacher historical contexts identified in the coverage audit
are outside this deployment. They require their separate compatibility review.

## Scheduling and isolation

The new serial service waits for successful completion of
`kd-flexible-screening-pending3-20261027.service` and its complete 12-control-family,
24-repair-arm ledger. It starts no GPU work while that queue is active. It stops
if the dependency fails or is incomplete and inactive, rather than bypassing it.

Remote root: `/home/ubuntu/kd-ascending-baselines-20261027`.
Service: `kd-ascending-baselines-20261027.service`.
Python: `/home/ubuntu/kd-cinic10-balanced/venv/bin/python`.

The service survives SSH disconnection, uses `Restart=no`, and does not install
a reboot autostart. Resumption must inspect and reuse the same frozen package.
Status records completed dataset/profile settings out of 24. Per-setting
`progress.json` identifies the active teacher/KD run and recipe. Fitting and
canonical evaluation occur within a separate child process per setting.

## Frozen protocol

The 58 scientific runtime files are copied byte-for-byte from the completed and
active screening campaigns. New code only prepares, validates, schedules and
reports existing training/evaluation functions. Original controls and earlier
results remain external read-only inputs.

| Component | Setting |
|---|---|
| Teacher → student | ImageNet-initialized adapted ResNet-34 → ResNet-18 |
| Input | RGB 32×32; `imagenet-smallstem-v1`; original split/augmentation identity |
| Seeds | Teacher 41; KD students 42, 43, 44 |
| Training | 100 epochs, batch 128, LR 0.001, momentum 0.9, decay 0.0005 |
| Teacher conditions | Existing frozen recipe implementations and default recipe parameters |
| Teacher checkpoint | Best validation accuracy, independently for each method |
| KD | Classical KD, temperature 4, weight 0.5, warmup 5; original student initialization/data |
| Student checkpoint | Existing engine's best validation accuracy rule |
| Runtime | BF16, fast CUDA, channels-last, four workers/threads; persistent workers disabled |
| Test inference | Canonical CPU FP32, four CPU threads, fixed diagnostic transforms/order |

All methods are fitted and their selected model hashes frozen for a setting
before its test loader is constructed. Every prespecified method is reported;
teacher quality or inspected test results do not filter methods. Teacher and
student initial states, data/split identities and KD settings must match the
paired CE controls. Inference is harmonized for CE and all seven methods.

This is a fixed-hyperparameter project comparison, not an optimized reproduction
of published method numbers. Teacher compute is not equalized: SAM uses its
existing two-pass optimization, EMA retains its extra state, and RandAugment
adds its configured transformation cost. No new tuning sweeps are introduced.

## Retention, space and archives

Keep all selected, best and last checkpoints, optimizer state and resumable
histories. Require **at least 12 GiB free** before each setting and each model
fit. Stop safely on low space; do not delete completed studies automatically.

Observed current checkpoint sizes predict **158.8 GiB additional checkpoint
storage** for 672 fits. This excludes extra EMA state, predictions, archives and
growth of the active dependency. About 138.4 GiB was free at inventory time;
therefore the full campaign is expected to need more storage. Allow roughly
200 GiB free at baseline start for a practical reserve. Existing retention
authorization permits the queue to stop for expansion rather than delete models.

After a setting finishes, each of its eight conditions receives a verified
results/predictions-only archive with one teacher and three KD prediction panels.
All archives include the common case summary, source/checkpoint identities and
training metadata. Thus a complete setting has eight archives and 32 panels,
including reused CE references; these are not 32 new model fits. Model files stay
outside result archives. No automatic Mac transfer is enabled.

## Verification and completion

Before queueing, validate all 24 configurations and dataset/profile identities
without training or claiming an idle GPU. On admission after the dependency,
verify cached named initialization weights, GPU availability and all four CE
checkpoint/configuration identities. Freeze those CE pins before new fitting.

A setting is complete only after 28 new fits, common inference of all eight
conditions, and verified archive receipts. Common-pipeline test accuracy,
precision/recall/F1 and target recall are reported with paired student deltas
against CE. Practical improvements and significance remain different concepts;
one teacher seed provides conditional-on-teacher comparisons.

Deployment artifacts and receipts are under
`runs/server-deployments/ascending-baselines-20261027/`.

## Deployment verification

Verified on **28 October 2026 at 00:37 MYT**. The service is active with phase
`waiting_for_dependency`; no new baseline study has started. The original
CE/repair queue remains active on Fashion-MNIST LT-IF10, with 6/12 control
families and 12/24 repair arms complete. Its GPU process remains the sole
training process.

All 24 settings passed non-training preflight, and all 11 queue tests passed.
CE checkpoint readiness is intentionally checked again after the dependency
finishes. Free disk at deployment verification is **131.0 GiB**, lower than the
earlier inventory as the active campaign continues. More storage is expected to
be required under the retain-all policy.

The initial service start stopped before training because its deployment
identity receipt was missing. The receipt was written, the controller hash was
verified, and the service was restarted successfully. Scientific code and the
frozen package were unchanged. The local `launch-verification.json` records the
healthy service state and invocation identity.

Storage expansion verified on 28 October 2026: the larger virtual disk was
visible but the root partition/filesystem had retained their old capacity.
After a successful dry run and saving the original partition table, the root
partition and ext4 filesystem were expanded online. The filesystem now reports
approximately **552 GiB total and 277 GiB available**, meeting the planned
roughly 200 GiB reserve. Both experiment services remained active; the baseline
queue continues waiting for the current campaign. The 12 GiB guard and retain-all
policy remain in effect. Receipt: `storage-expansion-verification.json`.

## Historical control provenance recovery

On 29 October 2026 the queue stopped before OrganAMNIST LT-IF50 because the
legacy OrganAMNIST control directory predates baseline-level `protocol.json`
files. The same compatibility gap affected all four OrganAMNIST, SVHN and
PathMNIST profiles. The four CE/KD checkpoints, configurations, summaries and
completion records were present for every setting.

The missing provenance was reconstructed for these 12 control families from
the frozen historical campaign manifest, each study protocol and case receipt,
and every run's configuration, summary, completion record and checkpoint. The
records explicitly identify themselves as reconstructed and set
`training_time_file_claim` to false. They do not claim that these files existed
during training.

Validation covered all frozen campaign source hashes, all completion-listed
artifact hashes, shared data/split identities within each four-model family,
the teacher hash in every KD summary, and the teacher hash in the original case
receipt. The ascending queue's own ready-control admission check then passed
all 24 settings. A provenance-only backup containing 24 JSON files is saved as
`runs/server-deployments/ascending-baselines-20261027/historical-control-provenance.tar.gz`
with SHA-256
`0a6cfb9c802602d876876f7489b5408b69d899311fb05ebc6c706eea3cf4e11a`.

The queue resumed successfully, skipped the eight completed EuroSAT/GTSRB
settings through their completion records, and began OrganAMNIST LT-IF50 with
the RandAugment teacher. No CE control was retrained or changed.
