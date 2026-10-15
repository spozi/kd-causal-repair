# Knowledge distillation for computer vision

A classification implementation of [the project guidelines](knowledge-distillation-guidelines.md), built with PyTorch and torchvision. It includes supervised training, classical logits KD, Decoupled Knowledge Distillation (DKD), optional semantic-stage feature transfer, controlled ablations, and inference-cost reporting.

[Experiment settings](docs/experiment-settings.md) defines the implemented **full
16-dataset experiment**: the original 13 class/long-tail datasets plus
**Colored MNIST, Waterbirds, and UrbanCars**. The
[shortcut companion](docs/causal-benchmark-phase1.md) defines subgroup outcomes,
teacher-to-student transfer, and research/adaptation boundaries. The planner now
includes all 16, with native group-aware repair/selection and mechanism controls
on the three additions. Waterbirds/UrbanCars use ImageNet-pretrained ResNet-34
teachers and ResNet-18 students (`IMAGENET1K_V1`); provision their local releases
before training. Full-budget results are not measured. Historical settings and
saved results remain unchanged.

[Teacher-training baselines and distillability](docs/teacher-training-baselines.md)
adds an isolated, Consensus-researched benchmark for RandAugment, Mixup, CutMix,
label smoothing, Balanced Softmax, raw/probability MSE, SAM, EMA, and a combined
augmentation recipe. It compares teacher quality with matched downstream KD,
includes the original channel-repair method using the common CE teacher,
freezes validation selection before test access, and documents recent 2025–2026
follow-up methods. Inspect all 16 datasets (four profiles on the original 13,
one native setting on each addition) without writes or
training using `python -m kd.teacher_baseline_matrix plan --output
runs/teacher-baselines-v5/matrix-plan --epochs 100 --dry-run`. Add
`--pairs convnet6 resnet34_resnet18` to include both supported pairings.
New matrix runs use 100 teacher/student epochs uniformly and 20 repair epochs;
historical budgets stay frozen. Generated configs bind their parent plan and
computational sources before execution. Interrupted training requires new
checkpoint-bound data provenance and a committed `last.pt` digest of the selected
`best.pt`; unsealed legacy training checkpoints without those bindings cannot
resume, although completed hash-verified inference artifacts remain usable.
An interrupted best/last pair commit fails closed: restore a known matching pair
or use a fresh run. Repair adoption always checks the student's recorded
training-teacher digest, even when the original checkpoint path still exists.

The default full plan has 55 settings and 56 test endpoint panels; both
class-family pairs give 107/108. Create a fresh plan after code changes, then use
`python -m kd.full_benchmark preflight --plan PLAN`; the subsequent phases are
`train`, `evaluate` and `report` (`report` also needs `--output`). See
[data layouts and runnable commands](docs/experiment-settings.md#9-execution-readiness-and-pre-run-checks).
Preflight never downloads weights; explicit training does. Offline tests cover
native-resolution training/KD with local fixture weights, not GPU capacity or
actual pretrained-weight performance.

**CIFAR-10 is supported**, with a full-data, repeated-seed study and a held-out test phase. See [the CIFAR-10 protocol](docs/cifar10.md). Generated results live in `runs/cifar10/report.json`; render `report.md` and figures with `python -m kd.reporting --study runs/cifar10` after the study completes.

The [calibration-loss intervention](docs/calibration-intervention.md) adds an optional validation-driven MMCE penalty, with a separate matched-control study and resumable controller state. It is disabled by default.

[Post-hoc calibration](docs/posthoc-calibration.md) fits temperature scaling and Focal Temperature Scaling to validation logits from an existing frozen student, then evaluates the fixed calibrators on test data. It exports reusable calibration parameters without changing model weights.

[CPC during student KD](docs/cpc-kd.md) adds fixed pairwise calibration constraints, with a [completed matched three-seed study](runs/cifar10-cpc/report.md) comparing raw and temperature-scaled students. Extended evaluation includes class-wise calibration, confidence ranking, selective accuracy and conditional calibration-null references. The first 0.1/0.1 setting did not demonstrate added benefit on the primary AURC endpoint beyond KD+TS.

[AdaFocal and AdaDualFocal during student KD](docs/adaptive-focal-kd.md) are optional supervised-loss strategies with per-bin validation feedback and checkpointed gamma schedules. Ready-to-run CIFAR-10 configurations reuse the existing frozen teacher; these combinations have implementation tests, not a completed CIFAR-10 comparison.

[Sample/slice surgery before KD](docs/sample-surgery-kd.md) diagnoses a frozen checkpoint on canonical, non-augmented training images, reports weak true-class and confusion slices, and writes a bounded list of dataset-local samples that a subsequent KD run can drop. It is opt-in and records the complete intervention in run provenance.

[Channel-level teacher repair before KD](docs/neuron-surgery-kd.md) adapts AI-Lancet to the factor-100 CIFAR-10-LT resources in this workspace. It uses same-class feature companions and bootstrap consensus to localize `stage2`/`stage3` channels, fine-tunes only their connected weights under original-teacher preservation KD, and distills a selected repaired teacher into three matched students. No training samples are removed.

The initial workspace contained no application, dataset, or teacher checkpoint. The project now supports **image classification** using CIFAR-10, ImageFolder, and an offline synthetic smoke dataset. It does not yet implement autonomous driving, object detection, semantic segmentation, or transformer distillation tokens. See [design and guideline coverage](docs/design.md) for extension boundaries.

## Versioned datasets and private cache

The built-in catalog covers 13 compact vision benchmarks, from CIFAR and SVHN through MedMNIST,
Caltech-101, EuroSAT, and STL-10, with balanced and deterministic IF10/50/100 profiles. Raw archives
can come from their canonical upstream locations, an optional private Git LFS cache, or extra
mirrors; each archive is downloaded from whichever source is fastest, switches source mid-download
if one slows, and is accepted only if its checksum matches. The public source repository never
requires private credentials.

```bash
python -m kd dataset list
python -m kd dataset fetch cifar100 --version 1.0 --profile lt-if100 --root data
python -m kd dataset verify cifar100 --version 1.0 --profile lt-if100 --root data
```

Run `python -m kd dataset registry-config --url SSH_URL --ref catalog-v1.2.0` once on a machine to
use a private mirror automatically. Environment variables remain available as temporary overrides.
See [the dataset registry protocol](docs/dataset-registry.md) for setup, license review, selective
LFS fetching, and security boundaries.

## Run in the existing Conda environment

From this directory in Bash:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kd
python -m unittest discover -s tests -v
python -m kd smoke --output runs/my-smoke
```

No downloads or additional test packages are needed in the existing `kd` environment. The smoke command trains a small teacher, runs all **21 student variants**, and writes `runs/my-smoke/ablations/comparison.json` and `comparison.csv`. Use a new output directory on subsequent runs. Synthetic accuracies are execution checks, not evidence that one distillation method is better.

The implementation was verified on Python 3.12.12, PyTorch `2.14.0.dev20260618`, and torchvision `0.29.0.dev20260618`. Unit tests and synthetic smoke runs use CPU; the CIFAR-10 study uses the Mac's MPS GPU. CUDA execution paths are covered by configuration and CPU-side integration tests but cannot be physically exercised on this Mac. Run the CUDA preflight below on every GPU server before starting a study. Dependencies are declared in `pyproject.toml`; an optional `python -m pip install -e . --no-deps` enables the `kd` console command. Module commands work directly without installing the project.

## NVIDIA CUDA servers

Create the pinned environment (Python 3.12, PyTorch 2.14.0 with CUDA 13.2, the build Experiment 1
ran on) from [`environment.yml`](environment.yml), or install
[`requirements-cuda.txt`](requirements-cuda.txt) into any Python 3.12 virtualenv. Then verify the
complete optimized path rather than relying only on `torch.cuda.is_available()`:

```bash
conda env create -f environment.yml
conda activate kd
nvidia-smi
python -m kd cuda-check --device cuda:0 --precision auto
```

CUDA training defaults to BF16 on supported GPUs and scaled FP16 otherwise. It also enables TF32,
cuDNN autotuning, fused SGD, channels-last convolution tensors, pinned host batches, and asynchronous host-to-GPU
copies. These choices and the resolved GPU properties are stored in each run's `summary.json`.

For ordinary configuration files, use:

```toml
[train]
device = "cuda:0"
precision = "auto"
cuda_mode = "fast"
channels_last = true
workers = 4
persistent_workers = false
prefetch_factor = 2
```

`cuda_mode = "fast"` prioritizes throughput and permits nondeterministic cuDNN kernel selection. For a
strict reproducibility audit, use `cuda_mode = "deterministic"` and `precision = "float32"`; this disables
TF32 and cuDNN autotuning. Keeping `persistent_workers = false` preserves epoch-boundary data-loader RNG
restoration after a checkpoint resume. See [CUDA execution](docs/cuda.md) for server setup, multi-GPU job
placement, memory tuning, and experiment-specific behavior.

## Train on images

Arrange labeled images in ImageFolder format with identical class names in each split:

```text
data/images/
  train/
    left_turn/...
    right_turn/...
    stop/...
    yield/...
  val/
    left_turn/...
    right_turn/...
    stop/...
    yield/...
  test/                      # Optional for training; required for test evaluation
    left_turn/...
    right_turn/...
    stop/...
    yield/...
```

Split real driving data by recording/session before training to keep correlated frames out of different splits. Configure the dataset path, class count, image size, epochs, and hardware in [teacher.toml](configs/teacher.toml) and [student.toml](configs/student.toml). The four classes and training schedule are placeholders; class names come from the dataset. All configured paths resolve relative to the working directory.

```bash
python -m kd train --config configs/teacher.toml
python -m kd train --config configs/student.toml
```

The example trains ResNet-34 as a teacher and ResNet-18 as a student. Model factories use `weights=None` and make no network requests. Assess the trained teacher's validation accuracy, confidence, NLL, entropy, and calibration before relying on it. A completed training epoch verifies checkpoint provenance, not teacher quality. KD fails if no teacher checkpoint is supplied.

Dataset-neutral model IDs include `convnet6_w32`/`convnet6_w16` and
`resnet18_smallstem`, `resnet34_smallstem`, `resnet50_smallstem`. Standard-stem
`resnet18`/`resnet34`/`resnet50`, compact `resnet8`/`resnet20`, and the three
`tiny_*` smoke models remain available. All legacy `cifar_*` IDs are retained;
their prefix describes an architecture variant, not a dataset restriction. Set
`student.checkpoint` to initialize from a compatible checkpoint. Checkpoints still
require their declared model ID, class ordering, and preprocessing; aliases do
not silently rename old artifacts. External raw state dictionaries must first be
converted to this contract.

For a supervised student, set `distillation.method = "supervised"` and `feature_weight = 0.0`. For classical KD, set `method = "kd"`. For DKD, use `method = "dkd"`. Choose a separate run `name` for each experiment.

## Run sample/slice surgery with KD

Generate a plan from a compatible frozen checkpoint. A training-split plan is required to modify training; validation and test plans are diagnostic only.

```bash
python -m kd surgery-plan \
  --config configs/teacher.toml \
  --checkpoint runs/teacher/student.pt \
  --output runs/surgery/teacher-plan.json \
  --split train \
  --score high_confidence_error \
  --strategy hybrid \
  --fraction 0.05
```

Then add the intervention to the student configuration and run KD normally:

```toml
[surgery]
action = "drop"
plan = "runs/surgery/teacher-plan.json"
max_drop_fraction = 0.10
```

```bash
python -m kd train --config configs/student.toml
```

The default score only considers confident mistakes. `high_loss` also considers correctly predicted hard examples and should be treated as a separate ablation. See [the surgery protocol](docs/sample-surgery-kd.md) for the plan fields, slice strategy, guardrails, and matched controls.

For the completed CIFAR-10 teacher and split recipe in this workspace, [cifar10_kd_surgery.toml](configs/cifar10_kd_surgery.toml) defines a seed-42 KD arm using `runs/cifar10-surgery/teacher_high_confidence_hybrid_f05.json`.

For the factor-100 long-tail recipe, [cifar10_lt_kd_surgery.toml](configs/cifar10_lt_kd_surgery.toml) records the matched seed-42 arm. Both deletion pilots reduced balanced CIFAR-10 test performance; results and class-level analysis are in [the surgery protocol](docs/sample-surgery-kd.md).

## Run channel-level teacher repair with KD

Validate the fixed factor-100 protocol and existing teacher/student baseline hashes without writing output:

```bash
python -m kd neuron-surgery-study \
  --baseline runs/cifar10-lt-multiseed \
  --output runs/cifar10-lt-neuron-surgery \
  --device auto \
  --seeds 42 43 44 \
  --dry-run
```

Remove `--dry-run` to localize channels, train the six bounded repair candidates, apply the validation gate, and, only when a repaired teacher qualifies, run the three matched KD students. See [the neuron-surgery protocol](docs/neuron-surgery-kd.md) for the fixed losses, parameter masks, artifact identities, stopping rule, and success criterion.

The completed factor-100 run selected eight `stage3` channels at learning rate 0.001. Mean student tail recall improved by 1.76 percentage points and mean overall accuracy improved by 0.87 points, with tail gains in two of three seeds, so the predeclared transfer rule passed. See the [generated report](runs/cifar10-lt-neuron-surgery/report.md) and [paired comparison](runs/cifar10-lt-neuron-surgery/comparison.json).

The [follow-up campaign](docs/neuron-surgery-campaign.md) adds a fresh-output reproduction, a sealed balanced holdout drawn from the CIFAR-10 training archive, validation-only component ablations, factors 10 and 50, and late-stage ResNet channel maps. Validate the complete sequential campaign without training:

```bash
python -m kd neuron-surgery-campaign \
  --output runs/cifar10-lt-neuron-campaign \
  --root data \
  --device auto \
  --dry-run
```

Individual generalized studies use `--study-config configs/neuron_surgery/NAME.json`. Missing matched controls can be generated with `python -m kd neuron-surgery-baselines --study-config ...`.

To run Experiment 1 across all 13 catalog datasets at balanced, IF10, IF50,
and IF100 profiles across every detected GPU:

```bash
scripts/run_gpu_experiment1_4gpu.sh --gpu-ids 0,1,2,3
```

The default catalog includes CIFAR-10, CIFAR-100, SVHN, CINIC-10, GTSRB,
Fashion-MNIST, PathMNIST, BloodMNIST, DermaMNIST, OrganAMNIST, Caltech-101,
EuroSAT, and STL-10. The launcher provisions its own environment, freezes a
52-job plan, prepares each
matrix dataset once, dynamically schedules isolated jobs, resumes validated
artifacts, and writes a matrix-level summary without pooling samples across
datasets. It uses an activated virtualenv when one is present, otherwise creates
the Conda environment (or a local `.venv`) and installs dependencies itself.
Dataset downloads, Git LFS transfers, and package installs stream live byte
progress, and the queue prints a per-GPU status table every 30 seconds
(`--status-interval`, or `--stream-logs` to mirror worker output inline).

For the matching 52-setting ResNet teacher studies and the remaining
teacher--student and distillation-loss comparisons, see
[`paper/full-matrix-run-plan.md`](paper/full-matrix-run-plan.md). The teacher
architecture runner supports `--all-jobs --dry-run` to list missing settings
before starting GPU training.

The [supervised-only imbalance repair protocol](docs/vanilla-imbalance-repair.md)
reuses the original classifiers at IF10/50/100 and repairs them with labeled
cross-entropy alone. It trains no KD student and evaluates each selected repaired
classifier directly. Results will be reported after the 39 new repair settings run.

To compare direct student surgery without distillation against both supervised and KD controls, run:

```bash
python -m kd student-surgery-study \
  --source runs/cifar10-lt-neuron-campaign \
  --output runs/cifar10-lt-student-surgery \
  --device auto \
  --seeds 142 143 144
```

This reuses the completed sealed factor-100 campaign, trains seed-matched supervised students,
and repairs each one with `CE(target) + CE(preservation)`. It evaluates four arms: supervised,
direct student surgery, original-teacher KD, and repaired-teacher KD. No teacher logits or feature
distillation enter the direct student-surgery arm.

## Run the recommended ablations

```bash
python -m kd sweep --config configs/student.toml --dry-run
python -m kd sweep --config configs/student.toml
```

Run either the individual student command or the sweep with that name, or change `name` between them: nonempty run directories are never silently overwritten.

The sweep starts every student from the same seed/initial checkpoint and uses matching data-order and augmentation seeds. It runs:

1. A supervised student with basic augmentation.
2. Classical KD at all nine combinations of `T ∈ {2, 4, 8}` and `weight ∈ {0.25, 0.5, 0.75}`.
3. DKD at the same nine combinations.
4. The best validation DKD setting with feature transfer, holding its other settings fixed.
5. The best preceding setting with stronger augmentation, holding its other settings fixed.

The feature ablation uses configured `feature_weight`, or `1.0` if it was zero. You can reduce the grid using `--temperatures 2 4 --weights 0.25 0.5`. Ties preserve the earlier, simpler variant. Selection uses validation accuracy and configured deployment limits; a failed budget produces `no_model_within_budget`. The test set is never used to choose a variant. Repeat the study under multiple `train.seed` values and separate names before making statistical claims. The adaptive stages can overfit validation, so evaluate the final choice once on held-out test data.

Each sweep writes a `selected_config.json` and points to the selected `student.pt` in `comparison.json`. Evaluate it with:

```bash
python -m kd evaluate \
  --config runs/student_dkd/selected_config.json \
  --checkpoint runs/student_dkd/VARIANT_NAME/student.pt \
  --split test
```

Replace `VARIANT_NAME` with `best.name` from the comparison. JSON configs exported by runs and TOML configs are both accepted.

## Loss settings and augmentation

Let `r = min((epoch + 1) / warmup_epochs, 1)`, or `1` when warmup is disabled. The zero-based epoch convention means the first epoch already receives some teacher signal.

| Method | Objective |
|---|---|
| Supervised | `CE(student, labels)` |
| Classical KD | `(1 - weight) * CE + r * weight * T² * KL(teacher/T || student/T)` |
| DKD | `CE + r * weight * T² * (alpha * target_KL + beta * non_target_KL)` |
| Optional feature transfer | Add `r * feature_weight * stage_feature_loss` |

DKD defaults to `alpha=1`, `beta=8`, `T=4`, and `weight=0.5`. The outer DKD weight scales the decoupled response terms; it does not reduce CE. Feature transfer uses explicit stage pairs, learned 1×1 channel projections, bilinear spatial alignment, and normalized feature matching. This is a simple projected stage loss, **not a reproduction of ReviewKD**.

Both networks see the very same transformed tensor. Basic augmentation uses a resized crop. Strong augmentation increases crop variation and adds color jitter and occasional blur. Horizontal flipping defaults to **off**, because directional labels may change under reflection. Configure it only when class semantics permit. By default no examples are discarded; sample removal happens only when an explicit, compatible surgery plan is configured.

## Teacher assistants

For a large capacity gap, train an intermediate model using [assistant.toml](configs/assistant.toml). Then create another student configuration with a smaller model, `teacher.name = "resnet18"`, and `teacher.checkpoint = "runs/assistant/student.pt"`. Each stage uses a trained artifact from the preceding stage, so the same trainer supports a progressive chain.

The report records the teacher/student parameter ratio. A ratio above 10 triggers a suggestion to consider an assistant; this is an implementation heuristic, not a validated scientific threshold. Compare direct KD and assistant-based KD on validation before choosing a chain.

## Artifacts, resuming, and deployment budgets

Each run writes:

| Artifact | Purpose |
|---|---|
| `config.json` | Effective configuration |
| `data.json` | Split provenance and any applied surgery plan hash/counts |
| `teacher.json` | Teacher quality, capacity ratio, checkpoint hash |
| `history.json` | Per-epoch losses, accuracy, timing, learning rate |
| `best.pt` | Full training state from the best validation epoch |
| `last.pt` | Full training state from the last completed epoch |
| `student.pt` | Best student weights and label/preprocessing metadata only |
| `summary.json` | Quality, inference costs, training cost, budget checks |

```bash
python -m kd train --config configs/student.toml \
  --resume runs/student_dkd/last.pt
```

Resume uses the original directory and training configuration, restores optimizer/scheduler/feature projections and random-generator states, and verifies the teacher hash. An interrupted epoch is replayed from the preceding completed epoch. Exact CPU recovery is tested; identical outcomes across different hardware/PyTorch versions are not promised. Changing the total epoch schedule requires a new experiment initialized with `student.checkpoint`, not resume. Individual runs are resumable; the adaptive sweep coordinator itself is not. A failed sweep retains `progress.json` and each completed run for inspection.

Uncomment `benchmark.max_parameters`, `max_flops`, and/or `max_latency_ms` to enforce a deployment budget during sweep selection. Cost reports describe the **student only**, excluding the teacher and feature projections. FLOPs are explicitly an estimate of twice the Conv2d/Linear multiply-accumulate count; other operations are excluded. Latency is synchronized batch-one model-forward latency, with warmup and median/p95, excluding image loading, preprocessing, and transfer. Measure on the intended deployment hardware before interpreting latency limits. Training overhead in sweep reports is epoch time relative to the supervised baseline on the same machine.

See [docs/design.md](docs/design.md) for the adapter contracts, design choices, test coverage, and source references.
