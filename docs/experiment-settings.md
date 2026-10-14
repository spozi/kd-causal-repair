# Experiment settings: teacher training, repair, and knowledge distillation

Updated: 14 October 2026.

The experiment scope is **all 13 project datasets at four profiles**, giving
**52 dataset/profile settings**. This document defines the dataset-specific
policies once and the shared teacher, student, KD, and repair parameters once.

The saved main-project repair matrix is the reference protocol. Additional
teacher-training baselines should be compared across that same dataset scope,
not only CIFAR-10 and CIFAR-100. Their implementation and configuration status
are distinguished in Section 6; this document does not claim unrun comparisons
are completed.

Sources of truth are the [saved matrix plan](../runs/experiment1-multidataset/matrix-plan/protocol.json),
[dataset catalog](../kd/catalog/v1.json), and each run's frozen configuration and
protocol. Historical artifacts take precedence over newer code defaults.

## 1. Datasets and splits

All models receive RGB `32 x 32` inputs. Split seed is `2026`.

| Dataset | Source ID | Classes | Version | Validation/test policy before profile sampling | Horizontal flip |
|---|---|---:|---|---|---|
| CIFAR-10 | `cifar10` | 10 | `1.0` | Stratified 10% of official training data for validation; official test | `p=0.5` |
| CIFAR-100 | `cifar100` | 100 | `1.0` | Stratified 10% of official training data for validation; official test | `p=0.5` |
| SVHN | `svhn` | 10 | `1.0` | Stratified 10% of official training data for validation; official test | Disabled |
| CINIC-10 | `cinic10` | 10 | `1.0` | Official train/validation/test partitions | `p=0.5` |
| GTSRB | `gtsrb` | 43 | `1.0` | Stratified 10% of official training data for validation; official test | Disabled |
| Fashion-MNIST | `fashionmnist` | 10 | `1.0` | Stratified 10% of official training data for validation; official test | Disabled |
| PathMNIST | `pathmnist` | 9 | `2.0` | Official train/validation/test partitions | Disabled |
| BloodMNIST | `bloodmnist` | 8 | `2.0` | Official train/validation/test partitions | Disabled |
| DermaMNIST | `dermamnist` | 7 | `2.0` | Official train/validation/test partitions | Disabled |
| OrganAMNIST | `organamnist` | 11 | `2.0` | Official train/validation/test partitions | Disabled |
| Caltech-101 | `caltech101` | 101 | `1.0` | Deterministic stratified 70% train / 10% validation / 20% protocol test | Disabled |
| EuroSAT | `eurosat` | 10 | `1.0` | Deterministic stratified 80% train / 10% validation / 10% protocol test | Disabled |
| STL-10 | `stl10` | 10 | `1.0` | Stratified 10% of official labeled training data for validation; official test | Disabled |

Official validation partitions retain their original membership before
long-tail subsampling. Test partitions are never profile-resampled and retain
their native distribution, which is not necessarily balanced. The saved matrix
uses `confirmation_fraction=0`; a directory named `confirmatory` does not imply
an additional independent holdout. Actual sample counts, class counts, and split
hashes are recorded separately for every dataset/profile run.

## 2. Profiles and preprocessing

| Profile | Sampling factor | Teacher and student epochs | Repair target classes | Dataset/profile settings |
|---|---:|---:|---|---:|
| `balanced` | 1 | 81 | All classes | 13 |
| `lt-if10` | 10 | 49 | Lowest-frequency `ceil(C/2)` classes | 13 |
| `lt-if50` | 50 | 71 | Lowest-frequency `ceil(C/2)` classes | 13 |
| `lt-if100` | 100 | 81 | Lowest-frequency `ceil(C/2)` classes | 13 |
| **Total** | | | | **52** |

For class index `k` among `C` classes, profile sampling retains
`max(1, round(n_k * factor ** (-k / (C - 1))))` examples. It is applied to both
training and validation in dataset label order. Target classes are then resolved
from actual training counts, with class-index tie breaking. For naturally
unequal datasets, the sampling factor need not equal the realized largest/smallest
class ratio. `balanced` means no additional resampling, not artificial equalization.

### Base augmentation

| Dataset group | Training transform, `augmentation="basic"` | Validation/test/diagnostic transform |
|---|---|---|
| CIFAR-10, CIFAR-100, SVHN, CINIC-10 | Random crop `32 x 32`, padding `4`; flip only where enabled in Section 1 | Tensor conversion and normalization |
| Fashion-MNIST, PathMNIST, BloodMNIST, DermaMNIST, OrganAMNIST | Resize to `32 x 32`, then random crop with padding `4`; no flip | Resize to `32 x 32`, tensor conversion, normalization |
| GTSRB, Caltech-101, EuroSAT, STL-10 | Random resized crop `32 x 32`, scale `(0.85, 1.0)`, aspect ratio `(0.9, 1.1)`; no flip | Resize shorter edge to `36`, center crop `32 x 32`, RGB conversion, tensor conversion, normalization |

Fashion-MNIST and the MedMNIST adapters convert images to RGB. Ordinary
main-matrix training does not add color jitter, batch mixing, or sample deletion.
Additional teacher augmentations are specified only in Section 6.

### Normalization

| Dataset group | Mean | Standard deviation |
|---|---|---|
| CIFAR-10, SVHN, PathMNIST, BloodMNIST, DermaMNIST, OrganAMNIST | `(0.5, 0.5, 0.5)` | `(0.5, 0.5, 0.5)` |
| CIFAR-100 | `(0.5071, 0.4867, 0.4408)` | `(0.2675, 0.2565, 0.2761)` |
| CINIC-10 | `(0.47889522, 0.47227842, 0.43047404)` | `(0.24205776, 0.23828046, 0.25874835)` |
| Fashion-MNIST | `(0.2860, 0.2860, 0.2860)` | `(0.3530, 0.3530, 0.3530)` |
| GTSRB, Caltech-101, EuroSAT, STL-10 | `(0.485, 0.456, 0.406)` | `(0.229, 0.224, 0.225)` |

Sources: [data adapters](../kd/data.py), [matrix profile definitions](../kd/neuron_surgery_matrix.py),
and [dataset catalog](../kd/catalog/v1.json).

## 3. Teacher and student training

### 3.1. Dataset-independent architecture names

Use **ConvNet-6-W32** for the reference teacher and **ConvNet-6-W16** for the
reference student. These names describe six convolutional layers and the initial
channel width, not a dataset or a permanently assigned model role.

The existing configuration identifiers `cifar_teacher` and `cifar_student` are
historical factory/checkpoint names for these architectures. The factory creates
`CifarCNN(width=32 or 16, num_classes=C)`; it contains no CIFAR loader or fixed
10/100-class restriction. The same architecture is instantiated separately for
each of the 13 datasets, with its classifier sized to that dataset's class count.
This document changes display names only; it does not rename those identifiers
or invalidate saved checkpoint metadata and frozen protocols.

Architecture names are dataset-independent, but **trained checkpoints remain
dataset/profile-specific**, including their class ordering and preprocessing.
There is not one shared trained teacher checkpoint for all 13 datasets.

### 3.2. Reference pairing and shared training settings

The architecture rows below describe the preserved main-matrix reference pair;
they do not describe every literature architecture in Section 3.3. The repaired
teacher retains its original architecture. Only its bounded repair phase uses
the optimization overrides in Section 5.

| Parameter | Original teacher | Student distilled from original or repaired teacher |
|---|---|---|
| Architecture label | ConvNet-6-W32 | ConvNet-6-W16 |
| Architecture | Six-convolution CNN; widths `32 -> 64 -> 128` | Six-convolution CNN; widths `16 -> 32 -> 64` |
| Stage structure | Two bias-free `3 x 3` convolutions, each with BatchNorm/ReLU; then `2 x 2` max pooling | Same structure |
| Classification head | Adaptive average pooling to `2 x 2`, flatten, linear classifier with `C` outputs | Same structure |
| Trainable parameter count | `287456 + 513*C` | `72304 + 257*C` |
| Initialization | Random, from scratch; no pretrained checkpoint | Random, from scratch; matched by seed across compared conditions |
| Seed(s) | `41` | `42, 43, 44` |
| Epoch budget | Profile-specific, Section 2 | Same profile-specific budget |
| Batch size | `128` | `128` |
| Optimizer | SGD | SGD |
| Initial learning rate | `0.05` | `0.05` |
| Momentum / weight decay | `0.9` / `0.0005` | Same |
| Nesterov | Disabled | Disabled |
| Scheduler | Cosine annealing; `T_max=profile epochs`, minimum LR `0`, one step per epoch | Same |
| Learning-rate warmup / early stopping | None / none; run the full budget | Same; KD-loss warmup is separate |
| Objective | Hard-label cross-entropy | Cross-entropy plus classical KD, Section 4 |
| Augmentation | Dataset-specific base policy, Section 2 | Same base policy |
| Device | Assigned CUDA device in saved GPU runs | Same job device |
| Configured precision | `auto` | `auto` |
| CUDA mode / memory format | `fast` / channels-last | Same |
| CUDA data-loader workers | Generator chooses `4`; available frozen CIFAR protocols record `4` | Same |
| CPU threads / persistent workers / prefetch | `4` / disabled / `2` | Same |
| Inference benchmark | `20` warmup and `100` measured iterations | Same |

Parameter counts exclude BatchNorm buffers. CUDA `auto` selects BF16 where
supported and scaled FP16 otherwise; resolved precision and hardware come from
run metadata. Fast mode permits TF32, cuDNN autotuning, and nondeterministic
kernel selection. Each job's saved config is authoritative for runtime choices.

Sources: [models](../kd/models.py), [baseline construction](../kd/neuron_surgery_baselines.py),
[training engine](../kd/engine.py), and an
[available frozen baseline protocol](../runs/experiment1-multidataset/cifar10/lt-if10/baselines/confirmatory/protocol.json).

### 3.3. Established teacher-student pairings from the literature

Literature discovery used **Consensus MCP** on 14 October 2026, including fetched
records for CRD and DKD. Exact pairings were checked against the authors' benchmark
tables. The entries below are established benchmark examples, not a ranking of
usage frequency or evidence that each pairing works best on all 13 datasets.
The literature benchmark column records where the pair was evaluated; it does
not make the architecture dataset-specific.

| Teacher | Student | Comparison type | Literature benchmark / evidence | Current project support |
|---|---|---|---|---|
| ResNet-34 | ResNet-18 | Same-family depth reduction | ImageNet; [DKD, Table 8](https://arxiv.org/html/2203.08679) | Both factories and teacher-repair maps available; fresh pairing runs required |
| ResNet-56 | ResNet-20 | Same-family depth reduction | CIFAR-100; [CRD](https://github.com/HobbitLong/RepDistiller#benchmark-results-on-cifar-100), [DKD, Table 6](https://arxiv.org/html/2203.08679) | Student factory available; teacher variant not exposed; teacher-repair map needed |
| ResNet-110 | ResNet-32 | Same-family depth reduction | CIFAR-100; [CRD](https://github.com/HobbitLong/RepDistiller#benchmark-results-on-cifar-100), [DKD, Table 6](https://arxiv.org/html/2203.08679) | Neither variant exposed in the factory; teacher-repair map needed |
| ResNet-32x4 | ResNet-8x4 | Same-family, widened residual networks | CIFAR-100; [CRD](https://github.com/HobbitLong/RepDistiller#benchmark-results-on-cifar-100), [DKD, Table 6](https://arxiv.org/html/2203.08679) | Width-4 variants and teacher-repair map missing |
| WRN-40-2 | WRN-16-2 | Same-family depth reduction | CIFAR-100; [CRD](https://github.com/HobbitLong/RepDistiller#benchmark-results-on-cifar-100), [DKD, Table 6](https://arxiv.org/html/2203.08679) | WideResNet factories and teacher-repair map missing |
| WRN-40-2 | WRN-40-1 | Same-family width reduction | CIFAR-100; [CRD](https://github.com/HobbitLong/RepDistiller#benchmark-results-on-cifar-100), [DKD, Table 6](https://arxiv.org/html/2203.08679) | WideResNet factories and teacher-repair map missing |
| VGG-13 | VGG-8 | Same-family depth reduction | CIFAR-100; [CRD](https://github.com/HobbitLong/RepDistiller#benchmark-results-on-cifar-100), [DKD, Table 6](https://arxiv.org/html/2203.08679) | VGG factories and teacher-repair map missing |
| ResNet-32x4 | ShuffleNet-V1 / ShuffleNet-V2 | Cross-family compression | CIFAR-100; [CRD](https://github.com/HobbitLong/RepDistiller#benchmark-results-on-cifar-100), [DKD, Table 7](https://arxiv.org/html/2203.08679) | Widened teacher, mobile students, and teacher-repair map missing |
| ResNet-50 | MobileNet-V2 | Cross-family compression | CIFAR-100; [CRD](https://github.com/HobbitLong/RepDistiller#benchmark-results-on-cifar-100), [DKD, Table 7](https://arxiv.org/html/2203.08679) | Teacher factory/repair map available; student factory missing |
| ResNet-50 | MobileNet-V1 | Cross-family compression | ImageNet; [DKD, Table 9](https://arxiv.org/html/2203.08679) | Teacher factory/repair map available; student factory missing |

`x4` is a width multiplier: the project's `resnet8` is **not** ResNet-8x4.
Likewise, MobileNet-V1 and V2 are distinct architectures. The existing
`resnet8`/`resnet20` factories expose fixed-width small-image residual models;
their general internal builder does not make every literature variant selectable
through the factory or supported by the repair pipeline.

Research records: Tian, Krishnan, and Isola,
[Contrastive Representation Distillation](https://consensus.app/papers/contrastive-representation-distillation-tian-krishnan/188e1661a13253c9844c4ab6cb7725d8/?utm_source=chatgpt),
ICLR 2020 (2019 preprint; DOI `10.48550/arxiv.1910.10699`), and Zhao et al.,
[Decoupled Knowledge Distillation](https://consensus.app/papers/decoupled-knowledge-distillation-zhao-cui/17a111d2eec1508687419f4b97b82b28/?utm_source=chatgpt),
CVPR 2022 (DOI `10.1109/CVPR52688.2022.01165`). Pairing evidence comes from
the primary paper/author sources linked in the table, not only paper abstracts.

### 3.4. Applying pairings across all datasets

Retain ConvNet-6-W32 -> ConvNet-6-W16 as the historical control. A practical
first additional pairing is **ResNet-34 -> ResNet-18**, because both models and
the teacher's repair map already exist. This is a project-specific integration
priority, not a claim that it is empirically best. For this project's `32 x 32`
inputs, explicitly freeze the small-input `3 x 3`, stride-1 stem without initial
max pooling; report it as a small-stem adaptation, not an exact ImageNet
reproduction. The legacy small-stem config keys are `cifar_resnet34` and
`cifar_resnet18`; their prefix also does not restrict dataset choice.

WRN-40-2 -> WRN-16-2 provides a literature-backed compact same-family extension;
ResNet-50 -> MobileNet-V2 adds a cross-family/mobile extension once the missing
factories and tests are integrated. New residual, WideResNet, or VGG teachers
require verified channel/consumer parameter masks and preservation-feature maps;
ordinary KD compatibility alone is not sufficient for teacher repair.

Each selected pair must retain **all 13 datasets and four profiles**, with a
dataset-specific output head, fresh teacher training, seed-matched original and
repaired teacher students, and the shared split/KD rules above. Stem, block
variant, width, pooling, initialization, and training budget must be explicit.
The reference parameter-count formulas and stage widths apply only to ConvNet-6,
not these alternative architectures. Different paper schedules and resolutions
mean the controlled project comparison is not a reproduction of published scores.

For `P` fully crossed pairings, there are `52*P` dataset/profile/pair settings,
before repair candidates or teacher-training recipe expansion. The saved
52-setting main matrix contains only the reference ConvNet-6 pair; the literature
table does not add trained results or silently change its frozen configurations.

## 4. Knowledge distillation

The same downstream KD settings apply across all datasets and teacher conditions.

| Parameter | Setting |
|---|---|
| Method | Classical logits KD, `method="kd"` |
| Teacher source | Frozen selected original or repaired teacher checkpoint |
| Temperature | `T=4`; temperature correction `T^2=16` |
| Response weight / hard-label CE weight | `weight=0.5` / `0.5` |
| Response warmup | `5` epochs; multipliers `0.2, 0.4, 0.6, 0.8, 1.0`, then `1.0` |
| KL direction / reduction | Teacher distribution to student distribution / `batchmean` |
| Teacher execution | Evaluation mode, frozen parameters and BatchNorm statistics, no gradients |
| Input views | Teacher and student receive the same augmented tensor |
| Feature distillation | Disabled, `feature_weight=0` |
| Other student interventions | No label smoothing, EMA, SAM, batch mixing, calibration loss, CPC, adaptive focal loss, or sample deletion |
| Temperature/weight search | None in the reference protocol |

For one-based epoch `e`:

```text
r_e = min(e / 5, 1)
p_t = softmax(teacher_logits / 4)
p_s = softmax(student_logits / 4)
L_student = 0.5 * CE(student_logits, labels)
          + 0.5 * r_e * 16 * KL(p_t || p_s)
```

KL sums over classes and averages over examples. The CE coefficient stays
`0.5` during warmup; it is not renormalized. Teacher-training interventions
do not carry over to students. A supervised-student reference, where added,
uses the same student settings but full-strength CE without a teacher.

Source: [distillation losses](../kd/losses.py).

## 5. Teacher repair

The main-matrix repair protocol applies to every dataset/profile.

| Parameter | Setting |
|---|---|
| Starting checkpoint | The dataset/profile's selected original CE teacher |
| Target classes | Profile policy in Section 2 |
| Hard-example source | Canonical, non-augmented training diagnostics |
| Target fraction | `0.2` per target class; retain all mistakes, then fill with highest-NLL correct examples |
| Quota caveat | Not a strict 20% cap when mistakes exceed the quota |
| Companions | `5`; cosine matching of normalized pooled `stage3` features |
| Sparse-class fallback | Enabled for the main matrix; correct non-target same-class examples first, then same-class fallback with replacement; singleton classes use gradient-only attribution |
| Localization stages / score | `stage2, stage3` / differential activation discrepancy times absolute error-margin gradient |
| Bootstrap | `20` within-class repetitions; equal class weighting; seed `2026` |
| Screening thresholds | Top fraction `0.25`; stability at least `0.7` |
| Causal screening | Individually zero a channel; positive target-margin gain and preservation accuracy loss at most `0.005` |
| Channel-budget grid / LR grid | `[2, 4, 8]` / `[0.001, 0.005]`; at most six feasible candidates |
| Candidate initialization | Each candidate starts from the same original teacher checkpoint |
| Repair epochs / seed | `20` / `2026` |
| Replay | `640` examples per target/preservation stream per epoch; class-balanced sampling with replacement |
| Batch size / updates | `64` target + `64` preservation examples per update; `10` updates per epoch |
| Optimizer | SGD; momentum `0.9`; **weight decay `0`** |
| Scheduler | Cosine annealing; `T_max=20`; minimum LR `0` |
| Precision / memory format | Inherit the teacher run's settings |
| Target CE coefficient | `1.0` |
| Preservation KD | Original teacher as frozen anchor; weight `1.0`, `T=4`, including `T^2` |
| Feature preservation | Weight `0.25`, matching `stage2/stage3` features |
| Preservation hard-label CE | Disabled, weight `0` |
| Mutable coordinates | Selected producing convolution rows, corresponding BatchNorm affine coordinates, and directly connected consumer weights |
| Frozen state | Every unselected parameter coordinate and every model buffer |
| Candidate checkpoint | End of its repair fit; select among candidates using validation |
| Acceptance | Strict target macro-recall improvement; overall validation accuracy at least `baseline - 0.005` |
| Selection tie-breaking | Lower target-balanced NLL, fewer channels, lower learning rate |

```text
L_repair = CE(candidate_target_logits, target_labels)
         + 16 * KL(softmax(anchor_preservation_logits / 4)
                   || softmax(candidate_preservation_logits / 4))
         + 0.25 * feature_preservation_loss
```

There is no permanent pruning or sample deletion. Exact checks enforce frozen
coordinates and buffers. In the main matrix, rejected repairs terminate with
`no_repair_selected` before new repaired-teacher students or final test
construction; accepted repairs trigger three matched students.

Sources: [repair implementation](../kd/neuron_surgery.py),
[study orchestration](../kd/neuron_surgery_study.py), and the
[frozen repair protocol](../runs/experiment1-multidataset/cifar10/lt-if10/studies/confirmatory/protocol.json).

## 6. Additional teacher-training baseline comparisons

**Dataset scope remains all 13 datasets and all four profiles.** The following
recipes are implemented comparison methods, not conditions already run throughout
the saved main matrix. A full-scope extension should freeze dataset/profile configs
using Sections 1-5 and declare any overrides before training.

| Teacher condition | Intervention beyond the dataset's base policy |
|---|---|
| `control` | Ordinary hard-label CE |
| `randaugment` | RandAugment `N=2, M=9` before tensor conversion; fill uses normalization mean RGB |
| `mixup` | `alpha=0.2`; one Beta-distributed mixing coefficient per batch; mix images and labels |
| `cutmix` | `alpha=1.0`; patch replacement; labels weighted by actual clipped patch area |
| `label_smoothing` | `epsilon=0.1`; targets `0.9*one_hot + 0.1/C` |
| `balanced_softmax` | Train with CE of `logits + log(training_counts)`; inference/KD use raw logits |
| `mse_logits` | MSE of raw logits against one-hot labels |
| `mse_probabilities` | MSE of softmax probabilities against one-hot labels |
| `sam` | Radius `rho=0.05`; two gradient passes on the same batch; BatchNorm statistics advance only on the first pass |
| `ema` | Parameter EMA decay `0.999`; copy current buffers; validate/select/export EMA weights |
| `randaugment_cutmix_ema` | RandAugment + CutMix + EMA using the values above; CE without label smoothing |
| `teacher_repair` | Reuse the selected `control` teacher and the repair method in Section 5 |

Mixup/CutMix apply to every eligible batch. Both MSE losses average over examples
and classes, with scale `1`; probability MSE is summed-class Brier score divided
by `C`. Balanced Softmax is redundant with CE for exactly equal class counts,
but a naturally unequal `balanced` dataset can still have unequal training counts.
Augmentation operations must be reviewed for label validity on digits, traffic
signs, and medical images before freezing the full-scope comparison.

### Configuration status and non-shared settings

Only two ready-made teacher-baseline presets currently exist:
[CIFAR-10-LT100](../configs/teacher_baselines/cifar10_lt100.json) and
[balanced CIFAR-100](../configs/teacher_baselines/cifar100_balanced.json).
**The all-13-dataset baseline extension is the intended scope, not an already
configured or completed 52-setting recipe benchmark.** These presets are
implementation pilots and must not redefine the main experiment's dataset scope.

| Difference | Saved main matrix | Existing baseline pilots |
|---|---|---|
| Runtime configuration | Assigned CUDA device, precision `auto`, CUDA workers `4` | Device `auto`, precision `float32`, workers `0` |
| Sparse-class fallback | Enabled | Disabled |
| Supervised-student reference | Not included | One per student seed |
| Rejected repair handling | Terminate repair study | Keep a no-change condition and reuse control KD students |
| Recipe selection | Original-vs-repaired paired transfer evaluation | Mean student validation balanced accuracy for long-tail, accuracy for balanced |

All use the same shared optimization and KD values unless an explicit config
overrides them. Both existing pilots happen to use 81 epochs because their profiles
are IF100 and balanced. Full-scope configs must declare runtime/fallback choices;
they cannot silently inherit those two pilots' restricted settings.

Teacher data, architecture, initialization, and student training are matched
within each comparison. Compute is not equalized: SAM adds a second gradient
pass and repair adds localization plus candidate search. These are controlled
baselines, not tuned SOTA reproductions. SRA, ACL, and Bayesian teachers remain
researched, unimplemented follow-ups.

Sources: [teacher recipes](../kd/teacher_training.py),
[benchmark orchestration](../kd/teacher_baselines.py),
[repair adapter](../kd/teacher_repair_baseline.py), and
[research and benchmark protocol](teacher-training-baselines.md).

## 7. Selection, evaluation, and experiment size

| Decision or reporting rule | Setting |
|---|---|
| Ordinary teacher/student checkpoint | Highest validation accuracy; EMA conditions evaluate EMA weights |
| Repaired teacher | Validation gate and tie-breaking in Section 5 |
| Main-matrix student transfer success | Mean target recall gain at least 1 percentage point, positive gains in at least two of three seeds, mean overall accuracy loss at most 0.5 points |
| Test access | After the applicable teacher selection and student checkpoints are frozen |
| Outcomes | Accuracy, balanced accuracy, target macro recall, NLL, Brier score, ECE; parameter/latency/training-cost information |
| Provenance | Frozen configs, split identities, source/checkpoint hashes, and protocol/completion artifacts |
| Retuning from test results | Not part of the protocol |

The main matrix contains 52 planned teacher fits and 156 original-teacher KD
student fits. Repair adds at most six candidate fits per setting (312 maximum)
and three downstream students per accepted repair. It has no separate supervised
student reference. Recipe-expanded comparisons require their own compute budget;
52 dataset/profile settings do not mean only 52 training fits.

The [saved matrix summary](../runs/experiment1-multidataset/matrix_summary.json)
reports all 52 jobs finished, 23 accepted repairs, 29 no-repair outcomes, and 10
successful transfers. These are saved summary counts, not a new artifact audit.
The 23 acceptances imply 69 repaired-teacher student fits.

Teacher seed `41` and three student seeds are not three independent teacher
replications. Comparisons are paired within dataset/profile/seed; predictions
must not be pooled across datasets. Previously examined test images are not an
independent confirmatory endpoint simply because a new runner gates access.

## 8. Related experiments and preserved evidence

| Family | Scope | Distinction |
|---|---|---|
| Teacher architecture Study A | CIFAR-10, CIFAR-100, BloodMNIST, EuroSAT, GTSRB at IF10/IF100; 20 additional teacher/profile jobs | ResNet-18/ResNet-34 teachers and separate late-stage repair maps |
| Student architecture Study B | The 23 accepted-repair settings | Additional ResNet-8/ResNet-20 students, paired under original/repaired teachers |
| Alternative distillation losses | The 23 accepted-repair settings spanning 11 datasets | DKD, RLD, LoCa from the original teacher |
| Matched loss-balance follow-up | The same 23 accepted-repair settings | DKD response weight `0.15`, RLD response weight `0.2` |

Alternative-loss runs retain `T=4` and five-epoch warmup. KD/LoCa use CE and
response coefficients `0.5/0.5`; published-setting DKD/RLD use full CE plus
response weight `1`, with internal `alpha=1, beta=8`. These conditional
follow-ups are not replacements for the full 13-dataset coverage. Details are in
[architecture studies](architecture-studies-statistics.md) and
[distillation comparison design](distillation-baselines-comparison.md).

The original single-dataset experiment in `runs/cifar10-lt-neuron-surgery`,
its [saved teacher config](../runs/cifar10-lt-multiseed/teacher_f100/config.json),
and [report](../runs/cifar10-lt-neuron-surgery/report.md) remain separate,
preserved evidence. Its omitted runtime fields must not be assigned today's
defaults retroactively. Confirmation and component-ablation variants retain
their own [campaign protocol](neuron-surgery-campaign.md). No historical
configuration, checkpoint, or result is changed by this document.

## 9. Read-only inspection

From the repository root:

```bash
jq '{datasets, profiles, job_count: (.jobs | length)}' \
  runs/experiment1-multidataset/matrix-plan/protocol.json

jq '{job_count, all_jobs_finished, complete_count, successful_transfer_count}' \
  runs/experiment1-multidataset/matrix_summary.json
```
