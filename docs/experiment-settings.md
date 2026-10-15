# Experiment settings: teacher training, repair, and knowledge distillation

Full-scope and pretrained-default update: 15 October 2026 (`kd-baseline`).

The implemented **full experiment includes 16 datasets**: the original 13 datasets
plus Colored MNIST, Waterbirds, and UrbanCars. All belong to one teacher-quality,
teacher-repair, and downstream knowledge-distillation (KD) benchmark. The three
shortcut datasets are an additional evaluation family, not a replacement for
the original datasets or a separate deferred experiment.

| Evaluation family | Dataset count | Conditions / primary student score | Execution status |
|---|---:|---|---|
| Class / long-tail | 13 | Four class profiles; pretrained ResNet-34 -> ResNet-18 at 32 pixels; accuracy for `balanced`, class-balanced accuracy for long-tail | Primary transfer protocol implemented; compact presets retained separately |
| Shortcut / nuisance groups | 3 | Native shortcut construction, no IF profiles; worst-group accuracy | Adapters, group selection, repair and mechanism controls implemented; provision real data before running |
| Full experiment | 16 | Both families, with their declared resolutions and metrics | Planner, preflight, execution/resume and overview reporting implemented; full-budget results not measured |

Sections 1-10 describe the combined experiment. Appendix A contains the optional
32-pixel from-scratch references and historical evidence, not the pretrained default.
The [shortcut companion](causal-benchmark-phase1.md) supplies construction,
research sources, and group-aware adaptation details for the other three.

Putting all 16 in the same scope does not mean forcing the same resolution,
class resampling, or metric onto them. The full run needs a fresh plan whose
dataset list includes all 16 and whose jobs declare the applicable family,
profile/environment, pairing, initialization, and evaluation endpoints.
No dataset/weight download or full-budget training was initiated during implementation validation.

Historical settings come from the [saved plan](../runs/experiment1-multidataset/matrix-plan/protocol.json)
and each run's frozen protocol; they are not rewritten as new 100/20-epoch runs.
Current defaults come from the
[baseline generator](../kd/teacher_baseline_matrix.py). The
[detailed results panel](#101-teacher-quality-and-downstream-transfer) and
[full-dataset overview](#102-full-16-dataset-overview) below are blank templates,
not measured performance.

## 1. Datasets and splits

### 1.1. Complete 16-dataset roster

This is the canonical roster for the full experiment. `Class profiles` means
`balanced`, `lt-if10`, `lt-if50`, and `lt-if100`; `native shortcut` means
the dataset-specific construction without extra class-imbalance profiles.
Inputs below are the project input sizes, not necessarily upstream/native sizes.
The original 13 retain `32 x 32` inputs and unchanged archive/catalog and split
identities. Their new `imagenet-smallstem-v1` protocol combines the compact crop
geometry with RGB conversion and ImageNet normalization, without 224-pixel upsampling.
Source IDs/releases in the original 13 rows are immutable catalog entries. The
three additions use separate version-`1.0` adapters; this version names the
project adapter contract, not a new upstream release. Their loaded content,
metadata, renderer and split identities are hashed in run provenance.

| Dataset | Source ID / release | Classes | Input (channels x height x width) | Conditions | Split policy | Implementation |
|---|---|---:|---|---|---|---|
| CIFAR-10 | `cifar10` / `1.0` | 10 | RGB `3 x 32 x 32` | Class profiles | A | Available |
| CIFAR-100 | `cifar100` / `1.0` | 100 | RGB `3 x 32 x 32` | Class profiles | A | Available |
| SVHN | `svhn` / `1.0` | 10 | RGB `3 x 32 x 32` | Class profiles | A | Available |
| CINIC-10 | `cinic10` / `1.0` | 10 | RGB `3 x 32 x 32` | Class profiles | B | Available; pretraining-overlap warning |
| GTSRB | `gtsrb` / `1.0` | 43 | RGB `3 x 32 x 32` | Class profiles | A | Available |
| Fashion-MNIST | `fashionmnist` / `1.0` | 10 | RGB `3 x 32 x 32` | Class profiles | A | Available |
| PathMNIST | `pathmnist` / `2.0` | 9 | RGB `3 x 32 x 32` | Class profiles | B | Available |
| BloodMNIST | `bloodmnist` / `2.0` | 8 | RGB `3 x 32 x 32` | Class profiles | B | Available |
| DermaMNIST | `dermamnist` / `2.0` | 7 | RGB `3 x 32 x 32` | Class profiles | B | Available |
| OrganAMNIST | `organamnist` / `2.0` | 11 | RGB `3 x 32 x 32` | Class profiles | B | Available |
| Caltech-101 | `caltech101` / `1.0` | 101 | RGB `3 x 32 x 32` | Class profiles | C | Available |
| EuroSAT | `eurosat` / `1.0` | 10 | RGB `3 x 32 x 32` | Class profiles | D | Available |
| STL-10 | `stl10` / `1.0` | 10 | RGB `3 x 32 x 32` | Class profiles | A, labeled training only | Available; pretraining-overlap warning |
| Colored MNIST | `colored_mnist` / adapter `1.0`, torchvision MNIST + pinned renderer | 2 | RGB `3 x 32 x 32` adaptation | Native shortcut | E | Available |
| Waterbirds | `waterbirds` / adapter `1.0`, local released CSV/images | 2 | RGB `3 x 224 x 224` | Native shortcut | F | Available |
| UrbanCars | `urbancars` / adapter `1.0`, local author-layout composites | 2 | RGB `3 x 256 x 256` | Native shortcut | G | Available |

`Available` means existing implementation, not a newly completed full-budget
run or a completed hardware preflight. The three added inputs/pairings are
project adaptations, not reproductions of published benchmark scores.
Sources: [retained dataset catalog](../kd/catalog/v1.json) and
[shortcut adapters](../kd/shortcut_data.py).

### 1.2. Split policies and source separation

- **A:** Stratified 10% validation holdout from official training data; official test.
- **B:** Official train/validation/test partitions.
- **C:** Deterministic stratified 70% train / 10% validation / 20% protocol test.
- **D:** Deterministic stratified 80% train / 10% validation / 10% protocol test.
- **E:** Split underlying official MNIST training images before coloring; derive
  validation only from training environments; official test digits reserved.
- **F:** Released Waterbirds split metadata; no re-stratification or IF resampling.
- **G:** Released/author-defined UrbanCars partitions and group mapping; pin the
  artifact or generation recipe, not a presumed Waterbirds-style metadata file.

The class family uses split seed `2026`; profile sampling affects train and
validation only. Test partitions retain native distributions and are not
resampled. Its `confirmation_fraction=0` means a folder called `confirmatory`
does not establish an independent holdout.

For the shortcut family, split seed `2026` fixes membership, label noise,
environments and rendering independently of model seeds. Colored MNIST uses a
stratified 10% holdout of official training digits before coloring. Train and
validation must each contain every required group. Waterbirds rejects duplicate
composite paths and foreground reuse across partitions. Its background lineage
is unavailable unless supplied; UrbanCars' released files do not export original
constituent IDs. These limitations are recorded, not treated as a complete
source-disjointness or counterfactual audit. In every family,
test images cannot enter fitting, localization, companion matching, or selection.
Record actual counts, class/group ordering, archive identities, and split hashes.

Pretraining is an additional data exposure. CINIC-10 includes ImageNet samples
and STL-10 was collected from ImageNet; neither endpoint is claimed to be
disjoint from ImageNet-1K pretraining without an exact source/duplicate audit.
The planner, preflight, data provenance and reports carry `audit_status=not_audited`
and an ImageNet-derived warning for these two datasets; exact overlap has not
been established. Other datasets are also not marked as audited-clean.
Sources: [CINIC-10 authors](https://github.com/BayesWatch/cinic-10),
[STL-10 authors](https://cs.stanford.edu/~acoates/stl10/).

### 1.3. Shortcut groups and evaluation environments

| Dataset | Required groups | Training / validation construction | Final test endpoints |
|---|---|---|---|
| Colored MNIST | Observed label x color: 4; retain environment ID separately | Training mismatch probabilities `0.2/0.1`; validation derives only from those environments | Reversed color (`0.9` mismatch) and unbiased color (`0.5` mismatch) on the same official test digits |
| Waterbirds | Bird label x land/water background: 4 | Released train/validation composition | Released test |
| UrbanCars | Vehicle label x background x co-object category: 8 | Released/author-defined train/validation composition | Released/author-defined test |

Colored MNIST uses digit `0-4` versus `5-9` with label-flip probability
`0.25`. Primary classification/group metrics use noisy observed labels; clean
digit category is a separately named diagnostic. Its two test renderings share
original images and fixed labels: they are two correlated endpoints, not two
datasets or independent replications. Construction/provenance details are in
[the shortcut companion](causal-benchmark-phase1.md#2-data-and-preprocessing-decisions-to-freeze).

## 2. Profiles, preprocessing, and augmentation

### 2.1. Family-specific conditions

| Family | Condition policy | Preprocessing / base views |
|---|---|---|
| Original 13 datasets, primary pretrained pair | Four class profiles; `balanced` means no extra resampling, not guaranteed equal natural class counts | `imagenet-smallstem-v1`: RGB conversion before geometry; Appendix A.2's 32-pixel crop/resize/flip policy; ImageNet normalization replaces source-specific normalization |
| Optional compact pairs | Same split/profile membership, different input/model/initialization protocol | Appendix A.2's RGB `32 x 32` views and source-specific normalization; do not pool with the pretrained setting |
| Three shortcut datasets | `native-shortcut`: native label/nuisance construction; no IF10/50/100 profiles | Dataset-specific policies below |

Appendix A.2 records the class family's exact resampling formula and target-class
policy. Class imbalance and shortcut-group imbalance are distinct: do not apply
that formula to the three added datasets merely because they share the roster.

### 2.2. Implemented shortcut preprocessing

| Item | Training base views | Evaluation / diagnostics |
|---|---|---|
| Colored MNIST | Resize digit to `32 x 32`; intensity in red or green, blue zero; tensor conversion; normalize RGB mean/std `(0.5,0.5,0.5)`; no crop/flip/recoloring | Same canonical view; only the prespecified test color renderer changes |
| Waterbirds | Resize to `256 x 256`, center crop `224 x 224`, tensor conversion, ImageNet normalization; no base horizontal flip | Same canonical view |
| UrbanCars | Require native `256 x 256`; horizontal flip `p=0.5`, tensor conversion, ImageNet normalization; no resize/crop | No flip; retain native dimensions |
| ImageNet normalization | Mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)` | Identical normalization |
| Extra teacher augmentation | Section 6's explicit recipe; Colored MNIST RandAugment allows only Identity/Brightness/Contrast/Sharpness, avoiding implicit hue/color removal | No extra recipe augmentation |

These base views are shared by ordinary teachers and KD/supervised students.
Group/environment/source metadata is carried separately; datasets return only
`(image, class_label)` to predictors. Source:
[shortcut preprocessing](../kd/shortcut_data.py).

The three additions do not automatically inherit the class family's SGD,
batch-size, normalization, or operation-space settings. Within each comparison,
declare hardware, precision, workers, batch size, and training exposure.
Across all 16 datasets, students retain their own base views rather than inherit
a teacher's Mixup, CutMix, RandAugment, or EMA intervention.

## 3. Models and training budgets

### 3.1. Declared teacher-student pairings

| Scope | Teacher -> student | Architecture convention / support |
|---|---|---|
| Original 13 datasets, default | Adapted ResNet-34 -> ResNet-18 | `resnet34_smallstem` / `resnet18_smallstem`, both derived from `IMAGENET1K_V1`; `resnet34_resnet18_smallstem_imagenet` preset at 32 pixels |
| Original 13 datasets, optional compact reference | ConvNet-6-W32 -> ConvNet-6-W16 | `convnet6_w32` / `convnet6_w16`; `convnet6` preset, 32 pixels, from scratch |
| Original 13 datasets, optional compact ResNet reference | ResNet-34 -> ResNet-18 | `resnet34_smallstem` / `resnet18_smallstem`; `resnet34_resnet18` preset, 32 pixels, from scratch |
| Colored MNIST | ConvNet-6-W32 -> ConvNet-6-W16 | `convnet6_w32` / `convnet6_w16`, from scratch |
| Waterbirds and UrbanCars | ResNet-34 -> ResNet-18 | `resnet34` / `resnet18`, ordinary stems, both initialized with named ImageNet weights |

Architectures are not dataset-bound: each is fitted independently with the
declared class count/order. The table specifies comparison choices, not model
restrictions. Legacy aliases and exact ConvNet structure are in Appendix A.3.
An optional extra pair does not automatically cross every architecture into the
three shortcut datasets; declare any expansion before creating the final plan.

The adapted pretrained stem is a project-specific initialization, not an official
32-pixel ImageNet weight release. Load the named full ImageNet checkpoint, resize
its `7 x 7` conv1 filters to `3 x 3` by bilinear interpolation (`align_corners=false`),
then restore each output filter's original L2 norm. Use stride `1`, padding `1`,
and identity max-pooling; retain all other pretrained backbone/BatchNorm tensors
and replace the classifier with a new seeded class-count-specific head. The
method is frozen as `bilinear_l2_7to3_v1`; source and adapted-stem tensor digests
are recorded. A nonzero filter that maps to zero is rejected rather than silently
randomized. This is an empirical adaptation, not a guarantee of preserved
ImageNet accuracy or improved target performance. At 32 pixels the four stage
maps are `32/16/8/4`, rather than the ordinary stem's `8/4/2/1`.

### 3.2. Shared budgets and family-specific optimization

| Parameter | Teacher | Student, KD or supervised | Repair |
|---|---|---|---|
| Epoch budget | `100` | `100` | `20` |
| Seeds | `41` | Paired `42, 43, 44` | `2026`; continuation candidate slots additionally use `2027/2028` |
| Initialization matching | Same initial teacher tensors across ordinary recipes within a dataset/seed | Same initial student tensors for each seed across KD arms and supervised reference | Each candidate resets to the same selected CE anchor |
| Objective | Section 6's teacher recipe | Section 4's classical KD, or full CE without a teacher | Section 5's family-specific repair |
| Runtime / evaluation | Record resolved device, precision, workers, and deterministic/fast-mode policy | Match declared evaluation policy within each comparison | Record fitting, screening, validation precision separately |

| Family | Initialization / optimization | Readiness |
|---|---|---|
| Original 13 datasets, primary pair | Named torchvision `ResNet34_Weights.IMAGENET1K_V1` / `ResNet18_Weights.IMAGENET1K_V1`, adapted stems as above; new seeded class-count-specific heads; full-network fine-tuning; SGD LR `0.001`, momentum `0.9`, decay `0.0005`, batch `128`, cosine | Implemented; not the frozen TPAMI/from-scratch settings |
| Optional compact pairs | From scratch; SGD LR `0.05`, momentum `0.9`, decay `0.0005`, batch `128`, cosine | Retained in Appendix A.3 |
| Colored MNIST teacher/students | From scratch; SGD LR `0.05`, momentum `0.9`, decay `0.0005`, batch `128`, cosine | Implemented |
| Waterbirds / UrbanCars teacher/students | Torchvision `ResNet34_Weights.IMAGENET1K_V1` / `ResNet18_Weights.IMAGENET1K_V1`; seeded new two-class heads; SGD LR `0.001`, momentum `0.9`, decay `0.0005`, batch `32`, cosine | Implemented; weight retrieval occurs only when training is explicitly launched |
| Shortcut repair | Section 5.2's group-balanced replay, six-candidate grid and preservation limits; SGD momentum `0.9`, no decay, cosine | Implemented |

The 100/20 budgets are implemented across both families, not convergence guarantees or literature-optimal
epochs. Validation-only tuning needs a prespecified, comparable search budget.
Report actual examples, optimizer updates, gradient passes, and search cost.
The planner's `--epochs` option configures teacher/student budgets for both
families; repair remains `20`. Default runtime is device `auto`, float32, workers
`0`, threads `4`. Initial tensor digests must match within each role/seed across
arms. Pretrained runs additionally record the named release, URL and pretrained
tensor-state digest; cached weight files are hashed and checked against their
advertised filename hash prefix. Exported teachers are loaded from their frozen
checkpoints for KD/evaluation, without re-downloading pretrained weights.

### 3.3. Literature pairings and implementation status

Consensus MCP discovery and fetched CRD/DKD records were checked against primary
benchmark tables. These are established examples, not a frequency ranking or a
claim of best performance across this project's datasets. The literature dataset
is evidence context, not a restriction on the architecture.

| Teacher -> student | Literature evidence | Local support |
|---|---|---|
| ResNet-34 -> ResNet-18 | ImageNet; [DKD T8][dkd] | Small-stem class-family preset and ordinary-stem pretrained shortcut preset implemented |
| ResNet-56 -> ResNet-20 | CIFAR-100; [CRD][crd], [DKD T6][dkd] | Student available; teacher factory entry and repair map needed |
| ResNet-110 -> ResNet-32 | CIFAR-100; [CRD][crd], [DKD T6][dkd] | Factory entries and teacher repair map needed |
| ResNet-32x4 -> ResNet-8x4 | CIFAR-100; [CRD][crd], [DKD T6][dkd] | Widened models and teacher repair map needed |
| WRN-40-2 -> WRN-16-2 | CIFAR-100; [CRD][crd], [DKD T6][dkd] | WideResNet models and teacher repair map needed |
| WRN-40-2 -> WRN-40-1 | CIFAR-100; [CRD][crd], [DKD T6][dkd] | WideResNet models and teacher repair map needed |
| VGG-13 -> VGG-8 | CIFAR-100; [CRD][crd], [DKD T6][dkd] | VGG models and teacher repair map needed |
| ResNet-32x4 -> ShuffleNet-V1/V2 | CIFAR-100; [CRD][crd], [DKD T7][dkd] | Widened teacher, mobile students, and teacher repair map needed |
| ResNet-50 -> MobileNet-V2 | CIFAR-100; [CRD][crd], [DKD T7][dkd] | Teacher available; student needed |
| ResNet-50 -> MobileNet-V1 | ImageNet; [DKD T9][dkd] | Teacher available; student needed |

[crd]: https://github.com/HobbitLong/RepDistiller#benchmark-results-on-cifar-100
[dkd]: https://arxiv.org/html/2203.08679

T6-T9 denote DKD table numbers. Research records:
[CRD, Tian/Krishnan/Isola, ICLR 2020](https://consensus.app/papers/contrastive-representation-distillation-tian-krishnan/188e1661a13253c9844c4ab6cb7725d8/?utm_source=chatgpt)
(2019 preprint, DOI `10.48550/arxiv.1910.10699`) and
[DKD, Zhao et al., CVPR 2022](https://consensus.app/papers/decoupled-knowledge-distillation-zhao-cui/17a111d2eec1508687419f4b97b82b28/?utm_source=chatgpt)
(DOI `10.1109/CVPR52688.2022.01165`).

`x4` denotes width: local `resnet8` is not ResNet-8x4. MobileNet-V1 and V2
are distinct. New teachers require verified repair masks/feature maps, not merely
a working KD factory. Small-stem, `32 x 32`, profile-budget adaptations are not
reproductions of published ImageNet/CIFAR scores. The full-benchmark pairings are those
declared in Section 3.1; literature examples do not expand the study automatically.
The three implemented matrix presets cover the original 13 datasets/four profiles;
additional unsupported pairings remain future work.

## 4. Downstream knowledge distillation

This is the implemented shared classical KD definition for all 16 datasets;
checkpoint and recipe selection remain family-specific.

| Parameter | Setting |
|---|---|
| Method / teacher | Classical logits KD / frozen selected original or repaired teacher |
| Temperature / correction | `T=4` / `T^2=16` |
| Response / hard-label CE coefficients | `weight=0.5` / `0.5` |
| Response warmup | Five epochs; multipliers `0.2, 0.4, 0.6, 0.8, 1.0`, then `1.0` |
| KL direction / reduction | Teacher to student / `batchmean` |
| Teacher execution | Evaluation mode; frozen parameters/BatchNorm statistics; no gradients |
| Input views | Teacher and student receive the same base-augmented tensor |
| Feature KD | Disabled, `feature_weight=0` |
| Other student interventions | No label smoothing, EMA, SAM, batch mixing, calibration loss, CPC, adaptive focal loss, or sample deletion |
| Temperature/weight search | None |

For one-based epoch `e`:

```text
r_e = min(e / 5, 1)
p_t = softmax(teacher_logits / 4)
p_s = softmax(student_logits / 4)
L_student = 0.5 * CE(student_logits, labels)
          + 0.5 * r_e * 16 * KL(p_t || p_s)
```

KL sums classes and averages examples. CE stays at coefficient `0.5` during
warmup. Teacher interventions are not inherited by students. The full benchmark's
supervised reference uses the same student setup but full-strength CE and no teacher.

Source: [distillation losses](../kd/losses.py).

## 5. Teacher repair

Every dataset compares its validation-selected CE teacher with its repaired form
under matched downstream KD. Repair losses/channel masks share the method
reference in Appendix A.4, but targeting and selection follow the dataset family.

### 5.1. Implemented class-family repair

For the original 13 datasets, retain Appendix A.4's hard-example/class policy,
same-class feature companions, replay/loss/masks, six-candidate grid, and strict
validation target-recall improvement with the overall accuracy guardrail.
For the primary adapted 32-pixel protocol, repair batch size is `64`, localization
batch size `16`, and ablation-screening batch size `128`, matching the compact
repair schedule. With `640` samples per stream and `20` epochs this gives `200`
optimizer updates per candidate. These are explicit defaults, not a measured
GPU-memory guarantee.
Balanced profiles target all classes; long-tail profiles target the declared
lowest-frequency classes. This is class-based repair, not a nuisance-group
robustness result.

### 5.2. Implemented shortcut-family repair

The following adaptations apply to Colored MNIST, Waterbirds, and UrbanCars.
The following are frozen generator defaults; standalone changes require a new
protocol. Sources: [group repair](../kd/shortcut_repair.py) and
[group-aware study](../kd/shortcut_baselines.py).

| Component | Implemented shortcut-family policy |
|---|---|
| Targets | Groups `1,2` for Colored MNIST/Waterbirds; `1..6` for UrbanCars (binary label is the most significant group bit). Within each target group, include all errors and fill with highest-NLL correct examples to at least `ceil(0.2*n_group)` |
| Companions | Request five nuisance-discordant same-label views: known same-foreground variants first, otherwise correctly classified feature-nearest matches, with declared fallback. Colored MNIST has one distinct red/green companion; repeated padding is not five independent variants. Record distinct coverage and source-match limitations |
| Preservation pool | Up to `64` correct nontarget examples per required group; fall back to other nontarget examples, then target overlap if necessary. Record fallback/overlap explicitly; do not claim disjoint preservation when a group has no suitable examples |
| Localization | Late `stage2/stage3` (ConvNet) or `layer3/layer4` (ResNet) channels; differential activation times absolute error-margin gradient; equal-group weights; `20` bootstraps, top fraction `0.25`, stability threshold `0.7`, diagnostic batch `2` |
| Screening | Ablate at most `32` stability-eligible channels; require improved target-group mean margin and preservation of raw/per-group accuracy within `0.005`. Record screened, rejected and unscreened counts |
| Mutable/frozen state | Selected connected channel coordinates only; unselected coordinates and all buffers unchanged. The continuation control updates all parameters but still freezes buffers |
| Replay / loss | Group-balanced target and preservation streams, `640` samples each per repair epoch. Target CE + original-teacher preservation KD (`T=4`, weight `1`) + normalized feature preservation (`0.25`); no decay |
| Candidate grid | Channel budgets `2/4/8` x LRs `0.001/0.005`: at most six independent anchor-reset candidates, `20` epochs each. Batch `64` for Colored MNIST (`200` updates/candidate), `16` for Waterbirds/UrbanCars (`800` updates/candidate) |
| Preservation / acceptance | Strict validation-WGA improvement over CE anchor; overall accuracy and every required group's accuracy may fall by at most `0.005` (0.5 pp) |
| Candidate tie order | Higher WGA, higher raw accuracy, lower NLL, smaller channel budget, lower LR; deterministic first exact tie |
| Attribution/mechanism controls | Section 6.2; same anchor, annotation/pool access, six-candidate/update budget and validation guardrails. No feasible/accepted candidate produces a verified control alias |

Teacher feature proximity does not prove that two images share foreground
content. Recoloring the same digit is a valid constructed nuisance intervention;
same-foreground background/co-object swaps require source identities, masks, and
a validated renderer. Generic group comparisons are not paired counterfactuals.
Channel ablation intervenes inside a model; it does not alone identify a
real-world cause or prove that a channel represents a nuisance.

### 5.3. Repair-epoch rationale

Repair-budget rationale: Consensus MCP identified
[AI-Lancet (Zhao et al., CCS 2021)](https://consensus.app/papers/ailancet-locating-errorinducing-neurons-to-optimize-zhao-zhu/e7d83f3ae92a5cc2b147eb52065aa78f/?utm_source=chatgpt).
Its primary paper, [Section 6.3](https://doi.org/10.1145/3460120.3484818), reports
**20 epochs** for neuron fine-tuning on VGG11, VGG16, ResNet18, and GoogleNet.
This supports retaining 20 as a reference budget, not asserting it is best for
this project's different datasets, architectures, or preservation losses.

The compact class-family replay converts 20 epochs into 200 updates (Appendix A.4);
the primary adapted class-family repair also uses 200 updates (Section 5.1).
Section 5.2 gives the shortcut family's explicit replay/update counts; 20 repair
epochs are not 20 full passes over Waterbirds or UrbanCars.
A prespecified `5/10/20/40` sensitivity study remains optional future work:
independent fits/schedules, validation-only selection, and explicit search cost.
Test outcomes must not determine the repair budget.

### 5.4. Failed-step diagnostics

Both repair families record numerical/runtime fitting failures under each
candidate's `failures/` directory. A unique attempt contains `failure.json`
(phase, one-based epoch/batch/global-step, exception, component losses and tensor
diagnostics) and a SHA-bound `state.pt` (pre-step and observed model/optimizer/
scheduler/scaler state, frozen anchor, gradients, exact replay batches/indices,
RNG and sampler state). These are diagnostic-only failed artifacts, never
completed candidates, selected inference exports, or automatic resume inputs.

Recording adds CPU-memory copies of pre-step state and per-step finite checks;
it does not change losses, LR grids, momentum, masks or the infinite clipping
limit, and normal AMP overflow-skip behavior is retained. Successful CPU fixture
runs preserve exact weights, history, sample order and RNG. GPU bitwise replay
is not guaranteed. The original exception is re-raised even if saving fails.
See [snapshot format, limitations and inspection](repair-failure-snapshots.md).
New source-bound protocols include the recorder; historical plans are not
silently upgraded, and the deployed baseline-only run is not affected.

## 6. Teacher-training comparison arms

### 6.1. Shared recipe definitions

The full experiment uses this common candidate recipe set. Eligibility and
augmentation policies are recorded in the frozen plan; shortcut Balanced Softmax
eligibility is resolved from actual training counts before any fitting.

| Condition | Intervention beyond the base policy |
|---|---|
| `control` | Hard-label CE |
| `randaugment` | `N=2, M=9`, before tensor conversion; mean-RGB fill; freeze each dataset family's operation policy |
| `mixup` | `alpha=0.2`; one Beta mixing coefficient per batch; mix images and labels |
| `cutmix` | `alpha=1`; patch replacement, labels weighted by actual clipped area |
| `label_smoothing` | `epsilon=0.1`; target `0.9*one_hot + 0.1/C` |
| `balanced_softmax` | Train CE of `logits + log(training_counts)`; inference/KD use raw logits |
| `mse_logits` | MSE of raw logits against one-hot labels |
| `mse_probabilities` | MSE of softmax probabilities against one-hot labels |
| `sam` | `rho=0.05`; two passes on the same batch; BatchNorm buffers advance on first pass only |
| `ema` | Decay `0.999`; average parameters, copy current buffers; validate/export EMA weights |
| `randaugment_cutmix_ema` | The three interventions above; CE without label smoothing |
| `teacher_repair` | Reuse CE control pretraining, then Section 5 |

Mixup/CutMix apply to every eligible batch. MSE uses scale `1` and averages
examples **and classes**; probability MSE is summed-class Brier divided by `C`.
Balanced Softmax addresses class counts, not shortcut-group imbalance. Existing
class-family eligibility follows Appendix A.5; any retained CE-equivalent row is
not independent method evidence. For the three additions, omit it with a recorded
reason if exact equal class counts make it CE-equivalent. Freeze eligibility and
tie order before training; class-family eligibility changes require a fresh plan
and corresponding implementation rather than silently changing the generator.

The `teacher_repair` recipe follows Section 5: Appendix A.4's class
repair for the original 13, and Section 5.2's group repair for the other three.
Do not silently reuse the retained catalog's RandAugment operation policies for
the three new datasets
(Appendix A.5). FP16 SAM, including auto-resolved FP16, is rejected by the
implemented recipe; float32/BF16 are supported.

These are controlled baselines, not tuned SOTA reproductions. SRA, ACL, and
Bayesian teachers remain researched, unimplemented follow-ups. Sources:
[teacher recipes](../kd/teacher_training.py), [study runner](../kd/teacher_baselines.py),
[repair adapter](../kd/teacher_repair_baseline.py), and
[research background](teacher-training-baselines.md).

### 6.2. Shortcut repair-mechanism controls

| Arm | Purpose |
|---|---|
| All-parameter continuation (`repair_ce_continuation`) | Extra fitting without channel selection, using the same target CE and preservation losses; all parameters mutable, buffers frozen. Three seed slots `2026/2027/2028` x two LRs give six trials; channel count is not applicable |
| Random-channel repair | Compare equal channel-count/update budgets without attribution-based localization |
| Attribution-only repair | Test localization without causal-ablation screening |
| Full causal-screened repair | Complete method; the same fitted condition as Section 6.1's repaired teacher |

These three additional controls are implemented for the shortcut family.
Give them the same CE anchor, group-label access, fitting pools, companion/matching
access, validation objective, and declared candidate/update budgets. If extended
to class-family settings, declare that additional scope explicitly. Report
localization/search costs
separately: equal updates are not equal FLOPs or wall-clock time.
They form a mechanism comparison, not automatic additional recipe-winner
candidates: only Section 6.1's eligible recipes compete for the winner. Count
full repair once. Random-channel and continuation controls do not perform
attribution/screening; attribution-only omits screening. The continuation name
does not mean a pure-CE-only loss or full-dataset fine-tuning.

## 7. Selection, metrics, and fairness

### 7.1. Validation-only decisions

| Decision | Original 13 datasets: class rules | Three shortcut datasets: group rules |
|---|---|---|
| Ordinary teacher / individual student checkpoint | Highest validation accuracy; EMA teachers evaluate EMA weights | Highest validation WGA, then raw accuracy, then earliest exact tie; EMA evaluates EMA weights |
| Repair candidate | Strict validation target macro-recall improvement and overall-accuracy guardrail; ordering in Appendix A.4 | Strict validation-WGA improvement over CE anchor, subject to frozen overall/per-group limits and tie rules |
| Recipe winner | Highest mean KD-student validation accuracy for `balanced`, class-balanced accuracy for long-tail; recipe-order tie break | Highest mean KD-student validation WGA; frozen recipe-order tie break |
| Rejected / infeasible repair | Verified control-equivalent aliases for new standardized runs; historical stopped cases remain separate | Keep reasons; control alias only after explicit checkpoint/evaluation verification |

Use paired students `42/43/44` for recipe selection. Freeze all checkpoints,
repair decisions, and the winner before evaluating every prespecified eligible
test arm. Both Colored MNIST renderings use the same frozen checkpoints and
winner, without test-specific reselection.

Known nuisance annotations in the shortcut family support diagnostics, targeting,
selection, and evaluation, not model inputs: those conditions are group-informed,
not annotation-free. The two families intentionally use different decision rules.

### 7.2. Primary score and supporting metrics

Define the primary score `M` **per setting**, not as one interchangeable score:

| Setting | Primary `M` | Required supporting measurements |
|---|---|---|
| Original dataset / `balanced` | Raw accuracy | Class-balanced accuracy, target macro recall, calibration |
| Original dataset / `lt-if10/50/100` | Class-balanced accuracy (BAcc) | Raw accuracy, target macro recall, calibration |
| Colored MNIST / Waterbirds / UrbanCars | Worst-group accuracy (WGA): `min_g(correct_g / n_g)` | Raw/group-balanced accuracy, each group's count/accuracy, nuisance diagnostics |

Group-balanced accuracy is the unweighted mean of group accuracies; it is not
class-balanced accuracy. An absent required group makes full-group WGA
unavailable, never silently omitted or assigned zero. Use Section 1.3's observed
label policy for Colored MNIST. Literature-weighted accuracy is distinct from
raw accuracy. UrbanCars additionally reports the author's ID-weighted score:
equal-class means in four background/co-object agreement slices, weighted by the
fixed independent `0.95/0.95` mixture. Each shortcut gap is its shifted slice
score **minus** that ID score (negative usually denotes degradation). Actual
training-group-weighted accuracy is a separately named statistic, not substituted
for the paper ID score. Source: [author evaluation](https://github.com/facebookresearch/Whac-A-Mole/blob/main/urbancars_trainers/base_trainer.py)
and [implemented definitions](../kd/urban_metrics.py).

Report NLL, summed-class Brier, and 15-bin ECE separately from recipe selection
unless prespecified. Valid paired nuisance checks report prediction flips and
accuracy before/after nuisance changes; invariance alone can reward a useless
constant classifier. Class-based results do not automatically establish shortcut
robustness, and these constructed benchmarks do not establish general causal
identification or autonomous-driving safety.

Teacher scores describe seed `41`. Student scores and same-seed deltas against
control KD are `mean ± sample SD` over `42/43/44`, conditional on that teacher.
Form each seed's delta first, then summarize; do not subtract SDs. These are not
confidence intervals or teacher-seed uncertainty. Stronger inferential claims
need additional teacher seeds and a declared analysis. Do not average Acc, BAcc,
and WGA into a single 16-dataset score.

### 7.3. Fair comparison and cost limits

Match splits, profile/environment, model pair, initialization within each
role/seed, base views, student budget, KD settings, and annotation access within
each setting. Teacher interventions intentionally differ. Shared settings estimate
transfer under the declared protocol, not every method's optimally tuned
performance; loss/logit scales can respond differently to fixed optimization/KD.
The class family's different checkpoint/repair objectives remain an explicit
fairness limitation, not a claim of metric-matched causal controls.

Loss-only recipes are near CE compute; EMA, mixing, and augmentation add
different overheads. SAM needs two gradient passes per batch. Repair adds
localization/screening and candidate fitting; cost depends on pools and search.
A strict cost order requires matched hardware/input measurements, not recipe
names or equal epochs.

Use a common declared evaluation device/precision and timing scope within each
comparison, separating pretraining, localization, search, KD, and inference.
Do not mix historical CPU-float32 scores or incompatible timing fields into a
matched new comparison (Appendix A.5). Seeds alone do not guarantee bitwise GPU
reproducibility; freeze gates do not undo prior test inspection.

## 8. Full experiment size and coverage

Dataset counts, training settings, and evaluation endpoints are different units.
Under Section 3.1's pair choices, the implemented planner scope is:

| Family | Datasets | Training settings: primary pair | Training settings: primary + compact ConvNet | Final endpoint panels: one / two pairs |
|---|---:|---:|---:|---|
| Class / long-tail: 13 x 4 profiles | 13 | 52 | 104 | 52 / 104 |
| Shortcut: one native setting per dataset | 3 | 3 | 3 | 4 / 4 |
| Full experiment | 16 | 55 | 107 | 56 / 108 |

These are **setting counts**, not completed experiments or training-fit totals.
Pair expansion applies only to the original 13; the
three shortcut datasets retain their declared single pair. Colored MNIST has
one training setting and two correlated test panels.

Do not multiply all 16 by four profiles. Maximum fresh-fit counts (one teacher
seed and three paired students) are `3,153` for the default full plan and `6,057`
for the primary plus compact ConvNet pair, including the three mechanism controls on each
shortcut dataset. The shortcut family adds at most `249` fits: `33` ordinary
teachers + `72` repair candidates + `135` KD students + `9` supervised students.
Conditional Balanced Softmax omissions, infeasible candidates and verified
control aliases reduce actual fits. Appendix A.6's `2,904/5,808` maxima remain
correct for the original 13-dataset subset only.
Selecting all three class-family presets produces `159` settings, `160` endpoint
panels and at most `8,961` fits. Different resolutions/initializations remain
separate panels; equal fit counts do not imply equal computational cost.

Report completed / eligible recipes separately from mechanism controls and
supervised references. Count shared fits once and check endpoint completeness
individually. Keep accepted, no-change, infeasible, failed, missing, and pending
outcomes with reasons/denominators; missing measurements are not zero gains.

Repair acceptance and downstream gain are separate outcomes. The historical
target-recall transfer-success criterion cannot be reused as a WGA criterion
without prespecification. Historical counts, follow-ups, and pilots remain in
Appendix A.6; their checkpoints/protocols/results are not rewritten or mixed
into new 100/20-epoch comparisons.

## 9. Execution readiness and pre-run checks

The [planner](../kd/teacher_baseline_matrix.py) now defaults to all 16 datasets:
55 settings, or 107 with the primary plus compact ConvNet pair. The
[full benchmark driver](../kd/full_benchmark.py) provides local-data preflight,
execution/resume, gated test evaluation and a source-bound overview. It stops
on an execution error; it does not silently skip a missing dataset. Existing
historical plans remain immutable: create a fresh plan after these code changes.

### 9.1. Provision data and inspect the plan

Provision the original 13 datasets through the existing catalog commands. The
three new adapters accept these local layouts under `--root data`:

| Source | Expected layout / acquisition |
|---|---|
| `colored_mnist` | `data/colored_mnist/1.0/MNIST/...`; explicitly acquire torchvision MNIST with the command below, then hash the rendered data at training |
| `waterbirds` | `data/waterbirds/1.0/metadata.csv` and relative image paths; required columns `img_filename,y,place,split`, split IDs `0/1/2`. Manually provision the licensed release |
| `urbancars` | `data/urbancars/1.0/bg-0.95_co_occur_obj-0.95/train/...` and `bg-0.5_co_occur_obj-0.5/{val,test}/...`; eight `obj-{urban,country}_bg-{urban,country}_co_occur_obj-{urban,country}` folders per split with native 256-pixel JPGs. Masks are not predictor inputs |

Waterbirds/UrbanCars have no automatic downloader. Unversioned source directories
and direct released roots are also accepted; provenance records the loaded
content. Review acquisition licenses and retain release/archive digests externally.

```bash
# Explicit optional MNIST acquisition; no teacher/student fitting.
python -m kd.shortcut_data colored_mnist --root data --download-mnist

# Inspect the complete scope without writes, downloads or fitting.
python -m kd.teacher_baseline_matrix plan \
  --root data --output runs/teacher-baselines-v7/matrix-plan \
  --epochs 100 --dry-run

# Materialize a fresh source-bound plan. Choose runtime here, not via later overrides.
python -m kd.teacher_baseline_matrix plan \
  --root data --output runs/teacher-baselines-v7/matrix-plan \
  --epochs 100 --device cuda --precision float32 --workers 4
```

Add `--pairs resnet34_resnet18_smallstem_imagenet convnet6` when creating the plan for 107 settings;
the option expands only the original 13 datasets. `--datasets` can explicitly
restrict a diagnostic/subset plan, but such a plan is not the full experiment.
Plan, dataset cache and study output roots must be disjoint.

### 9.2. Preflight, train, evaluate and report

```bash
python -m kd.full_benchmark preflight --plan runs/teacher-baselines-v7/matrix-plan
python -m kd.full_benchmark train --plan runs/teacher-baselines-v7/matrix-plan
python -m kd.full_benchmark evaluate --plan runs/teacher-baselines-v7/matrix-plan
python -m kd.full_benchmark report --plan runs/teacher-baselines-v7/matrix-plan \
  --output runs/teacher-baselines-v7/overview
```

Preflight checks all local training/validation datasets without fetching weights
or test images. Existing class adapters may create/verify deterministic split
caches. It is a data/shape check, not a GPU memory or convergence benchmark.
Training first repeats preflight, then retrieves named ImageNet weights if not
already cached. Use `--jobs JOB_ID ...` to execute selected frozen queue rows;
training still requires the full plan's data preflight to pass. Rerunning `train`
resumes or verifies completed work using checkpoint/data/initialization hashes.

Before full budgets, run a separate short diagnostic plan on the target GPU with
the actual releases and pretrained weights. Offline fixtures test software
contracts, not real-image performance, GPU capacity or ImageNet-weight efficacy.
Group metadata/preservation, missing-group metrics, paired color endpoints,
high-resolution ordinary training/KD, interrupted repair recovery and report
tampering have automated checks. Full 100/20-epoch experiments remain unrun.

Evaluation requires sealed validation selections and checkpoint identities.
Nested repair/localization/candidate manifests protect reuse; final evaluation
has its own manifest. Overview reporting checks the exact frozen matrix job and
selected checkpoints, writes to a separate directory and retains pending,
invalid or unavailable endpoints with reasons and empty measurements, never
zero gains. Repair accept/no-change details remain in each study's report.
Preflight checks each distinct dataset/input-protocol/resolution contract; it
does not audit ImageNet duplicates. Pretraining-overlap warnings persist in the
JSON and Markdown overview, including when results are still pending.

## 10. Final-results tables

The tables cover the entire 16-dataset scope with family-specific `M` from
Section 7.2. Every numeric cell is a template: `—` means unmeasured; `N/A`
means not applicable. Per-study `comparison.json` provides the detailed metrics;
`kd.full_benchmark report` writes `overview.json` and `overview.md` for all
frozen jobs/endpoints. The tables below are publication layouts, not exact
renderer output or measured results.

### 10.1. Teacher quality and downstream transfer

Use one panel per **dataset / profile or environment / architecture pair /
initialization policy**, naming `M` in the caption. Scores are percentages and
deltas percentage points. Report every eligible ordinary recipe; include
mechanism controls only in their prespecified scope.

| Condition | Teacher raw Acc (%) | Teacher M (%) | Student raw Acc (%) | Student M (%) | Paired student Δ M vs control KD (pp) | Status / scope |
|---|---:|---:|---:|---:|---:|---|
| Supervised student, no teacher | N/A | N/A | — | — | N/A | Not measured |
| CE control -> KD | — | — | — | — | Reference | Not measured |
| RandAugment -> KD | — | — | — | — | — | Not measured |
| Mixup -> KD | — | — | — | — | — | Not measured |
| CutMix -> KD | — | — | — | — | — | Not measured |
| Label smoothing -> KD | — | — | — | — | — | Not measured |
| Balanced Softmax -> KD, if eligible | — | — | — | — | — | Eligibility per setting |
| MSE logits -> KD | — | — | — | — | — | Not measured |
| MSE probabilities -> KD | — | — | — | — | — | Not measured |
| SAM -> KD | — | — | — | — | — | Not measured |
| EMA -> KD | — | — | — | — | — | Not measured |
| RandAugment + CutMix + EMA -> KD | — | — | — | — | — | Not measured |
| Repaired CE teacher -> KD | — | — | — | — | — | Implemented in both families; not measured |
| Mechanism control: all-parameter continuation -> KD | — | — | — | — | — | Shortcut scope; not measured |
| Mechanism control: random-channel repair -> KD | — | — | — | — | — | Shortcut scope; not measured |
| Mechanism control: attribution-only repair -> KD | — | — | — | — | — | Shortcut scope; not measured |

Use Section 7.2's seed summaries and same-seed deltas. Verified no-change aliases
are reused control evidence, not independent fits. Supplements retain class
target recall or per-group outcomes as appropriate, calibration, nuisance checks,
repair/search budgets, guardrails, and equal-scope cost accounting. Appendix A.8
details the implemented class-family fields.

### 10.2. Full 16-dataset overview

This example includes all 16 datasets: the 13 class datasets at `lt-if100`
with the primary pretrained ResNet-34 -> ResNet-18 pair at 32 pixels, plus the three native shortcut settings with
Section 3.1's assigned pairs. Colored MNIST contributes two endpoint rows.
For other class profiles/pairs, repeat their class rows with the correct `M`;
do not retrain/count the shortcut settings once per class profile.

| Dataset / profile or test endpoint | Primary M | Control KD M (%) | Repaired-teacher KD M (%) | Validation-selected recipe | Selected KD M (%) | Paired Δ repair vs control (pp) | Paired Δ selected vs control (pp) | Coverage / status |
|---|---|---:|---:|---|---:|---:|---:|---|
| CIFAR-10 / `lt-if100` | BAcc | — | — | — | — | — | — | Full-run results pending |
| CIFAR-100 / `lt-if100` | BAcc | — | — | — | — | — | — | Full-run results pending |
| SVHN / `lt-if100` | BAcc | — | — | — | — | — | — | Full-run results pending |
| CINIC-10 / `lt-if100` | BAcc | — | — | — | — | — | — | Not run; pretraining overlap unaudited |
| GTSRB / `lt-if100` | BAcc | — | — | — | — | — | — | Full-run results pending |
| Fashion-MNIST / `lt-if100` | BAcc | — | — | — | — | — | — | Full-run results pending |
| PathMNIST / `lt-if100` | BAcc | — | — | — | — | — | — | Full-run results pending |
| BloodMNIST / `lt-if100` | BAcc | — | — | — | — | — | — | Full-run results pending |
| DermaMNIST / `lt-if100` | BAcc | — | — | — | — | — | — | Full-run results pending |
| OrganAMNIST / `lt-if100` | BAcc | — | — | — | — | — | — | Full-run results pending |
| Caltech-101 / `lt-if100` | BAcc | — | — | — | — | — | — | Full-run results pending |
| EuroSAT / `lt-if100` | BAcc | — | — | — | — | — | — | Full-run results pending |
| STL-10 / `lt-if100` | BAcc | — | — | — | — | — | — | Not run; pretraining overlap unaudited |
| Colored MNIST / reversed color (`0.9` mismatch) | WGA | — | — | — | — | — | — | Not run |
| Colored MNIST / unbiased color (`0.5` mismatch) | WGA | — | — | — | — | — | — | Not run; shared test digits |
| Waterbirds / released test | WGA | — | — | — | — | — | — | Not run; provision local release |
| UrbanCars / released/author-defined test | WGA | — | — | — | — | — | — | Not run; provision local release |

`M=BAcc` for the shown class rows; it becomes raw accuracy for `balanced`.
The shortcut rows retain `M=WGA`. "Selected" always means validation-selected
under Section 7.1. Both Colored MNIST rows use the same winning recipe.
Use same-seed deltas and Section 8's coverage accounting. Do not combine different
primary metrics, resolutions, profiles, or architecture policies into a
single unqualified average; Appendix A.9 defines class-family summaries.

## Appendix A. Retained 13-dataset reference and historical evidence

The original 13 datasets remain an active part of the full 16-dataset scope.
This appendix centralizes the optional compact presets and historical evidence.
It is not the primary pretrained protocol; Sections 1-9 define that protocol.
Shared losses/recipes are in Sections 4 and 6. Historical frozen settings remain
authoritative and distinct from new standardized runs.

### A.1. Dataset reference

The full roster, source IDs/releases, class counts, dimensions, and split rules
are maintained once in Sections 1.1-1.2. These compact presets use RGB `32 x 32`,
split seed `2026`, and profile sampling only on train/validation.
Their horizontal-flip policy is in Section 2.1; exact views/normalization follow.
Source: [versioned dataset catalog](../kd/catalog/v1.json).

### A.2. Profiles and preprocessing

| Profile | Sampling factor | Saved main teacher/student epochs | Standardized retained matrix teacher/student epochs | Repair target classes |
|---|---:|---:|---:|---|
| `balanced` | 1 | 81 | 100 | All classes |
| `lt-if10` | 10 | 49 | 100 | Lowest-frequency `ceil(C/2)` classes |
| `lt-if50` | 50 | 71 | 100 | Lowest-frequency `ceil(C/2)` classes |
| `lt-if100` | 100 | 81 | 100 | Lowest-frequency `ceil(C/2)` classes |

For class index `k` among `C` classes, retain
`max(1, round(n_k * factor ** (-k / (C - 1))))` examples, independently in
training and validation, in dataset label order. Target classes are resolved
from actual training counts with class-index tie breaking. `balanced` means no
additional resampling; naturally unequal classes remain unequal. The requested
factor need not equal the realized maximum/minimum count ratio.

Standardized retained matrix runs use a uniform **100 epochs** for every teacher,
KD student, and supervised student across datasets/profiles/pairs. This is a standardized project
budget, not a universally optimal literature setting. `--epochs N` overrides it
for the entire generated matrix and must be declared before training. Historical
profile budgets and static pilots retain their frozen settings. Equal epochs do
not equalize samples, optimizer updates, or compute (Appendix A.5).

#### Base augmentation

| Dataset group | Training, `augmentation="basic"` | Validation/test/diagnostic views |
|---|---|---|
| CIFAR-10, CIFAR-100, SVHN, CINIC-10 | Random crop `32 x 32`, padding `4`; flip only where enabled in Section 2.1 | Tensor conversion and normalization |
| Fashion-MNIST and the four MedMNIST sources | Resize to `32 x 32`, then random crop with padding `4`; no flip | Resize to `32 x 32`, tensor conversion, normalization |
| GTSRB, Caltech-101, EuroSAT, STL-10 | Random resized crop `32 x 32`, scale `(0.85, 1.0)`, aspect ratio `(0.9, 1.1)`; no flip | Resize shorter edge to `36`, center crop `32 x 32`, RGB conversion, tensor conversion, normalization |

Fashion-MNIST and MedMNIST adapters convert images to RGB. Extra teacher-training
interventions are specified in Section 6; they do not change these student views.

#### Normalization

| Dataset group | Mean | Standard deviation |
|---|---|---|
| CIFAR-10, SVHN, PathMNIST, BloodMNIST, DermaMNIST, OrganAMNIST | `(0.5, 0.5, 0.5)` | `(0.5, 0.5, 0.5)` |
| CIFAR-100 | `(0.5071, 0.4867, 0.4408)` | `(0.2675, 0.2565, 0.2761)` |
| CINIC-10 | `(0.47889522, 0.47227842, 0.43047404)` | `(0.24205776, 0.23828046, 0.25874835)` |
| Fashion-MNIST | `(0.2860, 0.2860, 0.2860)` | `(0.3530, 0.3530, 0.3530)` |
| GTSRB, Caltech-101, EuroSAT, STL-10 | `(0.485, 0.456, 0.406)` | `(0.229, 0.224, 0.225)` |

Sources: [data adapters](../kd/data.py), [profile definitions](../kd/neuron_surgery_matrix.py).

### A.3. Models and shared training

#### Architecture identity

The reference pair is **ConvNet-6-W32 -> ConvNet-6-W16**, selected with neutral
IDs `convnet6_w32` and `convnet6_w16`. Legacy `cifar_teacher`/`cifar_student`
IDs construct identical architectures; they do not restrict dataset choice.
Models are instantiated and trained separately for each dataset/profile, with
`C` output classes. Checkpoints remain bound to their declared model ID, class
ordering, and preprocessing; aliases do not silently rename old artifacts.

The optional `resnet34_resnet18` pairing uses `resnet34_smallstem` and
`resnet18_smallstem`: a `3 x 3`, stride-1 stem without initial max pooling.
`resnet50_smallstem` is also available. All legacy small-stem IDs remain supported.

#### Reference architecture and common optimization

Architecture-specific rows below apply only to ConvNet-6. Optimization defaults
are shared within a paired setting; different architectures do not have identical
initial tensors or parameter counts.

| Parameter | Teacher | Student |
|---|---|---|
| Architecture | Six convolutions; widths `32 -> 64 -> 128` | Six convolutions; widths `16 -> 32 -> 64` |
| Stage | Two bias-free `3 x 3` convolutions, each with BatchNorm/ReLU; `2 x 2` max pooling | Same structure |
| Head | Adaptive average pooling to `2 x 2`, flatten, linear classifier with `C` outputs | Same structure |
| Trainable parameters, excluding buffers | `287456 + 513*C` | `72304 + 257*C` |
| Initialization / seeds | From scratch; `41` | From scratch; `42, 43, 44` |
| Epochs | Standardized retained matrix: `100`; historical budgets in Appendix A.2 | Same |
| Batch size | `128` | `128` |
| Optimizer | SGD, LR `0.05`, momentum `0.9`, weight decay `0.0005`; no Nesterov | Same |
| Scheduler | Cosine annealing; `T_max=declared training epochs`, minimum LR `0`; one step per epoch | Same |
| LR warmup / early stopping | None / none | Same; KD-loss warmup is separate |
| Objective | Hard-label CE, or Section 6's teacher recipe | Section 4 |
| CPU threads / persistent workers / prefetch | `4` / disabled / `2` | Same |
| CUDA mode / memory format | `fast` / channels-last | Same |
| Engine inference benchmark configuration | `20` warmup, `100` measured iterations | Same |

| Runtime defaults | Saved main matrix | Standardized retained baseline matrix |
|---|---|---|
| Training device / precision | Assigned CUDA device / `auto` in available saved GPU protocols | `auto` device / `float32` |
| Data-loader workers | `4` in available saved CUDA protocols | `0` |
| Overrides | Per-run frozen config is authoritative | Generator exposes epochs, device, precision, workers, and seeds; bound configs require a fresh plan for changes |

CUDA `auto` chooses BF16 where supported, scaled FP16 otherwise. Fast mode
permits TF32 and nondeterministic cuDNN choices; seeds alone do not guarantee
bitwise GPU reproducibility. Resolved runtime belongs in each run's metadata.
Evaluation runtimes and timing scopes are specified in Appendix A.5.

Sources: [models](../kd/models.py), [baseline construction](../kd/neuron_surgery_baselines.py),
[training engine](../kd/engine.py), [saved baseline example](../runs/experiment1-multidataset/cifar10/lt-if10/baselines/confirmatory/protocol.json).

### A.4. Class-based teacher repair

| Parameter | Setting |
|---|---|
| Anchor / target classes | Selected CE control checkpoint / Appendix A.2's class policy |
| Hard targets | Canonical training diagnostics; all mistakes, then highest-NLL correct examples to a `20%` class quota; not a cap if mistakes exceed it |
| Repair-fitting views | Dataset's ordinary training augmentation, for both replay streams |
| Companions | `5` requested; cosine matching of normalized pooled final-stage features: ConvNet-6 `stage3`, ResNet `stage4` |
| Sparse fallback | Enabled in saved/new full matrices; correct non-target same-class examples first, then same-class replacement; singleton classes use gradient-only attribution |
| Localization stages | ConvNet-6 `stage2/stage3`; ResNet `stage3/stage4` |
| Score / bootstrap | Differential activation discrepancy times absolute error-margin gradient; 20 within-class bootstraps, equal class weighting, seed `2026` |
| Screening | Top fraction `0.25`, stability at least `0.7`; zero-channel ablation must improve target margin with preservation accuracy loss at most `0.005` |
| Grid / initialization | Budgets `[2,4,8]` x LRs `[0.001,0.005]`; at most six feasible candidates, each reset to the anchor |
| Repair epochs / seed | `20` / `2026` |
| Replay / batch | Class-balanced sampling with replacement; `640` samples per stream per epoch; `64` target + `64` preservation per update; `10` updates/epoch |
| Optimizer / scheduler | SGD, momentum `0.9`, **weight decay `0`**; cosine, `T_max=20`, minimum LR `0` |
| Precision | Diagnostics/localization: float32; candidate fitting/selection validation: teacher precision |
| Target CE / preservation KD | Coefficients `1` / `1`; original teacher anchor, `T=4` with `T^2` |
| Feature preservation / preservation CE | Coefficients `0.25` / `0`; use configured localization stages |
| Mutable state | Selected producing convolution rows, matching BatchNorm affine coordinates, and directly connected consumer weights |
| Frozen state | All other parameter coordinates and all buffers; candidate stays in evaluation mode during repair fitting |
| Candidate checkpoint / eligibility | End of fit / strictly improve validation target macro recall and keep overall accuracy at least `baseline - 0.005` |
| Selection order | **Highest eligible target macro recall**, then lower target-balanced NLL, fewer channels, lower LR |

```text
L_repair = CE(candidate_target_logits, target_labels)
         + 16 * KL(softmax(anchor_preservation_logits / 4)
                   || softmax(candidate_preservation_logits / 4))
         + 0.25 * feature_preservation_loss
```

Feature preservation L2-normalizes along channels, sums squared discrepancies
over channels, averages examples/spatial locations, then averages configured
stages. There is no permanent pruning or sample deletion. Frozen coordinates
and buffers are checked for exact equality. Fallback companions are not guaranteed
to be five distinct correctly classified images; coverage/omissions are recorded.

In this retained implementation, 20 repair epochs mean **200 updates** with
replacement-sampled replay, not 20 full dataset passes. The 20-epoch literature
rationale and optional sensitivity-study boundary are in Section 5.

Sources: [repair losses/masks](../kd/neuron_surgery.py),
[repair selection and runtime](../kd/neuron_surgery_study.py),
[frozen protocol example](../runs/experiment1-multidataset/cifar10/lt-if10/studies/confirmatory/protocol.json).

### A.5. Recipe policies, selection, and integrity

#### Retained recipe-specific policies

Classical KD and the ordinary recipe definitions are shared with Sections 4 and 6.
For class-family runs, `teacher_repair` uses Appendix A.4, not shortcut-group repair.
The generator omits Balanced Softmax only for balanced CIFAR-10/CIFAR-100;
other exact equal-count settings can retain a CE-equivalent row, which is not
independent method evidence.

| Sources | Recorded RandAugment policy | Allowed operations |
|---|---|---|
| SVHN, GTSRB | `restricted_digit_sign_photometric` | Identity, Brightness, Contrast, Sharpness, AutoContrast, Equalize |
| Four MedMNIST sources | `restricted_medical_photometric` | Identity, Brightness, Contrast, Sharpness |
| Other catalog datasets | `torchvision_standard` | Identity, ShearX/Y, TranslateX/Y, Rotate, Brightness, Color, Contrast, Sharpness, Posterize, Solarize, AutoContrast, Equalize |

Policies are recorded in teacher/study protocols and summaries. Restricted
policies use torchvision magnitudes, not a complete standard RandAugment policy
or a guarantee of label invariance; operation-space changes fail closed.
Shortcut-family policies are recorded separately in Section 2.2 and the new plan.

#### Implemented decision rules

| Decision | Rule |
|---|---|
| Ordinary teacher / individual student checkpoint | Highest validation accuracy; EMA teachers evaluate EMA weights |
| Repaired teacher | Eligibility and ordering in Appendix A.4 |
| Standardized retained benchmark's winning recipe | Mean student validation balanced accuracy for long-tail; accuracy for balanced; ties retain recipe order |
| Main-matrix transfer success | Mean target recall gain at least 1 percentage point; at least two positive student seeds; mean accuracy loss at most 0.5 points |
| Rejected repair | Historical matrix: stop without repaired-student/test results; standardized retained benchmark: verified control-equivalent teacher/student aliases |
| Test access | Standardized retained benchmark freezes all required checkpoints and validation selections, then explicitly evaluates every prespecified recipe |
| Provenance | Standardized retained benchmark verifies manifests/checkpoint bindings and rebuilt train/validation identities before prediction |
| Interrupted-run resume | New engine/recipe-teacher checkpoints bind recorded data provenance; committed `last.pt` also records the exact selected `best.pt` SHA256. Validate both before adopting state or rewriting outputs, including explicit best-checkpoint rollback and final export |
| Matrix execution | Generated version-2 configs verify their parent plan, job/spec/config/output/queue bindings and exact current computational source coverage before study writes or fitting |
| Standalone repair inputs/sources | Adopted teacher/KD baselines reject calibration/CPC/adaptive-focal supervision; require the student's recorded training-teacher digest to match the repair anchor, plus the current source-file digest when present; new repair protocols include the full local computational dependency stack |

Report overall accuracy, balanced accuracy, target macro recall, NLL, Brier score,
and 15-bin ECE. Keep every prespecified setting/condition, including no-change,
infeasible, failed, and missing outcomes, with reasons and denominators. This
all-matrix coverage audit is a reporting requirement. The generator only creates
plans; Section 9's driver executes fresh plans and reports coverage, without
claiming a pooled cross-dataset statistical analysis. Do not reselect the winner
using test results.

Resume checks use recorded adapter identities, not an independent rescan/hash of
every image. Evaluation-only loader-size additions are ignored. Recipe-teacher
best checkpoints additionally bind the teacher protocol and initialization.
Unmanifested student recovery verifies both checkpoint data bindings and the
committed selected-best digest before sealing an export; no-change aliases rebind
only the digest of their deliberately rebased best file, after verifying the
control run. Model tensors remain identical.

Incomplete legacy checkpoints lacking data or selected-best bindings are rejected;
no adjacent `data.json`, matching model metadata, or matching export tensors are
silently adopted as proof. Completed manifest-verified artifacts and historical
reports remain usable as saved evidence, without backfilling new fields.
Checkpoint files are saved atomically individually, not as an atomic best/last
pair: interruption after replacing best but before committing last fails closed.
Checkpoint verification assumes one writer per run directory; do not train and
resume concurrently into the same output.
Restore a known matching checkpoint pair or start a fresh run; never manufacture
a new binding to an arbitrary best file. Computational-source changes require a
fresh baseline/repair plan rather than overwriting a frozen historical protocol.

#### What is matched, and what is not

| Aspect | Current comparison and permitted interpretation | Additional control needed for stronger claims |
|---|---|---|
| Within-setting matching | Same dataset/profile/pair, split membership, architecture, initialization within each role/seed, and student training; teacher augmented views intentionally differ by recipe | Keep comparisons paired; matching does not extend to different architecture tensors |
| Compute/search | Equal base epochs, but SAM doubles gradient passes and repair adds localization plus up to six candidate fits/validation comparisons | Equal-compute continuation and comparably budgeted searches within this benchmark |
| Selection objectives | Ordinary checkpoints maximize accuracy; repair candidates prioritize target recall subject to an accuracy guardrail | Metric/search-matched controls for attributing gains specifically to channel repair |
| Optimization/KD tuning | Shared LR and unit MSE scale do not equalize gradient magnitude across losses/classes; fixed KD settings can suit logit scales differently | Prespecified, comparable validation-only teacher and KD tuning budgets |
| Across profiles | New runs share 100 epochs; sample counts and optimizer updates still change. Historical runs also vary epochs | Equal-update/exposure sensitivity studies; keep historical and standardized runs separate |
| Teacher/student uncertainty | Three paired students condition on one teacher seed; reported SD is descriptive, not teacher uncertainty | Additional independent teacher seeds and declared paired/hierarchical analysis |
| Correlation/multiplicity | Profiles reuse dataset images; 52 settings are not 52 independent dataset replications | Cluster-aware analysis and a declared multiplicity policy for inferential claims |
| Selected-repair follow-ups | Accepted-only subsets are conditional, potentially selected analyses | Report full coverage; do not generalize subset averages to all 52 settings |
| Test reuse | Current freeze gates cannot undo prior inspection; reused tests remain exploratory | A genuinely independent endpoint and protocol frozen before inspecting it |
| Domain/resolution | Every input is reduced to this `32 x 32` protocol; base crops, mixing, and restricted policies remain empirical assumptions | Domain review and resolution/convergence sensitivity checks; no native-resolution SOTA claims |

The standardized retained benchmark estimates **which recipe transfers best
under the declared training and KD settings**, not each method's best achievable performance.
The rightmost column is not a list of completed controls or automatic safeguards.

#### Evaluation and cost comparability

Historical main-matrix final predictions use CPU float32; standardized retained benchmark
predictions use the run's configured/resolved device and precision. Do not mix
these results into a nominally matched comparison without a separately declared
common reevaluation. Fast-mode GPU reproducibility limits remain as in Appendix A.3.

New recipe-teacher `training_seconds` includes validation; historical engine-run
teacher and student `training.total_epoch_seconds` excludes it. The new repair
adapter's cumulative timing includes base pretraining plus the successful repair
invocation, but not earlier interrupted work; historical repair reports do not
uniformly record that cumulative scope. Engine-run summaries contain
parameter/latency information; new recipe/repaired teachers are not uniformly
latency-profiled and the aggregate benchmark does not supply a complete inference
cost table. These fields are descriptive, not equal-scope compute-normalized
evidence. Strong cost claims require matched hardware, precision, timing scope,
candidate-search accounting, and fresh comparable inference measurements.

### A.6. Size, saved evidence, and related experiments

Counts below describe the 13-dataset subset and historical summary evidence,
not the complete 16-dataset fit count.

| Experiment | Settings | Ordinary teachers | Repair candidates, max | KD students | Supervised references | Total fits, max |
|---|---:|---:|---:|---|---:|---|
| Saved main matrix | 52 | 52 planned | 312 | 156 controls + 69 implied by accepted cases | 0 | Actual total not recomputed |
| Standardized retained ConvNet-6 baseline matrix | 52 | 570 | 312 | 1,866 maximum | 156 | 2,904 |
| Standardized retained matrix, both supported pairs | 104 | 1,140 | 624 | 3,732 maximum | 312 | 5,808 |

The saved [matrix summary](../runs/experiment1-multidataset/matrix_summary.json)
records 52 finished jobs: 23 accepted repairs, 29 no-repair outcomes, and 10
successful transfers. Coverage is **23/52 accepted and 10/52 successful**;
10/23 describes the conditional accepted subset. The 29 stopped cases have no
newly measured repaired-student gains and must not be presented as observed
zero-gain tests. Standardized retained matrix no-change aliases are a different, explicit mechanism.
Infeasible candidates/rejected repairs reduce new fit counts. For `P` fully
crossed pairings, there are `52*P` settings before recipe/candidate expansion.

| Related family | Scope / distinguishing setting |
|---|---|
| Teacher architecture Study A | CIFAR-10, CIFAR-100, BloodMNIST, EuroSAT, GTSRB at IF10/IF100; 20 additional jobs; ResNet-18/34 teachers with separate repair maps |
| Student architecture Study B | 23 accepted-repair settings; additional ResNet-8/20 students |
| Alternative losses | 23 accepted-repair settings spanning 11 datasets; DKD, RLD, LoCa from original teachers |
| Matched loss-balance follow-up | Same 23 settings; DKD response weight `0.15`, RLD `0.2` |

Alternative-loss runs inherit temperature/warmup from Section 4. LoCa uses the
classical KD balance; published-setting DKD/RLD use full CE and response weight
`1`, with internal `alpha=1, beta=8`. Full details:
[architecture studies](architecture-studies-statistics.md),
[distillation comparisons](distillation-baselines-comparison.md).
The accepted-only caveat applies to the student/loss panels, not a claim that
every architecture study uses the same subset.

The [legacy CIFAR-10-LT100](../configs/teacher_baselines/cifar10_lt100.json) and
[CIFAR-100 balanced](../configs/teacher_baselines/cifar100_balanced.json) static
pilots retain legacy model IDs and disabled sparse fallback; they do not define
the retained full-matrix defaults.

The original [single-dataset repair report](../runs/cifar10-lt-neuron-surgery/report.md)
and [campaign variants](neuron-surgery-campaign.md) retain their own frozen settings.
Saved counts here are summary evidence, not a new audit of every artifact.
No historical checkpoint, protocol, result, or missing runtime field is rewritten.

### A.7. Retained matrix inspection and execution

Read the saved matrix, or inspect the retained generator without creating outputs.
These commands do not configure Colored MNIST, Waterbirds, or UrbanCars:

```bash
jq '{datasets, profiles, job_count: (.jobs | length)}' \
  runs/experiment1-multidataset/matrix-plan/protocol.json

python -m kd.teacher_baseline_matrix plan \
  --output runs/teacher-baselines-v7/class-reference-plan --epochs 100 --pairs convnet6 --dry-run \
  --datasets cifar10 cifar100 svhn cinic10 gtsrb fashionmnist pathmnist \
  bloodmnist dermamnist organamnist caltech101 eurosat stl10

python -m kd.teacher_baseline_matrix plan \
  --output runs/teacher-baselines-v7/class-reference-plan --epochs 100 \
  --pairs convnet6 resnet34_resnet18 --dry-run \
  --datasets cifar10 cifar100 svhn cinic10 gtsrb fashionmnist pathmnist \
  bloodmnist dermamnist organamnist caltech101 eurosat stl10
```

Omitting `--dry-run` materializes configs/protocol/queue only, not training.
Plan, dataset-cache, and study-output roots must be disjoint; incompatible plans
and unexpected files/symlinks are rejected. A chosen queue row is executed through
`kd.teacher_baselines train --config CONFIG --output OUTPUT`, followed by the
separate `evaluate` phase. Hardware effectiveness and standardized retained benchmark results
require actual execution; implementation tests are not performance evidence.

Generate a fresh version-2 plan for the standardized protocol; do not overwrite
old plans. Recognized legacy generated configs without execution bindings are
rejected. Use each queue row's recorded output and settings: changing a bound
config or applying CLI device/recipe/output overrides requires a fresh plan.
The parent identity is a canonical payload digest excluding only circular
binding metadata, not a byte-for-byte hash of `protocol.json`. Standalone static
configs remain separate experiments and are not silently treated as matrix jobs.

### A.8. Retained detailed-results template

The retained standardized 13-dataset benchmark has no completed full-matrix
results yet. The following reference tables show the publication/reporting
layout, assembled from each setting's
`comparison.json` and frozen `selection.json`. They are not the exact output of
the existing `report.md` renderer. Section 9's overview driver now assembles
setting/endpoint summaries and coverage for fresh bound plans. Historical
reports without the new evaluation seals are not silently adopted as new results.

Use one panel per **dataset / profile / architecture pair**. Example panel:
CIFAR-100 / `lt-if100` / ConvNet-6-W32 -> ConvNet-6-W16. All values are final
protocol-test measurements after validation selection. Teacher/student training
is 100 epochs; repair is 20 epochs. `—` means not measured, and `N/A` means the
field does not apply. No numeric performance results are supplied here.

| Condition | Teacher Acc / BAcc (%) | Student Acc (%) | Student BAcc (%) | Student target recall (%) | Paired Δ primary vs control KD (pp) | Run / repair status | Validation-selected? |
|---|---:|---:|---:|---:|---:|---|:---:|
| Supervised student, no teacher | N/A | — | — | — | N/A | Pending | N/A |
| CE control -> KD | — / — | — | — | — | Reference | Pending | — |
| RandAugment -> KD | — / — | — | — | — | — | Pending | — |
| Mixup -> KD | — / — | — | — | — | — | Pending | — |
| CutMix -> KD | — / — | — | — | — | — | Pending | — |
| Label smoothing -> KD | — / — | — | — | — | — | Pending | — |
| Balanced Softmax -> KD | — / — | — | — | — | — | Pending | — |
| MSE logits -> KD | — / — | — | — | — | — | Pending | — |
| MSE probabilities -> KD | — / — | — | — | — | — | Pending | — |
| SAM -> KD | — / — | — | — | — | — | Pending | — |
| EMA -> KD | — / — | — | — | — | — | Pending | — |
| RandAugment + CutMix + EMA -> KD | — / — | — | — | — | — | Pending | — |
| Repaired CE teacher -> KD | — / — | — | — | — | — | Pending | — |

Presentation rules:

- Primary score `M` is **accuracy for balanced** and **balanced accuracy for
  long-tail**. Acc, BAcc, and recall are percentages; deltas are percentage points.
- Teacher values describe seed `41`, without an uncertainty claim. Student values
  and paired deltas use `mean ± sample SD` over seeds `42/43/44`, conditional on
  that teacher. Compute each seed's delta against the matching control KD seed
  first, then summarize the three differences; do not subtract or combine SDs.
- Keep all eligible recipe rows, favorable or unfavorable. Balanced CIFAR-10 and
  CIFAR-100 omit Balanced Softmax: 11 recipe rows plus the supervised reference;
  other settings have 12 recipe rows plus that reference.
- Mark exactly the frozen `validation_selected_recipe` as selected, not the
  highest test score. The supervised reference is not a candidate teacher recipe.
- After verified evaluation, a rejected repair is labelled
  `Complete / no change (control alias)` and has an exact zero paired delta.
  Accepted repair is `Complete / selected`. Aliases are reused control results,
  not additional independent fits. `Failed`, `Missing`, and `Pending` cells stay
  unmeasured, with reasons; they must not be converted to zero.

Teacher/student NLL, summed-class Brier, and 15-bin ECE belong in a companion
quality table; include teacher target recall there as well. Candidate feasibility,
selected channels/LR, rejection reasons, and paired target-recall deltas belong
in the repair supplement. Keep costs separate with Appendix A.5's timing/precision
limitations; do not add significance marks to descriptive seed SDs.

The recipe metrics come from `comparison.json`'s `rows[].teacher` and
`rows[].student`. Paired deltas are `rows[].paired_student_delta_vs_control`;
repair decisions are `rows[].repair_outcomes`. The supervised reference uses
`supervised_student` for the three accuracy/recall metrics; its calibration
metrics are available per run, not in that aggregate. No paired-supervised
comparison or confidence interval is emitted by the current runner.

### A.9. Retained full-dataset results template

Use one overview per **profile / architecture pair**, alongside the full recipe
panels above. Example: `lt-if100` / ConvNet-6-W32 -> ConvNet-6-W16, so `M=BAcc`.
`Selected` below always means validation-selected. Coverage is completed recipe
conditions / eligible recipe conditions, excluding the supervised reference;
this example has 12 eligible conditions per dataset.

The canonical overview is now Section 10.2's combined 16-dataset table.
For this family, use its 13 class rows rather than maintain a duplicate roster.

Repeat for all four profiles: 52 setting rows per pair, or 104 for both supported
pairs. Preserve each setting even when it has no final report. Record repair
acceptance/no-change, candidate infeasibility, failures, and missing work in a
coverage supplement with denominators; these categories must not be conflated.

Any cross-dataset recipe summary must stay separate by profile and pair, average
dataset-level scores with equal dataset weights, and report completed / eligible
datasets. Never pool predictions across datasets. Balanced Softmax is eligible
for 11/13 balanced datasets (50/52 settings per pair overall); its subset average
cannot be ranked against another recipe's full 13-dataset average. Cross-recipe
aggregate comparisons require an explicitly common eligible, completed set.
If descriptive aggregate seed SD is reported, first form one equal-dataset mean
per student seed on that fixed set, then compute SD across those three means;
averaging the individual setting SDs is not an aggregate SD.

Historical 49/71/81-epoch outcomes in Appendix A.6 remain a separate results panel.
They must not fill blank cells in these standardized 100/20-epoch tables. Nor may
from-scratch compact results be relabeled as adapted-pretrained outcomes. Equal
input sizes do not make this a pretraining-only ablation: normalization and LR
also differ between the declared presets. Preserve
pretraining-overlap disclosures and use common audited/unaudited eligibility
sets explicitly if making any cross-dataset summary.
