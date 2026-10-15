# Teacher training baselines and distillability

Research date: 12 October 2026. Literature discovery used the Consensus MCP;
the MSE implementation and recent publication records were additionally checked
against primary papers and author repositories.

Implementation update: 14 October 2026. Protocol version 2 includes the original
teacher-repair condition and stricter completion/provenance verification. Existing
repair code and historical experiment directories are retained without edits.

The all-dataset extension uses a separate immutable matrix planner with neutral
model IDs, profile-specific epoch budgets, sparse-class repair fallback, recorded
dataset-specific augmentation policies, and early runtime/data-identity guards.
See [experiment settings](experiment-settings.md) for the consolidated parameters.

## Research question

Which teacher-training interventions improve the teacher, and which improve its
distilled student under a fixed KD recipe? These are separate outcomes. All arms
must be distilled, including teachers that lose some accuracy, to avoid selecting
away improvements in transfer.

These are research-backed comparison methods, not a claim that every method is
currently state of the art or will improve this project's compact CNNs. The
training runner implements the practical first stage. Recent methods requiring
different architectures, multiview learning, or posterior inference are listed
explicitly as follow-up work.

## Evidence and priorities

- **Augmentation first.** Research on the interaction of augmentation and KD
  supports testing Mixup and CutMix rather than assuming that the teacher's
  clean-image accuracy tells the whole story. Much of this evidence changes
  augmentation *during student KD*. Our first comparison changes teacher training
  only, so its transfer benefit is a hypothesis to test, not an established
  consequence of those papers. See [Knowledge Distillation Thrives on Data
  Augmentation](https://consensus.app/papers/knowledge-distillation-thrives-on-data-augmentation-wang-lohit/d4bd33d29e2a5f9d8ccf379e76e3befb/),
  [What Makes a Good Data Augmentation in KD](https://consensus.app/papers/what-makes-a-good-data-augmentation-in-knowledge-wang-lohit/6c7592a8ec545289b594b97c164d4ff4/),
  and the [WACV Mixup/KD empirical study](https://consensus.app/papers/understanding-the-role-of-mixup-in-knowledge-distillation-choi-jeon/5d40d993efa955098b0a5d197e7a08fe/).
- **A recent method specifically for teachers.** [How to Train the Teacher Model
  for Effective Knowledge Distillation, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12363.pdf)
  reports improved student results from MSE-trained teachers on CIFAR-100 and
  ImageNet, even with lower teacher accuracy. This is a particularly relevant
  baseline for our question. Its reported gains are not estimates for our study.
- **A long-tail control.** [Balanced Meta-Softmax](https://consensus.app/papers/balanced-metasoftmax-for-longtailed-visual-recognition-ren-yu/9717b947d6ae517ea25b74e827696352/)
  motivates a training-prior correction. We implement the Balanced Softmax loss,
  not its additional meta-sampler. The [authors' loss implementation](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification)
  adds `log(training_class_count)` during training and uses raw model logits for
  inference. Whether that improvement transfers to a student remains empirical.
- **A useful diagnostic arm.** [When Does Label Smoothing Help?](https://research.google/pubs/when-does-label-smoothing-help/)
  finds that label smoothing can improve teacher generalization/calibration while
  reducing distillation effectiveness. Include it to detect a discrepancy between
  teacher quality and teaching quality; do not assume it is a KD improvement.
- **Optimization controls.** [SAM](https://consensus.app/papers/sharpnessaware-minimization-for-efficiently-improving-foret-kleiner/4b052f0a403b5557beaa1a42f96a5859/)
  is an established generalization method. EMA is included as a practical weight
  averaging control. Neither is presented here as a proven improvement for this
  project's downstream KD.

My first priorities are **CutMix/RandAugment for generalization, Balanced Softmax
for long-tail recognition, and MSE for distillability**. Run the single-factor
arms before interpreting the combined recipe.

## Implemented arms

The ordinary training-baseline arms retain the same teacher architecture,
initialization seed, dataset split, base crop/flip policy, optimizer schedule,
number of epochs, and checkpoint selection rule. The repair condition adds the
original method's separately declared intervention and validation gate, described
below. Teacher batch mixing is not inherited by downstream students.

| Recipe | Teacher intervention | Purpose |
|---|---|---|
| `control` | Existing crop/flip + hard-label CE + SGD | Matched reference |
| `randaugment` | RandAugment N=2, M=9 before tensor conversion | Search-free augmentation control |
| `mixup` | Mixed images and soft labels, alpha=0.2 | Interpolation/generalization |
| `cutmix` | Patch mixing and realized-area labels, alpha=1 | Preserve local visual structure |
| `label_smoothing` | CE with epsilon=0.1 | Accuracy/calibration versus transfer diagnostic |
| `balanced_softmax` | CE(logits + log training counts, labels) | Long-tail classification control |
| `mse_logits` | Mean squared error(raw logits, one-hot labels) | Released-code-inspired MSE teacher |
| `mse_probabilities` | Mean squared error(softmax(logits), one-hot labels) | Distinct probability-space MSE/Brier variant |
| `sam` | SGD with SAM rho=0.05 and two backward passes | Optimization/generalization control |
| `ema` | Parameter EMA decay=0.999, current BN buffers | Weight averaging control |
| `randaugment_cutmix_ema` | RandAugment + CutMix + EMA, no label smoothing | Prespecified practical combination |
| `teacher_repair` | Existing channel repair applied to the common CE control teacher | Direct comparison with the project's proposed method |

### Original teacher repair in the common comparison

The two presets now include `teacher_repair` and an explicit `repair` section.
This condition starts from **the same trained CE control checkpoint** used by
`kd_control`, not a separately trained teacher. It invokes the existing
`run_neuron_surgery_study` with `downstream_kd=false`; that mode localizes channels,
repairs candidates, and selects a teacher using validation only. The common
benchmark then runs fresh matched KD students and controls all final test access.
The original repair algorithm and saved results are preserved; input-protocol
metadata is carried explicitly for adapted pretrained runs.

The default repair grid retains channel budgets 2/4/8, learning rates
0.001/0.005, 20 repair epochs, 640 samples per stream per epoch, batch size 64 for
the class-family 32-pixel presets (including adapted pretrained ResNets),
same-class companions, bootstrap screening, causal validation, and the original
logit/feature preservation objective. Localization uses training examples.
Balanced profiles target all classes; long-tail profiles target the lowest-frequency
half, using the original repair protocol's policy. The same target classes are
used in the reported recall of every compared condition.

If no repair meets the original validation gate, the condition records
`no_change`, exports an exact copy of the control teacher, and reuses the
seed-matched control students. The reported intervention effect is exactly zero;
no redundant KD fitting is performed. Aliases record their original run and
completion digest. An accepted repair trains its own matched students.

Repair adds localization, optimization, and candidate selection beyond the base
teacher fit. Thus this comparison matches datasets, initializations and student
training, but **does not equalize teacher compute or search budgets**. Timings and
candidate counts are reported. A separate equal-compute continuation control and
equally budgeted hyperparameter searches are needed for mechanism attribution.

Override the grid only through the frozen `repair` JSON section, for example:

```json
"repair": {
  "channel_budgets": [2, 4, 8],
  "learning_rates": [0.001, 0.005],
  "repair_epochs": 20,
  "repair_samples_per_epoch": 640,
  "repair_batch_size": 64,
  "target_fraction": 0.2,
  "companion_count": 5,
  "allow_sparse_class_fallback": false
}
```

Repair supports the catalog's 32x32 datasets and CifarCNN/torchvision ResNet
adapters, including CIFAR-stem ResNets. Synthetic and ImageFolder experiments can
still run the training baselines without the repair condition. Sparse fallback is
an explicit opt-in, following the existing generalized repair protocol.

### MSE implementation distinction

The paper analyzes probability estimates, but its [released training
loop](https://github.com/ECCV2024MSE/ECCV_MSE_Teacher/blob/main/helper/loops.py)
passes model outputs directly to MSE, and its ResNet returns raw logits. The
released `train_teacher.py` also has an `upper`/`upper()` typo in criterion
selection. We therefore implement two named arms rather than silently treating
the probability-space variant as the released method. Neither arm reproduces the
paper's reported numbers: our architectures, cosine schedule, validation
selection, and 81-epoch budget differ from its experimental recipe.

Both losses average over examples **and classes**. Probability MSE is training
Brier score divided by the class count; the evaluation Brier score sums across
classes. Loss scale affects optimization, especially for CIFAR-100. A poor result
at the fixed CE learning rate does not establish that a tuned MSE method is poor.
A separate, equally budgeted, validation-only learning-rate sweep is needed for
an optimized-method comparison.

### Augmentation and optimization details

RandAugment is inserted into the training dataset before `ToTensor`; validation
and test keep their canonical transforms. CutMix labels use the actual clipped
patch area. Mixup/CutMix targets remain normalized probability distributions.
SAM reuses the same batch and soft targets for both passes, restores unperturbed
parameters before SGD, and advances BN running statistics only on the first pass.
EMA averages parameters and copies model buffers; validation always uses the EMA
model for that arm, not a post-hoc choice between raw and EMA weights.

Both pilot configs and new matrix plans request explicit `bfloat16` for all arms,
including SAM. CUDA AMP keeps model/optimizer state FP32; BF16 forward operations
do not use a gradient scaler. Teacher loss/KD reductions and exported prediction
probabilities retain their FP32 paths. Explicit BF16 requires native CUDA support
and does not silently fall back. Use `--precision float32` for a fresh CPU/debug
matrix plan or standalone unbound study; repeat standalone overrides for evaluation.
The runner also supports ordinary scaled FP16 training, but rejects FP16 SAM before
any study training or writes, including when `auto` resolves to FP16. CUDA-specific
execution still needs verification on the intended GPU server. See the
[precision policy](experiment-settings.md#321-bf16-execution-policy-for-new-baseline-runs).

Create BF16 studies in a fresh output root such as `runs/teacher-baselines-bf16`.
Never relabel or resume the completed FP32 Waterbirds experiment as BF16. Its
saved report and prediction archives are unchanged. Frozen matrix bindings reject
late precision overrides; change precision when generating a new plan.

## Recent SOTA-oriented follow-ups: researched, not implemented

| Candidate | Evidence/status | Why include later | Required comparison |
|---|---|---|---|
| Sample-aware RandAugment (SRA) | IJCV 2025; [Consensus record](https://consensus.app/papers/sampleaware-randaugment-searchfree-automatic-data-xiao-yu/4cc98c3d1e8456889dca1483ba4fc286/), [author code](https://github.com/ainieli/Sample-awareRandAugment) | Recent adaptive augmentation with reported image-recognition gains | Compare to plain RandAugment with its full scoring/asymmetric policy; then freeze students' KD inputs |
| Aligned Contrastive Learning (ACL) | 2025 [paper](https://arxiv.org/abs/2506.01071), [Consensus record](https://consensus.app/papers/aligned-contrastive-loss-for-longtailed-recognition-ma-cui/7e9cf7b04da85fa7bf88472f714c9047/) | Recent long-tail representation method addressing contrastive gradient conflicts | Match multiview count and augmentation in a non-contrastive control; export classifier logits and evaluate downstream KD |
| Bayesian teacher | ICLR 2026 [proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0bdbfac7d9df70021d694e5ac795bf9d-Abstract-Conference.html), [Consensus record](https://consensus.app/papers/sgdbased-knowledge-distillation-with-bayesian-teachers-morad-shlezinger/ba7a37db024151bb88de4aa98ecf3329/), [author code](https://github.com/itaimorad/bayesian-kd) | Recent work directly studying teacher uncertainty and student convergence | Fix posterior sample count and aggregate predictive probabilities; account for teacher inference cost |

Plain RandAugment is not SRA, single-view CE is not ACL, and EMA is not Bayesian
posterior inference. These follow-ups require their own tested integrations.
Language-model-specific teaching methods returned by the search were excluded
from the initial vision baseline set.

## Matched experimental protocol

1. Start with `cifar10_lt100.json` and `cifar100_balanced.json`. The first is
   aligned with the original factor-100 teacher/data recipe. The second adds a
   100-class setting relevant to the MSE paper. Add CIFAR-10 balanced and
   CIFAR-100 IF100 by copying and changing the dataset profile in fresh configs.
2. Retrain the CE control inside this benchmark. Do not mix old results obtained
   with different precision, schedules, or validation rules with these new arms.
3. The pilot uses teacher seed 41 and student seeds 42/43/44. Each training baseline
   fits one teacher and three matched students. Supervised students are shared once
   per seed. The repaired-teacher condition reuses the fitted CE teacher and can
   fit at most six repair candidates and three more KD students. The **12-condition
   CIFAR-10 LT preset has at most 56 fits** (11 base teachers + 6 repair candidates
   + 36 KD + 3 supervised). The balanced CIFAR-100 preset omits Balanced Softmax,
   which equals CE for equal counts, and has **11 conditions and at most 52 fits**.
   Rejected repairs or infeasible budgets reduce the actual fitting count.
4. Ordinary teacher checkpoints use the same predeclared validation metric across
   training baselines. Repair retains its original validation target-recall gate
   and 0.5-percentage-point overall-accuracy guardrail.
   Downstream students always use the existing best-validation-accuracy rule,
   with KD T=4, response weight=0.5, warmup=5, basic augmentation, no feature
   transfer, and identical initialization and image streams for a given seed.
5. Report teacher accuracy, balanced accuracy, target-class macro recall, NLL,
   Brier score, ECE, and measured training time. Report the same predictive
   metrics for students and matched student differences against CE-teacher KD.
   Repair-enabled studies use the repair protocol's target-class policy for all
   rows. Other generic studies use the lowest-frequency half of **training** classes
   with label-index tie breaking; equal-count settings target all classes. ImageFolder
   studies use actual counts recorded by the teacher, not a missing provenance field.
6. Choose the teacher-training recipe using mean **student validation** performance,
   not teacher accuracy or test performance. Keep the separately best teacher
   result visible. In long-tail settings balanced accuracy is the primary student
   selection metric; overall accuracy and target recall remain reported outcomes.
7. Freeze all checkpoint hashes and the validation report before the separate
   test phase. Report every prespecified arm on test; retain the validation-selected
   recipe even if another test row is higher. No training stage loads the test set.
8. For confirmation, use at least three teacher seeds as well as three student
   seeds. The runner supports the crossed grid and reports student results per
   teacher seed. Pooled fit SD is descriptive: shared teacher and student seeds
   are not independent replicates. Use paired hierarchical/two-way uncertainty
   analysis before making significance claims.

The 81-epoch budget is a controlled pilot, not a guarantee of convergence under
strong augmentation. Prespecify longer training and equal hyperparameter-search
budgets as a separate phase. Tune the common temperature grid using student
validation if logit scale changes substantially, especially in the raw-MSE arm;
do not tune from test outcomes.

Existing CIFAR test results have already been examined in this workspace. New
results on those same images remain exploratory, even though this runner gates
test access. An independent confirmation split is needed for confirmatory claims.
The all-dataset extension records restricted photometric RandAugment spaces for
SVHN/GTSRB and the four MedMNIST sources. These are declared variants, not standard
RandAugment or proven task invariances; dataset-level review remains necessary.

## All-dataset matrix

`kd.teacher_baseline_matrix` now covers the full 16-dataset benchmark: four
profiles on the original 13 catalog datasets, plus one native setting on each
shortcut dataset. Its primary class-family preset is `resnet34_resnet18_smallstem_imagenet`:
adapted small-stem ResNet-34 -> ResNet-18, both derived from `IMAGENET1K_V1`,
seeded new heads, 32-pixel ImageNet-normalized inputs and full fine-tuning. The
stem adaptation is declared in experiment settings; these are not official
small-image pretrained weight releases. New teacher/student
budgets are uniformly 100 epochs; repair remains 20. Historical 81/49/71/81
budgets, original static pilot JSONs and the TPAMI `main` branch remain unchanged.
The exact optimization, views, repair batches and overlap disclosures are in
[experiment settings](experiment-settings.md).

Optional compact presets `convnet6` and `resnet34_resnet18` retain 32-pixel,
from-scratch architectures. The default full plan has 55 settings and at most
3,117 fits; primary plus compact ConvNet has 107 settings and at most 6,021 fits.
All three class-family presets give 159 settings and at most 8,925 fits. These
are upper bounds, not completed training or equal-compute comparisons. Colored
MNIST stays at 32 pixels from scratch; Waterbirds/MultiCelebA use pretrained pairs at
224 pixels. MultiCelebA replaces UrbanCars in new default plans and is baseline/KD-only
(11 teachers, 33 KD students, 3 supervised students); repair/controls are disabled.
UrbanCars remains an explicit legacy option. CINIC-10/STL-10 pretraining overlap is explicitly unaudited.
Other literature pairings remain pending model/repair-map integration.

```bash
# Read-only: all datasets/profiles, no writes or dataset/model access.
python -m kd.teacher_baseline_matrix plan \
  --output runs/teacher-baselines-bf16/matrix-plan --dry-run

# Primary pretrained pair plus compact reference; still read-only.
python -m kd.teacher_baseline_matrix plan \
  --output runs/teacher-baselines-bf16/matrix-plan \
  --pairs resnet34_resnet18_smallstem_imagenet convnet6 --dry-run
```

Without `--dry-run`, planning writes only `protocol.json`, `configs/*.json`, and
`jobs.tsv`; it does not launch training. Each queue row identifies the config and
its separate study output for the ordinary `teacher_baselines train/evaluate`
phases. Dataset, plan, and study roots must not overlap. Incompatible protocols,
config edits, unexpected files, and unsafe symlink paths fail before overwrites.
The full-scope plan binds its own source and all inherited computational sources.
Study policies and teacher artifacts record the exact RandAugment operation space.

## Commands

The entry point is separate from the original `kd` CLI and training engine.
Use the project's `kd` Python environment, from the repository root.

```bash
# Read-only: inspect counts and the entire frozen protocol, no dataset access.
python -m kd.teacher_baselines plan \
  --config configs/teacher_baselines/cifar10_lt100.json \
  --output runs/teacher-baselines-v2/cifar10-lt100 --device cuda:0 --dry-run

# Optional: materialize protocol.json without starting training.
python -m kd.teacher_baselines plan \
  --config configs/teacher_baselines/cifar10_lt100.json \
  --output runs/teacher-baselines-v2/cifar10-lt100 --device cuda:0

# Train every teacher, the supervised controls, and the matched KD students.
python -m kd.teacher_baselines train \
  --config configs/teacher_baselines/cifar10_lt100.json \
  --output runs/teacher-baselines-v2/cifar10-lt100 --device cuda:0

# Explicit test access after validation selection has been frozen.
python -m kd.teacher_baselines evaluate \
  --config configs/teacher_baselines/cifar10_lt100.json \
  --output runs/teacher-baselines-v2/cifar10-lt100 --device cuda:0

# Offline execution check for the 11 training baselines and their downstream KD.
python -m kd.teacher_baselines smoke --output runs/teacher-baselines-smoke
# Includes the actual repair runner on a tiny offline catalog adapter, plus
# accepted/rejected integration fixtures and corruption regressions.
python -m unittest tests.test_teacher_baselines -v
```

Use the **same device override on every phase**: it is part of the frozen
protocol. If planning on a CPU-only machine for a CUDA server, include the CUDA
override in the initial plan; dry-run does not construct the device or models.
For a cheaper first pass, use `--recipes control cutmix balanced_softmax
mse_logits mse_probabilities teacher_repair` consistently on all phases with a
fresh output. Omitting `teacher_repair` also disables its options for that plan.

Ordinary teacher/student fits resume from `last.pt` at epoch boundaries, including
optimizer, scheduler, AMP, EMA, and RNG state. Repair stages use the existing repair
runner's cached-stage/candidate behavior. Complete artifacts and required manifest
entries are verified before completed-study reuse and before test access. Recovery
of an unmanifested student checks its recorded teacher digest, full training
configuration, and agreement between its export and selected best checkpoint.
The study completion manifest binds all per-run completion manifests, validation,
and selection. Source hashes include `metrics.py` and the active repair code.
The normal `student.pt` contract lets every new teacher be consumed by existing
KD code. Changes to source or protocol require a new output root. GPU overrides
now also govern report evaluation and precision; evaluation is not forced to CPU.
Freshly rebuilt training/validation provenance is compared with the frozen run
before reporting predictions; adding explicit test-loader metadata is permitted,
but a changed dataset, archive, or split is rejected.

Main study artifacts:

- `protocol.json`: immutable configuration, recipe list, seeds, counts, code hashes.
- `teacher_*/`: teacher protocol, checkpoints, initialization/data identities,
  history, runtime metadata, summary and completion hashes.
- `teacher_repair_seed*/repair/`: existing repair-stage artifacts and validation
  selection; its parent also records `repair_decision.json` and the exported teacher.
- `kd_*/`, `supervised_student_*/`: ordinary engine runs and completion hashes.
- `baseline_alias.json` inside rejected-repair KD runs: exact control reuse identity.
- `validation.json`, `validation.md`: all teacher/student validation outcomes.
- `selection.json`: validation-selected recipe and hashes of all selected models.
- `study_completion.json`: hashes of selection/validation and every run's manifest.
- `comparison.json`, `report.md`, `test/`: final results and probability archives.

The smoke run is only a correctness check. There is no measured real-dataset
winner until the pilot training and evaluation commands complete.

Teacher timing includes training and validation. Repair timing includes the base
teacher plus the successful repair invocation's wall time; time spent in earlier
interrupted repair invocations is excluded. Student timing follows the ordinary
engine's training-epoch scope and is zero for reused control aliases. These timing
scopes are descriptive and are explicitly recorded in summaries.
