# Shortcut subset: teacher repair and knowledge distillation

Shortcut design selected: 14 October 2026; full scope clarified: 15 October 2026.
**Implemented protocol defaults, not a completed full-budget experiment.** This
companion covers Colored MNIST, Waterbirds, and UrbanCars within the
[full 16-dataset experiment](experiment-settings.md#11-complete-16-dataset-roster).
The original 13 datasets remain in that experiment with their class/long-tail
rules; their new primary pair is adapted pretrained ResNet-34 -> ResNet-18 at 32 pixels.
The three shortcut settings below are unchanged by that transfer-default update.
Historical configuration identities, checkpoints, and saved results
are unchanged. The filename is retained for existing links. Exact numerical
defaults and execution commands are maintained in the main settings document;
this companion explains research provenance and interpretation limits.

Question: does channel repair improve robustness to a specified nuisance, and
does that improvement transfer to a student under matched classical KD?
Channel ablation measures an intervention inside a model; it does not alone
identify a real-world cause or prove that a channel encodes the nuisance.

## 1. Dataset scope

Consensus MCP discovery/fetched records were checked against the papers and
author code below. Dataset facts and implemented project adaptations are separated.
There are no IF10/50/100 profiles in this shortcut subset: shortcut-group imbalance
must not be conflated with class imbalance.

| Dataset | Prediction target | Nuisance metadata / groups | Role |
|---|---|---|---|
| Colored MNIST | Binary digit category, with the declared label-noise construction | Observed label x color: 4 groups; retain environment ID too | Controlled color-shortcut test |
| Waterbirds | Landbird / waterbird | Bird label x land/water background: 4 groups | Background-shortcut robustness |
| UrbanCars | Urban / country vehicle category | Vehicle label x background x co-occurring-object category: 8 groups | Multiple shortcuts; inspect both separately and jointly |

Colored MNIST reference: digit `0-4` versus `5-9`, label-flip probability `0.25`,
and color/observed-label mismatch probabilities `0.2/0.1` in training and `0.9`
in the shifted environment. The original IRM implementation uses a two-channel,
downsampled representation and MLPs; an RGB ConvNet variant is an adaptation,
not reproduction of those scores. [IRM paper](https://arxiv.org/html/1907.02893),
[author code](https://github.com/facebookresearch/InvariantRiskMinimization).

Waterbirds uses constructed bird/background combinations and released split and
background metadata. UrbanCars has two controlled training shortcuts, each with
agreement probability `0.95`, and uncorrelated validation/test compositions.
[Waterbirds paper](https://arxiv.org/html/1911.08731),
[Waterbirds author code](https://github.com/kohpangwei/group_DRO),
[UrbanCars paper](https://arxiv.org/html/2212.04825),
[UrbanCars author code](https://github.com/facebookresearch/Whac-A-Mole).

## 2. Data and preprocessing decisions to freeze

| Item | Implemented shortcut-family policy / audit boundary |
|---|---|
| Colored MNIST splits | Split underlying MNIST images before coloring; derive train/validation only from official training images; reserve official test images for final evaluation. Retain original image/digit identity, clean digit-derived category, noisy observed label, color, and environment separately. Freeze the partition, label-noise realization, environment allocation, rendering, and RNG seeds. This split policy is a project adaptation. |
| Colored MNIST environments | Retain the reference `0.2/0.1` training mismatch rates; validation comes only from training environments. Evaluate prespecified `0.9` reversed and `0.5` unbiased test renderings separately, with fixed observed labels. They share underlying test images and are not independent replications. |
| Waterbirds / UrbanCars splits | Use Waterbirds' released split metadata and UrbanCars' released/author-defined partitions and group mapping, without re-stratifying or long-tail resampling. Pin the UrbanCars artifact or generation recipe; its author pipeline uses generated split folders, not a presumed Waterbirds-style metadata file. Verify all required groups and actual split counts. |
| Starting resolution / pairing | Colored MNIST: RGB `32 x 32`, ConvNet-6-W32 -> W16, from scratch. Waterbirds: `224 x 224`; UrbanCars: `256 x 256`; ordinary-stem ResNet-34 -> ResNet-18 for both. These pairings are project choices, not native paper reproductions. |
| Real-image initialization | Both roles use named torchvision `IMAGENET1K_V1` weights and seeded two-class heads; initial tensor digests are matched across arms. Family-specific optimization is in main Section 3.2, not inherited blindly from the compact-data protocol. |
| Base augmentation | Main Section 2.2 freezes cue-preserving Colored MNIST views, Waterbirds resize/center crop and native UrbanCars views. Teacher recipe augmentation is explicit; students retain base views. |
| Pairing/counterfactual endpoint | Recoloring the same held-out digit is valid by construction. Same-foreground background/co-object swaps require source masks/IDs and a validated renderer; ordinary group comparisons are not automatically paired counterfactuals. |
| Source reuse | Original MNIST IDs and available Waterbirds foreground IDs are checked across splits. Waterbirds background IDs may be unavailable; released UrbanCars composites do not export original constituent IDs. The loader records this audit limitation rather than claiming verified source-disjointness from distinct filenames. |
| Provenance / access | Pin dataset release, metadata, split membership, group mapping, rendering recipe, preprocessing, and source identities. Check licenses before acquisition; no downloads are triggered by this document. Test images are unavailable to fitting, localization, matching, or selection. |

Waterbirds' author ResNet-50 configuration specifies a `224 x 224` target
resolution. UrbanCars' author generator uses `256 x 256`, and its ordinary
ResNet transforms do not resize/crop these composites. Do not silently reduce
either benchmark to the retained matrix's `32 x 32` protocol.
[Waterbirds model settings](https://github.com/kohpangwei/group_DRO/blob/master/models.py),
[UrbanCars generator](https://github.com/facebookresearch/Whac-A-Mole/blob/main/create_datasets/urbancars/gen_urbancars.py),
[UrbanCars transforms](https://github.com/facebookresearch/Whac-A-Mole/blob/main/model/classifiers.py).

## 3. Teacher, student, and KD comparison

The existing teacher recipes remain candidates: CE, logits/probability MSE,
label smoothing, Balanced Softmax, EMA, Mixup, CutMix, RandAugment, the combined
RandAugment+CutMix+EMA recipe, SAM, and CE-teacher repair. Their implemented
definitions are in [the reference settings](experiment-settings.md#6-teacher-training-comparison-arms).
Freeze the eligible list and tie order before running. Balanced Softmax addresses
class counts, not shortcut-group imbalance; if exact class counts make it CE-
equivalent, omit it with a recorded reason rather than count it as independent
method evidence. Cost tiers order presentation only, not selection tie-breaking.

Retain the project **100 teacher/student epochs and 20 repair epochs as fixed
defaults**, not evidence of convergence or a literature-native optimum. Teacher
seed `41` and paired student seeds `42/43/44` are the initial descriptive-seed
design. Initialization, per-dataset optimizer settings and augmentation are
frozen in the generated plan; any validation-only tuning needs a declared budget.
Actual examples/updates and search cost are reported, not inferred from epochs.

Reuse the [classical KD objective](experiment-settings.md#4-downstream-knowledge-distillation):
`T=4`, response weight `0.5`, full five-epoch response warmup, frozen/eval teacher,
and identical teacher/student input tensors. The supervised reference uses full
CE, no teacher, and the same student initialization/budget/data policy. Student
augmentation does not automatically inherit teacher interventions.

For attributing gains specifically to causal screening, the runner adds a repair
mechanism panel with the same original CE anchor:

- All-parameter continuation on the same examples and matched repair objective
  (target CE plus preservation KD/features, frozen buffers; not pure CE alone).
- Random-channel repair with the same channel-count and candidate/update budget.
- Attribution-ranked repair without the causal-ablation screening step.
- Full causal-screened repair.

Give these controls the same group-label access, repair pool, and companion/
matching-data access as full repair. Declare their candidate grids, selection
objective, localization/search cost, and preservation constraints. Equal updates
are not automatically equal FLOPs or wall-clock time. These controls are
implemented; they do not compete for the ordinary recipe winner. Exact grids,
seeds and replay updates are in main Sections 5.2 and 6.2.

## 4. Group-aware repair and selection

Known nuisance metadata is available for diagnostics, target specification,
validation selection, and metric computation; it is not a predictor input. This
is a group-informed experimental setting, not an annotation-free method claim.

| Decision | Required shortcut-family rule |
|---|---|
| Repair target | Prespecify shortcut-discordant training groups; do not substitute low-frequency classes or discover the target from test performance. Retain hard-example diagnostics within those groups. |
| Companions | Prefer valid same-foreground nuisance variants; otherwise use declared same-label nuisance-discordant matching and report coverage. Teacher feature proximity alone is not proof of content matching. |
| Causal screening | Measure target margin change and preservation separately for all required groups, in addition to aggregate accuracy. Record every screened channel and failure/no-feasible outcome. |
| Preservation | Prespecify per-group as well as overall guardrails. The existing `0.005` overall tolerance does not by itself constrain subgroup damage; new group thresholds must be frozen before fitting. |
| Ordinary checkpoint selection | Validation worst-group accuracy; then validation raw accuracy, then a frozen deterministic tie rule. Apply the same objective to ordinary teachers and individual students. |
| Repair candidate selection | Strict improvement in validation worst-group accuracy over the CE anchor, subject to declared overall/per-group guardrails; then frozen validation/tie rules. |
| Recipe selection | Highest mean KD-student validation worst-group accuracy across paired student seeds; frozen recipe-order tie break. |
| Final test | Evaluate all prespecified eligible arms only after checkpoint and validation selections are frozen. Never tune on the shifted test environment. |

These rules are implemented separately from the preserved class-target method.
Exact thresholds, pools and candidate tie rules are in
[main Section 5.2](experiment-settings.md#52-implemented-shortcut-family-repair).
Do not infer paired real-image counterfactuals from feature-nearest companions.

## 5. Metrics and interpretation

Primary endpoint: **worst-group accuracy**, `min_g(correct_g / n_g)`, over the
declared label x nuisance groups for each teacher and student. Colored MNIST
primary accuracy/group metrics use the noisy observed labels; any clean
digit-category accuracy is separately named, not silently substituted. Also report each
group's sample count/accuracy, raw sample-weighted accuracy, and group-balanced
accuracy (the unweighted mean of group accuracies). Group-balanced accuracy is
not the existing class-balanced accuracy. An absent required group makes the
full-group endpoint unavailable; it is not silently omitted or assigned zero.

Literature comparisons can additionally report accuracy weighted by training-group
proportions; label it separately from raw test accuracy. UrbanCars' paper ID
score uses a fixed independent `0.95/0.95` mixture of equal-class agreement
slices, not measured training counts. Its background, co-object and joint gaps
are shifted-slice accuracy **minus** that ID score. The implementation separately
reports actual training-group-weighted accuracy. Definitions:
[author evaluation](https://github.com/facebookresearch/Whac-A-Mole/blob/main/urbancars_trainers/base_trainer.py),
[local metrics](../kd/urban_metrics.py). Also report each
disagreement slice to detect a tradeoff between shortcuts; generic slice scores
are not silently substituted for the published gaps. Prespecified paired nuisance tests report
prediction-flip rates and accuracy before/after nuisance change, where valid;
constant predictions can be invariant but useless, so flips are not a stand-alone
success criterion.

Use paired student contrasts against the same-seed control KD. Report mean and
sample SD over three students conditional on one teacher, not a confidence interval
or evidence about teacher-seed variability. Additional teacher seeds are needed
for that claim. Report datasets separately; three constructed benchmarks are not
proof of general real-world causal identification or autonomous-driving safety.

## 6. Final-results layout

The canonical templates are in the main experiment settings:
[Section 10.1: detailed teacher/student comparison](experiment-settings.md#101-teacher-quality-and-downstream-transfer)
and [Section 10.2: full 16-dataset overview](experiment-settings.md#102-full-16-dataset-overview).
They are maintained there instead of duplicating tables in both documents.

No shortcut-subset experiment is completed. Its rows use raw and worst-group
accuracy alongside the class-family rows with their own metrics. The templates
retain paired contrasts, validation-only selection, and explicit coverage/status.
Colored MNIST has two correlated endpoints within one dataset. Supplement
definitions are in Section 5. Group-aware `comparison.json` now emits the
required group metrics, mechanism controls, paired deltas and Colored MNIST
endpoint reports; the full driver assembles a frozen-plan overview.

## 7. Implementation boundary and next work

Implemented modules: [shortcut loaders](../kd/shortcut_data.py),
[group metrics](../kd/group_metrics.py),
[group baseline runner](../kd/shortcut_baselines.py),
[group repair](../kd/shortcut_repair.py) and
[full benchmark driver](../kd/full_benchmark.py). The original catalog and
historical 13-dataset neuron-surgery matrix are not expanded or rewritten;
the new teacher-baseline planner includes all 16 in a fresh source-bound plan.

Offline fixtures cover split/group contracts, paired color rendering, repair
controls and recovery, pretrained-model plumbing and native-resolution ordinary
training/KD. They do not establish benchmark performance. Provision actual data,
check licenses/source-lineage limitations and run a short separate GPU diagnostic
with the actual ImageNet weights before launching full budgets. See
[main Section 9](experiment-settings.md#9-execution-readiness-and-pre-run-checks)
for preflight, train, evaluate and report commands. No full experiment or weight
download was launched during these implementation checks.
