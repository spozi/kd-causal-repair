# Supervised-only repair on imbalanced datasets

## Question

When a classifier is trained on long-tail data, does targeted channel repair improve
minority-class recall without sacrificing head-class recall or overall accuracy?

This is a separate experiment from Experiment 1. It reuses the **original supervised
classifiers** and the frozen dataset/split recipes, but retrains every repair candidate.
No student model is trained. No teacher logits or teacher features contribute to the
repair loss.

## Frozen protocol

- Datasets: the same 13 catalog datasets at IF10, IF50, and IF100 (39 settings).
- Original model: each setting's `cifar_teacher` checkpoint trained with supervised
  cross-entropy (seed 41) in `runs/experiment1-multidataset`.
- Repair: the existing channel localization, target/preservation sample sets, channel
  budgets (2, 4, 8), learning rates (0.001, 0.005), 20 epochs, and validation gate.
  The changed objective is **CE(target) + CE(preservation)** with equal weights:
  `kd_weight=0`, `feature_weight=0`, `preservation_ce_weight=1`.
- Selection: retain the current validation rule. If no candidate qualifies, keep the
  original classifier. Do not use the test split to choose a channel budget or learning rate.
- Test: evaluate the original and selected repaired classifier on the same unresampled
  official test split. A rejected repair does not trigger test evaluation.
- Primary descriptive endpoint: change in macro recall over the frozen minority
  `target_classes`, across all 39 settings under the no-change policy.
- Secondary endpoints: selected-repair minority/head macro recall, accuracy, macro-F1,
  and per-class recall regressions. Report class support and selection coverage.

This experiment is exploratory because the original Experiment 1 runs have already
examined these test sets. A fresh independent test source is needed for a confirmatory
claim. The IF groups may select different datasets, so a trend across profiles is
descriptive. Each setting reuses one original model seed, so the study does not
estimate variation from retraining the original classifier.

## Run

From the repository root in the project environment, inspect the 39-job plan:

```bash
python3 scripts/vanilla_imbalance_repair.py --all --dry-run
```

Run one setting first:

```bash
python3 scripts/vanilla_imbalance_repair.py \
  --dataset cifar10 --profile lt-if100 --device cuda:0 --data-root data
```

Run all settings sequentially on a GPU host with the catalog datasets available:

```bash
python3 scripts/vanilla_imbalance_repair.py --all --device cuda:0 --data-root data
```

Each job has its own resumable study directory under
`runs/vanilla-imbalance-repair/DATASET/PROFILE/`. `comparison.json` records the
validation-selected repair; `test_comparison.json` records the direct classifier
test comparison. The runner verifies checkpoint identities and refuses to replace
a differing completed result. `--data-root` stages a copy of the original
classifier config with a local dataset path; it leaves the frozen Experiment 1
artifacts untouched.

After all 39 `test_comparison.json` files exist, render the supervised-only report:

```bash
python3 scripts/imbalance_repair_study.py \
  --vanilla-root runs/vanilla-imbalance-repair \
  --output docs/vanilla-imbalance-repair-results.md
```

## Current status

The 39 settings and their original supervised checkpoints were verified by the
plan dry run. The CIFAR-10 IF100 CE-only specification also passed the study's
full protocol dry run with the local dataset path and zero student runs.
Supervised-only repair and test evaluation have not been executed, so there are no
results to report for this protocol yet.
