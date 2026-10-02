# Full experimental coverage plan

## Required comparison matrix

Use the same 13 datasets and four imbalance factors, $\rho\in\{1,10,50,100\}$,
for every planned teacher--student pairing. This is 52 settings per pairing
and 260 pairing--setting comparisons in total:

| Teacher | Student | Settings required | Current coverage |
| --- | --- | ---: | --- |
| CNN-T | CNN-S | 52 | 52 repair decisions and matched KD comparisons |
| CNN-T | ResNet-8 | 52 | 23 selected-repair comparisons; original-teacher baselines still needed for 29 no-change settings |
| CNN-T | ResNet-20 | 52 | 23 selected-repair comparisons; original-teacher baselines still needed for 29 no-change settings |
| ResNet-18 | CNN-S | 52 | 10 teacher settings and matched student comparisons; 42 settings remain |
| ResNet-34 | CNN-S | 52 | 10 teacher settings and matched student comparisons; 42 settings remain |

Repair selection is performed once per teacher--dataset--profile setting, before
student training. Therefore CNN-T has 52/52 teacher decisions, ResNet-18 has
10/52, and ResNet-34 has 10/52. The remaining **84 ResNet teacher settings**
are untested, not rejected repairs. The 20 existing ResNet settings cover only
CIFAR-10, CIFAR-100, BloodMNIST, EuroSAT, and GTSRB at $\rho=10$ and $100$.

For each pairing and setting, train three matched original-teacher KD students.
If repair is accepted, train three matched repaired-teacher KD students; if it
is rejected, reuse the original-teacher students as the no-change result with
zero repair effect. This gives all pairings the same 52-setting denominator
without training a redundant second student for a rejected repair.

The alternative-loss comparison in Section 4 also needs the same 260
pairing--setting cells, with three matched seeds per cell for LoCa, DKD, RLD,
lower-response DKD, and lower-response RLD. The target is 3,900
alternative-loss student fits before reuse of any verified existing artifacts.
The currently reported alternative-loss results cover the 23 selected
CNN-T--CNN-S settings only; they cannot stand in for this full matrix.

## Execution order

1. **Freeze the matrix and data identities.** Keep the existing
   `runs/experiment1-multidataset/matrix-plan/configs/` files, split
   identities, seeds, training schedules, repair grid, and validation gates.
   Every new architecture job is derived from the corresponding original job
   config; only the teacher architecture, run names, and late-stage search
   space change.
2. **Complete ResNet teacher settings.** On the cloud CUDA machine, from the
   repository root, check the pending jobs and then run both teachers:

   ```bash
   python scripts/teacher_architecture_study.py --plan runs/experiment1-multidataset/matrix-plan --teacher cifar_resnet18 --all-jobs --dry-run
   python scripts/teacher_architecture_study.py --plan runs/experiment1-multidataset/matrix-plan --teacher cifar_resnet34 --all-jobs --dry-run
   python scripts/teacher_architecture_study.py --plan runs/experiment1-multidataset/matrix-plan --teacher cifar_resnet18 --all-jobs --device cuda:0 --data-root data
   python scripts/teacher_architecture_study.py --plan runs/experiment1-multidataset/matrix-plan --teacher cifar_resnet34 --all-jobs --device cuda:0 --data-root data
   ```

   The new `--all-jobs` mode uses all 52 fixed configs and skips existing
   comparisons whose status is `complete` or `no_repair_selected`. Each dry
   run currently reports 42 pending jobs. Re-running these commands resumes
   unfinished work and does not duplicate finished comparisons.
3. **Complete CNN-T student-architecture baselines.** Extend
   `scripts/student_architecture_study.py`, which currently skips
   `no_repair_selected` jobs, to train original-teacher ResNet-8 and ResNet-20
   students for the other 29 settings per architecture. Preserve the three
   matched seeds and use zero repair difference for no-change settings.
   Retain and verify the existing 23 selected-repair comparisons.
4. **Complete alternative KD losses for every pairing.** Extend
   `kd/distillation_baselines.py`, which currently requires a completed
   selected-repair comparison and reads only the primary CNN-T--CNN-S study,
   so it can use the original teacher and matched KD control for every
   pairing--setting cell, including no-change cells. Add the two lower-response
   variants to the same runner. Use identical data, student initialization,
   schedule, temperature, checkpoint rule, and evaluation images within each
   pairing. Do not use test results to choose response weights.
5. **Audit coverage before revising Section 5.** Verify 52 finalized teacher
   decisions for each of the three teacher architectures, 52 original-teacher
   KD comparisons for each of the five pairings, and a repaired or no-change
   outcome for every pairing--setting cell. Verify three matched seeds per
   trained student condition, identical data identities within each matched
   comparison, and all five alternative losses in every cell. Only then
   replace the interim ResNet counts and restricted alternative-loss tables
   with full-matrix results. Aggregate each dataset's four profiles before
   comparing across datasets, as specified in Section 4.3.

The local workspace is macOS without CUDA, so the training commands above
must run on the cloud GPU platform. No missing result is inferred from an
untested setting.

## Channel-budget sensitivity for larger teachers

The main matrix above keeps the existing $n_{\mathrm{edit}}\in\{2,4,8\}$
grid for every teacher. This preserves the protocol used by the completed
settings while allowing architecture to be the only changed factor. It is
also a fixed *absolute* channel budget: the searched late stages contain
192 channels in CNN-T and 768 in either ResNet teacher. The grid therefore
covers about 1.0%, 2.1%, and 4.2% of CNN-T's searched channels, but only
0.26%, 0.52%, and 1.04% of a ResNet's.

Add a **separate, prespecified ResNet budget arm** with
$n_{\mathrm{edit}}\in\{8,16,32\}$, matching the three CNN-T channel
fractions. Use it on the same 52 dataset--profile settings for both
ResNet-18 and ResNet-34, with the same teacher checkpoints, localization
rule, optimization recipe, validation gate, and matched student seeds.
Give this arm its own protocol and output directories; do not overwrite
the existing fixed-budget artifacts or choose the preferred grid from
test-set outcomes. Both ResNets receive the same scaled grid because their
searched stage widths are the same, despite their different total parameter
counts. Record the fraction of searched channels and the fraction of actual
parameter coordinates edited for every selected repair. This additional arm
requires a budget override in `scripts/teacher_architecture_study.py` that
writes new study configs and a separate output root; the current `--all-jobs`
commands above run the fixed-budget main matrix only.

Scaling is a hypothesis, not a guaranteed improvement. In the 20 existing
ResNet localization results, only 18 have at least eight causally retained
channels, 13 have at least 16, and 11 have at least 32. Under the current
feasibility rule, the scaled arm has no candidate in two of those settings.
Report budget feasibility, validation acceptance, teacher changes, and
student transfer for the fixed and scaled arms separately. Treat a rejected
repair as the same no-change outcome in both arms.
