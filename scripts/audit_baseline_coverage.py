"""Separate CE-control completion from full teacher-method benchmark coverage."""
from __future__ import annotations

from datetime import datetime
import csv
import json
from zoneinfo import ZoneInfo

from scripts.compare_established_baselines import generate as compare_baselines, METHODS, WORK
from scripts.summarize_current_experiments import ROOT, NAMES, PROFILES, table


def generate():
    compare_baselines()
    with (WORK / 'established-baseline-comparison.csv').open() as handle:
        history = {r['setting']: r for r in csv.DictReader(handle)}
    with (WORK / 'current-arm-results.csv').open() as handle:
        repair_rows = list(csv.DictReader(handle))
    status = json.loads((WORK / 'baseline-coverage-status.json').read_text())
    extension_controls = set(status['queue']['completed_baselines'])
    original_controls = {r['dataset'] + '-' + r['profile'] for r in repair_rows if r['accepted'] != ''}
    controls = original_controls | extension_controls
    active = (status['queue'].get('current_job') or '').removesuffix('-baseline')
    rows = []
    for dataset in NAMES:
        for profile in PROFILES:
            setting = dataset + '-' + profile
            h = history.get(setting)
            retained = bool(h and h['same_ce_teacher'] == 'True' and h['same_train_val_indices'] == 'True')
            rows.append({'dataset': dataset, 'profile': profile, 'setting': setting,
                'current_ce_controls': 'Complete' if setting in controls else 'Running' if setting == active else 'Queued',
                'historical_eight_method_suite_complete': h is not None,
                'historical_suite_retains_current_ce_teacher_data': retained,
                'historical_full_suite_current_cohort_status': 'retained CE/data; harmonized evaluation pending' if retained
                    else 'different CE cohort; compatibility/model availability needs review' if h else 'no historical full suite',
                'non_ce_methods_not_established_in_current_ce_cohort': '' if retained else ','.join(METHODS[1:])})
    complete_history = sum(r['historical_eight_method_suite_complete'] for r in rows)
    retained = sum(r['historical_suite_retains_current_ce_teacher_data'] for r in rows)
    assert (len(rows), complete_history, retained) == (52, 28, 16)
    stamp = datetime.fromisoformat(status['captured_utc']).astimezone(ZoneInfo('Asia/Kuala_Lumpur'))
    result = {'snapshot_utc': status['captured_utc'], 'planned_settings': 52,
        'historical_full_suite_settings': complete_history, 'retained_current_ce_data_settings': retained,
        'different_ce_historical_settings': complete_history - retained,
        'settings_without_historical_full_suite': 52 - complete_history,
        'current_ce_control_settings_complete': len(controls),
        'current_compatible_full_suite_gaps': 52 - retained,
        'current_queue_recipes': status['authorized_recipes'],
        'definite_never_benchmarked_non_ce_teacher_fits': (52 - complete_history) * 7,
        'definite_never_benchmarked_non_ce_kd_fits': (52 - complete_history) * 7 * 3,
        'upper_bound_non_ce_teacher_fits_if_all_gaps_refit': (52 - retained) * 7,
        'upper_bound_non_ce_kd_fits_if_all_gaps_refit': (52 - retained) * 7 * 3,
        'new_jobs_launched': False}
    display = []
    for dataset in NAMES:
        subset = [r for r in rows if r['dataset'] == dataset]
        display.append([NAMES[dataset], f"{sum(r['current_ce_controls'] == 'Complete' for r in subset)}/4",
            f"{sum(r['historical_eight_method_suite_complete'] for r in subset)}/4",
            f"{sum(r['historical_suite_retains_current_ce_teacher_data'] for r in subset)}/4"])
    display.append(['Total', f'{len(controls)}/52', f'{complete_history}/52', f'{retained}/52'])
    text = ['# Baseline coverage audit', '', f'Audit snapshot: **{stamp:%d %B %Y, %H:%M:%S MYT}**.', '',
        '**The full established-baseline benchmark is not complete.** Previous completion counts referred to CE controls and strict/expanded repair arms. The active continuation fits CE controls and repair; it does not fit the other seven teacher-training methods.', '',
        '## Coverage by dataset', '',
        table(['Dataset', 'Current CE + three KD controls complete', 'Historical eight-method reports complete',
               'Historical suite retaining current CE/data lineage'], display),
        'Each denominator of four represents default, LT-IF10, LT-IF50 and LT-IF100. The ordinary eight-method suite is CE, RandAugment, label smoothing, Balanced Softmax, MSE-logit, MSE-probability, SAM and EMA, each followed by three classical-KD students.', '',
        '“Retaining current CE/data lineage” is narrower than having a historical result. It still does not mean final inference has been harmonized: the legacy baseline checkpoints have not yet been run through the current canonical CPU FP32 pipeline.', '',
        '## Corrections to the comparison table', '',
        '- The previous 22-setting table was a partial extraction. Six historical full suites were present in older saved reports but absent from the repair-only aggregate: CINIC-10 default/IF10 and all four DermaMNIST profiles.',
        '- Those six suites have now been restored from their original comparison reports; all eight method rows have three student scores. Historical full-suite coverage is therefore **28/52**, with **16 retained-current-CE/data settings** and **12 refitted-teacher contexts**.',
        '- The twelve contexts are all CIFAR-10 profiles, CIFAR-100 default/IF100, CINIC-10 default/IF10, and all DermaMNIST profiles. They need model/configuration identity review and a common evaluation before being used as controlled current-cohort comparisons.',
        '- The other **24 settings**—OrganAMNIST, PathMNIST, SVHN, GTSRB, Fashion-MNIST and EuroSAT, four profiles each—have no historical full eight-method suite in these records. Existing/new CE controls for these settings do not fill the seven-method gap.', '',
        '## What the running queue will complete', '',
        f"Current queue phase: `{status['queue']['phase']}`; current job: `{status['queue'].get('current_job')}`. It contains **12 CE teachers, 36 original-teacher KD students and 24 strict/expanded repair arms** for GTSRB/Fashion-MNIST/EuroSAT. It will not automatically add RandAugment, label smoothing, Balanced Softmax, either MSE variant, SAM or EMA.", '',
        'The earlier 80 completed repair arms are a different completion measure. This later coverage audit does not overwrite their timestamped numerical results snapshot.', '',
        '## Remaining work', '',
        '1. Harmonize evaluation of retained baseline students with the repair reports. All 336 non-control R18 checkpoint files for the 16 retained settings were found, with matching saved data/training/KD/initialization metadata. Verify file identities before evaluation.',
        '2. Review the twelve refitted-teacher contexts for compatible saved models and protocols; identify which conditions actually require refitting rather than assuming old scores are interchangeable.',
        '3. Fit and evaluate the missing seven non-CE teacher recipes for the 24 never-benchmarked settings. That is **168 teacher fits + 504 KD student fits = 672 fits**, under a full eight-recipe scope.',
        '4. If all 36 current-cohort gaps require those seven methods, the upper planning workload is **252 teacher fits + 756 KD fits = 1,008 fits**. These counts exclude additional supervised-student references and may shrink if redundant Balanced Softmax arms are explicitly omitted or compatible models are recovered.',
        '5. DKD/RLD/LoCa are a separate student-loss benchmark; their historical CNN results do not complete an R34→R18 comparison.', '',
        'No additional baseline training or evaluation jobs were started by this audit.', '',
        '## Evidence', '',
        '- [Corrected established-baseline score tables](resnet34-resnet18-established-baselines.md).',
        '- [All 52 coverage rows as CSV](../../runs/experiment-results/current-results-20261027/baseline-coverage.csv).',
        '- [Audit receipt](../../runs/experiment-results/current-results-20261027/baseline-coverage-validation.json).',
        '- [Live queue scope/completion snapshot](../../runs/experiment-results/current-results-20261027/baseline-coverage-status.json).',
        '- [Recovered full-suite report metadata](../../runs/experiment-results/current-results-20261027/historical-full-baseline-supplement.json).',
        '- Reproduce using `python -m scripts.audit_baseline_coverage`.', '']
    (ROOT / 'docs/experiments/baseline-coverage-audit-20261027.md').write_text('\n'.join(text))
    with (WORK / 'baseline-coverage.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (WORK / 'baseline-coverage-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(generate(), indent=2))
