"""Create an auditable local results ledger from saved experiment metadata."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'runs/experiment-results'
DEFAULT_INPUT = RESULTS / 'current-results-20261027'
NAMES = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'cinic10': 'CINIC-10',
         'svhn': 'SVHN', 'gtsrb': 'GTSRB', 'fashionmnist': 'Fashion-MNIST',
         'pathmnist': 'PathMNIST', 'bloodmnist': 'BloodMNIST', 'dermamnist': 'DermaMNIST',
         'organamnist': 'OrganAMNIST', 'caltech101': 'Caltech-101', 'eurosat': 'EuroSAT',
         'stl10': 'STL-10'}
PROFILES = ('balanced', 'lt-if10', 'lt-if50', 'lt-if100')
POLICIES = ('strict', 'screened_then_ranked')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classification(metrics):
    """Independently reconstruct macro metrics from each saved confusion matrix."""
    cm = np.asarray(metrics['confusion_matrix'], dtype=np.float64)
    diagonal = np.diag(cm)
    precision = np.divide(diagonal, cm.sum(0), out=np.zeros_like(diagonal), where=cm.sum(0) != 0)
    recall = np.divide(diagonal, cm.sum(1), out=np.zeros_like(diagonal), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(diagonal), where=precision + recall != 0)
    values = {'accuracy': float(diagonal.sum() / cm.sum()),
              'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
              'macro_f1': float(f1.mean()), 'target_recall': metrics['target_macro_recall']}
    assert cm.sum() == metrics['samples']
    assert abs(values['accuracy'] - metrics['accuracy']) < 1e-12
    assert abs(values['macro_f1'] - metrics['macro_f1']) < 1e-12
    assert abs(values['macro_recall'] - metrics['balanced_accuracy']) < 1e-12
    return values


def load_current(directory):
    rows, classification_rows, teacher_rows, identity_rows, control_rows = [], [], [], [], []
    with tarfile.open(directory / 'source-metadata.tar.gz') as archive:
        snapshot = json.load(archive.extractfile('snapshot.json'))
        for campaign in snapshot['campaigns']:
            for record in campaign['baseline_records']:
                prefix = f"{campaign['name']}/baselines/{record['id']}"
                raw = archive.extractfile(prefix + '/comparison.json').read()
                assert sha(raw) == record['source_sha256']['comparison.json']
                report = json.loads(raw)
                assert report['status'] == 'complete'
                for run in report['runs']:
                    metrics = classification(run['metrics'])
                    control_rows.append({'setting': record['id'], 'run': run['name'],
                        'seed': run['seed'], **{k + '_pct': 100 * v for k, v in metrics.items()}})
            for record in campaign['records']:
                prefix = f"{campaign['name']}/studies/{record['id']}"
                artifacts = {}
                for name, digest in record['source_sha256'].items():
                    raw = archive.extractfile(prefix + '/' + name).read()
                    assert sha(raw) == digest
                    artifacts[name] = json.loads(raw)
                report, selection = artifacts['comparison.json'], artifacts['teacher_selection.json']
                protocol = artifacts['protocol.json']
                assert protocol['study']['teacher_model'] == 'resnet34_smallstem'
                assert protocol['study']['student_model'] == 'resnet18_smallstem'
                assert report['protocol_sha256'] == record['source_sha256']['protocol.json']
                assert artifacts['report_manifest.json']['protocol_sha256'] == report['protocol_sha256']
                assert report['status'] in {'complete', 'no_repair_selected'}
                for name in ('comparison.json', 'teacher_selection.json', 'localization.json'):
                    digest = artifacts['report_manifest.json']['files'].get(name)
                    if digest is not None:
                        assert record['source_sha256'][name] == digest
                dataset, profile = record['case'].split('-', 1)
                policy = protocol['study'].get('screening_policy', 'strict')
                selected = selection['selected']
                accepted = selected is not None
                assert accepted == (report['status'] == 'complete')
                row = {'dataset': dataset, 'profile': profile, 'policy': policy,
                       'status': 'Accepted' if accepted else 'Rejected', 'accepted': accepted,
                       'original_teacher_accuracy_pct': None, 'repaired_teacher_accuracy_pct': None,
                       'baseline_accuracy_pct': None, 'baseline_accuracy_sd_pct': None,
                       'repaired_accuracy_pct': None, 'repaired_accuracy_sd_pct': None,
                       'accuracy_delta_pp': None, 'target_recall_delta_pp': None,
                       'macro_f1_delta_pp': None, 'selected_support_pct': None,
                       'practical_success': None, 'campaign': campaign['name'],
                       'job_id': record['id'], 'teacher_sha256': protocol['teacher']['checkpoint_sha256'],
                       'comparison_sha256': record['source_sha256']['comparison.json'],
                       'archive_sha256': record['archive_receipt']['sha256']}
                identity_rows.append({'dataset': dataset, 'profile': profile, 'policy': policy,
                                      'teacher': protocol['teacher']['checkpoint_sha256'],
                                      'students': {k: v['checkpoint_sha256'] for k, v in protocol['student_baselines'].items()}})
                if accepted:
                    assert [x['seed'] for x in report['students']] == [42, 43, 44]
                    original_teacher = classification(report['teacher']['baseline'])
                    repaired_teacher = classification(report['teacher']['candidate'])
                    values = {condition: [classification(x[condition]) for x in report['students']]
                              for condition in ('baseline', 'candidate')}
                    means = {c: {m: statistics.mean(x[m] for x in v) for m in v[0]}
                             for c, v in values.items()}
                    acc_deltas = [c['accuracy'] - b['accuracy'] for b, c in zip(values['baseline'], values['candidate'])]
                    target_deltas = [c['target_recall'] - b['target_recall'] for b, c in zip(values['baseline'], values['candidate'])]
                    for metric, deltas in [('accuracy_delta', acc_deltas), ('target_recall_delta', target_deltas)]:
                        assert np.allclose(deltas, report['aggregate'][metric]['values'], rtol=0, atol=1e-12)
                        assert abs(statistics.mean(deltas) - report['aggregate'][metric]['mean']) < 1e-12
                    success = (statistics.mean(target_deltas) >= .01 and sum(x > 0 for x in target_deltas) >= 2
                               and statistics.mean(acc_deltas) >= -.005)
                    assert success == report['success']['passed']
                    row.update(baseline_accuracy_pct=100 * means['baseline']['accuracy'],
                        original_teacher_accuracy_pct=100 * original_teacher['accuracy'],
                        repaired_teacher_accuracy_pct=100 * repaired_teacher['accuracy'],
                        baseline_accuracy_sd_pct=100 * statistics.stdev(x['accuracy'] for x in values['baseline']),
                        repaired_accuracy_pct=100 * means['candidate']['accuracy'],
                        repaired_accuracy_sd_pct=100 * statistics.stdev(x['accuracy'] for x in values['candidate']),
                        accuracy_delta_pp=100 * statistics.mean(acc_deltas),
                        target_recall_delta_pp=100 * statistics.mean(target_deltas),
                        macro_f1_delta_pp=100 * (means['candidate']['macro_f1'] - means['baseline']['macro_f1']),
                        selected_support_pct=100 * selected['parameter_budget']['achieved_fraction'],
                        practical_success=success)
                    class_row = {k: row[k] for k in ('dataset', 'profile', 'policy')}
                    for condition in ('baseline', 'candidate'):
                        for metric in ('macro_precision', 'macro_recall', 'macro_f1'):
                            class_row[condition + '_' + metric + '_pct'] = 100 * means[condition][metric]
                    class_row['macro_f1_delta_pp'] = row['macro_f1_delta_pp']
                    classification_rows.append(class_row)
                    preservation = selected['validation']['preservation']
                    teacher_rows.append({**{k: row[k] for k in ('dataset', 'profile', 'policy')},
                        'selected_support_pct': row['selected_support_pct'],
                        'channels': len(selected['channels']),
                        'validation_accuracy_delta_pp': 100 * (selected['validation']['accuracy'] - selection['baseline_validation']['accuracy']),
                        'validation_target_delta_pp': 100 * (selected['validation']['target_macro_recall'] - selection['baseline_validation']['target_macro_recall']),
                        'validation_forward_kl': preservation['mean_forward_kl'],
                        'test_teacher_accuracy_delta_pp': 100 * (report['teacher']['candidate']['accuracy'] - report['teacher']['baseline']['accuracy']),
                        'test_student_accuracy_delta_pp': row['accuracy_delta_pp'],
                        'test_student_target_delta_pp': row['target_recall_delta_pp']})
                rows.append(row)
    keyed = {(r['dataset'], r['profile'], r['policy']): r for r in rows}
    assert len(keyed) == len(rows)
    # Check that every completed policy pair actually reused the same four checkpoints.
    pairs = defaultdict(list)
    for item in identity_rows:
        pairs[item['dataset'], item['profile']].append(item)
    for pair in pairs.values():
        if len(pair) == 2:
            assert pair[0]['teacher'] == pair[1]['teacher']
            assert pair[0]['students'] == pair[1]['students']
    extension = next(c for c in snapshot['campaigns'] if 'pending3' in c['name'])
    active_case = (extension['status'].get('current_job') or '').removesuffix('-baseline')
    for dataset in NAMES:
        for profile in PROFILES:
            for policy in POLICIES:
                key = dataset, profile, policy
                if key not in keyed:
                    case_id = f'{dataset}-{profile}'
                    status = 'Awaiting controls' if case_id == active_case else 'Queued'
                    if extension['status'].get('current_job') == f'{case_id}-{policy}':
                        status = 'Running'
                    empty = {k: None for k in rows[0]}
                    empty.update(dataset=dataset, profile=profile, policy=policy, status=status,
                                 campaign=extension['name'], job_id=f'{case_id}-{policy}')
                    keyed[key] = empty
    ordered = [keyed[d, p, s] for d in NAMES for p in PROFILES for s in POLICIES]
    assert len(ordered) == 104
    return snapshot, ordered, classification_rows, teacher_rows, pairs, control_rows


def fmt(value, signed=False, places=3):
    return '—' if value is None or value == '' else format(float(value), ('+' if signed else '') + f'.{places}f')


def label(row):
    return [NAMES[row['dataset']], row['profile'].replace('balanced', 'Default').replace('lt-', '').upper().replace('DEFAULT', 'Default'),
            'Strict' if row['policy'] == 'strict' else 'Expanded']


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |',
                      *['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows]]) + '\n'


def shortcut_repair_context():
    """Keep the earlier two shortcut repair settings outside the class-family matrix."""
    directory = RESULTS / 'repair-completed-20261018/rented/repair-queue/archives'
    rows = []
    for slug, name in [('colored-mnist', 'Colored MNIST'), ('waterbirds', 'Waterbirds')]:
        path = directory / (slug + '-results-with-predictions.tar.gz')
        with tarfile.open(path) as archive:
            report = json.load(archive.extractfile('study/comparison.json'))
            config = json.load(archive.extractfile('study/teacher_control_seed41/config.json'))
            control = next(x for x in report['rows'] if x['recipe'] == 'control')
            repair = next(x for x in report['rows'] if x['recipe'] == 'teacher_repair')
            outcomes = list(repair['repair_outcomes'].values())
            selected = any(x.get('selected_candidate') is not None for x in outcomes)
            rows.append([name, report['endpoint'], config['data']['image_size'],
                fmt(100 * control['student']['accuracy']['mean']),
                fmt(100 * control['student']['worst_group_accuracy']['mean']),
                'Accepted' if selected else 'No change; control alias',
                fmt(100 * repair['student']['accuracy']['mean']) if selected else '—',
                fmt(100 * repair['paired_student_delta_vs_control']['accuracy']['mean'], True) if selected else '—'])
    return rows


def generate(directory=DEFAULT_INPUT):
    directory = Path(directory).resolve()
    snapshot, rows, classification_rows, teacher_rows, pairs, control_rows = load_current(directory)
    accepted = [r for r in rows if r['accepted'] is True]
    complete = [r for r in rows if r['accepted'] is not None]
    success = [r for r in accepted if r['practical_success']]
    stamp = datetime.fromisoformat(snapshot['captured_utc']).astimezone(ZoneInfo('Asia/Kuala_Lumpur'))
    historical = json.loads((RESULTS / 'normalized-repair-20261025/results.json').read_text())
    older = historical['previous_fixed_channel']
    supplement = json.loads((directory / 'historical-full-baseline-supplement.json').read_text())
    methods = [r for r in read_csv(RESULTS / 'normalized-repair-20261025/previous-teacher-methods-context.csv')
               if r['method'] != 'teacher_repair' and r['setting'] not in supplement]
    methods += [{'setting': setting, **recipe} for setting, report in supplement.items()
                for recipe in report['recipes']]
    original_architectures = read_csv(RESULTS / 'repaired-teacher-analysis-20261017/comparisons.csv')
    path = ROOT / 'docs/experiments/kd-baseline-current-results.md'
    improvement_rows = []
    for policy, name in [('strict', 'Strict'), ('screened_then_ranked', 'Expanded'), (None, 'Combined')]:
        subset = [r for r in accepted if policy is None or r['policy'] == policy]
        improved = sum(r['accuracy_delta_pp'] > 1e-10 for r in subset)
        decreased = sum(r['accuracy_delta_pp'] < -1e-10 for r in subset)
        unchanged = len(subset) - improved - decreased
        improvement_rows.append([name, len(subset), improved, decreased, unchanged,
            f'{100 * improved / len(subset):.1f}%' if subset else '—',
            sum(r['target_recall_delta_pp'] > 1e-10 for r in subset),
            sum(r['macro_f1_delta_pp'] > 1e-10 for r in subset),
            sum(r['practical_success'] for r in subset)])
    sections = ['# KD baseline: ResNet-34 → ResNet-18 performance and historical results', '',
        f"Snapshot: **{stamp:%d %B %Y, %H:%M:%S MYT}**. Values are copied or independently reconstructed from saved results; running experiments are not treated as completed results.", '',
        f"The current strict/expanded campaign has **{len(complete)}/104 completed policy arms**, including the four pilot arms. **{len(accepted)}** selected a repaired teacher, **{len(complete)-len(accepted)}** selected none, and **{len(success)} policy arms across {len({(x['dataset'], x['profile']) for x in success})} settings** met the saved practical student-success criterion. " + (f"The remaining {104-len(complete)} arms belong to the GTSRB/Fashion-MNIST/EuroSAT continuation." if len(complete) < 104 else 'All 13 datasets and four profiles have completed both screening policies.'), '',
        '**These are repair-arm completion counts, not completion of the full established-baseline benchmark.** See the [baseline coverage audit](baseline-coverage-audit-20261027.md) for the separate CE-control, historical full-suite, and current teacher-matched coverage. The GTSRB/Fashion-MNIST/EuroSAT continuation covers CE controls and repair only; the separate seven-method missing-baseline queue is documented in the [ascending baseline plan](ascending-baselines-20261027.md).', '',
        '## ResNet-34 teacher → ResNet-18 student: accepted-repair summary', '',
        'This first table includes only the current ImageNet-initialized, adapted ResNet-34 → ResNet-18 experiments. An arm is one dataset/profile/screening-policy comparison, averaged over three paired student seeds. Improvement rates use **accepted repair arms** as their denominator; they are not counts of individual students or independent teacher replications.', '',
        table(['Policy', 'Accepted repairs', 'Accuracy improved', 'Accuracy decreased', 'Accuracy unchanged',
               'Accuracy improvement rate', 'Target recall improved', 'Macro-F1 improved', 'Practical successes'], improvement_rows),
        '[Compare this pairing with the established teacher-training baselines](resnet34-resnet18-established-baselines.md): CE, Balanced Softmax, RandAugment, SAM, EMA, label smoothing and both MSE variants. That report separates retained-CE comparisons from refitted-teacher context and records the remaining inference-pipeline mismatch.', '',
        '## ResNet-34 → ResNet-18: performance of every accepted repair', '',
        'Teacher columns are test accuracy of the single original/repaired ResNet-34. Student columns are mean ± sample SD test accuracy of the three ResNet-18 KD students. Student gains compare KD from the repaired teacher against paired KD from the original teacher; changes are percentage points. These teachers were accepted using validation gates before test evaluation.', '',
        table(['Dataset', 'Profile', 'Policy', 'Original R34 accuracy %', 'Repaired R34 accuracy %',
               'Original-teacher R18 KD accuracy %', 'Repaired-teacher R18 KD accuracy %',
               'R18 accuracy Δ pp', 'R18 target recall Δ pp'],
              [[*label(r), fmt(r['original_teacher_accuracy_pct']), fmt(r['repaired_teacher_accuracy_pct']),
                f"{fmt(r['baseline_accuracy_pct'])} ± {fmt(r['baseline_accuracy_sd_pct'])}",
                f"{fmt(r['repaired_accuracy_pct'])} ± {fmt(r['repaired_accuracy_sd_pct'])}",
                fmt(r['accuracy_delta_pp'], True), fmt(r['target_recall_delta_pp'], True)] for r in accepted]),
        '## Reading the tables', '',
        '- Current models: ImageNet-initialized adapted ResNet-34 teacher → ResNet-18 student, RGB 32×32. One fixed teacher seed 41 and three paired student seeds 42/43/44 per setting.',
        '- Accuracy and classification scores are percentages. Δ values are **percentage points**, repaired-teacher KD minus original-teacher KD within that exact run. Accuracy is mean ± sample SD over the three student seeds.',
        '- Target recall means macro recall over the declared target classes; it is different from all-class macro recall on long-tail profiles.',
        '- **Rejected** means no teacher candidate passed selection; no repaired-teacher students were trained. Dashes are unavailable/not applicable, never measured zero gains. Queued/running arms have no final test result.',
        '- Practical success requires mean target-recall gain ≥1 pp, positive target-recall gain in at least two seeds, and mean accuracy loss ≤0.5 pp. It is **not a statistical-significance test**.',
        '- The latest strict/expanded comparisons share the exact original teacher and three KD checkpoint hashes. Historical runs can use different teachers, architectures, masks, schedules or inference pipelines; their values are shown in separate tables and are not pooled.', '',
        'Historical families can reuse models and reference predictions. Their coverage counts are not additive counts of independent trained models or statistical replications.', '',
        '## Experiment families and available records', '']
    family_rows = [
        ['Current paired strict/expanded repair', f'{len(complete)}/104 arms', f'{len(accepted)} accepted; {len(success)} practical successes', 'Full current tables below'],
        ['Normalized strict repair, 25 October', '40/40 settings', f"{sum(x['accepted'] for x in historical['new'])} accepted; {sum(bool(x.get('success')) for x in historical['new'])} practical successes", '[Verified backup](../../runs/experiment-results/normalized-repair-20261025/README.md)'],
        ['Earlier fixed-channel modern ResNet repair', '28 settings in saved aggregate', f"{sum(x['accepted'] for x in older.values())} accepted", 'Historical table below; missing entries stay unavailable'],
        ['Historical teacher-training method comparisons', f"{len(set(x['setting'] for x in methods))} settings; {len(methods)} ordinary method rows", 'Eight teacher recipes; 16 retained-current-CE settings and 12 refitted-teacher contexts', 'Historical accuracy table below; full current benchmark remains incomplete'],
        ['Original architecture/repair experiments', f'{len(original_architectures)} arms across five families', 'CNN-T/CNN-S and ResNet-teacher/CNN-S families; separate protocols', '[Full source comparisons](../../runs/experiment-results/repaired-teacher-analysis-20261017/comparisons.csv)'],
        ['Original CNN student-loss comparisons', '23 accepted-repair settings; KD/DKD/RLD/LoCa complete', 'Separate architecture/training protocol', '[Full accuracy and target-recall tables](../distillation-baselines-results.md)'],
        ['Earlier shortcut benchmarks', 'Colored MNIST and Waterbirds records available', 'Group/OOD endpoints; not part of the 13-dataset screening matrix', '[Colored MNIST](../../runs/experiment-results/colored-mnist-bf16-20261016/README.md); [earlier report](../../runs/experiment-results/balanced-softmax-significance-20261017/README.md)'],
        ['MultiCelebA', 'Paused; model backup retained', 'No completed full benchmark claimed', '[Saved state](../../runs/experiment-results/multiceleba-models-paused-bf16-20261016/README.md)'],
        ['Per-image and held-out retention audits', 'BloodMNIST and STL-10 saved audits', 'Diagnostics rather than additional independent training replications', '[BloodMNIST](../../runs/experiment-results/bloodmnist-per-image-20261021/README.md); [retention](../../runs/experiment-results/bloodmnist-repair-retention-20261021/README.md); [STL-10](../../runs/experiment-results/stl10-repair-audit-20261022/README.md)']]
    sections += [table(['Family', 'Coverage', 'Interpretation', 'Detailed record'], family_rows),
                 '## Current coverage by dataset', '']
    coverage = []
    for dataset in NAMES:
        group = [r for r in rows if r['dataset'] == dataset]
        counts = []
        for policy in POLICIES:
            subset = [r for r in group if r['policy'] == policy]
            counts += [f"{sum(r['accepted'] is not None for r in subset)}/4",
                       sum(r['accepted'] is True for r in subset), sum(r['practical_success'] is True for r in subset)]
        coverage.append([NAMES[dataset], *counts])
    sections += [table(['Dataset', 'Strict done', 'Strict accepted', 'Strict successes', 'Expanded done', 'Expanded accepted', 'Expanded successes'], coverage),
                 f"Across the {len(pairs)} completed settings, strict selected {sum(r['policy'] == 'strict' for r in accepted)} teachers and met the practical rule in {sum(r['policy'] == 'strict' for r in success)} arms; expanded selected {sum(r['policy'] != 'strict' for r in accepted)} teachers and met it in {sum(r['policy'] != 'strict' for r in success)} arms. These are paired descriptive counts, not independent policy replications.", '',
                 '## New continuation: original CE/KD controls', '',
                 (table(['Setting', 'Run', 'Seed', 'Accuracy %', 'Target recall %', 'Macro P %', 'Macro R %', 'Macro F1 %'],
                        [[r['setting'], r['run'], r['seed'], *[fmt(r[k + '_pct']) for k in
                          ['accuracy', 'target_recall', 'macro_precision', 'macro_recall', 'macro_f1']]] for r in control_rows])
                  if control_rows else 'No new control family had completed its final test evaluation and verified archive at snapshot time. The first GTSRB LT-IF50 control family was training. Partial epoch/validation scores are not final results.'), '',
                 '## Complete ResNet-34 → ResNet-18 matrix: all planned arms', '',
                 'The accepted-arm accuracy results are tabulated first above. This ledger also records rejected and unfinished arms. Rejected arms do not supply a new canonical test evaluation; no earlier score is substituted. Default denotes the configured native/default profile, not a guarantee of naturally equal class frequencies.', '']
    current_table = []
    for r in rows:
        current_table.append([*label(r), r['status'], fmt(r['selected_support_pct']),
            'Yes' if r['practical_success'] is True else 'No' if r['practical_success'] is False else '—'])
    sections += [table(['Dataset', 'Profile', 'Policy', 'Decision/state', 'Editable %', 'Practical success'], current_table),
        '## Current student precision, recall and F1', '',
        'Macro precision/recall/F1 are independently recomputed from each saved per-seed test confusion matrix and then averaged over seeds. They are not computed by pooling seeds or datasets. All-class macro recall equals balanced accuracy here. Rows exist only for accepted repairs with completed downstream students.', '',
        table(['Dataset', 'Profile', 'Policy', 'Original macro P %', 'Original macro R %', 'Original macro F1 %',
               'Repaired macro P %', 'Repaired macro R %', 'Repaired macro F1 %', 'F1 Δ pp'],
              [[*label(r), *[fmt(r[c + '_' + m + '_pct']) for c in ('baseline', 'candidate')
                             for m in ('macro_precision', 'macro_recall', 'macro_f1')], fmt(r['macro_f1_delta_pp'], True)]
               for r in sorted(classification_rows, key=lambda x: (list(NAMES).index(x['dataset']), PROFILES.index(x['profile']), POLICIES.index(x['policy'])))]),
        '## Accepted-teacher diagnostics and downstream effects', '',
        'Validation changes and KL describe the selected teacher, while test teacher/student changes are post-selection outcomes. Editable support is the actual union of parameter coordinates, not the nominal cap. Passing teacher gates does not guarantee a student gain.', '',
        table(['Dataset', 'Profile', 'Policy', 'Editable %', 'Channels', 'Validation teacher accuracy Δ pp',
               'Validation teacher target Δ pp', 'Validation forward KL', 'Test teacher accuracy Δ pp', 'Test student accuracy Δ pp'],
              [[*label(r), fmt(r['selected_support_pct']), r['channels'], fmt(r['validation_accuracy_delta_pp'], True),
                fmt(r['validation_target_delta_pp'], True), fmt(r['validation_forward_kl'], places=6),
                fmt(r['test_teacher_accuracy_delta_pp'], True), fmt(r['test_student_accuracy_delta_pp'], True)] for r in teacher_rows]),
        '## Current arms meeting the practical-success criterion', '',
        table(['Dataset', 'Profile', 'Policy', 'Student accuracy Δ pp', 'Target recall Δ pp', 'Macro-F1 Δ pp'],
              [[*label(r), fmt(r['accuracy_delta_pp'], True), fmt(r['target_recall_delta_pp'], True), fmt(r['macro_f1_delta_pp'], True)] for r in success]),
        '', 'These are exploratory, conditional-on-teacher results across many inspected settings. Three paired student seeds do not establish robustness across independently fitted teachers. No new p-values or significance claims are introduced in this report.', '',
        '## Historical normalized strict repair: all 40 settings', '',
        'This is the earlier 25 October campaign, not an additional policy arm of the current experiment. Saved controls can differ numerically because the earlier evaluation pipeline was not identical. The values below retain their original within-run references.', '']
    historical_table = []
    for r in historical['new']:
        ds, profile = r['setting'].split('-', 1)
        a = r.get('accuracy', {})
        historical_table.append([NAMES[ds], profile, 'Accepted' if r['accepted'] else 'Rejected',
            fmt(a.get('baseline')), fmt(a.get('repair')), fmt(a.get('delta_pp'), True),
            fmt(r.get('target_macro_recall', {}).get('delta_pp'), True),
            fmt(r.get('macro_f1', {}).get('delta_pp'), True),
            'Yes' if r.get('success') is True else 'No' if r.get('success') is False else '—'])
    sections += [table(['Dataset', 'Profile', 'Decision', 'Original KD accuracy %', 'Repaired KD accuracy %',
        'Accuracy Δ pp', 'Target recall Δ pp', 'Macro-F1 Δ pp', 'Practical success'], historical_table),
        '## Historical fixed-channel modern ResNet repair', '',
        'The 28-setting aggregate preserves earlier control/repair comparisons. Some controls were later refitted, so an earlier positive result and a later rejection are not necessarily a matched-teacher reversal. Practical-success flags are not reconstructed for this older summary.', '']
    fixed_table = []
    for setting, r in sorted(older.items()):
        ds, profile = setting.split('-', 1)
        a = r.get('accuracy', {})
        fixed_table.append([NAMES[ds], profile, 'Accepted' if r['accepted'] else 'Rejected',
            fmt(a.get('baseline')), fmt(a.get('repair')), fmt(a.get('delta_pp'), True),
            fmt(r.get('target_macro_recall', {}).get('delta_pp'), True)])
    sections += [table(['Dataset', 'Profile', 'Decision', 'Original KD accuracy %', 'Repaired KD accuracy %',
                       'Accuracy Δ pp', 'Target recall Δ pp'], fixed_table),
        '## Historical teacher-training methods: downstream KD accuracy', '',
        'Cells are saved mean student test accuracy (%) from the earlier teacher-method context CSV and six recovered full-suite reports. Each method trains/changes a teacher before ordinary KD. These historical values are **not uniformly matched comparisons against the current strict/expanded cohorts**; verify teacher/data/evaluation identity before interpreting differences. Retired Mixup/CutMix/combined-augmentation results remain in the linked earlier archives.', '']
    method_names = ['control', 'randaugment', 'label_smoothing', 'balanced_softmax', 'mse_logits', 'mse_probabilities', 'sam', 'ema']
    by_setting = defaultdict(dict)
    for r in methods:
        if r['method'] in method_names:
            by_setting[r['setting']][r['method']] = float(r['accuracy'])
    sections += [table(['Setting', 'CE', 'RandAugment', 'Label smoothing', 'Balanced Softmax', 'MSE logits', 'MSE probabilities', 'SAM', 'EMA'],
                       [[setting, *[fmt(v.get(m)) for m in method_names]] for setting, v in sorted(by_setting.items())]),
        '## Original architecture experiments and other completed baselines', '',
        'The historical architecture experiments use their own teacher/student architectures, initialization and schedules. Their saved practical-success counts are descriptive and must not be compared as isolated architecture effects.', '']
    architecture_table = []
    for arm in dict.fromkeys(x['arm'] for x in original_architectures):
        group = [x for x in original_architectures if x['arm'] == arm]
        architecture_table.append([arm, len(group), sum(x['status'] == 'complete' for x in group),
            sum(x['status'] == 'no_repair_selected' for x in group),
            sum(x['transfer_rule_passed'].lower() == 'true' for x in group)])
    sections += [table(['Historical family', 'Settings', 'Accepted repairs', 'No repair selected', 'Saved practical successes'], architecture_table),
        '[All 260 architecture-setting outcomes and saved statistical fields](../../runs/experiment-results/repaired-teacher-analysis-20261017/comparisons.csv). '
        '[Original CNN KD/DKD/RLD/LoCa accuracy and target-recall tables](../distillation-baselines-results.md) contain all 23 accepted-teacher settings; the remaining 29 settings did not select a repaired teacher.', '',
        '## Shortcut benchmarks and held-out diagnostics', '',
        'Colored MNIST reports distinct OOD endpoints; its worst-group accuracy is not interchangeable with target recall or the standard-dataset test accuracy above. Waterbirds and other earlier teacher-method classification results are retained in the separate eight-study classification table.', '']
    colored = read_csv(RESULTS / 'colored-mnist-bf16-20261016/results-summary.csv')
    sections += [table(['Colored MNIST endpoint', 'Teacher recipe', 'Teacher accuracy %', 'Teacher worst-group %',
                       'KD student accuracy %', 'KD student worst-group %', 'Worst-group Δ vs CE pp'],
        [[r['endpoint'], r['recipe'], fmt(r['teacher_accuracy_pct']), fmt(r['teacher_wga_pct']),
          fmt(r['student_accuracy_mean_pct']), fmt(r['student_wga_mean_pct']),
          fmt(r['paired_wga_delta_vs_control_pp'], True)] for r in colored]),
        '[All earlier eight-study teacher-method student classification metrics, including Waterbirds and retired recipes](../../runs/experiment-results/balanced-softmax-significance-20261017/student-classification-metrics.md).', '',
        'The separately fitted October 18 shortcut repair controls have their own scores, shown below. Both saved repair outcomes are control aliases; no additional repaired-teacher student scores are claimed. Their CE scores must not be substituted for the earlier shortcut benchmark controls above.', '',
        table(['Shortcut repair setting', 'Endpoint', 'Input size', 'Original KD accuracy %',
               'Original KD worst-group %', 'Repair outcome', 'New repaired KD accuracy %', 'Accuracy Δ pp'], shortcut_repair_context()),
        '[Verified shortcut repair backup](../../runs/experiment-results/repair-completed-20261018/README.md).', '',
        table(['Diagnostic', 'Saved finding', 'Scope'], [
            ['BloodMNIST LT-IF10 held-out repair retention', '1 teacher-fixed image; all six students already correct; retention lift 0 pp for seeds 42/43/44', '1,044 held-out images after recorded exclusions; tiny fixed-set denominator and ceiling prevent a transfer-failure conclusion'],
            ['STL-10 fixed-channel repair audit', 'Candidate-level and per-image audit retained', '[Full audit](../../runs/experiment-results/stl10-repair-audit-20261022/README.md); separate from current parameter-normalized screening'],
            ['MultiCelebA', 'Paused/resumable models retained; no complete final table', '[Model inventory](../../runs/experiment-results/multiceleba-models-paused-bf16-20261016/README.md)']]),
        '## Provenance and reproducibility', '',
        '- Current metadata snapshot: [source-metadata.tar.gz](../../runs/experiment-results/current-results-20261027/source-metadata.tar.gz). This small snapshot contains JSON reports/provenance, not trained models, datasets or prediction arrays.',
        '- [All 104 current-arm rows as CSV](../../runs/experiment-results/current-results-20261027/current-arm-results.csv), [accepted-student classification CSV](../../runs/experiment-results/current-results-20261027/current-student-classification.csv), and [teacher diagnostics CSV](../../runs/experiment-results/current-results-20261027/current-teacher-diagnostics.csv).',
        '- [Validation receipt](../../runs/experiment-results/current-results-20261027/report-validation.json): source hashes, checkpoint sharing, saved-vs-recomputed accuracy/F1, paired deltas, and practical-success flags were checked.',
        '- Current source metadata was read from the completed pilot, completed 76-arm matrix, and running three-dataset continuation. Full current prediction archives remain on the GPU server except the previously backed-up pilot; this metadata-only report is not a full prediction backup.',
        '- Historical inputs: [normalized results JSON](../../runs/experiment-results/normalized-repair-20261025/results.json), [teacher-method context CSV](../../runs/experiment-results/normalized-repair-20261025/previous-teacher-methods-context.csv), and the linked verified archives/reports.',
        '- [Six recovered historical full-suite reports](../../runs/experiment-results/current-results-20261027/historical-full-baseline-supplement.json) fill the prior table omissions for CINIC-10 default/IF10 and all DermaMNIST profiles.',
        '- Regenerate after refreshing the snapshot: `python -m scripts.summarize_current_experiments --input runs/experiment-results/current-results-20261027`.',
        '- Generating this report does not train models, change thresholds, stop the queue, or delete files.', '']
    path.write_text('\n'.join(sections))
    write_csv(directory / 'current-arm-results.csv', rows)
    write_csv(directory / 'current-student-classification.csv', classification_rows)
    write_csv(directory / 'current-teacher-diagnostics.csv', teacher_rows)
    if control_rows:
        write_csv(directory / 'new-control-results.csv', control_rows)
    receipt = {'snapshot_utc': snapshot['captured_utc'], 'snapshot_sha256': sha((directory / 'source-metadata.tar.gz').read_bytes()),
        'planned_arms': len(rows), 'completed_arms': len(complete), 'accepted_repairs': len(accepted),
        'rejected_repairs': len(complete) - len(accepted), 'practical_successes': len(success),
        'student_metric_panels_recomputed_from_confusion_matrices': 6 * len(accepted),
        'teacher_metric_panels_recomputed_from_confusion_matrices': 2 * len(accepted),
        'completed_setting_pairs_with_matching_original_checkpoint_hashes': sum(len(x) == 2 for x in pairs.values()),
        'all_checks_passed': True, 'report': str(path.relative_to(ROOT))}
    (directory / 'report-validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    print(json.dumps(generate(args.input), indent=2))
