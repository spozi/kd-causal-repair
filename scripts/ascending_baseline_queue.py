"""Serial missing-method benchmark; wait for the existing repair queue, reuse CE."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RECIPES = ('randaugment', 'label_smoothing', 'balanced_softmax', 'mse_logits',
           'mse_probabilities', 'sam', 'ema')


class LowDisk(RuntimeError):
    pass


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def storage(plan):
    if plan['disk_free_minimum_gib'] < 12:
        raise ValueError('Retained-model queue requires at least a 12 GiB guard')
    if shutil.disk_usage(ROOT).free < plan['disk_free_minimum_gib'] * 1024**3:
        raise LowDisk('Less than 12 GiB free; retain all files and stop for storage expansion')


def verified_plan():
    manifest = read(ROOT / 'manifest.json')
    for name, digest in manifest.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or (ROOT / rel).is_symlink() or sha(ROOT / rel) != digest:
            raise ValueError(f'Frozen input changed: {name}')
    plan = read(ROOT / 'experiment.json')
    if Path(plan['remote_root']).resolve() != ROOT.resolve():
        raise ValueError('Run at the frozen remote root')
    for name, digest in plan['scientific_source_sha256'].items():
        if manifest.get(name) != digest:
            raise ValueError('Scientific source differs from the active comparison snapshot')
    if tuple(plan['recipes']) != RECIPES or len(plan['cases']) != 24:
        raise ValueError('Unexpected recipe/setting scope')
    if len({x['id'] for x in plan['cases']}) != 24:
        raise ValueError('Duplicate settings')
    expected_order = sorted(plan['dataset_order'], key=lambda d: (plan['default_train_samples'][d], d))
    if plan['dataset_order'] != expected_order:
        raise ValueError('Datasets are not in ascending configured training size')
    if [c['dataset'] for c in plan['cases']] != [d for d in expected_order for _ in range(4)]:
        raise ValueError('Case order differs from the declared dataset order')
    for case in plan['cases']:
        output = Path(case['output']).resolve()
        if not output.is_relative_to(ROOT) or output == ROOT or Path(case['baseline']).resolve().is_relative_to(ROOT):
            raise ValueError('New outputs and reused CE controls must be separate')
    return plan


def helpers():
    import control_helpers
    control_helpers.verify_snapshot(ROOT)
    return control_helpers


def check_case(case, *, ready_controls=False):
    helper = helpers()
    teacher, controls, spec = helper.configurations(ROOT, case)
    data = helper.validate_data(ROOT, case, teacher)
    for key in helper.DATA_KEYS:
        if key in case['expected_data'] and data.provenance.get(key) != case['expected_data'][key]:
            raise ValueError(f'Original dataset identity changed: {case["id"]}/{key}')
    raw = read(ROOT / case['config'])
    if tuple(x['recipe'] for x in raw['conditions']) != RECIPES:
        raise ValueError('Condition list changed')
    from kd.config import from_dict
    for condition in raw['conditions']:
        candidate = from_dict(condition['teacher'])
        wanted = teacher.to_dict()
        wanted.update(name=f'teacher_{condition["recipe"]}_seed41', output_dir=case['output'])
        if candidate.to_dict() != wanted:
            raise ValueError('Method teacher differs beyond its declared recipe/output')
        if len(condition['students']) != 3:
            raise ValueError('Every recipe needs all three KD students')
        for student, control in zip(condition['students'], controls):
            actual = from_dict(student)
            expected = control.to_dict()
            expected.update(name=f'kd_{condition["recipe"]}_teacher41_seed{control.train.seed}', output_dir=case['output'])
            expected['teacher']['checkpoint'] = str(Path(case['output']) / candidate.name / 'student.pt')
            if actual.to_dict() != from_dict(expected).to_dict():
                raise ValueError('Method KD student data/schedule/initialization policy differs')
    if ready_controls:
        from kd.config import from_dict
        original = Path(case['baseline'])
        for cfg in [teacher, *controls]:
            recorded = from_dict(read(original / cfg.name / 'config.json'))
            if recorded.to_dict() != cfg.to_dict():
                raise ValueError('Reused CE configuration differs from the frozen inventory')
        if case['expected_teacher_sha256'] and sha(original / teacher.name / 'student.pt') != case['expected_teacher_sha256']:
            raise ValueError('Previously available CE teacher checkpoint changed')
        helper.verify_baseline(ROOT, case, create_pins=True)
    return {'id': case['id'], 'train_samples': len(data.train.dataset),
            'validation_samples': len(data.val.dataset), 'methods': 7,
            'controls_ready_checked': ready_controls}


def dependency_state(plan):
    dependency = plan['dependency']
    path = Path(dependency['root']) / 'status.json'
    if not path.is_file():
        raise RuntimeError('Dependency queue status is missing; inspect before launch')
    state = read(path)
    if state['phase'] == 'complete':
        if len(state.get('completed_baselines', [])) != 12 or len(state.get('completed', [])) != 24:
            raise ValueError('Dependency completion ledger is incomplete')
        return True
    if state['phase'] == 'failed':
        raise RuntimeError('Existing CE/repair campaign failed; do not bypass it')
    active = subprocess.check_output(['systemctl', 'show', dependency['unit'], '-p', 'ActiveState', '--value'], text=True).strip()
    if active not in {'active', 'activating'}:
        raise RuntimeError('Dependency is incomplete and inactive; manual recovery required')
    return False


def status(phase, completed, current=None, **extra):
    write(ROOT / 'status.json', {'phase': phase, 'completed': list(completed),
        'current_job': current, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra})


def run_case(case_id):
    plan = verified_plan()
    if not dependency_state(plan):
        raise RuntimeError('CE/repair dependency has not completed')
    case = next(c for c in plan['cases'] if c['id'] == case_id)
    helper = helpers()
    helper.environment(ROOT, plan, plan['device'])
    check_case(case, ready_controls=True)
    teacher, controls, spec = helper.configurations(ROOT, case)
    from kd.checkpoints import write_json
    from kd.config import from_dict
    from kd.data import build_data
    from kd.neuron_surgery_study import (_complete_student, resolve_target_classes,
                                         _model, _save_predictions, _quality_report)
    from kd.teacher_training import train_teacher, RECIPES as recipe_specs
    from dataclasses import replace
    import numpy as np
    import torch
    import archive_helpers

    output = Path(case['output'])
    protocol = {'version': 1, 'case': case, 'conditions': read(ROOT / case['config'])['conditions'],
        'scientific_source_sha256': plan['scientific_source_sha256'],
        'pretrained_weight_sha256': plan['pretrained_weight_sha256'],
        'ce_pin_sha256': sha(ROOT / 'baseline-pins' / (case_id + '.json')),
        'evaluation': 'CPU FP32; all selected models frozen before test loading'}
    if output.exists() and any(output.iterdir()) and not (output / 'protocol.json').exists():
        raise ValueError('Nonempty study has no frozen protocol')
    helper.freeze(output / 'protocol.json', protocol)
    protocol_sha = sha(output / 'protocol.json')
    final = output / 'completion.json'
    if final.exists():
        completed = read(final)
        if completed['protocol_sha256'] != protocol_sha:
            raise ValueError('Completed study belongs to another protocol')
        for path, digest in completed['selected_models'].items():
            if sha(path) != digest:
                raise ValueError('Retained selected model changed')
        for receipt in completed['archives']:
            if sha(receipt['archive']) != receipt['sha256']:
                raise ValueError('Completed results archive changed')
        return completed
    baseline = Path(case['baseline'])
    ce_teacher_summary = read(baseline / teacher.name / 'summary.json')
    ce_student_summaries = {c.train.seed: read(baseline / c.name / 'summary.json') for c in controls}
    all_runs = {'control': [teacher, *controls]}
    selected_models = {str(baseline / c.name / 'student.pt'): sha(baseline / c.name / 'student.pt') for c in all_runs['control']}
    progress = output / 'progress.json'
    try:
        for condition in protocol['conditions']:
            recipe = condition['recipe']
            cfg = from_dict(condition['teacher'])
            storage(plan)
            write(progress, {'phase': 'teacher_training', 'recipe': recipe, 'run': cfg.name})
            summary = train_teacher(cfg, recipe_specs[recipe], selection_metric='accuracy')
            if summary['initial_student_sha256'] != ce_teacher_summary['initial_student_sha256']:
                raise ValueError('Teacher initialization differs from CE')
            for key in ('source', 'train_index_sha256', 'val_index_sha256', 'train_per_class', 'val_per_class', 'recipe_sha256'):
                if summary['data'].get(key) != ce_teacher_summary['data'].get(key):
                    raise ValueError(f'Teacher data identity differs from CE: {key}')
            runs = [cfg]
            for raw in condition['students']:
                config = from_dict(raw)
                storage(plan)
                write(progress, {'phase': 'kd_training', 'recipe': recipe, 'run': config.name})
                result = _complete_student(config, protocol_sha)
                reference = ce_student_summaries[config.train.seed]
                if result['initial_student_sha256'] != reference['initial_student_sha256'] or result['data'] != reference['data']:
                    raise ValueError('KD initialization/data differs from paired CE control')
                runs.append(config)
            all_runs[recipe] = runs
            for config in runs:
                path = output / config.name / 'student.pt'
                selected_models[str(path)] = sha(path)
        helper.verify_baseline(ROOT, case)
        diagnostic = helper.validate_data(ROOT, case, teacher)
        targets = list(resolve_target_classes(spec, diagnostic.provenance['train_per_class']))
        helper.freeze(output / 'selection.json', {'protocol_sha256': protocol_sha,
            'selected_models': selected_models, 'target_classes': targets,
            'selection': 'Each teacher/student selected on validation accuracy; all recipes retained'})
        # Every method's model choices and targets are now immutable, before test access.
        write(progress, {'phase': 'evaluation', 'models_frozen': True})
        torch.set_num_threads(teacher.train.threads)
        data = build_data(teacher.data, replace(teacher.train, workers=0, device='cpu'),
                          include_test=True, diagnostic=True)
        report_rows, reference_metrics, reference_arrays = [], {}, {}
        for recipe, configs in all_runs.items():
            report_dir = output / 'reports' / recipe
            method_protocol = {'case_protocol_sha256': protocol_sha, 'recipe': recipe,
                'models': {c.name: {'checkpoint': str(Path(c.output_dir) / c.name / 'student.pt'),
                                  'sha256': selected_models[str(Path(c.output_dir) / c.name / 'student.pt')]}
                           for c in configs}, 'target_classes': targets}
            helper.freeze(report_dir / 'protocol.json', method_protocol)
            method_sha = sha(report_dir / 'protocol.json')
            measurements = []
            for config in configs:
                checkpoint = Path(config.output_dir) / config.name / 'student.pt'
                if sha(checkpoint) != selected_models[str(checkpoint)]:
                    raise ValueError('Selected model changed before evaluation')
                model, _ = _model(config, checkpoint, data.classes)
                role = 'teacher' if config.train.seed == 41 else f'student{config.train.seed}'
                target = report_dir / 'evaluation' / role
                identity = {'protocol_sha256': method_sha, 'checkpoint_sha256': sha(checkpoint),
                    'condition': recipe, 'seed': config.train.seed, 'evaluation_split': 'test',
                    'data': data.provenance}
                labels, probabilities = _save_predictions(target / 'predictions.npz', identity, model, data.test, data.classes)
                metrics = _quality_report(target, labels, probabilities, data.classes, targets)
                if recipe == 'control':
                    reference_metrics[role] = metrics
                    reference_arrays[role] = (labels, probabilities.argmax(1))
                elif not np.array_equal(labels, reference_arrays[role][0]):
                    raise ValueError('Methods evaluated on different test examples/order')
                measurements.append({'run': config.name, 'role': role, 'seed': config.train.seed,
                    'metrics': metrics, 'accuracy_delta_vs_ce': metrics['accuracy'] - reference_metrics[role]['accuracy'],
                    'target_recall_delta_vs_ce': metrics['target_macro_recall'] - reference_metrics[role]['target_macro_recall']})
                destination = report_dir / 'training_metadata' / config.name
                for name in ('config.json', 'data.json', 'history.json', 'summary.json', 'teacher_protocol.json', 'completion.json'):
                    source = Path(config.output_dir) / config.name / name
                    if source.is_file():
                        destination.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, destination / name)
                del model
            students = [x for x in measurements if x['role'] != 'teacher']
            row = {'recipe': recipe, 'teacher_accuracy': measurements[0]['metrics']['accuracy'],
                'student_accuracy_mean': float(np.mean([x['metrics']['accuracy'] for x in students])),
                'student_accuracy_sd': float(np.std([x['metrics']['accuracy'] for x in students], ddof=1)),
                'student_target_recall_mean': float(np.mean([x['metrics']['target_macro_recall'] for x in students])),
                'accuracy_delta_vs_ce': float(np.mean([x['accuracy_delta_vs_ce'] for x in students])),
                'target_recall_delta_vs_ce': float(np.mean([x['target_recall_delta_vs_ce'] for x in students]))}
            report_rows.append(row)
            write_json(report_dir / 'comparison.json', {'status': 'complete', 'comparison_type': 'teacher_training_baseline',
                'case': case_id, 'recipe': recipe, 'protocol_sha256': method_sha,
                'evaluation_split': 'test', 'target_classes': targets, 'runs': measurements,
                'summary': row, 'ce_control_pins': read(ROOT / 'baseline-pins' / (case_id + '.json'))})
        comparison = {'status': 'complete', 'case': case_id, 'protocol_sha256': protocol_sha,
                      'rows': report_rows, 'evaluation': 'Canonical CPU FP32; all eight methods; no test-based recipe reselection'}
        write_json(output / 'comparison.json', comparison)
        markdown = '# Eight-method teacher training baseline\n\n| Method | Teacher accuracy % | KD accuracy % | KD accuracy delta pp | Target recall delta pp |\n|---|---:|---:|---:|---:|\n'
        markdown += ''.join(f"| {r['recipe']} | {100*r['teacher_accuracy']:.3f} | {100*r['student_accuracy_mean']:.3f} | {100*r['accuracy_delta_vs_ce']:+.3f} | {100*r['target_recall_delta_vs_ce']:+.3f} |\n" for r in report_rows)
        (output / 'report.md').write_text(markdown)
        receipts = []
        archive_helpers.ROOT = ROOT
        for recipe in all_runs:
            report_dir = output / 'reports' / recipe
            shutil.copyfile(output / 'comparison.json', report_dir / 'case-summary.json')
            (report_dir / 'report.md').write_text(markdown)
            names = [p for p in report_dir.rglob('*') if p.is_file() and p.name != 'report_manifest.json']
            write_json(report_dir / 'report_manifest.json', {'protocol_sha256': sha(report_dir / 'protocol.json'),
                'files': {str(p.relative_to(report_dir)): sha(p) for p in names}})
            receipts.append(archive_helpers.archive_result({'id': case_id + '-' + recipe, 'output': str(report_dir)}, expected_prediction_panels=4))
        completion = {'protocol_sha256': protocol_sha, 'selected_models': selected_models,
                      'archives': receipts, 'new_teacher_fits': 7, 'new_kd_student_fits': 21}
        write_json(final, completion)
        write(progress, {'phase': 'complete', 'new_fits': 28, 'ce_controls_retrained': False})
        return completion
    except BaseException as error:
        write(progress, {'phase': 'blocked_low_disk' if isinstance(error, LowDisk) else 'failed', 'error': repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--ready-controls', action='store_true')
    parser.add_argument('--case')
    args = parser.parse_args()
    plan = verified_plan()
    if args.check_only:
        print(json.dumps({'ready': True, 'training_started': False,
            'cases': [check_case(c, ready_controls=args.ready_controls) for c in plan['cases']]}, indent=2))
        return
    if args.case:
        print(json.dumps(run_case(args.case), indent=2))
        return
    import fcntl
    with (ROOT / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deployment = read(ROOT / 'deployment.json')
        if sha(__file__) != deployment['controller_sha256']:
            raise ValueError('Controller differs from deployment identity')
        completed = []
        current = None
        try:
            while not dependency_state(plan):
                status('waiting_for_dependency', completed, dependency=plan['dependency'])
                time.sleep(30)
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'code'), PYTHONUNBUFFERED='1')
            for case in plan['cases']:
                current = case['id']
                storage(plan)
                status('running', completed, current)
                with (ROOT / (current + '.log')).open('a') as handle:
                    subprocess.run([sys.executable, str(ROOT / 'queue_runner.py'), '--case', current],
                        cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
                completed.append(current)
                status('queued', completed)
            status('complete', completed)
        except BaseException as error:
            progress = ROOT / 'studies' / (current or '') / 'progress.json'
            low = isinstance(error, LowDisk) or (progress.is_file() and read(progress).get('phase') == 'blocked_low_disk')
            status('blocked_low_disk' if low else 'failed', completed, current, error=repr(error))
            raise


if __name__ == '__main__':
    main()
