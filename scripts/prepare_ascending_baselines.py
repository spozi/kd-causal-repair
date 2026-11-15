"""Freeze the missing seven-method benchmark in ascending dataset-size order."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/server-deployments/ascending-baselines-20261027'
SOURCE = ROOT / 'runs/server-deployments/flexible-screening-pending3-20261027/package'
REMOTE = '/home/ubuntu/kd-ascending-baselines-20261027'
DEPENDENCY_ROOT = '/home/ubuntu/kd-flexible-screening-pending3-20261027'
RECIPES = ('randaugment', 'label_smoothing', 'balanced_softmax', 'mse_logits',
           'mse_probabilities', 'sam', 'ema')
PROFILES = ('lt-if50', 'lt-if100', 'lt-if10', 'balanced')
DATASETS = {'eurosat', 'gtsrb', 'organamnist', 'fashionmnist', 'svhn', 'pathmnist'}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build_plan(inventory, remote=REMOTE, control_template=None):
    source_cases = inventory['cases']
    expected = {f'{d}-{p}' for d in DATASETS for p in PROFILES}
    if len(source_cases) != 24 or {c['id'] for c in source_cases} != expected:
        raise ValueError('Inventory must contain exactly the six authorized datasets/four profiles')
    by_id = {c['id']: c for c in source_cases}
    sizes = {d: sum(by_id[d + '-balanced']['expected_data']['train_per_class']) for d in DATASETS}
    order = sorted(DATASETS, key=lambda d: (sizes[d], d))
    template = (deepcopy(control_template) if control_template is not None else
                json.loads((SOURCE / 'configs/gtsrb-lt-if50-control.json').read_text()))
    cases, files = [], {}
    for dataset in order:
        for profile in PROFILES:
            original = by_id[f'{dataset}-{profile}']
            case_id = original['id']
            control_teacher = deepcopy(original['teacher_config'])
            controls = []
            for seed in (42, 43, 44):
                student = deepcopy(original['student_config'])
                student['train']['seed'] = seed
                student['name'] = f'kd_control_teacher41_seed{seed}'
                controls.append(student)
            base_config = f'controls/{case_id}.json'
            spec_path = f'controls/{case_id}-study.json'
            control_spec = deepcopy(template)
            control_spec.update(name=case_id, dataset_source=dataset, dataset_profile=profile,
                dataset_num_classes=control_teacher['data']['num_classes'],
                dataset_version=control_teacher['data']['dataset_version'],
                imbalance_factor={'balanced': 1., 'lt-if10': 10., 'lt-if50': 50., 'lt-if100': 100.}[profile],
                target_class_policy='all' if profile == 'balanced' else 'frequency_tail')
            files[base_config] = {'teacher': control_teacher, 'students': controls}
            files[spec_path] = control_spec
            output = f'{remote}/studies/{case_id}'
            conditions = []
            for recipe in RECIPES:
                teacher = deepcopy(control_teacher)
                teacher.update(name=f'teacher_{recipe}_seed41', output_dir=output)
                students = []
                for control in controls:
                    student = deepcopy(control)
                    seed = control['train']['seed']
                    student.update(name=f'kd_{recipe}_teacher41_seed{seed}', output_dir=output)
                    student['teacher']['checkpoint'] = f'{output}/{teacher["name"]}/student.pt'
                    students.append(student)
                conditions.append({'recipe': recipe, 'teacher': teacher, 'students': students})
            config = f'configs/{case_id}.json'
            files[config] = {'conditions': conditions}
            cases.append({'id': case_id, 'dataset': dataset, 'profile': profile,
                'baseline': original['baseline'], 'baseline_config': base_config,
                'control_spec': spec_path, 'config': config, 'output': output,
                'expected_data': original['expected_data'], 'expected_teacher_sha256': original['teacher_pin']})
    teacher_bytes = sum(inventory['model_sizes']['teacher'].values())
    student_bytes = sum(inventory['model_sizes']['student'].values())
    plan = {'version': 1, 'name': 'ascending-baselines-20261027', 'remote_root': remote,
        'device': 'cuda:0', 'disk_free_minimum_gib': 12,
        'dependency': {'root': DEPENDENCY_ROOT, 'unit': 'kd-flexible-screening-pending3-20261027.service',
                       'required_state': 'complete'},
        'dataset_order': order, 'default_train_samples': sizes, 'profiles': list(PROFILES),
        'recipes': list(RECIPES), 'cases': cases,
        'teacher_fits': 168, 'kd_student_fits': 504, 'total_new_fits': 672,
        'ce_teachers_retrained': 0, 'ce_students_retrained': 0, 'new_repairs': 0,
        'scientific_source_sha256': inventory['scientific_source_sha256'],
        'pretrained_weight_sha256': inventory['pretrained_weight_sha256'],
        'checkpoint_estimate_gib': (teacher_bytes * 168 + student_bytes * 504) / 1024**3,
        'checkpoint_estimate_note': 'Observed CE-file sizes; excludes extra EMA state, predictions, archives and active dependency growth',
        'retention': 'Keep all selected/best/last models and optimizer states; stop at 12 GiB rather than delete',
        'evaluation': 'All selected models frozen per setting before common CPU FP32 test inference',
        'archive_prediction_panels_per_condition': 4,
        'conditions_per_setting_including_reused_ce': 8}
    return plan, files


def prepare(output, inventory_path):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh immutable package output')
    inventory = json.loads(Path(inventory_path).read_text())
    plan, files = build_plan(inventory)
    old_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    output.mkdir(parents=True)
    for name, digest in plan['scientific_source_sha256'].items():
        if old_manifest.get(name) != digest or sha(SOURCE / name) != digest:
            raise ValueError('Scientific source differs from the active frozen campaign')
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, destination)
    shutil.copyfile(SOURCE / 'preflight.py', output / 'control_helpers.py')
    shutil.copyfile(SOURCE / 'pending_screening_archives.py', output / 'archive_helpers.py')
    shutil.copyfile(ROOT / 'scripts/ascending_baseline_queue.py', output / 'queue_runner.py')
    shutil.copyfile(Path(__file__), output / 'prepare-source.py')
    for name, value in files.items():
        write(output / name, value)
    write(output / 'experiment.json', plan)
    shutil.copyfile(ROOT / 'docs/experiments/ascending-baselines-20261027.md', output / 'README.md')
    manifest = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output / 'manifest.json', manifest)
    with tarfile.open(output / 'bundle.tar.gz', 'w:gz') as archive:
        for name in [*manifest, 'manifest.json']:
            archive.add(output / name, arcname=name, recursive=False)
    receipt = {'bundle_sha256': sha(output / 'bundle.tar.gz'),
        'controller_sha256': manifest['queue_runner.py'], 'bundle_bytes': (output / 'bundle.tar.gz').stat().st_size,
        'scientific_source_files_unchanged': len(plan['scientific_source_sha256']),
        'dataset_order': plan['dataset_order'], 'settings': 24, 'new_fits': 672,
        'checkpoint_estimate_gib': plan['checkpoint_estimate_gib'], 'training_started': False}
    write(output / 'preparation.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORK / 'package')
    parser.add_argument('--inventory', type=Path, default=ROOT / 'runs/server-deployments/ascending-baselines-inventory-20261027.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.inventory), indent=2))
