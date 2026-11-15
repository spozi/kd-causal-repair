"""Package the authorized CE controls and 24 remaining repair comparisons."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/server-deployments/flexible-screening-pending3-20261027'
PREVIOUS = ROOT / 'runs/server-deployments/flexible-screening-matrix13-20261026/package'
REMOTE = '/home/ubuntu/kd-flexible-screening-pending3-20261027'
DATA_ROOT = '/home/ubuntu/kd-cinic10-balanced/data'
DATASETS = ('gtsrb', 'fashionmnist', 'eurosat')
PROFILES = ('lt-if50', 'lt-if100', 'lt-if10', 'balanced')
POLICIES = ('strict', 'screened_then_ranked')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build_plan(reference, previous=PREVIOUS, remote=REMOTE):
    previous = Path(previous)
    prior = json.loads((previous / 'experiment.json').read_text())
    catalog = json.loads((previous / 'code/kd/catalog/v1.json').read_text())
    template = json.loads((previous / 'configs/cifar10-lt-if50-strict.json').read_text())
    cases, jobs, configs = [], [], {}
    for dataset in DATASETS:
        entry = catalog['datasets'][dataset]
        for profile in PROFILES:
            case_id = f'{dataset}-{profile}'
            baseline = f'{remote}/baselines/{case_id}'
            control_spec = deepcopy(template)
            control_spec.update(name=case_id, dataset_source=dataset,
                dataset_num_classes=entry['classes'], dataset_version=entry['version'],
                dataset_profile=profile, imbalance_factor={'balanced': 1., 'lt-if10': 10.,
                    'lt-if50': 50., 'lt-if100': 100.}[profile],
                target_class_policy='all' if profile == 'balanced' else 'frequency_tail')
            teacher = deepcopy(reference['teacher'])
            teacher.update(name='teacher_control_seed41', output_dir=baseline)
            teacher['data'].update(source=dataset, root=DATA_ROOT,
                dataset_version=entry['version'], dataset_profile=profile,
                num_classes=entry['classes'], image_size=32, imbalance_factor=1.,
                horizontal_flip=False, download=False)
            teacher['student'].update(name='resnet34_smallstem', checkpoint=None, weights='IMAGENET1K_V1')
            teacher['train'].update(epochs=100, batch_size=128, learning_rate=.001,
                momentum=.9, weight_decay=.0005, workers=4, threads=4,
                seed=41, device='cuda:0', precision='bfloat16', cuda_mode='fast',
                channels_last=True, persistent_workers=False, prefetch_factor=2)
            students = []
            for seed in (42, 43, 44):
                student = deepcopy(reference['student'])
                student.update(name=f'kd_control_teacher41_seed{seed}', output_dir=baseline,
                               data=deepcopy(teacher['data']), train=deepcopy(teacher['train']))
                student['train']['seed'] = seed
                student['student'].update(name='resnet18_smallstem', checkpoint=None, weights='IMAGENET1K_V1')
                student['teacher'].update(name='resnet34_smallstem',
                    checkpoint=f'{baseline}/teacher_control_seed41/student.pt', weights=None)
                students.append(student)
            baseline_config = f'baseline-configs/{case_id}.json'
            spec_path = f'configs/{case_id}-control.json'
            configs[baseline_config] = {'teacher': teacher, 'students': students}
            configs[spec_path] = control_spec
            cases.append({'id': case_id, 'dataset': dataset, 'profile': profile,
                          'baseline': baseline, 'baseline_config': baseline_config,
                          'control_spec': spec_path})
            for policy in POLICIES:
                job_id = f'{case_id}-{policy}'
                spec = deepcopy(control_spec)
                spec.update(name=job_id, screening_policy=policy)
                config = f'configs/{job_id}.json'
                configs[config] = spec
                jobs.append({'id': job_id, 'case': case_id, 'screening_policy': policy,
                             'config': config, 'baseline': baseline,
                             'output': f'{remote}/studies/{job_id}'})
    weights = {name: digest for name, digest in reference['weights'].items()
               if name in {'resnet18-f37072fd.pth', 'resnet34-b627a593.pth'}}
    if len(weights) != 2:
        raise ValueError('Both named ImageNet V1 initialization files must be cached and pinned')
    return {'version': 1, 'name': 'flexible-screening-pending3-20261027',
        'remote_root': remote, 'data_root': DATA_ROOT, 'device': 'cuda:0',
        'disk_free_minimum_gib': 12, 'datasets': list(DATASETS),
        'profiles': list(PROFILES), 'policies': list(POLICIES), 'cases': cases, 'jobs': jobs,
        'scheduled_runs': 24, 'new_baseline_teacher_fits': 12, 'new_original_kd_fits': 36,
        'maximum_candidate_fits': 192, 'maximum_new_student_fits': 72,
        'scientific_source_sha256': prior['pilot_scientific_source_sha256'],
        'pretrained_weight_sha256': weights,
        'reference_configuration_policy': 'Existing original CE/KD templates; dataset/profile-only adaptation',
        'authorization': 'User explicitly authorized missing CE controls and 24 repair arms on 2026-10-27',
        'retention': 'Keep all models and resume files; stop on low disk; no automatic deletion',
        'evaluation': 'Original controls: four canonical CPU FP32 test prediction panels per setting; repaired comparisons: unchanged frozen pipeline',
        'interpretation': 'Exploratory fixed-teacher paired comparison; thresholds unchanged'}, configs


def prepare(output, reference_path=WORK / 'reference-configs.json'):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh package output; do not overwrite frozen inputs')
    reference = json.loads(Path(reference_path).read_text())
    plan, configs = build_plan(reference)
    manifest = json.loads((PREVIOUS / 'manifest.json').read_text())
    output.mkdir(parents=True)
    for name, digest in plan['scientific_source_sha256'].items():
        if manifest.get(name) != digest or sha(PREVIOUS / name) != digest:
            raise ValueError('Previous scientific source changed after the completed pilot')
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PREVIOUS / name, destination)
    helpers = {'preflight.py': 'pending_screening_preflight.py',
               'baseline_runner.py': 'pending_screening_baselines.py',
               'pending_screening_queue.py': 'pending_screening_queue.py',
               'pending_screening_archives.py': 'pending_screening_archives.py',
               'prepare-source.py': 'prepare_pending_screening.py'}
    for destination, source in helpers.items():
        shutil.copyfile(ROOT / 'scripts' / source, output / destination)
    shutil.copyfile(ROOT / 'docs/experiments/flexible-screening-pending3-20261027.md', output / 'README.md')
    for name, value in configs.items():
        write(output / name, value)
    write(output / 'experiment.json', plan)
    manifest = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob('*')) if path.is_file()}
    write(output / 'manifest.json', manifest)
    with tarfile.open(output / 'bundle.tar.gz', 'w:gz') as archive:
        for name in [*manifest, 'manifest.json']:
            archive.add(output / name, arcname=name, recursive=False)
    receipt = {'bundle_sha256': sha(output / 'bundle.tar.gz'),
               'bundle_bytes': (output / 'bundle.tar.gz').stat().st_size,
               'controller_sha256': manifest['pending_screening_queue.py'],
               'scientific_source_files_unchanged': len(plan['scientific_source_sha256']),
               'baseline_settings': 12, 'baseline_fits': 48, 'repair_arms': 24,
               'training_started': False}
    write(output / 'preparation.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORK / 'package')
    parser.add_argument('--reference', type=Path, default=WORK / 'reference-configs.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.reference), indent=2))
