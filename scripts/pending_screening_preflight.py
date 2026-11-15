"""Read-only admission checks for the three newly authorized dataset families."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA_KEYS = ('source', 'dataset_version', 'dataset_profile', 'catalog_version',
             'catalog_sha256', 'recipe_sha256', 'artifact_sha256', 'input_protocol',
             'pretraining_overlap', 'split_seed', 'validation_fraction',
             'train_index_sha256', 'val_index_sha256', 'train_per_class',
             'val_per_class', 'imbalance_factor', 'imbalance_profile',
             'imbalance_minimum_per_class', 'split_indices_sha256', 'split_sizes')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Previously frozen identity differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_snapshot(root=ROOT):
    root = Path(root).resolve()
    manifest = read(root / 'manifest.json')
    plan = read(root / 'experiment.json')
    if root != Path(plan['remote_root']).resolve():
        raise ValueError('Package is not at its frozen remote root')
    for name, digest in manifest.items():
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(root) or sha(path) != digest):
            raise ValueError(f'Frozen package changed: {name}')
    for name, digest in plan['scientific_source_sha256'].items():
        if manifest.get(name) != digest:
            raise ValueError('Scientific source differs from the completed pilot')
    sys.path.insert(0, str(root / 'code'))
    import kd
    if not Path(kd.__file__).resolve().is_relative_to(root / 'code'):
        raise ValueError('Imported a different scientific checkout')
    return plan


def environment(root, plan, device):
    import torch
    if device != plan['device'] or not device.startswith('cuda:'):
        raise ValueError('Use the declared BF16 CUDA device')
    if shutil.disk_usage(root).free < plan['disk_free_minimum_gib'] * 1024**3:
        raise RuntimeError('Free disk below admission threshold; retain all artifacts')
    index = int(device.split(':')[1])
    active = subprocess.check_output([
        'nvidia-smi', '-i', str(index), '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    if any(p.strip().isdigit() and int(p.strip()) != os.getpid() for p in active):
        raise RuntimeError('Another process owns the GPU')
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError('CUDA device unavailable')
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('GPU does not support BF16')
    for name, digest in plan['pretrained_weight_sha256'].items():
        if sha(Path(torch.hub.get_dir()) / 'checkpoints' / name) != digest:
            raise ValueError(f'Cached named ImageNet weights changed: {name}')


def configurations(root, case):
    from kd.config import from_dict
    from kd.neuron_surgery_study import load_neuron_surgery_spec
    raw = read(root / case['baseline_config'])
    teacher = from_dict(raw['teacher'])
    students = [from_dict(x) for x in raw['students']]
    spec = load_neuron_surgery_spec(root / case['control_spec'])
    catalog = read(root / 'code/kd/catalog/v1.json')['datasets'][case['dataset']]
    factor = {'balanced': 1., 'lt-if10': 10., 'lt-if50': 50., 'lt-if100': 100.}[case['profile']]
    if (spec.dataset_source != case['dataset'] or spec.dataset_profile != case['profile']
            or spec.dataset_num_classes != catalog['classes']
            or spec.dataset_version != catalog['version'] or spec.dataset_image_size != 32
            or spec.imbalance_factor != factor or spec.teacher_seed != 41
            or spec.student_seeds != (42, 43, 44) or spec.training_epochs != 100
            or spec.teacher_model != 'resnet34_smallstem' or spec.student_model != 'resnet18_smallstem'
            or spec.dataset_preprocessing != 'imagenet-smallstem-v1'
            or spec.parameter_budget_fractions != (.004, .008, .016, .032)
            or spec.learning_rates != (.001, .005) or spec.repair_epochs != 20
            or spec.repair_samples_per_epoch != 640 or spec.repair_batch_size != 64
            or spec.max_preservation_kl != .01 or spec.hard_source != 'train'
            or spec.kd_weight != 1 or spec.feature_weight != .25 or spec.preservation_ce_weight != 0
            or spec.repair_subject != 'teacher' or not spec.downstream_kd
            or not spec.causal_validation or spec.channel_selection != 'screened'
            or spec.evaluation_split != 'test' or spec.split_seed != 2026
            or spec.target_class_policy != ('all' if case['profile'] == 'balanced' else 'frequency_tail')):
        raise ValueError('Control study differs from the declared dataset/repair protocol')
    for config in [teacher, *students]:
        config.validate(require_teacher=False)
        if (config.train.epochs != 100 or config.train.batch_size != 128
                or config.train.precision != 'bfloat16' or config.train.learning_rate != .001
                or config.train.momentum != .9 or config.train.weight_decay != .0005
                or config.train.cuda_mode != 'fast' or not config.train.channels_last
                or config.train.persistent_workers or config.data.download
                or config.data.source != case['dataset']
                or config.data.dataset_profile != case['profile']
                or config.data.num_classes != catalog['classes']
                or config.data.dataset_version != catalog['version']
                or config.data.split_seed != 2026 or config.data.confirmation_fraction != 0
                or config.data.validation_fraction != .1
                or config.data.augmentation != 'basic' or config.data.horizontal_flip
                or config.calibration.enabled or config.cpc.enabled
                or config.supervised_loss.method != 'cross_entropy'
                or config.surgery.action != 'none'
                or config.data.image_size != 32
                or config.data.preprocessing != 'imagenet-smallstem-v1'
                or config.student.weights != 'IMAGENET1K_V1'
                or config.student.checkpoint is not None
                or str(Path(config.output_dir).resolve()) != case['baseline']):
            raise ValueError('Control config differs from the frozen transfer protocol')
    if (teacher.student.name != 'resnet34_smallstem' or teacher.train.seed != 41
            or teacher.distillation.method != 'supervised'
            or teacher.name != spec.teacher_run or len(students) != 3):
        raise ValueError('CE teacher identity differs')
    if tuple(x.train.seed for x in students) != (42, 43, 44):
        raise ValueError('Student seed pairing differs')
    for student in students:
        if (student.data != teacher.data or student.student.name != 'resnet18_smallstem'
                or student.name != spec.student_run_template.format(seed=student.train.seed)
                or student.teacher.name != teacher.student.name
                or student.teacher.checkpoint != str(Path(case['baseline']) / teacher.name / 'student.pt')
                or student.distillation.method != 'kd' or student.distillation.temperature != 4
                or student.distillation.weight != .5 or student.distillation.warmup_epochs != 5
                or student.distillation.feature_weight != 0):
            raise ValueError('Original KD control pairing differs')
    return teacher, students, spec


def validate_data(root, case, teacher):
    from kd.data import build_data
    data = build_data(teacher.data, replace(teacher.train, workers=0, device='cpu'),
                      include_test=False, diagnostic=True)
    identity = {k: data.provenance[k] for k in DATA_KEYS if k in data.provenance}
    if not {'train_index_sha256', 'val_index_sha256', 'artifact_sha256'} <= identity.keys():
        raise ValueError('Dataset provenance is incomplete')
    freeze(root / 'data-pins' / (case['id'] + '.json'), identity)
    return data


def verify_baseline(root, case, *, create_pins=False):
    from kd.neuron_surgery_study import _load_inputs
    teacher, students, spec = configurations(root, case)
    baseline = Path(case['baseline'])
    _load_inputs(baseline, spec.student_seeds, spec)
    names = [teacher.name, *(x.name for x in students)]
    pins = {'case': case['id'], 'baseline_protocol_sha256': sha(baseline / 'protocol.json'),
            'data_pin_sha256': sha(root / 'data-pins' / (case['id'] + '.json')),
            'files': {f'{name}/{file}': sha(baseline / name / file)
                      for name in names for file in ('config.json', 'summary.json', 'student.pt')},
            'initial_student_sha256': {
                x.name: read(baseline / x.name / 'summary.json')['initial_student_sha256']
                for x in students}}
    destination = root / 'baseline-pins' / (case['id'] + '.json')
    if create_pins:
        freeze(destination, pins)
    elif not destination.is_file() or read(destination) != pins:
        raise ValueError('Original CE/KD controls changed after baseline completion')
    return pins


def inspect(root=ROOT, device='cuda:0', baseline_id=None, job_id=None):
    root = Path(root).resolve()
    plan = verify_snapshot(root)
    environment(root, plan, device)
    cases = {x['id']: x for x in plan['cases']}
    jobs = {x['id']: x for x in plan['jobs']}
    if baseline_id and baseline_id not in cases or job_id and job_id not in jobs:
        raise ValueError('Unknown case/job')
    selected = ([cases[baseline_id]] if baseline_id else
                [cases[jobs[job_id]['case']]] if job_id else list(cases.values()))
    rows = []
    for case in selected:
        teacher, students, spec = configurations(root, case)
        data = validate_data(root, case, teacher)
        policies = [x for x in plan['jobs'] if x['case'] == case['id']]
        from kd.neuron_surgery_study import load_neuron_surgery_spec
        from dataclasses import asdict
        strict, expanded = [load_neuron_surgery_spec(root / x['config']) for x in policies]
        left, right = asdict(strict), asdict(expanded)
        control = asdict(spec)
        for key in ('name', 'screening_policy'):
            left.pop(key); right.pop(key); control.pop(key)
        if left != right or left != control or strict.screening_policy != 'strict' or expanded.screening_policy != 'screened_then_ranked':
            raise ValueError('Strict/expanded settings differ beyond declared policy')
        row = {'id': case['id'], 'train_samples': len(data.train.dataset),
               'val_samples': len(data.val.dataset), 'data_verified': True}
        if job_id:
            verify_baseline(root, case)
            from kd.neuron_surgery_study import run_neuron_surgery_study
            job = jobs[job_id]
            dry = run_neuron_surgery_study(job['baseline'], job['output'], device,
                                          spec.student_seeds, dry_run=True,
                                          spec=load_neuron_surgery_spec(root / job['config']))
            output = Path(job['output'])
            if output.exists() and any(output.iterdir()):
                if read(output / 'protocol.json') != dry['protocol']:
                    raise ValueError('Existing repair output differs from its frozen identity')
            row['dry_run_verified'] = True
        rows.append(row)
    return {'ready': True, 'training_started': False, 'cases': rows,
            'free_gib': shutil.disk_usage(root).free / 1024**3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--baseline')
    group.add_argument('--job')
    args = parser.parse_args()
    print(json.dumps(inspect(ROOT, args.device, args.baseline, args.job), indent=2))


if __name__ == '__main__':
    main()
