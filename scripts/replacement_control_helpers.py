"""Admission for replacement cohorts, preserving their original CE augmentations."""
from dataclasses import replace
from pathlib import Path

import control_base as base

ROOT = Path(__file__).resolve().parent
read, sha, freeze = base.read, base.sha, base.freeze
verify_snapshot, environment, validate_data = base.verify_snapshot, base.environment, base.validate_data
DATA_KEYS = base.DATA_KEYS


def configurations(root, case):
    from kd.config import from_dict
    from kd.teacher_baselines import load_spec
    from kd.teacher_repair_baseline import repair_spec
    raw = read(root / case['baseline_config'])
    teacher = from_dict(raw['teacher'])
    students = [from_dict(c) for c in raw['students']]
    spec = replace(repair_spec(load_spec(root / case['control_spec']), 41), downstream_kd=True)
    if (teacher.student.name != 'resnet34_smallstem' or teacher.train.seed != 41
            or teacher.distillation.method != 'supervised'
            or tuple(s.train.seed for s in students) != (42, 43, 44)):
        raise ValueError('Teacher architecture or seed pairing differs')
    for config in [teacher, *students]:
        config.validate(require_teacher=False)
        if (config.output_dir != case['baseline'] or config.data.source != case['dataset']
                or config.data.dataset_profile != case['profile']
                or config.data.image_size != 32 or config.data.preprocessing != 'imagenet-smallstem-v1'
                or config.data.download or config.data.split_seed != 2026
                or config.student.weights != 'IMAGENET1K_V1' or config.student.checkpoint is not None
                or config.calibration.enabled or config.cpc.enabled
                or config.supervised_loss.method != 'cross_entropy' or config.surgery.action != 'none'
                or (config.train.epochs, config.train.batch_size, config.train.learning_rate,
                    config.train.momentum, config.train.weight_decay, config.train.precision)
                   != (100, 128, .001, .9, .0005, 'bfloat16')):
            raise ValueError('Control differs from the fixed benchmark configuration')
    for config in students:
        if (config.student.name != 'resnet18_smallstem' or config.data != teacher.data
                or config.teacher.checkpoint != str(Path(case['baseline']) / teacher.name / 'student.pt')
                or (config.distillation.method, config.distillation.temperature,
                    config.distillation.weight, config.distillation.warmup_epochs,
                    config.distillation.feature_weight) != ('kd', 4., .5, 5, 0.)):
            raise ValueError('Student differs from paired CE control')
    return teacher, students, spec


base.configurations = configurations


def verify_baseline(root, case, *, create_pins=False):
    from kd.neuron_surgery_study import _load_inputs
    baseline = Path(case['baseline'])
    teacher, students, spec = configurations(root, case)
    _load_inputs(baseline, (42, 43, 44), spec)
    for run in case['reuse_pins']:
        for filename, digest in run['completion_files_sha256'].items():
            if sha(baseline / run['name'] / filename) != digest:
                raise ValueError(f'Planned CE source changed: {case["id"]}/{run["name"]}/{filename}')
        directory = baseline / run['name']
        completion = directory / ('completion.json' if run['name'].startswith('teacher_') else 'baseline_completion.json')
        if not completion.exists():
            completion = directory / 'completion.json'
        manifest = read(completion)
        files = manifest.get('files', manifest)
        for filename, digest in run['completion_files_sha256'].items():
            if files.get(filename) != digest:
                raise ValueError('CE completion manifest does not bind the planned source')
    pins = {'case': case['id'], 'control_spec_sha256': sha(root / case['control_spec']),
            'data_pin_sha256': sha(root / 'data-pins' / (case['id'] + '.json')),
            'files': {run['name'] + '/' + filename: digest for run in case['reuse_pins']
                      for filename, digest in run['completion_files_sha256'].items()},
            'initial_student_sha256': {run['name']: run['initial_student_sha256'] for run in case['reuse_pins']}}
    destination = root / 'baseline-pins' / (case['id'] + '.json')
    if create_pins:
        freeze(destination, pins)
    elif not destination.exists() or read(destination) != pins:
        raise ValueError('CE pins differ from admission record')
    return pins


def verify_initializations(root, case):
    """Reproduce the frozen pre-optimizer data/model creation order on CPU."""
    from kd.checkpoints import tensor_state_fingerprint
    from kd.data import build_data
    from kd.engine import seed_everything
    from kd.models import create_model
    from kd.teacher_baselines import _training_data_identity
    import torch
    teacher, students, _ = configurations(root, case)
    torch.set_num_threads(4)
    rows = []
    for config, pin in zip([teacher, *students], case['reuse_pins']):
        seed_everything(config.train.seed, deterministic=False)
        data = build_data(config.data, replace(config.train, workers=0, device='cpu'), include_test=False)
        if _training_data_identity(data.provenance) != _training_data_identity(case['expected_data']):
            raise ValueError('Loaded training/validation data differs from planned identity')
        model = create_model(config.student.name, config.data.num_classes, weights=config.student.weights)
        digest = tensor_state_fingerprint(model.state_dict())
        if digest != pin['initial_student_sha256']:
            raise ValueError(f'Unpaired regenerated initial state: {case["id"]}/{config.name}')
        rows.append({'run': config.name, 'initial_sha256': digest})
        del model, data
    freeze(root / 'initialization-pins' / (case['id'] + '.json'), rows)
    return rows
