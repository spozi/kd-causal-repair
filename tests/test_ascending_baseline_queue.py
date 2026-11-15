from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kd.config import from_dict
from kd.neuron_surgery_study import NeuronSurgerySpec
from scripts.prepare_ascending_baselines import build_plan, RECIPES, PROFILES
from scripts import ascending_baseline_queue as queue


@pytest.fixture
def inventory():
    sizes = {'eurosat': 21600, 'gtsrb': 23976, 'organamnist': 34561,
             'fashionmnist': 54000, 'svhn': 65931, 'pathmnist': 89996}
    rows = []
    for dataset, size in reversed(list(sizes.items())):
        classes = {'gtsrb': 43, 'organamnist': 11, 'pathmnist': 9}.get(dataset, 10)
        for profile in PROFILES:
            case = dataset + '-' + profile
            baseline = '/external/controls/' + case
            teacher = from_dict({'name': 'teacher_control_seed41', 'output_dir': baseline,
                'data': {'source': dataset, 'num_classes': classes, 'image_size': 32,
                         'dataset_version': '1.0', 'dataset_profile': profile,
                         'preprocessing': 'imagenet-smallstem-v1'},
                'student': {'name': 'resnet34_smallstem', 'weights': 'IMAGENET1K_V1'},
                'train': {'epochs': 100, 'batch_size': 128, 'learning_rate': .001,
                    'seed': 41, 'precision': 'bfloat16', 'device': 'cuda:0',
                    'cuda_mode': 'fast', 'channels_last': True}}).to_dict()
            student = deepcopy(teacher)
            student.update(name='kd_control_teacher41_seed42')
            student['student'].update(name='resnet18_smallstem')
            student['teacher'].update(name='resnet34_smallstem', checkpoint=baseline + '/teacher_control_seed41/student.pt')
            student['train']['seed'] = 42
            student['distillation'].update(method='kd', temperature=4., weight=.5, warmup_epochs=5)
            counts = [size // classes + int(i < size % classes) for i in range(classes)]
            rows.append({'id': case, 'dataset': dataset, 'profile': profile,
                'baseline': baseline, 'teacher_config': teacher, 'student_config': student,
                'teacher_pin': None, 'expected_data': {'train_per_class': counts}})
    return {'cases': rows, 'model_sizes': {'teacher': {'student.pt': 80}, 'student': {'student.pt': 40}},
            'scientific_source_sha256': {}, 'pretrained_weight_sha256': {}}


@pytest.fixture
def plan_files(inventory, tmp_path):
    spec = NeuronSurgerySpec(training_epochs=100, parameter_budget_fractions=(.004, .008, .016, .032),
        max_preservation_kl=.01, teacher_model='resnet34_smallstem', student_model='resnet18_smallstem',
        teacher_run='teacher_control_seed41', student_run_template='kd_control_teacher41_seed{seed}',
        dataset_preprocessing='imagenet-smallstem-v1').to_dict()
    return build_plan(inventory, remote=str(tmp_path), control_template=spec)


def test_actual_default_train_count_determines_ascending_order(plan_files):
    plan, _ = plan_files
    assert plan['dataset_order'] == ['eurosat', 'gtsrb', 'organamnist', 'fashionmnist', 'svhn', 'pathmnist']
    assert [c['dataset'] for c in plan['cases']] == [d for d in plan['dataset_order'] for _ in PROFILES]


def test_scope_is_seven_methods_and_no_ce_refits_or_repair(plan_files):
    plan, files = plan_files
    assert (plan['teacher_fits'], plan['kd_student_fits'], plan['total_new_fits']) == (168, 504, 672)
    assert plan['ce_teachers_retrained'] == plan['ce_students_retrained'] == plan['new_repairs'] == 0
    assert len(plan['cases']) == 24
    for case in plan['cases']:
        conditions = files[case['config']]['conditions']
        assert tuple(c['recipe'] for c in conditions) == RECIPES
        assert not {'control', 'teacher_repair', 'mixup', 'cutmix', 'randaugment_cutmix_ema'} & {c['recipe'] for c in conditions}


def test_new_conditions_preserve_control_data_schedule_and_seed(plan_files):
    plan, files = plan_files
    for case in plan['cases']:
        control = files[case['baseline_config']]
        for condition in files[case['config']]['conditions']:
            teacher = condition['teacher']
            assert teacher['data'] == control['teacher']['data']
            assert teacher['train'] == control['teacher']['train']
            assert teacher['student']['weights'] == 'IMAGENET1K_V1'
            assert teacher['output_dir'] != case['baseline']
            for student, original in zip(condition['students'], control['students']):
                assert student['train'] == original['train']
                assert student['data'] == original['data']
                assert student['student'] == original['student']
                assert student['distillation'] == original['distillation']
                assert student['teacher']['checkpoint'] == teacher['output_dir'] + '/' + teacher['name'] + '/student.pt'


def test_missing_or_duplicate_inventory_case_fails(inventory):
    altered = deepcopy(inventory)
    altered['cases'].pop()
    with pytest.raises(ValueError, match='exactly'):
        build_plan(altered)
    altered = deepcopy(inventory)
    altered['cases'][0] = deepcopy(altered['cases'][1])
    with pytest.raises(ValueError, match='exactly'):
        build_plan(altered)


def test_disk_guard_does_not_delete_files(monkeypatch, tmp_path):
    retained = tmp_path / 'model.pt';retained.write_bytes(b'keep')
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    monkeypatch.setattr(queue.shutil, 'disk_usage', lambda _: SimpleNamespace(free=11 * 1024**3))
    with pytest.raises(queue.LowDisk):
        queue.storage({'disk_free_minimum_gib': 12})
    assert retained.read_bytes() == b'keep'
    with pytest.raises(ValueError):
        queue.storage({'disk_free_minimum_gib': 2})


def test_active_dependency_causes_wait_without_training(monkeypatch, tmp_path):
    (tmp_path / 'status.json').write_text(json.dumps({'phase': 'running'}))
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *a, **k: 'active\n')
    assert queue.dependency_state({'dependency': {'root': str(tmp_path), 'unit': 'existing.service'}}) is False


@pytest.mark.parametrize('state', ['failed', 'running'])
def test_failed_or_stopped_dependency_cannot_be_bypassed(monkeypatch, tmp_path, state):
    (tmp_path / 'status.json').write_text(json.dumps({'phase': state}))
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *a, **k: 'inactive\n')
    with pytest.raises(RuntimeError):
        queue.dependency_state({'dependency': {'root': str(tmp_path), 'unit': 'existing.service'}})


def test_only_complete_dependency_with_all_control_families_is_ready(tmp_path):
    value = {'phase': 'complete', 'completed_baselines': list(range(12)), 'completed': list(range(24))}
    (tmp_path / 'status.json').write_text(json.dumps(value))
    plan = {'dependency': {'root': str(tmp_path), 'unit': 'existing.service'}}
    assert queue.dependency_state(plan)
    value['completed_baselines'].pop()
    (tmp_path / 'status.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):
        queue.dependency_state(plan)


def test_source_tampering_and_order_changes_rejected(plan_files, tmp_path, monkeypatch):
    plan, _ = plan_files
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    (tmp_path / 'experiment.json').write_text(json.dumps(plan))
    digest = queue.sha(tmp_path / 'experiment.json')
    (tmp_path / 'manifest.json').write_text(json.dumps({'experiment.json': digest}))
    assert queue.verified_plan()['total_new_fits'] == 672
    (tmp_path / 'experiment.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen input changed'):
        queue.verified_plan()


def test_archive_receipt_progress_is_atomic(tmp_path):
    target = tmp_path / 'status.json'
    queue.write(target, {'phase': 'waiting_for_dependency'})
    queue.write(target, {'phase': 'running'})
    assert json.loads(target.read_text())['phase'] == 'running'
    assert not (tmp_path / 'status.json.tmp').exists()
