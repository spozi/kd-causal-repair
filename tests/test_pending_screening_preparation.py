"""Protocol and input-identity checks for the missing-baseline continuation."""
from copy import deepcopy

import pytest

from kd.config import from_dict
from kd.neuron_surgery_study import NeuronSurgerySpec
from scripts.prepare_pending_screening import build_plan, write
from scripts.pending_screening_preflight import configurations, freeze


@pytest.fixture
def prepared(tmp_path):
    teacher = from_dict({'data': {'source': 'cifar100', 'num_classes': 100,
        'dataset_version': '1.0', 'dataset_profile': 'balanced', 'image_size': 32,
        'preprocessing': 'imagenet-smallstem-v1'},
        'student': {'name': 'resnet34_smallstem', 'weights': 'IMAGENET1K_V1'},
        'distillation': {'method': 'supervised'}}).to_dict()
    student = deepcopy(teacher)
    student['distillation'].update(method='kd', temperature=4., weight=.5, warmup_epochs=5)
    reference = {'teacher': teacher, 'student': student,
                 'weights': {'resnet18-f37072fd.pth': 'a' * 64,
                             'resnet34-b627a593.pth': 'b' * 64}}
    prior = tmp_path / 'previous-inputs'
    catalog = {'datasets': {name: {'classes': 43 if name == 'gtsrb' else 10, 'version': '1.0'}
                           for name in ['gtsrb', 'fashionmnist', 'eurosat']}}
    template = NeuronSurgerySpec(teacher_run='teacher_control_seed41',
        student_run_template='kd_control_teacher41_seed{seed}',
        teacher_model='resnet34_smallstem', student_model='resnet18_smallstem',
        training_epochs=100, parameter_budget_fractions=(.004, .008, .016, .032),
        max_preservation_kl=.01, learning_rates=(.001, .005),
        dataset_preprocessing='imagenet-smallstem-v1', target_classes=None,
        expected_train_counts=None, expected_target_counts=None).to_dict()
    write(prior / 'experiment.json', {'pilot_scientific_source_sha256': {}})
    write(prior / 'code/kd/catalog/v1.json', catalog)
    write(prior / 'configs/cifar10-lt-if50-strict.json', template)
    plan, files = build_plan(reference, previous=prior, remote=str(tmp_path))
    for name, value in files.items():
        write(tmp_path / name, value)
    # The catalog is checked during admission, independently of generated configurations.
    write(tmp_path / 'code/kd/catalog/v1.json', catalog)
    return tmp_path, plan, files


def test_only_pending_settings_and_required_ce_controls(prepared):
    _, plan, _ = prepared
    assert len(plan['cases']) == 12
    assert len(plan['jobs']) == 24
    assert {x['dataset'] for x in plan['cases']} == {'gtsrb', 'fashionmnist', 'eurosat'}
    assert (plan['new_baseline_teacher_fits'], plan['new_original_kd_fits']) == (12, 36)
    assert plan['maximum_new_student_fits'] == 72
    assert plan['maximum_candidate_fits'] == 192


def test_all_profiles_use_100_epochs_and_named_initialization(prepared):
    root, plan, _ = prepared
    for case in plan['cases']:
        teacher, students, spec = configurations(root, case)
        assert spec.training_epochs == 100
        assert spec.dataset_num_classes == (43 if case['dataset'] == 'gtsrb' else 10)
        assert not teacher.data.horizontal_flip
        assert teacher.train.seed == 41
        assert [s.train.seed for s in students] == [42, 43, 44]
        assert all(s.student.weights == 'IMAGENET1K_V1' for s in [teacher, *students])


def test_strict_expanded_share_every_control_and_repair_hyperparameter(prepared):
    _, plan, files = prepared
    for case in plan['cases']:
        pair = [x for x in plan['jobs'] if x['case'] == case['id']]
        assert [x['screening_policy'] for x in pair] == ['strict', 'screened_then_ranked']
        assert pair[0]['baseline'] == pair[1]['baseline'] == case['baseline']
        left, right = [deepcopy(files[x['config']]) for x in pair]
        for row in [left, right]:
            row.pop('name'); row.pop('screening_policy')
        assert left == right
        assert left['parameter_budget_fractions'] == [.004, .008, .016, .032]
        assert left['max_preservation_kl'] == .01
        assert left['repair_epochs'] == 20


@pytest.mark.parametrize('field,value', [('epochs', 81), ('precision', 'float32'),
                                       ('learning_rate', .05), ('batch_size', 512)])
def test_admission_rejects_changed_control_training(prepared, field, value):
    root, plan, files = prepared
    case = plan['cases'][0]
    altered = deepcopy(files[case['baseline_config']])
    altered['students'][1]['train'][field] = value
    write(root / case['baseline_config'], altered)
    with pytest.raises(ValueError, match='frozen transfer protocol'):
        configurations(root, case)


def test_admission_rejects_changed_student_teacher_pair(prepared):
    root, plan, files = prepared
    case = plan['cases'][0]
    altered = deepcopy(files[case['baseline_config']])
    altered['students'][0]['teacher']['checkpoint'] = '/different/teacher.pt'
    write(root / case['baseline_config'], altered)
    with pytest.raises(ValueError, match='pairing'):
        configurations(root, case)


def test_no_silent_refreezing_of_baseline_or_data_identity(tmp_path):
    path = tmp_path / 'identity.json'
    freeze(path, {'checkpoint': 'first'})
    freeze(path, {'checkpoint': 'first'})
    with pytest.raises(ValueError, match='Previously frozen identity differs'):
        freeze(path, {'checkpoint': 'replacement'})
