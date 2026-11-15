"""Offline accounting/pairing checks for the 76-run reuse-only matrix."""
from copy import deepcopy
from dataclasses import replace

import pytest

from kd.neuron_surgery_study import NeuronSurgerySpec, neuron_surgery_spec_from_dict
from scripts.prepare_flexible_screening_matrix import build_plan, ORDER, PROFILES, PENDING, DONE


def inventory():
    cases = []
    classes = {'cifar10': 10, 'cifar100': 100, 'svhn': 10, 'cinic10': 10,
               'pathmnist': 9, 'bloodmnist': 8, 'dermamnist': 7,
               'organamnist': 11, 'caltech101': 101, 'stl10': 10}
    for dataset in ORDER:
        for profile in PROFILES:
            factor = {'balanced': 1, 'lt-if10': 10, 'lt-if50': 50, 'lt-if100': 100}[profile]
            spec = NeuronSurgerySpec(name=dataset+'-'+profile, teacher_run='teacher_control_seed41',
                student_run_template='kd_control_teacher41_seed{seed}',
                teacher_model='resnet34_smallstem', student_model='resnet18_smallstem',
                training_epochs=100, dataset_source=dataset, dataset_num_classes=classes[dataset],
                dataset_profile=profile, dataset_preprocessing='imagenet-smallstem-v1',
                imbalance_factor=factor, target_classes=None,
                target_class_policy='all' if factor==1 else 'frequency_tail',
                expected_train_counts=None, expected_target_counts=None,
                parameter_budget_fractions=(.004, .008, .016, .032), max_preservation_kl=.01)
            pins = {spec.teacher_run: 'a'*64, **{spec.student_run_template.format(seed=seed): 'b'*64
                                              for seed in spec.student_seeds}}
            cases.append({'id': spec.name, 'study': spec.to_dict(),
                          'baseline': '/home/ubuntu/original/'+spec.name,
                          'expected_model_sha256': pins,
                          'expected_data': {'train_index_sha256': 'c'*64, 'val_index_sha256': 'd'*64,
                                            'recipe_sha256': 'e'*64, 'artifact_sha256': {'test': 'f'*64}},
                          'source_queue': 'fixture', 'protocol_sha256': 'a'*64, 'targets_sha256': 'b'*64})
    return {'cases': cases}


def test_unified_ledger_accounts_for_all_13_datasets_without_queuing_pending_or_completed_cases():
    plan, configs = build_plan(inventory())
    assert len(plan['datasets']) == 13
    assert len(plan['settings']) == 52
    assert plan['total_policy_runs'] == 104
    assert plan['completed_pilot_runs'] == 4
    assert plan['scheduled_runs'] == 76
    assert plan['pending_runs'] == 24
    assert len(configs) == len(plan['jobs']) == 76
    assert len(plan['cases']) == 38
    assert len({job['output'] for job in plan['jobs']}) == 76
    assert {row['id'] for row in plan['settings'] if row['status']=='completed_pilot'} == DONE
    pending = [row for row in plan['settings'] if row['status']=='pending_missing_baselines']
    assert len(pending) == 12 and {row['dataset'] for row in pending} == PENDING
    assert not any(job['case'] in DONE or job['case'].split('-')[0] in PENDING for job in plan['jobs'])
    assert plan['new_baseline_teacher_fits'] == plan['new_original_kd_fits'] == 0
    assert plan['maximum_candidate_fits'] == 608
    assert plan['maximum_new_student_fits'] == 228


def test_strict_and_expanded_are_adjacent_and_otherwise_identical_with_original_models():
    plan, configs = build_plan(inventory())
    for first, second in zip(plan['jobs'][::2], plan['jobs'][1::2]):
        assert first['case'] == second['case']
        assert first['baseline'] == second['baseline']
        strict = neuron_surgery_spec_from_dict(configs[first['id']])
        expanded = neuron_surgery_spec_from_dict(configs[second['id']])
        assert strict.screening_policy == 'strict'
        assert expanded.screening_policy == 'screened_then_ranked'
        assert replace(expanded, name=strict.name, screening_policy='strict') == strict
        assert strict.student_seeds == (42, 43, 44)
        assert strict.downstream_kd
    assert plan['jobs'][0]['id'] == 'stl10-lt-if50-strict'
    assert plan['jobs'][-1]['id'] == 'cinic10-balanced-screened_then_ranked'


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra'])
def test_missing_duplicate_or_unapproved_inventory_fails_closed(change):
    data = inventory()
    if change=='missing':data['cases'].pop()
    elif change=='duplicate':data['cases'][-1] = deepcopy(data['cases'][0])
    else:data['cases'].append(deepcopy(data['cases'][0]))
    with pytest.raises(ValueError, match='exactly four profiles'):
        build_plan(data)


@pytest.mark.parametrize('change,value', [('repair_epochs', 21), ('training_epochs', 81),
    ('max_preservation_kl', .02), ('hard_source', 'validation'), ('score_mode', 'gradient_only'),
    ('downstream_kd', False), ('feature_weight', .5), ('repair_subject', 'student')])
def test_silent_recipe_changes_are_rejected(change, value):
    data = inventory();data['cases'][0]['study'][change] = value
    with pytest.raises(ValueError):
        build_plan(data)


def test_missing_or_invalid_checkpoint_pins_and_data_provenance_are_rejected():
    data = inventory();data['cases'][0]['expected_model_sha256'].pop('teacher_control_seed41')
    with pytest.raises(ValueError, match='exact teacher/KD pins'):build_plan(data)
    data = inventory();data['cases'][0]['expected_model_sha256']['teacher_control_seed41'] = 'unknown'
    with pytest.raises(ValueError, match='Invalid checkpoint digest'):build_plan(data)
    data = inventory();data['cases'][0]['expected_data'].pop('train_index_sha256')
    with pytest.raises(ValueError, match='data identities'):build_plan(data)
