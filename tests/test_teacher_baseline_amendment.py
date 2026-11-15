from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from unittest.mock import patch

import pytest

from kd.checkpoints import fingerprint, write_json
from kd.teacher_baseline_amendment import run_amended_baseline_then_repair
from kd.teacher_baselines import plan_study, study_configs
from kd.teacher_training import RECIPES, train_teacher
from tests.test_teacher_baselines import TeacherRepairIntegrationTests, _TinyCifar


EXCLUDED = ('mixup', 'cutmix', 'randaugment_cutmix_ema')


def inputs(root):
    fixture = TeacherRepairIntegrationTests()
    spec = replace(fixture.spec(str(root / 'data')), recipes=(*RECIPES, 'teacher_repair'))
    prior = root / 'prior-workflow.json'
    write_json(prior, {'spec': spec.to_dict()})
    return fixture, spec, prior


def test_dry_run_keeps_eight_methods_without_model_or_data_access(tmp_path):
    _, original, prior = inputs(tmp_path)
    with patch('kd.teacher_baseline_amendment.train_teacher', side_effect=AssertionError('fit')):
        result = run_amended_baseline_then_repair(original, tmp_path/'models', report_root=tmp_path/'amended',
            excluded_recipes=EXCLUDED, reason='User requested cancellation', prior_workflow=prior, dry_run=True)
    workflow = result['workflow']
    assert len(workflow['baseline_protocol']['spec']['recipes']) == 8
    assert not set(EXCLUDED) & set(workflow['spec']['recipes'])
    assert workflow['counts']['baseline_fits'] == 17  # one student seed in fixture
    assert not (tmp_path/'models').exists() and not (tmp_path/'amended').exists()


@pytest.mark.parametrize('excluded', [('control',), ('teacher_repair',), ('unknown',), ('mixup','mixup'), ()])
def test_unsafe_or_invalid_cancellations_fail_before_writes(tmp_path, excluded):
    _, spec, prior = inputs(tmp_path)
    with pytest.raises(ValueError):
        run_amended_baseline_then_repair(spec, tmp_path/'models', report_root=tmp_path/'amended',
            excluded_recipes=excluded, reason='User cancellation', prior_workflow=prior)
    assert not (tmp_path/'amended').exists()


@pytest.mark.parametrize('selected', [True, False])
def test_partial_study_reuses_models_skips_canceled_arms_and_preserves_protocol(tmp_path, selected):
    fixture, original, prior = inputs(tmp_path)
    output, reports = tmp_path/'models', tmp_path/'amended'
    old = replace(original, recipes=tuple(RECIPES), repair=None)
    output.mkdir()
    write_json(output/'protocol.json', plan_study(old, output))
    teachers, _, _ = study_configs(old, output)
    with patch('kd.data.datasets.CIFAR10', _TinyCifar), redirect_stdout(io.StringIO()):
        for recipe, config in teachers:
            if recipe in ('control', 'mixup'):
                train_teacher(config, RECIPES[recipe], selection_metric=old.teacher_selection_metric)
    hashes={name:fingerprint(output/name) for name in ('protocol.json','teacher_control_seed41/student.pt',
                                                      'teacher_mixup_seed41/student.pt')}
    real_teacher=train_teacher
    calls=[]
    def teacher(config, recipe, **kwargs):
        assert recipe.name not in EXCLUDED
        calls.append(recipe.name)
        return real_teacher(config, recipe, **kwargs)
    fake=fixture.fake_repair(selected)
    def repair(*args, **kwargs):
        assert (reports/'baseline/evaluation_completion.json').exists()
        return fake(*args, **kwargs)
    with patch('kd.data.datasets.CIFAR10', _TinyCifar), redirect_stdout(io.StringIO()), \
            patch('kd.teacher_baseline_amendment.train_teacher', side_effect=teacher), \
            patch('kd.teacher_repair_baseline.run_neuron_surgery_study', side_effect=repair):
        result=run_amended_baseline_then_repair(original, output, report_root=reports,
            excluded_recipes=EXCLUDED, reason='User requested cancellation after earlier dataset results', prior_workflow=prior)
    assert len(calls)==8 and len(result['rows'])==9
    assert not set(EXCLUDED) & {row['recipe'] for row in result['rows']}
    assert all(fingerprint(output/name)==digest for name,digest in hashes.items())
    assert not list(output.glob('kd_mixup_*')) and not list(output.glob('kd_cutmix_*'))
    assert not list(output.glob('teacher_randaugment_cutmix_ema_*'))
    assert not (output/'selection.json').exists()  # old 11-recipe study is not claimed complete
    assert len(json.loads((reports/'baseline/comparison.json').read_text())['rows'])==8
    with patch('kd.teacher_baseline_amendment.train_teacher', side_effect=AssertionError('refit')), \
            patch('kd.teacher_baseline_amendment._report', side_effect=AssertionError('reevaluate')), \
            patch('kd.teacher_baseline_phases._report', side_effect=AssertionError('reevaluate')):
        resumed=run_amended_baseline_then_repair(original, output, report_root=reports,
            excluded_recipes=EXCLUDED, reason='User requested cancellation after earlier dataset results', prior_workflow=prior)
    assert resumed==result
    assert all(fingerprint(output/name)==digest for name,digest in hashes.items())


def test_completed_original_cannot_be_rewritten(tmp_path):
    _, original, prior=inputs(tmp_path)
    output=tmp_path/'models'
    output.mkdir()
    write_json(output/'selection.json', {'preserve':'completed'})
    with pytest.raises(ValueError,match='completed or selected'):
        run_amended_baseline_then_repair(original,output,report_root=tmp_path/'amended',
            excluded_recipes=EXCLUDED,reason='Cancel',prior_workflow=prior)
    assert not (tmp_path/'amended').exists()
