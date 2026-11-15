import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def queue(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / 'runs/server-deployments/ascending-baselines-20261027/package/queue_runner.py'
    spec = importlib.util.spec_from_file_location('baseline_runner', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    monkeypatch.setitem(sys.modules, 'baseline_runner', base)
    spec = importlib.util.spec_from_file_location('replacement_queue_test', root / 'scripts/replacement_recipe_queue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency(queue, tmp_path):
    root = tmp_path / 'history'
    output = root / 'results'
    output.mkdir(parents=True)
    (root / 'plan.json').write_text('{}')
    dep = {'root': str(output), 'unit': 'historical.service',
           'plan_sha256': queue.runner.sha(root / 'plan.json'),
           'expected_complete': [f'case-{i}' for i in range(16)]}
    return {'dependency': dep}, output


def test_active_evaluation_keeps_gpu_work_waiting(queue, tmp_path, monkeypatch):
    plan, output = dependency(queue, tmp_path)
    (output / 'status.json').write_text(json.dumps({'phase': 'evaluation'}))
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *a, **k: 'active\n')
    assert queue.dependency_state(plan) is False


def test_inactive_unfinished_evaluation_stops_replacement(queue, tmp_path, monkeypatch):
    plan, output = dependency(queue, tmp_path)
    (output / 'status.json').write_text(json.dumps({'phase': 'evaluation'}))
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *a, **k: 'failed\n')
    with pytest.raises(RuntimeError, match='incomplete and inactive'):
        queue.dependency_state(plan)


def test_only_expected_missing_original_models_are_accepted(queue, tmp_path):
    plan, output = dependency(queue, tmp_path)
    (output / 'status.json').write_text(json.dumps({'phase': 'incomplete'}))
    ledger = {'plan_sha256': plan['dependency']['plan_sha256'],
              'ready_or_complete': [{'case': c} for c in plan['dependency']['expected_complete']],
              'issues': [{'case': c, 'reason': 'Original checkpoint root has not been located'} for c in queue.EXPECTED]}
    for case in plan['dependency']['expected_complete']:
        directory = output / case
        directory.mkdir()
        (directory / 'protocol.json').write_text('{}')
        (directory / 'result.json').write_text('original result')
        receipt = {'protocol_sha256': queue.runner.sha(directory / 'protocol.json'),
                   'files': {'result.json': queue.runner.sha(directory / 'result.json')}}
        (directory / 'completion.json').write_text(json.dumps(receipt))
    (output / 'evaluation-status.json').write_text(json.dumps(ledger))
    assert queue.dependency_state(plan) is True
    ledger['issues'][0]['reason'] = 'Loaded dataset identity differs'
    (output / 'evaluation-status.json').write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='Unexpected historical evaluation failure'):
        queue.dependency_state(plan)


def test_completed_dependency_artifact_tampering_is_rejected(queue, tmp_path):
    plan, output = dependency(queue, tmp_path)
    (output / 'status.json').write_text(json.dumps({'phase': 'incomplete'}))
    ledger = {'plan_sha256': plan['dependency']['plan_sha256'],
              'ready_or_complete': [{'case': c} for c in plan['dependency']['expected_complete']],
              'issues': [{'case': c, 'reason': 'Original checkpoint root has not been located'} for c in queue.EXPECTED]}
    (output / 'evaluation-status.json').write_text(json.dumps(ledger))
    directory = output / 'case-0'
    directory.mkdir()
    (directory / 'protocol.json').write_text('{}')
    (directory / 'result.json').write_text('changed')
    (directory / 'completion.json').write_text(json.dumps({
        'protocol_sha256': queue.runner.sha(directory / 'protocol.json'),
        'files': {'result.json': '0' * 64}}))
    with pytest.raises(ValueError, match='artifacts changed'):
        queue.dependency_state(plan)
