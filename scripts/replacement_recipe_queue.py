"""Twelve replacement suites using the unchanged finalized baseline runner."""
import argparse
from pathlib import Path
import subprocess

import baseline_runner as runner

ROOT = Path(__file__).resolve().parent
RECIPES = runner.RECIPES
EXPECTED = tuple([
    'dermamnist-lt-if100', 'dermamnist-lt-if50', 'dermamnist-lt-if10', 'dermamnist-balanced',
    'cifar10-lt-if100', 'cifar10-lt-if50', 'cifar10-lt-if10', 'cifar10-balanced',
    'cifar100-lt-if100', 'cifar100-balanced', 'cinic10-lt-if10', 'cinic10-balanced'])
original_check_case = runner.check_case


def verified_plan():
    manifest = runner.read(ROOT / 'manifest.json')
    for name, digest in manifest.items():
        path = ROOT / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(ROOT) or runner.sha(path) != digest):
            raise ValueError(f'Frozen package changed: {name}')
    plan = runner.read(ROOT / 'experiment.json')
    if Path(plan['remote_root']).resolve() != ROOT or not plan['training_authorized']:
        raise ValueError('Wrong root or training not authorized')
    if (tuple(c['id'] for c in plan['cases']) != EXPECTED or tuple(plan['recipes']) != RECIPES
            or (plan['teacher_fits'], plan['kd_student_fits'], plan['total_new_fits']) != (84, 252, 336)):
        raise ValueError('Replacement scope differs from authorized plan')
    for name, digest in plan['scientific_source_sha256'].items():
        if manifest.get(name) != digest:
            raise ValueError('Scientific source identity changed')
    for case in plan['cases']:
        output = Path(case['output']).resolve()
        baseline = Path(case['baseline']).resolve()
        if output != ROOT / 'studies' / case['id'] or baseline.is_relative_to(ROOT):
            raise ValueError('New outputs must be separate from reused CE inputs')
    return plan


def dependency_state(plan):
    dep = plan['dependency']
    root = Path(dep['root'])
    if runner.sha(root.parent / 'plan.json') != dep['plan_sha256']:
        raise ValueError('Historical evaluation plan changed')
    state = runner.read(root / 'status.json')
    if state['phase'] in ('complete', 'incomplete'):
        ledger = runner.read(root / 'evaluation-status.json')
        completed = [c['case'] for c in ledger['ready_or_complete']]
        issues = [c['case'] for c in ledger['issues']]
        if (set(completed) != set(dep['expected_complete']) or len(completed) != 16
                or set(issues) != set(EXPECTED) or len(issues) != 12
                or ledger['plan_sha256'] != dep['plan_sha256']):
            raise ValueError('Historical evaluation has unexpected missing or incomplete settings')
        for issue in ledger['issues']:
            reason = issue['reason']
            if not (reason == 'Original checkpoint root has not been located'
                    or ('No such file or directory' in reason and '/workspace/experiments/' in reason)):
                raise ValueError('Unexpected historical evaluation failure')
        for case in completed:
            output = root / case
            receipt = runner.read(output / 'completion.json')
            if receipt['protocol_sha256'] != runner.sha(output / 'protocol.json'):
                raise ValueError('Historical completion protocol changed')
            for name, digest in receipt['files'].items():
                path = output / name
                if not path.resolve().is_relative_to(output) or runner.sha(path) != digest:
                    raise ValueError('Historical completion artifacts changed')
        return True
    active = subprocess.check_output(['systemctl', 'show', dep['unit'], '-p', 'ActiveState', '--value'], text=True).strip()
    if active not in ('active', 'activating'):
        raise RuntimeError('Historical evaluation is incomplete and inactive; inspect before fitting')
    return False


def check_case(case, *, ready_controls=False):
    result = original_check_case(case, ready_controls=ready_controls)
    if ready_controls:
        helper = runner.helpers()
        helper.verify_initializations(ROOT, case)
        result['regenerated_initializations_verified'] = 4
    return result


runner.ROOT = ROOT
runner.__file__ = str(Path(__file__).resolve())
runner.verified_plan = verified_plan
runner.dependency_state = dependency_state
runner.check_case = check_case

if __name__ == '__main__':
    runner.main()
