"""Explicit recipe cancellation for an unfinished standalone class study.

The original protocol and retained per-run artifacts are never rewritten. The
reduced experiment has separate, labeled selection/evaluation records. This is
an adaptive scope amendment, not a completed original full-suite experiment.
"""
from dataclasses import replace
from pathlib import Path

from .baseline_artifacts import read_json, verify_manifest, write_manifest
from .checkpoints import fingerprint, write_json
from .teacher_baseline_phases import (
    _plan, _freeze, _extend, _check_matched_students, _verify_extension_selection,
)
from .teacher_baselines import (
    _report, _markdown, _student_run, _verify_runs, _training_data_identity,
    plan_study, study_configs,
)
from .teacher_training import RECIPES, train_teacher
from .teacher_repair_baseline import REPAIR_NAME


def run_amended_baseline_then_repair(original, output, *, report_root, excluded_recipes,
                                    reason, prior_workflow, callback=None, dry_run=False):
    """Resume retained fits unchanged while excluding explicitly canceled arms.

    Requires the original frozen standalone combined specification and its
    pre-existing workflow record. New and unfinished settings are supported;
    completed/evaluated original studies must remain historical results.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('An explicit scope-amendment reason is required')
    excluded = tuple(excluded_recipes)
    if (not excluded or len(set(excluded)) != len(excluded)
            or any(name not in original.recipes or name in ('control', REPAIR_NAME) for name in excluded)):
        raise ValueError('Only unique ordinary non-control recipe exclusions are allowed')
    if not Path(prior_workflow).is_file():
        raise ValueError('Original frozen workflow is required')
    original.validate()
    original_baseline = replace(original, recipes=tuple(n for n in original.recipes if n != REPAIR_NAME), repair=None)
    active = replace(original, recipes=tuple(n for n in original.recipes if n not in excluded))
    report_root = Path(report_root).resolve()
    baseline, output, repairs, workflow = _plan(active, output, report_root / 'repair')
    # _plan checks repair/data paths; also protect the encompassing amendment root.
    data_root = Path(active.teacher.data.root).resolve()
    from .teacher_baseline_phases import _overlap
    if _overlap(report_root, output) or _overlap(report_root, data_root):
        raise ValueError('Amendment report root overlaps models or data')
    original_plan = plan_study(original_baseline, output)
    parent_protocol = output / 'protocol.json'
    if parent_protocol.exists() and read_json(parent_protocol) != original_plan:
        raise ValueError('Original study protocol changed')
    if any((output / name).exists() for name in ('selection.json', 'study_completion.json', 'evaluation_completion.json')):
        raise ValueError('Do not amend a completed or selected original study')
    original_hash = fingerprint(parent_protocol) if parent_protocol.exists() else None
    amendment = {
        'version': 1, 'purpose': 'User-authorized adaptive recipe cancellation',
        'reason': reason, 'excluded_recipes': list(excluded),
        'original_spec': original.to_dict(), 'effective_spec': active.to_dict(),
        'original_protocol': original_plan, 'original_protocol_sha256': original_hash,
        'prior_workflow_sha256': fingerprint(prior_workflow),
        'scope': 'Canceled arms are excluded, not failed/zero-effect results; original artifacts retained',
        'source_sha256': {'teacher_baseline_amendment.py': fingerprint(Path(__file__)),
                          **workflow['source_sha256']},
    }
    workflow.update(amendment=amendment,
                    selection='Validation only within the amended recipe subset; scope was adaptively reduced',
                    reuse='Retained baseline fits and exact CE teacher reused without changing configs; canceled arms omitted')
    if dry_run:
        return {'dry_run': True, 'workflow': workflow}
    if report_root.exists() and any(report_root.iterdir()) and not (report_root / 'amendment.json').exists():
        raise FileExistsError('Nonempty amendment root has no matching amendment record')
    _freeze(report_root / 'amendment.json', amendment)
    _freeze(repairs / 'workflow.json', workflow)
    reports = report_root / 'baseline'
    _freeze(reports / 'protocol.json', workflow['baseline_protocol'])
    teachers, students, supervised = study_configs(baseline, output)
    configs = [*(c for _, c in teachers), *(c for _, _, c in students), *supervised]
    notify = callback if callback is not None else lambda phase: None
    sealed = ('protocol.json', 'validation.json', 'validation.md', 'selection.json')

    def verify_parent(_spec, _output, expected):
        if read_json(reports / 'protocol.json') != expected:
            raise ValueError('Amended baseline protocol changed')
        _freeze(report_root / 'amendment.json', amendment)
        if original_hash is not None and fingerprint(parent_protocol) != original_hash:
            raise ValueError('Original protocol changed during amendment')
        _verify_runs(baseline, teachers, students, supervised)
        _check_matched_students(students, supervised, output)
        verify_manifest(reports, 'training_completion.json', sealed)
        _verify_extension_selection(baseline, output, reports, configs)
        verify_manifest(reports, 'evaluation_completion.json', ('comparison.json', 'report.md', 'training_completion.json'))

    if not (reports / 'training_completion.json').exists():
        notify('baseline_train')
        initial, identities = {}, []
        for recipe, config in teachers:
            summary = train_teacher(config, RECIPES[recipe], selection_metric=baseline.teacher_selection_metric)
            digest = summary['initial_student_sha256']
            if initial.setdefault(config.train.seed, digest) != digest:
                raise ValueError('Teacher initialization differs across retained recipes')
            identities.append(_training_data_identity(summary['data']))
        for config in supervised:
            summary = _student_run(config)
            identities.append(_training_data_identity(summary['data']))
        for _, _, config in sorted(students, key=lambda row: row[0] != 'control'):
            summary = _student_run(config)
            identities.append(_training_data_identity(summary['data']))
        if any(row != identities[0] for row in identities):
            raise ValueError('Retained teacher/student data identities differ')
        _verify_runs(baseline, teachers, students, supervised)
        _check_matched_students(students, supervised, output)
        validation = _report(baseline, teachers, students, supervised, split='val')
        _freeze(reports / 'validation.json', validation)
        markdown = _markdown(validation)
        if (reports / 'validation.md').exists() and (reports / 'validation.md').read_text() != markdown:
            raise ValueError('Amended validation report changed')
        (reports / 'validation.md').write_text(markdown)
        rows = {row['recipe']: row for row in validation['rows']}
        chosen = max(baseline.recipes, key=lambda n: rows[n]['student'][baseline.student_selection_metric]['mean'])
        _freeze(reports / 'selection.json', {
            'recipe': chosen, 'selected_on': 'validation only', 'metric': baseline.student_selection_metric,
            'protocol_sha256': fingerprint(reports / 'protocol.json'),
            'validation_sha256': fingerprint(reports / 'validation.json'),
            'checkpoints': {c.name: fingerprint(output / c.name / 'student.pt') for c in configs},
        })
        _verify_extension_selection(baseline, output, reports, configs)
        write_manifest(reports, 'training_completion.json', sealed)
    if not (reports / 'evaluation_completion.json').exists():
        notify('baseline_evaluate')
        verify_manifest(reports, 'training_completion.json', sealed)
        _verify_runs(baseline, teachers, students, supervised)
        _verify_extension_selection(baseline, output, reports, configs)
        report = _report(baseline, teachers, students, supervised, split='test', output=reports / 'test')
        report['interpretation'] = amendment['scope']
        write_json(reports / 'comparison.json', report)
        (reports / 'report.md').write_text(_markdown(report))
        files = [p.relative_to(reports).as_posix() for p in (reports / 'test').rglob('*') if p.is_file()]
        write_manifest(reports, 'evaluation_completion.json', ('comparison.json', 'report.md', 'training_completion.json', *files))
    verify_parent(baseline, output, workflow['baseline_protocol'])
    notify('repair_and_matched_kd')
    return _extend(active, baseline, output, repairs, workflow,
                   parent_directory=reports, parent_verifier=verify_parent)
