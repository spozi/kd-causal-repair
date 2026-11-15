"""Fit and evaluate the minimal CE/KD controls using the unchanged frozen runtime."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def run(case_id, root=ROOT):
    # These modules are deployment helpers; scientific imports resolve only after verification.
    from preflight import (verify_snapshot, configurations, validate_data, verify_baseline,
                           freeze, sha, read)
    root = Path(root).resolve()
    plan = verify_snapshot(root)
    case = next(x for x in plan['cases'] if x['id'] == case_id)
    teacher, students, spec = configurations(root, case)
    baseline = Path(case['baseline'])
    protocol = {'version': 1, 'purpose': 'Shared CE teacher and paired original KD controls',
                'case': case_id, 'teacher': teacher.to_dict(),
                'students': [x.to_dict() for x in students],
                'scientific_source_sha256': plan['scientific_source_sha256'],
                'pretrained_weight_sha256': plan['pretrained_weight_sha256']}
    if baseline.exists() and any(baseline.iterdir()) and not (baseline / 'protocol.json').exists():
        raise ValueError('Nonempty baseline directory has no frozen protocol')
    freeze(baseline / 'protocol.json', protocol)
    digest = sha(baseline / 'protocol.json')
    from kd.checkpoints import write_json
    from kd.teacher_training import train_teacher, RECIPES
    from kd.neuron_surgery_study import (_complete_student, resolve_target_classes,
                                         _model, _save_predictions, _quality_report)
    from kd.data import build_data
    import torch
    data = validate_data(root, case, teacher)
    targets = list(resolve_target_classes(spec, data.provenance['train_per_class']))
    freeze(baseline / 'target_class_plan.json', {
        'policy': spec.target_class_policy, 'classes': targets,
        'train_per_class': data.provenance['train_per_class'], 'tie_break': 'dataset label index'})
    train_teacher(teacher, RECIPES['control'], selection_metric='accuracy')
    write_json(baseline / 'progress.json', {'phase': 'original_kd', 'teacher_complete': True})
    for config in students:
        _complete_student(config, digest)
        write_json(baseline / 'progress.json', {'phase': 'original_kd', 'completed_seed': config.train.seed})
    verify_baseline(root, case, create_pins=True)
    final = baseline / 'report_manifest.json'
    if final.exists():
        manifest = read(final)
        if manifest['protocol_sha256'] != digest:
            raise ValueError('Baseline report protocol changed')
        for name, expected in manifest['files'].items():
            if sha(baseline / name) != expected:
                raise ValueError('Baseline report changed after final evaluation')
        write_json(baseline / 'progress.json', {'phase': 'complete', 'models_frozen': True})
        return read(baseline / 'comparison.json')
    # All four selected models and target labels are frozen before the test loader is built.
    write_json(baseline / 'progress.json', {'phase': 'evaluation', 'models_frozen': True})
    torch.set_num_threads(teacher.train.threads)
    evaluation = build_data(teacher.data, replace(teacher.train, workers=0, device='cpu'),
                            include_test=True, diagnostic=True)
    rows = []
    for config in [teacher, *students]:
        checkpoint = baseline / config.name / 'student.pt'
        model, _ = _model(config, checkpoint, evaluation.classes)
        directory = baseline / 'evaluation' / config.name
        identity = {'protocol_sha256': digest, 'checkpoint_sha256': sha(checkpoint),
                    'seed': config.train.seed, 'condition': config.name,
                    'evaluation_split': 'test', 'data': evaluation.provenance}
        labels, probabilities = _save_predictions(directory / 'predictions.npz', identity,
                                                   model, evaluation.test, evaluation.classes)
        metrics = _quality_report(directory, labels, probabilities, evaluation.classes, targets)
        rows.append({'name': config.name, 'seed': config.train.seed, 'metrics': metrics})
        del model
    result = {'status': 'complete', 'protocol_sha256': digest, 'case': case_id,
              'evaluation_split': 'test', 'target_classes': targets, 'runs': rows,
              'interpretation': 'One fixed CE teacher per setting; paired student seeds; exploratory'}
    write_json(baseline / 'comparison.json', result)
    (baseline / 'report.md').write_text(
        '# Original CE teacher and KD controls\n\n'
        '| Run | Accuracy (%) | Target recall (%) | Macro F1 (%) |\n|---|---:|---:|---:|\n' +
        ''.join(f"| {x['name']} | {100*x['metrics']['accuracy']:.4f} | "
                f"{100*x['metrics']['target_macro_recall']:.4f} | {100*x['metrics']['macro_f1']:.4f} |\n"
                for x in rows))
    names = ['comparison.json', 'report.md', 'target_class_plan.json',
             *[str(x.relative_to(baseline)) for x in (baseline / 'evaluation').rglob('*') if x.is_file()]]
    write_json(final, {'protocol_sha256': digest, 'files': {name: sha(baseline / name) for name in names}})
    write_json(baseline / 'progress.json', {'phase': 'complete', 'models_frozen': True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.case), indent=2))


if __name__ == '__main__':
    main()
