"""Build the authorized replacement package on the experiment server."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def prepare(planning_path, destination, operational):
    planning = read(planning_path)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a fresh package root; existing outputs must not be overwritten')
    source = Path(planning['server']['science_root']).parent
    recipes = planning['scope']['new_recipes']
    for name, digest in planning['scientific_source_sha256'].items():
        if sha(source / 'code' / name) != digest:
            raise ValueError(f'Frozen scientific source mismatch: {name}')
    if shutil.disk_usage(destination.parent).free < 110 * 1024**3:
        raise ValueError('Full campaign admission requires 110 GiB free')
    destination.mkdir()
    for name in planning['scientific_source_sha256']:
        target = destination / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / 'code' / name, target)
    for old, new in [('queue_runner.py', 'baseline_runner.py'),
                     ('control_helpers.py', 'control_base.py'), ('archive_helpers.py', 'archive_helpers.py')]:
        shutil.copyfile(source / old, destination / new)
    for old, new in [('replacement_recipe_queue.py', 'queue_runner.py'),
                     ('replacement_control_helpers.py', 'control_helpers.py')]:
        shutil.copyfile(operational / old, destination / new)
    shutil.copyfile(Path(__file__), destination / 'prepare-source.py')
    shutil.copyfile(planning_path, destination / 'planning-input.json')
    cases = []
    for original in planning['cases']:
        case_id, baseline = original['id'], Path(original['baseline_root'])
        controls = [read(baseline / run['name'] / 'config.json') for run in original['runs']]
        for run in original['runs']:
            for name, digest in run['completion_files_sha256'].items():
                if sha(baseline / run['name'] / name) != digest:
                    raise ValueError('Previously planned CE inputs changed')
        control_path = f'controls/{case_id}.json'
        spec_path = f'controls/{case_id}-study.json'
        write(destination / control_path, {'teacher': controls[0], 'students': controls[1:]})
        spec_source = baseline.parents[2] / 'configs' / (case_id + '.json')
        shutil.copyfile(spec_source, destination / spec_path)
        output = str(destination / 'studies' / case_id)
        conditions = []
        for recipe in recipes:
            teacher = deepcopy(controls[0])
            teacher.update(name=f'teacher_{recipe}_seed41', output_dir=output)
            students = []
            for raw in controls[1:]:
                student = deepcopy(raw)
                student.update(name=f'kd_{recipe}_teacher41_seed{raw["train"]["seed"]}', output_dir=output)
                student['teacher']['checkpoint'] = f'{output}/{teacher["name"]}/student.pt'
                students.append(student)
            conditions.append({'recipe': recipe, 'teacher': teacher, 'students': students})
        config_path = f'configs/{case_id}.json'
        write(destination / config_path, {'conditions': conditions})
        cases.append({'id': case_id, 'dataset': original['dataset'], 'profile': original['profile'],
                      'baseline': str(baseline), 'baseline_config': control_path, 'control_spec': spec_path,
                      'config': config_path, 'output': output, 'expected_data': original['expected_training_data'],
                      'expected_teacher_sha256': original['runs'][0]['sha256'], 'reuse_pins': original['runs']})
    history_root = Path('/home/ubuntu/kd-harmonized-evaluation-20261108')
    historical = read(history_root / 'plan.json')
    source_plan = read(source / 'experiment.json')
    plan = {'name': 'replacement-recipes-20261108', 'remote_root': str(destination),
            'training_authorized': True, 'authorization': 'User: run the experiment', 'device': 'cuda:0',
            'disk_free_minimum_gib': 12, 'recipes': recipes, 'cases': cases,
            'teacher_fits': 84, 'kd_student_fits': 252, 'total_new_fits': 336,
            'scientific_source_sha256': {'code/' + k: v for k, v in planning['scientific_source_sha256'].items()},
            'pretrained_weight_sha256': source_plan['pretrained_weight_sha256'],
            'dependency': {'root': str(history_root / 'results'), 'unit': 'kd-harmonized-evaluation-20261108.service',
                           'plan_sha256': sha(history_root / 'plan.json'),
                           'expected_complete': [c['id'] for c in historical['cases'] if c['previously_inventory_verified']]}}
    write(destination / 'experiment.json', plan)
    manifest = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    write(destination / 'manifest.json', manifest)
    write(destination / 'deployment.json', {'controller_sha256': sha(destination / 'queue_runner.py'),
                                          'manifest_sha256': sha(destination / 'manifest.json')})
    return {'root': str(destination), 'new_fits': 336, 'settings': len(cases), 'training_started': False,
            'manifest_sha256': sha(destination / 'manifest.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planning', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--operational', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.planning, args.output, args.operational), indent=2))
