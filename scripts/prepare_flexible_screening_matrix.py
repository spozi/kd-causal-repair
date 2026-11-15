"""Build the authorized 13-dataset ledger and 76-run checkpoint-reuse queue.

Preparation is local-only. Scientific runtime code is copied byte-for-byte from
the completed pilot, not from a potentially changed working tree.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from kd.neuron_surgery_study import neuron_surgery_spec_from_dict
from scripts.flexible_screening_pilot_preflight import verify_snapshot


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/server-deployments/flexible-screening-matrix13-20261026'
PILOT = ROOT / 'runs/server-deployments/flexible-screening-pilot-20261026'
BACKUP = ROOT / 'runs/experiment-results/normalized-repair-20261025'
REMOTE = '/home/ubuntu/kd-flexible-screening-matrix13-20261026'
DATASETS = ('cifar10', 'cifar100', 'svhn', 'cinic10', 'gtsrb', 'fashionmnist',
            'pathmnist', 'bloodmnist', 'dermamnist', 'organamnist', 'caltech101', 'eurosat', 'stl10')
ORDER = ('stl10', 'caltech101', 'dermamnist', 'bloodmnist', 'organamnist',
         'cifar100', 'cifar10', 'pathmnist', 'svhn', 'cinic10')
PROFILES = ('lt-if50', 'lt-if100', 'lt-if10', 'balanced')
POLICIES = ('strict', 'screened_then_ranked')
PENDING = {'gtsrb', 'fashionmnist', 'eurosat'}
DONE = {'stl10-balanced', 'cifar100-lt-if50'}
DATA_KEYS = ('source', 'dataset_version', 'dataset_profile', 'catalog_version', 'catalog_sha256',
             'recipe_sha256', 'artifact_sha256', 'input_protocol', 'pretraining_overlap',
             'split_seed', 'validation_fraction', 'train_index_sha256', 'val_index_sha256',
             'train_per_class', 'val_per_class', 'imbalance_factor', 'imbalance_profile',
             'imbalance_minimum_per_class', 'split_indices_sha256')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build_plan(inventory):
    cases = inventory['cases']
    expected_ids = {f'{dataset}-{profile}' for dataset in ORDER for profile in PROFILES}
    if len(cases) != 40 or {case['id'] for case in cases} != expected_ids:
        raise ValueError('Inventory must contain exactly four profiles of the ten retained datasets')
    by_id = {case['id']: case for case in cases}
    jobs, queued_cases, configs = [], [], {}
    for dataset in ORDER:
        for profile in PROFILES:
            case_id = f'{dataset}-{profile}'
            case = by_id[case_id]
            original = neuron_surgery_spec_from_dict(case['study'])
            names = {original.teacher_run, *(original.student_run_template.format(seed=seed)
                                            for seed in original.student_seeds)}
            if set(case['expected_model_sha256']) != names:
                raise ValueError(f'Missing exact teacher/KD pins: {case_id}')
            if any(len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                   for digest in case['expected_model_sha256'].values()):
                raise ValueError(f'Invalid checkpoint digest: {case_id}')
            if (original.dataset_source != dataset or original.dataset_profile != profile
                    or original.teacher_model != 'resnet34_smallstem'
                    or original.student_model != 'resnet18_smallstem'
                    or original.teacher_seed != 41 or original.student_seeds != (42, 43, 44)
                    or original.dataset_preprocessing != 'imagenet-smallstem-v1'
                    or original.training_epochs != 100 or original.repair_epochs != 20
                    or original.parameter_budget_fractions != (0.004, 0.008, 0.016, 0.032)
                    or original.max_preservation_kl != 0.01
                    or original.learning_rates != (0.001, 0.005)
                    or original.repair_samples_per_epoch != 640 or original.repair_batch_size != 64
                    or not original.downstream_kd or original.hard_source != 'train'
                    or original.kd_weight != 1 or original.feature_weight != 0.25
                    or original.preservation_ce_weight != 0 or original.repair_subject != 'teacher'
                    or original.channel_selection != 'screened' or not original.causal_validation
                    or original.evaluation_split != 'test' or original.score_mode != 'differential'):
                raise ValueError(f'Original study differs from declared paired protocol: {case_id}')
            required = {'train_index_sha256', 'val_index_sha256', 'recipe_sha256', 'artifact_sha256'}
            if not required <= set(case['expected_data']):
                raise ValueError(f'Missing original data identities: {case_id}')
            if case_id in DONE:
                continue
            queued_cases.append({key: case[key] for key in ('id', 'baseline', 'expected_model_sha256')}
                                | {'expected_data': {key: case['expected_data'][key]
                                                     for key in DATA_KEYS if key in case['expected_data']},
                                   'source_queue': case['source_queue'],
                                   'original_protocol_sha256': case['protocol_sha256'],
                                   'original_targets_sha256': case['targets_sha256']})
            for policy in POLICIES:
                job_id = f'{case_id}-{policy}'
                spec = replace(original, name=job_id, screening_policy=policy)
                spec.validate()
                configs[job_id] = spec.to_dict()
                jobs.append({'id': job_id, 'case': case_id, 'screening_policy': policy,
                             'config': f'configs/{job_id}.json', 'baseline': case['baseline'],
                             'output': f'{REMOTE}/studies/{job_id}'})
    ledger = []
    for dataset in DATASETS:
        for profile in PROFILES:
            case_id = f'{dataset}-{profile}'
            state = ('pending_missing_baselines' if dataset in PENDING else
                     'completed_pilot' if case_id in DONE else 'queued_checkpoint_reuse')
            ledger.append({'id': case_id, 'dataset': dataset, 'profile': profile,
                           'status': state, 'policies': list(POLICIES)})
    if len(jobs) != 76 or len(configs) != 76 or len(queued_cases) != 38 or len(ledger) != 52:
        raise ValueError('Unified matrix counts do not match the authorized scope')
    return {'version': 1, 'name': 'flexible-screening-matrix13-20261026',
            'remote_root': REMOTE, 'device': 'cuda:0', 'disk_free_minimum_gib': 12,
            'datasets': list(DATASETS), 'profiles': list(PROFILES), 'policies': list(POLICIES),
            'settings': ledger, 'cases': queued_cases, 'jobs': jobs,
            'total_settings': 52, 'total_policy_runs': 104, 'completed_pilot_runs': 4,
            'scheduled_runs': 76, 'pending_runs': 24,
            'maximum_candidate_fits': 608, 'maximum_new_student_fits': 228,
            'new_baseline_teacher_fits': 0, 'new_original_kd_fits': 0,
            'retention': 'Keep all models and resume files; pause rather than delete',
            'interpretation': 'Exploratory fixed-teacher paired study; no normalized random/localization-only controls'}, configs


def prepare(output, inventory_path=WORK / 'inventory.json'):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing frozen package: {output}')
    inventory = json.loads(Path(inventory_path).read_text())
    plan, configs = build_plan(inventory)
    # Bind live originals to the protocol/target bytes already verified in Mac backups.
    for case in inventory['cases']:
        name = case['id'] + '-results-with-predictions.tar.gz.json'
        verification = json.loads((BACKUP / 'verification' / case['source_queue'] / name).read_text())
        prefix = ('repair/' if case['source_queue'] in
                  {'kd-normalized-fresh-20261023', 'kd-normalized-three-20261024'} else '')
        if (verification[prefix + 'protocol.json'] != case['protocol_sha256']
                or verification[prefix + 'targets.json'] != case['targets_sha256']):
            raise ValueError(f'Original input provenance differs from verified Mac archive: {case["id"]}')
    pilot_manifest = verify_snapshot(PILOT)
    pilot_plan = json.loads((PILOT / 'experiment.json').read_text())
    original_cases = {case['id']: case for case in inventory['cases']}
    for case in pilot_plan['cases']:
        if case['expected_model_sha256'] != original_cases[case['id']]['expected_model_sha256']:
            raise ValueError('Completed pilot used different original models')
    pilot_backup = ROOT / 'runs/experiment-results/flexible-screening-pilot-20261026'
    prior = json.loads((pilot_backup / 'backup-verification.json').read_text())
    if len(prior['archives']) != 4:
        raise ValueError('Completed pilot results must be backed up before exclusion')
    plan['pilot_references'] = prior['archives']
    plan['pilot_scientific_source_sha256'] = {name: digest for name, digest in pilot_manifest.items()
                                            if name.startswith('code/')}
    output.mkdir(parents=True)
    for name in pilot_manifest:
        if name.startswith('code/'):
            target = output / name;target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PILOT / name, target)
    for name, source in [('preflight.py', PILOT / 'preflight.py'),
                         ('queue_runner.py', PILOT / 'queue_runner.py'),
                         ('README.md', ROOT / 'docs/experiments/flexible-screening-matrix13-20261026.md'),
                         ('inventory.json', Path(inventory_path)),
                         ('prepare-source.py', Path(__file__))]:
        shutil.copyfile(source, output / name)
    for job in plan['jobs']:
        write_json(output / job['config'], configs[job['id']])
    write_json(output / 'experiment.json', plan)
    manifest = {str(path.relative_to(output)): sha(path)
                for path in sorted(output.rglob('*')) if path.is_file()}
    write_json(output / 'manifest.json', manifest)
    for name, digest in plan['pilot_scientific_source_sha256'].items():
        if manifest[name] != digest:
            raise ValueError('Scientific runtime changed relative to the completed pilot')
    with tarfile.open(output / 'bundle.tar.gz', 'w:gz') as archive:
        for name in [*manifest, 'manifest.json']:
            archive.add(output / name, arcname=name, recursive=False)
    receipt = {'bundle_sha256': sha(output / 'bundle.tar.gz'),
               'bundle_bytes': (output / 'bundle.tar.gz').stat().st_size,
               'source_files_verified_against_pilot': len(plan['pilot_scientific_source_sha256']),
               'scheduled_runs': 76, 'pending_runs': 24, 'completed_pilot_runs': 4,
               'maximum_candidate_fits': 608, 'maximum_new_student_fits': 228,
               'controller_sha256': manifest['queue_runner.py'], 'training_started': False}
    write_json(output / 'preparation.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORK / 'package')
    parser.add_argument('--inventory', type=Path, default=WORK / 'inventory.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.inventory), indent=2))


if __name__ == '__main__':main()
