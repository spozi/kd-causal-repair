"""Baseline-only matrix scope must be frozen before execution binding."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kd.data import _effective_factor
from kd.neuron_surgery_matrix import DATASETS, PROFILES
from kd.teacher_baseline_matrix import matrix_jobs, materialize_matrix_plan, main
from kd.teacher_baselines import load_spec, plan_study, study_configs
from kd.teacher_repair_baseline import REPAIR_NAME


class BaselineOnlyMatrixTests(unittest.TestCase):
    def test_four_priority_class_datasets_roundtrip_profiles_and_exact_counts(self):
        datasets = ('cifar10', 'cifar100', 'dermamnist', 'cinic10')
        factors = {'balanced': 1.0, 'lt-if10': 10.0, 'lt-if50': 50.0, 'lt-if100': 100.0}
        expected_counts = {
            'teacher_conditions': 174,
            'teacher_training_fits': 174,
            'repair_candidate_fits_max': 0,
            'kd_student_fits_max': 522,
            'supervised_student_fits': 48,
            'total_fits_max': 744,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = materialize_matrix_plan(root/'plan', datasets=datasets,
                profiles=tuple(PROFILES), root=root/'data', run_root=root/'studies',
                baseline_only=True)
            protocol = json.loads(Path(result['protocol']).read_text())
            self.assertEqual(result['job_count'], 16)
            self.assertEqual(result['counts'], expected_counts)
            self.assertEqual(protocol['counts'], expected_counts)
            self.assertEqual({(job['dataset'], job['profile']) for job in protocol['jobs']},
                             {(dataset, profile) for dataset in datasets for profile in factors})
            self.assertEqual(len({job['id'] for job in protocol['jobs']}), 16)
            self.assertEqual(protocol['epoch_policy']['repair_epochs'], None)
            roundtrip_fits = 0
            for job in protocol['jobs']:
                with self.subTest(dataset=job['dataset'], profile=job['profile']):
                    spec = load_spec(job['config'])
                    self.assertEqual(spec.to_dict(), job['study'])
                    self.assertIsNone(spec.repair)
                    self.assertNotIn(REPAIR_NAME, spec.recipes)
                    self.assertEqual(spec.teacher.data.dataset_profile, job['profile'])
                    self.assertEqual(_effective_factor(spec.teacher.data), factors[job['profile']])
                    self.assertEqual(spec.teacher.train.epochs, 100)
                    self.assertEqual(spec.teacher.train.precision, 'bfloat16')
                    self.assertEqual(spec.teacher.data.image_size, 32)
                    self.assertEqual(spec.teacher.data.preprocessing, 'imagenet-smallstem-v1')
                    self.assertEqual(spec.teacher.student.weights, 'IMAGENET1K_V1')
                    self.assertEqual(spec.student_weights, 'IMAGENET1K_V1')
                    teachers, students, supervised = study_configs(spec, job['output'])
                    configs = [config for _, config in teachers]
                    configs.extend(config for _, _, config in students)
                    configs.extend(supervised)
                    for config in configs:
                        self.assertEqual(config.data, spec.teacher.data)
                        self.assertEqual(_effective_factor(config.data), factors[job['profile']])
                    study_plan = plan_study(spec, job['output'])
                    self.assertEqual(study_plan['counts']['total_fits_max'],
                                     job['counts']['total_fits_max'])
                    roundtrip_fits += study_plan['counts']['total_fits_max']
            self.assertEqual(roundtrip_fits, 744)

    def test_native_shortcuts_do_not_expand_into_requested_class_profiles(self):
        datasets = ('colored_mnist', 'waterbirds', 'multiceleba')
        jobs = matrix_jobs(datasets=datasets, profiles=tuple(PROFILES), baseline_only=True)
        self.assertEqual(len(jobs), 3)
        self.assertEqual([job['dataset'] for job in jobs], list(datasets))
        for job in jobs:
            self.assertEqual(job['profile'], 'native-shortcut')
            self.assertEqual(job['study']['teacher']['data']['dataset_profile'], 'native-shortcut')
            self.assertEqual(job['study']['teacher']['data']['imbalance_factor'], 1.0)
            self.assertIsNone(job['study']['repair'])
            self.assertFalse(job['study']['group_policy']['mechanism_controls'])
            self.assertNotIn(REPAIR_NAME, job['study']['recipes'])

    def test_priority_dry_plan_never_accesses_data_models_or_gpu(self):
        blocked = (
            'kd.teacher_baselines.build_data',
            'kd.data.build_data',
            'kd.teacher_baselines.create_model',
            'kd.models.create_model',
            'kd.teacher_baselines.run_experiment',
            'kd.teacher_baselines.resolve_device',
            'kd.runtime.resolve_device',
            'torch.cuda.is_available',
            'torch.cuda.get_device_capability',
            'torch.cuda.is_bf16_supported',
        )
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            for target in blocked:
                stack.enter_context(patch(target, side_effect=AssertionError(target)))
            protocol = materialize_matrix_plan(root/'plan',
                datasets=('cifar10', 'cifar100', 'dermamnist', 'cinic10'),
                profiles=tuple(PROFILES), root=root/'data', run_root=root/'studies',
                baseline_only=True, dry_run=True)
            self.assertEqual(protocol['job_count'], 16)
            self.assertEqual(protocol['counts']['total_fits_max'], 744)
            self.assertEqual(list(root.iterdir()), [])

    def test_52_studies_have_no_repair_and_exact_fit_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = materialize_matrix_plan(root/'plan', datasets=tuple(DATASETS),
                root=root/'data', run_root=root/'studies', baseline_only=True)
            self.assertEqual(result['job_count'], 52)
            self.assertEqual(result['counts']['total_fits_max'], 2436)
            self.assertEqual(result['counts']['repair_candidate_fits_max'], 0)
            for path in (root/'plan/configs').glob('*.json'):
                spec = load_spec(path)
                self.assertIsNone(spec.repair)
                self.assertNotIn(REPAIR_NAME, spec.recipes)
                plan = plan_study(spec, spec.teacher.output_dir)
                self.assertEqual(plan.get('mechanism_conditions', []), [])
                self.assertEqual(plan['repair_protocols'], [])
                self.assertEqual(spec.teacher.train.batch_size, 128)
                self.assertEqual(spec.teacher.train.precision, 'bfloat16')

    def test_shortcut_baseline_only_disables_repair_and_controls(self):
        for job in matrix_jobs(datasets=('colored_mnist','waterbirds','urbancars','multiceleba'), baseline_only=True):
            self.assertIsNone(job['study']['repair'])
            self.assertFalse(job['study']['group_policy']['mechanism_controls'])
            self.assertNotIn(REPAIR_NAME, job['study']['recipes'])

    def test_optional_compact_pair_retains_complete_matrix_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = materialize_matrix_plan(root/'plan',datasets=('cifar10',),
                pairs=('convnet6',),root=root/'data',baseline_only=True)
            self.assertEqual(result['job_count'],4)
            self.assertEqual(result['counts']['total_fits_max'],184)
            self.assertEqual(result['counts']['repair_candidate_fits_max'],0)

    def test_default_preserves_repair_and_flag_is_boolean(self):
        self.assertIsNotNone(matrix_jobs(datasets=('cifar10',))[0]['study']['repair'])
        with self.assertRaisesRegex(ValueError, 'boolean'):
            matrix_jobs(baseline_only='yes')

    def test_cli_propagates_scope(self):
        with patch('kd.teacher_baseline_matrix.materialize_matrix_plan', return_value={}) as planner:
            main(['plan','--output','unused','--baseline-only','--dry-run'])
        self.assertTrue(planner.call_args.kwargs['baseline_only'])


if __name__ == '__main__':
    unittest.main()
