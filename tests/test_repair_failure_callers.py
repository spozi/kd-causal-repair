"""Production wiring checks with mocked fitters: no repair experiment runs."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from kd.baseline_artifacts import TEACHER_FILES, write_manifest
from kd.checkpoints import fingerprint, metadata, save_checkpoint, write_json
from kd.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from kd.matrix_binding import STUDY_SOURCE_FILES
from kd.models import create_model
from kd.neuron_surgery_study import COMPUTE_SOURCES, NeuronSurgerySpec, _train_candidate
from kd.shortcut_baselines import DEFAULT_GROUP_POLICY
from kd.shortcut_repair import train_shortcut_repair
from kd.teacher_repair_baseline import TeacherRepairOptions


class RepairFailureCallerTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_class_candidate_passes_failure_path_and_frozen_identity(self):
        classes = [f'class_{i}' for i in range(10)]
        config = ExperimentConfig(student=ModelConfig('cifar_teacher'),
                                  data=DataConfig(num_classes=10, image_size=32))
        anchor = create_model('cifar_teacher', 10)
        checkpoint = self.root / 'anchor.pt'
        save_checkpoint(checkpoint, {**metadata('cifar_teacher', classes, 32, 'synthetic'),
                                    'kind': 'inference', 'student': anchor.state_dict(), 'epoch': 0})
        destination = self.root / 'candidate'
        identity = {'trial': 'frozen-class-candidate'}
        failure = RuntimeError('stop before fitting')
        with patch('kd.neuron_surgery_study.train_repair_candidate', side_effect=failure) as fit:
            with self.assertRaises(RuntimeError) as raised:
                _train_candidate(destination, identity, config, checkpoint, anchor,
                    None, np.asarray([0, 1]), [0], [1], [], torch.device('cpu'),
                    classes, None, 'localization-digest', 1, 0.001,
                    NeuronSurgerySpec(), tuple(range(10)))
        self.assertIs(raised.exception, failure)
        self.assertEqual(fit.call_args.kwargs['failure_dir'], destination / 'failures')
        context = fit.call_args.kwargs['failure_context']
        self.assertEqual(context['candidate_identity'], identity)
        self.assertEqual(context['base_checkpoint_sha256'], fingerprint(checkpoint))
        self.assertEqual(context['localization_sha256'], 'localization-digest')
        self.assertFalse((destination / 'completion.json').exists())

    def test_shortcut_continuation_passes_trial_path_and_anchor_binding(self):
        config = ExperimentConfig(name='teacher_repair_ce_continuation_seed41',
            output_dir=str(self.root), student=ModelConfig('convnet6_w16'),
            data=DataConfig(source='waterbirds', image_size=224, num_classes=2),
            train=TrainConfig(seed=41, device='cpu'))
        anchor = create_model('convnet6_w16', 2)
        control = self.root / 'teacher_control_seed41'
        control.mkdir()
        provenance = {'source': 'waterbirds', 'split_sizes': {'train': 8, 'val': 8}}
        state = {**metadata('convnet6_w16', ['landbird', 'waterbird'], 224, 'waterbirds'),
                 'student': anchor.state_dict(), 'kind': 'inference', 'epoch': 0}
        for name in TEACHER_FILES:
            if name.endswith('.pt'):
                save_checkpoint(control / name, state)
            else:
                write_json(control / name, {'data': provenance} if name == 'summary.json' else {})
        write_manifest(control, 'completion.json', TEACHER_FILES)
        groups = np.repeat(np.arange(4), 2)
        labels = groups // 2
        dataset = TensorDataset(torch.zeros(8, 3, 224, 224), torch.tensor(labels))
        dataset.group_ids, dataset.targets = groups, labels
        dataset.source_ids = np.asarray([f'fixture-{i}' for i in range(8)])
        data = SimpleNamespace(train=DataLoader(dataset, batch_size=2), provenance=provenance,
                               classes=['landbird', 'waterbird'])
        records = [{'index': i, 'true_label': int(label), 'predicted_label': int(label),
                    'correct': True, 'nll': 0.1} for i, label in enumerate(labels)]
        spec = SimpleNamespace(group_policy=dict(DEFAULT_GROUP_POLICY),
            repair=TeacherRepairOptions(channel_budgets=(1,), learning_rates=(0.001,),
                repair_epochs=1, repair_samples_per_epoch=4, repair_batch_size=2, companion_count=1))
        failure = RuntimeError('stop before fitting')
        with patch('kd.shortcut_repair.build_data', return_value=data), \
             patch('kd.shortcut_repair.collect_teacher_diagnostics', return_value=(records, np.ones((8, 2)))), \
             patch('kd.shortcut_repair._validation', return_value={}), \
             patch('kd.shortcut_repair.train_repair_candidate', side_effect=failure) as fit:
            with self.assertRaises(RuntimeError) as raised:
                train_shortcut_repair(spec, 'repair_ce_continuation', config)
        self.assertIs(raised.exception, failure)
        trial = self.root / config.name / 'repair/candidate-b1-lr0.001'
        self.assertEqual(fit.call_args.kwargs['failure_dir'], trial / 'failures')
        context = fit.call_args.kwargs['failure_context']
        self.assertEqual(context['condition'], 'repair_ce_continuation')
        self.assertEqual(context['base_teacher_sha256'], fingerprint(control / 'student.pt'))
        self.assertEqual(context['base_completion_sha256'], fingerprint(control / 'completion.json'))
        self.assertEqual(context['trial_protocol']['fit_seed'], 2026)
        self.assertFalse((trial / 'completion.json').exists())
        self.assertFalse((self.root / config.name / 'completion.json').exists())

    def test_both_study_protocols_bind_the_recorder_source(self):
        self.assertIn('repair_failure.py', STUDY_SOURCE_FILES)
        self.assertIn('repair_failure.py', COMPUTE_SOURCES)


if __name__ == '__main__':
    unittest.main()
