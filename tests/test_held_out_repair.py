from dataclasses import replace
import unittest

import numpy as np

from kd.neuron_surgery_study import NeuronSurgerySpec, validation_repair_split


class ValidationRepairSplitTests(unittest.TestCase):
    def test_halves_are_disjoint_complete_and_stratified(self):
        labels = np.array([0] * 7 + [1] * 4 + [2] * 1)
        repair, selection = validation_repair_split(labels)
        self.assertEqual(sorted(repair + selection), list(range(len(labels))))
        self.assertFalse(set(repair) & set(selection))
        counts = {c: (sum(labels[repair] == c), sum(labels[selection] == c)) for c in (0, 1, 2)}
        # The selection half takes the larger share, so a single image stays there.
        self.assertEqual(counts, {0: (3, 4), 1: (2, 2), 2: (0, 1)})

    def test_split_is_deterministic_and_seeded(self):
        labels = np.repeat(np.arange(5), 20)
        self.assertEqual(validation_repair_split(labels), validation_repair_split(labels))
        self.assertNotEqual(validation_repair_split(labels, seed=1),
                            validation_repair_split(labels, seed=2))


class HardSourceSpecTests(unittest.TestCase):
    def test_default_keeps_training_hard_images(self):
        self.assertEqual(NeuronSurgerySpec().hard_source, "train")

    def test_validation_source_is_accepted_and_others_rejected(self):
        replace(NeuronSurgerySpec(), hard_source="validation").validate()
        with self.assertRaises(ValueError):
            replace(NeuronSurgerySpec(), hard_source="test").validate()


if __name__ == "__main__":
    unittest.main()
