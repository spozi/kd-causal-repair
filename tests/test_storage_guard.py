import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from kd.storage_guard import GIB, StorageGuard, StoragePause, _tensor_bytes


class StorageGuardTests(unittest.TestCase):
    def guard(self, free):
        return StorageGuard('/tmp/guard-study', usage=lambda _: SimpleNamespace(free=free))

    def test_study_eight_gib(self):
        self.guard(8 * GIB).check('study', study=True)
        with self.assertRaises(StoragePause):
            self.guard(8 * GIB - 1).check('study', study=True)

    def test_fit_reserve(self):
        function = Mock(return_value=5)
        self.assertEqual(self.guard(2 * GIB).fit(function)(), 5)
        with self.assertRaises(StoragePause):
            self.guard(2 * GIB - 1).fit(function)()
        self.assertEqual(function.call_count, 1)

    def test_checkpoint_headroom(self):
        save = Mock()
        with self.assertRaises(StoragePause):
            self.guard(2 * GIB + 1).torch_save(save)({}, '/tmp/guard-study/last.pt.tmp')
        save.assert_not_called()
        self.guard(3 * GIB).torch_save(save)({}, '/tmp/guard-study/last.pt.tmp')
        save.assert_called_once()

    def test_pause_bypasses_exception_handlers(self):
        self.assertFalse(issubclass(StoragePause, Exception))

    def test_other_filesystems_unaffected(self):
        save = Mock()
        self.guard(0).torch_save(save)({}, '/other/last.pt')
        save.assert_called_once()

    def test_failed_guard_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            path.write_bytes(b'old valid checkpoint')
            guard = StorageGuard(directory, usage=lambda _: SimpleNamespace(free=GIB))
            with self.assertRaises(StoragePause):
                guard.torch_save(Mock())({}, path)
            self.assertEqual(path.read_bytes(), b'old valid checkpoint')

    def test_tensor_view_backing_storage_and_cycles(self):
        import torch
        full = torch.zeros(100)
        value = [full[:1]]
        value.append(value)
        self.assertEqual(_tensor_bytes(value), 400)

    def test_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            StorageGuard('/tmp', study_gib=2, reserve_gib=2)


if __name__ == '__main__':
    unittest.main()
