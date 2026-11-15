"""Operational disk reserves; no changes to training or selection settings."""
from functools import wraps
import os
from pathlib import Path
import shutil

GIB = 2 ** 30


class StoragePause(BaseException):
    """Stop safely, without treating storage exhaustion as a numerical failure."""


class StorageGuard:
    def __init__(self, root, *, study_gib=8, reserve_gib=2, usage=shutil.disk_usage):
        if not 0 < reserve_gib < study_gib:
            raise ValueError('Require 0 < reserve_gib < study_gib')
        self.root = Path(root)
        self.study_bytes = int(study_gib * GIB)
        self.reserve_bytes = int(reserve_gib * GIB)
        self.usage = usage

    def check(self, phase, *, write_bytes=0, study=False):
        required = self.study_bytes if study else self.reserve_bytes + write_bytes
        free = self.usage(self.root).free
        if free < required:
            raise StoragePause(f'{phase}: free={free / GIB:.3f} GiB, '
                               f'required={required / GIB:.3f} GiB; '
                               'resume after freeing space')

    def fit(self, function):
        @wraps(function)
        def guarded(*args, **kwargs):
            self.check('before_fit')
            return function(*args, **kwargs)
        return guarded

    def torch_save(self, function):
        @wraps(function)
        def guarded(value, destination, *args, **kwargs):
            # A checkpoint's old version still exists during the atomic write.
            # Reserve space for the whole new payload, not merely its growth.
            if isinstance(destination, (str, bytes, os.PathLike)):
                parent = Path(os.fsdecode(destination)).resolve().parent
                if parent.is_relative_to(self.root.resolve()):
                    tensor_bytes = _tensor_bytes(value)
                    self.check('before_checkpoint',
                               write_bytes=tensor_bytes + tensor_bytes // 5 + 64 * 2 ** 20)
            return function(value, destination, *args, **kwargs)
        return guarded


def _tensor_bytes(value):
    """Conservative tensor-storage estimate, including view backing storage."""
    seen = set()

    def visit(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, dict):
            return sum(visit(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return sum(visit(v) for v in item)
        storage = getattr(item, 'untyped_storage', None)
        if callable(storage):
            return storage().nbytes()
        return 0

    return visit(value)
