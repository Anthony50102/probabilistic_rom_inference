"""Lightweight artifact identities shared by chemo inference and visualization."""

import hashlib

import numpy as np


def array_fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()
