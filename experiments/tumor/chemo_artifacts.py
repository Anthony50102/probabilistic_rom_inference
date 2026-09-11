"""Lightweight artifact identities shared by chemo inference and visualization."""

import hashlib

import numpy as np

PROTOCOL_ID = "chemo_matched_80_5_70_110_v1"
INPUT_AWARE_RESULTS_SUBDIR = f"{PROTOCOL_ID}_input_aware_v1"


def array_fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()
