"""Thread settings under which the reported tumor benchmark numbers were produced.

Import and call :func:`apply` before NumPy or JAX is imported. NumPy's
Accelerate/OpenBLAS thread count and XLA's CPU threading change float
rounding, so the untreated-growth fits reproduce the reported values
bitwise only in this single-threaded environment (the chemo fits were
bitwise identical in both). Explicit user settings are left alone and are
recorded with every fit.
"""
import os

REPORTED = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}


def apply():
    for key, value in REPORTED.items():
        os.environ.setdefault(key, value)
    return current()


def current():
    return {key: os.environ.get(key) for key in REPORTED}


def matches_reported():
    return current() == REPORTED
