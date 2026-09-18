#!/usr/bin/env python3
"""F_2-linear algebra helpers shared by recovery and decoding.

Everything here is plain uint8/uint64 numpy: reduced row echelon form,
kernels, particular solutions, and bit packing between F_{2^m} elements
and F_2 vectors.
"""

from __future__ import annotations

import numpy as np


def bits_of(GF, elems):
    """Field elements -> (len, m) uint8 array of coefficient bits."""
    m = GF.degree
    u = np.asarray(elems.view(np.ndarray)).astype(np.uint64).ravel()
    return ((u[:, None] >> np.arange(m, dtype=np.uint64)) & 1).astype(np.uint8)


def elems_of_bits(GF, bits):
    """(len, m) uint8 bit array -> field elements."""
    m = GF.degree
    u = (bits.astype(np.uint64)
         * (np.uint64(1) << np.arange(m, dtype=np.uint64))).sum(axis=1)
    return GF(u.astype(np.uint64))


def f2_rref(M):
    """RREF of a uint8 matrix over F_2 -> (R, pivot_cols)."""
    R = (M.copy() % 2).astype(np.uint8)
    rows, cols = R.shape
    pivots, r = [], 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if R[i, c]), None)
        if piv is None:
            continue
        R[[r, piv]] = R[[piv, r]]
        for i in range(rows):
            if i != r and R[i, c]:
                R[i] ^= R[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return R, pivots


def f2_kernel(M):
    """Basis of {x : M x = 0} over F_2, one basis vector per row."""
    R, pivots = f2_rref(M)
    cols = M.shape[1]
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for i, p in enumerate(pivots):
            if R[i, f]:
                v[p] = 1
        basis.append(v)
    return np.array(basis, dtype=np.uint8).reshape(len(basis), cols)


def f2_rank(M):
    return len(f2_rref(M)[1])


def f2_solve(M, rhs):
    """A particular solution of M x = rhs over F_2 (free variables = 0).

    Raises ValueError if the system is inconsistent.
    """
    A = np.hstack([M % 2, (np.asarray(rhs) % 2).reshape(-1, 1).astype(np.uint8)])
    R, pivots = f2_rref(A)
    pc = set(pivots)
    for i in range(R.shape[0]):
        if R[i, -1] and all(R[i, c] == 0 for c in range(M.shape[1])):
            raise ValueError("inconsistent F_2 system")
    x = np.zeros(M.shape[1], dtype=np.uint8)
    for i, p in enumerate(pivots):
        x[p] = R[i, -1]
    return x
