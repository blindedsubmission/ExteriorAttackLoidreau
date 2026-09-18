#!/usr/bin/env python3
"""Gabidulin syndrome decoding for rank-t errors (t <= r/2).

Setting: a vector g in GF(2^m)^n whose entries are F_2-independent, and
the twisted syndromes of an unknown x in GF(2^m)^n of rank weight <= t,

    tau_s = sum_j g_j^{[s]} x_j,   s = 0..r-1   (r = n - k),

with 2t <= r.  The decoder recovers x in two steps:

  * Annihilator system.  The coordinates x_j span an F_2-space V of
    dimension <= t, so the monic linearized polynomial
    N(z) = z^{[t]} + sum_{l<t} nu_l z^{[l]} of q-degree t annihilates
    every x_j.  Contracting N(x_j) = 0 against g^{[i]} gives, for
    i = t..r-1,

        sum_{l<t} nu_l tau_{i-l}^{[l]} = tau_{i-t}^{[t]},

    because tau_{i-l}^{[l]} = sum_j g_j^{[i]} x_j^{[l]}.  These r-t
    equations over GF(2^m) determine nu uniquely: a difference of two
    solutions is a linearized polynomial of q-degree < t vanishing on
    V, which is zero when dim V = t.
  * Coordinates.  V = ker_F2(N) (dimension exactly t); writing
    x_j = sum_u c_{j,u} b_u over an F_2-basis b of V turns the
    syndromes into an r*m x n*t F_2 system for c with a unique
    solution.

All operations are exact and the decoder verifies its own output
(twisted syndromes of the result match the input) before returning;
failures raise ValueError.
"""

from __future__ import annotations

import numpy as np

from f2 import bits_of, elems_of_bits, f2_kernel, f2_solve


def fro(x, i: int):
    """i-th Frobenius of field elements / GF arrays: x -> x^(2^i)."""
    return x ** (2 ** i)


def gf_solve(A, B):
    """Unique solution X of A X = B over GF(2^m).

    A is full column rank (square or tall), B has one column or as many
    as A has rows.  Raises ValueError if A is rank deficient or the
    system is inconsistent.
    """
    GF = A.__class__
    B2 = B if B.ndim == 2 else B.reshape(-1, 1)
    Aug = np.hstack([A, B2])
    R = Aug.row_reduce()
    ncol = A.shape[1]
    X = GF.Zeros((ncol, B2.shape[1]))
    pivots = {}
    for i in range(R.shape[0]):
        nz = np.nonzero(R[i, :ncol])[0]
        if len(nz):
            pivots[nz[0]] = i
        elif np.any(R[i, ncol:]):
            raise ValueError("inconsistent GF system")
    if len(pivots) != ncol:
        raise ValueError(f"GF system rank deficient "
                         f"({len(pivots)} < {ncol} pivots)")
    for c, i in pivots.items():
        X[c] = R[i, ncol:]
    return X


def moore_syndromes(GF, g, x, rows: int):
    """Twisted syndromes tau_s = g^{[s]} . x for s = 0..rows-1."""
    M = GF.Zeros((rows, len(g)))
    for s in range(rows):
        M[s, :] = g ** (2 ** s)
    return M @ x


def decode_syndrome(GF, g, tau, t: int):
    """Recover x of rank weight exactly t from tau = (g^{[s]} . x)_s.

    tau holds the twisted syndromes for s = 0..len(tau)-1; the usable
    window is s = 0..r-1 with r = len(tau), requiring 2t <= r.
    """
    n, r = len(g), len(tau)
    if t == 0:
        x = GF.Zeros(n)
        if np.any(moore_syndromes(GF, g, x, r).view(np.ndarray)
                  != tau.view(np.ndarray)):
            raise ValueError("t=0 but syndromes nonzero")
        return x
    if 2 * t > r:
        raise ValueError(f"t={t} exceeds the unique-decoding radius "
                         f"r/2={r // 2}")

    # annihilator system: rows i = t..r-1, unknowns nu_0..nu_{t-1}
    M = GF.Zeros((r - t, t))
    rhs = GF.Zeros(r - t)
    for row, i in enumerate(range(t, r)):
        for l in range(t):
            M[row, l] = fro(tau[i - l], l)
        rhs[row] = fro(tau[i - t], t)
    nu = gf_solve(M, rhs).ravel()

    # N as an F_2-linear map: column b is N(2^b); V = ker N
    m = GF.degree
    K = np.zeros((m, m), dtype=np.uint8)
    for b in range(m):
        zb = GF(np.uint64(1) << b)
        val = fro(zb, t)
        for l in range(t):
            val = val + GF(np.uint64(int(nu[l]))) * fro(zb, l)
        K[:, b] = bits_of(GF, val.reshape(1)).ravel()
    Kbasis = f2_kernel(K)
    if Kbasis.shape[0] != t:
        raise ValueError(f"dim ker N = {Kbasis.shape[0]} != t = {t}; "
                         "error support smaller than expected")
    B = elems_of_bits(GF, Kbasis)          # F_2-basis of the support V

    # coordinates: x_j = sum_u c_{j,u} b_u -> r*m x n*t F_2 system
    # tau_s = sum_{j,u} c_{j,u} g_j^{[s]} b_u, F_2-linear in c
    M2 = np.zeros((r * m, n * t), dtype=np.uint8)
    for s in range(r):
        gs = g ** (2 ** s)                       # twisted g row s
        contrib = gs[:, None] * B[None, :]       # (n, t) field elements
        cbits = bits_of(GF, contrib).reshape(n, t, m)   # (n, t, m)
        for b in range(m):
            M2[s * m + b, :] = cbits[:, :, b].reshape(-1)
    rhs2 = bits_of(GF, tau).ravel()              # (r*m,) row-major (s, b)
    c = f2_solve(M2, rhs2).reshape(n, t)
    bits_x = np.zeros((n, m), dtype=np.uint8)
    bitsB = bits_of(GF, B)                       # (t, m)
    for u in range(t):
        bits_x ^= c[:, u, None] * bitsB[u][None, :]
    x = elems_of_bits(GF, bits_x)

    # self-verification: the result must reproduce the given syndromes
    if np.any(moore_syndromes(GF, g, x, r).view(np.ndarray)
              != tau.view(np.ndarray)):
        raise ValueError("decoded word does not match the syndromes")
    return x
