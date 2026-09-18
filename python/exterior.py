#!/usr/bin/env python3
"""Exterior-power linear algebra of the attack.

Compound (Pluecker) matrices of Frobenius-shifted generator matrices, the
stacked matrix whose kernel carries the hidden-plane witness, the block
threshold s* = max(row threshold, Schubert incidence threshold), and the
anchored-route helpers (anchor kernels, shifted blocks) used by the
instance builder and the GPU Q-table generator.  See ARCHITECTURE.md
Stages 1-2 and main.tex for the mathematics.
"""

from __future__ import annotations

from itertools import combinations
from math import comb, ceil

import numpy as np

from primitives import frobenius_shift


def _perm_sign(sigma):
    """Sign of a permutation sigma (as a tuple), via inversion count."""
    s = 1
    for i in range(len(sigma)):
        for j in range(i + 1, len(sigma)):
            if sigma[i] > sigma[j]:
                s = -s
    return s
def compound_matrix(GF, A, lam: int):
    """The lam-th compound (matrix of lam x lam minors) of A.

    A is a GF array p x q.  Result is C(p,lam) x C(q,lam).  Entry (I,T) is the
    determinant of the lam x lam submatrix A[I, T].

    Vectorized: gather all lam-submatrix blocks at once via fancy indexing, then
    compute all determinants in a batch.  For small lam (<=5) the Leibniz
    expansion over permutations has only lam! terms and is applied elementwise
    across the whole batch, avoiding a Python loop over minors.
    """
    from itertools import permutations

    p, q = A.shape
    row_idx = np.array(list(combinations(range(p), lam)))  # (C(p,lam), lam)
    col_idx = np.array(list(combinations(range(q), lam)))  # (C(q,lam), lam)
    nrows, ncols = len(row_idx), len(col_idx)

    # Gather all blocks: shape (nrows, ncols, lam, lam).
    # blocks[i, j, a, b] = A[row_idx[i,a], col_idx[j,b]]
    blocks = A[np.ix_(
        row_idx.reshape(-1),               # (nrows*lam,)
        col_idx.reshape(-1),               # (ncols*lam,)
    )]  # shape (nrows*lam, ncols*lam)
    blocks = blocks.reshape(nrows, lam, ncols, lam).swapaxes(1, 2)  # (nrows,ncols,lam,lam)

    # Determinant via Leibniz: sum over sigma in S_lam of sign(sigma)*prod A[i,sigma(i)].
    # In characteristic 2, sign is irrelevant (all +1), so det = sum of prod over permutations.
    # For general GF(2^m) (char 2), sign(sigma) = 1 always.  (For odd char, use sign.)
    sign_is_trivial = (GF.characteristic == 2)

    perms = list(permutations(range(lam)))
    det = GF.Zeros((nrows, ncols))
    Av = blocks.view(np.ndarray) if hasattr(blocks, 'view') else np.array(blocks)
    # Work in the GF array's arithmetic for correct field operations.
    for sigma in perms:
        # product of blocks[i,j,a,sigma(a)] over a
        prod = GF.Ones((nrows, ncols))
        for a in range(lam):
            prod = prod * blocks[:, :, a, sigma[a]]
        if sign_is_trivial:
            det = det + prod
        else:
            sign = _perm_sign(sigma)
            # GF(-1) is rejected by galois for prime fields; negate instead.
            det = det + (prod if sign == 1 else -prod)
    return det
def stacked_compound(GF, G, lam: int, s: int):
    """Stack s blocks: Comp_lam(G), Comp_lam(G^{[-1]}), ..., Comp_lam(G^{[-(s-1)]}).

    G^{[-i]} = i-fold inverse Frobenius applied entrywise.
    """
    m = GF.degree
    blocks = []
    for i in range(s):
        if i == 0:
            Gi = G
        else:
            Gi = frobenius_shift(GF, G, m - (i % m))  # inverse Frobenius
        blocks.append(compound_matrix(GF, Gi, lam))
    return GF(np.vstack(blocks))
def shifted_blocks(GF, G, s: int):
    """The s shifted matrices G, G^{[-1]}, ..., G^{[-(s-1)]}."""
    m = GF.degree
    return [G if i == 0 else frobenius_shift(GF, G, m - (i % m)) for i in range(s)]


# ===========================================================================
# Anchored route: compress the N = C(n,lam) wedge coordinates to
# D = (n-k)^lam by pinning the lam wedge factors into the kernels of the
# anchor blocks 0..lam-1 (main.tex, "Anchored compound recovery").
# ===========================================================================

def _anchor_data(GF, G, lam: int):
    """Row bases of ker(G), ker(G^{[-1]}), ..., ker(G^{[-(lam-1)]})."""
    m = GF.degree
    N0 = G.null_space()  # (n-k, n) row basis of ker G
    return [frobenius_shift(GF, N0, m - (a % m)) for a in range(lam)]

def required_blocks(n: int, k: int, lam: int) -> int:
    """Least block count not ruled out by rows or Schubert incidence."""
    N = comb(n, lam)
    B = comb(k, lam)
    s_row = ceil(N / B)
    s_inc = lam * (n - lam) // (k - lam + 1) + 1
    return max(s_row, s_inc)
