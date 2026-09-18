#!/usr/bin/env python3
"""Key generators of the two attacked schemes.

Both share the same key structure (ARCHITECTURE.md, Stage 0): a secret
Gabidulin code and an invertible mask with entries in a secret
lambda-dimensional F_q-subspace.  Loidreau's scheme publishes the primal
generator; LowMS publishes a Niederreiter parity-check.  The differences
are on the encryption side and never enter key recovery.
"""

from __future__ import annotations

import numpy as np

from primitives import (_f2_rank, moore_matrix, random_full_rank_weight_n,
                        random_invertible, random_mask_in_subspace)


def make_loidreau_key(GF, n: int, k: int, lam: int, rng):
    """Generate a Loidreau public key.

    Returns (G_pub, G_sec, S, P, P_inv) where G_pub = S * G_sec * P^{-1}.
    P is invertible with entries in a lambda-dim F_q-subspace.
    """
    b = random_full_rank_weight_n(GF, n, rng)
    G_sec = moore_matrix(GF, b, k)

    S = random_invertible(GF, k, rng)

    # Lambda-dim F_q-subspace of GF(2^m).
    while True:
        vsec = GF.Random(lam, seed=rng)
        if _f2_rank(vsec, GF.degree) == lam:
            break

    P = random_mask_in_subspace(GF, n, vsec, rng)
    P_inv = np.linalg.inv(P)

    G_pub = S @ G_sec @ P_inv
    return G_pub, G_sec, S, P, P_inv
def make_lowms_key(GF, n: int, k: int, lam: int, rng):
    """Generate a LowMS public key (Aragon et al. 2024, Algorithm 2).

    Niederreiter form: the published parity-check is a systematic form of
    S^T H P^T, where H is a parity-check of the secret [n,k] Gabidulin code
    and P is an invertible perturbation with entries in a lambda-dimensional
    F_q-subspace F.  The row scrambler only changes the displayed basis, so
    the published row space is rowsp(H P^T) and the primal public code is
    rowsp(Moore^k(b) P^{-1}) -- exactly the distribution the exterior
    attack targets.

    Returns (H_pub, G_pub, a, P, P_inv):
      H_pub  (n-k) x n systematic parity-check (the public key),
      G_pub  k x n generator of the primal public code, computed the way an
             attacker derives it from H_pub (null space),
      b      secret Gabidulin support (rank weight n),
      P, P_inv  secret mask and its inverse.
    """
    b = random_full_rank_weight_n(GF, n, rng)
    G_sec = moore_matrix(GF, b, k)
    H = G_sec.null_space()
    assert H.shape == (n - k, n)
    while True:
        vsec = GF.Random(lam, seed=rng)
        if _f2_rank(vsec, GF.degree) == lam:
            break
    P = random_mask_in_subspace(GF, n, vsec, rng)
    P_inv = np.linalg.inv(P)
    M = H @ P.T
    H_pub = M.row_reduce()
    assert H_pub.shape == (n - k, n)
    G_pub = H_pub.null_space()
    assert G_pub.shape == (k, n)
    return H_pub, G_pub, b, P, P_inv
