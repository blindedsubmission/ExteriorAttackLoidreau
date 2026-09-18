#!/usr/bin/env python3
"""Coding-theory primitives of the attacked schemes.

Field construction over GF(2^m), Gabidulin Moore matrices and supports,
F2-rank of secret coefficient spaces, random masks with entries in a
lambda-dimensional F_q-subspace, random codes, and entrywise Frobenius
shifts.  Shared building blocks for the key generators (schemes.py), the
exterior linear algebra (exterior.py), and everything built on top.
"""

from __future__ import annotations

import galois
import numpy as np


def make_field(m: int, poly: int | None = None):
    """Construct GF(2^m).

    poly: optional irreducible polynomial as an integer bitmask (bit i =
    the coefficient of x^i, bit m set -- the convention of instances.py's
    meta "poly" and of matrixfree-rs meta.json's poly_g_lo/poly_g_hi plus
    the implicit x^m term).  None selects Conway where available, else
    galois's primitive_poly search (the historical default)."""
    if poly is None:
        try:
            return galois.GF(2 ** m)
        except LookupError:
            # Explicit primitive polynomial for large m not in Conway's
            # database.  Warn: the field (and hence every kernel scalar)
            # then depends on the installed galois version.  Engine-run
            # artifacts are unaffected -- meta.json records the modulus.
            print(f"make_field: m={m} not in Conway's database; using "
                  "galois's primitive_poly fallback", flush=True)
            return galois.GF(2 ** m, irreducible_poly=galois.primitive_poly(2, m))
    degrees = [i for i in range(m, -1, -1) if (int(poly) >> i) & 1]
    p = galois.Poly.Degrees(degrees)
    if p.degree != m:
        raise ValueError(f"polynomial {poly:#x} has degree {p.degree}, want m={m}")
    return galois.GF(2 ** m, irreducible_poly=p)


def field_from_meta(meta: dict):
    """GF(2^m) in which an instance's key material actually lives: honor a
    matrixfree-rs meta (poly_g_lo/poly_g_hi), then a Python instances meta
    ("poly"), else the historical Conway/primitive default.  Recovering a
    key in any other field than its own makes every check fail loudly."""
    m = int(meta["m"])
    if "poly_g_lo" in meta:
        poly = int(meta["poly_g_lo"]) | (int(meta["poly_g_hi"]) << 64) | (1 << m)
        return make_field(m, poly)
    if "poly" in meta:
        return make_field(m, int(meta["poly"]))
    return make_field(m)


# ===========================================================================
# Gabidulin / Loidreau key generation
# ===========================================================================

def moore_matrix(GF, b, rows: int):
    """Moore matrix M_rows(b): row i is b^{[i]} (i-fold Frobenius, entrywise).

    b is a 1-D GF array of length n.  Frobenius is x -> x^2 in GF(2^m).
    """
    n = len(b)
    M = GF.Zeros((rows, n))
    for i in range(rows):
        M[i, :] = b ** (2 ** i)
    return M


def random_full_rank_weight_n(GF, n: int, rng) -> "galois.FieldArray":
    """Random vector a in GF(2^m)^n whose entries are F_2-linearly independent
    (rank weight n).  Required for a Gabidulin generator of an [n,k] code.
    """
    m = GF.degree
    assert n <= m
    while True:
        a = GF.Random(n, seed=rng)
        if _f2_rank(a, m) == n:
            return a


def _f2_rank(vecs, m: int) -> int:
    """F_2-rank of a set of m-bit integers (viewed as bit vectors).

    Accepts a galois FieldArray or any iterable of integers.
    """
    arr = [int(v) for v in np.asarray(vecs).ravel()]
    rank = 0
    for bit in range(m):
        pivot = None
        for idx in range(rank, len(arr)):
            if (arr[idx] >> bit) & 1:
                pivot = idx
                break
        if pivot is None:
            continue
        arr[rank], arr[pivot] = arr[pivot], arr[rank]
        pv = arr[rank]
        for idx in range(len(arr)):
            if idx == rank:
                continue
            if (arr[idx] >> bit) & 1:
                arr[idx] ^= pv
        rank += 1
    return rank


def random_invertible(GF, n: int, rng):
    """Random invertible n x n matrix over GF(2^m)."""
    while True:
        M = GF.Random((n, n), seed=rng)
        try:
            _ = np.linalg.inv(M)
            return M
        except np.linalg.LinAlgError:
            continue


def random_mask_in_subspace(GF, n: int, subspace, rng, return_components=False):
    """Random invertible n x n matrix whose entries span `subspace` over F_2.

    Optionally also return its binary coefficient matrices in that basis.
    """
    lam = len(subspace)
    sub = GF(subspace)
    while True:
        # Each entry = random F_2 combination of the basis elements.
        coeffs = GF.Random((n, n, lam), seed=rng)  # 0 or 1 in GF
        # Reduce coeffs to actual 0/1 integers
        coeffs01 = (coeffs.view(np.ndarray) & 1)
        # entry[i,j] = XOR (GF sum) of sub[t] where coeffs01[i,j,t]=1
        entries = GF.Zeros((n, n))
        for t in range(lam):
            mask = coeffs01[:, :, t].astype(bool)
            entries[mask] = entries[mask] + sub[t]
        if _f2_rank(entries, GF.degree) != lam:
            continue
        try:
            _ = np.linalg.inv(entries)
            if return_components:
                return entries, GF(coeffs01.transpose(2, 0, 1))
            return entries
        except np.linalg.LinAlgError:
            continue


def make_loidreau_key(GF, n: int, k: int, lam: int, rng):
    """Generate a Loidreau public key.

    Returns (G_pub, G_sec, S, P, P_inv) where G_pub = S * G_sec * P^{-1}.
    P is invertible with entries in a lambda-dim F_q-subspace.
    """
    a = random_full_rank_weight_n(GF, n, rng)
    G_sec = moore_matrix(GF, a, k)

    S = random_invertible(GF, k, rng)

    # Lambda-dim F_q-subspace of GF(2^m).
    while True:
        sub_vals = GF.Random(lam, seed=rng)
        if _f2_rank(sub_vals, GF.degree) == lam:
            break

    P = random_mask_in_subspace(GF, n, sub_vals, rng)
    P_inv = np.linalg.inv(P)

    G_pub = S @ G_sec @ P_inv
    return G_pub, G_sec, S, P, P_inv


def random_code(GF, n: int, k: int, rng):
    """Random full-rank k x n matrix over GF(2^m) (a 'random code')."""
    while True:
        G = GF.Random((k, n), seed=rng)
        if np.linalg.matrix_rank(G) == k:
            return G


def frobenius_shift(GF, M, shifts: int):
    """Apply Frobenius (x -> x^{2^shifts}) entrywise to matrix M.

    Inverse Frobenius by i is x -> x^{2^(m-i mod m)} for i>0.
    """
    m = GF.degree
    if shifts == 0:
        return M
    e = (shifts % m)
    return M ** (2 ** e)
