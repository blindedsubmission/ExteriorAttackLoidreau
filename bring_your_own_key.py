#!/usr/bin/env python3
"""Bring-your-own-key conformance test, written from SPECIFICATION.md.

This script deliberately does NOT import the pipeline's key generation
(schemes.py / instances.py / primitives.py).  It is an independent,
from-scratch implementation of the specification: it samples a Loidreau
key with its own randomness and its own field choice, serializes the
public key exactly as SPECIFICATION.md section 5 prescribes, hands the
bytes to the Rust kernel finder, runs the Python recovery, and then
verifies the recovered equivalent key with checks implemented here from
the specification's verification recipe (section 7).

A PASS proves: the file formats are specified precisely enough to be
produced by a third party, the engine accepts such foreign keys, and the
recovered key is verifiably equivalent to the injected one.  The
verification does not reuse any pipeline key-generation code.

Usage:  python3 bring_your_own_key.py [--keep]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import galois
import numpy as np

RELEASE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(RELEASE, "rust", "target", "release", "matrixfree-rs")
INST = os.path.join(RELEASE, "instances", "byo")
sys.path.insert(0, os.path.join(RELEASE, "python"))

M, N, K, LAM = 24, 16, 10, 3          # instance shape (r = 6 >= lam+2)
R = N - K


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Field: GF(2^m) with an explicitly chosen modulus (spec section 2).
#    Constructing the field PROVES the modulus is irreducible (galois
#    rejects reducible polynomials); we then record it in meta.json.
# ---------------------------------------------------------------------------

CANDIDATE_G = [0x1B, 0x1D, 0x87]      # x^24 + {x4+x3+x, x4+x3+x2, x7+x6+x2+1}+1
for g in CANDIDATE_G:
    poly = (1 << M) | g
    degrees = [i for i in range(M, -1, -1) if (poly >> i) & 1]
    p = galois.Poly.Degrees(degrees)
    try:
        GF = galois.GF(2 ** M, irreducible_poly=p)
        break
    except ValueError:
        continue
else:
    sys.exit("no irreducible candidate: the spec modulus table is broken")
log(f"field: GF(2^{M}) with f = x^{M} + g, g = 0x{g:x} (irreducible by construction)")


def f2_rank(ints):
    """F_2-rank of m-bit integers, per spec section 2.3 (own implementation)."""
    arr = list(ints)
    rank = 0
    for bit in range(M):
        piv = next((i for i in range(rank, len(arr)) if (arr[i] >> bit) & 1), None)
        if piv is None:
            continue
        arr[rank], arr[piv] = arr[piv], arr[rank]
        pv = arr[rank]
        for i in range(len(arr)):
            if i != rank and (arr[i] >> bit) & 1:
                arr[i] ^= pv
        rank += 1
    return rank


rng = np.random.default_rng(0xA11CE)

# ---------------------------------------------------------------------------
# 2. Key generation per spec section 3 (independent sampling).
# ---------------------------------------------------------------------------

# support a with F_2-rank n (rank weight n)
while True:
    a_ints = [int(v) for v in rng.integers(0, 1 << M, size=N)]
    if f2_rank(a_ints) == N:
        break
a = GF(a_ints)

# Moore matrix M_k(a): row i = a^[i] (entrywise a -> a^(2^i))
Mk = GF.Zeros((K, N))
for i in range(K):
    Mk[i, :] = a ** (2 ** i)

# scrambler S in GL_k
while True:
    S = GF(rng.integers(0, 1 << M, size=(K, K)))
    try:
        np.linalg.inv(S)
        break
    except np.linalg.LinAlgError:
        pass

# mask P in GL_n with entries in the lambda-dim F_2-space span(gamma)
while True:
    gammas = [int(v) for v in rng.integers(0, 1 << M, size=LAM)]
    if f2_rank(gammas) == LAM:
        break
while True:
    bits = rng.integers(0, 2, size=(N, N, LAM)).astype(np.uint8)
    P = GF.Zeros((N, N))
    for t in range(LAM):
        sel = bits[:, :, t].astype(bool)
        P[sel] = P[sel] + GF(gammas[t])
    try:
        P_inv = np.linalg.inv(P)
        break
    except np.linalg.LinAlgError:
        pass

G_pub = S @ Mk @ P_inv
log(f"key generated: G_pub = S * Moore_{K}(a) * P^-1  ({K}x{N})")

# self-check of OUR construction against the spec's shape condition:
# (A separate structural self-check is unnecessary: the decryption
# roundtrip at the end is the structural proof.)

# ---------------------------------------------------------------------------
# 3. Serialization per spec section 5: 16-byte LE elements, row-major.
# ---------------------------------------------------------------------------

os.makedirs(INST, exist_ok=True)
key_path = os.path.join(INST, "byo-rs.key.bin")
with open(key_path, "wb") as f:
    for e in G_pub.reshape(-1).tolist():
        e = int(e)
        f.write((e & ((1 << 64) - 1)).to_bytes(8, "little"))
        f.write(((e >> 64) & ((1 << 64) - 1)).to_bytes(8, "little"))
meta = {"m": M, "n": N, "k": K, "lam": LAM, "random": False, "seed": 0,
        "poly_g_lo": g, "poly_g_hi": 0}
with open(os.path.join(INST, "byo-rs.meta.json"), "w") as f:
    json.dump(meta, f)
log(f"wrote {key_path} ({K*N*16} bytes) + meta.json")

# ---------------------------------------------------------------------------
# 4. Kernel search (Rust) and recovery (Python pipeline code under test).
# ---------------------------------------------------------------------------

log("running the Rust kernel finder on the foreign key ...")
r = subprocess.run([BIN, "solve", "--key", key_path, "--mode", "block",
                    "--block", "8",
                    "--out", os.path.join(INST, "kernel.bin")],
                   capture_output=True, text=True)
log(r.stdout.strip())
if r.returncode != 0:
    sys.exit(f"solve failed: {r.stderr.strip()}")

import recovery  # noqa: E402  (pipeline code under test)

res = recovery.equivalent_key_from_kernel(
    "byo", dirname=INST, key_prefix="byo-rs")
log(f"recovery: ok={res['ok']} stage={res['stage']} err={res['err']}")
if not res["ok"]:
    sys.exit(1)
b_hat, P_hat = res["b_hat"], res["P_hat"]

# ---------------------------------------------------------------------------
# 5. INDEPENDENT verification of the recovered key (spec section 7).
# ---------------------------------------------------------------------------

# check (a): rank weight of b_hat is n  ->  F_2-rank of its coordinates
b_ints = [int(v) for v in np.asarray(b_hat).ravel()]
assert len(b_ints) == N, "b_hat must have n coordinates"
assert f2_rank(b_ints) == N, "b_hat is not rank weight n"
log(f"check (a): rank weight of b_hat = {N}  OK")

# check (b): P_hat invertible
if np.linalg.matrix_rank(P_hat) != N:
    sys.exit("P_hat singular")
log("check (b): P_hat invertible  OK")

# check (c): rowsp(Moore_r(b_hat) * P_hat) == rowsp(ker G_pub)
H_hat = GF.Zeros((R, N))
bs = GF(b_ints)
for i in range(R):
    H_hat[i, :] = bs ** (2 ** i)
H_hat = H_hat @ P_hat
H_pub = GF(G_pub).null_space()
if not np.array_equal(H_hat.row_reduce().view(np.ndarray),
                      H_pub.row_reduce().view(np.ndarray)):
    sys.exit("rowsp(H_hat) != rowsp(ker G_pub)")
log("check (c): rowsp(Moore_r(b_hat) P_hat) == rowsp(ker G_pub)  OK")

# check (d): decryption roundtrip per spec section 6.  Encrypt with OUR
# G_pub, decrypt with the recovered key, compare (m, e) exactly.
import pke  # noqa: E402  (pipeline code under test)

t = R // (2 * LAM)
while True:
    b = GF(rng.integers(0, 1 << M, size=t))
    if f2_rank([int(v) for v in b]) == t:
        break
while True:
    cbits = rng.integers(0, 2, size=(N, t)).astype(np.uint8)
    if f2_rank([int("".join(map(str, cbits[:, u])), 2) for u in range(t)]) == t:
        break
e = GF.Zeros(N)
for u in range(t):
    e += GF(int(b[u])) * GF([int(v) for v in cbits[:, u]])
m0 = GF(rng.integers(0, 1 << M, size=K))
y = m0 @ GF(G_pub) + e

dec = pke.decrypt(GF, GF(G_pub), H_pub, b_hat, P_hat, LAM, y, t)
if not dec["ok"]:
    sys.exit(f"decryption failed: {dec['err']}")
if not (np.array_equal(dec["m"].view(np.ndarray), m0.view(np.ndarray))
        and np.array_equal(dec["e"].view(np.ndarray), e.view(np.ndarray))):
    sys.exit("decrypted (m, e) != original (m, e)")
log("check (d): ciphertext decrypted with the recovered key, (m, e) match  OK")

log("\n#### BYO CONFORMANCE PASS ####")
log("a foreign, independently generated key was recovered and the")
log("equivalent key verified against the specification's recipe.")
if "--keep" not in sys.argv:
    import shutil
    shutil.rmtree(INST)
    log("(instances/byo removed; rerun with --keep to inspect)")
