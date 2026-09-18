#!/usr/bin/env python3
"""Loidreau encryption and decryption with the recovered equivalent key.

Encryption is the public Loidreau-2017 PKE: y = m G_pub + e with a
random message m and a random error of rank weight t.  Decryption uses
ONLY the equivalent key (b_hat, P_hat) recovered from the public code
by recovery.py Step 5 -- never the secret (b, P).  The chain:

    H_hat = Moore_r(b_hat) P_hat        (rowsp(H_hat) = rowsp(ker G_pub),
                                         verified during recovery)
    U H_pub = H_hat                     (U solved from public data)
    sigma = U H_pub y^T
          = H_hat e^T = Moore_r(b_hat) (e P_hat^T)^T
    x = e P_hat^T                       (rank <= lam*t: the mask blows the
                                         error up into the lam-dim space
                                         spanned by gamma_hat's entries)
    x <- decode_syndrome (decoding.py, needs lam*t <= r/2)
    e = x P_hat^{-T},  m from (y - e) = m G_pub.

Every step is verified against the public key (syndrome match, rank
weight, message consistency), so a run reporting ok=True is a
per-ciphertext proof that the recovered key decrypts.
"""

from __future__ import annotations

import numpy as np

from decoding import decode_syndrome, gf_solve
from f2 import bits_of, elems_of_bits, f2_rank
from primitives import _f2_rank, moore_matrix


def sample_rank_error(GF, n, t, rng):
    """Random vector of rank weight exactly t.

    Model: coordinates uniform in a t-dimensional F_2-space V with
    basis b: e_j = sum_u c_{j,u} b_u, c in F_2^{n x t}; the rank is t
    iff c has F_2-column rank t (generic).  Field-coefficient products
    b_u * alpha_u would NOT give rank t.
    """
    while True:
        b = GF.Random(t, seed=rng)
        if _f2_rank(b, GF.degree) != t:
            continue
        c = rng.integers(0, 2, size=(n, t)).astype(np.uint8)
        if f2_rank(c) != t:
            continue
        bits_e = np.zeros((n, GF.degree), dtype=np.uint8)
        bitsB = bits_of(GF, b)
        for u in range(t):
            bits_e ^= c[:, u, None] * bitsB[u][None, :]
        return elems_of_bits(GF, bits_e)


def encrypt(GF, G_pub, t, rng):
    """Public-key encryption: y = m G_pub + e with rank(e) = t."""
    k, n = G_pub.shape
    m = GF.Random(k, seed=rng)
    e = sample_rank_error(GF, n, t, rng)
    y = m @ G_pub + e
    return y, m, e


def decrypt(GF, G_pub, H_pub, b_hat, P_hat, lam, y, t):
    """Decrypt y with the equivalent key (b_hat, P_hat).

    Returns a result dict in the recovery.py style: ok / stage / err,
    with m and e on success.
    """
    r, n = H_pub.shape
    try:
        H_hat = moore_matrix(GF, b_hat, r) @ P_hat
        U = gf_solve(H_pub.T, H_hat.T).T            # U H_pub = H_hat
        c = H_pub @ y.T                            # = H_pub e^T
        sigma = U @ c                              # = Moore_r(b_hat) x^T
        x = decode_syndrome(GF, b_hat, sigma, lam * t)
        e = x @ np.linalg.inv(P_hat).T
        m = gf_solve(G_pub.T, (y - e).reshape(-1, 1)).ravel()
    except ValueError as exc:
        return {"ok": False, "stage": "decrypt", "err": str(exc)}
    checks = [
        (np.array_equal((m @ G_pub + e).view(np.ndarray), y.view(np.ndarray)),
         "m G_pub + e != y"),
        (np.array_equal((H_pub @ e.T).view(np.ndarray), c.view(np.ndarray)),
         "syndrome of recovered error != ciphertext syndrome"),
        (_f2_rank(e, GF.degree) == t, f"rank(e) = {_f2_rank(e, GF.degree)} != {t}"),
    ]
    for ok, msg in checks:
        if not ok:
            return {"ok": False, "stage": "verify", "err": msg}
    return {"ok": True, "stage": "done", "err": None, "m": m, "e": e}


# ===========================================================================
# Drivers
# ===========================================================================

def _roundtrip(GF, G_pub, b_hat, P_hat, lam, t, trials, rng_seed):
    """Encrypt `trials` ciphertexts publicly, decrypt each with the
    equivalent key; every trial must recover (m, e) exactly."""
    H_pub = G_pub.null_space()
    assert H_pub.shape == (G_pub.shape[1] - G_pub.shape[0], G_pub.shape[1])
    lam_ok = 2 * lam * t <= H_pub.shape[0]
    print(f"encrypting {trials} ciphertext(s): t={t}, masked rank "
          f"lam*t={lam * t} vs r/2={H_pub.shape[0] // 2}"
          f"{'' if lam_ok else '  -- EXCEEDS unique-decoding radius'}",
          flush=True)
    if not lam_ok:
        return False
    allok = True
    for trial in range(trials):
        rng = np.random.default_rng(rng_seed + trial)
        y, m, e = encrypt(GF, G_pub, t, rng)
        res = decrypt(GF, G_pub, H_pub, b_hat, P_hat, lam, y, t)
        if res["ok"]:
            same = (np.array_equal(res["m"].view(np.ndarray), m.view(np.ndarray))
                    and np.array_equal(res["e"].view(np.ndarray), e.view(np.ndarray)))
            print(f"trial {trial}: decrypted with the recovered key, "
                  f"(m, e) match: {'YES' if same else 'NO'}", flush=True)
            allok &= same
        else:
            print(f"trial {trial}: DECRYPTION FAILED at {res['stage']}: "
                  f"{res['err']}", flush=True)
            allok = False
    return allok


def pke_toy_case(m, n, k, lam, seed, t):
    """Dense-route toy: Loidreau key -> exterior kernel -> equivalent key
    -> public encryption -> equivalent-key decryption."""
    from exterior import stacked_compound
    from itertools import combinations
    from recovery import contract_pluecker, recover_key_from_plane, verify_plane
    from primitives import make_field
    from schemes import make_loidreau_key

    GF = make_field(m)
    rng = np.random.default_rng(seed)
    G_pub, G_sec, S, P, P_inv = make_loidreau_key(GF, n, k, lam, rng)
    print(f"=== pke toy ({m};{n},{k},{lam}) seed={seed} t={t} ===", flush=True)
    W = stacked_compound(GF, G_pub, lam, n - k)
    kern = W.null_space()
    if kern.shape[0] != 1:
        print(f"kernel dim {kern.shape[0]} != 1 -> plane extraction inconclusive",
              flush=True)
        return False
    idx = np.array(list(combinations(range(n), lam)))
    R = contract_pluecker(GF, kern[0], idx, n, lam, rng)
    ok, msg = verify_plane(GF, G_pub, R)
    if not ok:
        print(f"plane verification: {msg}", flush=True)
        return False
    res = recover_key_from_plane(GF, G_pub, R, verbose=False)
    print(f"equivalent-key recovery: ok={res['ok']} "
          f"stage={res['stage']} err={res['err']}", flush=True)
    if not res["ok"]:
        return False
    return _roundtrip(GF, G_pub, res["b_hat"], res["P_hat"], lam, t,
                      trials=2, rng_seed=seed)


def demo_case(name, t=None, trials=3, seed=7):
    """Instance demo: kernel.bin -> equivalent key -> encrypt with the
    public key -> decrypt with the equivalent key.  Works for the Python
    instances (key.npz) and for matrixfree-rs runs (key.bin, 16-byte
    elements, field from meta poly_g_lo/poly_g_hi)."""
    import json
    import os

    from instances import build, inst_dir, load_registry
    from primitives import field_from_meta
    from recovery import _read_fe_file, equivalent_key_from_kernel

    dirname = inst_dir(name)
    key_prefix = (f"{name}-rs"
                  if os.path.exists(os.path.join(dirname, f"{name}-rs.meta.json"))
                  else name)
    have_npz = os.path.exists(os.path.join(dirname, f"{name}.key.npz"))
    have_keybin = (os.path.exists(os.path.join(dirname, f"{name}.key.bin"))
                   or os.path.exists(os.path.join(dirname, f"{key_prefix}.key.bin")))
    if not have_npz and not have_keybin:
        print(f"key material missing; regenerating (deterministic)", flush=True)
        reg = load_registry()
        build(name, *reg[name][:6], s_override=reg[name][6], blocks=False)
        have_npz = True
    if not os.path.exists(os.path.join(dirname, "kernel.bin")):
        print(f"missing {dirname}/kernel.bin; run: python main.py solve {name}",
              flush=True)
        return False
    meta = json.load(open(os.path.join(dirname, f"{key_prefix}.meta.json")))
    GF = field_from_meta(meta)
    m, n, k, lam = meta["m"], meta["n"], meta["k"], meta["lam"]
    if have_npz:
        d = np.load(os.path.join(dirname, f"{name}.key.npz"))
        G_pub = GF(d["G_pub"])
    else:
        keybin = os.path.join(dirname, f"{key_prefix}.key.bin")
        G_pub = GF(_read_fe_file(keybin, m, k * n)).reshape(k, n)
    if t is None:
        t = (n - k) // (2 * lam)
    print(f"=== {name}: Loidreau PKE vs the recovered equivalent key "
          f"(({m};{n},{k},{lam}), t={t}) ===", flush=True)
    res = equivalent_key_from_kernel(name)
    print(f"recovery: ok={res['ok']} stage={res['stage']} err={res['err']}",
          flush=True)
    if not res["ok"]:
        return False
    return _roundtrip(GF, G_pub, res["b_hat"], res["P_hat"], lam, t,
                      trials=trials, rng_seed=seed)


TOY_CASES = [(12, 12, 7, 2, 1, 1), (14, 14, 8, 2, 3, 1), (16, 16, 10, 3, 4, 1)]


def run_pke_toys() -> bool:
    """Dense-route PKE toys; lam*t <= r/2 and r >= lam+2 required."""
    allok = True
    for (m, n, k, lam, seed, t) in TOY_CASES:
        allok &= pke_toy_case(m, n, k, lam, seed, t)
    print("ALL PKE TOY CASES PASSED" if allok else "SOME CASES FAILED")
    return allok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_pke_toys() else 1)
