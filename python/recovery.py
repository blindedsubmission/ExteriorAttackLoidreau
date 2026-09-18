#!/usr/bin/env python3
"""Full key recovery from the exterior kernel (main.tex Steps 1-5).

Input: the public primal generator G (k x n over GF(2^m)) plus a kernel
vector of the exterior stack -- a dense Pluecker vector (toy sizes) or an
anchored coefficient vector (t80 route, a legacy kernel.bin format).

  Step 1  extract the hidden plane L = Span{b_0,...,b_{lam-1}} from the
          kernel vector and verify rank(G^{[-i]}|_L) < lam for all i < r.
  Step 2  h_i (rows of S^{-1}H_pub) up to scalar as the intersection lines
          rowsp(ker G) cap L^{[i]}.
  Step 3  Fqm-linear Frobenius-shift system
          mu_i h~_{i+1} = h~_i^{[-1]} T  ->  T = beta (Q^{[-1]})^{-1} Q.
  Step 4  matrix Hilbert 90 over F_2: Q_hat with Q_hat^{[-1]} T = Q_hat.
  Step 5  B_hat = Q_hat R, gamma_hat = h~_0 Q_hat^{-1},
          H_hat rows gamma_hat B_hat^{[i]}; verify rowsp(H_hat) =
          rowsp(ker G), then build and verify the equivalent Loidreau key
          (b_hat of rank weight n, invertible mask P_hat in a lam-dim
          F_2-space).

Every stage is verified against public data, so a run reporting ok=True is
a per-instance proof that an equivalent decryption key was recovered.
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations, permutations, product
from math import comb

import numpy as np

from exterior import _anchor_data, stacked_compound
from primitives import field_from_meta, frobenius_shift, make_field
from schemes import make_loidreau_key, make_lowms_key


# ===========================================================================
# Helpers
# ===========================================================================

def onehot(GF, t, n):
    v = GF.Zeros(n)
    v[t] = 1
    return v


def rowsp_equal(A, B):
    ra, rb = A.row_reduce(), B.row_reduce()
    return ra.shape == rb.shape and np.array_equal(ra.view(np.ndarray),
                                                   rb.view(np.ndarray))


def batch_det(GF, blocks):
    """Determinants of (..., lam, lam) galois blocks (char 2: signs +1)."""
    lam = blocks.shape[-1]
    det = GF.Zeros(blocks.shape[:-2])
    for sigma in permutations(range(lam)):
        prod = GF.Ones(blocks.shape[:-2])
        for a in range(lam):
            prod = prod * blocks[..., a, sigma[a]]
        det = det + prod
    return det


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


def bits_of(GF, elems):
    """(len(elems), m) uint8 matrix of coefficient bits.  Works for any m
    (assembly via Python ints: element values exceed uint64 when m > 64)."""
    m = GF.degree
    flat = np.asarray(elems.view(np.ndarray)).ravel()
    out = np.zeros((flat.size, m), dtype=np.uint8)
    for i, e in enumerate(flat):
        v = int(e)
        j = 0
        while v:
            out[i, j] = v & 1
            v >>= 1
            j += 1
    return out


def elems_of_bits(GF, bits):
    m = GF.degree
    rows = np.asarray(bits).reshape(-1, m)
    vals = [sum(int(b) << j for j, b in enumerate(row)) for row in rows]
    return GF(vals)


def fro(GF, A, i):
    """i-th Frobenius twist (negative i allowed)."""
    m = GF.degree
    return frobenius_shift(GF, A, i % m)


# ===========================================================================
# Step 1: plane extraction
# ===========================================================================

def contract_pluecker(GF, p, idx, n, lam, rng):
    """Hidden plane from a Pluecker vector via contraction with random
    covectors: R[j]_t = f(cov_1,..,e_t^*,..,cov_lam)."""
    covs = GF.Random((lam, n), seed=rng)
    N = len(idx)
    # sanity: full contraction must be nonzero
    blocks = GF.Zeros((N, lam, lam))
    for a in range(lam):
        for b in range(lam):
            blocks[:, a, b] = covs[a][idx[:, b]]
    if int(np.sum(batch_det(GF, blocks) * p)) == 0:
        return None
    R = GF.Zeros((lam, n))
    for j in range(lam):
        for t in range(n):
            for a in range(lam):
                row = covs[a] if a != j else onehot(GF, t, n)
                for b in range(lam):
                    blocks[:, a, b] = row[idx[:, b]]
            R[j, t] = int(np.sum(batch_det(GF, blocks) * p))
    return R


def plane_from_anchored_kernel(GF, anchors, p, rng):
    """Hidden plane from an anchored kernel vector p (appendix, implicit
    wedge evaluation).  anchors: lam row bases (d x n) of K_0..K_{lam-1}."""
    lam = len(anchors)
    n = anchors[0].shape[1]
    d = anchors[0].shape[0]
    idxT = np.array(list(product(range(d), repeat=lam)))
    blocks = GF.Zeros((len(idxT), lam, lam))

    def f_eval(rows):
        E = [[rows[a] @ anchors[b].T for b in range(lam)] for a in range(lam)]
        for a in range(lam):
            for b in range(lam):
                blocks[:, a, b] = E[a][b][idxT[:, b]]
        return batch_det(GF, blocks)

    covs = GF.Random((lam, n), seed=rng)
    dets = f_eval(covs)
    f0 = int(np.sum(dets * GF(p)))
    if f0 == 0:
        return None, 0
    R = GF.Zeros((lam, n))
    for j in range(lam):
        for t in range(n):
            rows = [covs[a] if a != j else onehot(GF, t, n) for a in range(lam)]
            R[j, t] = int(np.sum(f_eval(rows) * GF(p)))
    return R, f0


def verify_plane(GF, G, R):
    """dim R = lam and rank(G^{[-i]} R^T) < lam for all i < r."""
    lam, n = R.shape
    if np.linalg.matrix_rank(GF(R)) != lam:
        return False, f"rank(R) = {np.linalg.matrix_rank(GF(R))} != {lam}"
    for i in range(n - G.shape[0]):
        prod = fro(GF, G, -i) @ R.T
        rk = np.linalg.matrix_rank(prod)
        if rk >= lam:
            return False, f"rank(G^[-{i}] R^T) = {rk} (not < {lam})"
    return True, "ok"


# ===========================================================================
# Steps 2-5
# ===========================================================================

def recover_key_from_plane(GF, G, R, verbose=True):
    m = GF.degree
    k, n = G.shape
    r, lam = n - k, R.shape[0]
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    H = G.null_space()
    assert H.shape[0] == r

    # Step 2: lines rowsp(H) cap L^{[i]}
    hs = []
    for i in range(r):
        S = np.vstack([H, fro(GF, R, i)])
        C = S.T.null_space()          # (u, v) with u H = v R^{[i]}
        if C.shape[0] != 1:
            return {"ok": False, "stage": "step2",
                    "err": f"intersection dim {C.shape[0]} at i={i}"}
        hs.append(C[0, :r] @ H)
    hs = GF(np.vstack(hs))

    # Step 3: Frobenius-shift system
    # completion of R by unit rows to an invertible n x n stack (its
    # inverse expresses the h_i^{[-i]} in the row basis of R)
    rows = [R]
    for t in range(n):
        if rows[-1].shape[0] == n:
            break
        trial = np.vstack(rows + [onehot(GF, t, n)[None, :]])
        if np.linalg.matrix_rank(GF(trial)) == trial.shape[0]:
            rows.append(onehot(GF, t, n)[None, :])
    Minv = np.linalg.inv(GF(np.vstack(rows)))

    htil = GF.Zeros((r, lam))
    for i in range(r):
        w = fro(GF, hs[i], -i) @ Minv       # h_i^{[-i]} M = (x_i, 0)
        if np.any(w[lam:].view(np.ndarray)):
            return {"ok": False, "stage": "step3",
                    "err": f"h_[{i}]^[-i] not in rowsp(R)"}
        htil[i] = w[:lam]

    ncols = lam * lam + (r - 1)
    eqs = []
    for i in range(r - 1):
        bi = fro(GF, htil[i], -1)
        for a in range(lam):
            row = GF.Zeros(ncols)
            for b in range(lam):
                row[b * lam + a] = bi[b]
            row[lam * lam + i] = htil[i + 1][a]
            eqs.append(row)
    solspace = GF(np.vstack(eqs)).null_space()  # solutions with A x = 0
    if solspace.shape[0] != 1:
        return {"ok": False, "stage": "step3",
                "err": f"shift-system solution dim {solspace.shape[0]} (want 1)"}
    sol = solspace[0]
    T = GF(sol.view(np.ndarray)[:lam * lam].reshape(lam, lam))
    if np.linalg.matrix_rank(T) < lam:
        return {"ok": False, "stage": "step3", "err": "T singular"}
    log(f"step3: shift system dim 1, T invertible")

    # Step 4: twisted norm + matrix Hilbert 90 over F_2
    prod = T.copy()
    for _ in range(m - 1):
        prod = fro(GF, prod, -1) @ T
    I = GF(np.eye(lam, dtype=np.uint64))
    if not np.array_equal(prod.view(np.ndarray), I.view(np.ndarray)):
        return {"ok": False, "stage": "step4",
                "err": f"twisted norm != I: {prod.view(np.ndarray)}"}

    nb = lam * m
    cols = []
    for j in range(nb):                      # image of each basis bit
        a = GF.Zeros(lam)
        a[j // m] = GF(1 << (j % m))         # Python int: fine for m > 64
        v = fro(GF, a, -1) @ T + a
        cols.append(bits_of(GF, v).ravel())
    Mmap = np.array(cols, dtype=np.uint8).T
    # cols already include the "+ a", so fixed space = ker(Mmap) itself
    Kf = f2_kernel(Mmap)
    if Kf.shape[0] != lam:
        return {"ok": False, "stage": "step4",
                "err": f"fixed space dim {Kf.shape[0]} (want {lam})"}
    Qhat = GF.Zeros((lam, lam))
    for i in range(lam):
        Qhat[i] = elems_of_bits(GF, Kf[i].reshape(lam, m))
    if np.linalg.matrix_rank(Qhat) < lam:
        return {"ok": False, "stage": "step4", "err": "Qhat singular"}
    if not np.array_equal((fro(GF, Qhat, -1) @ T).view(np.ndarray),
                          Qhat.view(np.ndarray)):
        return {"ok": False, "stage": "step4", "err": "Qhat^{[-1]}T != Qhat"}
    log("step4: twisted norm = I, fixed space dim = lam")

    # Step 5: equivalent key
    Bhat = Qhat @ R
    gammahat = htil[0] @ np.linalg.inv(Qhat)
    Hhat = GF.Zeros((r, n))
    for i in range(r):
        Hhat[i] = gammahat @ fro(GF, Bhat, i)
    if not rowsp_equal(Hhat, H):
        return {"ok": False, "stage": "step5", "err": "rowsp(H_hat) != rowsp(ker G)"}
    log("step5: rowsp(H_hat) == rowsp(ker G)")

    # equivalent Loidreau key (b_hat, P_hat)
    bits = bits_of(GF, Bhat).reshape(lam * n, m).T          # (m, lam*n)
    rref, pivots = f2_rref(bits)
    if len(pivots) != n:
        return {"ok": False, "stage": "key",
                "err": f"F_2-span of B_hat entries dim {len(pivots)} != {n}",
                "H_hat": Hhat, "B_hat": Bhat, "gamma_hat": gammahat}
    bhat = GF(np.array([int(Bhat.reshape(-1)[c]) for c in pivots]))
    Abits = bits_of(GF, bhat).T                           # (m, n)
    Pj_hat = np.zeros((lam, n, n), dtype=np.uint64)
    for j in range(lam):
        for c in range(n):
            Aug = np.hstack([Abits, bits_of(GF, Bhat[j, c]).T]).astype(np.uint8)
            Ra, pa = f2_rref(Aug)
            if any(pa[c2] == n for c2 in range(len(pa))):
                return {"ok": False, "stage": "key",
                        "err": "entry not in span(b_hat)"}
            # RREF: columns 0..n-1 pivoted; read solution from last column
            x = np.zeros(n, dtype=np.uint8)
            for i2, p2 in enumerate(pa):
                x[p2] = Ra[i2, n]
            Pj_hat[j][:, c] = x
    Phat = GF.Zeros((n, n))
    for j in range(lam):
        Phat = Phat + gammahat[j] * GF(Pj_hat[j])
    if np.linalg.matrix_rank(Phat) != n:
        return {"ok": False, "stage": "key", "err": "P_hat singular",
                "H_hat": Hhat, "B_hat": Bhat, "gamma_hat": gammahat,
                "b_hat": bhat}
    log("key: wt(b_hat)=n, P_hat invertible -> equivalent key verified")
    return {"ok": True, "stage": "done", "err": None,
            "H_hat": Hhat, "B_hat": Bhat, "gamma_hat": gammahat,
            "b_hat": bhat, "P_hat": Phat}


# ===========================================================================
# Drivers
# ===========================================================================

def toy_case(m, n, k, lam, seed):
    from itertools import combinations
    GF = make_field(m)
    rng = np.random.default_rng(seed)
    G, G_sec, S, P, P_inv = make_loidreau_key(GF, n, k, lam, rng)
    # The one-dimensional-kernel conjecture (conj:structured-one-dimensional-
    # kernel in main.tex) is stated for s = n - k; at s = s* the stack can
    # carry extra structured kernel (empirically dim C(r-s+lam, lam) at
    # lambda = 2), so the dense route uses s = r.
    s = n - k
    print(f"=== toy ({m};{n},{k},{lam}) seed={seed} s=r={s} ===", flush=True)
    W = stacked_compound(GF, G, lam, s)
    kern = W.null_space()
    print(f"dense stack {W.shape}, kernel dim {kern.shape[0]}", flush=True)
    if kern.shape[0] != 1:
        print("kernel dim != 1 -> plane extraction inconclusive", flush=True)
        return False
    from itertools import combinations as comb
    idx = np.array(list(comb(range(n), lam)))
    R = contract_pluecker(GF, kern[0], idx, n, lam, rng)
    ok, msg = verify_plane(GF, G, R)
    print(f"plane verification: {msg}", flush=True)
    if not ok:
        return False
    res = recover_key_from_plane(GF, G, R)
    print(f"recovery: ok={res['ok']} stage={res['stage']} err={res['err']}",
          flush=True)
    return res["ok"]


def lowms_toy_case(m, n, k, lam, seed):
    """Dense-route recovery from a LowMS Niederreiter public key.

    Faithful chain: generate the LowMS key, keep only the published
    systematic parity-check H_pub, derive the primal generator the way an
    attacker must (null space), and run the exterior pipeline on it.  Also
    checks rowsp(G) == rowsp(Moore^k(b) P^{-1}) against the secret.
    """
    from itertools import combinations as comb
    GF = make_field(m)
    rng = np.random.default_rng(seed)
    H_pub, G, b, P, P_inv = make_lowms_key(GF, n, k, lam, rng)
    from primitives import moore_matrix
    assert rowsp_equal(G, moore_matrix(GF, b, k) @ P_inv), \
        "rowsp(G) != rowsp(Moore^k(b) P^-1)"
    s = n - k
    print(f"=== lowms toy ({m};{n},{k},{lam}) seed={seed} s=r={s} ===",
          flush=True)
    W = stacked_compound(GF, G, lam, s)
    kern = W.null_space()
    print(f"dense stack {W.shape}, kernel dim {kern.shape[0]}", flush=True)
    if kern.shape[0] != 1:
        print("kernel dim != 1 -> plane extraction inconclusive", flush=True)
        return False
    idx = np.array(list(comb(range(n), lam)))
    R = contract_pluecker(GF, kern[0], idx, n, lam, rng)
    ok, msg = verify_plane(GF, G, R)
    print(f"plane verification: {msg}", flush=True)
    if not ok:
        return False
    res = recover_key_from_plane(GF, G, R)
    print(f"recovery: ok={res['ok']} stage={res['stage']} err={res['err']}",
          flush=True)
    return res["ok"]


def _read_fe_file(path, m, expected):
    """Field elements from an engine vector file of `expected` elements.

    Supported layouts: 16 bytes per element (LE lo u64, then LE hi u64 --
    matrixfree-rs write_fe_vec) or the legacy 8-byte LE u64 format of
    an earlier development engine.  The width is picked by matching the expected element
    count, so the two are never confused.  Returns Python ints (bitmasks).
    """
    raw = open(path, "rb").read()
    for width in (16, 8):
        if len(raw) == expected * width:
            if width == 8:
                return [int.from_bytes(raw[i:i + 8], "little")
                        for i in range(0, len(raw), 8)]
            vals = []
            for i in range(0, len(raw), 16):
                lo = int.from_bytes(raw[i:i + 8], "little")
                hi = int.from_bytes(raw[i + 8:i + 16], "little")
                if hi != 0 and m <= 64:
                    raise ValueError(f"{path}: hi word nonzero at element "
                                     f"{i // 16} but m <= 64")
                vals.append(lo | (hi << 64))
            return vals
    raise ValueError(f"{path}: {len(raw)} bytes matches neither {expected} "
                     f"x 16-byte nor {expected} x 8-byte elements")


def equivalent_key_from_kernel(name, dirname=None, kernel_path=None,
                               key_prefix=None, tries=8):
    """Plane extraction + Steps 2-5 for an engine instance (auto-detects
    the kernel flavor from its size against the instance parameters).

    Flavors:
      - full Pluecker kernel: C(n,lam) coordinates in lex order -- the
        matrixfree-rs engines' kernel.bin / p_lex (16-byte elements), or
        an 8-byte u64 file of the same count.  Plane extraction uses
        contract_pluecker with `tries` fresh random draws.
      - anchored kernel: meta["D"] = r^lam u64 coordinates -- the
        legacy anchored kernel.bin consumed by plane_from_anchored_kernel.

    The instance meta is <dirname>/<key_prefix>.meta.json; key_prefix
    defaults to <name>, or "<name>-rs" when instances.export_rust_key
    artifacts are present (Python-generated key solved by the Rust
    engine).  The field is taken from the meta (poly_g_lo/poly_g_hi or
    "poly"), never from a global default.  G_pub comes from
    <key_prefix>.key.bin when present, else <name>.key.npz.

    Returns the recover_key_from_plane result dict (with name/meta
    attached), so drivers beyond engine_case (e.g. the PKE demo) can use
    the equivalent key (b_hat, P_hat).
    """
    if dirname is None:
        from instances import inst_dir
        dirname = inst_dir(name)
    if kernel_path is None:
        kernel_path = os.path.join(dirname, "kernel.bin")
    if key_prefix is None:
        key_prefix = (f"{name}-rs"
                      if os.path.exists(os.path.join(dirname, f"{name}-rs.meta.json"))
                      else name)
    meta = json.load(open(f"{dirname}/{key_prefix}.meta.json"))
    meta.setdefault("kind", "random" if meta.get("random") else "loidreau")
    m, n, k, lam = meta["m"], meta["n"], meta["k"], meta["lam"]
    size = os.path.getsize(kernel_path)
    if size % 16 == 0 and size // 16 == comb(n, lam):
        flavor = ("pluecker", 16)
    elif size % 8 == 0 and size // 8 == comb(n, lam):
        flavor = ("pluecker", 8)
    elif "D" in meta and size % 8 == 0 and size // 8 == meta["D"]:
        flavor = ("anchored", 8)
    else:
        return {"ok": False, "stage": "plane",
                "err": (f"kernel size {size} matches neither C(n,lam)="
                        f"{comb(n, lam)} (Pluecker, 8/16 B) nor D=r^lam="
                        f"{meta.get('D')} (anchored, 8 B)"),
                "name": name, "meta": meta}
    if flavor[0] == "pluecker":
        return _equivalent_key_from_pluecker_kernel(
            name, dirname, kernel_path, key_prefix, meta,
            width=flavor[1], tries=tries)
    return _equivalent_key_from_anchored_kernel(name, dirname, kernel_path,
                                                meta)


def _equivalent_key_from_pluecker_kernel(name, dirname, kernel_path,
                                         key_prefix, meta, width, tries):
    m, n, k, lam = meta["m"], meta["n"], meta["k"], meta["lam"]
    t0 = time.time()
    GF = field_from_meta(meta)
    print(f"=== {name}: extracting plane from full-Pluecker kernel "
          f"(C(n,lam)={comb(n, lam)}, {width}-byte elements, field m={m}) ===",
          flush=True)
    keybin = os.path.join(dirname, f"{key_prefix}.key.bin")
    if os.path.exists(keybin):
        G = GF(_read_fe_file(keybin, m, k * n)).reshape(k, n)
        gsrc = os.path.basename(keybin)
    else:
        d = np.load(f"{dirname}/{name}.key.npz")
        G = GF(d["G_pub"])
        gsrc = f"{name}.key.npz"
    print(f"public key: {gsrc}  [{time.time() - t0:.1f}s]", flush=True)
    p = GF(_read_fe_file(kernel_path, m, comb(n, lam)))
    idx = np.array(list(combinations(range(n), lam)))
    R = None
    for attempt in range(tries):
        rng = np.random.default_rng(2026 + attempt)
        R = contract_pluecker(GF, p, idx, n, lam, rng)
        if R is not None:
            break
    print(f"plane contraction: {'ok' if R is not None else 'Phi(p) = 0'} "
          f"(attempt {attempt + 1}/{tries})  [{time.time() - t0:.1f}s]",
          flush=True)
    if R is None:
        return {"ok": False, "stage": "plane",
                "err": f"Phi(p) = 0 for {tries} independent draws",
                "name": name, "meta": meta}
    ok, msg = verify_plane(GF, G, R)
    print(f"plane verification: {msg}  [{time.time() - t0:.1f}s]", flush=True)
    if not ok:
        return {"ok": False, "stage": "plane", "err": msg,
                "name": name, "meta": meta}
    res = recover_key_from_plane(GF, G, R)
    print(f"steps 2-5 + key  [{time.time() - t0:.1f}s]", flush=True)
    res["name"], res["meta"] = name, meta
    return res


def _equivalent_key_from_anchored_kernel(name, dirname, kernel_path, meta):
    m, n, k, lam = meta["m"], meta["n"], meta["k"], meta["lam"]
    GF = field_from_meta(meta)
    d = np.load(f"{dirname}/{name}.key.npz")
    G = GF(d["G_pub"])
    p = GF(np.fromfile(kernel_path, dtype=np.uint64))
    assert p.shape[0] == meta["D"] == (n - k) ** lam
    anchors = _anchor_data(GF, G, lam)
    rng = np.random.default_rng(2026)
    print(f"=== {name}: extracting plane from anchored kernel vector ===",
          flush=True)
    R, f0 = plane_from_anchored_kernel(GF, anchors, p, rng)
    if R is None:
        return {"ok": False, "stage": "plane", "err": "Phi(p) = 0",
                "name": name, "meta": meta}
    ok, msg = verify_plane(GF, G, R)
    print(f"plane verification: {msg}", flush=True)
    if not ok:
        return {"ok": False, "stage": "plane", "err": msg,
                "name": name, "meta": meta}
    res = recover_key_from_plane(GF, G, R)
    res["name"], res["meta"] = name, meta
    return res


def engine_case(name="t80", dirname=None, kernel_path=None):
    res = equivalent_key_from_kernel(name, dirname, kernel_path)
    print(f"recovery: ok={res['ok']} stage={res['stage']} err={res['err']}",
          flush=True)
    if res["ok"]:
        meta = res["meta"]
        print(f"EQUIVALENT KEY RECOVERED for {name} "
              f"({meta.get('kind', 'loidreau')} key, "
              f"({meta['m']};{meta['n']},{meta['k']},{meta['lam']}))", flush=True)
    return res["ok"]


# Backwards-compatible alias for the documented t80 invocation.
t80_case = engine_case


TOY_CASES = [(12, 12, 7, 2, 1), (14, 14, 8, 2, 3),
             (16, 16, 11, 3, 4), (16, 16, 11, 3, 5)]


def run_lowms_toys() -> bool:
    """Dense-route LowMS toys; r >= lam+2 is required by the shift system."""
    allok = True
    for (m, n, k, lam, seed) in TOY_CASES:
        allok &= lowms_toy_case(m, n, k, lam, seed)
    print("ALL LOWMS TOY CASES PASSED" if allok else "SOME CASES FAILED")
    return allok


def run_toys() -> bool:
    allok = True
    for (m, n, k, lam, seed) in TOY_CASES:
        allok &= toy_case(m, n, k, lam, seed)
    print("ALL TOY CASES PASSED" if allok else "SOME CASES FAILED")
    return allok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("toys", "lowms-toys"):
        sys.exit(0 if engine_case(sys.argv[1]) else 1)
    if len(sys.argv) > 1 and sys.argv[1] == "lowms-toys":
        sys.exit(0 if run_lowms_toys() else 1)
    sys.exit(0 if run_toys() else 1)
