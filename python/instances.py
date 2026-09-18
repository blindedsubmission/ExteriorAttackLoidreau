#!/usr/bin/env python3
"""Phase 0: build Loidreau instances and (optionally) anchored blocks for the
engine routes (Rust kernel finder).

Writes per case:
  <name>.blocks.bin   row-major uint64 anchored stack (R x D), R = (s-lam)*C(k,lam)
  <name>.blocksT.bin  transposed copy (D x R) for sequential transpose matvecs
  <name>.meta.json    m, poly, n, k, lam, s, D, R
  <name>.key.npz      G_pub and secret material for post-hoc ground truth
"""
import json
import os
import sys
from itertools import combinations, permutations, product
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exterior import _anchor_data, shifted_blocks
from primitives import make_field, random_code
from schemes import make_loidreau_key, make_lowms_key


def anchored_blocks_u64(GF, G, lam, s):
    k, n = G.shape
    d = n - k
    idxI = np.array(list(combinations(range(k), lam)))          # (B, lam)
    idxT = np.array(list(product(range(d), repeat=lam)))        # (D, lam)
    anchors = _anchor_data(GF, G, lam)
    gshifts = shifted_blocks(GF, G, s)
    out = []
    for i in range(lam, s):
        Pa = [(gshifts[i] @ anchors[a].T) for a in range(lam)]  # k x d each
        det = GF.Zeros((len(idxI), len(idxT)))
        for pi in permutations(range(lam)):  # char 2: sign +1
            prod = GF.Ones((len(idxI), len(idxT)))
            for r in range(lam):
                a = pi[r]
                # M[I,t][r,a] = Pa_a[I_r, t_a]
                col = Pa[a][:, idxT[:, a]]                     # k x D
                gather = col[idxI[:, r]]                       # (B, D)
                prod = prod * gather
            det = det + prod
        out.append(det.view(np.ndarray).astype(np.uint64))
    return np.vstack(out)


def inst_dir(name, root=None):
    """Artifact directory of an instance: <implementation>/instances/<name>/."""
    root = root or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "instances", name)


def build(name, mdeg, n, k, lam, seed, kind, s_override=None, blocks=True):
    out = inst_dir(name)
    os.makedirs(out, exist_ok=True)
    GF = make_field(mdeg)
    rng = np.random.default_rng(seed)
    s = s_override if s_override is not None else max(
        -(-comb(n, lam) // comb(k, lam)),
        lam * (n - lam) // (k - lam + 1) + 1)
    assert s <= n - k and s > lam, f"s*={s} infeasible for r={n-k}"
    extra = {}
    G_sec = S = None
    if kind == "loidreau":
        G, G_sec, S, P, P_inv = make_loidreau_key(GF, n, k, lam, rng)
    elif kind == "lowms":
        H_pub, G, b, P, P_inv = make_lowms_key(GF, n, k, lam, rng)
        extra = dict(H_pub=H_pub.view(np.ndarray).astype(np.uint64),
                     b=b.view(np.ndarray).astype(np.uint64))
    else:
        G = random_code(GF, n, k, rng)
        P = P_inv = None
    D = (n - k) ** lam
    R = (s - lam) * comb(k, lam)
    if blocks:
        Blks = anchored_blocks_u64(GF, G, lam, s)
        assert Blks.shape == (R, D), Blks.shape
        Blks.tofile(os.path.join(out, f"{name}.blocks.bin"))
        Blks.T.tofile(os.path.join(out, f"{name}.blocksT.bin"))
        nbytes = Blks.nbytes
    else:
        nbytes = 0
    coeffs = GF.irreducible_poly.coeffs
    poly = int(sum(int(c) << i for i, c in enumerate(coeffs[::-1])))
    json.dump(dict(m=mdeg, poly=poly, n=n, k=k, lam=lam, s=s, D=D, R=R,
                   kind=kind),
              open(os.path.join(out, f"{name}.meta.json"), "w"))
    np.savez(os.path.join(out, f"{name}.key.npz"), G_pub=G.view(np.ndarray).astype(np.uint64),
             G_sec=None if G_sec is None else G_sec.view(np.ndarray).astype(np.uint64),
             S=None if S is None else S.view(np.ndarray).astype(np.uint64),
             P=None if P is None else P.view(np.ndarray).astype(np.uint64),
             **extra)
    print(f"{name}: kind={kind} m={mdeg} D={D} R={R} s={s} poly=0x{poly:x} "
          + (f"blocks {nbytes/1e6:.0f} MB" if blocks else "key only"),
          flush=True)


def load_registry(path=None):
    """Parse instances.txt: name m n k lam seed kind [s]."""
    import os
    path = path or os.path.join(os.path.dirname(__file__), "instances.txt")
    out = {}
    for line in open(path):
        line = line.split("#")[0].split()
        if not line:
            continue
        name, m, n, k, lam, seed, kind = line[:7]
        out[name] = (int(m), int(n), int(k), int(lam), int(seed), kind,
                     int(line[7]) if len(line) > 7 else None)
    return out


def export_rust_key(name, dirname=None):
    """Export an instance's public key for the matrixfree-rs engine (Mode B:
    Python-generated key solved by the Rust kernel engine).

    Writes <dirname>/<name>-rs.key.bin (k*n field elements, 16 bytes each,
    little-endian lo/hi u64) and <name>-rs.meta.json in matrixfree-rs `gen`
    format (poly_g_lo/poly_g_hi = meta["poly"] minus the implicit x^m term,
    split into two u64 limbs).  `matrixfree-rs solve` then runs in exactly
    the instance's own field; recovery picks the -rs artifacts up
    automatically via the kernel-flavor dispatch in recovery.py.
    """
    d = dirname or inst_dir(name)
    meta = json.load(open(os.path.join(d, f"{name}.meta.json")))
    m = int(meta["m"])
    poly = int(meta["poly"])
    if not (poly >> m) & 1:
        sys.exit(f"{name}: meta poly 0x{poly:x} lacks its x^m bit")
    g = poly ^ (1 << m)
    mask = (1 << 64) - 1
    lo, hi = g & mask, (g >> 64) & mask
    G = np.load(os.path.join(d, f"{name}.key.npz"))["G_pub"]
    out_key = os.path.join(d, f"{name}-rs.key.bin")
    with open(out_key, "wb") as f:
        for e in G.reshape(-1).tolist():
            f.write(int(e).to_bytes(16, "little"))
    rs_meta = dict(m=m, n=int(meta["n"]), k=int(meta["k"]), lam=int(meta["lam"]),
                   random=(meta.get("kind") == "random"),
                   seed=int(meta.get("seed", 0)),
                   poly_g_lo=lo, poly_g_hi=hi, kind=meta.get("kind", "loidreau"))
    out_meta = os.path.join(d, f"{name}-rs.meta.json")
    json.dump(rs_meta, open(out_meta, "w"))
    print(f"export_rust_key: {out_key} ({G.size} elements), "
          f"poly x^{m}+g g=(0x{lo:x},0x{hi:x})", flush=True)
    return out_key


if __name__ == "__main__":
    import sys
    reg = load_registry()
    names = sys.argv[1:] or list(reg)
    blocks = "--key-only" not in names
    for name in names:
        if name.startswith("--"):
            continue
        build(name, *reg[name][:6], s_override=reg[name][6], blocks=blocks)
