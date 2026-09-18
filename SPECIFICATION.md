# SPECIFICATION — keys, ciphertexts, and interfaces

This document specifies the artifact's interfaces precisely enough to be
implemented from scratch: a reviewer can generate their own keys in their
own code, serialize them per section 5, run the pipeline on them, and
verify the recovered equivalent key per section 7 without trusting any
pipeline key-generation code.  `bring_your_own_key.py` in this directory
is such an independent implementation and doubles as a conformance test.

## 0. Relation to the other documents

This file is the normative contract: if ARCHITECTURE.md or README.md
disagree with it on an encoding, a format, or a parameter constraint,
this file wins.  ARCHITECTURE.md documents design and rationale.
README.md documents how to run the pipeline.  VALIDATION.md records
the validation evidence.

## 1. Scope and trust model

The pipeline has two trust boundaries:

- **keygen** (only this stage handles secrets) samples an instance of
  the Loidreau PKE / LowMS KEM.  A reviewer may replace it entirely with
  their own generator (sections 3, 5, 6).
- everything after — kernel search, recovery, decryption — consumes only
  public bytes and proves its results against public data (section 7).

## 2. Field and element encoding

- Base field: GF(2^m), m in 1..=128, represented as polynomials over F2
  of degree < m modulo an irreducible f.
- The modulus has the shape f = x^m + g with deg g < m.  It is recorded
  wherever the key is: as the integer g (coefficients of g, bit i =
  coefficient of x^i) or as the full polynomial poly = g + 2^m.
- Element encoding: an element is the integer whose bit i is the
  coefficient of x^i.  Serialized **little-endian in two 8-byte words**:
  bytes 0..7 = lo = bits 0..63, bytes 8..15 = hi = bits 64..127.  For
  m <= 64, hi is always 0 but is still written.
- Multiplication/addition are polynomial multiplication modulo f and XOR.
  (The engine detects PCLMULQDQ at runtime.  Results are exact and
  order-free either way.)

## 3. The scheme (Loidreau-type public-key encryption)

Notation: Frobenius a^[i] := (a_1^(2^i), ..., a_n^(2^i)) (entrywise).
The F_2-rank of a set of field elements is the dimension of their span
as bit vectors of length m.

- **Support.** a in GF(2^m)^n with F_2-rank(a) = n (rank weight n).
  Requires n <= m.
- **Gabidulin generator.** M_k(a) in GF(2^m)^(k x n): row i = a^[i]
  (a Moore matrix).  Its row space is a Gabidulin code of length n.
- **Mask.** P in GL_n(GF(2^m)) with all n^2 entries in a
  lambda-dimensional F_2-subspace Lambda of GF(2^m) (the Loidreau
  perturbation).  Any Lambda works for the attack.  Constructions often
  normalize 1 in Lambda.
- **Scrambler.** S in GL_k(GF(2^m)).
- **Public key (Loidreau PKE).**  G_pub = S * M_k(a) * P^(-1).
- **LowMS variant.**  The published object is a systematic parity check
  H_pub = RREF(H * P^T) with H = nullspace(M_k(a)).  The attacker derives
  the primal generator G_pub = nullspace(H_pub), whose row space equals
  rowsp(M_k(a) * P^(-1)).  Recovery and verification are identical from
  there.
- **Ciphertext.**  Pick m in GF(2^m)^k (message) and a rank-t error e
  (coordinates spanning a t-dimensional F_2-space, i.e. rank weight t);
  y = m * G_pub + e in GF(2^m)^n.  Decryption radius: lambda * t <=
  (n - k) / 2.  t >= 1 therefore needs n - k >= 2*lambda.

## 4. Instance parameters

A pipeline instance is the tuple (m, n, k, lam) plus the modulus and a
seed.  Constraints:

- 1 <= lam <= k and k < n;  n <= m for structured keys.
- **recovery feasibility: n - k >= lam + 2** (uniqueness of the
  Frobenius-shift system).  The drivers enforce this.
- one-dimensional kernel of the stacked system at s = n - k (empirical,
  verified by the raw-circuit gate).
- s defaults to n - k everywhere (the full stack).

## 5. File formats

All "elements" are 16 bytes: little-endian lo u64, then little-endian
hi u64 (section 2).  All matrices are row-major.

### 5.1 Public key — `<prefix>.key.bin`

Exactly k*n elements: row i of G_pub is elements i*n .. (i+1)*n - 1, in
column order.  This is the ONLY key material the solver and the recovery
consume.

### 5.2 Instance metadata — `<prefix>.meta.json`

One JSON object:

    {"m":24,"n":16,"k":10,"lam":3,"random":false,"seed":7,
     "poly_g_lo":27,"poly_g_hi":0}

`poly_g_lo`/`poly_g_hi` are the two u64 limbs of g in f = x^m + g (bit i
of the concatenated 128-bit integer = coefficient of x^i).  The solver
reads this file and refuses to solve a key in any other field.  The
Python driver additionally accepts `"poly"` (the full polynomial
integers, bit m set) — same convention as `instances.py`.

### 5.3 Kernel vector — `kernel.bin`

Exactly C(n, lam) elements.  Entry at index rank(J) is the Pluecker
coordinate p_J = det(U[:, J]) of the recovered lambda-plane L =
rowsp(U), where J = (j_1 < ... < j_lam) ranges over the lam-subsets of
{0, ..., n-1} in lexicographic order (the order of
`itertools.combinations(range(n), lam)`).  Defined up to one global
nonzero scalar.  Contract: p != 0 and W p = 0 for the stacked system
W_s^(lam)(G_pub) built from G_pub with s = n - k (the pipeline
re-verifies this before accepting).

### 5.4 Ground truth (optional, Python side)

`<name>.key.npz` may store G_pub and secret material (G_sec, S, P,
P_inv, uint64 — hence m <= 64) for self-checks.  The pipeline never
needs it: recovery and decryption work from 5.1 + 5.2 alone.

### 5.5 Ciphertexts

The demo generates ciphertexts internally (y = m G_pub + e, section 3).
If exchanged, serialize y as n elements (16 bytes each), row vector;
m is k elements, e is n elements.

## 6. Key generation recipe

Any randomness may be used.  Only the distribution matters:

    a:  n random field elements, retried until F_2-rank(a) = n
    S:  k x k random, retried until invertible
    Gamma: lambda field elements spanning a lambda-dim F_2-space
    P:  n x n, entry (i,j) = sum_t c[i,j,t] * gamma_t with random bits
        c in {0,1}, retried until invertible over GF(2^m)
    G_pub = S * M_k(a) * P^(-1)

For bit-reproduction of shipped instances, the generators are seeded:
the Rust `gen` uses the documented xorshift64 stream (seed, conditioning
weights, start vectors), the Python `keygen` uses `numpy.random.
default_rng(seed)` (PCG64) with the call order visible in
`schemes.py`/`primitives.py`.  Bit-reproduction additionally pins the
galois/numpy versions.  The field itself is always pinned by meta.json.

## 7. Recovery contract and independent verification

Given only G_pub (5.1) and a kernel vector (5.3), recovery outputs an
**equivalent key**: coordinates b_hat in GF(2^m)^n and P_hat in GL_n
such that (with H_hat := Moore_r(b_hat) * P_hat, r = n - k)

    rowsp(H_eq) = rowsp(ker G_pub),                    (same public code)
    F_2-rank(b_hat) = n         (rank-weight-n support),
    P* invertible.

A reviewer verifies this with plain linear algebra, no pipeline code:

    (a) f2_rank(b_hat) = n                       (section 2 rank)
    (b) det(P_hat) != 0
    (c) RREF(Moore_r(b_hat) * P_hat) == RREF(nullspace(G_pub))
    (d) encrypt any (m, e) with G_pub (section 3); decrypt with
        (b_hat, P_hat) — syndrome decode w.r.t. Moore_r(b_hat), unmask by
        P_hat^-T — and compare (m, e) exactly.

These are exactly the checks `bring_your_own_key.py` implements.  (d) is
the operative proof: the recovered key decrypts fresh ciphertexts
produced with the true public key.

## 8. Bring-your-own-key workflow

1. Pick a shape satisfying section 4 and a row in `instances.txt`
   (or add one).
2. Write your public key and meta as
   `instances/<name>/<name>-rs.key.bin` and `<name>-rs.meta.json`
   (sections 5.1, 5.2 — the `-rs` names are the engine-key convention).
3. `python3 pipeline.py stage solve <name>` (kernel search),
   `stage recover <name>` (equivalent key), `stage demo <name>`
   (decryption).  Or run `bring_your_own_key.py` for the fully
   independent end-to-end demonstration.

## 9. Honesty boundaries

- The implementation was developed with LLM assistance
  (AI_DISCLOSURE.md).
- Kernel-solve wall times dominate.  Recovery and decryption are
  seconds.  A recovered key is *equivalent*, not *the* secret key:
  many (a, P, S) triples produce the same public key, and the attack
  recovers one of them.
- The attack is randomized (Las Vegas): a solver seed that finds no
  kernel is a retry, not a break.  The W p == 0 gate makes false
  acceptance impossible.
