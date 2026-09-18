# Architecture of the attack pipeline

How the released pipeline works, stage by stage, and why it is built
this way.  Division of labor between the documents:

    SPECIFICATION.md   the contract: field and scheme semantics, file
                       formats, and the recovery-verification recipe.
                       Normative and implementable from scratch.
    ARCHITECTURE.md    the design: algorithms, data structures, and the
                       rationale behind the engineering decisions.
    README.md          how to run the pipeline.
    VALIDATION.md      evidence that the pipeline works.

Normative statements (encodings, formats, parameter constraints) are in
SPECIFICATION.md, and this document links there instead of restating
them.  Function-level detail is in the module docstrings of `python/`
and `rust/src/`.  The mathematics is in the paper.

## The five stages at a glance

    stage     component            input            output
    keygen    Python               registry seed    secret + PUBLIC key
    export    Python               public key       Rust-format key + meta
    solve     Rust                 PUBLIC key only  kernel vector p, W p = 0
    recover   Python               PUBLIC key + p   equivalent key (verified)
    demo      Python               PUBLIC key       ciphertexts decrypt with
                                                  the recovered key alone

Trust boundary: secret material exists only in the keygen stage and is
written only to `instances/<name>/`.  Everything after `export` is the
attacker's view: it consumes the public key (or data derived from it)
exactly as a real adversary would, and every correctness check from
there on uses public data only.

## Module map (python/)

    instances.py   registry, keygen driver, artifact directories,
                   export to the Rust key format
    schemes.py     Loidreau and LowMS key constructions
    primitives.py  GF(2^m) construction, Moore matrices, Frobenius
                   shifts, F2-subspace masks, random codes
    exterior.py    compound (Pluecker) matrices, anchored-stack helpers
    recovery.py    plane extraction + Steps 1-5, entry points
    pke.py         Loidreau PKE encryption/decryption, the demo roundtrip
    decoding.py    rank-metric decoding (used by the demo)
    f2.py          F2 bit linear algebra (used by decoding/recovery)

## Stage 1: keygen (Python)

The construction is normative in SPECIFICATION.md sections 3 and 6
(rank-weight-n support, Moore generator, lambda-dimensional F2-subspace
mask, scrambler, G_pub = S M_k(a) P^-1).  Design notes only here.  Both
published forms are supported:

- LowMS kind (the paper's target).  The published object is the
  systematic parity-check H_pub of a Niederreiter-style key.  The
  attacker, and therefore this pipeline, derives the primal generator as
  G = ker(H_pub).  For the secret support a this equals
  rowsp(Moore^k(a) P^-1) (asserted in the toy route).
- random kind.  A random full-rank code.  The engine must find no
  kernel; this is the negative control used by `pipeline.py verify`.

All randomness comes from `numpy.random.default_rng` seeded by the
registry seed, so keys regenerate deterministically.

## Stage 2: export (Python)

Writes the public key and metadata per SPECIFICATION.md section 5
(16-byte little-endian elements, the modulus as poly_g_lo/poly_g_hi).
The modulus is exported explicitly so the solver provably runs in the
field the key lives in, and the solver reads it back and rejects
mismatches.  This stage is the only bridge between the two languages:
after it, the Rust engine and the Python recovery share no state beyond
these files.

## Stage 3: kernel finder (Rust)

The solver must find a nonzero p with W p = 0 for the stacked exterior
system

    W = ( C_lam(G^{[-i]}) )_{i=0..s-1}          (s = n-k, C_lam = compound)

W is never materialized.  It is applied by the substitute-and-discard
circuit (below), so memory is O(N) with N = C(n, lam).  Wiedemann-type
recurrence methods need only the matrix-vector products that the circuit
provides.

### The square operator and conditioning

Wiedemann requires a square operator, but E is s*C(k,lam) x N.  The
engine forms H = W^T Delta W with a random nonzero diagonal Delta and
runs the sequence on H.  In characteristic 2 this diagonal conditioning
is required: without it, H = W^T W has a large radical (a Gram matrix can
have nullity far above the planted kernel; the smallest studied instance
has nullity 20 against the planted 1).  With rank(W) = N-1, a
Cauchy-Binet expansion gives ker(H) = ker(W) with probability at least
1 - (N-1)/(2^m - 1).  The weights come from a keyed xorshift stream
seeded by the solver seed, with zero weights replaced by 1, so runs are
reproducible from the seed.

### The circuit oracle

H v costs two passes of the grouped substitute-and-discard circuit
(2*s*T_and field multiply-adds), where
T_and = k*(C(n-1, lam) - C(k-1, lam)).  Each pass re-applies the
elementary transvection factors of the systematic-form compound in
reverse or forward order (a "stage") and applies the weighting by
Delta between the passes.  The schedule, meaning which coefficient multiplies
which source into which destination per stage, is computed once at
setup and then streamed: the CPU walks a flat triple array, and the GPU
dispatches one workgroup grid per stage.  Frobenius blocks are
independent.  The block index is carried by workgroup grid.y on the GPU,
and the schedule is grouped per block, so no loop iterates over the
s = n-k blocks serially.

### Field arithmetic

- Element and modulus conventions are normative in SPECIFICATION.md
  section 2.  Implementation-wise, elements are two u64 limbs, with
  m <= 64 as the fast path.
- The default modulus is f = x^m + g, searched deterministically at
  keygen: all trinomials x^m + x^t + 1 in ascending t, then
  pentanomials, then weight-7, then a dense fallback.  Seroussi's tables
  guarantee a pentanomial for m <= 128.  A previous search omitted the
  constant term of trinomials and silently selected weight-28 dense
  polynomials, which cost a factor 8-16 in multiply speed for m in
  33..64.
- Multiplication uses PCLMULQDQ with runtime detection on x86-64 (1-3
  clmuls plus XOR folds at m <= 64).  Other architectures use a portable
  path.  A Barrett-style single-fold for m near 64 is a known optional
  improvement, not currently load-bearing.

### Solver: block Wiedemann with a PM-BASIS annihilator

The default mode is `--mode block --block 8`:

1. Krylov phase.  Random start blocks U and V (B columns, seeded
   stream).  The sequence S_t = U^T H^t V is recorded for t = 0..e-1
   with window e = 3c + 4B + 1, c = ceil(N/B).  Each step costs B
   H-steps.  The advantage over scalar Wiedemann (2N = 2NB steps) comes
   from the shorter window, not from cheaper steps: per-column cost does
   not amortize with B (measured).  B is therefore chosen for the window
   and annihilator structure, not for step throughput.
2. Annihilator: stacked approximant basis (pmbasis.rs).  Form the series
   [G(x) | -I_B] with G = sum_t S_t x^t and compute a right order-e
   approximant basis W (Giorgi-Jeannerod-Villard).  The -I_B block
   absorbs product heads.  Without it, a bare series with invertible S_0
   collapses the approximant module to trivial x-shifts (observed
   experimentally); this is the block analogue of the scalar window
   starting at t = 1.  The B columns of lowest degree, degree-reversed,
   are engine annihilators sum_j S_{t+j} a_j = 0.  The expected degree
   profile is [c-1, c, ..., c], which matches the textbook prediction
   and was observed in all validation runs.  The iteration is
   order-by-order Gaussian elimination with min-degree pivots,
   O(B^2 e^2) schoolbook, and single-threaded by measurement: rayon over
   the elimination targets was slower (about 59k small dispatches plus
   8 threads streaming the shared pivot column through L3).  A 2.4x
   speedup was obtained from instruction-count work instead (clone-free
   disjoint-borrow elimination, a unified axpy, branch-free m <= 64
   coefficient multiplies).  At B = 1 the min-degree column equals the
   scalar Berlekamp-Massey output (verified by a test).
3. Reconstruction: batched block Horner.  Candidates are random
   combinations of the annihilator rows.  Z accumulates
   sum_j M^j (V A_j) over the Horner depth ell + 1, starting from the
   unmodified start block V.  Batches of B candidates are verified
   independently (up to 4 batches; batch 0 uses the unit combination, so
   a typical instance resolves on attempt 1).
4. The certificate.  A candidate is accepted only if it is nonzero and
   annihilates the raw unconditioned stack (W p == 0).  This gate is the
   solver's correctness certificate, and nothing downstream may weaken
   it.  The conditioning hazard above is the standing example of why.

Determinism: all randomness is seeded (instance, conditioning weights,
start blocks, reconstruction combinations), and every arithmetic
operation is order-free over GF(2^m).  Kernels are therefore
bit-identical across machines and thread counts at equal seeds.  This
was observed on Intel and AMD laptops, on EPYC and Xeon servers, and on
the GPU backend.

### Output coordinates

Internally the engine works in colex-ranked coordinates over the pivot-
permuted systematic form of G, which is cache-friendly for the circuit.
The written kernel.bin is p_lex: the full C(n, lam) Pluecker coordinate
vector in lex order over the original coordinates, matching the order of
`compound_matrix` and of the Python recovery's contraction index.  The
map (to_lex_plucker) and its inverse are unit-tested against the dense
compound construction, and the dense nullspace of E at the toy shape is
verified to be one-dimensional with the wedge as the only kernel
direction.

### GPU backend (opt-in: `--features gpu`)

wgpu runs the same circuit as one op-switched WGSL kernel.  Tests
enforce bit-identity with the CPU operator, including chunked multi-step
chaining, for m up to 128 and lam = 2 and 3.  Design points determined
empirically:

- One queue submission per chunk of up to 32 H-steps.  Per-dispatch
  parameter blocks are stored in a read-only STORAGE buffer selected by
  dynamic offsets (uniform bindings are capped at 16 KiB).  Steps chain
  on device through a copy kernel, because same-buffer
  copy_buffer_to_buffer is forbidden in WebGPU.
- Compute passes per submission are capped at 384, with a poll between
  submissions.  Above about 500 passes, one submission silently corrupts
  on Mesa/ANV (passes are dropped, results are wrong, no error is
  reported).  Rewriting the parameter blob while the previous submission
  is in flight corrupts even small submissions.  Benchmarks with 1-step
  chunks do not reach either limit; only full-scale solves do.  Any new
  backend or driver must re-verify the cap.
- The multiply inside the GPU kernel is shift-and-add over bit limbs
  (2-4 u32 planes per element).  Dispatch shaping and batching removed
  the submission and host bottlenecks.  A windowed or bitsliced multiply
  is the next known lever (the superseded dense route sustained 6-8
  G GF(2^50) mult/s with a bitsliced kernel).
- Adapter selection uses wgpu's default device and can select an iGPU on
  machines with several devices.  At m <= 64 the CPU route is usually
  faster than an iGPU anyway.

### Progress reporting

Long phases (Krylov, approximant basis, reconstruction, scalar Horner)
print throttled progress to stderr.  The total step count is printed
when the phase starts, then one line is printed per MF_PROGRESS_SECS
seconds (default 10, 0 disables) with percent complete, elapsed time,
and a linear ETA.  Stdout carries results only.  Phases shorter than the
interval print only the initial line.

## Stage 4: recovery (Python, recovery.py)

Input: the public G and the kernel vector p (a Pluecker coordinate
vector of the hidden lambda-plane L, defined up to a scalar).  Output:
an equivalent decryption key, with every step verified against public
data.  Numbering follows the paper's Steps 1-5.  The normative output
contract and the independent verification recipe are in SPECIFICATION.md
section 7.  This section explains how the steps work internally.

1. Plane extraction.  z is contracted with lambda random covectors:
   U[j, t] = <z, wedge(cov_1, ..., e_t^*, ..., cov_lambda)> is evaluated
   as a sum of 3x3 (lambda x lambda) determinants, giving a lambda x n
   basis R of L.  A degenerate random draw produces a zero contraction
   and is retried with a fresh seed.  Acceptance is the Las Vegas test
   rank(R) = lam and rank(G^{[-i]} R^T) < lam for all i < r.  The plane
   must satisfy this simultaneously for the whole Frobenius family.
2. Intersection lines.  For each i, h_i spans rowsp(ker G) intersect
   L^{[i]}.  It is computed as the unique solution of a small joint
   nullspace system, and uniqueness is checked rather than assumed.
3. Frobenius-shift system.  R is completed to an invertible n x n stack
   by appending unit rows (rank-checked).  Each h_i^{[-i]} is expressed
   in that basis (membership checked), and the Fqm-linear system
   mu_i h~_{i+1} = h~_i^{[-1]} T is solved over the unknowns: the
   entries of T (lambda x lambda) and the scalars mu_i.  Uniqueness of
   the solution (dimension exactly 1) is where the condition
   r >= lam + 2 enters.  T must be invertible.
4. Twisted norm and matrix Hilbert 90.  The norm equation
   N(T) = T T^{[-1]} ... T^{[-(m-1)]} = I is checked.  The cocycle
   Q^{[-1]} T = Q is then solved homogeneously over F2 on the bit
   matrices of the lambda x lambda ring, giving a lambda-dimensional
   fixed space.  An invertible Q_hat is drawn from it, and its cocycle
   identity is re-verified.
5. Assembly and equivalent key.  B_hat = Q_hat R and gamma_hat =
   h~_0 Q_hat^{-1}.  The public-side identity rowsp(gamma_hat
   B_hat^{[i]}) = rowsp(ker G) is verified for every i.  The
   equivalent Loidreau key is then extracted: b_hat, a rank-weight-n
   support (n entries of B_hat spanning an n-dimensional F2 space,
   checked), and the mask P_hat = sum_j gamma_hat_j P_j (invertibility
   checked).  ok=True at this point is a per-instance proof, from public
   data alone, that an equivalent decryption key has been recovered.

The module also provides a legacy entry point for anchored kernels
(r^lam coordinates, a legacy input format from an earlier
development version).  The
dispatcher selects the entry point by kernel size.

## Stage 5: decryption demo (Python, pke.py)

The Loidreau PKE adds a rank-metric error e of weight t to the codeword:
y = m G_pub + e, with the Loidreau parameter condition
lam * t <= (n-k)/2.  The demo encrypts fresh random plaintexts with the
public key, decrypts the ciphertexts with the recovered equivalent key
(b_hat, P_hat) alone via the rank decoder, and checks that message and
error match on every trial.  This demonstrates that the recovered key is
a working decryption key, not only a kernel certificate.

## Known limits

- The CPU path is lam-generic.  The GPU path is exercised at lam = 2
  and 3.
- Instances with n > m cannot have rank-weight-n supports (gen warns).
  Recovery additionally requires n - k >= lam + 2, which the driver
  enforces.
- Row-1-class shapes fit in 16 GB of VRAM with room to spare.  Compute
  the planar-vector and schedule budgets before scaling B or N on
  smaller cards (schedules are about 12*T_and bytes, and planar vectors
  are m/8 bytes per element per column, times a few copies).
