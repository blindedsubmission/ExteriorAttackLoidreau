# Validation log

Tree validated: this release directory as of 2026-09-17 (evening),
master commit range b5d23f3..57647bf plus this note.  Hardware: Tiger
Lake laptop, i5-1135G7 (4c/8t), Iris Xe iGPU unused (CPU route), galois
0.4.11 / numpy 2.5.3, rustc release builds.  Solver seed 1, conditioning
on everywhere.

## Software tests

- `cargo test --release` (CPU-only default build): **27 passed, 0
  failed**, with the GPU tests correctly excluded and 5 ignored profiling probes).
- `cargo test --release --features gpu`: **39 passed, 0 failed**,
  including CPU/GPU bit-identity of full H-steps (chunked chaining,
  lam 2 and 3, m up to 128) and GPU-multiply/host-parity sweeps
  m in {24, 40, 64, 65, 96, 100, 128}.
- Python dense/lowms/pke toy suites (`main.py toys`, `toys lowms`,
  `toys pke` against the same recovery modules): **all passed**.
- Clippy: `cargo clippy --release` is **zero warnings** with the
  pedantic lint group enabled (policy: crate attributes in main.rs,
  with targeted allowances for limb-packed field arithmetic and indexed
  hot loops).  The same holds for `--features gpu` with and without
  `--tests`.

## Release pipeline runs (`pipeline.py`)

- `all` on `pipeline-toy` (24;16,10,3, Python keygen seed 7):
  **PASS** (keygen, export, solve with block B=8, recovery ok=True,
  PKE demo 3/3 decryptions (t=1).  Wall time under a minute.
- `verify`: **PASS**.  The random control instance yields no kernel
  and exits nonzero.  A one-bit-corrupted kernel.bin is rejected by
  recovery at plane verification (`rank(G^[-0] R^T) = 3 (not < 3)`),
  demonstrating the checks are not vacuous.
- `stage recover lowms128a` + `stage demo lowms128a` at the paper's
  row-1 shape (61;50,25,3): **ok=True** and **3/3 decryptions (t=4)**,
  recovery 4.2 s, demo 6.0 s (kernel from the solve below).

## Bring-your-own-key conformance (specification spot-check)

`bring_your_own_key.py` implements the specification independently of the
pipeline's key generation (own field candidate table, own sampling, own
serialization, and own rank/verification code.  Only the solver binary and the
recovery/decryption modules under test are reused).  Run on
(24;16,10,3), modulus x^24+x^4+x^3+x+1 chosen by the script:

    foreign key -> Rust solve -> recovery ok=True
    check (a) rank weight of b_hat = n          OK
    check (b) P_hat invertible                  OK
    check (c) rowsp(Moore_r(b_hat) P_hat) == rowsp(ker G_pub)   OK
    check (d) foreign ciphertext (m, e) match under recovered key  OK

## End-to-end ladder (same sources, earlier driver)

All runs use solver seed 1 with conditioning on.  Kernel times are solve-only,
recovery/demo are the small Python stages.

| instance | (m; n, k, lam) | route | solve | recovery | demo |
|---|---|---|---|---|---|
| pipeline-toy | (24;16,10,3) | A, block B=8 CPU | ~1 s | ok | 3/3 (t=1) |
| pipeline-toy-rand | (24;16,10,3) random | control | no kernel, exit 1 | skipped | - |
| toy40rs | (40;32,16,3) | A, block B=8 CPU | 116 s | ok | - |
| t80rs | (50;50,32,3) | A (see note) | prior session | ok in ~4 s | - |
| m128toy | (128;14,10,2) | A, block B=8 CPU | 1.1 s | ok (m=128 path) | - |
| lowms128a | (61;50,25,3) | B (real LowMS key) | 6044.7 s | ok in 3.9 s | 3/3 (t=4) |

lowms128a solve phase split (laptop CPU): Krylov 7,383 steps in
4,362.4 s (590.8 ms/step, equal to the single-step bench, with no
sustained-load penalty at this shape), approximant basis 285.2 s with
degree profile [2449, 2450 x7] (c = 2450), batch-Horner attempt 1 ->
kernel found.  Determinism: kernel scalars at equal seeds reproduce
bit-for-bit across machines (toy40 c = 0x3c900a3367 matched an
independent Xeon server run.  t80m and the GPU backend behave likewise).

Notes, honestly labeled:

- The t80m kernel was produced by a prior session the same day.  Its key
  was regenerated deterministically (seed 21) and the recovery run is
  the certificate (public data only).  No new t80m solve is claimed.
- Cross-version kernel comparisons are meaningless across the modulus
  change in the engine history.  Every run here used meta-driven fields.
- The historical RTX 4090 figure for the row-1 shape (12.3 min solve)
  was measured on an earlier development build before this release
freeze, and the
  released CPU default is the reproducible baseline.
