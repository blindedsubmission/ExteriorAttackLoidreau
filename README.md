# Exterior key recovery for Loidreau-type rank-metric encryption

Released artifact of the paper's attack.  The pipeline maps a public key
to a verified equivalent decryption key in five stages across two
components: Python for cryptography and verification, Rust for the
kernel search.  All instances use the same code path.

Independent key generation is supported.  SPECIFICATION.md defines the
field, the scheme, and all file formats precisely enough for an
independent implementation.  bring_your_own_key.py is such an
independent implementation: it generates its own keys, feeds the public
bytes to the pipeline, and verifies the recovered key with its own
checks.

    Python keygen          Rust kernel finder         Python recovery + demo
  -------------------    -----------------------    ------------------------
  sample secret key  ->  exterior kernel vector -> equivalent key (Steps 1-5)
  + public key           of the public compound     verified against PUBLIC
  (instances.py)         stack (block Wiedemann     data; then decrypt real
                         + PM-BASIS annihilator)    ciphertexts with it

A run that ends in `PASS` establishes, for the given instance and
without any secret information:

- the kernel vector annihilates the public stack (W p = 0, p != 0),
- the extracted plane satisfies all rank conditions,
- the Frobenius-shift system, twisted norm, and Hilbert-90 steps close,
- the assembled key generates the same public code as the published key,
- that key decrypts ciphertexts produced with the public key.

All checks use public data only.

## Layout

    pipeline.py        one-command driver (stages, summary, exit codes)
    SPECIFICATION.md   precise key/ciphertext formats and the recovery
                       contract (implementable from scratch)
    bring_your_own_key.py  independent key generator + conformance test
    ARCHITECTURE.md    how the solver works and why it is built this way
    AI_DISCLOSURE.md   LLM-assistance provenance and result verification
    traces/            reference run logs and SHA256 checksums
    instances.txt      instance registry (shape, seed, kind)
    requirements.txt   pinned Python dependencies
    python/            the Python modules of the pipeline
    rust/              the kernel finder crate (CPU by default, GPU opt-in)
    instances/         per-instance artifacts (created at run time)
    VALIDATION.md      validation log of this exact tree
    LICENSE            MIT license

## Requirements

- Linux x86-64 is the tested platform.  The shipped
  `rust/.cargo/config.toml` is scoped to `x86_64-unknown-linux-gnu`
  (gcc linker plus `target-cpu=native`).  Other targets use platform
  defaults.  The field multiplication uses PCLMULQDQ with runtime
  detection on x86-64; non-x86 builds use a portable slow path.
- Rust >= 1.85 (edition 2024), from rustup or a distribution rustc,
  with a working C linker.  On Linux the shipped config selects gcc.
- Python >= 3.10 with `galois` and `numpy` (pinned in requirements.txt).
- No GPU is required.  The default Rust build is CPU-only (dependency:
  rayon).  A wgpu/Vulkan GPU backend is available with
  `cargo build --release --features gpu` and is faster on discrete
  GPUs.  The CPU figures below are the reproducible baseline.

## Build and install

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    (cd rust && cargo build --release)     # optional; pipeline.py also
                                           # auto-builds on first use

## Run

    .venv/bin/python pipeline.py all                 # toy, ~1 minute
    .venv/bin/python pipeline.py all lowms128a       # the paper's row-1 instance
    .venv/bin/python pipeline.py verify              # negative + tamper controls

`all` runs keygen, export, solve, recover, and demo, prints per-stage
timings, and exits 0 only if every stage succeeded.
`stage <keygen|export|solve|recover|demo> [instance]` runs a single
stage.

## Artifacts and cleaning up

The pipeline writes to two locations inside the release tree:

- `instances/<name>/`: all run artifacts, namely the public key
  (`<name>-rs.key.bin`), metadata, `kernel.bin`, and the keygen npz.
- `rust/target/`: cargo build artifacts.

`python3 pipeline.py clean` removes `instances/`.  `clean --all` also
removes `rust/target/`.  Source files under `python/` and `rust/src/`
are never written (the driver disables bytecode caches).

## What a run looks like (pipeline-toy)

Annotated excerpt of a full toy run (times from a 4-core laptop):

    === [keygen pipeline-toy (Python)] ===            # secret/public key, GF(2^24)
    pipeline-toy: kind=loidreau m=24 D=2160 ...       # instance built
    === [export ...] ===                              # key.bin + meta in Rust format
    export_rust_key: .../pipeline-toy-rs.key.bin (160 elements), poly x^24+g ...
    === [solve ... (Rust kernel finder, block B=8)] ===
    field: x^24+g g=(0x1e6a9,0x0) (from meta.json)    # solved in the key's own field
    block(pmb) krylov NNN steps ... approximant basis ... KERNEL FOUND
    .../kernel.bin written (560 standard coords)
    === [recover ... (Python Steps 1-5)] ===
    plane contraction: ok (attempt 1/8)               # kernel -> hidden plane
    plane verification: ok                            # rank(G^[-i] R^T) < lam, all i
    step3: shift system dim 1, T invertible           # Frobenius-shift system
    step4: twisted norm = I, fixed space dim = lam    # norm equation + Hilbert 90
    step5: rowsp(H_hat) == rowsp(ker G)                # same public code
    key: wt(b_hat)=n, P_hat invertible -> equivalent key verified
    recovery: ok=True stage=done err=None
    === [demo ...] ===                                # encrypt public, decrypt recovered
    trial 0: decrypted with the recovered key, (m, e) match: YES
    #### PASS: pipeline-toy: keygen -> solve -> recover -> decrypt all succeeded ####

## Verification and labeling

- The Rust engine accepts a candidate kernel only if it is a nonzero
  null vector of the raw unconditioned public stack (W p = 0).
- The Python recovery re-derives the key independently and checks it
  against the public key alone: plane rank conditions for every
  Frobenius shift, uniqueness of the shift-system solution,
  invertibility, the twisted-norm equation, the F_2 fixed space, the
  rowspace identity, and the equivalent-key structure (rank-weight-n
  support, invertible mask).
- The PKE demo encrypts fresh random plaintexts with the public key and
  decrypts them with the recovered key alone.
- `pipeline.py verify` shows that the checks are not vacuous: a random
  control code must produce no kernel, and a one-bit-corrupted kernel
  must be rejected by recovery.
- Determinism: key generation and the solver are seeded, and all field
  and solver arithmetic uses order-free XOR/multiply operations.  Kernels
  and verification outcomes reproduce bit-for-bit across machines
  (observed on an Intel laptop, a Ryzen desktop, and EPYC and Xeon
  servers).
- Labeling: the wall times reported below are kernel-search times plus
  small recovery and demo stages.  They are not end-to-end
  cryptanalysis times of a production adversary, and the demo instances
  have research sizes.

## Progress during long runs

Solve phases report progress on stderr.  The total step count is
printed when a phase starts (`[krylov] 0/7383`), then one line is
printed every 10 seconds with percent complete, elapsed time, and ETA
(`[krylov] 3200/7383 (43%) elapsed 1620s, eta 2140s`).  This applies to
the Krylov, approximant-basis, and reconstruction phases.  Set
`MF_PROGRESS_SECS=<seconds>` to change the interval (`0` disables).
Stdout carries results only.

## Instances and measured runtimes

| instance | (m; n, k, lam) | kind | what it is | solve time | whole `all` |
|---|---|---|---|---|---|
| pipeline-toy | (24; 16, 10, 3) | loidreau | smoke test | ~1 s | ~1 min |
| lowms128a | (61; 50, 25, 3) | lowms | the paper's row-1 instance | 100.7 min on a 4-core laptop CPU, 12.4 min on an RX 6800 XT with `--features gpu` | plus ~10 s recovery/demo |
| pipeline-toy-rand | (24; 16, 10, 3) | random | negative control only | exits nonzero, no kernel | - |

(CPU figure: 2026-09-17, Tiger Lake i5-1135G7 laptop.  GPU figure:
2026-09-18, RX 6800 XT, this release's engine, same seeds as the CPU
run and the same kernel.  Scale with core count and clock.)

Add your own instances by appending a row to `instances.txt`
(`name m n k lam seed kind [s]`).  Key recovery additionally requires
`n - k >= lam + 2`, and the driver rejects shapes that violate this
bound.

## License

MIT.  See `LICENSE`.

## Troubleshooting

- `cargo build` complains about edition 2024: update Rust to >= 1.85.
- `galois` install fails: use the pinned versions from requirements.txt.
  galois 0.4.x is required for API stability.
- Solve reports "no verified kernel found" on a structured or lowms
  instance: rerun once.  The attack is randomized (solver seed, default
  1) and Las Vegas.  A persistent failure indicates a build or
  environment problem.  Do not remove the W p == 0 gate.
- `--gpu` reports that it needs `--features gpu`: this is the CPU-only
  default build.  Rebuild with `cargo build --release --features gpu`.

## Reference traces

`traces/` holds full stage logs of the reference runs (the toy
pipeline and the row-1 instance on the GPU route) and `SHA256SUMS.txt`,
the checksums of the deterministic artifacts.  After running the
pipeline, `sha256sum -c traces/SHA256SUMS.txt` must report OK for every
file: key exports and kernels reproduce bit-for-bit from the instance
and solver seeds alone, on any machine or backend.  `traces/README.md`
states the exact inputs (registry rows, solver seed 1, build and
library versions) and which log lines are machine-specific (timings
only).  Home-directory paths in the traces are replaced by `$RELEASE`.

## Provenance

The implementation was developed with LLM assistance.  See
AI_DISCLOSURE.md for the provenance statement and for how results are
verified independently of the code that produced them.

This artifact is self-contained: the `python/` modules and the
`rust/` crate in this directory are everything the pipeline needs.
VALIDATION.md records the end-to-end validation of this exact tree:
the instance ladder, the negative and tamper controls, the hardware,
and measured wall times.
