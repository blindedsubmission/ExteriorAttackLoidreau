# Reference traces and exact reproduction inputs

The pipeline is deterministic: given the instance seed and the solver
seed, every output byte (public key export, kernel vector, metadata) is
reproducible on any machine, build, thread count, or backend, because
all field arithmetic is exact and order-free (XOR/add/multiply over
GF(2^m), no floating point, seeded randomness throughout).  This was
verified by bit-identity of outputs across a 4-core laptop, a 12-core
desktop, and an RX 6800 XT GPU route.  Only wall-time lines in the
traces below are machine-specific.

## Exact inputs

1.  Instance registry rows (`instances.txt`, the only instance data
    under version control):

        pipeline-toy   24  16  10   3     7  loidreau
        lowms128a      61  50  25   3    44  lowms     25

    Fields: name m n k lam seed kind [s].  The field modulus f = x^m + g
    is derived at keygen and recorded in each instance's meta.json
    (toy: g = 0x1e6a9; lowms128a: g = 0x27; bit i of g = coefficient
    of x^i, SPECIFICATION section 2).

2.  Solver seed: the default 1 (conditioning on, block Wiedemann B=8).
    Nothing else is passed.

3.  Environment (tested, 2026-09-18): Python 3.13, galois 0.4.11,
    numpy 2.5.3 (pinned in requirements.txt); rustc/cargo 1.98.1,
    edition 2024, `-C target-cpu=native` (shipped .cargo/config.toml);
    GPU route: wgpu 30 on Mesa/RADV, RX 6800 XT (16 GB).  Rebuild
    native binaries on your machine (see README, Requirements).

4.  Commands, from the release root:

        python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
        (cd rust && cargo build --release)          # add --features gpu for the GPU route
        .venv/bin/python pipeline.py all             # toy
        .venv/bin/python pipeline.py all lowms128a   # the row-1 instance

## What the traces contain

    pipeline-toy-all.txt      full `pipeline.py all` (toy), this tree
    lowms128a-solve-gpu.txt   full `pipeline.py all lowms128a`, GPU route
    SHA256SUMS.txt            checksums of the deterministic artifacts

Home-directory paths are replaced by $RELEASE/$REPO.  A reviewer checks
reproduction with `sha256sum -c traces/SHA256SUMS.txt` after running
the pipeline; log lines other than the throttled progress lines and
timings must match verbatim.

For the provenance of the implementation itself (LLM-assisted
development under the authors' direction, certificate-based
correctness), see README, Provenance.
