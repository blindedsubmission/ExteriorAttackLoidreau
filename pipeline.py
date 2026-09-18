#!/usr/bin/env python3
"""One-command driver for the released attack pipeline.

Stages (in order, one code path for every instance):

  keygen   sample the instance secret + public key (Python, instances.py)
  export   write the public key in the Rust engine's format (16-byte LE
           field elements + meta.json with the field modulus)
  solve    find an exterior kernel vector of the public stack
           (Rust matrixfree-rs, block Wiedemann + PM-BASIS annihilator)
  recover  kernel -> equivalent decryption key, verified against public
           data only (Python, recovery.py Steps 1-5)
  demo     encrypt with the public key, decrypt with the recovered key
           alone (Python, pke.py)

Usage:
  python3 pipeline.py all                 # toy instance, ~1 minute
  python3 pipeline.py all lowms128a       # the paper's row-1 instance
  python3 pipeline.py stage <name> [inst] # a single stage
  python3 pipeline.py verify              # negative + tamper controls
  python3 pipeline.py clean [--all]       # remove instances/ [+ rust/target]

Every stage prints what it does and how long it took.  `all` ends with a
PASS/FAIL summary and exit status 0 only if every stage succeeded.  See
README.md for requirements, expected output, and honest caveats.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

RELEASE_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(RELEASE_ROOT, "python")
INSTANCES_DIR = os.path.join(RELEASE_ROOT, "instances")
RUST_DIR = os.path.join(RELEASE_ROOT, "rust")
BIN = os.path.join(RUST_DIR, "target", "release", "matrixfree-rs")

# Keep the source tree pristine: all run artifacts go to instances/,
# build artifacts to rust/target/, and no __pycache__ next to the sources.
sys.dont_write_bytecode = True
sys.path.insert(0, PYTHON_DIR)

DEFAULT_INSTANCE = "pipeline-toy"


def inst_dir(name):
    return os.path.join(INSTANCES_DIR, name)


# The research-tree modules resolve artifact directories relative to their
# own file location; pin them to the release tree instead.
import instances as instances_mod  # noqa: E402

instances_mod.inst_dir = inst_dir


def log(msg):
    print(msg, flush=True)


class Stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        log(f"\n=== [{self.name}] ===")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        self.seconds = dt
        log(f"=== [{self.name}] {'FAILED' if exc_type else 'done'} "
            f"({dt:.1f}s) ===")
        return False


def load_registry():
    reg = instances_mod.load_registry(
        os.path.join(RELEASE_ROOT, "instances.txt"))
    return reg


def check_registry(name):
    reg = load_registry()
    if name not in reg:
        sys.exit(f"unknown instance {name!r}; known: {', '.join(reg)}")
    return reg[name]


def ensure_binary():
    if os.path.exists(BIN):
        return
    log("rust binary missing; building (cargo build --release) ...")
    t0 = time.time()
    r = subprocess.run(["cargo", "build", "--release"], cwd=RUST_DIR)
    if r.returncode != 0:
        sys.exit("cargo build failed, see output above "
                 "(rust >= 1.85 required, edition 2024)")
    log(f"built in {time.time() - t0:.0f}s")


def run(cmd, **kw):
    log("+ " + " ".join(cmd))
    return subprocess.run(cmd, **kw)


def clean(all_only=False):
    """Remove run artifacts (instances/) and, with --all, build artifacts
    (rust/target/) too.  Source files are never touched."""
    removed = []
    if os.path.isdir(INSTANCES_DIR):
        shutil.rmtree(INSTANCES_DIR)
        removed.append("instances/")
    if all_only and os.path.isdir(os.path.join(RUST_DIR, "target")):
        shutil.rmtree(os.path.join(RUST_DIR, "target"))
        removed.append("rust/target/")
    log("removed: " + (", ".join(removed) if removed else "(nothing to do)"))


# ----------------------------------------------------------------- stages

def stage_keygen(name):
    with Stage(f"keygen {name} (Python)") as s:
        m, n, k, lam, seed, kind, s_ovr = check_registry(name)
        instances_mod.build(name, m, n, k, lam, seed, kind,
                            s_override=s_ovr, blocks=False)


def stage_export(name):
    with Stage(f"export {name} (Python -> Rust key format)") as s:
        instances_mod.export_rust_key(name)


def stage_solve(name, mode="block", block=8):
    with Stage(f"solve {name} (Rust kernel finder, {mode} B={block})") as s:
        d = inst_dir(name)
        meta = json.load(open(os.path.join(d, f"{name}-rs.meta.json")))
        if meta["n"] - meta["k"] < meta["lam"] + 2:
            sys.exit(f"{name}: n-k = {meta['n'] - meta['k']} < lam+2 = "
                     f"{meta['lam'] + 2}: key recovery is infeasible at "
                     "this shape (README: recovery bounds)")
        cmd = [BIN, "solve",
               "--key", os.path.join(d, f"{name}-rs.key.bin"),
               "--mode", mode, "--block", str(block),
               "--out", os.path.join(d, "kernel.bin")]
        r = run(cmd)
        if r.returncode != 0:
            sys.exit(f"solve failed (exit {r.returncode}), no kernel "
                     "written.  This is the correct outcome for random "
                     "control instances, and a failure otherwise")


def stage_recover(name):
    with Stage(f"recover {name} (Python Steps 1-5)") as s:
        import recovery
        res = recovery.engine_case(name, dirname=inst_dir(name))
        if not res:
            sys.exit("recovery failed, see the stage error above")


def stage_demo(name):
    with Stage(f"demo {name} (decrypt with the recovered key alone)") as s:
        import pke
        if not pke.demo_case(name):
            sys.exit("PKE demo failed")


def stage_all(name):
    log(f"\n#### pipeline `all` on {name!r} ####")
    stage_keygen(name)
    stage_export(name)
    stage_solve(name)
    stage_recover(name)
    stage_demo(name)
    log(f"\n#### PASS: {name}: keygen -> solve -> recover -> decrypt "
        "all succeeded ####")


# ----------------------------------------------------------------- verify

def verify():
    """Controls that show the checks are not vacuous."""
    ok = True

    # 1. negative control: a random code must yield no kernel
    neg = "pipeline-toy-rand"
    stage_keygen(neg)
    stage_export(neg)
    d = inst_dir(neg)
    kernel = os.path.join(d, "kernel.bin")
    if os.path.exists(kernel):
        os.remove(kernel)
    with Stage(f"verify: random control {neg} must find no kernel"):
        m, n, k, lam, seed, kind, s_ovr = check_registry(neg)
        cmd = [BIN, "solve",
               "--key", os.path.join(d, f"{neg}-rs.key.bin"),
               "--mode", "block", "--block", "8", "--out", kernel]
        r = run(cmd)
        if r.returncode == 0 or os.path.exists(kernel):
            log("FAIL: random control produced a kernel (or exited 0)")
            ok = False
        else:
            log("random control: no kernel, nonzero exit, as expected")

    # 2. tamper control: a corrupted kernel must fail recovery loudly
    src = inst_dir(DEFAULT_INSTANCE)
    good = os.path.join(src, "kernel.bin")
    if not os.path.exists(good):
        sys.exit(f"verify needs a solved {DEFAULT_INSTANCE}: run "
                 f"`python3 pipeline.py all` first")
    tampered = os.path.join(src, "kernel-tampered.bin")
    with Stage("verify: tampered kernel must fail recovery"):
        raw = bytearray(open(good, "rb").read())
        raw[0] ^= 0x01  # flip the lowest bit of the first coordinate
        with open(tampered, "wb") as f:
            f.write(raw)
        import recovery
        res = recovery.equivalent_key_from_kernel(
            DEFAULT_INSTANCE, dirname=src, kernel_path=tampered)
        if res["ok"]:
            log("FAIL: tampered kernel passed recovery, checks are broken")
            ok = False
        else:
            log(f"tampered kernel rejected: stage={res['stage']} "
                f"err={res['err']}, as expected")

    log("\n#### VERIFY " + ("PASS" if ok else "FAIL") + " ####")
    sys.exit(0 if ok else 1)


# ------------------------------------------------------------------- main

def main():
    args = [a for a in sys.argv[1:]]
    if not args:
        sys.exit(__doc__)
    cmd = args.pop(0)
    if cmd == "clean":
        clean(all_only="--all" in args)
        return
    if cmd == "stage":
        if not args:
            sys.exit("stage <keygen|export|solve|recover|demo> [instance]")
        st = args.pop(0)
        name = args.pop(0) if args and not args[0].startswith("-") \
            else DEFAULT_INSTANCE
        fn = {"keygen": stage_keygen, "export": stage_export,
              "solve": stage_solve, "recover": stage_recover,
              "demo": stage_demo}.get(st)
        if fn is None:
            sys.exit(f"unknown stage {st!r}")
        check_registry(name)
        ensure_binary()
        fn(name)
        return
    if cmd == "verify":
        ensure_binary()
        verify()
        return
    name = args.pop(0) if args and not args[0].startswith("-") \
        else DEFAULT_INSTANCE
    check_registry(name)
    ensure_binary()
    if cmd == "all":
        stage_all(name)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
