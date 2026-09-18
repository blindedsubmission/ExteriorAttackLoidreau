//! Matrix-free exterior kernel-recovery engine over GF(2^m), m <= 128.
//!
//! Rust reimplementation of the "substitute and discard" circuit pipeline
//! (`../matrixfree/` in this repository): applying the stacked compound
//! system E = (`C_lam`([`I_k` | `D_i`]))_i and its transpose to exterior
//! vectors without materializing any compound matrix, then finding
//! ker E by (scalar or block) Wiedemann on the char-2-conditioned square
//! wrapper M = E^T diag(c) E.  CPU backend (rayon) and GPU backend
//! (wgpu, bit-identical arithmetic) share all setup and postprocessing.
//!
//! See README.md for the algorithm, the cost model, and the design
//! rationale; each module carries its own documentation.

// Lint policy: pedantic clippy on, with allowances for the deliberate
// patterns of this codebase -- integer narrowing/widening in limb-packed
// field arithmetic, single-letter/similar names in numeric code, indexed
// loops in hot schedule walks, and long solver orchestration functions.
// Crate-level attributes rather than the Cargo [lints] table: immune to
// command-line flag ordering and clippy warning-cache quirks.
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::unused_self,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::wrong_self_convention,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::format_push_string
)]
// Test code keeps the default clippy set but not the pedantic group.
#![cfg_attr(test, allow(clippy::pedantic))]

mod circuit;
mod dense;
mod engine;
mod field;
#[cfg(feature = "gpu")]
mod gpu;
mod instance;
mod linalg;
mod operator;
mod pmbasis;
mod poly;
mod progress;
mod rng;
mod subset;

use field::{Fe, Field};
use rng::Xorshift64;
use std::io::Write;
use std::path::{Path, PathBuf};

fn parse_size(s: &str) -> usize {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        usize::from_str_radix(hex, 16).unwrap_or_else(|e| panic!("bad number {s}: {e}"))
    } else {
        s.parse().unwrap_or_else(|e| panic!("bad number {s}: {e}"))
    }
}

fn parse_u64(s: &str) -> u64 {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).unwrap_or_else(|e| panic!("bad number {s}: {e}"))
    } else {
        s.parse().unwrap_or_else(|e| panic!("bad number {s}: {e}"))
    }
}

/// Serialize a key/secret/kernel vector: 16 bytes per element (two
/// little-endian u64 words), row-major.  Compatible with reading back
/// both the m <= 64 and m > 64 layouts.
fn write_fe_vec(path: &Path, v: &[Fe]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for &(lo, hi) in v {
        f.write_all(&lo.to_le_bytes())?;
        f.write_all(&hi.to_le_bytes())?;
    }
    Ok(())
}

fn read_fe_vec(path: &Path, expect: usize) -> Vec<Fe> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    assert_eq!(bytes.len(), expect * 16, "file {} has wrong size", path.display());
    bytes
        .chunks_exact(16)
        .map(|c| {
            (
                u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]),
                u64::from_le_bytes([c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]]),
            )
        })
        .collect()
}

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() -> ! {
    eprintln!(
        "matrixfree-rs {VERSION} -- matrix-free exterior kernel recovery, GF(2^m), m <= 128\n\
         \n\
         USAGE:\n\
           matrixfree-rs <COMMAND> [OPTIONS]\n\
         \n\
         COMMANDS:\n\
           gen     Sample a Loidreau key or random control code\n\
           solve   Run the Wiedemann engine on a key file\n\
           bench   Time one H-step at the given shape\n\
           info    Inspect a key / meta file\n\
           help    Show this help (or `help <command>`)\n\
         \n\
         Run `matrixfree-rs <command> --help` for command-specific options.\n\
         \n\
         All instance parameters follow the (m; n, k, lam) convention of the\n\
         research notes; s defaults to n-k (full stack).\n\
         \n\
         EXAMPLES:\n\
           matrixfree-rs gen --m 24 --n 16 --k 12 --lam 3 --seed 7 --out /tmp/toy\n\
           matrixfree-rs solve --m 24 --n 16 --k 12 --lam 3 --key /tmp/toy.key.bin --expect-wedge /tmp/toy --mode block --block 8\n\
           matrixfree-rs bench --m 50 --n 50 --k 32 --lam 3 --block 8\n\
           matrixfree-rs bench --m 50 --n 50 --k 32 --lam 3 --gpu --block 8"
    );
    std::process::exit(2);
}

fn usage_gen() -> ! {
    eprintln!(
        "matrixfree-rs gen -- generate a key instance\n\
         \n\
         USAGE:\n\
           matrixfree-rs gen --m M --n N --k K --lam L --seed S --out PREFIX [--random]\n\
         \n\
         OPTIONS:\n\
           --m M        extension degree, 1..=128\n\
           --n N        code length\n\
           --k K        code dimension (k < n)\n\
           --lam L      subspace dimension, 1 <= lam <= k\n\
           --seed S     u64 seed (decimal or 0x hex)\n\
           --out PREFIX output prefix; writes PREFIX.key.bin, PREFIX.meta.json,\n\
                        and for structured keys PREFIX.secret.bin + PREFIX.wedge.bin\n\
           --random     sample a random full-rank code instead of Loidreau structured\n\
           -h, --help   show this help\n\
         \n\
         NOTES:\n\
           Structured keys sample the parity-check model of the paper: rank-weight-n\n\
           support g, F2-independent gamma with gamma0=1, and a random mask\n\
           with entries spanning that lam-space.  The wedge ground truth is\n\
           written to PREFIX.wedge.bin (standard lex Pluecker order)."
    );
    std::process::exit(2);
}

fn usage_solve() -> ! {
    eprintln!(
        "matrixfree-rs solve -- run the Wiedemann engine on a key file\n\
         \n\
         USAGE:\n\
           matrixfree-rs solve --m M --n N --k K --lam L --key FILE [OPTIONS]\n\
         \n\
         REQUIRED:\n\
           --m M              extension degree (or omit if FILE.meta.json is present)\n\
           --n N              code length\n\
           --k K              dimension\n\
           --lam L            subspace dimension\n\
           --key FILE         path to k*n*16 byte key file (written by `gen`)\n\
         \n\
         OPTIONS:\n\
           --s S              stack height, 1..=n-k (default: n-k, full stack)\n\
           --seed SEED        Wiedemann / conditioning seed (default: 1)\n\
           --mode MODE        scalar | block (default: scalar)\n\
           --block B          block size for --mode block (default: 8, 1..=64)\n\
           --gpu              run the block Krylov steps on GPU (wgpu)\n\
           --no-cond          disable the char-2 diagonal conditioning (not recommended)\n\
           --out FILE         kernel output path (default: kernel.bin in CWD)\n\
           --expect-wedge PREFIX  check result against PREFIX.wedge.bin (structured keys)\n\
           -h, --help         show this help\n\
         \n\
         OUTPUT:\n\
           Writes the kernel vector in standard lex Pluecker order (16 bytes per\n\
           coordinate, same as PREFIX.wedge.bin) to --out.  Prints whether the\n\
           vector satisfies E*z==0 and whether it is proportional to the planted\n\
           wedge when --expect-wedge is given.

         NOTES:
         The field modulus f = x^m + g is taken from the key's sibling
         meta.json (poly_g_lo/poly_g_hi) whenever present -- a key is only
         solved in the field it was generated in.  Without a recorded poly,
         the build's default modulus is used with a loud warning."
    );
    std::process::exit(2);
}

fn usage_bench() -> ! {
    eprintln!(
        "matrixfree-rs bench -- time one H-step at the given shape (random key)\n\
         \n\
         USAGE:\n\
           matrixfree-rs bench --m M --n N --k K --lam L [OPTIONS]\n\
         \n\
         OPTIONS:\n\
           --m M, --n N, --k K, --lam L   instance shape (required)\n\
           --block B     block size B for the H-step (default: 8)\n\
           --gpu         time the GPU H-step instead of CPU (rayon)\n\
           --s S         stack height (default: n-k); kept for parity with solve\n\
           -h, --help    show this help"
    );
    std::process::exit(2);
}

fn usage_info() -> ! {
    eprintln!(
        "matrixfree-rs info -- inspect a key / meta file\n\
         \n\
         USAGE:\n\
           matrixfree-rs info --key FILE [--m M --n N --k K --lam L]\n\
                          info --meta FILE\n\
         \n\
         OPTIONS:\n\
           --key FILE   key file to inspect (reads sibling .meta.json if present)\n\
           --meta FILE  meta.json file to dump\n\
           -h, --help   show this help\n\
         \n\
         Prints file sizes, expected dimensions, and field polynomial."
    );
    std::process::exit(2);
}

fn bail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    eprintln!("run `matrixfree-rs --help` or `matrixfree-rs <command> --help` for usage");
    std::process::exit(2);
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn arg_or(args: &[String], name: &str, cmd_help: fn() -> !) -> String {
    let i = args.iter().position(|a| a == name).unwrap_or_else(|| cmd_help());
    args.get(i + 1).unwrap_or_else(|| cmd_help()).clone()
}

fn opt_arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .map(|i| args.get(i + 1).unwrap_or_else(|| bail(&format!("{name} requires a value"))).clone())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    // global flags
    match args[0].as_str() {
        "-h" | "--help" | "help" => {
            if args.len() >= 2 {
                match args[1].as_str() {
                    "gen" => usage_gen(),
                    "solve" => usage_solve(),
                    "bench" => usage_bench(),
                    "info" => usage_info(),
                    _ => usage(),
                }
            } else {
                usage();
            }
        }
        "-V" | "--version" | "version" => {
            println!("matrixfree-rs {VERSION}");
            std::process::exit(0);
        }
        _ => {}
    }
    // allow `matrixfree-rs --help` without subcommand already handled;
    // also handle `matrixfree-rs gen --help` etc inside dispatch
    match args[0].as_str() {
        "gen" => {
            if has_flag(&args, "-h") || has_flag(&args, "--help") {
                usage_gen();
            }
            cmd_gen(&args);
        }
        "solve" => {
            if has_flag(&args, "-h") || has_flag(&args, "--help") {
                usage_solve();
            }
            cmd_solve(&args);
        }
        "bench" => {
            if has_flag(&args, "-h") || has_flag(&args, "--help") {
                usage_bench();
            }
            cmd_bench(&args);
        }
        "info" => {
            if has_flag(&args, "-h") || has_flag(&args, "--help") {
                usage_info();
            }
            cmd_info(&args);
        }
        _ => {
            bail(&format!("unknown command {:?}", args[0]));
        }
    }
}

fn ensure_parent_dir(path: &Path) {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("cannot create directory {}: {e}", parent.display()));
        }
    }
}

fn cmd_gen(args: &[String]) {
    let m: u32 = parse_size(&arg_or(args, "--m", usage_gen)) as u32;
    let n = parse_size(&arg_or(args, "--n", usage_gen));
    let k = parse_size(&arg_or(args, "--k", usage_gen));
    let lam = parse_size(&arg_or(args, "--lam", usage_gen));
    let seed: u64 = parse_u64(&arg_or(args, "--seed", usage_gen));
    let out = PathBuf::from(arg_or(args, "--out", usage_gen));
    let random = has_flag(args, "--random");
    if !(1..=128).contains(&(m as usize)) {
        bail("m must be in 1..=128");
    }
    if k >= n {
        bail("need k < n");
    }
    if !(1 <= lam && lam <= k) {
        bail("need 1 <= lam <= k");
    }
    if n as u32 > m && !random {
        // random still needs full rank check; structured needs n<=m for rank weight n
        eprintln!("warning: n > m: rank-weight-n support impossible, sampling will loop; consider m >= n");
    }
    let field = Field::new(m);
    let mut rng = Xorshift64::new(seed);
    let t0 = std::time::Instant::now();
    let inst = if random {
        instance::random_instance(&field, n, k, &mut rng)
    } else {
        loop {
            if let Some(i) = instance::loidreau_with_secret(&field, n, k, lam, &mut rng) {
                break i;
            }
        }
    };
    let key_path = out.with_extension("key.bin");
    ensure_parent_dir(&key_path);
    write_fe_vec(&key_path, &inst.g_pub).expect("write key");
    let meta = format!(
        "{{\"m\":{m},\"n\":{n},\"k\":{k},\"lam\":{lam},\"random\":{random},\"seed\":{seed},\"poly_g_lo\":{},\"poly_g_hi\":{}}}\n",
        field.g.0, field.g.1
    );
    let meta_path = out.with_extension("meta.json");
    std::fs::write(&meta_path, meta).unwrap_or_else(|e| panic!("cannot write {}: {e}", meta_path.display()));
    if let Some(b_rows) = &inst.b_rows {
        let sec_path = out.with_extension("secret.bin");
        write_fe_vec(&sec_path, b_rows).unwrap_or_else(|e| panic!("cannot write {}: {e}", sec_path.display()));
        let wedge = inst.wedge_lex(&field, lam).unwrap();
        let wedge_path = out.with_extension("wedge.bin");
        write_fe_vec(&wedge_path, &wedge).unwrap_or_else(|e| panic!("cannot write {}: {e}", wedge_path.display()));
        println!(
            "gen: kind={} m={m} n={n} k={k} lam={lam} poly=x^{m}+g g=(0x{:x},0x{:x}) [{:.1}s]",
            if random { "random" } else { "loidreau" },
            field.g.0,
            field.g.1,
            t0.elapsed().as_secs_f64()
        );
        println!("  wrote {} ({} bytes), {} and {}", key_path.display(), inst.g_pub.len()*16, sec_path.display(), wedge_path.display());
    } else {
        println!(
            "gen: kind=random m={m} n={n} k={k} lam={lam} poly=x^{m}+g g=(0x{:x},0x{:x}) [{:.1}s]",
            field.g.0,
            field.g.1,
            t0.elapsed().as_secs_f64()
        );
        println!("  wrote {} ({} bytes)", key_path.display(), inst.g_pub.len()*16);
    }
    println!("  meta: {}", meta_path.display());
}

struct MetaInfo {
    m: u32,
    n: usize,
    k: usize,
    lam: usize,
    /// Field modulus f = x^m + g recorded by `gen`; the key's ground truth.
    poly: Option<(u64, u64)>,
}

fn read_meta_json(path: &PathBuf) -> Option<MetaInfo> {
    let txt = std::fs::read_to_string(path).ok()?;
    // minimal ad-hoc parser: look for "key": integer pairs
    let get = |key: &str| -> Option<u64> {
        let pat = format!("\"{key}\"");
        let idx = txt.find(&pat)?;
        let after = &txt[idx + pat.len()..];
        let colon = after.find(':')?;
        let rest = after[colon + 1..].trim_start();
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        rest[..end].parse().ok()
    };
    let poly = match (get("poly_g_lo"), get("poly_g_hi")) {
        (Some(lo), Some(hi)) => Some((lo, hi)),
        _ => None,
    };
    Some(MetaInfo {
        m: get("m")? as u32,
        n: get("n")? as usize,
        k: get("k")? as usize,
        lam: get("lam").or_else(|| get("lambda"))? as usize,
        poly,
    })
}

fn sibling_meta_path(key_file: &Path) -> PathBuf {
    let s = key_file.to_string_lossy();
    if s.ends_with(".key.bin") {
        PathBuf::from(s.replacen(".key.bin", ".meta.json", 1))
    } else {
        key_file.with_extension("meta.json")
    }
}

/// The field the key actually lives in: the poly recorded in its meta.json
/// when present, else this build's default (with a loud warning -- keys
/// generated by a different build may use a different default modulus).
fn field_for_key(m: u32, meta: Option<&MetaInfo>) -> Field {
    if let Some((lo, hi)) = meta.and_then(|mi| mi.poly) { match Field::with_poly(m, (lo, hi)) {
        Some(f) => {
            eprintln!("field: x^{m}+g g=(0x{lo:x},0x{hi:x}) (from meta.json)");
            f
        }
        None => {
            bail(&format!(
                "meta.json poly g=(0x{lo:x},0x{hi:x}) fails the irreducibility test for m={m}"
            ));
        }
    } } else {
        let f = Field::new(m);
        eprintln!(
            "warning: meta.json records no poly_g_lo/poly_g_hi; using this build's default x^{m}+g g=(0x{:x},0x{:x}) -- if the key was generated elsewhere, check the modulus matches",
            f.g.0, f.g.1
        );
        f
    }
}

/// GPU solve dispatch.  The default build has no GPU backend: --gpu is a
/// clear error instead of a missing-symbol build failure.  Scalar mode
/// runs the block engine at width 1 (bit-identical sequences).
#[cfg(not(feature = "gpu"))]
fn run_gpu(_op: operator::Operator, _seed: u64, _mode: &str, _block: usize,
           _dense_ann: bool) -> ! {
    bail("--gpu requires a build with --features gpu (wgpu/Vulkan); \
          the default build is CPU-only")
}

#[cfg(feature = "gpu")]
fn run_gpu(op: operator::Operator, seed: u64, mode: &str, block: usize,
           dense_ann: bool) -> engine::EngineResult {
    let r = match (mode, dense_ann) {
        ("block", true) => gpu::block_wiedemann_gpu(op, seed, block),
        ("block", false) => gpu::block_wiedemann_gpu_pmb(op, seed, block),
        _ => gpu::block_wiedemann_gpu_pmb(op, seed, 1),
    };
    match r {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gpu error: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_solve(args: &[String]) {
    let key_file = PathBuf::from(arg_or(args, "--key", usage_solve));
    // Try to auto-fill m/n/k/lam from sibling meta.json if not given explicitly
    let meta_guess = read_meta_json(&sibling_meta_path(&key_file));
    let m: u32 = if let Some(v) = opt_arg(args, "--m") {
        parse_size(&v) as u32
    } else if let Some(mi) = meta_guess.as_ref() {
        eprintln!("solve: --m not given, using m={} from sibling meta.json", mi.m);
        mi.m
    } else {
        usage_solve();
    };
    let n = if let Some(v) = opt_arg(args, "--n") {
        parse_size(&v)
    } else if let Some(mi) = meta_guess.as_ref() {
        eprintln!("solve: --n not given, using n={} from sibling meta.json", mi.n);
        mi.n
    } else {
        usage_solve();
    };
    let k = if let Some(v) = opt_arg(args, "--k") {
        parse_size(&v)
    } else if let Some(mi) = meta_guess.as_ref() {
        eprintln!("solve: --k not given, using k={} from sibling meta.json", mi.k);
        mi.k
    } else {
        usage_solve();
    };
    let lam = if let Some(v) = opt_arg(args, "--lam") {
        parse_size(&v)
    } else if let Some(mi) = meta_guess.as_ref() {
        eprintln!("solve: --lam not given, using lam={} from sibling meta.json", mi.lam);
        mi.lam
    } else {
        usage_solve();
    };
    if !(1..=128).contains(&(m as usize)) {
        bail("m must be in 1..=128");
    }
    if k >= n {
        bail("need k < n");
    }
    if !(1 <= lam && lam <= k) {
        bail("need 1 <= lam <= k");
    }
    let s = opt_arg(args, "--s").map_or(n - k, |x| parse_size(&x));
    if s == 0 || s > n - k {
        bail(&format!("s must be in 1..=n-k ({})", n - k));
    }
    let seed: u64 = opt_arg(args, "--seed").map_or(1, |x| parse_u64(&x));
    let mode = opt_arg(args, "--mode").unwrap_or_else(|| "scalar".into());
    if mode != "scalar" && mode != "block" {
        bail(" --mode must be scalar or block");
    }
    let block = opt_arg(args, "--block").map_or(8, |x| parse_size(&x));
    if !(1..=64).contains(&block) {
        bail(" --block must be in 1..=64");
    }
    let use_gpu = has_flag(args, "--gpu");
    let dense_ann = has_flag(args, "--dense-ann");
    let cond = !has_flag(args, "--no-cond");
    let expect_wedge = opt_arg(args, "--expect-wedge");
    let out_path = opt_arg(args, "--out").map_or_else(|| PathBuf::from("kernel.bin"), PathBuf::from);

    let field = field_for_key(m, meta_guess.as_ref());
    let circuit = circuit::Circuit::new(n, k, lam);
    let g_pub = read_fe_vec(&key_file, k * n);
    let t0 = std::time::Instant::now();
    let op = operator::Operator::from_generator(field.clone(), circuit, &g_pub, s, seed, cond);
    println!(
        "setup: m={m} n={n} k={k} lam={lam} s={s} N={} K={} T_and={} (dense {}) | conditioning {}",
        op.n,
        op.k_coords,
        op.circuit.t_and,
        op.k_coords * op.n,
        if cond { "on" } else { "OFF" },
    );

    // block modes default to the approximant-basis annihilator;
    // --dense-ann selects the historical dense block-Hankel nullspace
    let res = match (mode.as_str(), use_gpu) {
        ("scalar", false) => engine::scalar_wiedemann(&op, seed),
        ("block", false) if dense_ann => engine::block_wiedemann(&op, seed, block),
        ("block", false) => engine::block_wiedemann_pmb(&op, seed, block),
        (_, true) => run_gpu(op, seed, mode.as_str(), block, dense_ann),
        _ => usage_solve(),
    };
    println!("{}", res.stats);
    if res.ok {
        ensure_parent_dir(&out_path);
        write_fe_vec(&out_path, &res.p_lex).unwrap_or_else(|e| panic!("cannot write {}: {e}", out_path.display()));
        println!("{} written ({} standard coords)", out_path.display(), res.p_lex.len());
        if let Some(prefix) = expect_wedge {
            // accept both prefix (without extension) and full wedge path
            let wedge_file = {
                let p = PathBuf::from(&prefix);
                if p.extension().is_some() {
                    p
                } else {
                    PathBuf::from(format!("{prefix}.wedge.bin"))
                }
            };
            let wedge = read_fe_vec(&wedge_file, res.p_lex.len());
            if let Some(c) = field.proportional(&res.p_lex, &wedge) { println!("wedge check: kernel = c * planted wedge, c = ({:#x}, {:#x})", c.0, c.1) } else {
                println!("wedge check: MISMATCH (kernel not proportional to planted wedge)");
                std::process::exit(1);
            }
        }
    } else {
        println!("no verified kernel found");
        std::process::exit(1);
    }
    println!("total {:.1}s", t0.elapsed().as_secs_f64());
}

fn cmd_bench(args: &[String]) {
    let m: u32 = parse_size(&arg_or(args, "--m", usage_bench)) as u32;
    let n = parse_size(&arg_or(args, "--n", usage_bench));
    let k = parse_size(&arg_or(args, "--k", usage_bench));
    let lam = parse_size(&arg_or(args, "--lam", usage_bench));
    let bs = opt_arg(args, "--block").map_or(8, |x| parse_size(&x));
    let s_opt = opt_arg(args, "--s").map(|x| parse_size(&x));
    let use_gpu = has_flag(args, "--gpu");
    if !(1..=128).contains(&(m as usize)) {
        bail("m must be in 1..=128");
    }
    if k >= n {
        bail("need k < n");
    }
    if !(1 <= lam && lam <= k) {
        bail("need 1 <= lam <= k");
    }
    if !(1..=64).contains(&bs) {
        bail("--block must be in 1..=64");
    }
    let field = Field::new(m);
    let circuit = circuit::Circuit::new(n, k, lam);
    let mut rng = Xorshift64::new(1);
    let inst = instance::random_instance(&field, n, k, &mut rng);
    let s = s_opt.unwrap_or(n - k);
    let op = operator::Operator::from_generator(field, circuit, &inst.g_pub, s, 1, true);
    let mask = op.field.mask();
    let mut r = Xorshift64::new(2);
    let y: Vec<Fe> = (0..op.n * bs).map(|_| r.next_fe(mask)).collect();
    let reps: u32 = 10;
    if use_gpu {
        #[cfg(not(feature = "gpu"))]
        {
            bail("--gpu requires a build with --features gpu (wgpu/Vulkan); \
                  the default build is CPU-only")
        }
        #[cfg(feature = "gpu")]
        match gpu::GpuEngine::new(op.clone(), bs) {
            Ok(mut g) => {
                let mut yy = y.clone();
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    let _ = g.hstep_dots(&mut yy);
                }
                let elapsed = t0.elapsed();
                let dt = elapsed / reps;
                let ups = (op.s * op.circuit.t_and * bs) as f64 / dt.as_secs_f64();
                println!("bench gpu: B={bs} {dt:.2?}/H-step ({ups:.2} column-updates/s)");
                // per-op attribution: GPU timestamps of the last
                // step, aggregated by op; empty when timestamps unsupported
                let ts = &g.last_step_ts_us;
                if ts.is_empty() {
                    println!("  gpu timestamps: unsupported on this adapter");
                } else {
                    let r = op.circuit.r;
                    // actual per-step dispatch order (step_list):
                    // dots, broadcast, fwd×r, weight, embed, trn×r,
                    // reduce, handoff = 2r + 6 passes; the query buffer is
                    // over-allocated (ndispatch(2r)*2), unwritten tail is 0
                    let n_pass = 2 * r + 6;
                    let total: f32 = ts[..n_pass].iter().sum();
                    let seg = |lo: usize, hi: usize| -> f32 { ts[lo..hi.min(n_pass)].iter().sum() };
                    let row = |name: &str, us: f32| {
                        println!("  {:<10} {:>9.2} us  {:>5.1}%", name, us, 100.0 * us / total);
                    };
                    row("dots", seg(0, 1));
                    row("broadcast", seg(1, 2));
                    row("fwd", seg(2, 2 + r));
                    row("weight", seg(2 + r, 3 + r));
                    row("embed", seg(3 + r, 4 + r));
                    row("trn", seg(4 + r, 4 + 2 * r));
                    row("reduce", seg(4 + 2 * r, 5 + 2 * r));
                    row("handoff", seg(5 + 2 * r, 6 + 2 * r));
                    println!("  {:<10} {:>9.2} us  ({} passes; rest of wall = host/inter-pass)", "sum", total, n_pass);
                }
            }
            Err(e) => {
                eprintln!("gpu error: {e}");
                std::process::exit(1);
            }
        }
    } else {
        let mut yy = y.clone();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            op.hstep_block(&mut yy, bs);
        }
        let elapsed = t0.elapsed();
        let dt = elapsed / reps;
        let ups = (op.s * op.circuit.t_and * bs) as f64 / dt.as_secs_f64();
        println!(
            "bench cpu ({} threads): B={bs} {dt:.2?}/H-step ({ups:.2} column-updates/s)",
            rayon::current_num_threads(),
            ups = ups
        );
    }
}

fn cmd_info(args: &[String]) {
    if let Some(meta) = opt_arg(args, "--meta") {
        let p = PathBuf::from(&meta);
        let txt = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
        println!("{txt}");
        if let Some(mi) = read_meta_json(&p) {
            let circuit = circuit::Circuit::new(mi.n, mi.k, mi.lam);
            println!(
                "decoded: m={} n={} k={} lam={} N={} K={} T_and={}",
                mi.m, mi.n, mi.k, mi.lam, circuit.n_coords, circuit.k_coords, circuit.t_and
            );
            if let Some((lo, hi)) = mi.poly { match Field::with_poly(mi.m, (lo, hi)) {
                Some(f) => println!("poly (from meta): x^{}+g g=(0x{:x},0x{:x})", f.m, f.g.0, f.g.1),
                None => println!("poly (from meta): INVALID g=(0x{lo:x},0x{hi:x}) for m={}", mi.m),
            } } else {
                let f = Field::new(mi.m);
                println!("poly (build default; meta has no poly): x^{}+g g=(0x{:x},0x{:x})", f.m, f.g.0, f.g.1);
            }
        }
        return;
    }
    let key = PathBuf::from(arg_or(args, "--key", usage_info));
    let meta_path = sibling_meta_path(&key);
    let bytes = std::fs::metadata(&key).map_or(0, |m| m.len());
    println!("key: {} ({} bytes, {} field elements)", key.display(), bytes, bytes / 16);
    if let Ok(txt) = std::fs::read_to_string(&meta_path) {
        println!("meta: {} -> {}", meta_path.display(), txt.trim());
        if let Some(mi) = read_meta_json(&meta_path) {
            println!("shape: m={} n={} k={} lam={} s={} N={} K={} T_and={}", mi.m, mi.n, mi.k, mi.lam, mi.n - mi.k, {
                let c = circuit::Circuit::new(mi.n, mi.k, mi.lam); c.n_coords
            }, {
                let c = circuit::Circuit::new(mi.n, mi.k, mi.lam); c.k_coords
            }, {
                let c = circuit::Circuit::new(mi.n, mi.k, mi.lam); c.t_and
            });
            if let Some((lo, hi)) = mi.poly { match Field::with_poly(mi.m, (lo, hi)) {
                Some(f) => println!("poly (from meta): x^{}+g g=(0x{:x},0x{:x})", f.m, f.g.0, f.g.1),
                None => println!("poly (from meta): INVALID g=(0x{lo:x},0x{hi:x}) for m={}", mi.m),
            } } else {
                let f = Field::new(mi.m);
                println!("poly (build default; meta has no poly): x^{}+g g=(0x{:x},0x{:x})", f.m, f.g.0, f.g.1);
            }
            if bytes as usize != mi.k * mi.n * 16 {
                eprintln!("warning: key file size {} != k*n*16 ({})", bytes, mi.k * mi.n * 16);
            }
            // also check companion files
            let wedge = {
                let s = key.to_string_lossy();
                if s.ends_with(".key.bin") {
                    PathBuf::from(s.replacen(".key.bin", ".wedge.bin", 1))
                } else {
                    key.with_extension("wedge.bin")
                }
            };
            if wedge.exists() {
                let wb = std::fs::metadata(&wedge).map_or(0, |m| m.len());
                println!("wedge: {} ({} bytes, {} coords)", wedge.display(), wb, wb/16);
            }
            let secret = {
                let s = key.to_string_lossy();
                if s.ends_with(".key.bin") {
                    PathBuf::from(s.replacen(".key.bin", ".secret.bin", 1))
                } else {
                    key.with_extension("secret.bin")
                }
            };
            if secret.exists() {
                let sb = std::fs::metadata(&secret).map_or(0, |m| m.len());
                println!("secret: {} ({} bytes)", secret.display(), sb);
            }
        }
    } else {
        eprintln!("no meta.json at {}", meta_path.display());
        // try to infer from --m etc if given
        if let (Some(ms), Some(ns), Some(ks), Some(ls)) = (opt_arg(args, "--m"), opt_arg(args, "--n"), opt_arg(args, "--k"), opt_arg(args, "--lam")) {
            let m: u32 = parse_size(&ms) as u32;
            let n = parse_size(&ns);
            let k = parse_size(&ks);
            let lam = parse_size(&ls);
            println!("inferred shape from CLI: m={m} n={n} k={k} lam={lam}");
            if bytes as usize != k*n*16 {
                eprintln!("warning: key size mismatch");
            }
        }
    }
}

/// TSC-based profiling probes for the hot loops (no privileges
/// needed); #[ignore]d so the regular suite stays short.
#[cfg(test)]
mod proftmp {
    use crate::circuit::Circuit;
    use crate::field::{Fe, Field};
    use crate::instance::random_instance;
    use crate::operator::Operator;
    use crate::rng::Xorshift64;
    use std::arch::x86_64::_rdtsc;
    use std::time::Instant;

    fn tsc() -> u64 {
        unsafe { _rdtsc() }
    }

    fn raw_clmul(a: u64, b: u64) -> (u64, u64) {
        unsafe {
            use std::arch::x86_64::*;
            let p = _mm_clmulepi64_si128(_mm_set_epi64x(0, a as i64), _mm_set_epi64x(0, b as i64), 0x00);
            (_mm_extract_epi64(p, 0) as u64, _mm_extract_epi64(p, 1) as u64)
        }
    }

    /// cycles per schedule entry at the benchmark shape, B=1 (run pinned)
    /// Run explicitly: cargo test --release proftmp -- --ignored --nocapture
    #[test]
    #[ignore]
    fn prof_entry_cost() {
        let field = Field::new(50);
        let (n_, k_, lam) = (50usize, 32usize, 3usize);
        let circuit = Circuit::new(n_, k_, lam);
        let mut rng = Xorshift64::new(3);
        let inst = random_instance(&field, n_, k_, &mut rng);
        let op = Operator::from_generator(field.clone(), circuit, &inst.g_pub, n_ - k_, 1, true);
        let entries = (op.s * op.circuit.t_and * 2) as f64;
        for &bs in &[1usize, 8] {
            let mask = field.mask();
            let mut r = Xorshift64::new(5);
            let mut y: Vec<Fe> = (0..op.n * bs).map(|_| r.next_fe(mask)).collect();
            op.hstep_block(&mut y, bs);
            let reps = if bs == 1 { 20 } else { 8 };
            let c0 = tsc();
            let t0 = Instant::now();
            for _ in 0..reps {
                op.hstep_block(&mut y, bs);
            }
            let dtsc = tsc() - c0;
            let dt = t0.elapsed();
            let total = entries * bs as f64 * reps as f64;
            let ghz = dtsc as f64 / dt.as_secs_f64() / 1e9;
            eprintln!(
                "entry-cost B={}: {:.2} ms/step, {:.1} cyc/entry, tsc={:.2} GHz",
                bs,
                dt.as_secs_f64() * 1000.0 / reps as f64,
                dtsc as f64 / total,
                ghz
            );
        }
    }

    /// mul throughput (independent) vs latency (dependent) vs raw clmul
    /// Run explicitly: cargo test --release proftmp -- --ignored --nocapture
    #[test]
    #[ignore]
    fn prof_mul_floors() {
        let f = Field::new(50);
        let mask = f.mask();
        let mut r = Xorshift64::new(7);
        let a: Vec<Fe> = (0..4096).map(|_| r.next_fe(mask)).collect();
        let b: Vec<Fe> = (0..4096).map(|_| r.next_fe(mask)).collect();
        let n = 200.0 * 4096.0;
        // throughput (xor into acc keeps independence)
        let c0 = tsc();
        let t0 = Instant::now();
        let mut acc = 0u64;
        for rep in 0..200 {
            for i in 0..a.len() {
                let v = f.mul(a[i], b[(i + rep) & 4095]);
                acc ^= v.0;
            }
        }
        let dtsc = tsc() - c0;
        let dt = t0.elapsed();
        eprintln!(
            "mul throughput m=50: {:.2} cyc/mul ({:.2} ns) [acc {:#x}]",
            dtsc as f64 / n,
            dt.as_secs_f64() * 1e9 / n,
            acc
        );
        // dependent chain
        let c0 = tsc();
        let mut x = f.one();
        for rep in 0..200 {
            for i in 0..a.len() {
                x = f.mul(x, a[i]);
                x = f.mul(x, b[(i + rep) & 4095]);
            }
        }
        let dtsc = tsc() - c0;
        eprintln!("mul dependent m=50: {:.2} cyc/mul", dtsc as f64 / n);
        // raw clmul throughput
        let c0 = tsc();
        let mut acc2 = 0u64;
        for rep in 0..1000 {
            for i in 0..a.len() {
                let (lo, _hi) = raw_clmul(a[i].0 ^ (rep as u64), b[i & 4095].0);
                acc2 ^= lo;
            }
        }
        let dtsc = tsc() - c0;
        let n2 = 1000.0 * 4096.0;
        eprintln!("raw clmul: {:.2} cyc/clmul [acc {:#x}]", dtsc as f64 / n2, acc2);
        // fold round counts at m=50 (transcribed fold with counters)
        let g = f.g.0;
        let mut hist_hi = vec![0u32; 8];
        let mut hist_lo = vec![0u32; 8];
        for i in 0..4096 {
            for rep in 0..50 {
                let (mut lo, mut hi) = raw_clmul(a[i].0, b[(i + rep) & 4095].0);
                let m = 50u32;
                let shift = 64 - m;
                let mut rounds_hi = 0;
                while hi != 0 {
                    let (u0, u1) = raw_clmul(hi, g);
                    lo ^= u0 << shift;
                    hi = (u1 << shift) | (u0 >> m);
                    rounds_hi += 1;
                }
                let mut rounds_lo = 0;
                while (lo >> m) != 0 {
                    let (q0, _q1) = raw_clmul(lo >> m, g);
                    lo = (lo & ((1u64 << m) - 1)) ^ q0;
                    rounds_lo += 1;
                }
                hist_hi[rounds_hi.min(7) as usize] += 1;
                hist_lo[rounds_lo.min(7) as usize] += 1;
            }
        }
        eprintln!("fold rounds (hi-loop) histogram: {:?}", hist_hi);
        eprintln!("fold rounds (lo-loop) histogram: {:?}", hist_lo);
    }

    /// per-phase split of one block-column pass (serial, B=1)
    /// Run explicitly: cargo test --release proftmp -- --ignored --nocapture
    #[test]
    #[ignore]
    fn prof_phases() {
        let field = Field::new(50);
        let (n_, k_, lam) = (50usize, 32usize, 3usize);
        let circuit = Circuit::new(n_, k_, lam);
        let mut rng = Xorshift64::new(3);
        let inst = random_instance(&field, n_, k_, &mut rng);
        let op = Operator::from_generator(field.clone(), circuit, &inst.g_pub, n_ - k_, 1, true);
        let (n, rr, r) = (op.n, op.k_coords, op.circuit.r);
        let kr = op.circuit.k * r;
        let f = &op.field;
        let flat = &op.circuit.flat;
        let starts = &op.circuit.stage_starts;
        let dib = &op.di[0..kr];
        let mask = f.mask();
        let mut rng2 = Xorshift64::new(9);
        let y0: Vec<Fe> = (0..n).map(|_| rng2.next_fe(mask)).collect();
        let (mut t_gather, mut t_fwd, mut t_weight, mut t_trn, mut t_scatter) =
            (0u128, 0u128, 0u128, 0u128, 0u128);
        let reps = 100;
        for _ in 0..reps {
            let t = Instant::now();
            let mut yc = y0.clone();
            t_gather += t.elapsed().as_nanos();
            let t = Instant::now();
            for b in (0..r).rev() {
                let seg = &flat[(starts[b] * 3) as usize..(starts[b + 1] * 3) as usize];
                for tr in seg.chunks_exact(3) {
                    let cf = dib[tr[0] as usize];
                    let idx = tr[2] as usize;
                    yc[idx] = f.add(yc[idx], f.mul(cf, yc[tr[1] as usize]));
                }
            }
            t_fwd += t.elapsed().as_nanos();
            let t = Instant::now();
            for x in 0..n {
                yc[x] = if x < rr { f.mul(yc[x], op.cw[x]) } else { f.zero() };
            }
            t_weight += t.elapsed().as_nanos();
            let t = Instant::now();
            for b in 0..r {
                let seg = &flat[(starts[b] * 3) as usize..(starts[b + 1] * 3) as usize];
                for tr in seg.chunks_exact(3) {
                    let cf = dib[tr[0] as usize];
                    let idx = tr[1] as usize;
                    yc[idx] = f.add(yc[idx], f.mul(cf, yc[tr[2] as usize]));
                }
            }
            t_trn += t.elapsed().as_nanos();
            let t = Instant::now();
            let _ = yc.iter().fold(f.zero(), |acc, x| f.add(acc, *x));
            t_scatter += t.elapsed().as_nanos();
        }
        let tot = t_gather + t_fwd + t_weight + t_trn + t_scatter;
        eprintln!(
            "phase split (1 block, 1 column, {} reps): gather {:.1}% fwd {:.1}% weight {:.1}% trn {:.1}% scatter {:.1}% | per-rep total {:.2} ms",
            reps,
            100.0 * t_gather as f64 / tot as f64,
            100.0 * t_fwd as f64 / tot as f64,
            100.0 * t_weight as f64 / tot as f64,
            100.0 * t_trn as f64 / tot as f64,
            100.0 * t_scatter as f64 / tot as f64,
            tot as f64 / reps as f64 / 1e6
        );
    }

    /// engine-side dots cost (not part of hstep_block bench)
    /// Run explicitly: cargo test --release proftmp -- --ignored --nocapture
    #[test]
    #[ignore]
    fn prof_dots() {
        let field = Field::new(50);
        let (n_, k_, lam) = (50usize, 32usize, 3usize);
        let circuit = Circuit::new(n_, k_, lam);
        let mut rng = Xorshift64::new(3);
        let inst = random_instance(&field, n_, k_, &mut rng);
        let op = Operator::from_generator(field, circuit, &inst.g_pub, n_ - k_, 1, true);
        let bs = 8usize;
        let mask = op.field.mask();
        let mut r = Xorshift64::new(11);
        let u: Vec<Fe> = (0..op.n * bs).map(|_| r.next_fe(mask)).collect();
        let v: Vec<Fe> = (0..op.n * bs).map(|_| r.next_fe(mask)).collect();
        let t0 = Instant::now();
        let c0 = tsc();
        let reps = 20;
        let mut acc = vec![op.field.zero(); bs * bs];
        for _ in 0..reps {
            for c1 in 0..bs {
                for c2 in 0..bs {
                    let mut a = op.field.zero();
                    for x in 0..op.n {
                        a = op.field.add(a, op.field.mul(u[x * bs + c1], v[x * bs + c2]));
                    }
                    acc[c1 * bs + c2] = a;
                }
            }
        }
        eprintln!(
            "dots B=8: {:.2} ms/step ({:.0} cyc/dot-element) [first {:#x}]",
            t0.elapsed().as_secs_f64() * 1000.0 / reps as f64,
            (tsc() - c0) as f64 / (reps * bs * bs * op.n) as f64,
            acc[0].0
        );
    }
}
