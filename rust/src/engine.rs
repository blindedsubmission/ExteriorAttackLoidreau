//! Wiedemann engines over the matrix-free operator: scalar and block.
//!
//! # Scalar
//!
//! From u = `start_vector(seed)`, iterate v <- M v for 2N steps recording
//! `seq[i] = u . M^i u` (dot BEFORE the step); Berlekamp-Massey recovers
//! the minimal polynomial f w.r.t. u; a zero constant coefficient
//! (`f(0) = 0`, i.e. C[L] == 0) certifies a kernel eigenvalue; the kernel
//! candidate is the Horner combination `z = h(M) u` with h = f/x.  The
//! gate is the raw-circuit test `W p == 0 && p != 0`.  Streams and
//! protocol matches the earlier development engines exactly, so at
//! equal seed and m <= 63 both produce the identical kernel vector
//! (cross-validated in tests).
//!
//! # Block (Coppersmith-style)
//!
//! The scalar engine pays 3N H-steps (2N Krylov + ~N Horner).  With a B x
//! N block V of start vectors and a B x N block U of projections, the
//! B x B matrix sequence `S_t = U^T M^t V` determines M after only
//! e ~ N/B steps: solve the homogeneous block-Hankel system
//! `sum_j a_j^T S_{t+j} = 0 (t = 0..e-1)` for unknown vectors
//! `a_0..a_e in F^B`; a solution whose Horner reconstruction
//! `z = sum_j M^j V a_j` is nonzero satisfies `M z = 0` (z is orthogonal
//! to the block Krylov space of U, which fails to span exactly when M is
//! singular), and `W p == 0` gates it as always.  Cost: 2e + O(e) H-steps
//! at B columns each (so ~3N/B H-steps total, a B-fold cut), plus a
//! one-off O((Be)^2 * Be) nullspace on the CPU.  That post-processing is
//! the deliberate simplicity/performance trade of this implementation:
//! production scale would use PM-BASIS (O~(B^2 N)) instead of the dense
//! nullspace; at toy scale the difference is immaterial and the dense
//! solve is much easier to trust.
//!
//! Why B random columns are safe: each H-step applies the same circuit B
//! times (B independent buffers); no step mixes columns, so the arithmetic
//! per column is bit-identical to the scalar engine's.

use crate::field::{Fe, Field};
use crate::linalg::nullspace;
use crate::operator::Operator;
use crate::rng::Xorshift64;

/// Result of an engine run.
pub struct EngineResult {
    /// Whether a kernel vector was found and verified (W p == 0).
    pub ok: bool,
    /// Kernel vector in systematic colex coordinates (empty when !ok).
    /// Colex-order kernel; kept for cross-engine parity tooling.
    #[allow(dead_code)]
    pub z_colex: Vec<Fe>,
    /// The same vector in standard lex Pluecker coordinates.
    pub p_lex: Vec<Fe>,
    pub stats: String,
}

/// Berlekamp-Massey over GF(2^m): returns (L, C) with
/// `sum_{i=0..L} C[i] seq[t-i] = 0` for all t >= L.
/// Direct transcription of the reference engine's routine.
pub fn berlekamp_massey(field: &Field, seq: &[Fe]) -> (usize, Vec<Fe>) {
    let nseq = seq.len();
    let mut c = vec![field.zero(); nseq + 1];
    let mut b = vec![field.zero(); nseq + 1];
    c[0] = field.one();
    b[0] = field.one();
    let mut l = 0usize;
    let mut m = 1usize;
    let mut bb = field.one();
    for q in 0..nseq {
        let mut d = seq[q];
        for i in 1..=l {
            d = field.add(d, field.mul(c[i], seq[q - i]));
        }
        if field.is_zero(d) {
            m += 1;
        } else {
            let coef = field.mul(d, field.inv(bb));
            if 2 * l <= q {
                let tc = c.clone();
                for j in m..=nseq {
                    c[j] = field.add(c[j], field.mul(coef, b[j - m]));
                }
                l = q + 1 - l;
                b = tc;
                bb = d;
                m = 1;
            } else {
                for j in m..=nseq {
                    c[j] = field.add(c[j], field.mul(coef, b[j - m]));
                }
                m += 1;
            }
        }
    }
    (l, c)
}

/// Scalar Wiedemann over the operator.
pub fn scalar_wiedemann(op: &Operator, seed: u64) -> EngineResult {
    let field = &op.field;
    let n = op.n;
    let u = op.start_vector(seed);
    let mut v = u.clone();
    let mut w = vec![field.zero(); n];
    let nsteps = 2 * n;
    let mut seq = vec![field.zero(); nsteps];
    let t0 = std::time::Instant::now();
    let mut prog = crate::progress::Progress::start("krylov", nsteps);
    for i in 0..nsteps {
        let mut acc = field.zero();
        for x in 0..n {
            acc = field.add(acc, field.mul(u[x], v[x]));
        }
        seq[i] = acc;
        op.hstep(&v, &mut w);
        std::mem::swap(&mut v, &mut w);
        prog.tick(i + 1);
    }
    let krylov_secs = t0.elapsed().as_secs_f64();
    let (l, c) = berlekamp_massey(field, &seq);
    let zero_root = l >= 1 && field.is_zero(c[l]);
    let mut stats = format!(
        "krylov {:.1}s ({:.2} ms/H-step, {} steps), BM: L={} of N={}, f(0)={} -> {}",
        krylov_secs,
        krylov_secs * 1000.0 / nsteps as f64,
        nsteps,
        l,
        n,
        c[l].0,
        if zero_root { "ZERO ROOT" } else { "no zero root" },
    );
    if !zero_root {
        stats.push_str("; not deficient (or missed)");
        return EngineResult { ok: false, z_colex: vec![], p_lex: vec![], stats };
    }
    // Horner: z = h(M) u, h = f / x  (skip C[0], the known-zero constant)
    let th = std::time::Instant::now();
    v.clone_from(&u);
    let mut prog = crate::progress::Progress::start("horner", l.saturating_sub(1));
    for i in 1..l {
        op.hstep(&v, &mut w);
        std::mem::swap(&mut v, &mut w);
        let ci = c[i];
        for x in 0..n {
            v[x] = field.add(v[x], field.mul(ci, u[x]));
        }
        prog.tick(i);
    }
    stats.push_str(&format!(
        ", horner {:.1}s, verify Ez==0: {}",
        th.elapsed().as_secs_f64(),
        if op.verify_kernel(&v) { "YES" } else { "NO" },
    ));
    if !op.verify_kernel(&v) {
        return EngineResult { ok: false, z_colex: vec![], p_lex: vec![], stats };
    }
    let p_lex = op.to_lex_plucker(&v);
    stats.push_str(" -> KERNEL FOUND");
    EngineResult { ok: true, z_colex: v, p_lex, stats }
}

/// Block Wiedemann over the operator with a B-column block.
///
/// `steps` overrides the default e = ceil(N/B) + B (extra margin for the
/// annihilator to exist); `retries` random solution-space combinations
/// are tried for the kernel reconstruction.
pub fn block_wiedemann(op: &Operator, seed: u64, bs: usize) -> EngineResult {
    let field = &op.field;
    let n = op.n;
    assert!((1..=64).contains(&bs), "block size 1..=64");
    // blocks U (projections) and V (starts) from the mf-style stream
    let mut rng = Xorshift64::new_mf(seed);
    let mask = field.mask();
    let next = |rng: &mut Xorshift64| -> Fe {
        if field.m <= 64 {
            (rng.next_u64() & mask.0, 0)
        } else {
            rng.next_fe(mask)
        }
    };
    let mut u = vec![field.zero(); n * bs];
    let mut v = vec![field.zero(); n * bs];
    for x in 0..n {
        for c in 0..bs {
            u[x * bs + c] = next(&mut rng);
            v[x * bs + c] = next(&mut rng);
        }
    }
    // ---- Krylov: e steps of M on the whole block, collecting S_t -------
    // Window sizing (see module docs): with c = ceil(N/B) and generator
    // degree d <= c, the annihilator system below must (a) start at t = 1
    // -- the t = 0 equations correspond to the scalar polynomial f, whose
    // h = f/x kernel carrier only annihilates from t = 1, and enforcing
    // t = 0 collapses every solution to a zero reconstruction when the
    // U-block Krylov spans -- and (b) cover enough t for the recurrence
    // to extend past the window (t-range >= ell + d).  Hence
    // ell = c + 2B and e = 3c + 4B + 1.
    let c = n.div_ceil(bs);
    let ell = c + 2 * bs;
    let e = 3 * c + 4 * bs + 1;
    let t_krylov = std::time::Instant::now();
    let v0 = v.clone(); // pristine start block for the reconstruction
    // S_t[c1][c2] = (U column c1) . (V column c2), both interleaved
    let mut seqs: Vec<Vec<Fe>> = Vec::with_capacity(e); // e matrices of BxB (row-major)
    let mut prog = crate::progress::Progress::start("krylov", e);
    for step in 0..e {
        let mut s = vec![field.zero(); bs * bs];
        for c1 in 0..bs {
            for c2 in 0..bs {
                let mut acc = field.zero();
                for x in 0..n {
                    acc = field.add(acc, field.mul(u[x * bs + c1], v[x * bs + c2]));
                }
                s[c1 * bs + c2] = acc;
            }
        }
        seqs.push(s);
        op.hstep_block(&mut v, bs);
        prog.tick(step + 1);
    }
    let t_seq = t_krylov.elapsed();
    // ---- solve the block-Hankel annihilator system ----------------------
    // unknowns a[j*B + c'] for j = 0..=ell; equations for t = 1..e-1-ell,
    // b = 0..B:  sum_{j,c'} S_{t+j}[b][c'] * a[j*B+c'] = 0.  The window
    // deliberately starts at t = 1 (scalar analogue: h = f/x annihilates
    // the sequence only from index 1; this is what keeps the
    // reconstruction from collapsing to zero -- see module docs).
    let unknowns = bs * (ell + 1);
    let trange = (e - 1).saturating_sub(ell);
    let eqs = bs * trange;
    let mut sys = vec![field.zero(); eqs * unknowns];
    for t in 1..=trange {
        for b in 0..bs {
            let row = (t - 1) * bs + b;
            for j in 0..=ell {
                let s = &seqs[t + j];
                for cp in 0..bs {
                    sys[row * unknowns + j * bs + cp] = s[b * bs + cp];
                }
            }
        }
    }
    let t_solve = std::time::Instant::now();
    let sol = nullspace(field, &sys, eqs, unknowns);
    let t_solve_dur = t_solve.elapsed();
    if sol.is_empty() {
        return EngineResult {
            ok: false,
            z_colex: vec![],
            p_lex: vec![],
            stats: format!(
                "block krylov {e} steps {t_seq:.1?}, annihilator system {eqs}x{unknowns} trivial (t={t_solve_dur:.1?}); no kernel direction found"
            ),
        };
    }
    // ---- reconstruct candidates in batches via a BLOCK Horner pass ------
    // For a batch of candidate coefficient vectors c (as block columns),
    //   Z^(c) = sum_j M^j V a_j^(c)
    // is computed by one block-Horner chain over all candidates at once,
    // so a batch of B candidates costs ~ell block steps (one scalar-Krylov
    // pass equivalent), not B separate chains.
    let mut rng2 = Xorshift64::new(seed ^ 0xB10C);
    let nbatches = 4;
    let per_batch = bs.min(sol.len());
    let mut attempt = 0usize;
    for batch in 0..nbatches {
        // coefficient combinations: batch 0 walks basis rows, later
        // batches use random combinations (fresh per batch)
        let ncand = per_batch;
        let mut cmat = vec![field.zero(); sol.len() * ncand];
        if batch == 0 {
            for r in 0..ncand.min(sol.len()) {
                cmat[r * ncand + r] = field.one();
            }
        } else {
            for cf in &mut cmat {
                *cf = next(&mut rng2);
            }
        }
        // a^{(c)}_j = sum_r cmat[r*ncand + c] * sol[r][j*B + b']
        // layout: A[j*bs*ncand + c*bs + b']
        let mut a = vec![field.zero(); bs * ncand * (ell + 1)];
        for r in 0..sol.len() {
            for c in 0..ncand {
                let cr = cmat[r * ncand + c];
                if field.is_zero(cr) {
                    continue;
                }
                for idx in 0..bs * (ell + 1) {
                    let val = field.mul(cr, sol[r][idx]);
                    let dst = (idx / bs) * bs * ncand + c * bs + (idx % bs);
                    a[dst] = field.add(a[dst], val);
                }
            }
        }
        // block Horner: Z (n x ncand interleaved) = sum_j M^j (V A_j)
        let mut z = vec![field.zero(); n * ncand];
        let mut prog = crate::progress::Progress::start("reconstruct", ell + 1);
        for round in 0..=ell {
            let j = ell - round;
            op.hstep_block(&mut z, ncand);
            for x in 0..n {
                for c in 0..ncand {
                    let mut add = field.zero();
                    for bp in 0..bs {
                        add = field.add(add, field.mul(v0[x * bs + bp], a[j * bs * ncand + c * bs + bp]));
                    }
                    z[x * ncand + c] = field.add(z[x * ncand + c], add);
                }
            }
            prog.tick(round + 1);
        }
        // verify each candidate column
        for c in 0..ncand {
            attempt += 1;
            let zc: Vec<Fe> = (0..n).map(|x| z[x * ncand + c]).collect();
            if op.verify_kernel(&zc) {
                let p_lex = op.to_lex_plucker(&zc);
                return EngineResult {
                    ok: true,
                    z_colex: zc,
                    p_lex,
                    stats: format!(
                        "block krylov {e} steps {t_seq:.1?}, annihilator {eqs}x{unknowns} nullity {} ({t_solve_dur:.1?}), batch-Horner attempt {} -> KERNEL FOUND",
                        sol.len(),
                        attempt
                    ),
                };
            }
        }
    }
    EngineResult {
        ok: false,
        z_colex: vec![],
        p_lex: vec![],
        stats: format!(
            "block krylov {e} steps {t_seq:.1?}, annihilator {eqs}x{unknowns} nullity {} ({t_solve_dur:.1?}), all {attempt} batch-Horner candidates failed Ez",
            sol.len()
        ),
    }
}

/// Block Wiedemann with the annihilator from the stacked approximant
/// basis instead of the dense block-Hankel nullspace: same
/// Krylov and same batched block-Horner reconstruction / Ez gate; the
/// O((B e)^3) dense solve is replaced by the O(B^2 e^2) schoolbook
/// iterative basis, which is what makes real shapes feasible.
pub fn block_wiedemann_pmb(op: &Operator, seed: u64, bs: usize) -> EngineResult {
    let field = &op.field;
    let n = op.n;
    assert!((1..=64).contains(&bs), "block size 1..=64");
    let mut rng = Xorshift64::new_mf(seed);
    let mask = field.mask();
    let next = |rng: &mut Xorshift64| -> Fe {
        if field.m <= 64 {
            (rng.next_u64() & mask.0, 0)
        } else {
            rng.next_fe(mask)
        }
    };
    let mut u = vec![field.zero(); n * bs];
    let mut v = vec![field.zero(); n * bs];
    for x in 0..n {
        for c in 0..bs {
            u[x * bs + c] = next(&mut rng);
            v[x * bs + c] = next(&mut rng);
        }
    }
    let c0 = n.div_ceil(bs);
    let e = 3 * c0 + 4 * bs + 1;
    let t_krylov = std::time::Instant::now();
    let v0 = v.clone();
    let mut seqs: Vec<Vec<Fe>> = Vec::with_capacity(e);
    let mut prog = crate::progress::Progress::start("krylov", e);
    for step in 0..e {
        let mut s = vec![field.zero(); bs * bs];
        for c1 in 0..bs {
            for c2 in 0..bs {
                let mut acc = field.zero();
                for x in 0..n {
                    acc = field.add(acc, field.mul(u[x * bs + c1], v[x * bs + c2]));
                }
                s[c1 * bs + c2] = acc;
            }
        }
        seqs.push(s);
        op.hstep_block(&mut v, bs);
        prog.tick(step + 1);
    }
    let t_seq = t_krylov.elapsed();
    // annihilators via the stacked [G | -I] approximant basis
    let t_solve = std::time::Instant::now();
    let basis = crate::pmbasis::stacked_basis(field, &seqs, bs, e);
    let ann = crate::pmbasis::annihilators_from_basis(field, &basis, &seqs, bs);
    let t_solve_dur = t_solve.elapsed();
    let sol = ann.sol;
    let ell = ann.ell;
    if sol.is_empty() {
        return EngineResult {
            ok: false,
            z_colex: vec![],
            p_lex: vec![],
            stats: format!(
                "block(pmb) krylov {e} steps {t_seq:.1?}, approximant basis ({t_solve_dur:.1?}) yielded no rows"
            ),
        };
    }
    // batched block Horner reconstruction (identical to the dense path)
    let mut rng2 = Xorshift64::new(seed ^ 0xB10C);
    let nbatches = 4;
    let per_batch = bs.min(sol.len());
    let mut attempt = 0usize;
    for batch in 0..nbatches {
        let ncand = per_batch;
        let mut cmat = vec![field.zero(); sol.len() * ncand];
        if batch == 0 {
            for r in 0..ncand.min(sol.len()) {
                cmat[r * ncand + r] = field.one();
            }
        } else {
            for cf in &mut cmat {
                *cf = next(&mut rng2);
            }
        }
        let mut a = vec![field.zero(); bs * ncand * (ell + 1)];
        for r in 0..sol.len() {
            for c in 0..ncand {
                let cr = cmat[r * ncand + c];
                if field.is_zero(cr) {
                    continue;
                }
                for idx in 0..bs * (ell + 1) {
                    let val = field.mul(cr, sol[r][idx]);
                    let dst = (idx / bs) * bs * ncand + c * bs + (idx % bs);
                    a[dst] = field.add(a[dst], val);
                }
            }
        }
        let mut z = vec![field.zero(); n * ncand];
        let mut prog = crate::progress::Progress::start("reconstruct", ell + 1);
        for round in 0..=ell {
            let j = ell - round;
            op.hstep_block(&mut z, ncand);
            for x in 0..n {
                for c in 0..ncand {
                    let mut add = field.zero();
                    for bp in 0..bs {
                        add = field.add(add, field.mul(v0[x * bs + bp], a[j * bs * ncand + c * bs + bp]));
                    }
                    z[x * ncand + c] = field.add(z[x * ncand + c], add);
                }
            }
            prog.tick(round + 1);
        }
        for c in 0..ncand {
            attempt += 1;
            let zc: Vec<Fe> = (0..n).map(|x| z[x * ncand + c]).collect();
            if op.verify_kernel(&zc) {
                let p_lex = op.to_lex_plucker(&zc);
                return EngineResult {
                    ok: true,
                    z_colex: zc,
                    p_lex,
                    stats: format!(
                        "block(pmb) krylov {e} steps {t_seq:.1?}, approximant basis degs {:?} ({t_solve_dur:.1?}), batch-Horner attempt {} -> KERNEL FOUND",
                        ann.degrees,
                        attempt
                    ),
                };
            }
        }
    }
    EngineResult {
        ok: false,
        z_colex: vec![],
        p_lex: vec![],
        stats: format!(
            "block(pmb) krylov {e} steps {t_seq:.1?}, approximant basis degs {:?} ({t_solve_dur:.1?}), all {attempt} batch-Horner candidates failed Ez",
            ann.degrees
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::instance::{loidreau_with_secret, random_instance, Instance};

    #[allow(clippy::too_many_arguments)]
    fn make_op(m: u32, n: usize, k: usize, lam: usize, seed: u64, structured: bool) -> (Operator, Option<Instance>) {
        let field = Field::new(m);
        let circuit = Circuit::new(n, k, lam);
        let mut rng = Xorshift64::new(0x51E9);
        let inst = if structured {
            loop {
                match loidreau_with_secret(&field, n, k, lam, &mut rng) {
                    Some(i) => break i,
                    None => continue,
                }
            }
        } else {
            random_instance(&field, n, k, &mut rng)
        };
        let op = Operator::from_generator(field, circuit, &inst.g_pub, n - k, seed, true);
        (op, Some(inst))
    }

    #[test]
    fn scalar_engine_finds_wedge_toy() {
        // structured toy: kernel must be proportional to the planted wedge
        let (op, inst) = make_op(24, 16, 12, 3, 1, true);
        let res = scalar_wiedemann(&op, 1);
        assert!(res.ok, "scalar engine failed: {}", res.stats);
        let inst = inst.unwrap();
        let wedge = inst.wedge_lex(&op.field, 3).unwrap();
        assert!(op.field.proportional(&res.p_lex, &wedge).is_some(),
            "kernel is not the planted wedge line");
        println!("{}", res.stats);
    }

    #[test]
    fn scalar_engine_random_negative() {
        // random control at the toy shape: full-rank stack, the
        // conditioned M is nonsingular, no kernel
        let (op, _) = make_op(24, 16, 12, 3, 1, false);
        let res = scalar_wiedemann(&op, 1);
        assert!(!res.ok, "random code must not yield a kernel: {}", res.stats);
    }

    #[test]
    fn block_engine_finds_wedge_toy() {
        let (op, inst) = make_op(24, 16, 12, 3, 1, true);
        let res = block_wiedemann(&op, 1, 8);
        assert!(res.ok, "block engine failed: {}", res.stats);
        let inst = inst.unwrap();
        let wedge = inst.wedge_lex(&op.field, 3).unwrap();
        assert!(op.field.proportional(&res.p_lex, &wedge).is_some(),
            "kernel is not the planted wedge line");
        println!("{}", res.stats);
    }

    /// Approximant-basis gates: the annihilator (1) reproduces the
    /// dense nullspace's kernel space on the toy (z proportional at equal
    /// seed), (2) end-to-end wedge, (3) random control stays negative.
    #[test]
    fn block_pmb_finds_wedge_and_matches_dense() {
        let (op, inst) = make_op(24, 16, 12, 3, 1, true);
        let res = block_wiedemann_pmb(&op, 1, 8);
        assert!(res.ok, "pmb engine failed: {}", res.stats);
        let inst = inst.unwrap();
        let wedge = inst.wedge_lex(&op.field, 3).unwrap();
        assert!(op.field.proportional(&res.p_lex, &wedge).is_some(),
            "pmb kernel is not the planted wedge line");
        println!("{}", res.stats);
        // gate (1): same kernel space as the dense annihilator
        let dense = block_wiedemann(&op, 1, 8);
        assert!(dense.ok);
        assert!(op.field.proportional(&res.p_lex, &dense.p_lex).is_some(),
            "pmb kernel not proportional to the dense-path kernel");
    }

    #[test]
    fn block_pmb_random_negative() {
        let (op, _) = make_op(24, 16, 12, 3, 1, false);
        let res = block_wiedemann_pmb(&op, 1, 8);
        assert!(!res.ok, "random code must not yield a kernel: {}", res.stats);
    }

    #[test]
    fn bm_recovers_simple_recurrence() {
        let field = Field::new(16);
        // seq_i = alpha^i with alpha of order 51 in GF(2^16)^*
        let alpha = field.mul((3, 0), (5, 0));
        let mut seq = vec![field.one()];
        for _ in 0..20 {
            seq.push(field.mul(*seq.last().unwrap(), alpha));
        }
        let (l, c) = berlekamp_massey(&field, &seq);
        assert!(l <= 16);
        // verify the recurrence
        for t in l..seq.len() {
            let mut acc = field.zero();
            for i in 0..=l {
                acc = field.add(acc, field.mul(c[i], seq[t - i]));
            }
            assert!(field.is_zero(acc), "recurrence fails at t={t}");
        }
    }
}
