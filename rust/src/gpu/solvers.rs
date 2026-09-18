//! Block-Wiedemann drivers: Krylov (and reconstruction) on the GPU, the
//! annihilator on the host (dense nullspace or the stacked approximant
//! basis), verification via the raw `W p == 0` gate.

use super::device::GpuEngine;
use super::params::MAX_CHUNK;
use super::GpuError;
use crate::field::{Fe, Field};
use crate::operator::Operator;
use crate::rng::Xorshift64;


/// Fill the U (projections) and V (starts) blocks from the mf-style
/// interleaved stream; returns (u, v).
fn random_blocks(field: &Field, n: usize, bs: usize, rng: &mut Xorshift64) -> (Vec<Fe>, Vec<Fe>) {
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
            u[x * bs + c] = next(rng);
            v[x * bs + c] = next(rng);
        }
    }
    (u, v)
}

/// Combine annihilator rows with a coefficient matrix into the per-order
/// `A_j` blocks: layout a[j*bs*ncand + c*bs + bp].
fn combine_coefficients(
    field: &Field,
    sol: &[Vec<Fe>],
    cmat: &[Fe],
    bs: usize,
    ncand: usize,
    ell: usize,
) -> Vec<Fe> {
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
    a
}

/// Block Wiedemann with the Krylov iterations (and dots) on the GPU; the
/// annihilator solve, batched Horner reconstruction and verification on
/// the CPU (via the operator).  Mirrors `engine::block_wiedemann`.
#[allow(clippy::needless_pass_by_value)] // owned-operator API, symmetric with the CPU engine
pub fn block_wiedemann_gpu(op: Operator, seed: u64, bs: usize) -> Result<crate::engine::EngineResult, GpuError> {
    use crate::engine::EngineResult;
        let field = op.field.clone();
    let n = op.n;
    let mut gpu = GpuEngine::new(op.clone(), bs)?;
    let mut rng = Xorshift64::new_mf(seed);
    let mask = field.mask();
    let next = |rng: &mut Xorshift64| -> Fe {
        if field.m <= 64 {
            (rng.next_u64() & mask.0, 0)
        } else {
            rng.next_fe(mask)
        }
    };
    let (u, mut v) = random_blocks(&field, n, bs, &mut rng);
    gpu.upload_u(&u);
    let c0 = n.div_ceil(bs);
    let ell = c0 + 2 * bs;
    let e = 3 * c0 + 4 * bs + 1;
    let t_krylov = std::time::Instant::now();
    let mut seqs: Vec<Vec<Fe>> = Vec::with_capacity(e);
    let v0 = v.clone();
    // chunked submissions: device-resident state between the steps of a
    // chunk, one readback per chunk (MAX_CHUNK steps per submit)
    let mut done = 0;
    let mut prog = crate::progress::Progress::start("krylov", e);
    while done < e {
        let c = (e - done).min(MAX_CHUNK);
        seqs.extend(gpu.run_chunk(&mut v, c));
        done += c;
        prog.tick(done);
    }
    let t_seq = t_krylov.elapsed();
    // annihilator system (identical to the CPU path)
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
    let sol = crate::linalg::nullspace(&field, &sys, eqs, unknowns);
    let t_solve_dur = t_solve.elapsed();
    if sol.is_empty() {
        return Ok(EngineResult {
            ok: false,
            z_colex: vec![],
            p_lex: vec![],
            stats: format!(
                "gpu block krylov {e} steps {t_seq:.1?}, annihilator {eqs}x{unknowns} trivial ({t_solve_dur:.1?})"
            ),
        });
    }
    // batched block Horner reconstruction on the GPU: the
    // coefficient-combination matrices are built on the host, the
    // ell+1 H-step rounds and accumulates run device-resident
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
        let a = combine_coefficients(&field, &sol, &cmat, bs, ncand, ell);
        // block Horner on the GPU: Z (n x ncand interleaved) =
        // sum_j M^j (V0 A_j), device-resident, one readback per batch;
        // v0 is the PRISTINE start block (the Krylov mutated v in place)
        let z = gpu.reconstruct(&v0, &a, ell, ncand);
        for c in 0..ncand {
            attempt += 1;
            let zc: Vec<Fe> = (0..n).map(|x| z[x * ncand + c]).collect();
            if op.verify_kernel(&zc) {
                let p_lex = op.to_lex_plucker(&zc);
                return Ok(EngineResult {
                    ok: true,
                    z_colex: zc,
                    p_lex,
                    stats: format!(
                        "gpu block krylov {e} steps {t_seq:.1?}, annihilator {eqs}x{unknowns} nullity {} ({t_solve_dur:.1?}), batch-Horner attempt {} -> KERNEL FOUND",
                        sol.len(),
                        attempt
                    ),
                });
            }
        }
    }
    Ok(EngineResult {
        ok: false,
        z_colex: vec![],
        p_lex: vec![],
        stats: format!(
            "gpu block krylov {e} steps {t_seq:.1?}, annihilator {eqs}x{unknowns} nullity {} ({t_solve_dur:.1?}), all {attempt} batch-Horner candidates failed Ez",
            sol.len()
        ),
    })
}

/// Block Wiedemann with the GPU Krylov (and GPU reconstruction) plus the
/// stacked-approximant-basis annihilator; mirrors
/// `block_wiedemann_gpu` with the dense nullspace replaced.
#[allow(clippy::needless_pass_by_value)] // owned-operator API, symmetric with the CPU engine
pub fn block_wiedemann_gpu_pmb(op: Operator, seed: u64, bs: usize) -> Result<crate::engine::EngineResult, GpuError> {
    use crate::engine::EngineResult;
        let field = op.field.clone();
    let n = op.n;
    let mut gpu = GpuEngine::new(op.clone(), bs)?;
    let mut rng = Xorshift64::new_mf(seed);
    let mask = field.mask();
    let next = |rng: &mut Xorshift64| -> Fe {
        if field.m <= 64 {
            (rng.next_u64() & mask.0, 0)
        } else {
            rng.next_fe(mask)
        }
    };
    let (u, mut v) = random_blocks(&field, n, bs, &mut rng);
    gpu.upload_u(&u);
    let c0 = n.div_ceil(bs);
    let e = 3 * c0 + 4 * bs + 1;
    let t_krylov = std::time::Instant::now();
    let mut seqs: Vec<Vec<Fe>> = Vec::with_capacity(e);
    let v0 = v.clone();
    let mut done = 0;
    let mut prog = crate::progress::Progress::start("krylov", e);
    while done < e {
        let c = (e - done).min(MAX_CHUNK);
        seqs.extend(gpu.run_chunk(&mut v, c));
        done += c;
        prog.tick(done);
    }
    let t_seq = t_krylov.elapsed();
    // annihilators via the stacked [G | -I] approximant basis
    let t_solve = std::time::Instant::now();
    let basis = crate::pmbasis::stacked_basis(&field, &seqs, bs, e);
    let ann = crate::pmbasis::annihilators_from_basis(&field, &basis, &seqs, bs);
    let t_solve_dur = t_solve.elapsed();
    let sol = ann.sol;
    let ell = ann.ell;
    if sol.is_empty() {
        return Ok(EngineResult {
            ok: false,
            z_colex: vec![],
            p_lex: vec![],
            stats: format!(
                "gpu block(pmb) krylov {e} steps {t_seq:.1?}, approximant basis ({t_solve_dur:.1?}) yielded no rows"
            ),
        });
    }
    // batched block Horner on the GPU
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
        let a = combine_coefficients(&field, &sol, &cmat, bs, ncand, ell);
        let z = gpu.reconstruct(&v0, &a, ell, ncand);
        for c in 0..ncand {
            attempt += 1;
            let zc: Vec<Fe> = (0..n).map(|x| z[x * ncand + c]).collect();
            if op.verify_kernel(&zc) {
                let p_lex = op.to_lex_plucker(&zc);
                return Ok(EngineResult {
                    ok: true,
                    z_colex: zc,
                    p_lex,
                    stats: format!(
                        "gpu block(pmb) krylov {e} steps {t_seq:.1?}, approximant basis degs {:?} ({t_solve_dur:.1?}), batch-Horner attempt {} -> KERNEL FOUND",
                        ann.degrees,
                        attempt
                    ),
                });
            }
        }
    }
    Ok(EngineResult {
        ok: false,
        z_colex: vec![],
        p_lex: vec![],
        stats: format!(
            "gpu block(pmb) krylov {e} steps {t_seq:.1?}, approximant basis degs {:?} ({t_solve_dur:.1?}), all {attempt} batch-Horner candidates failed Ez",
            ann.degrees
        ),
    })
}

