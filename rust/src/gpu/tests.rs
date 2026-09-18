//! GPU test suite.  The inner test modules glob-import this module, so
//! the shared imports (engine, field, operator items) live at the top.

use super::device::GpuEngine;
use super::params::{sub_chunk_size, MAX_SUB_PASSES};
use super::solvers::{block_wiedemann_gpu, block_wiedemann_gpu_pmb};
use crate::field::{Fe, Field};
use crate::operator::Operator;
use crate::rng::Xorshift64;

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::instance::loidreau_with_secret;
    use crate::rng::Xorshift64;

    /// The GPU H-step must be bit-identical to the CPU H-step, and the
    /// GPU dots must equal the CPU dots (XOR associativity).
    #[test]
    fn gpu_hstep_bit_identical() {
        let field = Field::new(24);
        let (n_, k_, lam) = (16usize, 12usize, 3usize);
        let circuit = crate::circuit::Circuit::new(n_, k_, lam);
        let mut rng = Xorshift64::new(0x51E9);
        let inst = loop {
            match loidreau_with_secret(&field, n_, k_, lam, &mut rng) {
                Some(i) => break i,
                None => continue,
            }
        };
        let op = Operator::from_generator(field, circuit, &inst.g_pub, n_ - k_, 1, true);
        let bs = 4usize;
        let mut gpu = match GpuEngine::new(op.clone(), bs) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("gpu unavailable, skipping: {e}");
                return;
            }
        };
        let mask = op.field.mask();
        let mut r = Xorshift64::new(5);
        let mut y: Vec<Fe> = (0..op.n * bs).map(|_| r.next_fe(mask)).collect();
        let u: Vec<Fe> = (0..op.n * bs).map(|_| r.next_fe(mask)).collect();
        gpu.upload_u(&u);
        let y0 = y.clone();
        let dots = gpu.hstep_dots(&mut y);
        // CPU reference
        let mut want = y0.clone();
        op.hstep_block(&mut want, bs);
        let bad: Vec<usize> = (0..y.len()).filter(|&i| y[i] != want[i]).collect();
        eprintln!("hstep mismatches: {}/{}", bad.len(), y.len());
        if !bad.is_empty() {
            for &i in bad.iter().take(5) {
                eprintln!("  [{i}] gpu {:?} vs cpu {:?}", y[i], want[i]);
            }
        }
        assert_eq!(y, want, "GPU H-step differs from CPU");
        for c1 in 0..bs {
            for c2 in 0..bs {
                let mut acc = op.field.zero();
                for x in 0..op.n {
                    acc = op.field.add(acc, op.field.mul(u[x * bs + c1], y0[x * bs + c2]));
                }
                        assert_eq!(dots[c1 * bs + c2], acc, "GPU dot ({c1},{c2}) differs");
            }
        }
        // second step chained (state must carry over correctly)
        let dots2 = gpu.hstep_dots(&mut y);
        let mut cur = want;
        op.hstep_block(&mut cur, bs);
        assert_eq!(y, cur, "GPU chained H-step differs");
        assert_ne!(dots, dots2);
    }

    /// GPU reconstruction gate: device-side reconstruction must be bit-identical to
    /// the CPU block-Horner loop at EQUAL inputs (v0, A_j, ell, ncand) --
    /// including ncand < bs batches (different Z vs V0 strides) -- because
    /// both compute the same XOR/multiply DAG in the same order.
    #[test]
    fn gpu_reconstruct_matches_cpu() {
        let field = Field::new(24);
        let (n_, k_, lam) = (16usize, 12usize, 3usize);
        let circuit = crate::circuit::Circuit::new(n_, k_, lam);
        let mut rng = Xorshift64::new(0x51E9);
        let inst = loop {
            match loidreau_with_secret(&field, n_, k_, lam, &mut rng) {
                Some(i) => break i,
                None => continue,
            }
        };
        let op = Operator::from_generator(field, circuit, &inst.g_pub, n_ - k_, 1, true);
        let bs = 8usize;
        let mut gpu = match GpuEngine::new(op.clone(), bs) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("gpu unavailable, skipping: {e}");
                return;
            }
        };
        let f = &op.field;
        let mask = f.mask();
        let mut r = Xorshift64::new(42);
        // small Horner window (bit-identity does not depend on ell) and
        // both a full batch and a narrow ncand < bs batch
        for (ell, ncand) in [(5usize, 8usize), (7, 5), (0, 3)] {
            let v0: Vec<Fe> = (0..op.n * bs).map(|_| r.next_fe(mask)).collect();
            let a: Vec<Fe> = (0..bs * ncand * (ell + 1)).map(|_| r.next_fe(mask)).collect();
            let n = op.n;
            // CPU reference: the plain block-Horner loop
            let mut zc = vec![f.zero(); n * ncand];
            for j in (0..=ell).rev() {
                op.hstep_block(&mut zc, ncand);
                for x in 0..n {
                    for c in 0..ncand {
                        let mut add = f.zero();
                        for bp in 0..bs {
                            add = f.add(add, f.mul(v0[x * bs + bp], a[j * bs * ncand + c * bs + bp]));
                        }
                        zc[x * ncand + c] = f.add(zc[x * ncand + c], add);
                    }
                }
            }
            let zg = gpu.reconstruct(&v0, &a, ell, ncand);
            let bad = (0..zg.len()).filter(|&i| zg[i] != zc[i]).count();
            eprintln!("reconstruct ell={ell} ncand={ncand}: {bad}/{} mismatches", zg.len());
            assert_eq!(bad, 0, "GPU reconstruction differs (ell={ell} ncand={ncand})");
        }
    }

    /// Submission integrity: chunks larger than
    /// the pass cap split into multiple polled submissions; the op-7
    /// handoff must chain them exactly like single steps.  30 toy steps =
    /// 420 passes > MAX_SUB_PASSES, so this exercises the split path.
    #[test]
    fn gpu_chunk_split_matches_singles() {
        let field = Field::new(24);
        let (n_, k_, lam) = (16usize, 12usize, 3usize);
        let circuit = crate::circuit::Circuit::new(n_, k_, lam);
        let mut rng = Xorshift64::new(0x51E9);
        let inst = loop {
            match loidreau_with_secret(&field, n_, k_, lam, &mut rng) {
                Some(i) => break i,
                None => continue,
            }
        };
        let op = Operator::from_generator(field, circuit, &inst.g_pub, n_ - k_, 1, true);
        let bs = 4usize;
        let mut gpu = match GpuEngine::new(op.clone(), bs) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("gpu unavailable, skipping: {e}");
                return;
            }
        };
        let mask = op.field.mask();
        let mut r = Xorshift64::new(6);
        let v: Vec<Fe> = (0..op.n * bs).map(|_| r.next_fe(mask)).collect();
        let u: Vec<Fe> = (0..op.n * bs).map(|_| r.next_fe(mask)).collect();
        gpu.upload_u(&u);
        let mut y1 = v.clone();
        let mut d1 = Vec::new();
        for _ in 0..30 {
            d1.push(gpu.hstep_dots(&mut y1));
        }
        let mut y2 = v.clone();
        let d2 = gpu.run_chunk(&mut y2, 30);
        assert_eq!(y1, y2, "split-chunk vector differs");
        assert_eq!(d1, d2, "split-chunk dots differ");
    }

    /// The sub-submission cap arithmetic (pure logic).
    #[test]
    fn sub_chunk_cap_arithmetic() {
        assert_eq!(sub_chunk_size(1), MAX_SUB_PASSES);
        assert_eq!(sub_chunk_size(42), MAX_SUB_PASSES / 42);
        assert_eq!(sub_chunk_size(500), 1);
        assert!(sub_chunk_size(42) * 42 <= MAX_SUB_PASSES);
    }

    /// Full GPU block Wiedemann (PM-BASIS annihilator + GPU
    /// reconstruction) on the structured toy: kernel = wedge.
    #[test]
    fn gpu_block_wiedemann_pmb_toy() {
        let field = Field::new(24);
        let (n_, k_, lam) = (16usize, 12usize, 3usize);
        let circuit = crate::circuit::Circuit::new(n_, k_, lam);
        let mut rng = Xorshift64::new(0x51E9);
        let inst = loop {
            match loidreau_with_secret(&field, n_, k_, lam, &mut rng) {
                Some(i) => break i,
                None => continue,
            }
        };
        let wedge = inst.wedge_lex(&field, lam).unwrap();
        let op = Operator::from_generator(field, circuit, &inst.g_pub, n_ - k_, 1, true);
        let res = match block_wiedemann_gpu_pmb(op.clone(), 1, 8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("gpu unavailable, skipping: {e}");
                return;
            }
        };
        assert!(res.ok, "gpu pmb engine failed: {}", res.stats);
        assert!(op.field.proportional(&res.p_lex, &wedge).is_some());
        eprintln!("{}", res.stats);
    }

    /// Full GPU block Wiedemann on the structured toy: kernel = wedge.
    #[test]
    fn gpu_block_wiedemann_toy() {
        let field = Field::new(24);
        let (n_, k_, lam) = (16usize, 12usize, 3usize);
        let circuit = crate::circuit::Circuit::new(n_, k_, lam);
        let mut rng = Xorshift64::new(0x51E9);
        let inst = loop {
            match loidreau_with_secret(&field, n_, k_, lam, &mut rng) {
                Some(i) => break i,
                None => continue,
            }
        };
        let wedge = inst.wedge_lex(&field, lam).unwrap();
        let op = Operator::from_generator(field, circuit, &inst.g_pub, n_ - k_, 1, true);
        let res = match block_wiedemann_gpu(op.clone(), 1, 8) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("gpu unavailable, skipping: {e}");
                return;
            }
        };
        assert!(res.ok, "gpu block engine failed: {}", res.stats);
        assert!(op.field.proportional(&res.p_lex, &wedge).is_some());
        eprintln!("{}", res.stats);
    }
}

#[cfg(test)]
mod wmultest {
    use super::*;

    /// The WGSL wmul via the dots kernel at every limb-boundary degree:
    /// with U and Y zero except at x=0, dots[c1][c2] = U[0][c1] * Y[0][c2]
    /// exactly.  m = 64/65/96/100/128 pin the 4-limb shift/fold (regression:
    /// the mulx carry used to run in the wrong limb direction -- invisible
    /// at m <= 32, wrong for every element spanning two or more limbs).
    #[test]
    fn gpu_wmul_matches_host() {
        for md in [24u32, 40, 64, 65, 96, 100, 128] {
            let field = Field::new(md);
            let (n_, k_, lam) = (10usize, 7usize, 2usize);
            let circuit = crate::circuit::Circuit::new(n_, k_, lam);
            let mut rng = Xorshift64::new(11);
            let inst = loop {
                match crate::instance::loidreau_with_secret(&field, n_, k_, lam, &mut rng) {
                    Some(i) => break i,
                    None => continue,
                }
            };
            let op = Operator::from_generator(field, circuit, &inst.g_pub, n_ - k_, 1, true);
            let bs = 2usize;
            let mut gpu = match GpuEngine::new(op.clone(), bs) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("gpu unavailable, skipping: {e}");
                    return;
                }
            };
            let f = &op.field;
            let mask = f.mask();
            let mut rr = Xorshift64::new(77);
            // random full-width pairs (both words used for m > 64)
            let cases: Vec<(Fe, Fe)> = (0..bs)
                .map(|_| (rr.next_fe(mask), rr.next_fe(mask)))
                .chain(std::iter::once((f.one(), (12345u64 & mask.0, 0))))
                .collect();
            let mut u = vec![f.zero(); op.n * bs];
            let mut y = vec![f.zero(); op.n * bs];
            for (c, (a, b)) in cases.iter().enumerate().take(bs) {
                u[c] = *a;
                y[c] = *b;
            }
            gpu.upload_u(&u);
            // hstep_dots computes dots on the INPUT y (then advances y,
            // which we discard here)
            let dots = gpu.hstep_dots(&mut y);
            for (c, (a, b)) in cases.iter().enumerate().take(bs) {
                let want = f.mul(*a, *b);
                let got = dots[c * bs + c];
                assert_eq!(got, want, "gpu wmul mismatch m={md} case {c}");
            }
            eprintln!("wmul parity ok at m={md}");
        }
    }
}

#[cfg(test)]
mod phaseprobe {
    use super::*;

    #[test]
    fn phases_bit_identical_through_transpose() {
        let field = Field::new(24);
        let (n_, k_, lam) = (16usize, 12usize, 3usize);
        let circuit = crate::circuit::Circuit::new(n_, k_, lam);
        let mut rng = Xorshift64::new(0x51E9);
        let inst = loop {
            match crate::instance::loidreau_with_secret(&field, n_, k_, lam, &mut rng) {
                Some(i) => break i,
                None => continue,
            }
        };
        let op = Operator::from_generator(field, circuit, &inst.g_pub, n_ - k_, 1, true);
        let bs = 4usize;
        let mut gpu = match GpuEngine::new(op.clone(), bs) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("gpu unavailable, skipping: {e}");
                return;
            }
        };
        gpu.debug_through_embed = true;
        let f = &op.field;
        let mask = f.mask();
        let mut r = Xorshift64::new(5);
        let y: Vec<Fe> = (0..op.n * bs).map(|_| r.next_fe(mask)).collect();
        let gpu_fwd = gpu.debug_forward_state(&y);
        // CPU forward + weight + embed per block per column
        let kr = op.circuit.k * op.circuit.r;
        let rr = op.k_coords;
        for i in 0..op.s {
            for c in 0..bs {
                let mut yc: Vec<Fe> = (0..op.n).map(|x| y[x * bs + c]).collect();
                let dib = &op.di[i * kr..(i + 1) * kr];
                for b in (0..op.circuit.r).rev() {
                    let st = &op.circuit.stages[b];
                    for e in 0..st.dst.len() {
                        let cf = dib[st.cidx[e] as usize];
                        let add = f.mul(cf, yc[st.src[e] as usize]);
                        let idx = st.dst[e] as usize;
                        yc[idx] = f.add(yc[idx], add);
                    }
                }
                for x in 0..op.n {
                    let t = if x < rr && op.conditioned {
                        f.mul(yc[x], op.cw[i * rr + x])
                    } else if x < rr {
                        yc[x]
                    } else {
                        f.zero()
                    };
                    yc[x] = t;
                }
                // transpose in place on the embedded buffer
                for b in 0..op.circuit.r {
                    let st = &op.circuit.stages[b];
                    for e in 0..st.dst.len() {
                        let cf = dib[st.cidx[e] as usize];
                        let add = f.mul(cf, yc[st.dst[e] as usize]);
                        let idx = st.src[e] as usize;
                        yc[idx] = f.add(yc[idx], add);
                    }
                }
                for x in 0..op.n {
                    let g = gpu_fwd[(i * op.n + x) * bs + c];
                    assert_eq!(g, yc[x], "post-transpose mismatch block {i} col {c} coord {x}");
                }
            }
        }
        eprintln!("post-transpose state bit-identical");
    }
}

#[cfg(test)]
mod lam2probe {
    use super::*;
    use crate::circuit::Circuit;
    use crate::instance::loidreau_with_secret;
    use crate::rng::Xorshift64;

    /// Bit-identity (H-step, dots, and chunked multi-step chaining) at
    /// shapes the other GPU tests do not cover: lam = 2 and m > 64
    /// (two-word elements exercise the 4-limb shift/fold path).
    #[test]
    fn gpu_hstep_bit_identical_lam2_m128() {
        for (m, n_, k_, lam) in [(24u32, 14usize, 10usize, 2usize), (100, 14, 10, 2), (128, 14, 10, 2)] {
        let field = Field::new(m);
        let circuit = Circuit::new(n_, k_, lam);
        let mut rng = Xorshift64::new(5);
        let inst = loop {
            match loidreau_with_secret(&field, n_, k_, lam, &mut rng) {
                Some(i) => break i,
                None => continue,
            }
        };
        let op = Operator::from_generator(field, circuit, &inst.g_pub, n_ - k_, 1, true);
        let bs = 8usize;
        let mut gpu = match GpuEngine::new(op.clone(), bs) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("gpu unavailable, skipping: {e}");
                return;
            }
        };
        let f = &op.field;
        let mask = f.mask();
        let mut r = Xorshift64::new(5);
        // replicate the engine's stream: u and v interleaved per element
        let mut rng2 = Xorshift64::new_mf(1);
        let mut u = vec![f.zero(); op.n * bs];
        let mut v = vec![f.zero(); op.n * bs];
        for x in 0..op.n {
            for c in 0..bs {
                u[x * bs + c] = rng2.next_fe(mask);
                v[x * bs + c] = rng2.next_fe(mask);
            }
        }
        let _ = &mut r;
        gpu.upload_u(&u);
        let mut y = v.clone();
        let dots = gpu.hstep_dots(&mut y);
        // CPU dots of the input and CPU H-step
        for c1 in 0..bs {
            for c2 in 0..bs {
                let mut acc = f.zero();
                for x in 0..op.n {
                    acc = f.add(acc, f.mul(u[x * bs + c1], v[x * bs + c2]));
                }
                assert_eq!(dots[c1 * bs + c2], acc, "lam2 dot ({c1},{c2}) differs");
            }
        }
        let mut want = v.clone();
        op.hstep_block(&mut want, bs);
        let bad = (0..y.len()).filter(|&i| y[i] != want[i]).count();
        eprintln!("lam2 hstep mismatches: {bad}/{}", y.len());
        assert_eq!(bad, 0, "lam2 GPU H-step differs from CPU");
        // chunked multi-step: run_chunk(y, k) must equal k single steps
        let mut y1 = v.clone();
        let mut y2 = v.clone();
        let mut dots1 = Vec::new();
        for _ in 0..8 {
            dots1.push(gpu.hstep_dots(&mut y1));
        }
        let dots2 = gpu.run_chunk(&mut y2, 8);
        assert_eq!(dots1.len(), dots2.len());
        let dd = dots1.iter().zip(dots2.iter()).filter(|(a, b)| a != b).count();
        eprintln!("chunked dots mismatches: {dd}/{}", dots1.len());
        assert_eq!(dd, 0, "chunked dots differ from single-step dots");
        let yy = (0..y1.len()).filter(|&i| y1[i] != y2[i]).count();
        eprintln!("chunked final-vector mismatches: {yy}/{}", y1.len());
        assert_eq!(yy, 0, "chunked final vector differs");
        }
    }
}

#[cfg(test)]
mod wgslsim {
    // Host-side transcription of the WGSL bit_of / mulx / wmul, to
    // localize shader-logic bugs without GPU round-trips.
    fn bit_of(a: [u32; 4], limb: u32, bit: u32) -> u32 {
        let w = match limb {
            1 => a[1],
            2 => a[2],
            3 => a[3],
            _ => a[0],
        };
        (w >> bit) & 1
    }
    fn mulx(a: [u32; 4], m: u32, g: [u32; 4]) -> [u32; 4] {
        let m1_limb = (m - 1) / 32;
        let m1_bit = (m - 1) % 32;
        let ovf = bit_of(a, m1_limb, m1_bit);
        let mut q = [
            a[0] << 1,
            (a[1] << 1) | (a[0] >> 31),
            (a[2] << 1) | (a[1] >> 31),
            (a[3] << 1) | (a[2] >> 31),
        ];
        if ovf == 1 {
            let m_limb = m / 32;
            let m_bit = m % 32;
            if m_limb <= 3 {
                q[m_limb as usize] ^= 1 << m_bit;
            }
            for i in 0..4 {
                q[i] ^= g[i];
            }
        }
        q
    }
    fn wmul(a: [u32; 4], b: [u32; 4], m: u32, g: [u32; 4]) -> [u32; 4] {
        let mut acc = [0u32; 4];
        let mut y = b;
        for i in 0..m {
            let abit = bit_of(a, i / 32, i % 32);
            let mask = [0u32.wrapping_sub(abit); 4];
            for j in 0..4 {
                acc[j] ^= y[j] & mask[j];
            }
            y = mulx(y, m, g);
        }
        acc
    }

    #[test]
    fn wgsl_mulx_semantics_match_host() {
        use crate::field::Field;
        for &m in &[24u32, 33, 40, 63, 64, 100] {
            let f = Field::new(m);
            let g = Field::to_limbs(f.g);
            let mut r = crate::rng::Xorshift64::new(31 + m as u64);
            let mask = f.mask();
            for _ in 0..200 {
                let a = r.next_fe(mask);
                let sim = mulx(Field::to_limbs(a), m, g);
                let want = f.mul(a, (2, 0)); // a * x
                assert_eq!(
                    Field::from_limbs(sim),
                    want,
                    "mulx mismatch m={m} a={a:?}"
                );
            }
        }
    }

    #[test]
    fn wgsl_wmul_semantics_match_host_field() {
        use crate::field::Field;
        for &m in &[24u32, 40, 63, 64, 65, 96, 100, 128] {
            let f = Field::new(m);
            let g = Field::to_limbs(f.g);
            let mut r = crate::rng::Xorshift64::new(1234 + m as u64);
            let mask = f.mask();
            for _ in 0..50 {
                let a = r.next_fe(mask);
                let b = r.next_fe(mask);
                let sim = wmul(Field::to_limbs(a), Field::to_limbs(b), m, g);
                let want = f.mul(a, b);
                assert_eq!(
                    Field::from_limbs(sim),
                    want,
                    "sim mismatch m={m} a={a:?} b={b:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tsprobe {
    use super::*;

    #[test]
    fn probe_ts_counts() {
        let field = Field::new(24);
        let (n_, k_, lam) = (16usize, 12usize, 3usize);
        let circuit = crate::circuit::Circuit::new(n_, k_, lam);
        let mut rng = Xorshift64::new(0x51E9);
        let inst = loop {
            match crate::instance::loidreau_with_secret(&field, n_, k_, lam, &mut rng) {
                Some(i) => break i,
                None => continue,
            }
        };
        let op = Operator::from_generator(field, circuit, &inst.g_pub, n_ - k_, 1, true);
        let bs = 8usize;
        let mut gpu = match GpuEngine::new(op.clone(), bs) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("gpu unavailable, skipping: {e}");
                return;
            }
        };
        let mask = op.field.mask();
        let mut r = Xorshift64::new(3);
        let mut y: Vec<Fe> = (0..op.n * bs).map(|_| r.next_fe(mask)).collect();
        let _ = gpu.hstep_dots(&mut y);
        eprintln!(
            "probe: r={} nd={} ts.len={} ts={:?}",
            op.circuit.r,
            crate::gpu::ndispatch_pub(2 * op.circuit.r as u32),
            gpu.last_step_ts_us.len(),
            gpu.last_step_ts_us,
        );
        assert_eq!(gpu.last_step_ts_us.len(), crate::gpu::ndispatch_pub(2 * op.circuit.r as u32) as usize);
    }
}
