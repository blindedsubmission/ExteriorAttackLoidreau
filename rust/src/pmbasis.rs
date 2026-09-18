//! Block annihilators via a stacked approximant basis (iterative
//! PM-BASIS over GF(2^m)).
//!
//! # Construction (Giorgi-Jeannerod-Villard; block Wiedemann)
//!
//! The block engine needs right annihilators of the B x B sequence
//! `S_t` = U^T M^t V: vectors a = (`a_0..a_D`) with `sum_j S_{t+j} a_j = 0`
//! for t >= 1.  Those are the columns of the minimal generating matrix
//! polynomial, obtained as follows: form the STACKED series
//!
//!     M(x) = [ G(x)  -I_B ],   G(x) = sum_t S_t x^t,
//!
//! and compute a right order-d approximant basis W (2B columns w with
//! `M w = 0 mod x^d`).  The -`I_B` block absorbs the head of G*p for the
//! s-part of each column (w = (p, s) with s = G p mod x^d plus a free
//! x^d tail), which is what keeps the iteration non-degenerate at every
//! order -- a bare series with invertible `S_0` would collapse the module
//! to x^d-shifts.  For a column of degree D whose p-part is `p_0..p_D`,
//! the approximant relations `sum_i G_{k-i} p_i = 0` hold for all
//! k >= D, i.e. with `a_j` := p_{D-j}, `sum_j a_j S_{t+j} = 0` for
//! t = k-D >= 0: the engine's system (a fortiori the t >= 1 window the
//! dense engine uses; the reconstruction argument in engine.rs applies).
//!
//! Generic columns carry degrees ~d/2 (the minimal generator degree
//! ~N/B); kernel directions of a singular M appear as columns whose
//! degree freezes far below that.  The iterative basis is order-by-order
//! Gaussian elimination with min-degree pivots: at each order d the
//! B residual coordinates each pick one pivot column, eliminate the
//! others, and the pivots are x-shifted.  The I-backed columns have
//! nonvanishing residuals at every order, so they keep pivoting (degree
//! ~d) and act as the tail carriers, leaving the low-degree columns to
//! the generator.  Schoolbook O(B^2 d^2) -- fine for B <= 32, d ~ 1e4.

use crate::field::{Fe, Field};

/// Candidate annihilator rows in the ENGINE's flat layout:
/// `sol[r][j*bs + b] = a_j[b]`, padded to the common Horner depth `ell`.
#[derive(Clone, Debug)]
pub struct Annihilators {
    pub sol: Vec<Vec<Fe>>,
    pub degrees: Vec<usize>,
    pub ell: usize,
}

fn poly_deg(p: &[Fe], field: &Field) -> usize {
    let mut d = 0usize;
    for (i, c) in p.iter().enumerate() {
        if !field.is_zero(*c) {
            d = i;
        }
    }
    d
}

/// dst -= alpha * src, coefficients from `from` upward, capped at `cap`
/// (`from` > 0 exploits the invariant that residual coefficients below
/// the current order are zero; the cap drops coefficients the fit never
/// reads; `mul_u64` is the branch-free m <= 64 fast path).
fn col_axpy(field: &Field, dst: &mut [Vec<Fe>], src: &[Vec<Fe>], alpha: Fe, from: usize, cap: usize) {
    for b in 0..dst.len() {
        let sb = &src[b];
        let db = &mut dst[b];
        if sb.len() > db.len() {
            db.resize(sb.len().min(cap), field.zero());
        }
        for k in from..sb.len().min(db.len()) {
            if !field.is_zero(sb[k]) {
                db[k] = field.add(db[k], field.mul_u64(alpha, sb[k]));
            }
        }
    }
}

fn col_scale(field: &Field, col: &mut [Vec<Fe>], alpha: Fe) {
    for p in col.iter_mut() {
        for c in p.iter_mut() {
            if !field.is_zero(*c) {
                *c = field.mul_u64(*c, alpha);
            }
        }
    }
}

/// The 2B basis columns after the iteration: `cols[i][row]` is the
/// polynomial of entry `row` (rows 0..B-1 = the p-part over G's
/// columns, rows B..2B-1 = the s-part over the -I block).
pub struct StackedBasis {
    pub cols: Vec<Vec<Vec<Fe>>>,
    pub degrees: Vec<usize>,
}

/// Right order-`orders` approximant basis of [G | -I]: every column w
/// satisfies G*p - s == 0 mod x^orders, where p/s are w's two halves.
pub fn stacked_basis(field: &Field, seqs: &[Vec<Fe>], bs: usize, orders: usize) -> StackedBasis {
    let nb = 2 * bs;
    let e = seqs.len().min(orders);
    let zero = field.zero();
    /// One basis column: the output polynomial entries (2B: p-part over
    /// G's columns, s-part over the -I block), its residual series
    /// (capped at the fit order), and its degree.
    struct Column {
        w: Vec<Vec<Fe>>,
        rho: Vec<Vec<Fe>>,
        deg: usize,
    }
    let mut cols: Vec<Column> = (0..nb)
        .map(|i| Column {
            w: (0..nb).map(|j| if i == j { vec![field.one()] } else { vec![] }).collect(),
            // residual rho[r] = coeffs of (M w)[r]; the -I columns start
            // with -e_r at order 0 (char 2: -1 == 1)
            rho: if i < bs {
                (0..bs)
                    .map(|r| seqs.iter().take(e).map(|s| s[r * bs + i]).collect())
                    .collect()
            } else {
                (0..bs)
                    .map(|r| if r == i - bs { vec![field.one()] } else { vec![] })
                    .collect()
            },
            deg: 0,
        })
        .collect();
    let mut prog = crate::progress::Progress::start("approximant basis", e);
    for d in 0..e {
        let pi = |c: &Column, r: usize| -> Fe {
            c.rho[r].get(d).copied().unwrap_or(zero)
        };
        let mut marked = vec![false; nb];
        for r in 0..bs {
            // min-degree pivot among unmarked columns with nonzero residual
            let mut best: Option<usize> = None;
            for (i, c) in cols.iter().enumerate() {
                if !marked[i] && !field.is_zero(pi(c, r)) {
                    match best {
                        None => best = Some(i),
                        Some(b) if c.deg < cols[b].deg => best = Some(i),
                        _ => {}
                    }
                }
            }
            let Some(pivot) = best else { continue };
            marked[pivot] = true;
            // normalize the pivot (one inversion per pivot, not per target)
            let invp = field.inv(pi(&cols[pivot], r));
            col_scale(field, &mut cols[pivot].w, invp);
            col_scale(field, &mut cols[pivot].rho, invp);
            // eliminate every other column with a nonzero residual in row r.
            // split_at_mut borrows the pivot immutably while every target
            // mutates only its own column (rayon over the targets was
            // tried and measured SLOWER: ~59k small dispatches plus 8
            // threads streaming the shared pivot clone through L3).
            let (lo, mid) = cols.split_at_mut(pivot);
            let (pcol, hi) = mid.split_first_mut().expect("pivot in range");
            let (w_pivot, rho_pivot) = (&pcol.w, &pcol.rho);
            let elim = |c: &mut Column| {
                let alpha = pi(c, r); // pivot coefficient is now 1
                if field.is_zero(alpha) {
                    return;
                }
                col_axpy(field, &mut c.w, w_pivot, alpha, 0, usize::MAX);
                col_axpy(field, &mut c.rho, rho_pivot, alpha, d, e);
            };
            for c in lo.iter_mut() {
                elim(c);
            }
            for c in hi.iter_mut() {
                elim(c);
            }
        }
        // x-shift the pivot columns; their x^d entry moves to x^{d+1}
        // (residuals capped at the fit order: always insert then truncate
        // -- skipping the insert would break the M*w = rho correspondence)
        for (i, m) in marked.iter().enumerate() {
            if *m {
                for poly in &mut cols[i].w {
                    poly.insert(0, zero);
                }
                for poly in &mut cols[i].rho {
                    poly.insert(0, zero);
                    poly.truncate(e);
                }
                cols[i].deg += 1;
            }
        }
        // keep degrees honest (elimination can grow a column past its pivot)
        for c in &mut cols {
            let dmax = c.w.iter().map(|q| poly_deg(q, field)).max().unwrap_or(0);
            c.deg = c.deg.max(dmax);
        }
        prog.tick(d + 1);
    }
    let degrees = cols.iter().map(|c| c.deg).collect();
    StackedBasis {
        cols: cols.into_iter().map(|c| c.w).collect(),
        degrees,
    }
}

/// Extract engine annihilator rows from a stacked basis: the `bs`
/// lowest-degree columns' p-parts, degree-reversed into the engine's
/// flat `a_j` layout and padded to a common Horner depth.  Rows are
/// filtered by a direct correlation check on a small window of the
/// engine equations (cheap sanity; the raw Ez gate remains the only
/// certificate).  If every row fails the window check they are kept
/// unfiltered -- a wasted Horner attempt costs time, a dropped carrier
/// would cost the kernel.
pub fn annihilators_from_basis(
    field: &Field,
    basis: &StackedBasis,
    seqs: &[Vec<Fe>],
    bs: usize,
) -> Annihilators {
    let mut idx: Vec<usize> = (0..basis.cols.len()).collect();
    idx.sort_by_key(|&i| basis.degrees[i]);
    let chosen: Vec<usize> = idx.into_iter().take(bs).collect();
    let ell = chosen.iter().map(|&i| basis.degrees[i]).max().unwrap_or(0);
    let build = |i: usize| -> Vec<Fe> {
        let d = basis.degrees[i];
        let mut row = vec![field.zero(); bs * (ell + 1)];
        for j in 0..=d {
            for b in 0..bs {
                // a_j[b] = p_{D-j}[b]
                row[j * bs + b] = basis.cols[i][b].get(d - j).copied().unwrap_or_else(|| field.zero());
            }
        }
        row
    };
    let rows: Vec<Vec<Fe>> = chosen.iter().map(|&i| build(i)).collect();
    // correlation window check (a head and a tail slice of the equations)
    let e = seqs.len();
    let win = 16usize.min(e.saturating_sub(2));
    let keep: Vec<bool> = rows
        .iter()
        .map(|row| {
            annihilates(field, row, bs, ell, seqs, 1, 1 + win)
                && annihilates(field, row, bs, ell, seqs, e.saturating_sub(win + 2), e.saturating_sub(1))
        })
        .collect();
    let use_filter = keep.iter().any(|&k| k);
    let mut sol = Vec::new();
    let mut degrees = Vec::new();
    for (r, k) in rows.into_iter().zip(keep.iter().copied()) {
        if !use_filter || k {
            sol.push(r);
        }
    }
    for (i, k) in chosen.iter().copied().zip(keep.iter().copied()) {
        if !use_filter || k {
            degrees.push(basis.degrees[i]);
        }
    }
    Annihilators { sol, degrees, ell }
}

/// Direct check of the engine equation for a candidate row:
/// `(sum_j S_{t+j} a_j)[b] == 0` for every b and every t in [t0, t1)
/// (data-clamped: terms beyond the collected sequence are skipped).
pub fn annihilates(field: &Field, sol: &[Fe], bs: usize, ell: usize, seqs: &[Vec<Fe>], t0: usize, t1: usize) -> bool {
    let e = seqs.len();
    for t in t0..t1.min(e.saturating_sub(1)) {
        for b in 0..bs {
            let mut acc = field.zero();
            for j in 0..=ell {
                let tj = t + j;
                if tj >= e {
                    break;
                }
                let s = &seqs[tj];
                for bp in 0..bs {
                    let a = sol[j * bs + bp];
                    if field.is_zero(a) {
                        continue;
                    }
                    acc = field.add(acc, field.mul(a, s[b * bs + bp]));
                }
            }
            if !field.is_zero(acc) {
                return false;
            }
        }
    }
    true
}


#[cfg(test)]
mod timing {
    use super::*;
    use crate::rng::Xorshift64;
    /// Ignored timing probe: stacked_basis cost at engine-scale windows.
    #[test]
    #[ignore]
    fn basis_scale() {
        let f = Field::new(50);
        let mut r = Xorshift64::new(0xC0FFEE);
        let mask = f.mask();
        for &(label, e) in &[("toy40-e", 1893usize), ("t80m-e", 7383)] {
            let seqs: Vec<Vec<Fe>> = (0..e).map(|_| (0..64).map(|_| r.next_fe(mask)).collect()).collect();
            let t0 = std::time::Instant::now();
            let b = stacked_basis(&f, &seqs, 8, e);
            eprintln!("{label}: stacked_basis {:.2?} (degs {:?})", t0.elapsed(), b.degrees);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::berlekamp_massey;
    use crate::rng::Xorshift64;

    fn rnd_fe(f: &Field, r: &mut Xorshift64) -> Fe {
        let mask = f.mask();
        let mut x = r.next_fe(mask);
        if f.is_zero(x) {
            x = f.one();
        }
        x
    }

    /// Definition: every basis column satisfies G*p - s == 0 mod x^orders.
    #[test]
    fn stacked_basis_satisfies_definition() {
        let f = Field::new(24);
        let mut r = Xorshift64::new(0x1234);
        for bs in [1usize, 2, 3] {
            let n = 16usize;
            let seqs: Vec<Vec<Fe>> = (0..n)
                .map(|_| (0..bs * bs).map(|_| rnd_fe(&f, &mut r)).collect())
                .collect();
            let basis = stacked_basis(&f, &seqs, bs, n);
            for col in 0..2 * bs {
                let w = &basis.cols[col];
                for k in 0..n {
                    for row in 0..bs {
                        // (G p - s)[row] at order k
                        let mut acc = f.zero();
                        for i in 0..bs {
                            // p-part entry i has coeffs p_c; G's column i
                            let pc = w[i].get(k).copied().unwrap_or(f.zero());
                            if !f.is_zero(pc) {
                                continue; // k-th coeff of G*entry uses G at k... full conv:
                            }
                        }
                        // full convolution: sum_c p_c * G_{k-c}[row][i]
                        for i in 0..bs {
                            for (c, pc) in w[i].iter().enumerate() {
                                if f.is_zero(*pc) || c > k {
                                    continue;
                                }
                                acc = f.add(acc, f.mul(*pc, seqs[k - c][row * bs + i]));
                            }
                        }
                        let sc = w[bs + row].get(k).copied().unwrap_or(f.zero());
                        // G p - s (s-part is rows bs..; -I means subtract s)
                        let got = f.add(acc, sc); // char 2: subtract == add
                        assert!(f.is_zero(got), "column {col} row {row} order {k} (bs={bs})");
                    }
                }
            }
        }
    }

    /// B = 1 anchor: the lowest-degree column's p-part, reversed, is the
    /// Berlekamp-Massey polynomial of the sequence, up to a scalar.
    #[test]
    fn stacked_basis_matches_berlekamp_massey() {
        let f = Field::new(24);
        let mut r = Xorshift64::new(0xBEEF);
        for trial in 0..4usize {
            let l = 3 + trial;
            let mut state: Vec<Fe> = (0..l).map(|_| rnd_fe(&f, &mut r)).collect();
            let taps: Vec<Fe> = (0..l).map(|_| rnd_fe(&f, &mut r)).collect();
            let n = 40;
            let mut seq = Vec::with_capacity(n);
            for _ in 0..n {
                seq.push(state[0]);
                let mut nxt = f.zero();
                for (i, t) in taps.iter().enumerate() {
                    nxt = f.add(nxt, f.mul(*t, state[i]));
                }
                state.remove(0);
                state.push(nxt);
            }
            let (l_bm, c) = berlekamp_massey(&f, &seq);
            assert_eq!(l_bm, l, "BM degree");
            let seqs: Vec<Vec<Fe>> = seq.iter().map(|&s| vec![s]).collect();
            let basis = stacked_basis(&f, &seqs, 1, n);
            let mut best = 0usize;
            for col in 0..basis.cols.len() {
                if basis.degrees[col] < basis.degrees[best] {
                    best = col;
                }
            }
            assert_eq!(basis.degrees[best], l_bm, "generator column degree vs BM");
            // p-part entry 0 equals BM's C(x) itself up to a scalar (the
            // engine vector is its reversal: a_j = p_{D-j} = C_{L-j})
            let lead = basis.cols[best][0].get(l_bm).copied().unwrap();
            assert!(!f.is_zero(lead));
            let scale = f.mul(lead, f.inv(c[l_bm]));
            for j in 0..=l_bm {
                let pj = basis.cols[best][0].get(j).copied().unwrap_or(f.zero());
                let a = f.mul(pj, f.inv(scale));
                assert_eq!(a, c[j], "p-coefficient {j} mismatch vs BM");
            }
        }
    }
}
