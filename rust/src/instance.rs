//! Instance generation: the Loidreau parity-check model and random controls.
//!
//! Sampling matches the project's Python fixture generator, so a seed
//! reproduces a key of the same *distribution* -- and, more importantly,
//! so the structured instances carry a machine-checkable ground truth:
//!
//! * sample `b` of rank weight n (F_2-independent coordinates),
//! * sample `gamma` with `gamma_0 = 1`, F_2-independent (the lambda-dim
//!   subspace of the perturbation),
//! * sample the binary component matrices `P_j` of an invertible mask
//!   with entries in `span(gamma)`,
//! * secret rows `b_j = b · P_j` (the rows of `B`),
//! * `H = sum_j gamma_j Moore_{n-k}(b_j)`, `G_pub = nullspace(H)`.
//!
//! The planted kernel of the full stacked compound of `G_pub` at s = n-k is
//! then the Pluecker line of `b_0 ^ ... ^ b_{lam-1}` (Conjecture 1/2 of
//! the research notes; validated against the dense compound in tests and
//! used as ground truth by the end-to-end toy runs).

use crate::field::{Fe, Field};
use crate::linalg::{f2_rank, nullspace};
use crate::rng::Xorshift64;

/// A generated instance: public generator + optional ground truth.
pub struct Instance {
    pub g_pub: Vec<Fe>, // k x n row-major
    #[allow(dead_code)] // documentary shape field
    pub k: usize,
    pub n: usize,
    pub lam: usize,
    /// Structured case: the rows of `B` (lam x n, flat row-major).
    pub b_rows: Option<Vec<Fe>>,
}

impl Instance {
    /// The planted wedge line in standard lex Pluecker order
    /// (row 0 of `C_lam(B)`); None for random instances.
    pub fn wedge_lex(&self, field: &Field, lam: usize) -> Option<Vec<Fe>> {
        let b_rows = self.b_rows.as_ref()?;
        let mut w = crate::dense::compound_matrix(field, b_rows, self.lam, self.n, lam);
        w.truncate(crate::subset::subsets_lex(self.n, lam).len());
        Some(w)
    }
}

/// Random [n, n] Moore-support vector: entries F_2-independent (rank
/// weight n).  Redraws until independent; needs n <= m.
fn random_full_rank_weight_n(field: &Field, n: usize, rng: &mut Xorshift64) -> Vec<Fe> {
    assert!(n as u32 <= field.m, "rank weight n requires n <= m");
    loop {
        let mask = field.mask();
        let a: Vec<Fe> = (0..n).map(|_| rng.next_fe(mask)).collect();
        if f2_rank(&a, field.m) == n {
            return a;
        }
    }
}

/// Moore matrix with `rows` rows over support `a`: row i = a^{[i]}.
fn moore_matrix(field: &Field, a: &[Fe], rows: usize) -> Vec<Fe> {
    let n = a.len();
    let mut m = vec![field.zero(); rows * n];
    for i in 0..rows {
        for j in 0..n {
            m[i * n + j] = field.frob(a[j], i as i32);
        }
    }
    m
}

/// Invertible n x n mask with entries in span(gamma), as binary component
/// matrices `comps` (lam x n x n, entries 0/1) with
/// `P = sum_j gamma_j * comps[j]`.  Retried until P is invertible and its
/// entries F_2-span exactly gamma's dimension.
fn random_mask_components(
    field: &Field,
    n: usize,
    gamma: &[Fe],
    rng: &mut Xorshift64,
) -> Vec<Vec<u8>> {
    let lam = gamma.len();
    loop {
        // random 0/1 component tensors (pj[j] = P_j)
        let pj: Vec<Vec<u8>> = (0..lam)
            .map(|_| (0..n * n).map(|_| (rng.next_u64() & 1) as u8).collect())
            .collect();
        // entries of P and their F_2 span
        let mut entries: Vec<Fe> = Vec::with_capacity(n * n);
        for idx in 0..n * n {
            let mut e = field.zero();
            for (j, &gj) in gamma.iter().enumerate() {
                if pj[j][idx] == 1 {
                    e = field.add(e, gj);
                }
            }
            entries.push(e);
        }
        if f2_rank(&entries, field.m) != lam {
            continue;
        }
        // invertibility: nullspace of P must be trivial
        let p: Vec<Fe> = entries;
        let ns = nullspace(field, &p, n, n);
        if ns.is_empty() {
            return pj;
        }
    }
}

/// Structured (Loidreau parity-check model) instance.  Returns None when
/// the sampled secret rows are not F_2-independent (resample outside).
pub fn loidreau_with_secret(
    field: &Field,
    n: usize,
    k: usize,
    lam: usize,
    rng: &mut Xorshift64,
) -> Option<Instance> {
    let mask = field.mask();
    let b = random_full_rank_weight_n(field, n, rng);
    let gamma: Vec<Fe> = loop {
        // gamma_0 = 1, remaining entries random, F_2-independent overall
        let mut gam = vec![field.one()];
        for _ in 1..lam {
            gam.push(rng.next_fe(mask));
        }
        if f2_rank(&gam, field.m) == lam {
            break gam;
        }
    };
    let pj = random_mask_components(field, n, &gamma, rng);
    let mut b_rows = vec![field.zero(); lam * n];
    for j in 0..lam {
        for l in 0..n {
            // (b · P_j)[l] = sum_i b[i] * P_j[i][l]
            let mut acc = field.zero();
            for i in 0..n {
                if pj[j][i * n + l] == 1 {
                    acc = field.add(acc, b[i]);
                }
            }
            b_rows[j * n + l] = acc;
        }
    }
    // the lam secret rows must be linearly independent over GF(2^m)
    // (field rank, NOT F_2 rank): rank(b_rows) == lam iff ker(b_rows^T)
    // is trivial.
    let mut b_rows_t = vec![field.zero(); n * lam];
    for i in 0..lam {
        for j in 0..n {
            b_rows_t[j * lam + i] = b_rows[i * n + j];
        }
    }
    if !nullspace(field, &b_rows_t, n, lam).is_empty() {
        return None;
    }
    let rows = n - k;
    let mut h = vec![field.zero(); rows * n];
    for j in 0..lam {
        let moore = moore_matrix(field, &b_rows[j * n..(j + 1) * n], rows);
        for t in 0..rows * n {
            h[t] = field.add(h[t], field.mul(gamma[j], moore[t]));
        }
    }
    // G_pub = nullspace(H): a k x n basis (rows)
    let basis = nullspace(field, &h, rows, n);
    assert_eq!(basis.len(), k, "parity-check rank defect");
    let mut g_pub = vec![field.zero(); k * n];
    for (i, row) in basis.iter().enumerate() {
        g_pub[i * n..(i + 1) * n].copy_from_slice(row);
    }
    Some(Instance { g_pub, k, n, lam, b_rows: Some(b_rows) })
}

/// Random full-rank k x n control code (no planted structure).
pub fn random_instance(field: &Field, n: usize, k: usize, rng: &mut Xorshift64) -> Instance {
    let mask = field.mask();
    loop {
        let g: Vec<Fe> = (0..k * n).map(|_| rng.next_fe(mask)).collect();
        // full row rank iff nullspace of the k x n matrix... cheap check:
        // rank k. Use nullspace of the transpose-free check via systematic
        // form (panics inside if rank-deficient, so guard with a trial).
        if let Ok((_, _)) = try_systematic(field, &g, k, n) {
            return Instance { g_pub: g, k, n, lam: 0, b_rows: None };
        }
    }
}

fn try_systematic(field: &Field, a: &[Fe], k: usize, n: usize) -> Result<(Vec<usize>, Vec<Fe>), ()> {
    let mut t = a.to_vec();
    let mut piv = Vec::new();
    let mut row = 0usize;
    for col in 0..n {
        if row == k {
            break;
        }
        let mut p = None;
        for i in row..k {
            if !field.is_zero(t[i * n + col]) {
                p = Some(i);
                break;
            }
        }
        let Some(p) = p else { continue };
        if p != row {
            for c in 0..n {
                t.swap(row * n + c, p * n + c);
            }
        }
        let iv = field.inv(t[row * n + col]);
        for c in 0..n {
            t[row * n + c] = field.mul(t[row * n + c], iv);
        }
        for i in 0..k {
            if i != row && !field.is_zero(t[i * n + col]) {
                let fac = t[i * n + col];
                for c in 0..n {
                    t[i * n + c] = field.add(t[i * n + c], field.mul(fac, t[row * n + c]));
                }
            }
        }
        piv.push(col);
        row += 1;
    }
    if row < k {
        return Err(());
    }
    Ok((piv, t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wedge_is_planted_kernel() {
        // The load-bearing ground-truth property: at s = n-k the dense
        // stacked compound of G_pub annihilates the secret wedge line.
        let field = Field::new(16);
        let mut rng = Xorshift64::new(0x51E9);
        let (n, k, lam) = (12usize, 9usize, 3usize);
        let inst = loop {
            match loidreau_with_secret(&field, n, k, lam, &mut rng) {
                Some(i) => break i,
                None => continue,
            }
        };
        let s = n - k;
        let w = crate::dense::stacked_compound(&field, &inst.g_pub, k, n, lam, s);
        let wedge = inst.wedge_lex(&field, lam).unwrap();
        let n_cols = crate::subset::subsets_lex(n, lam).len();
        let rows = crate::subset::subsets_lex(k, lam).len() * s;
        for r in 0..rows {
            let mut acc = field.zero();
            for c in 0..n_cols {
                acc = field.add(acc, field.mul(w[r * n_cols + c], wedge[c]));
            }
            assert!(field.is_zero(acc), "row {r} not annihilated");
        }
    }
}
