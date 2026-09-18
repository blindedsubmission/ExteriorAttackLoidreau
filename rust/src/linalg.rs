//! Dense linear algebra over GF(2^m): pivoting to systematic form,
//! nullspace, and square-system solve.  Only used at setup time and on
//! toy shapes; everything here is straightforward Gaussian elimination.

use crate::field::{Fe, Field};

/// Column-pivoted reduction of a full-row-rank k x n matrix to reduced
/// systematic form.  Returns (cols, D) where cols = piv + rest, the first
/// k columns of A[:, cols] are invertible, and
/// A[:, cols[:k]]^{-1} A[:, cols[k:]] = D  (k x r row-major).
///
/// This mirrors `systematic_form` in the Python reference: after full
/// reduction (normalize pivot + eliminate in all other rows), row `row`
/// of the reduced matrix has its pivot 1 at pivcols[row] and equals the
/// corresponding row of A_piv^{-1} A -- so D is read off the rest
/// columns directly.
pub fn systematic_form(field: &Field, a: &[Fe], k: usize, n: usize) -> (Vec<usize>, Vec<Fe>) {
    let mut t = a.to_vec();
    let mut piv = Vec::with_capacity(k);
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
    assert!(row == k, "input must have full row rank");
    let mut is_piv = vec![false; n];
    for &c in &piv {
        is_piv[c] = true;
    }
    let rest: Vec<usize> = (0..n).filter(|&c| !is_piv[c]).collect();
    let r = rest.len();
    let mut d = vec![field.zero(); k * r];
    for (row, rv) in d.chunks_exact_mut(r).enumerate() {
        for (b, &c) in rest.iter().enumerate() {
            rv[b] = t[row * n + c];
        }
    }
    let mut cols = piv.clone();
    cols.extend_from_slice(&rest);
    (cols, d)
}

/// Nullspace basis (as rows) of a k x n matrix: rows x with A x^T = 0.
pub fn nullspace(field: &Field, a: &[Fe], k: usize, n: usize) -> Vec<Vec<Fe>> {
    let mut t = a.to_vec();
    let mut row = 0usize;
    let mut pivcols: Vec<usize> = Vec::new();
    let mut piv_of_col = vec![None; n];
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
        pivcols.push(col);
        piv_of_col[col] = Some(row);
        row += 1;
    }
    let free: Vec<usize> = (0..n).filter(|&c| piv_of_col[c].is_none()).collect();
    let mut basis = Vec::new();
    for &fc in &free {
        let mut v = vec![field.zero(); n];
        v[fc] = field.one();
        for (ri, &pc) in pivcols.iter().enumerate() {
            v[pc] = t[ri * n + fc];
        }
        basis.push(v);
    }
    basis
}

/// `F_2` (bit) rank of a set of m-bit field elements viewed as length-m bit
/// vectors -- the "rank weight" / subspace-dimension check of the scheme
/// samplers.
pub fn f2_rank(elems: &[Fe], m: u32) -> usize {
    let mut rows: Vec<u128> = elems
        .iter()
        .map(|&(lo, hi)| u128::from(lo) | (u128::from(hi) << 64))
        .collect();
    if m <= 128 {
        let mask = if m == 128 { u128::MAX } else { (1u128 << m) - 1 };
        for r in &mut rows {
            *r &= mask;
        }
    }
    let mut rank = 0usize;
    for bit in (0..m as usize).rev() {
        // find pivot with this bit set
        let mut p = None;
        for i in rank..rows.len() {
            if (rows[i] >> bit) & 1 == 1 {
                p = Some(i);
                break;
            }
        }
        if let Some(p) = p {
            rows.swap(rank, p);
            for i in 0..rows.len() {
                if i != rank && (rows[i] >> bit) & 1 == 1 {
                    rows[i] ^= rows[rank];
                }
            }
            rank += 1;
        }
    }
    rank
}

/// Solve A x = b for square invertible A (Gaussian elimination with a
/// working copy); returns None when A is singular.
#[allow(dead_code)] // kept: dense-route cross-check API
pub fn solve_square(field: &Field, a: &[Fe], b: &[Fe], n: usize) -> Option<Vec<Fe>> {
    let mut t = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        let mut p = None;
        for i in col..n {
            if !field.is_zero(t[i * n + col]) {
                p = Some(i);
                break;
            }
        }
        let p = p?;
        if p != col {
            for c in 0..n {
                t.swap(col * n + c, p * n + c);
            }
            x.swap(col, p);
        }
        let iv = field.inv(t[col * n + col]);
        for c in 0..n {
            t[col * n + c] = field.mul(t[col * n + c], iv);
        }
        x[col] = field.mul(x[col], iv);
        for i in 0..n {
            if i != col && !field.is_zero(t[i * n + col]) {
                let fac = t[i * n + col];
                for c in 0..n {
                    t[i * n + c] = field.add(t[i * n + c], field.mul(fac, t[col * n + c]));
                }
                x[i] = field.add(x[i], field.mul(fac, x[col]));
            }
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorsh(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn systematic_form_matches_definition() {
        let f = Field::new(16);
        let mut sm = 42u64;
        for &(k, n) in &[(4usize, 7usize), (5, 8), (3, 9)] {
            for _ in 0..5 {
                let a: Vec<Fe> = (0..k * n)
                    .map(|_| (xorsh(&mut sm) & f.mask().0, 0))
                    .collect();
                let (cols, d) = systematic_form(&f, &a, k, n);
                assert_eq!(cols.len(), n);
                // check: A[:, cols] has invertible first k columns and
                // A[:, cols[:k]] @ D == A[:, cols[k:]]
                let r = n - k;
                let mut apiv = vec![f.zero(); k * k];
                let mut arest = vec![f.zero(); k * r];
                for i in 0..k {
                    for j in 0..k {
                        apiv[i * k + j] = a[i * n + cols[j]];
                    }
                    for b in 0..r {
                        arest[i * r + b] = a[i * n + cols[k + b]];
                    }
                }
                let prod = crate::dense::matmul(&f, &apiv, &d, k, k, r);
                assert_eq!(prod, arest, "k={k} n={n}");
            }
        }
    }

    #[test]
    fn nullspace_basis_is_exact() {
        let f = Field::new(11);
        let mut sm = 4242u64;
        let (k, n) = (3usize, 7usize);
        for _ in 0..10 {
            let a: Vec<Fe> = (0..k * n).map(|_| (xorsh(&mut sm) & f.mask().0, 0)).collect();
            let basis = nullspace(&f, &a, k, n);
            assert_eq!(basis.len(), n - k);
            for v in &basis {
                for i in 0..k {
                    let mut acc = f.zero();
                    for j in 0..n {
                        acc = f.add(acc, f.mul(a[i * n + j], v[j]));
                    }
                    assert!(f.is_zero(acc));
                }
            }
        }
    }

    #[test]
    fn solve_square_roundtrip() {
        let f = Field::new(61);
        let mut sm = 1u64;
        let n = 4usize;
        for _ in 0..20 {
            let a: Vec<Fe> = (0..n * n).map(|_| (xorsh(&mut sm) & f.mask().0, 0)).collect();
            let b: Vec<Fe> = (0..n).map(|_| (xorsh(&mut sm) & f.mask().0, 0)).collect();
            if let Some(x) = solve_square(&f, &a, &b, n) {
                for i in 0..n {
                    let mut acc = f.zero();
                    for j in 0..n {
                        acc = f.add(acc, f.mul(a[i * n + j], x[j]));
                    }
                    assert_eq!(acc, b[i]);
                }
            }
        }
    }
}
