//! Dense compound matrices -- the validation oracle.
//!
//! `compound_matrix` builds the lambda-th compound `C_lam(A)` explicitly
//! (Leibniz over lam! permutations; signs are trivial in characteristic 2).
//! This is exponentially slower than the circuit and exists ONLY so the
//! circuit, the instance sampler, and the wedge ground truth can be
//! checked against the definition on small toy shapes (mirroring
//! `exterior.compound_matrix` in the Python reference).

use crate::field::{Fe, Field};
use crate::subset::subsets_lex;

/// `C_lam(A)` for a row-major `rows x cols` matrix over GF(2^m).
/// Output is row-major C(rows,lam) x C(cols,lam) in **lex** subset order:
/// entry (I, T) = det(A[I, T]).
pub fn compound_matrix(field: &Field, a: &[Fe], rows: usize, cols: usize, lam: usize) -> Vec<Fe> {
    assert!(lam >= 1 && rows >= lam && cols >= lam);
    let row_sets = subsets_lex(rows, lam);
    let col_sets = subsets_lex(cols, lam);
    let perms = permutations(lam);
    let mut out = vec![field.zero(); row_sets.len() * col_sets.len()];
    for (ri, i) in row_sets.iter().enumerate() {
        for (ti, t) in col_sets.iter().enumerate() {
            // Leibniz: sum over permutations sigma of prod_a A[I[a], T[sigma(a)]]
            let mut det = field.zero();
            for p in &perms {
                let mut prod = field.one();
                for (a_idx, &sa) in p.iter().enumerate() {
                    prod = field.mul(prod, a[i[a_idx] * cols + t[sa]]);
                }
                det = field.add(det, prod);
            }
            out[ri * col_sets.len() + ti] = det;
        }
    }
    out
}

/// All permutations of 0..lam as index vectors.
fn permutations(lam: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur: Vec<usize> = (0..lam).collect();
    out.push(cur.clone());
    if lam <= 1 {
        return out;
    }
    // Heap's algorithm
    let mut c = vec![0usize; lam];
    let mut i = 0usize;
    while i < lam {
        if c[i] < i {
            if i % 2 == 0 {
                cur.swap(0, i);
            } else {
                cur.swap(c[i], i);
            }
            out.push(cur.clone());
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    out
}

/// The stacked compound of inverse-Frobenius shifts:
/// rows of `Comp_lam(G)`, `Comp_lam(G`^{[-1]}), ..., `Comp_lam(G`^{[-(s-1)]}),
/// in that order (lex subset order within each block).
#[allow(dead_code)] // dense cross-check + test oracle
pub fn stacked_compound(field: &Field, g: &[Fe], k: usize, n: usize, lam: usize, s: usize) -> Vec<Fe> {
    let rows = k;
    let cols = n;
    let sets = subsets_lex(rows, lam);
    let colsets = subsets_lex(cols, lam);
    let rb = sets.len();
    let cb = colsets.len();
    let mut out = vec![field.zero(); s * rb * cb];
    for i in 0..s {
        let shifted: Vec<Fe> = g.iter().map(|&x| field.frob(x, -(i as i32))).collect();
        let block = compound_matrix(field, &shifted, rows, cols, lam);
        out[i * rb * cb..(i + 1) * rb * cb].copy_from_slice(&block);
    }
    out
}

/// Dense row-major p x r product A (p x q) times B (q x r).
#[allow(dead_code)] // dense cross-check + test oracle
pub fn matmul(f: &Field, a: &[Fe], b: &[Fe], p: usize, q: usize, r: usize) -> Vec<Fe> {
    let mut out = vec![f.zero(); p * r];
    for i in 0..p {
        for l in 0..r {
            let mut acc = f.zero();
            for j in 0..q {
                acc = f.add(acc, f.mul(a[i * q + j], b[j * r + l]));
            }
            out[i * r + l] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_of_identity_is_identity() {
        let f = Field::new(8);
        let k = 5usize;
        let a: Vec<Fe> = (0..k * k)
            .map(|i| if i / k == i % k { f.one() } else { f.zero() })
            .collect();
        let c = compound_matrix(&f, &a, k, k, 2);
        for (i, &x) in c.iter().enumerate() {
            assert_eq!(x, if i / c.len().isqrt() == i % c.len().isqrt() { f.one() } else { f.zero() });
        }
    }

    #[test]
    fn compound_multiplicative() {
        // C_lam(AB) == C_lam(A) C_lam(B) on small dense inputs
        let f = Field::new(13);
        let (p, q, rr, lam) = (4usize, 5usize, 6usize, 2usize);
        let mut sm = 777u64;
        let mut rnd = || {
            sm ^= sm << 13;
            sm ^= sm >> 7;
            sm ^= sm << 17;
            (sm & f.mask().0, 0)
        };
        let a: Vec<Fe> = (0..p * q).map(|_| rnd()).collect();
        let b: Vec<Fe> = (0..q * rr).map(|_| rnd()).collect();
        let ab = matmul(&f, &a, &b, p, q, rr);
        let ca = compound_matrix(&f, &a, p, q, lam);
        let cb = compound_matrix(&f, &b, q, rr, lam);
        let cab = compound_matrix(&f, &ab, p, rr, lam);
        let lsets = subsets_lex(p, lam);
        let msets = subsets_lex(q, lam);
        let nsets = subsets_lex(rr, lam);
        for (li, l) in lsets.iter().enumerate() {
            for (ni, nn) in nsets.iter().enumerate() {
                let mut acc = f.zero();
                for (mi, _) in msets.iter().enumerate() {
                    acc = f.add(acc, f.mul(ca[li * msets.len() + mi], cb[mi * nsets.len() + ni]));
                }
                // lex positions of l, nn in cab
                let cab_l = lsets.iter().position(|x| x == l).unwrap();
                let cab_n = subsets_lex(rr, lam).iter().position(|x| x == nn).unwrap();
                assert_eq!(acc, cab[cab_l * nsets.len() + cab_n]);
            }
        }
    }
}
