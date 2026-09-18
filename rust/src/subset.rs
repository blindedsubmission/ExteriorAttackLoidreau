//! Binomials and exterior-coordinate orderings.
//!
//! Exterior (Pluecker) coordinates are indexed by lambda-subsets of
//! [0, n).  Two orders appear:
//!
//! * **colex** — the coordinate order of the substitute-and-discard
//!   circuit.  `colex_rank(I) = sum_t C(I[t], t+1)`; its load-bearing
//!   property is that the subsets of a prefix [0, j) occupy exactly the
//!   rank range [0, C(j, lam)), which is what lets circuit stages update
//!   one buffer in place and gives forward/transposed write regions that
//!   are disjoint rank ranges (the GPU race-freedom argument).
//! * **lex** — the coordinate order of `compound_matrix` and of the
//!   dense engines / recovery pipeline (`kernel.bin` compatibility).
//!
//! Everything here is precomputed once per instance; ranks fit in u32
//! for every parameter set the pipeline accepts (enforced here).

/// Pascal triangle up to degree `n`, columns 0..=lam, in u64.
#[derive(Clone)]
pub struct Binomial {
    pub t: Vec<Vec<u64>>,
}

impl Binomial {
    pub fn new(n: usize, lam: usize) -> Binomial {
        let mut t = vec![vec![0u64; lam + 1]; n + 1];
        for i in 0..=n {
            t[i][0] = 1;
            for k in 1..=lam.min(i) {
                t[i][k] = t[i - 1][k - 1] + t[i - 1][k];
            }
        }
        Binomial { t }
    }
    /// C(i, k); k must be <= the lam given at construction.
    pub fn c(&self, i: usize, k: usize) -> u64 {
        if k > i {
            0
        } else {
            self.t[i][k]
        }
    }
}

/// Colex rank of an increasing subset (slice sorted ascending).
pub fn colex_rank(subset: &[usize], binom: &Binomial) -> u32 {
    let mut r: u64 = 0;
    for (t, &i) in subset.iter().enumerate() {
        r += binom.c(i, t + 1);
    }
    assert!(u32::try_from(r).is_ok(), "coordinate count exceeds u32; reduce n or lambda");
    r as u32
}

/// All lam-subsets of [0, n) in lex order, as sorted index vectors.
pub fn subsets_lex(n: usize, lam: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if lam == 0 {
        out.push(vec![]);
        return out;
    }
    if lam > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..lam).collect();
    loop {
        out.push(idx.clone());
        // advance: rightmost position that can move
        let mut t = lam as isize - 1;
        while t >= 0 && idx[t as usize] == n - lam + t as usize {
            t -= 1;
        }
        if t < 0 {
            break;
        }
        let t = t as usize;
        idx[t] += 1;
        for u in t + 1..lam {
            idx[u] = idx[u - 1] + 1;
        }
    }
    out
}

/// Permutation colex -> lex position: entry c is the lex index of the
/// subset with colex rank c.  (`argsort` of `colex_of_lex`.)
pub fn lex_of_colex(n: usize, lam: usize, binom: &Binomial) -> Vec<u32> {
    let mut v: Vec<(u32, u32)> = subsets_lex(n, lam)
        .iter()
        .enumerate()
        .map(|(l, s)| (colex_rank(s, binom), l as u32))
        .collect();
    v.sort_by_key(|x| x.0);
    v.into_iter().map(|x| x.1).collect()
}

/// Permutation lex -> colex position: entry l is the colex rank of the
/// l-th subset in lex order.
#[allow(dead_code)] // documented inverse of lex_of_colex
pub fn colex_of_lex(n: usize, lam: usize, binom: &Binomial) -> Vec<u32> {
    let mut out = vec![0u32; binom.c(n, lam) as usize];
    for (l, s) in subsets_lex(n, lam).iter().enumerate() {
        out[l] = colex_rank(s, binom);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_are_bijections() {
        let (n, lam) = (9usize, 3usize);
        let binom = Binomial::new(n, lam);
        assert_eq!(binom.c(n, lam) as usize, subsets_lex(n, lam).len());
        // colex ranks over all lex subsets are distinct and in range
        let mut seen = std::collections::HashSet::new();
        for s in subsets_lex(n, lam) {
            let r = colex_rank(&s, &binom);
            assert!((r as u64) < binom.c(n, lam));
            assert!(seen.insert(r));
        }
        // prefix property: subsets of [0, j) have rank < C(j, lam)
        for j in 0..=n {
            for s in subsets_lex(n, lam) {
                if s.iter().all(|&x| x < j) {
                    assert!((colex_rank(&s, &binom) as u64) < binom.c(j, lam), "j={j} s={s:?}");
                }
            }
        }
        // inverse permutations
        let lox = lex_of_colex(n, lam, &binom);
        let xol = colex_of_lex(n, lam, &binom);
        for l in 0..(binom.c(n, lam) as usize) {
            assert_eq!(lox[xol[l] as usize], l as u32);
        }
    }
}
