//! GF(2^m) arithmetic for extension degrees m in 1..=128.
//!
//! # Representation
//!
//! A field element is a pair of `u64` words `(lo, hi)` holding the
//! coefficient bits of a polynomial of degree < m (`lo` carries bits 0..63,
//! `hi` bits 64..127).  m <= 64 keeps `hi == 0`; m up to 128 uses both
//! words.  This single uniform representation is the reason the whole
//! pipeline supports "m up to 128" without generic gymnastics: the CPU
//! paths branch once on `m <= 64`, the GPU path always works on four
//! `u32` limbs and simply skips the two high limbs when `m <= 64`.
//!
//! The field is fixed by the modulus `f(x) = x^m + g(x)` with
//! `deg g < m`; `g` is stored as two words.  Addition is XOR;
//! multiplication is a carry-less product followed by the single
//! substitution `x^m -> g` applied from the top bit down.
//!
//! # Why two multiply paths
//!
//! * `m <= 64`: the product fits in 127 bits.  With PCLMULQDQ (detected at
//!   runtime) one 64x64 carry-less multiply plus at most two fold steps
//!   gives a very fast mul -- this is the path the heavy CPU loops (Krylov
//!   dots, Berlekamp-Massey) run on.  Without clmul (non-x86 builds) a
//!   portable `u128` shift-and-add product is used instead.
//! * `64 < m <= 128`: a 128x128 carry-less schoolbook product (4 clmuls or
//!   64 shift-and-adds per word pair) into 256 bits, then a generic
//!   word-wise fold.  This path is deliberately simple; large-m CPU
//!   performance is a non-goal (the GPU does the heavy lifting), it only
//!   needs to be correct and fast enough for toy instances.
//!
//! All operations are AND/XOR/shift only -- every route is bit-deterministic,
//! which is what the CPU-vs-GPU bit-identity tests rely on.

/// A field element: `(lo, hi)` coefficient words of a polynomial of
/// degree < m over `F_2`.
pub type Fe = (u64, u64);

#[derive(Clone, Debug)]
pub struct Field {
    /// Extension degree, 1 <= m <= 128.
    pub m: u32,
    /// `f(x) = x^m + g(x)`; `g` has degree < m, stored `(lo, hi)`.
    pub g: Fe,
    /// Runtime-detected PCLMULQDQ support (`x86_64` only).
    clmul: bool,
}

#[inline]
fn shr64(x: u64, s: u32) -> u64 {
    if s >= 64 { 0 } else { x >> s }
}

#[inline]
fn shl64_or(x: u64, s: u32) -> u64 {
    if s >= 64 { x } else { x << s }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn clmul64(a: u64, b: u64) -> (u64, u64) {
    // Safety: guarded by the runtime `pclmulqdq` detection that fills
    // `Field::clmul` before any call.
    unsafe {
        use std::arch::x86_64::{_mm_clmulepi64_si128, _mm_set_epi64x, _mm_extract_epi64};
        let p = _mm_clmulepi64_si128(_mm_set_epi64x(0, a as i64), _mm_set_epi64x(0, b as i64), 0x00);
        (
            _mm_extract_epi64(p, 0) as u64,
            _mm_extract_epi64(p, 1) as u64,
        )
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn clmul64(a: u64, b: u64) -> (u64, u64) {
    // single shared portable implementation (see poly::clmul_word)
    crate::poly::clmul_word(a, b)
}

impl Field {
    /// Build GF(2^m) with a deterministically searched irreducible modulus
    /// (see [`Field::default_poly`]).
    pub fn new(m: u32) -> Field {
        assert!((1..=128).contains(&m), "extension degree must be in 1..=128");
        let g = Field::default_poly(m);
        Field { m, g, clmul: detect_clmul() }
    }

    /// Build GF(2^m) with the explicit modulus `x^m + g(x)`, where `g`
    /// (degree < m) is given as its coefficient bitmask over `(lo, hi)`
    /// words -- the x^m term is implicit, which is what makes m = 128
    /// representable.  Returns None unless `g` has degree < m and
    /// `x^m + g` is irreducible over `F_2`.
    pub fn with_poly(m: u32, g: Fe) -> Option<Field> {
        assert!((1..=128).contains(&m));
        if m > 1 && !crate::poly::is_irreducible(g, m) {
            return None;
        }
        if m == 1 && g != (1, 0) {
            return None;
        }
        Some(Field { m, g, clmul: detect_clmul() })
    }

    /// Deterministic irreducible-modulus search, memoized per m.  Tries a
    /// short list of sparse candidates (x^m + x^t + 1) and then
    /// splitmix64-guided low-density candidates, returning the first
    /// irreducible polynomial.  The result only depends on m, so CPU and
    /// GPU and repeated runs all agree.
    pub fn default_poly(m: u32) -> Fe {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<HashMap<u32, Fe>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(p) = cache.lock().unwrap().get(&m) {
            return *p;
        }
        let p = Self::default_poly_uncached(m);
        cache.lock().unwrap().insert(m, p);
        p
    }

    /// The uncached modulus search (see [`Field::default_poly`]).  Returns
    /// the low part `g` of the modulus `x^m + g`; the leading term is
    /// implicit.
    fn default_poly_uncached(m: u32) -> Fe {
        assert!((1..=128).contains(&m));
        if m == 1 {
            return (1, 0); // x + 1
        }
        // all trinomials x^m + x^t + 1, t ascending (none exist when
        // m = 0 mod 8, Swan's theorem); sparse g keeps the reduction fold
        // in mul() one or two clmuls deep instead of ~(m/(64-m)) rounds
        for t in 1..m {
            let g = crate::poly::set_bit((1, 0), t as usize);
            if crate::poly::is_irreducible(g, m) {
                return g;
            }
        }
        // pentanomials x^m + x^a + x^b + x^c + 1 (a > b > c >= 1); an
        // irreducible one exists for every m <= 128 (Seroussi's tables),
        // so this bounded scan finds one in practice
        let mut tries = 0u32;
        'penta: for a in 1..m {
            for b in 1..a {
                for c in 1..b {
                    let mut g = crate::poly::set_bit((1, 0), a as usize);
                    g = crate::poly::set_bit(g, b as usize);
                    g = crate::poly::set_bit(g, c as usize);
                    if crate::poly::is_irreducible(g, m) {
                        return g;
                    }
                    tries += 1;
                    if tries > 200_000 {
                        break 'penta;
                    }
                }
            }
        }
        // weight-7 moduli, same enumeration plus two inner exponents
        // (bounded; not expected to run for m <= 128)
        let mut tries7 = 0u32;
        'hepta: for a in 1..m {
            for b in 1..a {
                for c in 1..b {
                    for d in 1..c {
                        for e in 1..d {
                            let mut g = crate::poly::set_bit((1, 0), a as usize);
                            g = crate::poly::set_bit(g, b as usize);
                            g = crate::poly::set_bit(g, c as usize);
                            g = crate::poly::set_bit(g, d as usize);
                            g = crate::poly::set_bit(g, e as usize);
                            if crate::poly::is_irreducible(g, m) {
                                return g;
                            }
                            tries7 += 1;
                            if tries7 > 200_000 {
                                break 'hepta;
                            }
                        }
                    }
                }
            }
        }
        // dense splitmix-guided candidates (density ~1/m) -- historical
        // fallback, kept so the search always terminates with SOME
        // irreducible modulus.
        let imask = crate::poly::low_mask(m as usize);
        let interior = (imask.0 & !1u64, imask.1);
        let mut sm = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(m).wrapping_mul(0x1000_0000_0000_0001);
        loop {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let g = (z & interior.0 | 1, z & interior.1);
            if crate::poly::is_irreducible(g, m) {
                return g;
            }
        }
    }

/// Bitmask of the m coefficient bits.
    pub fn mask(&self) -> Fe {
        if self.m >= 64 {
            (
                u64::MAX,
                if self.m == 64 {
                    0
                } else if self.m == 128 {
                    u64::MAX
                } else {
                    (1u64 << (self.m - 64)) - 1
                },
            )
        } else {
            ((1u64 << self.m) - 1, 0)
        }
    }

    pub fn zero(&self) -> Fe {
        (0, 0)
    }
    pub fn one(&self) -> Fe {
        (1, 0)
    }
    pub fn is_zero(&self, a: Fe) -> bool {
        a.0 == 0 && a.1 == 0
    }
    pub fn add(&self, a: Fe, b: Fe) -> Fe {
        (a.0 ^ b.0, a.1 ^ b.1)
    }

    /// Multiplicative inverse by the addition-chain-free route
    /// `a^(2^m - 2)`; used only in setup and Berlekamp-Massey, never in
    /// the hot Krylov loops.
    pub fn inv(&self, a: Fe) -> Fe {
        assert!(!self.is_zero(a), "0 has no inverse");
        let (elo, ehi): (u64, u64) = match self.m {
            m if m < 64 => ((1u128 << m) as u64 - 2, 0),
            64 => (!1, 0),
            m if m <= 128 => {
                // 2^m - 2 for 64 < m <= 128: bits 1..m-1 set, i.e.
                // (all ones) << 1 within m bits.
                let hi = if m >= 128 { u64::MAX } else { (1u64 << (m - 64)) - 1 };
                (!1u64, hi)
            }
            _ => unreachable!(),
        };
        let mut r = self.one();
        let mut base = a;
        for bit in 0..128 {
            let e = if bit < 64 { elo } else { ehi };
            if (e >> (bit % 64)) & 1 != 0 {
                r = self.mul(r, base);
            }
            base = self.mul(base, base);
        }
        r
    }

    /// a * a (used to generate the Frobenius-shifted coefficient tables).
    pub fn sqr(&self, a: Fe) -> Fe {
        self.mul(a, a)
    }

    /// a^(2^e): e-fold Frobenius, e may be any i32 (reduced mod m).
    pub fn frob(&self, a: Fe, e: i32) -> Fe {
        let m = self.m as i32;
        let e = ((e % m) + m) % m;
        let mut x = a;
        for _ in 0..e {
            x = self.sqr(x);
        }
        x
    }

    /// Field multiplication.
    pub fn mul(&self, a: Fe, b: Fe) -> Fe {
        if self.m <= 64 {
            let (mut lo, mut hi) = if self.clmul { clmul64(a.0, b.0) } else { crate::poly::clmul_word(a.0, b.0) };
            if self.m == 64 {
                // shift-free special case: x^64 -> g word-wise; every round
                // strictly decreases the top bit (deg g <= 62).
                while hi != 0 {
                    let (u0, u1) = if self.clmul { clmul64(hi, self.g.0) } else { crate::poly::clmul_word(hi, self.g.0) };
                    lo ^= u0;
                    hi = u1;
                }
                (lo, 0)
            } else {
                // fast fold: bits >= m of the 127-bit product -> * g
                let shift = 64 - self.m;
                let mask = (1u64 << self.m) - 1;
                while hi != 0 {
                    let (u0, u1) = if self.clmul { clmul64(hi, self.g.0) } else { crate::poly::clmul_word(hi, self.g.0) };
                    lo ^= u0 << shift;
                    // u1 << shift cannot overflow: deg(u1) + shift <= (m - 2) + (64 - m) = 62
                    hi = shl64_or(u1, shift) | shr64(u0, self.m);
                }
                while lo >> self.m != 0 {
                    let (q0, _q1) = if self.clmul { clmul64(lo >> self.m, self.g.0) } else { crate::poly::clmul_word(lo >> self.m, self.g.0) };
                    lo = (lo & mask) ^ q0;
                }
                (lo, 0)
            }
        } else {
            // 128x128 carry-less schoolbook into 4 words.
            let (a0, a1, b0, b1) = (a.0, a.1, b.0, b.1);
            let (p00l, p00h) = self.clmul64_(a0, b0);
            let (p01l, p01h) = self.clmul64_(a0, b1);
            let (p10l, p10h) = self.clmul64_(a1, b0);
            let (p11l, p11h) = self.clmul64_(a1, b1);
            let mut p = [
                p00l,
                p00h ^ p01l ^ p10l,
                p01h ^ p10h ^ p11l,
                p11h,
            ];
            self.reduce_wide(&mut p);
            (p[0], p[1])
        }
    }

    /// Fast path for m <= 64: both operands carry hi == 0; skips the
    /// generic dispatch branches of `mul` per coefficient (hot in the
    /// approximant-basis elimination).
    #[inline]
    pub fn mul_u64(&self, a: Fe, b: Fe) -> Fe {
        debug_assert!(self.m <= 64);
        self.mul(a, b)
    }

    #[inline]
    fn clmul64_(&self, a: u64, b: u64) -> (u64, u64) {
        if self.clmul { clmul64(a, b) } else { crate::poly::clmul_word(a, b) }
    }

    /// Test-only constructor that forces the portable (no-clmul) paths, so
    /// both multiply implementations can be cross-checked.
    #[cfg(test)]
    pub(crate) fn with_clmul_pref(m: u32, use_clmul: bool) -> Field {
        let g = Field::default_poly(m);
        Field { m, g, clmul: use_clmul }
    }

    /// Fold bits >= m of a 256-bit product down via the substitution
    /// x^k -> x^(k-m) * (x^m + g)  (the x^m term is what clears bit k).
    /// Simple top-bit loop; only used on the m > 64 path (setup + toy
    /// instances), so clarity beats speed here.
    fn reduce_wide(&self, p: &mut [u64; 4]) {
        let m = self.m as usize;
        // f = x^m + g as a 3-word polynomial (m <= 128 -> word index <= 2)
        let mut f = [self.g.0, self.g.1, 0u64];
        f[m / 64] |= 1u64 << (m % 64);
        loop {
            // highest set bit of p at position >= m
            let mut top: Option<usize> = None;
            for w in (0..4).rev() {
                if p[w] != 0 {
                    top = Some(w * 64 + 63 - p[w].leading_zeros() as usize);
                    break;
                }
            }
            let k = match top {
                Some(k) if k >= m => k,
                _ => break,
            };
            // p ^= f << (k - m): clears bit k, only touches bits < k
            let s = k - m;
            for (w, &fw) in f.iter().enumerate() {
                if fw == 0 {
                    continue;
                }
                let s2 = s + w * 64;
                if s2 >= 256 {
                    break;
                }
                let off = s2 % 64;
                let wi = s2 / 64;
                if off == 0 {
                    p[wi] ^= fw;
                } else {
                    p[wi] ^= fw << off;
                    if wi + 1 < 4 {
                        p[wi + 1] ^= fw >> (64 - off);
                    }
                }
            }
        }
    }

    /// Uniform random nonzero element from a u64 (used for conditioning
    /// weights; low bits suffice for m <= 64, both words for larger m).
    #[allow(dead_code)] // conditioning-weight helper (research tooling)
    pub fn from_seed_word(&self, w: u64) -> Fe {
        if self.m <= 64 {
            let v = w & self.mask().0;
            (if v == 0 { 1 } else { v }, 0)
        } else {
            // mix again for the high word so both words are well distributed
            let mut z = w.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            z ^= z >> 33;
            let hi = z & self.mask().1;
            let lo = w;
            if lo == 0 && hi == 0 {
                (1, 0)
            } else {
                (lo, hi)
            }
        }
    }

    /// If `a = c * b` for some nonzero c, return c (else None).  Both
    /// vectors must be nonzero.
    pub fn proportional(&self, a: &[Fe], b: &[Fe]) -> Option<Fe> {
        let p = b.iter().position(|&x| !self.is_zero(x))?;
        if self.is_zero(a[p]) {
            return None;
        }
        let c = self.mul(a[p], self.inv(b[p]));
        for (x, y) in a.iter().zip(b) {
            if *x != self.mul(c, *y) {
                return None;
            }
        }
        Some(c)
    }

    /// Pack into the GPU's uniform 4x u32 limb layout (lo0, lo1, hi0, hi1).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // GPU layout helpers
    pub fn to_limbs(a: Fe) -> [u32; 4] {
        [a.0 as u32, (a.0 >> 32) as u32, a.1 as u32, (a.1 >> 32) as u32]
    }
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // GPU layout helpers
    pub fn from_limbs(l: [u32; 4]) -> Fe {
        (u64::from(l[0]) | (u64::from(l[1]) << 32), u64::from(l[2]) | (u64::from(l[3]) << 32))
    }
}

fn detect_clmul() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("pclmulqdq")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-by-bit shift-and-reduce reference: builds the 256-bit carry-less
    /// product one term at a time, then folds every bit >= m down.  Slow and
    /// obviously correct; the fast paths are checked against it.
    fn slow_mul(f: &Field, a: Fe, b: Fe) -> Fe {
        let mut acc = [0u64; 4];
        for i in 0..f.m as usize {
            if crate::poly::bit(a, i) == 1 {
                for w in 0..2 {
                    let bv = if w == 0 { b.0 } else { b.1 };
                    if bv == 0 {
                        continue;
                    }
                    let s = i + w * 64;
                    let off = s % 64;
                    let wi = s / 64;
                    if off == 0 {
                        acc[wi] ^= bv;
                    } else {
                        acc[wi] ^= bv << off;
                        if wi + 1 < 4 {
                            acc[wi + 1] ^= bv >> (64 - off);
                        }
                    }
                }
            }
        }
        let mm = f.m as usize;
        // modulus f = x^m + g as 4 words (x^m explicit so bit k cancels)
        let mut fm = [0u64; 4];
        fm[0] = f.g.0;
        fm[1] = f.g.1;
        fm[mm / 64] |= 1u64 << (mm % 64);
        for k in (mm..256).rev() {
            if crate::poly::bit4(&acc, k) == 1 {
                let s = k - mm;
                for w in 0..4 {
                    let fword = fm[w];
                    if fword == 0 {
                        continue;
                    }
                    let s2 = s + w * 64;
                    if s2 >= 256 {
                        break;
                    }
                    let off = s2 % 64;
                    let wi = s2 / 64;
                    if off == 0 {
                        acc[wi] ^= fword;
                    } else {
                        acc[wi] ^= fword << off;
                        if wi + 1 < 4 {
                            acc[wi + 1] ^= fword >> (64 - off);
                        }
                    }
                }
            }
        }
        (acc[0], acc[1])
    }

    #[test]
    fn mul_matches_slow_reference() {
        let mut sm = 12345u64;
        let mut rnd = move || {
            sm ^= sm << 13;
            sm ^= sm >> 7;
            sm ^= sm << 17;
            sm
        };
        for &m in &[2u32, 3, 8, 16, 32, 61, 63, 64, 65, 96, 100, 127, 128] {
            for &use_clmul in &[true, false] {
                if use_clmul && !detect_clmul() {
                    continue;
                }
                let f = Field::with_clmul_pref(m, use_clmul);
                for _ in 0..200 {
                    let a = (rnd() & f.mask().0, rnd() & f.mask().1);
                    let b = (rnd() & f.mask().0, rnd() & f.mask().1);
                    let fast = f.mul(a, b);
                    let slow = slow_mul(&f, a, b);
                    assert_eq!(fast, slow, "m={m} clmul={use_clmul} a={a:?} b={b:?}");
                }
            }
        }
    }

    #[test]
    fn field_axioms() {
        for &m in &[2u32, 8, 16, 61, 100, 128] {
            let f = Field::new(m);
            let mut sm = 999u64;
            let mut rnd = move || {
                sm ^= sm << 13;
                sm ^= sm >> 7;
                sm ^= sm << 17;
                sm
            };
            let mut one_seen = false;
            for _ in 0..100 {
                let a = (rnd() & f.mask().0, rnd() & f.mask().1);
                let b = (rnd() & f.mask().0, rnd() & f.mask().1);
                let c = (rnd() & f.mask().0, rnd() & f.mask().1);
                // distributivity
                assert_eq!(f.mul(a, f.add(b, c)), f.add(f.mul(a, b), f.mul(a, c)));
                // associativity
                assert_eq!(f.mul(f.mul(a, b), c), f.mul(a, f.mul(b, c)));
                if !f.is_zero(a) {
                    let ia = f.inv(a);
                    assert_eq!(f.mul(a, ia), f.one());
                    one_seen = true;
                }
            }
            assert!(one_seen);
        }
    }

    #[test]
    fn frob_is_field_automorphism() {
        let f = Field::new(16);
        let mut sm = 7u64;
        let mut rnd = move || {
            sm ^= sm << 13;
            sm ^= sm >> 7;
            sm ^= sm << 17;
            sm
        };
        for _ in 0..50 {
            let a = (rnd() & f.mask().0, rnd() & f.mask().1);
            let b = (rnd() & f.mask().0, rnd() & f.mask().1);
            assert_eq!(f.frob(f.mul(a, b), 3), f.mul(f.frob(a, 3), f.frob(b, 3)));
            assert_eq!(f.frob(a, 16), a); // sigma^m = id
            assert_eq!(f.frob(a, -3), f.frob(a, 13));
        }
    }

    #[test]
    fn limb_roundtrip() {
        let f = Field::new(100);
        let mut sm = 5u64;
        let mut rnd = move || {
            sm ^= sm << 13;
            sm ^= sm >> 7;
            sm ^= sm << 17;
            sm
        };
        for _ in 0..50 {
            let a = (rnd(), rnd() & f.mask().1);
            assert_eq!(Field::from_limbs(Field::to_limbs(a)), a);
        }
    }
}
