//! `F_2` polynomial helpers on fixed 4-word (<= 256 bit) polynomials.
//!
//! Only used at instance-setup time: validating / searching the irreducible
//! modulus of GF(2^m) for m in 1..=128.  The arithmetic is deliberately
//! simple (schoolbook, shift-subtract remainder, Euclid gcd); performance
//! is irrelevant here, correctness is.
//!
//! Also hosts the small bit utilities on `(u64, u64)` pairs that the field
//! and schedule code share.

/// Bit i of a two-word element.
#[inline]
pub fn bit(x: (u64, u64), i: usize) -> u64 {
    if i < 64 {
        (x.0 >> i) & 1
    } else if i < 128 {
        (x.1 >> (i - 64)) & 1
    } else {
        0
    }
}

/// Bit i of a 4-word value.
#[inline]
#[allow(dead_code)] // F2 bit-toolbox kept complete
pub fn bit4(x: &[u64; 4], i: usize) -> u64 {
    if i >= 256 {
        0
    } else {
        (x[i / 64] >> (i % 64)) & 1
    }
}

#[inline]
pub fn set_bit(mut x: (u64, u64), i: usize) -> (u64, u64) {
    if i < 64 {
        x.0 |= 1u64 << i;
    } else {
        x.1 |= 1u64 << (i - 64);
    }
    x
}

#[inline]
#[allow(dead_code)] // F2 bit-toolbox kept complete
pub fn clear_bit(mut x: (u64, u64), i: usize) -> (u64, u64) {
    if i < 64 {
        x.0 &= !(1u64 << i);
    } else {
        x.1 &= !(1u64 << (i - 64));
    }
    x
}

/// Mask of the low `n` bits, n <= 128 (word-shift safe: never shifts by 64).
pub fn low_mask(n: usize) -> (u64, u64) {
    match n {
        0 => (0, 0),
        1..=63 => ((1u64 << n) - 1, 0),
        64 => (u64::MAX, 0),
        65..=127 => (u64::MAX, (1u64 << (n - 64)) - 1),
        _ => (u64::MAX, u64::MAX),
    }
}

/// Carry-less 64x64 -> 127-bit product (portable u128 shift-and-add;
/// fast enough for the modulus search, which runs once per m).
#[inline]
pub(crate) fn clmul_word(a: u64, b: u64) -> (u64, u64) {
    let mut acc: u128 = 0;
    let mut bb = u128::from(b);
    let mut aa = a;
    while aa != 0 {
        if aa & 1 != 0 {
            acc ^= bb;
        }
        aa >>= 1;
        bb <<= 1;
    }
    (acc as u64, (acc >> 64) as u64)
}

/// Carry-less product of two <= 256-bit polynomials -> 512 bits (8 words).
/// Word-wise schoolbook with 16 word multiplies; carry-less addition is
/// plain XOR (no ripple).
fn pmul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
    let mut acc = [0u64; 8];
    for i in 0..4 {
        if a[i] == 0 {
            continue;
        }
        for j in 0..4 {
            if b[j] == 0 {
                continue;
            }
            let (lo, hi) = clmul_word(a[i], b[j]);
            acc[i + j] ^= lo;
            acc[i + j + 1] ^= hi;
        }
    }
    acc
}

/// Remainder of `a` (any degree <= 511) mod `f` (degree m, f[4] holds the
/// explicit bits; the x^m term included).  Returns 4 words.
fn pmod(a: &[u64; 8], f: &[u64; 4], m: usize) -> [u64; 4] {
    // 8-word (512-bit) working space: every XOR during the reduction
    // clears the current top bit and only touches bits below it, so the
    // leading bit position never increases and everything stays in bounds.
    let mut r = *a;
    for k in (m..512).rev() {
        let bitk = (r[k / 64] >> (k % 64)) & 1;
        if bitk == 0 {
            continue;
        }
        // r ^= f << (k - m)
        let s = k - m;
        for w in 0..4 {
            let fw = f[w];
            if fw == 0 {
                continue;
            }
            let s2 = s + w * 64;
            if s2 >= 512 {
                break;
            }
            let off = s2 % 64;
            let wi = s2 / 64;
            if off == 0 {
                r[wi] ^= fw;
            } else {
                r[wi] ^= fw << off;
                if wi + 1 < 8 {
                    r[wi + 1] ^= fw >> (64 - off);
                }
            }
        }
    }
    [r[0], r[1], r[2], r[3]]
}

fn pdeg(x: &[u64; 4]) -> usize {
    for w in (0..4).rev() {
        if x[w] != 0 {
            return w * 64 + 63 - x[w].leading_zeros() as usize;
        }
    }
    0 // zero polynomial: degree "0" sentinel; callers check == 0 first
}

fn pzero(x: &[u64; 4]) -> bool {
    x.iter().all(|&w| w == 0)
}

/// Euclid gcd of two <= 256-bit `F_2` polynomials.
pub fn pgcd(mut a: [u64; 4], mut b: [u64; 4]) -> [u64; 4] {
    while !pzero(&b) {
        // a mod b
        let db = pdeg(&b);
        loop {
            let da = pdeg(&a);
            if pzero(&a) || da < db {
                break;
            }
            let s = da - db;
            for w in 0..4 {
                let bw = b[w];
                if bw == 0 {
                    continue;
                }
                let s2 = s + w * 64;
                if s2 >= 256 {
                    break;
                }
                let off = s2 % 64;
                let wi = s2 / 64;
                if off == 0 {
                    a[wi] ^= bw;
                } else {
                    a[wi] ^= bw << off;
                    if wi + 1 < 4 {
                        a[wi + 1] ^= bw >> (64 - off);
                    }
                }
            }
        }
        std::mem::swap(&mut a, &mut b);
    }
    a
}

/// Irreducibility of `x^m + g(x)` over `F_2`, where `g` (degree < m) is
/// given WITHOUT the leading x^m term -- the leading term is implicit,
/// which is what lets m = 128 fit the two-word representation (and
/// matches how the modulus is stored in [`crate::field::Field`]).
///
/// Rabin's test:  x^(2^m) == x (mod f)  and
/// gcd(x^(2^(m/q)) - x, f) == 1 for every prime divisor q of m.
pub fn is_irreducible(g: (u64, u64), m: u32) -> bool {
    if m == 1 {
        return g == (1, 0); // x + 1 over GF(2)
    }
    // f = x^m + g as a 4-word array: bit m may land in word 1 (m < 128)
    // or word 2 (m == 128), both within the 256-bit working space.
    let mut fw = [g.0, g.1, 0, 0];
    fw[(m as usize) / 64] |= 1u64 << ((m as usize) % 64);
    // degree check: g must have degree < m
    for i in (m as usize)..128 {
        if bit(g, i) == 1 {
            return false;
        }
    }
    if bit(g, 0) == 0 {
        return false; // f must have a nonzero constant term
    }
    // x as a polynomial: bit 1
    let xw = [2u64, 0, 0, 0];
    // x^(2^k) mod f via k squarings
    let frobenius = |k: usize| -> [u64; 4] {
        let mut h = xw;
        for _ in 0..k {
            let prod = pmul(&h, &h);
            h = pmod(&prod, &fw, m as usize);
        }
        h
    };
    // x^(2^m) == x mod f
    let xm = frobenius(m as usize);
    if xm != xw {
        return false;
    }
    // for each prime q | m: gcd(x^(2^(m/q)) + x, f) must be 1
    let mut mm = m;
    let mut q = 2u32;
    while q * q <= mm {
        if mm % q == 0 {
            let h = frobenius((m / q) as usize);
            let g2 = [h[0] ^ xw[0], h[1] ^ xw[1], h[2], h[3]];
            let d = pgcd(g2, fw);
            if !pzero(&d) && pdeg(&d) > 0 {
                return false;
            }
            while mm % q == 0 {
                mm /= q;
            }
        }
        q += 1;
    }
    if mm > 1 {
        let h = frobenius((m / mm) as usize);
        let g2 = [h[0] ^ xw[0], h[1] ^ xw[1], h[2], h[3]];
        let d = pgcd(g2, fw);
        if !pzero(&d) && pdeg(&d) > 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_irreducibles() {
        // x^16 + x^5 + x^3 + x^2 + 1: the Conway polynomial of GF(2^16)
        assert!(is_irreducible((0x2D, 0), 16));
        // reducible: x^16 + 1 = (x+1)^16
        assert!(!is_irreducible((1, 0), 16));
        // x^8 + x^4 + x^3 + x^2 + 1: AES / Conway GF(2^8)
        assert!(is_irreducible((0x1B, 0), 8));
        // x^64 + x^4 + x^3 + x + 1: Conway GF(2^64)
        assert!(is_irreducible((0x1B, 0), 64));
        // x^128 + x^7 + x^2 + x + 1: GCM polynomial, irreducible
        assert!(is_irreducible((0x87, 0), 128));
        // x^4 + x^2 + 1 = (x^2+x+1)^2, reducible
        assert!(!is_irreducible((0b101, 0), 4));
        assert!(is_irreducible((0b11, 0), 4)); // x^4+x+1
    }

    #[test]
    fn default_poly_is_irreducible_for_all_m() {
        for m in 1..=128u32 {
            let g = super::super::field::Field::default_poly(m);
            assert!(is_irreducible(g, m), "m={m} g={g:?}");
        }
    }
}
