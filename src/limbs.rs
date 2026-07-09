//! Base-`2^ℓ` limb decomposition of a secp256k1 scalar.
//!
//! ec-segve segments the canonical scalar `m ∈ [0, n)` into [`LIMB_COUNT`]
//! limbs of [`LIMB_BITS`] bits each (little-endian: limb 0 is the least
//! significant). Each limb `v_k ∈ [0, 2^ℓ)`, and `m = Σ v_k·2^{ℓk}` as an
//! exact integer — which, because `m < n`, equals `recompose` evaluated in the
//! scalar field. `L·ℓ = 264 ≥ 256` covers a full scalar; the top limb's high
//! `L·ℓ − 256 = 8` bits are always zero for an in-range `m`.

use crate::params::Params;
use k256::Scalar;
use k256::elliptic_curve::PrimeField;

/// The frozen segmentation tuple this module is built against.
const PARAMS: Params = Params::FROZEN;

/// Limb bit-width `ℓ`.
pub const LIMB_BITS: u32 = PARAMS.limb_bits;

/// Number of limbs `L`.
pub const LIMB_COUNT: usize = PARAMS.limb_count as usize;

/// Per-limb modulus `2^ℓ`; every limb satisfies `v_k ∈ [0, LIMB_MODULUS)`.
pub const LIMB_MODULUS: u64 = 1u64 << LIMB_BITS;

/// Bit mask selecting one limb's worth of bits.
const LIMB_MASK: u64 = LIMB_MODULUS - 1;

/// Decompose a scalar into `LIMB_COUNT` little-endian base-`2^ℓ` limbs.
///
/// Each returned limb is `< 2^ℓ`. For any `m ∈ [0, n)`, the limbs reproduce
/// `m` exactly via [`recompose`].
#[must_use]
pub fn decompose(m: &Scalar) -> [u32; LIMB_COUNT] {
    // `to_repr` is big-endian; reverse to little-endian so limb k reads from
    // bit offset k·ℓ directly.
    let be = m.to_repr();
    let mut le = [0u8; 32];
    for (dst, src) in le.iter_mut().zip(be.iter().rev()) {
        *dst = *src;
    }

    let mut limbs = [0u32; LIMB_COUNT];
    for (k, limb) in limbs.iter_mut().enumerate() {
        let bit_offset = k * LIMB_BITS as usize;
        let mut acc: u64 = 0;
        for b in 0..LIMB_BITS as usize {
            let pos = bit_offset + b;
            if pos >= 256 {
                break;
            }
            let byte = le[pos / 8];
            let bit = u64::from((byte >> (pos % 8)) & 1);
            acc |= bit << b;
        }
        *limb = (acc & LIMB_MASK) as u32;
    }
    limbs
}

/// Recompose limbs into a scalar: `Σ v_k·2^{ℓk}` reduced mod `n`.
///
/// Inverse of [`decompose`] for in-range scalars. Evaluated by Horner's method
/// over the per-limb base `2^ℓ`.
#[must_use]
pub fn recompose(limbs: &[u32; LIMB_COUNT]) -> Scalar {
    let base = Scalar::from(LIMB_MODULUS);
    let mut acc = Scalar::ZERO;
    for limb in limbs.iter().rev() {
        acc = acc * base + Scalar::from(u64::from(*limb));
    }
    acc
}

/// The per-limb recomposition weights `2^{ℓk} = (2^ℓ)^k` for `k ∈ [0, L)`, as
/// scalars.
///
/// `Σ_k weight_k·v_k` is the full scalar a limb set recomposes to — the weights
/// the linking sigma binds to `C` and the assembly absorbs. Distinct from the
/// range proof's per-digit `b^j` weights (base `b`, within one limb).
#[must_use]
pub fn limb_weights() -> Vec<Scalar> {
    let base = Scalar::from(LIMB_MODULUS);
    let mut weights = Vec::with_capacity(LIMB_COUNT);
    let mut acc = Scalar::ONE;
    for _ in 0..LIMB_COUNT {
        weights.push(acc);
        acc *= base;
    }
    weights
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;
    use k256::elliptic_curve::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Assert each limb is in range and that `recompose ∘ decompose` is the
    /// identity for `m`.
    fn assert_roundtrip(m: &Scalar) {
        let limbs = decompose(m);
        for v in limbs {
            assert!(u64::from(v) < LIMB_MODULUS, "limb {v} >= 2^l");
        }
        assert_eq!(&recompose(&limbs), m);
    }

    #[test]
    fn decompose_zero_is_all_zero() {
        assert_eq!(decompose(&Scalar::ZERO), [0u32; LIMB_COUNT]);
        assert_eq!(recompose(&[0u32; LIMB_COUNT]), Scalar::ZERO);
    }

    #[test]
    fn roundtrip_boundary_values() {
        let n_minus_one = Scalar::ZERO - Scalar::ONE;
        for m in [
            Scalar::ZERO,
            Scalar::ONE,
            Scalar::from(LIMB_MODULUS - 1), // largest single limb
            Scalar::from(LIMB_MODULUS),     // first carry into limb 1
            n_minus_one,                    // largest in-range scalar
        ] {
            assert_roundtrip(&m);
        }
    }

    #[test]
    fn top_limb_high_bits_are_zero_for_n_minus_one() {
        // L·ℓ − 256 = 8 high bits of the top limb must be zero in range.
        let n_minus_one = Scalar::ZERO - Scalar::ONE;
        let top = u64::from(decompose(&n_minus_one)[LIMB_COUNT - 1]);
        let used_bits = 256 - (LIMB_COUNT - 1) * LIMB_BITS as usize;
        assert!(top < (1u64 << used_bits));
    }

    #[test]
    fn roundtrip_random_scalars() {
        // Fixed seed keeps the test deterministic while sweeping the space.
        let mut rng = StdRng::seed_from_u64(0xEC_5E_6E_01);
        for _ in 0..2000 {
            assert_roundtrip(&Scalar::random(&mut rng));
        }
    }
}
