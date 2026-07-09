//! Hiding Pedersen commitments `Com = v·G + s·H`.
//!
//! Used by both the range proof (W2e) and the carry chain (W2f) to commit to
//! a limb value `v` under a blinding scalar `s`, with `G` and the NUMS
//! generator `H` ([`crate::generators`]). Because `log_G(H)` is unknown, the
//! commitment is computationally binding and perfectly hiding: for any `v`,
//! a uniform `s` makes `Com` uniform over the group, revealing nothing about
//! `v`.
//!
//! Commitments are additively homomorphic — `Com(v₁,s₁) + Com(v₂,s₂) =
//! Com(v₁+v₂, s₁+s₂)` — which is exactly what lets the carry chain prove the
//! exact-integer relation `m + m̄ = n − 1` over the limb commitments.

use crate::generators::{g, h};
#[cfg(test)]
use k256::elliptic_curve::Field;
use k256::{ProjectivePoint, Scalar};
#[cfg(test)]
use rand_core::{CryptoRng, RngCore};
use zeroize::Zeroize;

/// A Pedersen commitment `Com = v·G + s·H` together with — for the prover —
/// the value and blinding it opens to. Only [`Commitment::point`] is public;
/// `value`/`blinding` are the prover's opening witness.
pub struct Commitment {
    /// The public commitment point `v·G + s·H`.
    pub point: ProjectivePoint,
    /// The committed value `v` (prover witness).
    pub value: Scalar,
    /// The blinding scalar `s` (prover witness).
    pub blinding: Scalar,
}

impl Drop for Commitment {
    fn drop(&mut self) {
        self.value.zeroize();
        self.blinding.zeroize();
    }
}

impl core::fmt::Debug for Commitment {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Commitment").finish_non_exhaustive()
    }
}

impl Commitment {
    /// Commit to `value` under a freshly sampled blinding scalar.
    #[cfg(test)]
    pub fn commit<R: RngCore + CryptoRng>(value: Scalar, rng: &mut R) -> Self {
        Self::with_blinding(value, Scalar::random(rng))
    }

    /// Commit to `value` under a caller-supplied blinding (used when the
    /// blinding must satisfy a homomorphic relation, e.g. the carry chain).
    #[must_use]
    pub fn with_blinding(value: Scalar, blinding: Scalar) -> Self {
        let point = Self::point_of(value, blinding);
        Self {
            point,
            value,
            blinding,
        }
    }

    /// Recompute the commitment point from a public value and blinding — the
    /// verifier's check that a claimed opening matches a point.
    #[must_use]
    pub fn point_of(value: Scalar, blinding: Scalar) -> ProjectivePoint {
        g() * value + h() * blinding
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn scalar(v: u64) -> Scalar {
        Scalar::from(v)
    }

    #[test]
    fn opening_matches_point() {
        let mut rng = StdRng::seed_from_u64(0x9E_DE_55_01);
        let c = Commitment::commit(scalar(42), &mut rng);
        assert_eq!(Commitment::point_of(c.value, c.blinding), c.point);
    }

    #[test]
    fn additively_homomorphic() {
        // Com(v1,s1) + Com(v2,s2) == Com(v1+v2, s1+s2): the property the
        // carry chain relies on.
        let mut rng = StdRng::seed_from_u64(0x9E_DE_55_02);
        let a = Commitment::commit(scalar(7), &mut rng);
        let b = Commitment::commit(scalar(35), &mut rng);
        let sum = Commitment::point_of(a.value + b.value, a.blinding + b.blinding);
        assert_eq!(a.point + b.point, sum);
    }

    #[test]
    fn hiding_blinding_changes_point() {
        let a = Commitment::with_blinding(scalar(1), scalar(100));
        let b = Commitment::with_blinding(scalar(1), scalar(200));
        assert_ne!(a.point, b.point);
    }

    #[test]
    fn zero_value_is_pure_blinding() {
        // Com(0, s) = s·H, never the identity for s != 0 (H is non-identity).
        let c = Commitment::with_blinding(Scalar::ZERO, scalar(5));
        assert_eq!(c.point, h() * scalar(5));
        assert_ne!(c.point, ProjectivePoint::IDENTITY);
    }

    #[test]
    fn distinct_values_distinct_commitments_same_blinding() {
        let s = scalar(99);
        let a = Commitment::with_blinding(scalar(10), s);
        let b = Commitment::with_blinding(scalar(11), s);
        // Differ by exactly G (binding intuition: the value term is in play).
        assert_eq!(a.point + g(), b.point);
        assert_ne!(a.point, b.point);
    }

    #[test]
    fn debug_redacts_opening_witness() {
        let c = Commitment::with_blinding(scalar(0x42), scalar(0x77));
        let debug = format!("{c:?}");
        assert!(
            !debug.contains("value") && !debug.contains("blinding"),
            "Debug output must not expose witness fields: {debug}"
        );
    }

    #[test]
    fn opening_witness_container_is_not_cheaply_duplicable() {
        static_assertions::assert_not_impl_any!(Commitment: Clone, Copy);
    }
}
