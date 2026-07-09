//! Frozen segmentation parameters for ec-segve-v1.
//!
//! The shipped tuple `(ℓ, L, d, D)`: `ℓ` limb bits, `L` limbs, BP++ digit
//! base `d`, `D` base-`d` digits per limb. `ℓ` and `L` were chosen by the
//! parameter-freeze benchmark; [`Params::FROZEN`] keeps that winner, with
//! the digit shape refit for the aggregated BP++ range proof.
//!
//! # Parameter freeze: `ℓ = 24` (`L24_D16`)
//!
//! The constrained-device recovery critical path is dominated by the per-limb BSGS
//! search, so the gate is: pick the **largest `ℓ`** whose recovery path is
//! comfortably under ~1 s on a reference desktop (a primitive that cannot hit
//! that budget there is too slow for the constrained target). Measured
//! naive-BSGS totals (baby table built once + `L` worst-case giant searches for
//! `v = 2^ℓ − 1`): `ℓ = 16` ≈ 20 ms, `ℓ = 24` ≈ 227 ms, `ℓ = 32` ≈ 2.7 s.
//! `ℓ = 32` blows the gate; `ℓ = 24` clears it with real margin and is the
//! largest that does, so it is frozen.
//!
//! Caveat: the bench BSGS is the NAIVE form (fresh affine conversion +
//! `Vec`-keyed hash per step, no Montgomery batch-inversion), so the
//! `ℓ = 32` rejection is against un-optimized BSGS; a reader optimizing
//! BSGS later must re-run this gate before reconsidering a larger `ℓ`.
//! As built, release-mode `verify` measures ~143 ms and `open` ~13 ms at
//! the frozen shape — comfortably inside the gate.

use crate::transcript::push_framed;
use sha2::{Digest, Sha256};

/// A candidate / frozen segmentation parameter tuple.
///
/// Invariants (checked by [`Params::is_valid`] and the candidate const
/// asserts): `d^D == 2^ℓ` (the BP++ base-`d` digits cover a limb exactly —
/// the decode-stage BSGS window gate) and `L·ℓ >= 256` (limbs cover a full
/// secp256k1 scalar).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Params {
    /// Limb bit-width `ℓ`.
    pub limb_bits: u32,
    /// Limb count `L`.
    pub limb_count: u32,
    /// BP++ digit base `d`.
    pub digit_base: u32,
    /// Base-`d` digits per limb `D`.
    pub digits_per_limb: u32,
}

impl Params {
    /// The frozen v1 tuple: `ℓ = 24` ([`Self::L24_D16`]), the largest-`ℓ`
    /// candidate with comfortable margin under the ~1 s constrained-device gate
    /// (critical path ~271 ms; see the module-level Wb freeze table).
    /// Downstream chunks hard-code their segmentation via this.
    pub const FROZEN: Self = Self::L24_D16;

    /// Candidate `ℓ = 16`, `L = 16`, `d = 16`, `D = 4` (`16^4 = 2^16`).
    pub const L16_D16: Self = Self {
        limb_bits: 16,
        limb_count: 16,
        digit_base: 16,
        digits_per_limb: 4,
    };

    /// Candidate `ℓ = 24`, `L = 11`, `d = 16`, `D = 6` (`16^6 = 2^24`) —
    /// the BP++ shared-multiplicity digit shape (frozen-shape table, §4.1).
    pub const L24_D16: Self = Self {
        limb_bits: 24,
        limb_count: 11,
        digit_base: 16,
        digits_per_limb: 6,
    };

    /// Candidate `ℓ = 32`, `L = 8`, `d = 16`, `D = 8` (`16^8 = 2^32`).
    pub const L32_D16: Self = Self {
        limb_bits: 32,
        limb_count: 8,
        digit_base: 16,
        digits_per_limb: 8,
    };

    /// The candidates the Wb spike benches; `FROZEN` MUST be one of these.
    pub const CANDIDATES: [Self; 3] = [Self::L16_D16, Self::L24_D16, Self::L32_D16];

    /// Checks the two structural invariants: `d^D == 2^ℓ` and
    /// `L·ℓ >= 256`. Total over arbitrary `Params` — a degenerate tuple
    /// (e.g. `d^D` past `u128`, or `ℓ >= 128`) returns `false` rather
    /// than panicking via `checked_*`, so `is_valid` is a real predicate.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        let (Some(bd), Some(two_pow_l)) = (
            (self.digit_base as u128).checked_pow(self.digits_per_limb),
            1u128.checked_shl(self.limb_bits),
        ) else {
            return false;
        };
        bd == two_pow_l && (self.limb_count as u64) * (self.limb_bits as u64) >= 256
    }

    /// Per-limb modulus `2^ℓ`; every limb value lies in `[0, limb_modulus())`.
    /// Valid only for `ℓ < 64` (all candidates satisfy this).
    #[must_use]
    pub const fn limb_modulus(&self) -> u64 {
        1u64 << self.limb_bits
    }

    /// Baby-step table size for per-limb BSGS: `2^ceil(ℓ/2)`, i.e.
    /// `ceil(sqrt(2^ℓ))` baby steps so the table fully covers `[0, 2^ℓ)`
    /// even for odd `ℓ` (all current candidates have even `ℓ`).
    #[must_use]
    pub const fn bsgs_table_size(&self) -> u64 {
        1u64 << self.limb_bits.div_ceil(2)
    }

    /// A stable 32-byte identifier for this parameter tuple: the SHA-256 of a
    /// domain-separated, length-prefixed encoding of `(ℓ, L, b, D)`. A recovery
    /// artifact pins the `id()` of the params it was sealed under, and a verifier
    /// rejects any capsule whose params id differs — the analog of CL-HSMq's
    /// `Ciphertext::v1_params_id()`.
    #[must_use]
    pub fn id(&self) -> [u8; 32] {
        let mut framed = Vec::new();
        push_framed(&mut framed, PARAMS_ID_DOMAIN);
        push_framed(&mut framed, &self.limb_bits.to_be_bytes());
        push_framed(&mut framed, &self.limb_count.to_be_bytes());
        push_framed(&mut framed, &self.digit_base.to_be_bytes());
        push_framed(&mut framed, &self.digits_per_limb.to_be_bytes());
        Sha256::digest(&framed).into()
    }
}

/// Domain tag for [`Params::id`]. Distinct from the FS transcript domain so a
/// params id can never collide with a challenge derivation. Bump on any change
/// to the params layout or this encoding. (v1 was redefined in place on
/// 2026-06-10 — nothing had shipped — so the bump rule starts applying at
/// first release.)
const PARAMS_ID_DOMAIN: &[u8] = b"ve-capsule.params-id.v1";

// Compile-time guard: every candidate — and the frozen tuple — satisfies
// `b^D == 2^ℓ` and `L·ℓ >= 256`. (`FROZEN ∈ CANDIDATES` is the
// `frozen_is_a_candidate` test.)
const _: () = assert!(Params::L16_D16.is_valid());
const _: () = assert!(Params::L24_D16.is_valid());
const _: () = assert!(Params::L32_D16.is_valid());
const _: () = assert!(Params::FROZEN.is_valid());

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

    use super::*;

    #[test]
    fn all_candidates_valid() {
        for c in Params::CANDIDATES {
            assert!(
                c.is_valid(),
                "candidate {c:?} violates b^D==2^l or L*l>=256"
            );
        }
    }

    #[test]
    fn frozen_is_a_candidate() {
        assert!(
            Params::CANDIDATES.contains(&Params::FROZEN),
            "FROZEN must be one of the benched candidates"
        );
    }

    #[test]
    fn rejects_over_cover_tuple() {
        // d=8, D=6 over ℓ=16 gives 8^6 = 2^18 != 2^16 → must be invalid.
        let bad = Params {
            limb_bits: 16,
            limb_count: 16,
            digit_base: 8,
            digits_per_limb: 6,
        };
        assert!(!bad.is_valid());
    }

    #[test]
    fn rejects_insufficient_coverage() {
        // L*ℓ = 8*24 = 192 < 256 → must be invalid even with d^D==2^ℓ.
        let bad = Params {
            limb_bits: 24,
            limb_count: 8,
            digit_base: 16,
            digits_per_limb: 6,
        };
        assert!(!bad.is_valid());
    }

    #[test]
    fn params_id_is_deterministic_and_distinct() {
        assert_eq!(Params::FROZEN.id(), Params::FROZEN.id());
        assert_eq!(
            Params::FROZEN.id(),
            Params::L24_D16.id(),
            "FROZEN == L24_D16 ⇒ same id"
        );
        let ids = Params::CANDIDATES.map(|c| c.id());
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                assert_ne!(a, b, "distinct param tuples must have distinct ids");
            }
        }
    }

    #[test]
    fn frozen_params_id_golden() {
        // Pin the frozen tuple's id so an accidental params/domain/encoding
        // change is caught. Recompute deliberately if the params are refrozen.
        let hex = Params::FROZEN.id().iter().fold(String::new(), |mut s, b| {
            use std::fmt::Write;
            let _ = write!(s, "{b:02x}");
            s
        });
        assert_eq!(
            hex,
            "e04bf2759873bd2ffb93101e5b958d443059d4752751adda5666f70226e171ab"
        );
    }

    #[test]
    fn bsgs_table_size_is_half_width() {
        assert_eq!(Params::L24_D16.bsgs_table_size(), 1u64 << 12);
        assert_eq!(Params::L16_D16.bsgs_table_size(), 1u64 << 8);
        assert_eq!(Params::L32_D16.bsgs_table_size(), 1u64 << 16);
    }

    #[test]
    fn is_valid_returns_false_instead_of_panicking_on_degenerate() {
        // b^D overflows u128 (16^64 = 2^256) → false, not a panic.
        assert!(
            !Params {
                limb_bits: 24,
                limb_count: 11,
                digit_base: 16,
                digits_per_limb: 64,
            }
            .is_valid()
        );
        // 1 << 128 overflows the shift → false, not a panic.
        assert!(
            !Params {
                limb_bits: 128,
                limb_count: 2,
                digit_base: 2,
                digits_per_limb: 128,
            }
            .is_valid()
        );
    }
}
