//! Strong Fiat–Shamir transcript for ec-segve-v1.
//!
//! This is the single most soundness-critical primitive in the crate: the
//! whole NIZK is only as sound as the challenge derivation. The contract is
//! pinned normatively in `docs/design/ec-segve-soundness.md` §1–§3; this
//! module implements that contract.
//!
//! Rules enforced here:
//! - **Own domain.** Every transcript is seeded with the domain string
//!   [`DOMAIN`], distinct from the CL-HSMq/BCL24 domain, so a challenge can
//!   never be replayed across protocols.
//! - **Length-prefixed absorption.** Every absorbed field is written as a
//!   4-byte big-endian length followed by its bytes, so no two distinct field
//!   sequences can hash to the same transcript (concatenation is
//!   unambiguous). Lists are a raw 4-byte big-endian count followed by each
//!   element absorbed length-prefixed, in index order.
//! - **Canonical encodings.** Points are absorbed as their re-serialized
//!   33-byte SEC1 ([`crate::codec::encode_point`]), never received wire bytes,
//!   so two byte-strings for the same group element cannot fork the challenge.
//!   Scalars are absorbed as 32-byte big-endian.
//! - **Labeled multi-squeeze ratchet (soundness doc §2).** The BP++ rounds
//!   need several challenges from one linear transcript.
//!   [`Transcript::squeeze`] derives a labeled interior challenge and ratchets
//!   the state so every later challenge depends on all earlier absorptions,
//!   labels, AND challenges; [`Transcript::finalize`] consumes the transcript
//!   for the LAST squeeze (`sigma.x`), so the final-squeeze rule — nothing
//!   prover-chosen absorbed after it — is enforced by the type system, exactly
//!   as the single-squeeze [`Transcript::challenge`] enforces it for the
//!   separate single-challenge transcripts (the contribute-DLEQ).
//! - **Full-width challenge.** Every challenge is an entire 32-byte SHA-256
//!   digest reduced mod `n` (≈256-bit space), NOT bcl24's 160-bit truncation.

use crate::codec::encode_point;
use k256::elliptic_curve::bigint::U256;
use k256::elliptic_curve::ops::Reduce;
use k256::{FieldBytes, ProjectivePoint, Scalar};
use sha2::{Digest, Sha256};

/// The ec-segve FS domain string. Distinct from any other protocol's domain
/// so challenges cannot be replayed across protocols.
pub const DOMAIN: &[u8] = b"ve-capsule.ec-segve.secp256k1.v1";

/// Ratchet-chaining domain (soundness doc §2). Distinct from [`DOMAIN`] and
/// [`CHALLENGE_DOMAIN`] so a chaining value can never alias a challenge value
/// or a transcript seed.
pub const RATCHET_DOMAIN: &[u8] = b"ve-capsule.ec-segve.secp256k1.v1.ratchet";

/// Challenge-derivation domain (soundness doc §2).
pub const CHALLENGE_DOMAIN: &[u8] = b"ve-capsule.ec-segve.secp256k1.v1.challenge";

/// The canonical 4-byte big-endian length prefix for a framed field, shared by
/// [`Transcript::absorb_bytes`] and any `Vec`-built binding that must frame
/// identically (e.g. a composite context binding) so the two cannot drift.
/// Saturating — every field is far below the four-byte ceiling.
pub fn length_prefix(len: usize) -> [u8; 4] {
    u32::try_from(len).unwrap_or(u32::MAX).to_be_bytes()
}

/// Append `field` to `out` length-prefixed ([`length_prefix`] ‖ bytes), so a
/// `Vec`-built payload (a composite binding, a contribution binding) frames a
/// sequence of fields unambiguously and identically to the transcript.
pub fn push_framed(out: &mut Vec<u8>, field: &[u8]) {
    out.extend_from_slice(&length_prefix(field.len()));
    out.extend_from_slice(field);
}

/// A running strong-Fiat–Shamir transcript over `SHA-256`.
///
/// Construct with [`Transcript::new`] (which seeds [`DOMAIN`]), absorb the
/// statement and prover messages with the typed `absorb_*` methods in the
/// order fixed by the soundness spec, then consume it with
/// [`Transcript::challenge`].
pub struct Transcript {
    hasher: Sha256,
}

impl Transcript {
    /// Start a transcript, seeding the protocol [`DOMAIN`].
    #[must_use]
    pub fn new() -> Self {
        let mut t = Self {
            hasher: Sha256::new(),
        };
        t.absorb_bytes(DOMAIN);
        t
    }

    /// Absorb a raw field, length-prefixed (4-byte big-endian length ‖ bytes).
    ///
    /// Every absorbed field flows through here so the framing is uniform. All
    /// fields in this protocol are tiny (points 33 B, scalars 32 B, headers a
    /// few bytes); the saturating length is unreachable in practice and only
    /// keeps the method total instead of panicking.
    pub fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.hasher.update(length_prefix(bytes.len()));
        self.hasher.update(bytes);
    }

    /// Absorb a curve point as its canonical 33-byte SEC1 encoding.
    pub fn absorb_point(&mut self, point: &ProjectivePoint) {
        self.absorb_bytes(&encode_point(point));
    }

    /// Absorb a scalar as 32-byte big-endian.
    pub fn absorb_scalar(&mut self, scalar: &Scalar) {
        self.absorb_bytes(&scalar.to_bytes());
    }

    /// Absorb a `u16` header field (big-endian), length-prefixed.
    pub fn absorb_u16(&mut self, value: u16) {
        self.absorb_bytes(&value.to_be_bytes());
    }

    /// Absorb a `u8` header field, length-prefixed.
    pub fn absorb_u8(&mut self, value: u8) {
        self.absorb_bytes(&[value]);
    }

    /// Absorb a list-length frame: a raw 4-byte big-endian count (NOT
    /// length-prefixed), to be followed by `count` elements each absorbed with
    /// a typed `absorb_*` method, in index order.
    ///
    /// A zero count is byte-identical to `absorb_bytes(&[])` (both emit four
    /// zero bytes). The fixed absorption schedules keep a list frame and an
    /// empty byte field from ever occupying the same transcript position —
    /// preserve that invariant when extending a schedule.
    pub fn absorb_list_len(&mut self, count: usize) {
        let count = u32::try_from(count).unwrap_or(u32::MAX);
        self.hasher.update(count.to_be_bytes());
    }

    /// Consume the transcript and squeeze the challenge: the full 32-byte
    /// SHA-256 digest reduced mod `n`.
    ///
    /// This is the single-challenge form, used by the transcripts that squeeze
    /// exactly once (the contribute-DLEQ, §6). The capsule's multi-squeeze
    /// schedule uses [`Transcript::squeeze`] + [`Transcript::finalize`].
    #[must_use]
    pub fn challenge(self) -> Scalar {
        let digest = self.hasher.finalize();
        let mut bytes = FieldBytes::default();
        bytes.copy_from_slice(&digest);
        <Scalar as Reduce<U256>>::reduce_bytes(&bytes)
    }

    /// Squeeze a labeled interior challenge and ratchet the state
    /// (soundness doc §2, byte-pinned):
    ///
    /// ```text
    /// d := SHA256-finalize(running state)
    /// c := SHA256( LP(CHALLENGE_DOMAIN) ‖ LP(label) ‖ LP(d) )  mod n
    /// running state := SHA256 absorbing
    ///                  LP(RATCHET_DOMAIN) ‖ LP(label) ‖ LP(d) ‖ LP(c)
    /// ```
    ///
    /// Every later absorption and squeeze therefore depends on this label and
    /// challenge; the challenge/ratchet domain split keeps a challenge value
    /// from ever aliasing a chaining value.
    #[must_use]
    pub fn squeeze(&mut self, label: &[u8]) -> Scalar {
        let d = std::mem::take(&mut self.hasher).finalize();
        let c = Self::derive_labeled(label, &d);
        self.absorb_bytes(RATCHET_DOMAIN);
        self.absorb_bytes(label);
        self.absorb_bytes(&d);
        self.absorb_bytes(&c.to_bytes());
        c
    }

    /// Squeeze the FINAL labeled challenge, consuming the transcript — the
    /// type-level final-squeeze rule (soundness doc §2): nothing prover-chosen
    /// can be absorbed after the last squeeze, because the transcript no
    /// longer exists.
    #[must_use]
    pub fn finalize(self, label: &[u8]) -> Scalar {
        let d = self.hasher.finalize();
        Self::derive_labeled(label, &d)
    }

    /// `c = SHA256( LP(CHALLENGE_DOMAIN) ‖ LP(label) ‖ LP(d) ) mod n` — the
    /// shared derivation of [`Transcript::squeeze`] and
    /// [`Transcript::finalize`].
    fn derive_labeled(label: &[u8], d: &[u8]) -> Scalar {
        let mut h = Sha256::new();
        for field in [CHALLENGE_DOMAIN, label, d] {
            h.update(length_prefix(field.len()));
            h.update(field);
        }
        let digest = h.finalize();
        let mut bytes = FieldBytes::default();
        bytes.copy_from_slice(&digest);
        <Scalar as Reduce<U256>>::reduce_bytes(&bytes)
    }
}

impl Default for Transcript {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k256::elliptic_curve::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn empty_transcript_is_deterministic() {
        // Two fresh (domain-only) transcripts agree.
        assert_eq!(Transcript::new().challenge(), Transcript::new().challenge());
    }

    #[test]
    fn same_absorptions_same_challenge() {
        let build = || {
            let mut t = Transcript::new();
            t.absorb_u8(1);
            t.absorb_u16(256);
            t.absorb_scalar(&Scalar::from(42u64));
            t.absorb_point(&ProjectivePoint::GENERATOR);
            t.challenge()
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn order_changes_challenge() {
        let mut a = Transcript::new();
        a.absorb_u8(1);
        a.absorb_u8(2);
        let mut b = Transcript::new();
        b.absorb_u8(2);
        b.absorb_u8(1);
        assert_ne!(a.challenge(), b.challenge());
    }

    #[test]
    fn length_prefix_prevents_concatenation_ambiguity() {
        // ("ab","c") and ("a","bc") share the concatenation "abc"; the length
        // prefix must keep their challenges distinct.
        let mut a = Transcript::new();
        a.absorb_bytes(b"ab");
        a.absorb_bytes(b"c");
        let mut b = Transcript::new();
        b.absorb_bytes(b"a");
        b.absorb_bytes(b"bc");
        assert_ne!(a.challenge(), b.challenge());
    }

    #[test]
    fn domain_is_seeded() {
        // A transcript that absorbs nothing still differs from a bare SHA-256
        // reduce of the empty string — i.e. the domain was mixed in.
        let with_domain = Transcript::new().challenge();
        let mut bare = Sha256::new();
        bare.update([]);
        let digest = bare.finalize();
        let mut fb = FieldBytes::default();
        fb.copy_from_slice(&digest);
        let no_domain = <Scalar as Reduce<U256>>::reduce_bytes(&fb);
        assert_ne!(with_domain, no_domain);
    }

    #[test]
    fn distinct_points_distinct_challenges() {
        let mut rng = StdRng::seed_from_u64(0xF5_7A_2D_01);
        let p = ProjectivePoint::GENERATOR * Scalar::random(&mut rng);
        let q = ProjectivePoint::GENERATOR * Scalar::random(&mut rng);
        let chal = |pt: &ProjectivePoint| {
            let mut t = Transcript::new();
            t.absorb_point(pt);
            t.challenge()
        };
        assert_ne!(chal(&p), chal(&q));
    }

    #[test]
    fn challenge_kat() {
        // Locks the exact transcript byte layout + challenge reduction. Any
        // change to the domain, framing, or reduction must update this vector
        // deliberately.
        let mut t = Transcript::new();
        t.absorb_u8(1);
        t.absorb_u16(256);
        t.absorb_scalar(&Scalar::from(7u64));
        t.absorb_point(&ProjectivePoint::GENERATOR);
        t.absorb_list_len(2);
        t.absorb_scalar(&Scalar::from(0u64));
        t.absorb_scalar(&Scalar::ONE);
        let hex: String = t
            .challenge()
            .to_bytes()
            .iter()
            .fold(String::new(), |mut s, b| {
                use std::fmt::Write as _;
                let _ = write!(s, "{b:02x}");
                s
            });
        assert_eq!(hex, KNOWN_CHALLENGE);
    }

    #[test]
    fn reordering_full_sequence_changes_challenge() {
        // Negative KAT (soundness doc §3, W2d note): reordering any two absorbed
        // items must fork the challenge across the full mixed-type sequence, not
        // just the minimal u8 case in `order_changes_challenge`.
        let challenge = |swap_tail: bool| {
            let mut t = Transcript::new();
            t.absorb_u8(1);
            t.absorb_u16(256);
            t.absorb_scalar(&Scalar::from(7u64));
            t.absorb_point(&ProjectivePoint::GENERATOR);
            t.absorb_list_len(2);
            let (first, second) = if swap_tail {
                (Scalar::ONE, Scalar::from(0u64))
            } else {
                (Scalar::from(0u64), Scalar::ONE)
            };
            t.absorb_scalar(&first);
            t.absorb_scalar(&second);
            t.challenge()
        };
        assert_ne!(challenge(false), challenge(true));
    }

    const KNOWN_CHALLENGE: &str =
        "8b048584002b09b82eac73cd3bbcd73b00f207697f1c9cc28dc642f35ed38919";

    fn hex(s: &Scalar) -> String {
        s.to_bytes().iter().fold(String::new(), |mut acc, b| {
            use std::fmt::Write as _;
            let _ = write!(acc, "{b:02x}");
            acc
        })
    }

    #[test]
    fn ratchet_vector_kat() {
        // Pins the §2 ratchet byte layout: domains, label framing, chaining,
        // and the finalize derivation. Any change to RATCHET_DOMAIN /
        // CHALLENGE_DOMAIN / framing must update these deliberately.
        let mut t = Transcript::new();
        t.absorb_u8(7);
        let alpha = t.squeeze(b"alpha");
        t.absorb_point(&ProjectivePoint::GENERATOR);
        let rho = t.squeeze(b"rho");
        let lambda = t.squeeze(b"lambda");
        let x = t.finalize(b"sigma.x");
        assert_eq!(
            hex(&alpha),
            "4882cb5f893e8161f3006f4ba9b62a24b8f2f8edc5130d655a021b7cacf791f0"
        );
        assert_eq!(
            hex(&rho),
            "78c5e105c3bff562ebd235df15a678f5509b3f31a34c96da4f61ae95b7878512"
        );
        assert_eq!(
            hex(&lambda),
            "c92cb14eabc0199ce64519cfa6ece9e087a1c03c5100832dc3f1523c22b6b0c9"
        );
        assert_eq!(
            hex(&x),
            "5fc05805296d2187d67b5985ddd347acfec83394d62664837625503c9adbef4d"
        );
    }

    #[test]
    fn squeeze_and_finalize_agree_on_the_same_state() {
        // finalize is the consuming form of the same labeled derivation.
        let build = || {
            let mut t = Transcript::new();
            t.absorb_u8(3);
            t
        };
        let mut a = build();
        assert_eq!(a.squeeze(b"tau"), build().finalize(b"tau"));
    }

    #[test]
    fn squeeze_label_forks_challenge() {
        let build = || {
            let mut t = Transcript::new();
            t.absorb_u8(3);
            t
        };
        assert_ne!(build().squeeze(b"rho"), build().squeeze(b"lambda"));
    }

    #[test]
    fn ratchet_chains_earlier_absorptions_into_later_squeezes() {
        // Frozen-Heart guard: a field absorbed before an EARLIER squeeze must
        // fork every LATER challenge, even with identical interim absorptions.
        let seq = |seed: u8| {
            let mut t = Transcript::new();
            t.absorb_u8(seed);
            let _ = t.squeeze(b"alpha");
            t.absorb_u8(0xAA);
            t.squeeze(b"rho")
        };
        assert_ne!(seq(1), seq(2));
    }

    #[test]
    fn sequential_squeeze_order_is_load_bearing() {
        // rho->lambda in sequence is NOT lambda->rho: each squeeze ratchets,
        // so the pinned order produces different values than a swapped order.
        let pair = |first: &[u8], second: &[u8]| {
            let mut t = Transcript::new();
            (t.squeeze(first), t.squeeze(second))
        };
        let (r1, l1) = pair(b"rho", b"lambda");
        let (l2, r2) = pair(b"lambda", b"rho");
        assert_ne!(r1, r2);
        assert_ne!(l1, l2);
    }

    #[test]
    fn indexed_gamma_labels_do_not_collide() {
        // gamma ‖ BE16(1) vs gamma ‖ BE16(256): framing keeps every indexed
        // label distinct.
        let squeeze_with = |i: u16| {
            let mut t = Transcript::new();
            let mut label = b"gamma".to_vec();
            label.extend_from_slice(&i.to_be_bytes());
            t.squeeze(&label)
        };
        assert_ne!(squeeze_with(1), squeeze_with(256));
    }

    #[test]
    fn single_squeeze_challenge_differs_from_labeled_derivation() {
        // The legacy single-squeeze (DLEQ) path and the labeled derivation use
        // different domains — same absorptions can never alias.
        let build = || {
            let mut t = Transcript::new();
            t.absorb_u8(9);
            t
        };
        assert_ne!(build().challenge(), build().finalize(b"sigma.x"));
    }
}
