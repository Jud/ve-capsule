//! The two independent generators `G` and `H` for ec-segve-v1.
//!
//! `G` is the secp256k1 base point. `H` is a second generator used for the
//! hiding term of the Pedersen commitments `Com_k = v_k·G + s_k·H`. Hiding
//! requires that **no party knows `log_G(H)`**: if `log_G(H) = d` were known,
//! `Com_k = (v_k + d·s_k)·G` collapses to a single unknown and the commitment
//! stops hiding `v_k`. `H` is therefore derived by RFC 9380 hash-to-curve
//! (random-oracle suite `secp256k1_XMD:SHA-256_SSWU_RO_`) from a fixed domain
//! string — a nothing-up-my-sleeve construction whose discrete log w.r.t. `G`
//! is unknown to everyone, including this crate's authors.
//!
//! `H` is recomputed from these fixed constants by every party; it is never
//! transmitted, so there is no wire value a malicious prover could substitute
//! (audit SA-2026-307). The derivation is fully pinned by [`H_DST`] and
//! [`H_MSG`] and locked by a known-answer test (audit SA-2026-322).

use k256::elliptic_curve::hash2curve::{ExpandMsgXmd, GroupDigest};
use k256::{ProjectivePoint, Secp256k1};
use sha2::Sha256;
use std::sync::LazyLock;

/// RFC 9380 domain-separation tag for the `H` derivation.
///
/// Encodes the protocol, the role of the point, and the hash-to-curve suite
/// so this DST cannot collide with any other hash-to-curve use in the system.
pub const H_DST: &[u8] = b"ec-segve-secp256k1-v1:NUMS:H:RFC9380:secp256k1_XMD:SHA-256_SSWU_RO_";

/// The message hashed to derive `H`. Fixed and arbitrary; the unknown-dlog
/// property comes from the random-oracle map, not from the message content.
pub const H_MSG: &[u8] = b"hiding-generator";

/// `H` is computed once and cached. `hash_from_bytes` is deterministic, so the
/// cached value equals a fresh derivation; caching only avoids repeating the
/// hash-to-curve map on every commitment.
static H_CACHED: LazyLock<ProjectivePoint> = LazyLock::new(derive_h);

/// Derive the hiding generator `H` via RFC 9380 hash-to-curve.
///
/// `hash_from_bytes` with `ExpandMsgXmd` only errors when the DST is empty or
/// exceeds 255 bytes; [`H_DST`] is a fixed 67-byte string, so the error arm is
/// statically unreachable. A const assertion pins the DST length so a future
/// edit that violates the RFC 9380 bound fails to compile rather than reaching
/// the `expect`.
fn derive_h() -> ProjectivePoint {
    const _: () = assert!(H_DST.len() <= 255 && !H_DST.is_empty());
    #[allow(clippy::expect_used)]
    Secp256k1::hash_from_bytes::<ExpandMsgXmd<Sha256>>(&[H_MSG], &[H_DST])
        .expect("fixed 67-byte DST satisfies the RFC 9380 1..=255 length bound")
}

/// The secp256k1 base point `G`.
#[must_use]
pub const fn g() -> ProjectivePoint {
    ProjectivePoint::GENERATOR
}

/// The hiding generator `H` (NUMS, unknown `log_G(H)`).
#[must_use]
pub fn h() -> ProjectivePoint {
    *H_CACHED
}

/// RFC 9380 DST for the BP++ n-side ("norm") vector generators `g⃗[i]`
/// (soundness doc §0, "Vector generators").
pub const GVEC_DST: &[u8] =
    b"ec-segve-secp256k1-v1:NUMS:GVEC:RFC9380:secp256k1_XMD:SHA-256_SSWU_RO_";

/// RFC 9380 DST for the BP++ l-side ("linear") vector generators `h⃗[i]`,
/// `i ≥ 1` (`h⃗[0]` is [`h`] itself — the capsule's Pedersen commitments are
/// then valid BP++ value commitments with no re-commitment).
pub const HVEC_DST: &[u8] =
    b"ec-segve-secp256k1-v1:NUMS:HVEC:RFC9380:secp256k1_XMD:SHA-256_SSWU_RO_";

/// Domain tag for [`generators_digest`]. Distinct from every transcript and
/// params-id domain in the crate.
const GENERATORS_DIGEST_DOMAIN: &[u8] = b"ve-capsule.bppp.generators-digest.v1";

/// Derive one vector generator by role DST and index: hash-to-curve over the
/// message `prefix ‖ BE16(index)`. Deterministic; never read from the wire.
fn derive_vector_generator(dst: &'static [u8], prefix: &[u8], index: u16) -> ProjectivePoint {
    const _: () = assert!(GVEC_DST.len() <= 255 && HVEC_DST.len() <= 255);
    let mut msg = Vec::with_capacity(prefix.len() + 2);
    msg.extend_from_slice(prefix);
    msg.extend_from_slice(&index.to_be_bytes());
    #[allow(clippy::expect_used)]
    Secp256k1::hash_from_bytes::<ExpandMsgXmd<Sha256>>(&[&msg], &[dst])
        .expect("fixed sub-255-byte DSTs satisfy the RFC 9380 length bound")
}

/// The BP++ n-side generator vector `g⃗[0..count)`.
///
/// The production `count` is a frozen circuit-shape constant; it is taken as
/// a parameter so KATs and the norm-argument tests can exercise small shapes.
#[must_use]
pub fn gvec(count: u16) -> Vec<ProjectivePoint> {
    (0..count)
        .map(|i| derive_vector_generator(GVEC_DST, b"vector-generator-g-", i))
        .collect()
}

/// The BP++ l-side generator vector `h⃗[0..count)`, with `h⃗[0] = H`.
#[must_use]
pub fn hvec(count: u16) -> Vec<ProjectivePoint> {
    (0..count)
        .map(|i| {
            if i == 0 {
                h()
            } else {
                derive_vector_generator(HVEC_DST, b"vector-generator-h-", i)
            }
        })
        .collect()
}

/// The 32-byte digest pinning a generator set (soundness doc §0 / §3 item
/// 13a): SHA-256 over the domain tag and both vectors, every field framed
/// exactly like the transcript (`push_framed`), lists as a raw 4-byte BE
/// count then framed elements.
#[must_use]
pub fn generators_digest(g_vec: &[ProjectivePoint], h_vec: &[ProjectivePoint]) -> [u8; 32] {
    use crate::codec::encode_point;
    use crate::transcript::push_framed;
    use sha2::Digest as _;

    let mut framed = Vec::new();
    push_framed(&mut framed, GENERATORS_DIGEST_DOMAIN);
    framed.extend_from_slice(&u32::try_from(g_vec.len()).unwrap_or(u32::MAX).to_be_bytes());
    for p in g_vec {
        push_framed(&mut framed, &encode_point(p));
    }
    framed.extend_from_slice(&u32::try_from(h_vec.len()).unwrap_or(u32::MAX).to_be_bytes());
    for p in h_vec {
        push_framed(&mut framed, &encode_point(p));
    }
    Sha256::digest(&framed).into()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::codec::encode_point;

    #[test]
    fn h_is_deterministic() {
        assert_eq!(h(), derive_h());
        assert_eq!(h(), h());
    }

    #[test]
    fn h_is_not_identity() {
        // A known-dlog or identity H breaks the hiding property (SA-2026-322).
        assert_ne!(h(), ProjectivePoint::IDENTITY);
    }

    #[test]
    fn h_differs_from_g() {
        assert_ne!(h(), g());
    }

    #[test]
    fn h_is_on_curve_and_canonically_encodable() {
        // Round-tripping through the strict codec proves H is a valid,
        // non-identity, on-curve point with a canonical 33-byte encoding.
        let enc = encode_point(&h());
        assert!(enc[0] == 0x02 || enc[0] == 0x03);
        assert_eq!(crate::codec::decode_point(&enc).unwrap(), h());
    }

    #[test]
    fn h_known_answer() {
        // Pins the exact H derivation: a change to H_DST/H_MSG or the
        // hash-to-curve suite must update this vector deliberately
        // (SA-2026-322 — H derivation is locked, not free to drift).
        use std::fmt::Write as _;
        let enc = encode_point(&h());
        let hex = enc.iter().fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        });
        assert_eq!(hex, KNOWN_H_SEC1);
    }

    // Locked from the RFC 9380 derivation; any drift in H_DST/H_MSG/suite
    // must update this vector deliberately.
    const KNOWN_H_SEC1: &str = "02460a164ac67bea239d4995793e179a3f4adfc260e0a2074c93e83228af8a5482";

    #[test]
    fn vector_generators_are_deterministic_and_distinct() {
        let gs = gvec(8);
        let hs = hvec(8);
        assert_eq!(gs, gvec(8));
        assert_eq!(hs, hvec(8));
        // No identity, no collision with G or H, no cross/role/index collisions.
        let mut seen = vec![g(), h()];
        for p in gs.iter().chain(hs.iter().skip(1)) {
            assert_ne!(*p, ProjectivePoint::IDENTITY);
            assert!(!seen.contains(p), "vector generator collision");
            seen.push(*p);
        }
    }

    #[test]
    fn hvec_zero_is_h() {
        // h⃗[0] = H makes the capsule's Pedersen commitments valid BP++ value
        // commitments with no re-commitment (soundness doc §0).
        assert_eq!(hvec(3)[0], h());
    }

    #[test]
    fn longer_vectors_extend_shorter_ones() {
        // Index-wise derivation: gvec(8) is a prefix of gvec(16), so freezing
        // a larger count later never moves existing generators.
        assert_eq!(gvec(16)[..8], gvec(8)[..]);
        assert_eq!(hvec(16)[..8], hvec(8)[..]);
    }

    #[test]
    fn generators_digest_pins_set_and_shape() {
        let d = generators_digest(&gvec(4), &hvec(3));
        // Deterministic.
        assert_eq!(d, generators_digest(&gvec(4), &hvec(3)));
        // Any count or element change forks the digest.
        assert_ne!(d, generators_digest(&gvec(5), &hvec(3)));
        assert_ne!(d, generators_digest(&gvec(4), &hvec(4)));
        let mut swapped = gvec(4);
        swapped.swap(0, 1);
        assert_ne!(d, generators_digest(&swapped, &hvec(3)));
        // Role separation: swapping the two vectors forks the digest.
        assert_ne!(
            generators_digest(&gvec(3), &hvec(4)),
            generators_digest(&hvec(4), &gvec(3))
        );
    }
}
