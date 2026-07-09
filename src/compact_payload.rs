//! Compact recovery payloads — the canonical, self-describing byte projection of
//! recovery hints.
//!
//! A payload carries the **certified target** `VS`, the per-piece unseal hints
//! `{idx, E*, ct}`, and — when gated — the access-gate roster and the
//! quorum's hint-binding signature (§8). It does **not** carry `ctx`/`epoch`/recipient:
//! those are pinned by the recipient verifier ([`crate::PinnedHintVerifier`]), and
//! that verifier cross-checks the payload's `VS` + roster against its pinned binding
//! ([`CompactRecoveryPayload::matches_pinned`]) rather than trusting the payload's copy (§10
//! — the security-critical `VS` is always the pinned/certified one, never the
//! payload's).
//!
//! ```text
//! version(1) ‖ SEC1(VS)(33) ‖ piece_count(4) ‖ {BE32(idx) ‖ SEC1(E*) ‖ ct}*
//!   ‖ gate_count(4) ‖ SEC1(gate)* ‖ [gated: scheme(1) ‖ sig(64)]
//! ```
//!
//! Recipient-only payloads (`gate_count == 0`) carry no signature; gated payloads
//! (`gate_count > 0`) carry exactly one.

use crate::codec::{POINT_LEN, decode_point, encode_point};
use crate::error::Error;
use crate::hint::{GatedBinding, HINT_LEN, RecoveryHint};
use crate::signature::Signature;
use crate::transcript::length_prefix;
use k256::ProjectivePoint;

/// Payload wire magic — the self-identifying prefix marking these bytes as a compact
/// recovery payload, distinct from every other ve-capsule wire type so a blob fed
/// to the wrong decoder fails immediately (the crate's `*_WIRE_MAGIC` convention).
const COMPACT_RECOVERY_PAYLOAD_WIRE_MAGIC: &[u8] = b"ve-capsule.compact-recovery-payload.v1";

/// Payload wire version. Bump on any layout change.
const COMPACT_RECOVERY_PAYLOAD_VERSION: u8 = 0x01;

/// Signature-scheme tag for BIP-340 Schnorr (the one quorum scheme today).
const SCHEME_SCHNORR: u8 = 0x01;

/// Byte length of a BIP-340 signature (`R_x ‖ s`).
const SCHNORR_SIG_LEN: usize = 64;

/// Wire length of one piece record: `BE32(idx)` (4) ‖ `SEC1(E*) ‖ ct` (`HINT_LEN`).
const PIECE_RECORD_LEN: usize = 4 + HINT_LEN;

/// Maximum pieces on one compact recovery payload. Generous versus realistic
/// recovery payloads and a hard cap so a forged count cannot drive a large
/// allocation in a caller-facing parser.
const MAX_COMPACT_RECOVERY_PAYLOAD_PIECES: usize = 32;

/// Maximum access gates on one compact recovery payload — matches `composite`'s
/// access-key roster cap (a larger roster is rejected at opening-binding regardless).
const MAX_COMPACT_RECOVERY_PAYLOAD_GATES: usize = 5;

/// A compact recovery payload: the certified `VS`, the
/// per-piece unseal hints, and — when gated — the access-gate roster plus the
/// quorum's hint-binding signature.
///
/// Construct with [`CompactRecoveryPayload::new`] (which enforces the gated ⟺ roster ∧
/// signature invariant), serialize with [`CompactRecoveryPayload::to_canonical_bytes`], and
/// parse untrusted bytes with [`CompactRecoveryPayload::from_canonical_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactRecoveryPayload {
    vs: ProjectivePoint,
    access_keys: Vec<ProjectivePoint>,
    pieces: Vec<(u32, RecoveryHint)>,
    signature: Option<Signature>,
}

impl CompactRecoveryPayload {
    /// Assemble a payload from its parts, enforcing the canonical invariants.
    ///
    /// `pieces` must be non-empty (at most `MAX_COMPACT_RECOVERY_PAYLOAD_PIECES`) with **strictly
    /// increasing** indices. The roster is **canonicalized** — sorted by SEC1 encoding
    /// and rejected on a duplicate or identity gate, at most
    /// `MAX_COMPACT_RECOVERY_PAYLOAD_GATES` — so a payload has one canonical wire
    /// form regardless of the order the caller lists
    /// gates. The payload is **gated** iff it has both a non-empty roster and a
    /// `signature`, and **recipient-only** iff it has neither — a roster without a
    /// signature (or vice versa) is rejected. `vs` must be non-identity.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] on empty/oversized/misordered pieces, an oversized,
    /// duplicate, or identity roster, an identity `vs`, or a roster/signature mismatch.
    pub fn new(
        vs: ProjectivePoint,
        access_keys: Vec<ProjectivePoint>,
        pieces: Vec<(u32, RecoveryHint)>,
        signature: Option<Signature>,
    ) -> Result<Self, Error> {
        if vs == ProjectivePoint::IDENTITY {
            return Err(Error::DegenerateInput(
                "compact recovery payload VS is the identity",
            ));
        }
        if pieces.is_empty() {
            return Err(Error::DegenerateInput(
                "compact recovery payload has no pieces",
            ));
        }
        if pieces.len() > MAX_COMPACT_RECOVERY_PAYLOAD_PIECES {
            return Err(Error::DegenerateInput(
                "compact recovery payload has too many pieces",
            ));
        }
        for window in pieces.windows(2) {
            if window[1].0 <= window[0].0 {
                return Err(Error::DegenerateInput(
                    "compact recovery payload pieces are not in strictly increasing index order",
                ));
            }
        }
        if access_keys.len() > MAX_COMPACT_RECOVERY_PAYLOAD_GATES {
            return Err(Error::DegenerateInput(
                "compact recovery payload has too many gates",
            ));
        }
        let access_keys = canonical_roster(access_keys)?;
        if access_keys.is_empty() != signature.is_none() {
            return Err(Error::DegenerateInput(
                "compact recovery payload must be gated (roster + signature) or recipient-only (neither)",
            ));
        }
        Ok(Self {
            vs,
            access_keys,
            pieces,
            signature,
        })
    }

    /// The certified target point `VS` carried by the payload (self-describing; the
    /// recipient authenticates against its **pinned** `VS`, not this one).
    #[must_use]
    pub const fn vs(&self) -> &ProjectivePoint {
        &self.vs
    }

    /// The access-gate roster (empty for a recipient-only payload).
    #[must_use]
    pub fn access_keys(&self) -> &[ProjectivePoint] {
        &self.access_keys
    }

    /// The per-piece unseal hints `(idx, hint)`, in strictly increasing index order.
    #[must_use]
    pub fn pieces(&self) -> &[(u32, RecoveryHint)] {
        &self.pieces
    }

    /// The quorum's hint-binding signature, present iff the payload is gated.
    #[must_use]
    pub const fn signature(&self) -> Option<&Signature> {
        self.signature.as_ref()
    }

    /// Whether the payload is gated (carries a roster and a signature).
    #[must_use]
    pub const fn is_gated(&self) -> bool {
        self.signature.is_some()
    }

    /// Cross-check the payload's self-describing `VS` and roster against a **pinned**
    /// binding (design §10): the recipient verifier asserts the payload belongs to
    /// its pinned recovery context before recovering. The roster is compared as a
    /// canonical set (order-independent, matching `composite`'s gate canonicalization).
    ///
    /// This is an early/UX guard, not the soundness gate — gated recovery still
    /// reverifies the quorum signature under the pinned key and self-checks against the
    /// pinned `VS`, both independent of the payload's copies.
    #[must_use]
    pub fn matches_pinned(&self, binding: &GatedBinding<'_>) -> bool {
        if self.vs != *binding.vs {
            return false;
        }
        let mut here: Vec<[u8; POINT_LEN]> = self.access_keys.iter().map(encode_point).collect();
        let mut pinned: Vec<[u8; POINT_LEN]> =
            binding.access_keys.iter().map(encode_point).collect();
        here.sort_unstable();
        pinned.sort_unstable();
        here == pinned
    }

    /// The exact byte length [`Self::to_canonical_bytes`] produces, without
    /// allocating the serialized payload.
    #[must_use]
    pub fn wire_len(&self) -> usize {
        let sig_len = if self.signature.is_some() {
            1 + SCHNORR_SIG_LEN
        } else {
            0
        };
        COMPACT_RECOVERY_PAYLOAD_WIRE_MAGIC.len()
            + 1
            + POINT_LEN
            + 4
            + self.pieces.len() * PIECE_RECORD_LEN
            + 4
            + self.access_keys.len() * POINT_LEN
            + sig_len
    }

    /// Serialize to canonical compact recovery payload bytes.
    #[must_use]
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.wire_len());
        out.extend_from_slice(COMPACT_RECOVERY_PAYLOAD_WIRE_MAGIC);
        out.push(COMPACT_RECOVERY_PAYLOAD_VERSION);
        out.extend_from_slice(&encode_point(&self.vs));
        out.extend_from_slice(&length_prefix(self.pieces.len()));
        for (idx, hint) in &self.pieces {
            out.extend_from_slice(&idx.to_be_bytes());
            out.extend_from_slice(&hint.to_canonical_bytes());
        }
        out.extend_from_slice(&length_prefix(self.access_keys.len()));
        for gate in &self.access_keys {
            out.extend_from_slice(&encode_point(gate));
        }
        if let Some(signature) = &self.signature {
            out.push(SCHEME_SCHNORR);
            out.extend_from_slice(signature.bytes());
        }
        out
    }

    /// Parse a compact recovery payload from untrusted canonical bytes.
    ///
    /// Strict: the magic and version must match, `VS` and every gate must decode
    /// canonically and be non-identity, piece indices must be strictly increasing, the
    /// gated ⟺ roster ∧ signature invariant must hold, and there must be no trailing
    /// bytes. List counts are bounded by the remaining input before allocation, and a
    /// final re-encode-equality check rejects any non-canonical input (e.g. a reordered
    /// roster) — the crate's standard wire-decode discipline.
    ///
    /// # Errors
    ///
    /// [`Error::PointDecode`] on bad magic/version, truncation, an over-long count,
    /// trailing bytes, a non-canonical encoding, or an unsupported signature scheme;
    /// [`Error::DegenerateInput`] on an identity `VS`/gate or a roster/signature mismatch.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut cursor = bytes
            .strip_prefix(COMPACT_RECOVERY_PAYLOAD_WIRE_MAGIC)
            .ok_or(Error::PointDecode("compact recovery payload: bad magic"))?;
        if take(&mut cursor, 1)?[0] != COMPACT_RECOVERY_PAYLOAD_VERSION {
            return Err(Error::PointDecode(
                "compact recovery payload: unsupported version",
            ));
        }
        let vs = decode_point(take(&mut cursor, POINT_LEN)?)?;

        let piece_count = take_count(&mut cursor)?;
        bound_count(
            piece_count,
            PIECE_RECORD_LEN,
            cursor.len(),
            MAX_COMPACT_RECOVERY_PAYLOAD_PIECES,
        )?;
        let mut pieces = Vec::with_capacity(piece_count);
        for _ in 0..piece_count {
            let idx = take_u32(&mut cursor)?;
            let hint = RecoveryHint::from_canonical_bytes(take(&mut cursor, HINT_LEN)?)?;
            pieces.push((idx, hint));
        }

        let gate_count = take_count(&mut cursor)?;
        bound_count(
            gate_count,
            POINT_LEN,
            cursor.len(),
            MAX_COMPACT_RECOVERY_PAYLOAD_GATES,
        )?;
        let mut access_keys = Vec::with_capacity(gate_count);
        for _ in 0..gate_count {
            access_keys.push(decode_point(take(&mut cursor, POINT_LEN)?)?);
        }

        let signature = if gate_count > 0 {
            if take(&mut cursor, 1)?[0] != SCHEME_SCHNORR {
                return Err(Error::PointDecode(
                    "compact recovery payload: unsupported signature scheme",
                ));
            }
            let mut sig = [0u8; SCHNORR_SIG_LEN];
            sig.copy_from_slice(take(&mut cursor, SCHNORR_SIG_LEN)?);
            Some(Signature::schnorr(sig))
        } else {
            None
        };

        if !cursor.is_empty() {
            return Err(Error::PointDecode(
                "compact recovery payload: trailing bytes",
            ));
        }
        // new() canonicalizes the roster + re-checks the semantic invariants; the
        // re-encode-equality check then rejects any non-canonical input (e.g. a
        // reordered or duplicated wire roster, which new() would silently re-sort) —
        // the same structural canonicality guard the crate's other decoders use.
        let payload = Self::new(vs, access_keys, pieces, signature)?;
        if payload.to_canonical_bytes() != bytes {
            return Err(Error::PointDecode(
                "compact recovery payload: non-canonical encoding",
            ));
        }
        Ok(payload)
    }
}

/// Canonicalize an access-gate roster: reject identity gates, sort by SEC1 encoding,
/// and reject duplicates — the same canonical ordering `composite` uses, so the
/// payload roster is independent of the order the caller listed it and the wire form
/// is unique (non-malleable).
fn canonical_roster(access_keys: Vec<ProjectivePoint>) -> Result<Vec<ProjectivePoint>, Error> {
    let mut tagged: Vec<([u8; POINT_LEN], ProjectivePoint)> = Vec::with_capacity(access_keys.len());
    for gate in access_keys {
        if gate == ProjectivePoint::IDENTITY {
            return Err(Error::DegenerateInput(
                "compact recovery payload gate is the identity",
            ));
        }
        tagged.push((encode_point(&gate), gate));
    }
    tagged.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for window in tagged.windows(2) {
        if window[0].0 == window[1].0 {
            return Err(Error::DegenerateInput(
                "compact recovery payload has a duplicate gate",
            ));
        }
    }
    Ok(tagged.into_iter().map(|(_, gate)| gate).collect())
}

/// Split `n` bytes off the front of `cursor`, advancing it; error on truncation.
fn take<'a>(cursor: &mut &'a [u8], n: usize) -> Result<&'a [u8], Error> {
    let (head, tail) = cursor
        .split_at_checked(n)
        .ok_or(Error::PointDecode("compact recovery payload: truncated"))?;
    *cursor = tail;
    Ok(head)
}

/// Read a big-endian `u32` from the cursor.
fn take_u32(cursor: &mut &[u8]) -> Result<u32, Error> {
    let mut arr = [0u8; 4];
    arr.copy_from_slice(take(cursor, 4)?);
    Ok(u32::from_be_bytes(arr))
}

/// Read a 4-byte big-endian count as `usize`.
fn take_count(cursor: &mut &[u8]) -> Result<usize, Error> {
    Ok(take_u32(cursor)? as usize)
}

/// Reject a list count that exceeds its `max` cap or whose records cannot fit in the
/// remaining input, before allocating — so a forged 4-byte count cannot drive a large
/// allocation.
fn bound_count(count: usize, record_len: usize, remaining: usize, max: usize) -> Result<(), Error> {
    if count > max {
        return Err(Error::PointDecode(
            "compact recovery payload: list count exceeds cap",
        ));
    }
    let needed = count.checked_mul(record_len).ok_or(Error::PointDecode(
        "compact recovery payload: list count overflow",
    ))?;
    if needed > remaining {
        return Err(Error::PointDecode(
            "compact recovery payload: list count exceeds input",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;
    use crate::hint::HintBinding;
    use k256::elliptic_curve::Field;
    use k256::{ProjectivePoint, Scalar};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn point(rng: &mut StdRng) -> ProjectivePoint {
        ProjectivePoint::GENERATOR * Scalar::random(rng)
    }

    fn sample_hint(rng: &mut StdRng, idx: u32) -> RecoveryHint {
        let y = point(rng);
        let vs = point(rng);
        let binding = HintBinding {
            y_star: &y,
            vs: &vs,
            ctx: b"ctx",
            epoch: b"epoch",
        };
        let s = Scalar::random(&mut *rng);
        RecoveryHint::seal(&s, &binding, idx, rng).unwrap()
    }

    fn dummy_sig() -> Signature {
        Signature::schnorr([0xABu8; 64])
    }

    #[test]
    fn gated_round_trip() {
        let mut rng = StdRng::seed_from_u64(0x57C0_0001);
        let vs = point(&mut rng);
        let access = vec![point(&mut rng), point(&mut rng)];
        let pieces = vec![
            (1u32, sample_hint(&mut rng, 1)),
            (4u32, sample_hint(&mut rng, 4)),
        ];
        let payload = CompactRecoveryPayload::new(vs, access, pieces, Some(dummy_sig())).unwrap();
        assert!(payload.is_gated());

        let bytes = payload.to_canonical_bytes();
        assert_eq!(bytes.len(), payload.wire_len());
        let decoded = CompactRecoveryPayload::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn recipient_only_round_trip() {
        let mut rng = StdRng::seed_from_u64(0x57C0_0002);
        let vs = point(&mut rng);
        let pieces = vec![(2u32, sample_hint(&mut rng, 2))];
        let payload = CompactRecoveryPayload::new(vs, Vec::new(), pieces, None).unwrap();
        assert!(!payload.is_gated());
        assert!(payload.signature().is_none());

        let bytes = payload.to_canonical_bytes();
        assert_eq!(bytes.len(), payload.wire_len());
        let decoded = CompactRecoveryPayload::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn rejects_roster_signature_mismatch() {
        let mut rng = StdRng::seed_from_u64(0x57C0_0003);
        let vs = point(&mut rng);
        let pieces = vec![(1u32, sample_hint(&mut rng, 1))];
        // Roster without a signature.
        assert!(matches!(
            CompactRecoveryPayload::new(vs, vec![point(&mut rng)], pieces.clone(), None),
            Err(Error::DegenerateInput(_))
        ));
        // Signature without a roster.
        assert!(matches!(
            CompactRecoveryPayload::new(vs, Vec::new(), pieces, Some(dummy_sig())),
            Err(Error::DegenerateInput(_))
        ));
    }

    #[test]
    fn rejects_empty_and_misordered_pieces() {
        let mut rng = StdRng::seed_from_u64(0x57C0_0004);
        let vs = point(&mut rng);
        assert!(matches!(
            CompactRecoveryPayload::new(vs, Vec::new(), Vec::new(), None),
            Err(Error::DegenerateInput(_))
        ));
        let out_of_order = vec![
            (4u32, sample_hint(&mut rng, 4)),
            (1u32, sample_hint(&mut rng, 1)),
        ];
        assert!(matches!(
            CompactRecoveryPayload::new(vs, Vec::new(), out_of_order, None),
            Err(Error::DegenerateInput(_))
        ));
    }

    #[test]
    fn rejects_identity_vs_and_gate() {
        let mut rng = StdRng::seed_from_u64(0x57C0_0005);
        let pieces = vec![(1u32, sample_hint(&mut rng, 1))];
        assert!(matches!(
            CompactRecoveryPayload::new(
                ProjectivePoint::IDENTITY,
                Vec::new(),
                pieces.clone(),
                None
            ),
            Err(Error::DegenerateInput(_))
        ));
        let vs = point(&mut rng);
        assert!(matches!(
            CompactRecoveryPayload::new(
                vs,
                vec![ProjectivePoint::IDENTITY],
                pieces,
                Some(dummy_sig())
            ),
            Err(Error::DegenerateInput(_))
        ));
    }

    #[test]
    fn strict_decode_rejects_corruption() {
        let mut rng = StdRng::seed_from_u64(0x57C0_0006);
        let vs = point(&mut rng);
        let pieces = vec![(1u32, sample_hint(&mut rng, 1))];
        let payload =
            CompactRecoveryPayload::new(vs, vec![point(&mut rng)], pieces, Some(dummy_sig()))
                .unwrap();
        let good = payload.to_canonical_bytes();

        // Truncated.
        assert!(CompactRecoveryPayload::from_canonical_bytes(&good[..good.len() - 1]).is_err());
        // Trailing byte.
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(matches!(
            CompactRecoveryPayload::from_canonical_bytes(&trailing),
            Err(Error::PointDecode(_))
        ));
        // Bad magic (first byte).
        let mut bad_magic = good.clone();
        bad_magic[0] = 0xFF;
        assert!(matches!(
            CompactRecoveryPayload::from_canonical_bytes(&bad_magic),
            Err(Error::PointDecode(_))
        ));
        // Bad version (the byte right after the magic).
        let mut bad_version = good;
        bad_version[COMPACT_RECOVERY_PAYLOAD_WIRE_MAGIC.len()] = 0xFF;
        assert!(matches!(
            CompactRecoveryPayload::from_canonical_bytes(&bad_version),
            Err(Error::PointDecode(_))
        ));
        // Empty input.
        assert!(CompactRecoveryPayload::from_canonical_bytes(&[]).is_err());
    }

    #[test]
    fn decode_rejects_overlong_count() {
        let mut rng = StdRng::seed_from_u64(0x57C0_0007);
        let vs = point(&mut rng);
        let pieces = vec![(1u32, sample_hint(&mut rng, 1))];
        let payload = CompactRecoveryPayload::new(vs, Vec::new(), pieces, None).unwrap();
        let mut bytes = payload.to_canonical_bytes();
        // Overwrite the piece count (just after version + VS) with a huge value.
        let count_at = COMPACT_RECOVERY_PAYLOAD_WIRE_MAGIC.len() + 1 + POINT_LEN;
        bytes[count_at..count_at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(
            CompactRecoveryPayload::from_canonical_bytes(&bytes),
            Err(Error::PointDecode(_))
        ));
    }

    #[test]
    fn matches_pinned_detects_mismatch() {
        let mut rng = StdRng::seed_from_u64(0x57C0_0008);
        let vs = point(&mut rng);
        let gate_a = point(&mut rng);
        let gate_b = point(&mut rng);
        let recipient = point(&mut rng);
        let pieces = vec![(1u32, sample_hint(&mut rng, 1))];
        // Roster stored in one order; pinned binding lists it in the other.
        let payload =
            CompactRecoveryPayload::new(vs, vec![gate_a, gate_b], pieces, Some(dummy_sig()))
                .unwrap();

        let pinned_roster = [gate_b, gate_a];
        let pinned = GatedBinding {
            recipient: &recipient,
            access_keys: &pinned_roster,
            vs: &vs,
            ctx: b"ctx",
            epoch: b"epoch",
        };
        assert!(payload.matches_pinned(&pinned));

        // Different VS.
        let other_vs = point(&mut rng);
        let mismatch_vs = GatedBinding {
            vs: &other_vs,
            ..pinned
        };
        assert!(!payload.matches_pinned(&mismatch_vs));

        // Different roster.
        let other_roster = [gate_a, point(&mut rng)];
        let mismatch_roster = GatedBinding {
            access_keys: &other_roster,
            ..pinned
        };
        assert!(!payload.matches_pinned(&mismatch_roster));
    }

    #[test]
    fn small_gated_payload_wire_len_is_pinned() {
        // A single-piece, single-gate gated payload stays small enough for
        // low-capacity encodings.
        let mut rng = StdRng::seed_from_u64(0x57C0_0009);
        let vs = point(&mut rng);
        let pieces = vec![(1u32, sample_hint(&mut rng, 1))];
        let payload =
            CompactRecoveryPayload::new(vs, vec![point(&mut rng)], pieces, Some(dummy_sig()))
                .unwrap();
        assert_eq!(
            payload.wire_len(),
            COMPACT_RECOVERY_PAYLOAD_WIRE_MAGIC.len() + 209
        );
    }

    #[test]
    fn new_sorts_and_dedups_roster() {
        let mut rng = StdRng::seed_from_u64(0x57C0_000A);
        let vs = point(&mut rng);
        let a = point(&mut rng);
        let b = point(&mut rng);
        let pieces = vec![(1u32, sample_hint(&mut rng, 1))];
        // Built in arbitrary order, stored canonically sorted by SEC1 encoding.
        let payload =
            CompactRecoveryPayload::new(vs, vec![a, b], pieces.clone(), Some(dummy_sig())).unwrap();
        let got: Vec<[u8; POINT_LEN]> = payload.access_keys().iter().map(encode_point).collect();
        let mut want = vec![encode_point(&a), encode_point(&b)];
        want.sort_unstable();
        assert_eq!(got, want);
        // A duplicate gate is rejected.
        assert!(matches!(
            CompactRecoveryPayload::new(vs, vec![a, a], pieces, Some(dummy_sig())),
            Err(Error::DegenerateInput(_))
        ));
    }

    #[test]
    fn decode_rejects_noncanonical_roster() {
        // Swapping the two canonically sorted gate encodings on the wire yields a
        // descending roster — a second encoding of the same logical payload — and is
        // rejected (non-malleability).
        let mut rng = StdRng::seed_from_u64(0x57C0_000B);
        let vs = point(&mut rng);
        let pieces = vec![(1u32, sample_hint(&mut rng, 1))];
        let payload = CompactRecoveryPayload::new(
            vs,
            vec![point(&mut rng), point(&mut rng)],
            pieces,
            Some(dummy_sig()),
        )
        .unwrap();
        let mut bytes = payload.to_canonical_bytes();
        let gate0 =
            COMPACT_RECOVERY_PAYLOAD_WIRE_MAGIC.len() + 1 + POINT_LEN + 4 + PIECE_RECORD_LEN + 4;
        let mut first = [0u8; POINT_LEN];
        let mut second = [0u8; POINT_LEN];
        first.copy_from_slice(&bytes[gate0..gate0 + POINT_LEN]);
        second.copy_from_slice(&bytes[gate0 + POINT_LEN..gate0 + 2 * POINT_LEN]);
        bytes[gate0..gate0 + POINT_LEN].copy_from_slice(&second);
        bytes[gate0 + POINT_LEN..gate0 + 2 * POINT_LEN].copy_from_slice(&first);
        assert!(matches!(
            CompactRecoveryPayload::from_canonical_bytes(&bytes),
            Err(Error::PointDecode(_))
        ));
    }

    #[test]
    fn rejects_over_cap_counts() {
        let mut rng = StdRng::seed_from_u64(0x57C0_000C);
        let vs = point(&mut rng);
        let pieces = vec![(1u32, sample_hint(&mut rng, 1))];
        let too_many_gates: Vec<ProjectivePoint> = (0..=MAX_COMPACT_RECOVERY_PAYLOAD_GATES)
            .map(|_| point(&mut rng))
            .collect();
        assert!(matches!(
            CompactRecoveryPayload::new(vs, too_many_gates, pieces, Some(dummy_sig())),
            Err(Error::DegenerateInput(_))
        ));
        let piece_count = u32::try_from(MAX_COMPACT_RECOVERY_PAYLOAD_PIECES).unwrap() + 1;
        let many_pieces: Vec<(u32, RecoveryHint)> = (1..=piece_count)
            .map(|i| (i, sample_hint(&mut rng, i)))
            .collect();
        assert!(matches!(
            CompactRecoveryPayload::new(vs, Vec::new(), many_pieces, None),
            Err(Error::DegenerateInput(_))
        ));
    }
}
