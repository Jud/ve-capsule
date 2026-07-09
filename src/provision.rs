//! Blob-in / blob-out provisioning of a recovery payload.
//!
//! The producer counterpart to [`crate::recover`]. The hint algebra (`hint`)
//! operates on raw curve types the crate does not export, so provisioning is
//! driven through this seam: a producer seals one cleartext additive contribution
//! into a hint ([`seal_recovery_hint`]), and the coordinator collects the sealed
//! hints into a recipient-only payload ([`assemble_recipient_recovery_payload`]) —
//! never handling a raw scalar or point.
//!
//! Each piece is sealed by the producer that holds it. If the caller's recovery
//! scheme needs interpolation weights or other coefficients, it applies them before
//! sealing. Assembly takes only the **already sealed** hint bytes, so the
//! coordinator never needs cleartext. Gated payloads carry a quorum attestation and
//! are produced by the quorum's signing path, not here.

use crate::assembly::{cross_piece_elgamal_mask_relation, degenerate_elgamal_mask};
use crate::capsule::{Capsule, PrivateKey, PublicKey};
use crate::compact_payload::CompactRecoveryPayload;
use crate::elgamal::LimbCiphertext;
use crate::error::Error;
use crate::hint::{HINT_LEN, HintBinding, RecoveryHint};
use crate::recover::RecoveryContext;
use k256::ProjectivePoint;
use rand_core::{CryptoRng, RngCore};

/// Seal one cleartext recovery contribution into a recipient-only unseal hint
/// (blob output) for piece index `idx`, under the certified recovery `context`.
///
/// The producer holds only its own contribution and the public context. `recipient`
/// is the recipient recovery key `Y_rcpt` the hint is sealed to;
/// `context.certified_target` is the public key the recovered secret must match,
/// bound into the mask alongside `ctx` and `epoch`. The recipient later opens this
/// piece with one ECDH, a hash, and a subtraction, recovering against the **same**
/// context.
///
/// # Errors
///
/// [`Error::DegenerateInput`] if `context.certified_target` or `recipient` is the
/// identity, if `ctx`/`epoch` are over-long, or if the ephemeral draws zero.
pub fn seal_recovery_hint<R: RngCore + CryptoRng>(
    contribution: &PrivateKey,
    recipient: &PublicKey,
    context: &RecoveryContext<'_>,
    idx: u32,
    rng: &mut R,
) -> Result<[u8; HINT_LEN], Error> {
    let y_star = recipient.point();
    let vs = context.certified_target.point();
    let binding = HintBinding {
        y_star: &y_star,
        vs: &vs,
        ctx: context.ctx,
        epoch: context.epoch,
    };
    let hint = RecoveryHint::seal(contribution.scalar(), &binding, idx, rng)?;
    Ok(hint.to_canonical_bytes())
}

/// Structurally validate a single sealed recovery-hint blob (blob input).
///
/// Confirms the bytes are exactly [`HINT_LEN`] long and decode to a non-identity
/// ephemeral `E*` and a canonical (`< n`) ciphertext scalar — the same strict
/// decode [`assemble_recipient_recovery_payload`] applies per piece. This lets an ingress
/// boundary for paired hint/capsule artifacts reject a malformed or absent
/// hint up front, without handling the raw curve types. An empty blob fails the
/// length check, so a piece that is required to carry a hint cannot pass with
/// none.
///
/// # Errors
///
/// [`Error::PointDecode`] if the blob is the wrong length or `E*`/the ciphertext
/// scalar is non-canonical; [`Error::DegenerateInput`] if `E*` is the identity.
pub fn validate_recovery_hint(bytes: &[u8]) -> Result<(), Error> {
    RecoveryHint::from_canonical_bytes(bytes)?;
    Ok(())
}

/// Screen every recovery hint in a paired hint/capsule set against every
/// capsule's `ElGamal` limb masks, and against the other hints (blob input).
///
/// A piece's hint and its capsule encrypt the same contribution to the **same**
/// recipient `Y*`. The hint mask absorbs only `x(z)`, and `x(P) == x(-P)`, so if a
/// hint's ephemeral `E*` aliases (`±`) or has a small public relation to capsule
/// limb ephemerals (e.g. `E* = 2·E_j` or `E* = E_i + E_k`), the shared masking
/// x-coordinate lets a full-core holder brute-force a bounded limb and unmask the
/// hint — confirming against the public commitment. Honest producers draw `E*` and
/// the limb masks independently, so a relation is negligible.
///
/// This is a **bounded** screen: it rejects the relations a non-recipient holder
/// could actually *find* — exact aliases and small public combinations within the
/// relation engine's coefficient and support bounds. A relation `E* = k·E_j` with a
/// large unknown `k` is not caught, but recovering such a `k` from the public points
/// is a discrete log, so it is not exploitable by a holder lacking the recipient
/// key; a producer that hands `k` to the coordinator could instead hand over the
/// contribution, so the bound covers the meaningful non-colluding threat.
///
/// Each `(hint_bytes, capsule_bytes)` pair is one piece. The capsule limb masks form
/// one screen "piece" and the hint `E*` another, so every hint–limb relation spans
/// pieces and is caught by [`cross_piece_elgamal_mask_relation`]. Capsule↔capsule
/// relations are out of scope here — that is `Case::verify`'s screen. The hints are
/// additionally self-screened for a duplicate or inverse `E*` (the fresh-per-piece
/// invariant). `LimbCiphertext::d` is unused by the mask-relation scanners, so the
/// hint pieces carry an identity placeholder.
///
/// Because the screen flattens to two pieces, the relation scan's own piece-count
/// guard never fires; cost is driven by the mask count (~12 per piece). The
/// piece count is therefore capped at the same bound `Case::verify` enforces, so
/// an untrusted over-large package is rejected cheaply rather than driving an
/// exponential scan (a denial-of-service guard).
///
/// # Errors
///
/// [`Error::DegenerateInput`] if the package exceeds the screen's piece bound;
/// propagates the strict-decode error of any malformed hint or capsule;
/// [`Error::Verification`] if a cross-scheme hint↔limb or a hint↔hint mask relation
/// is found.
pub fn validate_recovery_hints_against_capsules(pairs: &[(&[u8], &[u8])]) -> Result<(), Error> {
    // DoS guard: cost scales with the mask count (~12 per piece), and the
    // two-piece flattening means the scan's piece-count guard cannot fire. Reject an
    // over-large package before any capsule parse or relation enumeration.
    if pairs.len() > crate::assembly::CROSS_PIECE_MASK_RELATION_PIECE_BOUND {
        return Err(Error::DegenerateInput(
            "too many pieces for the recovery-hint cross-scheme screen",
        ));
    }
    let mut capsule_masks: Vec<LimbCiphertext> = Vec::new();
    let mut hint_masks: Vec<LimbCiphertext> = Vec::with_capacity(pairs.len());
    for (hint_bytes, capsule_bytes) in pairs {
        let hint = RecoveryHint::from_canonical_bytes(hint_bytes)?;
        let capsule = Capsule::from_canonical_bytes(capsule_bytes)?;
        capsule_masks.extend_from_slice(capsule.proof().elgamal());
        hint_masks.push(LimbCiphertext {
            e: hint.e_star(),
            d: ProjectivePoint::IDENTITY,
        });
    }

    // Hint ⟷ capsule-limb: capsule masks are one piece, hint masks another, so a
    // relating pair always spans pieces (the only relations the cross-piece scan
    // flags). Flattening the capsules into one piece intentionally leaves capsule
    // ⟷ capsule to `Case::verify`; here we only add the hint screen.
    if let Some(detail) =
        cross_piece_elgamal_mask_relation([capsule_masks.as_slice(), hint_masks.as_slice()])
    {
        return Err(Error::Verification(detail));
    }

    // Hint ⟷ hint: the hints occupy one piece above, so the cross-piece scan does
    // not cover them. Self-screen the hint masks for a duplicate or inverse `E*`.
    if let Some(detail) = degenerate_elgamal_mask(&hint_masks) {
        return Err(Error::Verification(detail));
    }
    Ok(())
}

/// Assemble a recipient-only recovery payload (blob output) from the certified
/// target and the per-piece sealed hint bytes.
///
/// Each `(idx, hint_bytes)` is the output of [`seal_recovery_hint`]; the bytes are
/// strictly decoded (rejecting a malformed or identity-`E*` hint) before assembly.
/// Pieces must be non-empty with strictly increasing indices. The result is a
/// recipient-only payload — no roster, no signature — ready for external storage.
///
/// # Errors
///
/// [`Error::PointDecode`]/[`Error::DegenerateInput`] if a hint is malformed, or if
/// the piece set is empty, over-large, or not strictly increasing, or the certified
/// target is the identity.
pub fn assemble_recipient_recovery_payload(
    certified_target: &PublicKey,
    pieces: &[(u32, [u8; HINT_LEN])],
) -> Result<Vec<u8>, Error> {
    let parsed = pieces
        .iter()
        .map(|(idx, bytes)| Ok((*idx, RecoveryHint::from_canonical_bytes(bytes)?)))
        .collect::<Result<Vec<(u32, RecoveryHint)>, Error>>()?;
    let payload = CompactRecoveryPayload::new(certified_target.point(), Vec::new(), parsed, None)?;
    Ok(payload.to_canonical_bytes())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

    use super::*;
    use crate::generators::g;
    use crate::recover::recover_recipient_secret_from_payload;
    use k256::Scalar;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const CTX: &[u8] = b"ve-capsule.provision.ctx";
    const EPOCH: &[u8] = b"ve-capsule.provision.epoch.0001";

    fn priv_key(scalar: Scalar) -> PrivateKey {
        PrivateKey::from_scalar(scalar)
    }

    fn rctx<'a>(certified_target: &'a PublicKey, epoch: &'a [u8]) -> RecoveryContext<'a> {
        RecoveryContext {
            certified_target,
            ctx: CTX,
            epoch,
        }
    }

    #[test]
    fn provision_then_recover_round_trip() {
        let mut rng = StdRng::seed_from_u64(7);
        let x_rcpt = priv_key(Scalar::from(7u64));
        let recipient = x_rcpt.public_key();

        // Three producers, each sealing only its own additive contribution.
        let contributions = [Scalar::from(3u64), Scalar::from(5u64), Scalar::from(11u64)];
        let s: Scalar = contributions
            .iter()
            .fold(Scalar::ZERO, |acc, contribution| acc + contribution);
        let certified_target = priv_key(s).public_key();
        let context = rctx(&certified_target, EPOCH);

        let pieces: Vec<(u32, [u8; HINT_LEN])> = contributions
            .iter()
            .enumerate()
            .map(|(i, contribution)| {
                let idx = u32::try_from(i).unwrap() + 1;
                let hint = seal_recovery_hint(
                    &priv_key(*contribution),
                    &recipient,
                    &context,
                    idx,
                    &mut rng,
                )
                .unwrap();
                (idx, hint)
            })
            .collect();

        let payload_bytes =
            assemble_recipient_recovery_payload(&certified_target, &pieces).unwrap();
        let payload = CompactRecoveryPayload::from_canonical_bytes(&payload_bytes).unwrap();
        assert!(!payload.is_gated());

        let recovered = recover_recipient_secret_from_payload(&payload, &x_rcpt, &context).unwrap();
        assert_eq!(recovered.public_key().point(), g() * s);
    }

    #[test]
    fn validate_recovery_hint_accepts_sealed_and_rejects_degenerate() {
        let mut rng = StdRng::seed_from_u64(11);
        let s = Scalar::from(21u64);
        let recipient = priv_key(Scalar::from(9u64)).public_key();
        let certified_target = priv_key(s).public_key();
        let context = rctx(&certified_target, EPOCH);

        // A freshly sealed hint is structurally valid.
        let hint = seal_recovery_hint(&priv_key(s), &recipient, &context, 1, &mut rng).unwrap();
        validate_recovery_hint(&hint).unwrap();

        // Empty / wrong-length blobs are rejected — an absent hint cannot pass.
        assert!(validate_recovery_hint(&[]).is_err());
        assert!(validate_recovery_hint(&hint[..HINT_LEN - 1]).is_err());
        let mut too_long = hint.to_vec();
        too_long.push(0);
        assert!(validate_recovery_hint(&too_long).is_err());

        // An all-zero blob has an invalid (non-canonical / identity) ephemeral E*.
        assert!(validate_recovery_hint(&[0u8; HINT_LEN]).is_err());

        // A valid E* with a non-canonical (`>= n`) ciphertext scalar is rejected.
        let mut bad_ct = hint;
        for byte in bad_ct.iter_mut().skip(33) {
            *byte = 0xFF;
        }
        assert!(validate_recovery_hint(&bad_ct).is_err());
    }

    #[test]
    fn hints_against_capsules_rejects_aliasing_and_duplicates() {
        use crate::Context;
        use rand::RngCore;
        use std::borrow::Cow;

        struct TestCtx;
        impl Context for TestCtx {
            fn domain(&self) -> &'static str {
                "ve-capsule.hint-screen-test"
            }
            fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, Error> {
                Ok(Cow::Borrowed(b"hint-screen-binding"))
            }
        }

        // The capsule seal rejects a publicly enumerable recovery key, so draw
        // large random keys (the recovery recipient and the sealed value).
        let mut rng = StdRng::seed_from_u64(0x5C_8E_11_22);
        let random_key = |rng: &mut StdRng| loop {
            let mut bytes = [0u8; 32];
            rng.fill_bytes(&mut bytes);
            if let Ok(key) = PrivateKey::from_secret(&bytes) {
                break key;
            }
        };
        let recipient = random_key(&mut rng).public_key();
        let secret = random_key(&mut rng);
        let certified_target = secret.public_key();
        let context = rctx(&certified_target, EPOCH);

        // A real capsule sealing `s` to the recipient, and a properly sealed hint
        // over the same `s` to the same recipient (independent fresh ephemerals).
        let capsule = Capsule::builder(&secret, &recipient, &TestCtx)
            .seal()
            .unwrap();
        let capsule_bytes = capsule.to_canonical_bytes();
        let good_hint = seal_recovery_hint(&secret, &recipient, &context, 1, &mut rng).unwrap();

        // VALID: a fresh hint ephemeral has no relation to the capsule masks.
        validate_recovery_hints_against_capsules(&[(&good_hint, &capsule_bytes)]).unwrap();

        // ALIAS: a hint whose E* equals a capsule limb ephemeral E_j is rejected.
        let limb_e = capsule.proof().elgamal()[0].e;
        let alias_hint = |point: &ProjectivePoint| -> [u8; HINT_LEN] {
            let mut blob = [0u8; HINT_LEN];
            blob[..33].copy_from_slice(&crate::codec::encode_point(point));
            blob[33..].copy_from_slice(&Scalar::from(7u64).to_bytes());
            blob
        };
        let aliasing = alias_hint(&limb_e);
        let Err(err) = validate_recovery_hints_against_capsules(&[(&aliasing, &capsule_bytes)])
        else {
            panic!("a hint E* aliasing a capsule limb mask must be rejected");
        };
        assert!(matches!(err, Error::Verification(_)));

        // ALIAS (negated): E* == -E_j shares the masking x-coordinate, also rejected.
        let neg_aliasing = alias_hint(&(-limb_e));
        assert!(
            validate_recovery_hints_against_capsules(&[(&neg_aliasing, &capsule_bytes)]).is_err()
        );

        // SMALL RELATIONS: the reason this screen uses the relation engine rather
        // than a plain ±-alias check — `E* = 2·E_j` and `E* = E_i + E_k` leak the
        // same way and must also be rejected.
        let limb_e1 = capsule.proof().elgamal()[1].e;
        let double_relation = alias_hint(&(limb_e + limb_e));
        assert!(
            validate_recovery_hints_against_capsules(&[(&double_relation, &capsule_bytes)])
                .is_err(),
            "a hint E* = 2·E_j must be rejected"
        );
        let sum_relation = alias_hint(&(limb_e + limb_e1));
        assert!(
            validate_recovery_hints_against_capsules(&[(&sum_relation, &capsule_bytes)]).is_err(),
            "a hint E* = E_i + E_k must be rejected"
        );

        // HINT-HINT: a duplicate hint E* across the package is rejected.
        let Err(err) = validate_recovery_hints_against_capsules(&[
            (&good_hint, &capsule_bytes),
            (&good_hint, &capsule_bytes),
        ]) else {
            panic!("a duplicate hint E* must be rejected package-wide");
        };
        assert!(matches!(err, Error::Verification(_)));

        // DoS GUARD: a package with more pieces than the screen's bound is
        // rejected cheaply, before any hint decode or capsule parse. Using invalid
        // (empty) blobs proves the cap precedes parsing — a cap placed after the
        // parse would surface a decode error (PointDecode) on the empty bytes
        // instead of the count error (DegenerateInput).
        let empty: &[u8] = &[];
        let over_bound: Vec<(&[u8], &[u8])> =
            vec![(empty, empty); crate::assembly::CROSS_PIECE_MASK_RELATION_PIECE_BOUND + 1];
        let Err(err) = validate_recovery_hints_against_capsules(&over_bound) else {
            panic!("a package exceeding the screen's piece bound must be rejected");
        };
        assert!(matches!(err, Error::DegenerateInput(_)));
    }

    #[test]
    fn assemble_rejects_empty_pieces() {
        let certified_target = priv_key(Scalar::from(5u64)).public_key();
        let Err(err) = assemble_recipient_recovery_payload(&certified_target, &[]) else {
            panic!("an empty piece set must be rejected");
        };
        assert!(matches!(err, Error::DegenerateInput(_)));
    }

    #[test]
    fn assemble_rejects_malformed_hint() {
        let certified_target = priv_key(Scalar::from(5u64)).public_key();
        // An all-zero hint has an identity E*, which strict decode rejects.
        let Err(err) =
            assemble_recipient_recovery_payload(&certified_target, &[(1, [0u8; HINT_LEN])])
        else {
            panic!("a malformed hint must be rejected at assembly");
        };
        assert!(matches!(
            err,
            Error::PointDecode(_) | Error::DegenerateInput(_)
        ));
    }

    #[test]
    fn ctx_epoch_mismatch_fails_recovery() {
        let mut rng = StdRng::seed_from_u64(8);
        let x_rcpt = priv_key(Scalar::from(7u64));
        let recipient = x_rcpt.public_key();
        let s = Scalar::from(42u64);
        let certified_target = priv_key(s).public_key();

        let hint = seal_recovery_hint(
            &priv_key(s),
            &recipient,
            &rctx(&certified_target, EPOCH),
            1,
            &mut rng,
        )
        .unwrap();
        let payload_bytes =
            assemble_recipient_recovery_payload(&certified_target, &[(1, hint)]).unwrap();
        let payload = CompactRecoveryPayload::from_canonical_bytes(&payload_bytes).unwrap();

        // Recovering under a different epoch changes the mask → fails the VS self-check.
        let Err(err) = recover_recipient_secret_from_payload(
            &payload,
            &x_rcpt,
            &rctx(&certified_target, b"ve-capsule.wrong.epoch"),
        ) else {
            panic!("an epoch mismatch must fail recovery");
        };
        assert!(matches!(err, Error::Verification(_)));
    }
}
