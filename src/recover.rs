//! Blob-in / blob-out recovery from a [`CompactRecoveryPayload`].
//!
//! The hint algebra (`hint`) operates on raw curve types the crate does not
//! export, so it cannot be driven from outside `ve-capsule`. This module is the
//! consent-gated public seam an integrator drives instead: it takes a payload
//! plus the **certified** recovery context as the crate's public key types and
//! returns the recovered secret, never exposing a raw scalar or a pre-self-check
//! single-piece open.
//!
//! Only the **recipient-only** path is a pure blob open. A gated payload additionally
//! requires live authorizer contributions, which a payload does not carry — that
//! flow is interactive and lives with the recovery ceremony, not here.

use crate::capsule::{PrivateKey, PublicKey};
use crate::compact_payload::CompactRecoveryPayload;
use crate::error::Error;
use crate::hint::{HintBinding, recover_recipient_only};
use k256::Scalar;

/// The public recovery context a recovery hint binds: the certified target public
/// key and the `ctx`/`epoch` domain separators the pieces were sealed under.
///
/// Shared by [`recover_recipient_secret_from_payload`] and [`crate::seal_recovery_hint`] so the
/// producer and recipient bind identical context — a mismatch in any
/// field changes the per-piece mask and fails the self-check.
#[derive(Clone, Copy)]
pub struct RecoveryContext<'a> {
    /// The certified target public key `s·G`, the self-check anchor
    /// (never the payload's self-described `VS` — design §10).
    pub certified_target: &'a PublicKey,
    /// Context domain separator.
    pub ctx: &'a [u8],
    /// Recovery epoch identifier.
    pub epoch: &'a [u8],
}

/// Recover the combined secret from a **recipient-only** recovery payload,
/// recipient-only and self-securing.
///
/// Each `(idx, hint)` piece is opened with `x_rcpt` as one additive contribution.
/// If a caller's recovery scheme needs interpolation weights or other coefficients,
/// it applies them before sealing the payload pieces. Recovery then sums the opened
/// contributions and authenticates the result against `context.certified_target`,
/// never the payload's self-described `VS` (design
/// §10). The recipient key the hints were sealed to is `x_rcpt`'s public key; a
/// wrong recipient key, a substituted payload, or a tampered piece yields a wrong
/// scalar that fails the `s·G == certified_target` self-check and is rejected — recovery
/// fails closed, it never returns a wrong secret.
///
/// A gated payload is rejected here: gates require live authorizer contributions
/// that the payload does not carry, so gated recovery runs the interactive flow.
///
/// # Errors
///
/// [`Error::Verification`] if the payload is gated, the piece set is malformed,
/// or the recovered secret fails the certified-target self-check;
/// [`Error::DegenerateInput`] on a degenerate binding.
pub fn recover_recipient_secret_from_payload(
    payload: &CompactRecoveryPayload,
    x_rcpt: &PrivateKey,
    context: &RecoveryContext<'_>,
) -> Result<PrivateKey, Error> {
    if payload.is_gated() {
        return Err(Error::Verification(
            "recipient-only recovery rejects a gated payload; gated recovery is interactive",
        ));
    }
    let recipient = x_rcpt.public_key().point();
    let vs = context.certified_target.point();
    let binding = HintBinding {
        y_star: &recipient,
        vs: &vs,
        ctx: context.ctx,
        epoch: context.epoch,
    };
    // Payload pieces are already additive contributions, so recovery sums them with
    // unit weights.
    let weights = vec![Scalar::ONE; payload.pieces().len()];
    let recovered = recover_recipient_only(payload.pieces(), &weights, x_rcpt.scalar(), &binding)?;
    Ok(PrivateKey::from_scalar(*recovered))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

    use super::*;
    use crate::compact_payload::CompactRecoveryPayload;
    use crate::generators::g;
    use crate::hint::RecoveryHint;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const CTX: &[u8] = b"ve-capsule.recover.ctx";
    const EPOCH: &[u8] = b"ve-capsule.recover.epoch.0001";

    /// Seal `pieces` additive contributions to `x_rcpt`, returning the recipient-only
    /// payload and the combined secret `s = Σ sₕ` it recovers to.
    fn seal_recipient_only(
        contributions: &[Scalar],
        x_rcpt: &Scalar,
        rng: &mut StdRng,
    ) -> (CompactRecoveryPayload, Scalar) {
        let recipient = g() * x_rcpt;
        let s: Scalar = contributions
            .iter()
            .fold(Scalar::ZERO, |acc, contribution| acc + contribution);
        let vs = g() * s;
        let binding = HintBinding {
            y_star: &recipient,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let pieces: Vec<(u32, RecoveryHint)> = contributions
            .iter()
            .enumerate()
            .map(|(i, contribution)| {
                let idx = u32::try_from(i).unwrap() + 1;
                (
                    idx,
                    RecoveryHint::seal(contribution, &binding, idx, rng).unwrap(),
                )
            })
            .collect();
        let payload = CompactRecoveryPayload::new(vs, Vec::new(), pieces, None).unwrap();
        (payload, s)
    }

    fn priv_key(scalar: Scalar) -> PrivateKey {
        PrivateKey::from_scalar(scalar)
    }

    fn rctx(certified_target: &PublicKey) -> RecoveryContext<'_> {
        RecoveryContext {
            certified_target,
            ctx: CTX,
            epoch: EPOCH,
        }
    }

    #[test]
    fn recipient_only_round_trip_single_piece() {
        let mut rng = StdRng::seed_from_u64(1);
        let x_rcpt = Scalar::from(7u64);
        let (payload, s) = seal_recipient_only(&[Scalar::from(42u64)], &x_rcpt, &mut rng);

        let certified_target = priv_key(s).public_key();
        let recovered = recover_recipient_secret_from_payload(
            &payload,
            &priv_key(x_rcpt),
            &rctx(&certified_target),
        )
        .unwrap();
        assert_eq!(recovered.public_key().point(), g() * s);
    }

    #[test]
    fn recipient_only_round_trip_multi_piece_additive() {
        let mut rng = StdRng::seed_from_u64(2);
        let x_rcpt = Scalar::from(9u64);
        let contributions = [Scalar::from(3u64), Scalar::from(5u64), Scalar::from(11u64)];
        let (payload, s) = seal_recipient_only(&contributions, &x_rcpt, &mut rng);
        assert_eq!(s, Scalar::from(19u64));

        let certified_target = priv_key(s).public_key();
        let recovered = recover_recipient_secret_from_payload(
            &payload,
            &priv_key(x_rcpt),
            &rctx(&certified_target),
        )
        .unwrap();
        assert_eq!(recovered.public_key().point(), g() * s);
    }

    #[test]
    fn recipient_only_rejects_wrong_recipient_secret() {
        let mut rng = StdRng::seed_from_u64(3);
        let x_rcpt = Scalar::from(7u64);
        let (payload, s) = seal_recipient_only(&[Scalar::from(42u64)], &x_rcpt, &mut rng);

        let certified_target = priv_key(s).public_key();
        let wrong = priv_key(Scalar::from(8u64));
        let Err(err) =
            recover_recipient_secret_from_payload(&payload, &wrong, &rctx(&certified_target))
        else {
            panic!("wrong recipient secret must fail the VS self-check");
        };
        assert!(matches!(err, Error::Verification(_)));
    }

    #[test]
    fn recipient_only_rejects_wrong_certified_target() {
        let mut rng = StdRng::seed_from_u64(4);
        let x_rcpt = Scalar::from(7u64);
        let (payload, _s) = seal_recipient_only(&[Scalar::from(42u64)], &x_rcpt, &mut rng);

        // A certified VS that is not s·G: the self-check rejects even a correct open.
        let wrong_vs = priv_key(Scalar::from(99u64)).public_key();
        let Err(err) =
            recover_recipient_secret_from_payload(&payload, &priv_key(x_rcpt), &rctx(&wrong_vs))
        else {
            panic!("wrong certified VS must fail the self-check");
        };
        assert!(matches!(err, Error::Verification(_)));
    }

    #[test]
    fn recipient_only_rejects_tampered_piece() {
        let mut rng = StdRng::seed_from_u64(5);
        let x_rcpt = Scalar::from(7u64);
        let (payload, s) = seal_recipient_only(
            &[Scalar::from(42u64), Scalar::from(8u64)],
            &x_rcpt,
            &mut rng,
        );

        // Flip a ct in one piece; the combined secret no longer reconstructs VS.
        let mut pieces = payload.pieces().to_vec();
        let bytes = pieces[0].1.to_canonical_bytes();
        let mut tampered = bytes;
        tampered[64] ^= 0x01;
        pieces[0].1 = RecoveryHint::from_canonical_bytes(&tampered).unwrap();
        let tampered_payload =
            CompactRecoveryPayload::new(*payload.vs(), Vec::new(), pieces, None).unwrap();

        let certified_target = priv_key(s).public_key();
        let Err(err) = recover_recipient_secret_from_payload(
            &tampered_payload,
            &priv_key(x_rcpt),
            &rctx(&certified_target),
        ) else {
            panic!("a tampered piece must fail the VS self-check");
        };
        assert!(matches!(err, Error::Verification(_)));
    }

    #[test]
    fn recipient_only_rejects_gated_payload() {
        let mut rng = StdRng::seed_from_u64(6);
        let x_rcpt = Scalar::from(7u64);
        let (recipient_only, s) = seal_recipient_only(&[Scalar::from(42u64)], &x_rcpt, &mut rng);

        // Re-wrap the same pieces as a gated payload (roster + signature present).
        let roster = vec![g() * Scalar::from(123u64)];
        let sig = crate::Signature::schnorr([0u8; 64]);
        let gated = CompactRecoveryPayload::new(
            *recipient_only.vs(),
            roster,
            recipient_only.pieces().to_vec(),
            Some(sig),
        )
        .unwrap();
        assert!(gated.is_gated());

        let certified_target = priv_key(s).public_key();
        let Err(err) = recover_recipient_secret_from_payload(
            &gated,
            &priv_key(x_rcpt),
            &rctx(&certified_target),
        ) else {
            panic!("a gated payload must be rejected by the recipient-only entry");
        };
        assert!(matches!(err, Error::Verification(_)));
    }
}
