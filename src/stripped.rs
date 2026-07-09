//! Proof-stripped recovery core: a verified capsule reduced to its opening core
//! (commitment `C` + per-limb `ElGamal` ciphertexts) for compact storage.
//!
//! A sealed [`Capsule`](crate::Capsule) is 5,414 bytes at the frozen shape; the
//! opening core is 778 bytes. The seal proof `π`'s job is finished once the
//! quorum verifies the capsule at provisioning, so compact recovery storage keeps
//! only the **opening core** and drops `π`. A core is produced solely by stripping
//! a [`VerifiedCapsule`](crate::VerifiedCapsule) — the honest encoder — so a
//! stripped core is always one whose `π` already passed.
//!
//! Recovery mirrors the proof-backed path:
//! - [`StrippedCapsule::bind`] re-runs the screens `π` would otherwise enforce
//!   (mask degeneracy, `Y*` enumerability, context limits) and pins the
//!   `(recipient, gates, ctx)` and certified target (`C == expected_pubkey`),
//!   yielding a [`BoundCapsule`].
//! - [`BoundCapsule::unseal`] opens with the recipient's secret + authorizer
//!   partials. It is **self-securing**: the final `g·m == C` recheck means a
//!   tampered core can make recovery *fail* but never yield a wrong secret, so it
//!   needs no signature.
//!
//! A stripped capsule is **recovery media, not trustless-provenance media**:
//! decoding asserts nothing about authenticity. Promoting a core to the full
//! `contribute` surface needs a quorum signature (`verify_signed`); the
//! self-securing `unseal` is for recovering *your own* core locally.

use crate::assembly::{
    cross_piece_elgamal_mask_relation, degenerate_elgamal_mask, reject_degenerate_recovery_key,
    validate_context_limits,
};
use crate::capsule::{FrozenContext, PrivateKey, PublicKey, VerifiedCapsule};
use crate::case::VerifiedCase;
use crate::codec::{POINT_LEN, decode_point, encode_point};
use crate::composite::{self, OpeningBinding};
use crate::context::Context;
use crate::elgamal::{LimbCiphertext, encode_limb, reject_identity_mask};
use crate::error::Error;
use crate::limbs::{LIMB_COUNT, LIMB_MODULUS};
use crate::opening::{self, CapsuleRef, Partial};
use crate::signature::{self, Backing, Signature};
use k256::ProjectivePoint;

/// Wire magic for a stripped opening core; distinct from the full-capsule magic
/// so the two decode doors are unmistakable.
const CORE_WIRE_MAGIC: &[u8] = b"ve-capsule.core.v1";

/// Stripped-core wire version. Bump only after an incompatible released format.
const CORE_WIRE_VERSION: u8 = 1;

/// A proof-stripped capsule: the opening core — commitment `C = m·G` plus the
/// per-limb `ElGamal` ciphertexts `{(E_k, D_k)}` — with the seal proof discarded.
///
/// Produced by [`VerifiedCapsule::strip`](crate::VerifiedCapsule::strip) (the
/// honest encoder) or decoded from canonical bytes. A decoded core is
/// **unauthenticated** — the decoder validates framing and the structural decode
/// gates but asserts nothing about provenance; authenticity comes only from
/// [`StrippedCapsule::bind`] + the self-securing [`BoundCapsule::unseal`] (or a
/// quorum signature via [`BoundCapsule::verify_signed`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrippedCapsule {
    elgamal: Vec<LimbCiphertext>,
    c: ProjectivePoint,
}

impl StrippedCapsule {
    /// Build a stripped core from a verified capsule's core view (crate-internal;
    /// the public encoder is [`VerifiedCapsule::strip`](crate::VerifiedCapsule::strip)).
    pub(crate) const fn from_core(elgamal: Vec<LimbCiphertext>, c: ProjectivePoint) -> Self {
        Self { elgamal, c }
    }

    /// Canonical wire bytes: `magic ‖ version ‖ C ‖ (E_k ‖ D_k)×L`. Deterministic
    /// (re-encode equality) and length-fixed: 778 bytes for the frozen params.
    #[must_use]
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            CORE_WIRE_MAGIC.len() + 1 + POINT_LEN + self.elgamal.len() * 2 * POINT_LEN,
        );
        out.extend_from_slice(CORE_WIRE_MAGIC);
        out.push(CORE_WIRE_VERSION);
        out.extend_from_slice(&encode_point(&self.c));
        for ct in &self.elgamal {
            encode_limb(ct, &mut out);
        }
        out
    }

    /// Parse a stripped core from canonical wire bytes — the only deserialization
    /// door. Validates framing, decodes `C` and every limb strictly (off-curve /
    /// non-canonical SEC1 rejected), applies the identity-mask gate (`E_k = O`
    /// rejected, shared with the full-proof decoder), and enforces **re-encode
    /// equality** (the input must be the canonical encoding). Asserts nothing
    /// about authenticity.
    ///
    /// # Errors
    ///
    /// [`Error::PointDecode`] on bad magic/version, a malformed point, a length
    /// mismatch (short/trailing), or non-canonical input; [`Error::DegenerateInput`]
    /// if a segment mask `E_k` decodes to the identity.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let rest = bytes
            .strip_prefix(CORE_WIRE_MAGIC)
            .ok_or(Error::PointDecode("stripped capsule: bad magic"))?;
        let (&version, rest) = rest
            .split_first()
            .ok_or(Error::PointDecode("stripped capsule: truncated header"))?;
        if version != CORE_WIRE_VERSION {
            return Err(Error::PointDecode("stripped capsule: unsupported version"));
        }
        let (c_bytes, mut limbs) = rest
            .split_at_checked(POINT_LEN)
            .ok_or(Error::PointDecode("stripped capsule: truncated commitment"))?;
        let c = decode_point(c_bytes)?;
        let mut elgamal = Vec::with_capacity(LIMB_COUNT);
        for _ in 0..LIMB_COUNT {
            let (e_bytes, after_e) = limbs
                .split_at_checked(POINT_LEN)
                .ok_or(Error::PointDecode("stripped capsule: truncated mask"))?;
            let e = decode_point(e_bytes)?;
            // Identity-mask gate before reading D_k (precedence over truncation),
            // the same gate the full-proof decoder applies.
            reject_identity_mask(&e)?;
            let (d_bytes, after_d) =
                after_e
                    .split_at_checked(POINT_LEN)
                    .ok_or(Error::PointDecode(
                        "stripped capsule: truncated masked point",
                    ))?;
            let d = decode_point(d_bytes)?;
            elgamal.push(LimbCiphertext { e, d });
            limbs = after_d;
        }
        if !limbs.is_empty() {
            return Err(Error::PointDecode("stripped capsule: trailing bytes"));
        }
        let capsule = Self { elgamal, c };
        if capsule.to_canonical_bytes() != bytes {
            return Err(Error::PointDecode(
                "stripped capsule: non-canonical encoding",
            ));
        }
        Ok(capsule)
    }

    /// The core identity digest: `SHA-256` over the core sub-range (`C ‖ masks`)
    /// of the canonical wire — never the magic/version envelope, never a signature
    /// (which signs this digest). Shared with the attestation statement via the
    /// crate-internal `signature::core_digest`, so the bytes signed at
    /// provisioning and the bytes rechecked at recovery are derived identically.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        signature::core_digest(&self.elgamal, &self.c)
    }

    /// Re-run the screens the seal proof `π` would enforce and pin the
    /// authorization, yielding a [`BoundCapsule`] ready to open. Mirrors
    /// [`Capsule::verify`](crate::Capsule::verify) minus the (absent) proof: it
    /// checks the certified target `C == expected_pubkey`, re-runs the structural
    /// mask-degeneracy scan and the aggregate-`Y*` enumerability/NUMS rejection,
    /// validates the context limits, and derives the opening binding from the
    /// expected `(recipient, access_keys)`.
    ///
    /// # Errors
    ///
    /// [`Error::Verification`] if `C` is not `expected_pubkey`, the masks are
    /// degenerate, or `Y*` is publicly enumerable; [`Error::DegenerateInput`] on a
    /// degenerate expected roster or an invalid context.
    pub fn bind<C: Context + ?Sized>(
        &self,
        expected_pubkey: &PublicKey,
        expected_recipient: &PublicKey,
        expected_access_keys: &[PublicKey],
        ctx: &C,
    ) -> Result<BoundCapsule<'_>, Error> {
        // The certified target: C must be the expected recovery public key.
        if self.c != expected_pubkey.point() {
            return Err(Error::Verification(
                "stripped capsule commitment does not match expected public key",
            ));
        }
        // Structural mask gates that live in the proof path but not at decode.
        if let Some(detail) = degenerate_elgamal_mask(&self.elgamal) {
            return Err(Error::Verification(detail));
        }
        // Context limits π's transcript would enforce (nonempty/≤256 B domain,
        // ≤64 KiB binding).
        validate_context_limits(ctx)?;
        // The opening binding (Y*, g*, gate coefficients) from the verified roster.
        let access: Vec<ProjectivePoint> =
            expected_access_keys.iter().map(PublicKey::point).collect();
        let binding = composite::opening_binding(&expected_recipient.point(), &access)?;
        // The aggregate-Y* enumerability/NUMS rejection π runs inside verify.
        reject_degenerate_recovery_key(&binding.y_star, Error::Verification)?;
        let ctx = FrozenContext::capture(ctx)?;
        Ok(BoundCapsule {
            core: CapsuleRef {
                elgamal: &self.elgamal,
                c: self.c,
            },
            binding,
            ctx,
        })
    }
}

/// A stripped core with its authorization pinned: ready to open.
///
/// [`StrippedCapsule::bind`] passed, so this carries the opening binding plus the
/// frozen context, and borrows the core from the [`StrippedCapsule`] it was bound
/// from.
pub struct BoundCapsule<'a> {
    core: CapsuleRef<'a>,
    binding: OpeningBinding,
    ctx: FrozenContext,
}

impl<'a> BoundCapsule<'a> {
    /// Open from the recipient's secret plus authorizer `partials` —
    /// **self-securing**, no signature. The final `g·m == C` recheck (inside the
    /// opening core) means a tampered core fails closed and never yields a wrong
    /// secret, and `bind` already pinned `C == expected_pubkey`, so a successful
    /// open returns the certified target's secret. A corrupted or absent signature
    /// therefore cannot brick recovery of an intact core you hold the key to.
    ///
    /// Use this to recover *your own* core locally. It is not constant-time and,
    /// on a *fabricated* core, whether a limb resolves is a chosen-ciphertext
    /// predicate on the recipient secret — do not run it on untrusted cores in an
    /// attacker-observable setting (promote with a signature there instead).
    ///
    /// # Errors
    ///
    /// [`Error::Verification`] if a gate bucket is not qualifying, a limb is
    /// unrecoverable (incomplete opening / wrong key), or the recovered scalar's
    /// commitment is not `C`.
    pub fn unseal(
        &self,
        recipient: &PrivateKey,
        partials: &[Partial],
    ) -> Result<PrivateKey, Error> {
        let m = opening::unseal_verified(
            self.core,
            &self.binding,
            recipient.scalar(),
            partials,
            &self.ctx,
        )?;
        Ok(PrivateKey::from_scalar(m))
    }

    /// Promote a bound core to the full [`VerifiedCapsule`] by verifying a quorum
    /// `signature` over its [canonical statement](crate::VerifiedCapsule::attestation_message),
    /// against the caller-supplied `verifying_key`: the 32-byte x-only key the
    /// signature verifies under — for a `frost-secp256k1-tr` Taproot key-path
    /// quorum signature this is the **tweaked** output key
    /// `tap_tweak(group_x, None).0`, not the raw group key.
    /// This is the stripped-path dual of [`Capsule::verify`](crate::Capsule::verify):
    /// it re-derives the exact bytes the quorum signed and verifies the signature
    /// in-crate, so the resulting token's `contribute` surface is gated
    /// cryptographically — closing the partial-decryption oracle a bare stripped
    /// core would open. The returned token carries [`Backing::Signature`].
    ///
    /// # Errors
    ///
    /// [`Error::Verification`] if the signature is malformed or does not verify
    /// under `verifying_key`; [`Error::DegenerateInput`] if the context's
    /// `binding_bytes` fails.
    pub fn verify_signed(
        self,
        verifying_key: &[u8; 32],
        signature: &Signature,
    ) -> Result<VerifiedCapsule<'a>, Error> {
        let digest = signature::core_digest(self.core.elgamal, &self.core.c);
        let statement = signature::attestation_statement(&digest, &self.binding, &self.ctx)?;
        let signed = signature::attestation_digest(&statement);
        signature::verify_signature(signature, verifying_key, &signed)?;
        Ok(VerifiedCapsule::from_parts(
            self.core,
            self.binding,
            self.ctx,
            Backing::Signature,
        ))
    }
}

// ── Case-level stripped recovery (the additively-split secret s = Σ σⱼ) ──────

/// Wire magic for a proof-stripped Case bundle; distinct from the single-core
/// magic so the decode doors stay unmistakable.
const CASE_WIRE_MAGIC: &[u8] = b"ve-capsule.case-core.v1";

/// Stripped-Case wire version. Bump only after an incompatible released format.
const CASE_WIRE_VERSION: u8 = 1;

/// Byte length of one stripped-capsule core on the wire (frozen params): the
/// magic/version envelope + `C` + the `L` limb pairs.
const STRIPPED_CAPSULE_WIRE_LEN: usize =
    CORE_WIRE_MAGIC.len() + 1 + POINT_LEN + LIMB_COUNT * 2 * POINT_LEN;

/// A proof-stripped [`Case`](crate::Case): the per-piece opening cores of one
/// additively-split secret (`s = Σ σⱼ`), every proof discarded.
///
/// Produced by [`VerifiedCase::strip`](crate::VerifiedCase::strip). Piece count is
/// `1..=255` (a Case is a small additive split). Like a single stripped core, a
/// decoded bundle is **unauthenticated**; authenticity comes from
/// [`StrippedCase::bind`] + the self-securing [`BoundCase::unseal`] (opened as one
/// summed core anchored on the certified target `M`), or a quorum signature via
/// [`BoundCase::verify_signed`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrippedCase {
    pieces: Vec<StrippedCapsule>,
}

impl StrippedCase {
    /// Bundle stripped piece cores (crate-internal; the public encoder is
    /// [`VerifiedCase::strip`](crate::VerifiedCase::strip)).
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `pieces` is empty or exceeds 255.
    pub(crate) fn from_pieces(pieces: Vec<StrippedCapsule>) -> Result<Self, Error> {
        if pieces.is_empty() {
            return Err(Error::DegenerateInput(
                "a stripped case needs at least one piece",
            ));
        }
        if u8::try_from(pieces.len()).is_err() {
            return Err(Error::DegenerateInput(
                "a stripped case has at most 255 pieces",
            ));
        }
        Ok(Self { pieces })
    }

    /// Canonical wire bytes: `magic ‖ version ‖ count(u8) ‖ {stripped core}×count`.
    /// Deterministic (re-encode equality); each piece is its own 778 B core.
    #[must_use]
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            CASE_WIRE_MAGIC.len() + 2 + self.pieces.len() * STRIPPED_CAPSULE_WIRE_LEN,
        );
        out.extend_from_slice(CASE_WIRE_MAGIC);
        out.push(CASE_WIRE_VERSION);
        // pieces.len() is 1..=255 by construction (from_pieces / from_canonical_bytes).
        out.push(u8::try_from(self.pieces.len()).unwrap_or(u8::MAX));
        for piece in &self.pieces {
            out.extend_from_slice(&piece.to_canonical_bytes());
        }
        out
    }

    /// Parse a stripped Case from canonical wire bytes — the only deserialization
    /// door. Validates framing, decodes each piece strictly (per-piece identity-mask
    /// gate + re-encode equality), and enforces bundle re-encode equality. Asserts
    /// nothing about authenticity.
    ///
    /// # Errors
    ///
    /// [`Error::PointDecode`] on bad magic/version, a zero count, a malformed/short/
    /// trailing piece, or non-canonical input; [`Error::DegenerateInput`] if a piece
    /// mask decodes to the identity.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let rest = bytes
            .strip_prefix(CASE_WIRE_MAGIC)
            .ok_or(Error::PointDecode("stripped case: bad magic"))?;
        let (&version, rest) = rest
            .split_first()
            .ok_or(Error::PointDecode("stripped case: truncated header"))?;
        if version != CASE_WIRE_VERSION {
            return Err(Error::PointDecode("stripped case: unsupported version"));
        }
        let (&count, mut rest) = rest
            .split_first()
            .ok_or(Error::PointDecode("stripped case: truncated count"))?;
        if count == 0 {
            return Err(Error::PointDecode("stripped case: zero pieces"));
        }
        let mut pieces = Vec::with_capacity(usize::from(count));
        for _ in 0..count {
            let (piece, after) = rest
                .split_at_checked(STRIPPED_CAPSULE_WIRE_LEN)
                .ok_or(Error::PointDecode("stripped case: truncated piece"))?;
            pieces.push(StrippedCapsule::from_canonical_bytes(piece)?);
            rest = after;
        }
        if !rest.is_empty() {
            return Err(Error::PointDecode("stripped case: trailing bytes"));
        }
        let case = Self { pieces };
        if case.to_canonical_bytes() != bytes {
            return Err(Error::PointDecode("stripped case: non-canonical encoding"));
        }
        Ok(case)
    }

    /// Re-run the screens each piece's seal proof would enforce and pin the
    /// authorization, yielding a [`BoundCase`] ready to open. Mirrors
    /// [`Case::verify`](crate::Case::verify) minus the (absent) per-piece proofs:
    /// the shared opening binding `(Y*, g*)` is derived once from the roster (the
    /// Case invariant — every piece shares it), the context limits and `Y*`
    /// enumerability are screened once, each piece's masks are screened, and the
    /// piece commitments must sum to the certified target (`Σ Cⱼ == M`).
    ///
    /// # Errors
    ///
    /// [`Error::Verification`] if a piece's masks are degenerate, `Y*` is publicly
    /// enumerable, or `Σ Cⱼ ≠ M`; [`Error::DegenerateInput`] on a degenerate roster
    /// or an invalid context.
    pub fn bind<C: Context + ?Sized>(
        &self,
        target_commitment: &PublicKey,
        recipient: &PublicKey,
        access_keys: &[PublicKey],
        ctx: &C,
    ) -> Result<BoundCase<'_>, Error> {
        validate_context_limits(ctx)?;
        let access: Vec<ProjectivePoint> = access_keys.iter().map(PublicKey::point).collect();
        let binding = composite::opening_binding(&recipient.point(), &access)?;
        reject_degenerate_recovery_key(&binding.y_star, Error::Verification)?;

        let mut sum = ProjectivePoint::IDENTITY;
        let mut pieces = Vec::with_capacity(self.pieces.len());
        let mut digests = Vec::with_capacity(self.pieces.len());
        for piece in &self.pieces {
            if let Some(detail) = degenerate_elgamal_mask(&piece.elgamal) {
                return Err(Error::Verification(detail));
            }
            sum += piece.c;
            digests.push(signature::core_digest(&piece.elgamal, &piece.c));
            pieces.push(CapsuleRef {
                elgamal: &piece.elgamal,
                c: piece.c,
            });
        }
        signature::reject_duplicate_cores(&digests)?;
        if let Some(detail) =
            cross_piece_elgamal_mask_relation(pieces.iter().map(|piece| piece.elgamal))
        {
            return Err(Error::Verification(detail));
        }
        if sum != target_commitment.point() {
            return Err(Error::Verification(
                "stripped case piece commitments do not sum to the expected commitment",
            ));
        }
        Ok(BoundCase {
            pieces,
            binding,
            commitment: target_commitment.point(),
            ctx: FrozenContext::capture(ctx)?,
        })
    }
}

/// A stripped Case with its authorization pinned: ready to open.
///
/// [`StrippedCase::bind`] passed, so this carries the shared opening binding, the
/// certified commitment `M`, and the frozen context, and borrows the per-piece
/// cores from the [`StrippedCase`].
pub struct BoundCase<'a> {
    pieces: Vec<CapsuleRef<'a>>,
    binding: OpeningBinding,
    commitment: ProjectivePoint,
    ctx: FrozenContext,
}

impl<'a> BoundCase<'a> {
    /// Open the summed case core → the reconstructed secret `s = Σ σⱼ`, rechecked
    /// `g·s == M`. **Self-securing**, no signature: unauthenticated per-piece
    /// commitments are treated as storage decomposition, not independent
    /// statements, so opening happens once over the summed ciphertext and certified
    /// aggregate target. Recipient-only when `partials` is empty. The same
    /// recover-your-own-core / not-constant-time caveat as [`BoundCapsule::unseal`]
    /// applies.
    ///
    /// # Errors
    ///
    /// [`Error::Verification`] if a piece is not fully opened (a gate short, a limb
    /// unrecoverable) or the reconstructed secret's commitment is not `M`.
    pub fn unseal(
        &self,
        recipient: &PrivateKey,
        partials: &[Partial],
    ) -> Result<PrivateKey, Error> {
        let mut aggregate = vec![
            LimbCiphertext {
                e: ProjectivePoint::IDENTITY,
                d: ProjectivePoint::IDENTITY,
            };
            LIMB_COUNT
        ];
        let mut weighted_access_masks = [ProjectivePoint::IDENTITY; LIMB_COUNT];
        for piece in &self.pieces {
            for (sum, ct) in aggregate.iter_mut().zip(piece.elgamal.iter()) {
                sum.e += ct.e;
                sum.d += ct.d;
            }

            let mut verified: Vec<&Partial> = Vec::new();
            for partial in partials {
                if partial.verify(*piece, &self.binding, &self.ctx).is_err() {
                    continue;
                }
                if verified.contains(&partial) {
                    continue;
                }
                verified.push(partial);
            }
            for gate in &self.binding.gates {
                let mut w_sum = ProjectivePoint::IDENTITY;
                for partial in &verified {
                    if partial.gate() == *gate {
                        w_sum += partial.w_g();
                    }
                }
                if w_sum != *gate {
                    return Err(Error::Verification(
                        "access gate not qualifying: Σ W ≠ Y_access",
                    ));
                }
            }
            for (j, access_mask) in weighted_access_masks.iter_mut().enumerate() {
                for partial in &verified {
                    let weight = self
                        .binding
                        .gate_weight(&partial.gate())
                        .ok_or(Error::Verification("partial gate weight missing"))?;
                    *access_mask += partial.masks()[j] * weight;
                }
            }
        }

        // A decoded case is unauthenticated, so the summed handle needs the same
        // structural screen as the compact aggregate path. Per-piece masks can be
        // individually safe while their sum leaks a bounded aggregate limb.
        if let Some(detail) = degenerate_elgamal_mask(&aggregate) {
            return Err(Error::Verification(detail));
        }
        let max_limb_exclusive = u64::try_from(self.pieces.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(LIMB_MODULUS);
        let s = opening::open_core_with_access_masks(
            CapsuleRef {
                elgamal: &aggregate,
                c: self.commitment,
            },
            &self.binding,
            recipient.scalar(),
            &weighted_access_masks,
            max_limb_exclusive,
        )?;
        Ok(PrivateKey::from_scalar(s))
    }

    /// Promote a bound case to the full [`VerifiedCase`] by verifying a quorum
    /// `signature` over its [canonical statement](crate::VerifiedCase::attestation_message),
    /// against the caller-supplied `verifying_key`: the 32-byte x-only key the
    /// signature verifies under — for a `frost-secp256k1-tr` Taproot key-path
    /// quorum signature this is the **tweaked** output key
    /// `tap_tweak(group_x, None).0`, not the raw group key.
    /// The stripped-path dual of [`Case::verify`](crate::Case::verify): it re-derives
    /// the exact bytes the quorum signed (the sorted per-piece core digests + `M` +
    /// `recipient ‖ g* ‖ Y*` + `ctx`/`params`), reduces them to the 32-byte
    /// attestation digest, and verifies the signature in-crate — so the resulting
    /// token's `contribute` surface is gated cryptographically. The returned token
    /// carries [`Backing::Signature`].
    ///
    /// # Errors
    ///
    /// [`Error::Verification`] if the signature is malformed or does not verify
    /// under `verifying_key`; [`Error::DegenerateInput`] if the context's
    /// `binding_bytes` fails.
    pub fn verify_signed(
        self,
        verifying_key: &[u8; 32],
        signature: &Signature,
    ) -> Result<VerifiedCase<'a>, Error> {
        let piece_digests: Vec<[u8; 32]> = self
            .pieces
            .iter()
            .map(|piece| signature::core_digest(piece.elgamal, &piece.c))
            .collect();
        let statement = signature::case_attestation_statement(
            &piece_digests,
            &self.commitment,
            &self.binding,
            &self.ctx,
        )?;
        let signed = signature::attestation_digest(&statement);
        signature::verify_signature(signature, verifying_key, &signed)?;
        Ok(VerifiedCase::from_parts(
            self.pieces,
            self.binding,
            self.commitment,
            self.ctx,
            Backing::Signature,
        ))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;
    use crate::capsule::Capsule;
    use crate::case::Case;
    use crate::elgamal::IDENTITY_MASK_DETAIL;
    use crate::generators::g;
    use crate::signature::frost_test_support::{group_xonly, keygen, sign};
    use crate::signature::{Backing, Signature, attestation_digest};
    use k256::Scalar;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::borrow::Cow;

    /// Core size for the frozen params: magic(18) + ver(1) + C(33) + 11·(33+33).
    const CORE_LEN: usize = 18 + 1 + 33 + 11 * (33 + 33);

    struct TestCtx;
    impl Context for TestCtx {
        fn domain(&self) -> &'static str {
            "ve-capsule.stripped-test"
        }
        fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
            Ok(Cow::Borrowed(b"stripped-binding"))
        }
    }

    fn private_key(rng: &mut StdRng) -> PrivateKey {
        use rand::RngCore;
        loop {
            let mut bytes = [0u8; 32];
            rng.fill_bytes(&mut bytes);
            if let Ok(k) = PrivateKey::from_secret(&bytes) {
                return k;
            }
        }
    }

    fn private_key_from_u64(value: u64) -> PrivateKey {
        let mut bytes = [0u8; 32];
        bytes[24..].copy_from_slice(&value.to_be_bytes());
        PrivateKey::from_secret(&bytes).unwrap()
    }

    #[test]
    fn core_is_778_bytes() {
        let mut rng = StdRng::seed_from_u64(0x57_01_00_01);
        let recipient = private_key(&mut rng);
        let m = private_key(&mut rng);
        let capsule = Capsule::builder(&m, &recipient.public_key(), &TestCtx)
            .seal()
            .unwrap();
        let stripped = capsule
            .verify(&m.public_key(), &recipient.public_key(), &[], &TestCtx)
            .unwrap()
            .strip();
        assert_eq!(stripped.to_canonical_bytes().len(), 778);
        assert_eq!(CORE_LEN, 778, "the hand-computed budget matches the wire");
    }

    #[test]
    fn wire_round_trip_and_reencode_equality() {
        let mut rng = StdRng::seed_from_u64(0x57_01_00_02);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let m = private_key(&mut rng);
        let capsule = Capsule::builder(&m, &recipient.public_key(), &TestCtx)
            .access_key(&access.public_key())
            .seal()
            .unwrap();
        let stripped = capsule
            .verify(
                &m.public_key(),
                &recipient.public_key(),
                &[access.public_key()],
                &TestCtx,
            )
            .unwrap()
            .strip();
        let bytes = stripped.to_canonical_bytes();
        let parsed = StrippedCapsule::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(parsed, stripped, "decode is the inverse of encode");
        assert_eq!(parsed.to_canonical_bytes(), bytes, "re-encode equality");
    }

    #[test]
    fn digest_is_stable_and_distinguishes_cores() {
        let mut rng = StdRng::seed_from_u64(0x57_01_00_03);
        let recipient = private_key(&mut rng);
        let m1 = private_key(&mut rng);
        let m2 = private_key(&mut rng);
        let strip = |m: &PrivateKey| {
            let capsule = Capsule::builder(m, &recipient.public_key(), &TestCtx)
                .seal()
                .unwrap();
            capsule
                .verify(&m.public_key(), &recipient.public_key(), &[], &TestCtx)
                .unwrap()
                .strip()
        };
        let s1 = strip(&m1);
        let s2 = strip(&m2);
        // Stable across calls and across a wire round trip.
        assert_eq!(s1.digest(), s1.digest());
        assert_eq!(
            StrippedCapsule::from_canonical_bytes(&s1.to_canonical_bytes())
                .unwrap()
                .digest(),
            s1.digest(),
        );
        // Distinct cores → distinct digests.
        assert_ne!(s1.digest(), s2.digest());
    }

    #[test]
    fn decode_rejects_identity_mask() {
        let mut rng = StdRng::seed_from_u64(0x57_01_00_04);
        let recipient = private_key(&mut rng);
        let m = private_key(&mut rng);
        let capsule = Capsule::builder(&m, &recipient.public_key(), &TestCtx)
            .seal()
            .unwrap();
        let good = capsule
            .verify(&m.public_key(), &recipient.public_key(), &[], &TestCtx)
            .unwrap()
            .strip();
        // Poison the first mask to the identity and re-encode; decode must reject.
        let mut elgamal = good.elgamal.clone();
        elgamal[0].e = ProjectivePoint::IDENTITY;
        let poisoned = StrippedCapsule::from_core(elgamal, good.c);
        let bytes = poisoned.to_canonical_bytes();
        assert!(matches!(
            StrippedCapsule::from_canonical_bytes(&bytes),
            Err(Error::DegenerateInput(IDENTITY_MASK_DETAIL))
        ));
    }

    #[test]
    fn decode_rejects_framing_faults() {
        let mut rng = StdRng::seed_from_u64(0x57_01_00_05);
        let recipient = private_key(&mut rng);
        let m = private_key(&mut rng);
        let bytes = Capsule::builder(&m, &recipient.public_key(), &TestCtx)
            .seal()
            .unwrap()
            .verify(&m.public_key(), &recipient.public_key(), &[], &TestCtx)
            .unwrap()
            .strip()
            .to_canonical_bytes();
        // bad magic
        assert!(StrippedCapsule::from_canonical_bytes(b"not-a-core").is_err());
        // truncated
        assert!(StrippedCapsule::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());
        // trailing byte
        let mut trailing = bytes;
        trailing.push(0);
        assert!(StrippedCapsule::from_canonical_bytes(&trailing).is_err());
    }

    #[test]
    fn strip_then_recover_no_signature_gated() {
        // The headline path: seal → verify → strip → bytes → parse → bind →
        // unseal, with a partial made from the proof-side verified capsule (so
        // this also exercises proof-side contribute → stripped-side unseal).
        let mut rng = StdRng::seed_from_u64(0x57_01_01_01);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let m = private_key(&mut rng);
        let (rpk, apk, mpk) = (recipient.public_key(), access.public_key(), m.public_key());

        let capsule = Capsule::builder(&m, &rpk, &TestCtx)
            .access_key(&apk)
            .seal()
            .unwrap();
        let vc = capsule.verify(&mpk, &rpk, &[apk], &TestCtx).unwrap();
        let partial = vc.contribute(&access).unwrap();
        let bytes = vc.strip().to_canonical_bytes();

        let parsed = StrippedCapsule::from_canonical_bytes(&bytes).unwrap();
        let bound = parsed.bind(&mpk, &rpk, &[apk], &TestCtx).unwrap();
        let recovered = bound.unseal(&recipient, &[partial]).unwrap();
        assert_eq!(
            recovered.public_key(),
            mpk,
            "stripped gated core opens to the sealed m with no signature"
        );
    }

    #[test]
    fn strip_then_recover_no_signature_ungated() {
        let mut rng = StdRng::seed_from_u64(0x57_01_01_02);
        let recipient = private_key(&mut rng);
        let m = private_key(&mut rng);
        let (rpk, mpk) = (recipient.public_key(), m.public_key());

        let capsule = Capsule::builder(&m, &rpk, &TestCtx).seal().unwrap();
        let bytes = capsule
            .verify(&mpk, &rpk, &[], &TestCtx)
            .unwrap()
            .strip()
            .to_canonical_bytes();

        let parsed = StrippedCapsule::from_canonical_bytes(&bytes).unwrap();
        let bound = parsed.bind(&mpk, &rpk, &[], &TestCtx).unwrap();
        let recovered = bound.unseal(&recipient, &[]).unwrap();
        assert_eq!(recovered.public_key(), mpk);
    }

    #[test]
    fn bind_rejects_wrong_expected_pubkey() {
        let mut rng = StdRng::seed_from_u64(0x57_01_02_01);
        let recipient = private_key(&mut rng);
        let m = private_key(&mut rng);
        let other = private_key(&mut rng);
        let (rpk, mpk) = (recipient.public_key(), m.public_key());
        let stripped = Capsule::builder(&m, &rpk, &TestCtx)
            .seal()
            .unwrap()
            .verify(&mpk, &rpk, &[], &TestCtx)
            .unwrap()
            .strip();
        let bytes = stripped.to_canonical_bytes();
        let parsed = StrippedCapsule::from_canonical_bytes(&bytes).unwrap();
        assert!(
            matches!(
                parsed.bind(&other.public_key(), &rpk, &[], &TestCtx),
                Err(Error::Verification(_))
            ),
            "C == expected_pubkey gate must reject a mismatched target"
        );
    }

    #[test]
    fn bind_rejects_wrong_roster() {
        let mut rng = StdRng::seed_from_u64(0x57_01_02_02);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let wrong = private_key(&mut rng);
        let m = private_key(&mut rng);
        let (rpk, apk, mpk) = (recipient.public_key(), access.public_key(), m.public_key());
        let stripped = Capsule::builder(&m, &rpk, &TestCtx)
            .access_key(&apk)
            .seal()
            .unwrap()
            .verify(&mpk, &rpk, &[apk], &TestCtx)
            .unwrap()
            .strip();
        let bound = stripped
            .bind(&mpk, &rpk, &[wrong.public_key()], &TestCtx)
            .unwrap();
        // A wrong roster yields a different Y*/g*, so a partial for the real gate
        // fails and the recipient-only strip leaves the bucket short.
        assert!(bound.unseal(&recipient, &[]).is_err());
    }

    // ── Signature backing: promote a stripped core with a real FROST signature ──

    struct OtherCtx;
    impl Context for OtherCtx {
        fn domain(&self) -> &'static str {
            "ve-capsule.stripped-test"
        }
        fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
            Ok(Cow::Borrowed(b"a-different-binding"))
        }
    }

    #[test]
    fn signed_core_promotes_contributes_and_unseals() {
        // The headline signature path: provision → sign attestation_message →
        // strip → bind → verify_signed → contribute + unseal, all with a real
        // FROST(secp256k1)-TR quorum signature and no proof on the wire.
        let mut rng = StdRng::seed_from_u64(0x57_51_01_01);
        let (packages, pubkeys) = keygen(2, 3);
        let vk = group_xonly(&pubkeys);

        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let m = private_key(&mut rng);
        let (rpk, apk, mpk) = (recipient.public_key(), access.public_key(), m.public_key());

        let capsule = Capsule::builder(&m, &rpk, &TestCtx)
            .access_key(&apk)
            .seal()
            .unwrap();
        let vc = capsule.verify(&mpk, &rpk, &[apk], &TestCtx).unwrap();
        let msg = vc.attestation_message().unwrap();
        let sig = Signature::schnorr(sign(&packages, &pubkeys, &attestation_digest(&msg)));

        let bytes = vc.strip().to_canonical_bytes();
        let parsed = StrippedCapsule::from_canonical_bytes(&bytes).unwrap();
        let bound = parsed.bind(&mpk, &rpk, &[apk], &TestCtx).unwrap();
        let vc2 = bound.verify_signed(&vk, &sig).unwrap();

        assert_eq!(
            vc2.backing(),
            Backing::Signature,
            "a signature-promoted token reports the signature backing"
        );
        let partial = vc2.contribute(&access).unwrap();
        let recovered = vc2.unseal(&recipient, &[partial]).unwrap();
        assert_eq!(
            recovered.public_key(),
            mpk,
            "signature-backed token contributes and unseals like a proof-backed one"
        );
    }

    #[test]
    fn cross_path_partial_interchange() {
        // A partial is bound to (core, binding, ctx), not to how the token was
        // established — so a proof-side partial opens a signature-side token and
        // vice versa.
        let mut rng = StdRng::seed_from_u64(0x57_51_02_01);
        let (packages, pubkeys) = keygen(2, 3);
        let vk = group_xonly(&pubkeys);

        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let m = private_key(&mut rng);
        let (rpk, apk, mpk) = (recipient.public_key(), access.public_key(), m.public_key());

        let capsule = Capsule::builder(&m, &rpk, &TestCtx)
            .access_key(&apk)
            .seal()
            .unwrap();
        let vc = capsule.verify(&mpk, &rpk, &[apk], &TestCtx).unwrap();
        let sig = Signature::schnorr(sign(
            &packages,
            &pubkeys,
            &attestation_digest(&vc.attestation_message().unwrap()),
        ));

        let bytes = vc.strip().to_canonical_bytes();
        let parsed = StrippedCapsule::from_canonical_bytes(&bytes).unwrap();
        let vc2 = parsed
            .bind(&mpk, &rpk, &[apk], &TestCtx)
            .unwrap()
            .verify_signed(&vk, &sig)
            .unwrap();

        let p_proof = vc.contribute(&access).unwrap();
        let p_sig = vc2.contribute(&access).unwrap();
        // proof-side partial → signature-side unseal
        assert_eq!(
            vc2.unseal(&recipient, &[p_proof]).unwrap().public_key(),
            mpk
        );
        // signature-side partial → proof-side unseal
        assert_eq!(vc.unseal(&recipient, &[p_sig]).unwrap().public_key(), mpk);
    }

    #[test]
    fn verify_signed_rejects_fabricated_core() {
        // A signature over capsule A's statement must not promote capsule B's core
        // (different digest ⇒ different statement ⇒ the signature does not cover it).
        let mut rng = StdRng::seed_from_u64(0x57_51_03_01);
        let (packages, pubkeys) = keygen(2, 3);
        let vk = group_xonly(&pubkeys);

        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let (rpk, apk) = (recipient.public_key(), access.public_key());

        let m_a = private_key(&mut rng);
        let capsule_a = Capsule::builder(&m_a, &rpk, &TestCtx)
            .access_key(&apk)
            .seal()
            .unwrap();
        let vc_a = capsule_a
            .verify(&m_a.public_key(), &rpk, &[apk], &TestCtx)
            .unwrap();
        let sig_a = Signature::schnorr(sign(
            &packages,
            &pubkeys,
            &attestation_digest(&vc_a.attestation_message().unwrap()),
        ));

        let m_b = private_key(&mut rng);
        let capsule_b = Capsule::builder(&m_b, &rpk, &TestCtx)
            .access_key(&apk)
            .seal()
            .unwrap();
        let vc_b = capsule_b
            .verify(&m_b.public_key(), &rpk, &[apk], &TestCtx)
            .unwrap();
        let bytes_b = vc_b.strip().to_canonical_bytes();
        let parsed_b = StrippedCapsule::from_canonical_bytes(&bytes_b).unwrap();
        let bound_b = parsed_b
            .bind(&m_b.public_key(), &rpk, &[apk], &TestCtx)
            .unwrap();
        assert!(
            bound_b.verify_signed(&vk, &sig_a).is_err(),
            "A's signature must not promote B's core"
        );
    }

    #[test]
    fn verify_signed_rejects_wrong_key_and_mismatched_context() {
        let mut rng = StdRng::seed_from_u64(0x57_51_04_01);
        let (packages, pubkeys) = keygen(2, 3);
        let vk = group_xonly(&pubkeys);

        let recipient = private_key(&mut rng);
        let m = private_key(&mut rng);
        let (rpk, mpk) = (recipient.public_key(), m.public_key());

        let capsule = Capsule::builder(&m, &rpk, &TestCtx).seal().unwrap();
        let vc = capsule.verify(&mpk, &rpk, &[], &TestCtx).unwrap();
        let sig = Signature::schnorr(sign(
            &packages,
            &pubkeys,
            &attestation_digest(&vc.attestation_message().unwrap()),
        ));
        let bytes = vc.strip().to_canonical_bytes();
        let parsed = StrippedCapsule::from_canonical_bytes(&bytes).unwrap();

        // Wrong verifying key (a fresh group).
        let (_p2, pubkeys2) = keygen(2, 3);
        assert!(
            parsed
                .bind(&mpk, &rpk, &[], &TestCtx)
                .unwrap()
                .verify_signed(&group_xonly(&pubkeys2), &sig)
                .is_err(),
            "a signature must not verify under a different group key"
        );

        // Mismatched context: the statement binds ctx, so binding under a different
        // context than provisioning changes the message the signature must cover.
        assert!(
            parsed
                .bind(&mpk, &rpk, &[], &OtherCtx)
                .unwrap()
                .verify_signed(&vk, &sig)
                .is_err(),
            "binding under a different context must reject the provisioning signature"
        );
    }

    // ── Case-level stripped recovery (recipient-only, self-securing) ──────────

    /// `Σ pieces·G`, as a `PublicKey` — the certified target `M`.
    fn case_commitment(pieces: &[&PrivateKey]) -> PublicKey {
        let sum = pieces.iter().fold(ProjectivePoint::IDENTITY, |acc, p| {
            acc + p.public_key().point()
        });
        PublicKey::from_canonical_bytes(&encode_point(&sum)).unwrap()
    }

    #[test]
    fn case_strip_recover_recipient_only_ungated() {
        // The recipient-only case shape: each producer seals its additive piece σⱼ
        // to the recovery key with no gates; strip the proofs; the recipient opens
        // the stripped bundle alone and sums to s = Σ σⱼ. No signature.
        let mut rng = StdRng::seed_from_u64(0x5C_A5_01_01);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let s1 = private_key(&mut rng);
        let s2 = private_key(&mut rng);
        let s3 = private_key(&mut rng);
        let m = case_commitment(&[&s1, &s2, &s3]);

        let case = Case::new(vec![
            Capsule::builder(&s1, &rpk, &TestCtx).seal().unwrap(),
            Capsule::builder(&s2, &rpk, &TestCtx).seal().unwrap(),
            Capsule::builder(&s3, &rpk, &TestCtx).seal().unwrap(),
        ])
        .unwrap();
        let bytes = case
            .verify(&m, &rpk, &[], &TestCtx)
            .unwrap()
            .strip()
            .unwrap()
            .to_canonical_bytes();

        let parsed = StrippedCase::from_canonical_bytes(&bytes).unwrap();
        let bound = parsed.bind(&m, &rpk, &[], &TestCtx).unwrap();
        let recovered = bound.unseal(&recipient, &[]).unwrap();
        assert_eq!(
            recovered.public_key(),
            m,
            "stripped case opens to s = Σ σⱼ with no signature"
        );
    }

    #[test]
    fn stripped_case_balanced_piece_limb_shifts_do_not_expose_piece_range() {
        // A stripped Case has only an aggregate certified target M = ΣC_i. If
        // opening checks each unauthenticated piece independently, an attacker can
        // rebalance C_i and D_i across pieces and learn whether a shifted piece
        // limb stayed inside its individual BSGS range. Opening the summed core
        // once makes net-zero reallocations aggregate-equivalent instead.
        let mut rng = StdRng::seed_from_u64(0x5C_A5_01_02);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let s1 = private_key_from_u64(5);
        let s2 = private_key_from_u64(100);
        let m = case_commitment(&[&s1, &s2]);
        let case = Case::new(vec![
            Capsule::builder(&s1, &rpk, &TestCtx).seal().unwrap(),
            Capsule::builder(&s2, &rpk, &TestCtx).seal().unwrap(),
        ])
        .unwrap();
        let parsed = StrippedCase::from_canonical_bytes(
            &case
                .verify(&m, &rpk, &[], &TestCtx)
                .unwrap()
                .strip()
                .unwrap()
                .to_canonical_bytes(),
        )
        .unwrap();

        let balanced_shift = |case: &mut StrippedCase, delta: u64| {
            let shift = g() * Scalar::from(delta);
            case.pieces[0].c += shift;
            case.pieces[0].elgamal[0].d += shift;
            case.pieces[1].c -= shift;
            case.pieces[1].elgamal[0].d -= shift;
        };

        let mut still_piece_range = parsed.clone();
        balanced_shift(&mut still_piece_range, 1);
        let recovered = still_piece_range
            .bind(&m, &rpk, &[], &TestCtx)
            .unwrap()
            .unseal(&recipient, &[])
            .unwrap();
        assert_eq!(recovered.public_key(), m);

        let mut outside_piece_range = parsed.clone();
        balanced_shift(&mut outside_piece_range, LIMB_MODULUS - 5);
        let recovered = outside_piece_range
            .bind(&m, &rpk, &[], &TestCtx)
            .unwrap()
            .unseal(&recipient, &[])
            .unwrap();
        assert_eq!(
            recovered.public_key(),
            m,
            "balanced per-piece rewrites must not reveal individual limb range"
        );

        let mut changed_aggregate = parsed;
        let shift = g() * Scalar::ONE;
        changed_aggregate.pieces[0].c += shift;
        changed_aggregate.pieces[1].c -= shift;
        changed_aggregate.pieces[0].elgamal[0].d += shift;
        let err = changed_aggregate
            .bind(&m, &rpk, &[], &TestCtx)
            .unwrap()
            .unseal(&recipient, &[])
            .err()
            .unwrap();
        assert!(
            err.to_string().contains("opening failed"),
            "changing the aggregate ciphertext must still fail: {err}"
        );
    }

    #[test]
    fn case_strip_recover_gated() {
        // Gated case: an authorizer's partials (from the proof-verified case) open
        // the stripped bundle — proof-side contribute → stripped-side unseal.
        let mut rng = StdRng::seed_from_u64(0x5C_A5_02_01);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let (rpk, apk) = (recipient.public_key(), access.public_key());
        let s1 = private_key(&mut rng);
        let s2 = private_key(&mut rng);
        let m = case_commitment(&[&s1, &s2]);

        let case = Case::new(vec![
            Capsule::builder(&s1, &rpk, &TestCtx)
                .access_key(&apk)
                .seal()
                .unwrap(),
            Capsule::builder(&s2, &rpk, &TestCtx)
                .access_key(&apk)
                .seal()
                .unwrap(),
        ])
        .unwrap();
        let vc = case.verify(&m, &rpk, &[apk], &TestCtx).unwrap();
        let partials = vc.contribute(&access).unwrap();
        let bytes = vc.strip().unwrap().to_canonical_bytes();

        let parsed = StrippedCase::from_canonical_bytes(&bytes).unwrap();
        let bound = parsed.bind(&m, &rpk, &[apk], &TestCtx).unwrap();
        let recovered = bound.unseal(&recipient, &partials).unwrap();
        assert_eq!(recovered.public_key(), m, "gated stripped case opens");
    }

    #[test]
    fn case_wire_round_trip_and_framing() {
        let mut rng = StdRng::seed_from_u64(0x5C_A5_03_01);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let s1 = private_key(&mut rng);
        let s2 = private_key(&mut rng);
        let m = case_commitment(&[&s1, &s2]);
        let case = Case::new(vec![
            Capsule::builder(&s1, &rpk, &TestCtx).seal().unwrap(),
            Capsule::builder(&s2, &rpk, &TestCtx).seal().unwrap(),
        ])
        .unwrap();
        let stripped = case
            .verify(&m, &rpk, &[], &TestCtx)
            .unwrap()
            .strip()
            .unwrap();
        let bytes = stripped.to_canonical_bytes();
        // Two 778 B pieces + magic + ver(1) + count(1).
        assert_eq!(bytes.len(), CASE_WIRE_MAGIC.len() + 2 + 2 * 778);
        let parsed = StrippedCase::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(parsed, stripped);
        assert_eq!(parsed.to_canonical_bytes(), bytes, "re-encode equality");
        // bad magic / zero-count / trailing
        assert!(StrippedCase::from_canonical_bytes(b"not-a-case").is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(StrippedCase::from_canonical_bytes(&trailing).is_err());
    }

    #[test]
    fn case_bind_rejects_wrong_target() {
        let mut rng = StdRng::seed_from_u64(0x5C_A5_04_01);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let s1 = private_key(&mut rng);
        let s2 = private_key(&mut rng);
        let m = case_commitment(&[&s1, &s2]);
        let wrong = case_commitment(&[&s1]); // missing piece 2
        let case = Case::new(vec![
            Capsule::builder(&s1, &rpk, &TestCtx).seal().unwrap(),
            Capsule::builder(&s2, &rpk, &TestCtx).seal().unwrap(),
        ])
        .unwrap();
        let stripped = case
            .verify(&m, &rpk, &[], &TestCtx)
            .unwrap()
            .strip()
            .unwrap();
        assert!(
            matches!(
                stripped.bind(&wrong, &rpk, &[], &TestCtx),
                Err(Error::Verification(_))
            ),
            "ΣCⱼ must equal the expected commitment"
        );
    }

    #[test]
    fn case_bind_rejects_duplicate_cores() {
        // A misbehaving dealer ships two byte-identical piece cores. They sum to a
        // valid ΣCⱼ == M (= 2·C), but identical cores share DLEQ bases and would
        // overcount a gate bucket — bind must reject before that footgun is reachable.
        let mut rng = StdRng::seed_from_u64(0x5C_A5_05_01);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let s = private_key(&mut rng);
        let core = Capsule::builder(&s, &rpk, &TestCtx)
            .seal()
            .unwrap()
            .verify(&s.public_key(), &rpk, &[], &TestCtx)
            .unwrap()
            .strip();
        // The duplicate sums to the doubled commitment M = 2·C.
        let m = PublicKey::from_canonical_bytes(&encode_point(
            &(s.public_key().point() + s.public_key().point()),
        ))
        .unwrap();
        let dup = StrippedCase::from_pieces(vec![core.clone(), core]).unwrap();
        assert!(
            matches!(
                dup.bind(&m, &rpk, &[], &TestCtx),
                Err(Error::Verification(_))
            ),
            "bind must reject duplicate piece cores"
        );
    }

    // ── Case signature backing: promote a stripped Case with a real FROST sig ──

    #[test]
    fn signed_case_promotes_contributes_and_unseals() {
        // The Case signature path: provision a gated split → sign the Case
        // attestation digest → strip → bind → verify_signed → contribute + unseal,
        // all with a real FROST(secp256k1)-TR quorum signature and no proofs.
        let mut rng = StdRng::seed_from_u64(0x5C_51_01_01);
        let (packages, pubkeys) = keygen(2, 3);
        let vk = group_xonly(&pubkeys);

        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let (rpk, apk) = (recipient.public_key(), access.public_key());
        let s1 = private_key(&mut rng);
        let s2 = private_key(&mut rng);
        let m = case_commitment(&[&s1, &s2]);

        let case = Case::new(vec![
            Capsule::builder(&s1, &rpk, &TestCtx)
                .access_key(&apk)
                .seal()
                .unwrap(),
            Capsule::builder(&s2, &rpk, &TestCtx)
                .access_key(&apk)
                .seal()
                .unwrap(),
        ])
        .unwrap();
        let vc = case.verify(&m, &rpk, &[apk], &TestCtx).unwrap();
        let sig = Signature::schnorr(sign(
            &packages,
            &pubkeys,
            &attestation_digest(&vc.attestation_message().unwrap()),
        ));

        let bytes = vc.strip().unwrap().to_canonical_bytes();
        let parsed = StrippedCase::from_canonical_bytes(&bytes).unwrap();
        let vc2 = parsed
            .bind(&m, &rpk, &[apk], &TestCtx)
            .unwrap()
            .verify_signed(&vk, &sig)
            .unwrap();

        assert_eq!(
            vc2.backing(),
            Backing::Signature,
            "a signature-promoted case reports the signature backing"
        );
        let partials = vc2.contribute(&access).unwrap();
        let recovered = vc2.unseal(&recipient, &partials).unwrap();
        assert_eq!(
            recovered.public_key(),
            m,
            "signature-backed case contributes and unseals like a proof-backed one"
        );
    }

    #[test]
    fn signed_case_rejects_wrong_key_and_cross_case() {
        let mut rng = StdRng::seed_from_u64(0x5C_51_02_01);
        let (packages, pubkeys) = keygen(2, 3);
        let vk = group_xonly(&pubkeys);

        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let s1 = private_key(&mut rng);
        let s2 = private_key(&mut rng);
        let m_a = case_commitment(&[&s1, &s2]);
        let case_a = Case::new(vec![
            Capsule::builder(&s1, &rpk, &TestCtx).seal().unwrap(),
            Capsule::builder(&s2, &rpk, &TestCtx).seal().unwrap(),
        ])
        .unwrap();
        let vc_a = case_a.verify(&m_a, &rpk, &[], &TestCtx).unwrap();
        let sig_a = Signature::schnorr(sign(
            &packages,
            &pubkeys,
            &attestation_digest(&vc_a.attestation_message().unwrap()),
        ));
        let parsed_a =
            StrippedCase::from_canonical_bytes(&vc_a.strip().unwrap().to_canonical_bytes())
                .unwrap();

        // Wrong verifying key (a fresh group) cannot promote case A.
        let (_p2, pubkeys2) = keygen(2, 3);
        assert!(
            parsed_a
                .bind(&m_a, &rpk, &[], &TestCtx)
                .unwrap()
                .verify_signed(&group_xonly(&pubkeys2), &sig_a)
                .is_err(),
            "a case signature must not verify under a different group key"
        );

        // A different case (different pieces ⇒ different digests/M ⇒ different
        // statement) cannot be promoted by case A's signature.
        let t1 = private_key(&mut rng);
        let t2 = private_key(&mut rng);
        let m_b = case_commitment(&[&t1, &t2]);
        let case_b = Case::new(vec![
            Capsule::builder(&t1, &rpk, &TestCtx).seal().unwrap(),
            Capsule::builder(&t2, &rpk, &TestCtx).seal().unwrap(),
        ])
        .unwrap();
        let parsed_b = StrippedCase::from_canonical_bytes(
            &case_b
                .verify(&m_b, &rpk, &[], &TestCtx)
                .unwrap()
                .strip()
                .unwrap()
                .to_canonical_bytes(),
        )
        .unwrap();
        assert!(
            parsed_b
                .bind(&m_b, &rpk, &[], &TestCtx)
                .unwrap()
                .verify_signed(&vk, &sig_a)
                .is_err(),
            "case A's signature must not promote case B"
        );
    }
}
