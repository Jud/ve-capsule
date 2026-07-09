//! The `Case`: the verifiable opening of one additively-split secret.
//!
//! When a scalar is represented as additive pieces `s = Σⱼ sⱼ`, no single
//! producer needs to hold `s`; each producer seals only its piece `sⱼ` to the
//! recipient behind the same access policy. A `Case` bundles those
//! piece-capsules: **every capsule shares `(recipient, access policy, ctx)`** and
//! differs only in its piece `sⱼ` (commitment `Mⱼ = sⱼ·G`). The recipient
//! recovers by opening every piece and summing.
//!
//! Completeness is one homomorphic equation: the target commitment is
//! `M = s·G = Σ Mⱼ`, so [`Case::verify`] checks
//! **`Σ Mⱼ == M`** — anything that changes the committed secret (a missing or
//! tampered piece) fails. That single check gives completeness *and* correctness,
//! and subsumes a per-case signature given an authentic `M`.

use crate::assembly;
use crate::capsule::{Capsule, Contribute, FrozenContext, PrivateKey, PublicKey, Unseal};
use crate::composite::{self, OpeningBinding};
use crate::context::Context;
use crate::error::Error;
use crate::generators::g;
use crate::opening::{self, CapsuleRef, Partial};
use crate::signature::{self, Backing};
use crate::stripped::{StrippedCapsule, StrippedCase};
use k256::{ProjectivePoint, Scalar};
use rand_core::OsRng;

/// A bundle of piece-capsules sharing one `(recipient, access policy, ctx)` — the
/// additive split of one secret to one recipient.
pub struct Case {
    capsules: Vec<Capsule>,
}

impl Case {
    /// Bundle piece-capsules into a `Case`. Requires at least one.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `capsules` is empty.
    pub fn new(capsules: Vec<Capsule>) -> Result<Self, Error> {
        if capsules.is_empty() {
            return Err(Error::DegenerateInput("a case needs at least one capsule"));
        }
        Ok(Self { capsules })
    }

    /// Confirm the additive split against one authorization tuple: every capsule is
    /// well-formed and shares `(expected_recipient, expected_access_keys, ctx)`,
    /// and the piece commitments sum to `expected_commitment` (**`Σ Mⱼ == M`** —
    /// completeness + correctness). Returns the [`VerifiedCase`] token.
    ///
    /// # Errors
    ///
    /// [`Error::Verification`] if any piece's `π` fails, a piece does not share
    /// the tuple, or the commitments do not sum to `expected_commitment`;
    /// [`Error::DegenerateInput`] on a degenerate expected input.
    pub fn verify<C: Context + ?Sized>(
        &self,
        expected_commitment: &PublicKey,
        expected_recipient: &PublicKey,
        expected_access_keys: &[PublicKey],
        ctx: &C,
    ) -> Result<VerifiedCase<'_>, Error> {
        let frozen = FrozenContext::capture(ctx)?;
        let access: Vec<ProjectivePoint> =
            expected_access_keys.iter().map(PublicKey::point).collect();
        let binding = composite::opening_binding(&expected_recipient.point(), &access)?;

        let mut sum = ProjectivePoint::IDENTITY;
        let mut pieces = Vec::with_capacity(self.capsules.len());
        let mut proofs = Vec::with_capacity(self.capsules.len());
        let mut digests = Vec::with_capacity(self.capsules.len());
        for capsule in &self.capsules {
            composite::verify_case_piece_bound(
                capsule.proof(),
                &capsule.commitment(),
                &binding,
                &frozen,
            )?;
            proofs.push(capsule.proof());
            sum += capsule.commitment();
            let piece = capsule.as_capsule_ref();
            digests.push(signature::core_digest(piece.elgamal, &piece.c));
            pieces.push(piece);
        }
        signature::reject_duplicate_cores(&digests)?;
        if let Some(detail) =
            assembly::cross_piece_elgamal_mask_relation(pieces.iter().map(|piece| piece.elgamal))
        {
            return Err(Error::Verification(detail));
        }
        if let Some(detail) = assembly::cross_piece_pedersen_commitment_relation(proofs) {
            return Err(Error::Verification(detail));
        }
        if sum != expected_commitment.point() {
            return Err(Error::Verification(
                "case piece commitments do not sum to the expected commitment",
            ));
        }

        Ok(VerifiedCase {
            pieces,
            binding,
            commitment: expected_commitment.point(),
            ctx: frozen,
            backing: Backing::Proof,
        })
    }
}

/// A `Case` confirmed against its authorization — `verify` passed, or a quorum
/// signature verified.
///
/// The only place `contribute`/`unseal` exist; opens every piece without
/// re-verifying each `π`. Holds the per-piece **opening core views** (not the full
/// `Case`), so a proof-stripped case can mint the same token.
pub struct VerifiedCase<'a> {
    pieces: Vec<CapsuleRef<'a>>,
    binding: OpeningBinding,
    commitment: ProjectivePoint,
    ctx: FrozenContext,
    backing: Backing,
}

impl<'a> VerifiedCase<'a> {
    /// Assemble a verified case from already-verified parts (crate-internal). The
    /// signature path ([`BoundCase::verify_signed`](crate::BoundCase::verify_signed))
    /// builds the token here with [`Backing::Signature`]; the proof path uses the
    /// inherent constructor in [`Case::verify`] with [`Backing::Proof`].
    pub(crate) const fn from_parts(
        pieces: Vec<CapsuleRef<'a>>,
        binding: OpeningBinding,
        commitment: ProjectivePoint,
        ctx: FrozenContext,
        backing: Backing,
    ) -> Self {
        Self {
            pieces,
            binding,
            commitment,
            ctx,
            backing,
        }
    }
}

impl VerifiedCase<'_> {
    /// How this token was established — proof (trustless) or signature (delegated
    /// to the verifying-key holder).
    #[must_use]
    pub const fn backing(&self) -> Backing {
        self.backing
    }

    /// The canonical statement the quorum signs at provisioning to back this case
    /// with a signature: `domain ‖ count ‖ {per-piece core digests}↑ ‖ M ‖
    /// recipient ‖ g* ‖ Y* ‖ ctx ‖ params`. The framework reduces these bytes to a
    /// 32-byte digest (via a typed signing intent), runs its FROST round over that
    /// digest, and attaches the signature to the stripped case;
    /// [`BoundCase::verify_signed`](crate::BoundCase::verify_signed) re-derives the
    /// identical bytes through the same builder, so signer and verifier agree by
    /// construction. Only obtainable from a verified case — never a free blob.
    ///
    /// **Contract:** the signature must cover `SHA-256` of the returned bytes, not
    /// the bytes themselves — that 32-byte digest is what
    /// [`BoundCase::verify_signed`](crate::BoundCase::verify_signed) checks and what
    /// the framework's typed signing intent sets as its `message_hash`.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if the verified context's `binding_bytes` fails.
    pub fn attestation_message(&self) -> Result<Vec<u8>, Error> {
        let piece_digests: Vec<[u8; 32]> = self
            .pieces
            .iter()
            .map(|piece| signature::core_digest(piece.elgamal, &piece.c))
            .collect();
        signature::case_attestation_statement(
            &piece_digests,
            &self.commitment,
            &self.binding,
            &self.ctx,
        )
    }

    /// One authorizer's contribution to **every** piece, using `key` as a
    /// self-held access key (gate = `key.public_key()`): a [`Partial`] per
    /// capsule (each piece has its own segment masks, so its own partial).
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `key.public_key()` is not in the access set;
    /// otherwise a DLEQ error.
    pub fn contribute(&self, key: &PrivateKey) -> Result<Vec<Partial>, Error> {
        self.contribute_for_gate(key, &key.public_key())
    }

    /// One authorizer's contribution to **every** piece for an explicit `gate`
    /// (the threshold/share path): a [`Partial`] per capsule. The recipient
    /// accepts a gate's bucket only when the contributed points sum to it, so
    /// several participants can each pass their piece with the same aggregate
    /// `gate`.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `gate` is not in the access set; otherwise a
    /// DLEQ error.
    pub fn contribute_for_gate(
        &self,
        key: &PrivateKey,
        gate: &PublicKey,
    ) -> Result<Vec<Partial>, Error> {
        self.pieces
            .iter()
            .map(|piece| {
                opening::contribute(
                    *piece,
                    &self.binding,
                    &gate.point(),
                    key.scalar(),
                    &self.ctx,
                    &mut OsRng,
                )
            })
            .collect()
    }

    /// Open every piece and sum → the reconstructed secret `s = Σ sⱼ`. Each
    /// piece picks its own partials from the flat `partials`: a partial's DLEQ is
    /// proven over its capsule's per-segment masks `{E_j}` (fresh per seal, so
    /// distinct across pieces *even if two pieces share a commitment*) and the
    /// challenge absorbs them — so a partial verifies against exactly the piece
    /// it was made for and is skipped elsewhere. The recovered secret is
    /// rechecked against `Σ Mⱼ`.
    ///
    /// # Errors
    ///
    /// [`Error::Verification`] if any piece is not fully opened (a gate short, a
    /// limb unrecoverable) or the reconstructed secret's commitment is not `M`.
    pub fn unseal(
        &self,
        recipient: &PrivateKey,
        partials: &[Partial],
    ) -> Result<PrivateKey, Error> {
        let mut s = Scalar::ZERO;
        for piece in &self.pieces {
            let opened = opening::unseal_verified(
                *piece,
                &self.binding,
                recipient.scalar(),
                partials,
                &self.ctx,
            )?;
            s += opened;
        }
        if g() * s != self.commitment {
            return Err(Error::Verification(
                "reconstructed secret does not match the case commitment",
            ));
        }
        Ok(PrivateKey::from_scalar(s))
    }

    /// Strip every piece's seal proof, keeping only the per-piece opening cores,
    /// for compact recovery storage. Only a verified case can be stripped (the
    /// honest encoder). Recover via [`StrippedCase::bind`] + the self-securing
    /// [`BoundCase::unseal`](crate::BoundCase::unseal), anchored on the certified
    /// commitment `M`.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if the case has more than 255 pieces.
    pub fn strip(&self) -> Result<StrippedCase, Error> {
        let pieces = self
            .pieces
            .iter()
            .map(|piece| StrippedCapsule::from_core(piece.elgamal.to_vec(), piece.c))
            .collect();
        StrippedCase::from_pieces(pieces)
    }
}

impl Contribute<VerifiedCase<'_>> for PrivateKey {
    type Output = Result<Vec<Partial>, Error>;
    fn contribute(&self, target: &VerifiedCase<'_>) -> Self::Output {
        target.contribute(self)
    }
    fn contribute_for_gate(&self, target: &VerifiedCase<'_>, gate: &PublicKey) -> Self::Output {
        target.contribute_for_gate(self, gate)
    }
}

impl Unseal<VerifiedCase<'_>> for PrivateKey {
    type Output = Result<Self, Error>;
    fn unseal(&self, target: &VerifiedCase<'_>, partials: &[Partial]) -> Self::Output {
        target.unseal(self, partials)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::bsgs::baby_table;
    use crate::codec::encode_point;
    use crate::composite::{
        seal_with_prefix_mask_scalars_for_test, seal_with_prefix_value_blindings_for_test,
    };
    use crate::limbs::{LIMB_COUNT, LIMB_MODULUS, recompose};
    use k256::elliptic_curve::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::borrow::Cow;

    struct TestCtx;
    impl Context for TestCtx {
        fn domain(&self) -> &'static str {
            "ve-capsule.case-test"
        }
        fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
            Ok(Cow::Borrowed(b"case-binding"))
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

    /// `s·G = Σ pieces·G`, wrapped as a `PublicKey`.
    fn commitment(pieces: &[&PrivateKey]) -> PublicKey {
        let sum = pieces.iter().fold(ProjectivePoint::IDENTITY, |acc, p| {
            acc + p.public_key().point()
        });
        PublicKey::from_canonical_bytes(&encode_point(&sum)).unwrap()
    }

    fn piece_capsule(s: &PrivateKey, recipient: &PublicKey, access: &PublicKey) -> Capsule {
        Capsule::builder(s, recipient, &TestCtx)
            .access_key(access)
            .seal()
            .unwrap()
    }

    fn random_nonzero_scalar_for_test(rng: &mut StdRng) -> Scalar {
        loop {
            let scalar = Scalar::random(&mut *rng);
            if !bool::from(scalar.is_zero()) {
                return scalar;
            }
        }
    }

    fn private_key_from_limb_prefix(prefix: &[u32]) -> PrivateKey {
        let mut limbs = [0u32; LIMB_COUNT];
        limbs[..prefix.len()].copy_from_slice(prefix);
        PrivateKey::from_scalar(recompose(&limbs))
    }

    #[test]
    fn split_secret_round_trip() {
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_01);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let rpk = recipient.public_key();
        let apk = access.public_key();
        let s1 = private_key(&mut rng);
        let s2 = private_key(&mut rng);
        let s3 = private_key(&mut rng);
        let m = commitment(&[&s1, &s2, &s3]);

        let case = Case::new(vec![
            piece_capsule(&s1, &rpk, &apk),
            piece_capsule(&s2, &rpk, &apk),
            piece_capsule(&s3, &rpk, &apk),
        ])
        .unwrap();

        let vcase = case.verify(&m, &rpk, &[apk], &TestCtx).unwrap();
        // One gate participant contributes to every piece; recipient sums the secret.
        let partials = vcase.contribute(&access).unwrap();
        assert_eq!(partials.len(), 3, "one partial per piece");
        let s = vcase.unseal(&recipient, &partials).unwrap();
        assert_eq!(s.public_key(), m, "Σ sⱼ reconstructs s");
    }

    #[test]
    fn recipient_only_split_round_trip() {
        // The recipient-only shape: each producer seals its additive piece σⱼ to
        // the recovery key with NO access gates, and the recipient opens every
        // piece with ZERO partials and sums to s = Σ σⱼ. Every other `Case` test
        // gates on an access key, so this one pins the no-gate path.
        let mut rng = StdRng::seed_from_u64(0xCA_5E_02_01);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let s1 = private_key(&mut rng);
        let s2 = private_key(&mut rng);
        let s3 = private_key(&mut rng);
        let m = commitment(&[&s1, &s2, &s3]);

        let case = Case::new(vec![
            Capsule::builder(&s1, &rpk, &TestCtx).seal().unwrap(),
            Capsule::builder(&s2, &rpk, &TestCtx).seal().unwrap(),
            Capsule::builder(&s3, &rpk, &TestCtx).seal().unwrap(),
        ])
        .unwrap();

        // Empty access set, empty partials — the recipient opens alone.
        let vcase = case.verify(&m, &rpk, &[], &TestCtx).unwrap();
        let s = vcase.unseal(&recipient, &[]).unwrap();
        assert_eq!(s.public_key(), m, "Σ σⱼ reconstructs the target commitment");
    }

    #[test]
    fn verify_rejects_incomplete_commitment() {
        // Σ Mⱼ over only two of three pieces ≠ the three-piece commitment.
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_02);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let rpk = recipient.public_key();
        let apk = access.public_key();
        let s1 = private_key(&mut rng);
        let s2 = private_key(&mut rng);
        let s3 = private_key(&mut rng);
        let full_m = commitment(&[&s1, &s2, &s3]);

        // A case missing piece 3 cannot match the full three-piece commitment.
        let case = Case::new(vec![
            piece_capsule(&s1, &rpk, &apk),
            piece_capsule(&s2, &rpk, &apk),
        ])
        .unwrap();
        assert!(matches!(
            case.verify(&full_m, &rpk, &[apk], &TestCtx),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn verify_rejects_cross_piece_mask_reuse_that_leaks_producer_limb() {
        // Each piece's proof is internally valid, but the malicious piece reuses
        // the honest piece's ElGamal masks. Public observers can then subtract
        // D-points and cancel Y*, recovering the bounded honest-minus-malicious
        // limb relation by BSGS.
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_05);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let honest = PrivateKey::from_scalar(Scalar::from(12_345u64));
        let malicious = PrivateKey::from_scalar(Scalar::from(2_345u64));
        let target = commitment(&[&honest, &malicious]);
        let prefix = (0..LIMB_COUNT)
            .map(|_| random_nonzero_scalar_for_test(&mut rng))
            .collect::<Vec<_>>();

        let (honest_proof, honest_c) = seal_with_prefix_mask_scalars_for_test(
            honest.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &prefix,
        )
        .unwrap();
        let (malicious_proof, malicious_c) = seal_with_prefix_mask_scalars_for_test(
            malicious.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &prefix,
        )
        .unwrap();
        let honest_capsule = Capsule::from_parts_for_test(honest_proof, honest_c);
        let malicious_capsule = Capsule::from_parts_for_test(malicious_proof, malicious_c);

        let leaked = honest_capsule.as_capsule_ref().elgamal[0].d
            - malicious_capsule.as_capsule_ref().elgamal[0].d;
        assert_eq!(
            baby_table().recover_bounded_complete(&leaked, LIMB_MODULUS),
            Some(10_000),
            "shared cross-piece masks publicly reveal the honest-minus-malicious limb"
        );

        let stripped = StrippedCase::from_pieces(vec![
            {
                let core = honest_capsule.as_capsule_ref();
                StrippedCapsule::from_core(core.elgamal.to_vec(), core.c)
            },
            {
                let core = malicious_capsule.as_capsule_ref();
                StrippedCapsule::from_core(core.elgamal.to_vec(), core.c)
            },
        ])
        .unwrap();
        assert!(matches!(
            stripped.bind(&target, &rpk, &[], &TestCtx),
            Err(Error::Verification(_))
        ));

        let case = Case::new(vec![honest_capsule, malicious_capsule]).unwrap();
        assert!(matches!(
            case.verify(&target, &rpk, &[], &TestCtx),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn verify_rejects_cross_piece_mask_public_g_offset_that_leaks_producer_limb() {
        // Each piece is internally proof-valid. The malicious piece instead
        // chooses its first mask so E_honest,0 + E_malicious,0 = G. Public
        // observers subtract Y* from the matching D sum and recover the bounded
        // honest-plus-malicious limb; the malicious producer then subtracts its
        // own limb to learn the honest producer's limb.
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_12);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let honest = PrivateKey::from_scalar(Scalar::from(12_345u64));
        let malicious = PrivateKey::from_scalar(Scalar::from(2_345u64));
        let target = commitment(&[&honest, &malicious]);

        let honest_r0 = random_nonzero_scalar_for_test(&mut rng);
        let mut honest_prefix = vec![honest_r0];
        let mut malicious_prefix = vec![Scalar::ONE - honest_r0];
        for _ in 1..LIMB_COUNT {
            honest_prefix.push(random_nonzero_scalar_for_test(&mut rng));
            malicious_prefix.push(random_nonzero_scalar_for_test(&mut rng));
        }

        let (honest_proof, honest_c) = seal_with_prefix_mask_scalars_for_test(
            honest.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &honest_prefix,
        )
        .unwrap();
        let (malicious_proof, malicious_c) = seal_with_prefix_mask_scalars_for_test(
            malicious.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &malicious_prefix,
        )
        .unwrap();
        let honest_capsule = Capsule::from_parts_for_test(honest_proof, honest_c);
        let malicious_capsule = Capsule::from_parts_for_test(malicious_proof, malicious_c);

        let leaked = honest_capsule.as_capsule_ref().elgamal[0].d
            + malicious_capsule.as_capsule_ref().elgamal[0].d
            - rpk.point();
        assert_eq!(
            baby_table().recover_bounded_complete(&leaked, LIMB_MODULUS * 2),
            Some(12_345 + 2_345),
            "a public-G mask offset reveals the honest-plus-malicious limb"
        );

        let stripped = StrippedCase::from_pieces(vec![
            {
                let core = honest_capsule.as_capsule_ref();
                StrippedCapsule::from_core(core.elgamal.to_vec(), core.c)
            },
            {
                let core = malicious_capsule.as_capsule_ref();
                StrippedCapsule::from_core(core.elgamal.to_vec(), core.c)
            },
        ])
        .unwrap();
        assert!(matches!(
            stripped.bind(&target, &rpk, &[], &TestCtx),
            Err(Error::Verification(_))
        ));

        let case = Case::new(vec![honest_capsule, malicious_capsule]).unwrap();
        assert!(matches!(
            case.verify(&target, &rpk, &[], &TestCtx),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn verify_rejects_cross_piece_high_arity_mask_relation_that_leaks_producer_limb() {
        // Each piece is internally proof-valid and has no intra-piece mask
        // relation. The malicious piece instead chooses five of its masks so
        // that E_honest,0 + E_malicious,0 + E_malicious,1 + E_malicious,2
        // + E_malicious,3 - E_malicious,4 = O across pieces. The
        // corresponding six-mask D-point combination cancels Y* and exposes a
        // bounded relation containing the honest producer's limb.
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_06);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let honest = PrivateKey::from_scalar(Scalar::from(12_345u64));
        let malicious = private_key_from_limb_prefix(&[100, 5, 6, 7, 8]);
        let target = commitment(&[&honest, &malicious]);

        let honest_prefix = (0..LIMB_COUNT)
            .map(|_| random_nonzero_scalar_for_test(&mut rng))
            .collect::<Vec<_>>();
        let mut malicious_prefix = Vec::with_capacity(LIMB_COUNT);
        let malicious_r0 = random_nonzero_scalar_for_test(&mut rng);
        let malicious_r1 = random_nonzero_scalar_for_test(&mut rng);
        let malicious_r2 = random_nonzero_scalar_for_test(&mut rng);
        let malicious_r3 = random_nonzero_scalar_for_test(&mut rng);
        malicious_prefix.push(malicious_r0);
        malicious_prefix.push(malicious_r1);
        malicious_prefix.push(malicious_r2);
        malicious_prefix.push(malicious_r3);
        malicious_prefix
            .push(honest_prefix[0] + malicious_r0 + malicious_r1 + malicious_r2 + malicious_r3);
        for _ in 5..LIMB_COUNT {
            malicious_prefix.push(random_nonzero_scalar_for_test(&mut rng));
        }

        let (honest_proof, honest_c) = seal_with_prefix_mask_scalars_for_test(
            honest.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &honest_prefix,
        )
        .unwrap();
        let (malicious_proof, malicious_c) = seal_with_prefix_mask_scalars_for_test(
            malicious.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &malicious_prefix,
        )
        .unwrap();
        let honest_capsule = Capsule::from_parts_for_test(honest_proof, honest_c);
        let malicious_capsule = Capsule::from_parts_for_test(malicious_proof, malicious_c);

        let leaked = honest_capsule.as_capsule_ref().elgamal[0].d
            + malicious_capsule.as_capsule_ref().elgamal[0].d
            + malicious_capsule.as_capsule_ref().elgamal[1].d
            + malicious_capsule.as_capsule_ref().elgamal[2].d
            + malicious_capsule.as_capsule_ref().elgamal[3].d
            - malicious_capsule.as_capsule_ref().elgamal[4].d;
        assert_eq!(
            baby_table().recover_bounded_complete(&leaked, LIMB_MODULUS),
            Some(12_345 + 100 + 5 + 6 + 7 - 8),
            "the cross-piece high-arity relation publicly reveals an honest-limb equation"
        );

        let stripped = StrippedCase::from_pieces(vec![
            {
                let core = honest_capsule.as_capsule_ref();
                StrippedCapsule::from_core(core.elgamal.to_vec(), core.c)
            },
            {
                let core = malicious_capsule.as_capsule_ref();
                StrippedCapsule::from_core(core.elgamal.to_vec(), core.c)
            },
        ])
        .unwrap();
        assert!(matches!(
            stripped.bind(&target, &rpk, &[], &TestCtx),
            Err(Error::Verification(_))
        ));

        let case = Case::new(vec![honest_capsule, malicious_capsule]).unwrap();
        assert!(matches!(
            case.verify(&target, &rpk, &[], &TestCtx),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn verify_rejects_three_piece_mask_relation_that_leaks_producer_limb() {
        // The prior support-six mask screen only walked pairs of pieces. A
        // support-six relation split over three pieces evades every pairwise
        // projection while the full D-point equation still cancels Y* and leaks
        // an honest limb relation.
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_0F);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let honest = PrivateKey::from_scalar(Scalar::from(12_345u64));
        let splitter = private_key_from_limb_prefix(&[100, 5]);
        let canceller = private_key_from_limb_prefix(&[6, 7, 8]);
        let target = commitment(&[&honest, &splitter, &canceller]);

        let honest_prefix = (0..LIMB_COUNT)
            .map(|_| random_nonzero_scalar_for_test(&mut rng))
            .collect::<Vec<_>>();
        let mut splitter_prefix = Vec::with_capacity(LIMB_COUNT);
        splitter_prefix.push(random_nonzero_scalar_for_test(&mut rng));
        splitter_prefix.push(random_nonzero_scalar_for_test(&mut rng));
        for _ in 2..LIMB_COUNT {
            splitter_prefix.push(random_nonzero_scalar_for_test(&mut rng));
        }

        let mut canceller_prefix = Vec::with_capacity(LIMB_COUNT);
        let canceller_r0 = random_nonzero_scalar_for_test(&mut rng);
        let canceller_r1 = random_nonzero_scalar_for_test(&mut rng);
        canceller_prefix.push(canceller_r0);
        canceller_prefix.push(canceller_r1);
        canceller_prefix.push(
            honest_prefix[0]
                + splitter_prefix[0]
                + splitter_prefix[1]
                + canceller_r0
                + canceller_r1,
        );
        for _ in 3..LIMB_COUNT {
            canceller_prefix.push(random_nonzero_scalar_for_test(&mut rng));
        }

        let (honest_proof, honest_c) = seal_with_prefix_mask_scalars_for_test(
            honest.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &honest_prefix,
        )
        .unwrap();
        let (splitter_proof, splitter_c) = seal_with_prefix_mask_scalars_for_test(
            splitter.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &splitter_prefix,
        )
        .unwrap();
        let (canceller_proof, canceller_c) = seal_with_prefix_mask_scalars_for_test(
            canceller.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &canceller_prefix,
        )
        .unwrap();
        let honest_capsule = Capsule::from_parts_for_test(honest_proof, honest_c);
        let splitter_capsule = Capsule::from_parts_for_test(splitter_proof, splitter_c);
        let canceller_capsule = Capsule::from_parts_for_test(canceller_proof, canceller_c);

        let leaked = honest_capsule.as_capsule_ref().elgamal[0].d
            + splitter_capsule.as_capsule_ref().elgamal[0].d
            + splitter_capsule.as_capsule_ref().elgamal[1].d
            + canceller_capsule.as_capsule_ref().elgamal[0].d
            + canceller_capsule.as_capsule_ref().elgamal[1].d
            - canceller_capsule.as_capsule_ref().elgamal[2].d;
        assert_eq!(
            baby_table().recover_bounded_complete(&leaked, LIMB_MODULUS),
            Some(12_345 + 100 + 5 + 6 + 7 - 8),
            "the three-piece mask relation publicly reveals an honest-limb equation"
        );

        let case = Case::new(vec![honest_capsule, splitter_capsule, canceller_capsule]).unwrap();
        assert!(matches!(
            case.verify(&target, &rpk, &[], &TestCtx),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn verify_rejects_cross_piece_pedersen_blinding_reuse_that_leaks_producer_limb() {
        // The ElGamal masks are honest-random and unrelated. The leak is in the
        // public Pedersen value commitments: reusing the same H-blinding across
        // pieces makes Com_h,0 - Com_m,0 = (v_h,0 - v_m,0)G, so a malicious
        // producer that knows v_m,0 recovers the honest producer's limb by BSGS.
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_07);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let honest = PrivateKey::from_scalar(Scalar::from(12_345u64));
        let malicious = private_key_from_limb_prefix(&[100]);
        let target = commitment(&[&honest, &malicious]);
        let reused_blinding = random_nonzero_scalar_for_test(&mut rng);

        let (honest_proof, honest_c) = seal_with_prefix_value_blindings_for_test(
            honest.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &[reused_blinding],
        )
        .unwrap();
        let (malicious_proof, malicious_c) = seal_with_prefix_value_blindings_for_test(
            malicious.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &[reused_blinding],
        )
        .unwrap();

        let leaked = honest_proof.value_commitments()[0] - malicious_proof.value_commitments()[0];
        assert_eq!(
            baby_table().recover_bounded_complete(&leaked, LIMB_MODULUS),
            Some(12_345 - 100),
            "same-slot Pedersen blinding reuse publicly reveals an honest-limb equation"
        );

        let case = Case::new(vec![
            Capsule::from_parts_for_test(honest_proof, honest_c),
            Capsule::from_parts_for_test(malicious_proof, malicious_c),
        ])
        .unwrap();
        assert!(matches!(
            case.verify(&target, &rpk, &[], &TestCtx),
            Err(Error::Verification(
                "case pieces have a cross-piece Pedersen commitment relation"
            ))
        ));
    }

    #[test]
    fn verify_rejects_cross_piece_pedersen_cross_slot_blinding_reuse_that_leaks_producer_limb() {
        // Same leak without matching statement slots: the malicious producer reuses
        // the honest slot-0 Pedersen blinding in its own slot 1. Both proofs are
        // locally valid, but Com_h,0 - Com_m,1 cancels H and exposes the honest
        // limb after subtracting the malicious producer's known slot-1 limb.
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_13);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let honest = PrivateKey::from_scalar(Scalar::from(12_345u64));
        let malicious = private_key_from_limb_prefix(&[100, 200]);
        let target = commitment(&[&honest, &malicious]);
        let reused_blinding = random_nonzero_scalar_for_test(&mut rng);
        let unrelated_blinding = random_nonzero_scalar_for_test(&mut rng);

        let (honest_proof, honest_c) = seal_with_prefix_value_blindings_for_test(
            honest.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &[reused_blinding],
        )
        .unwrap();
        let (malicious_proof, malicious_c) = seal_with_prefix_value_blindings_for_test(
            malicious.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &[unrelated_blinding, reused_blinding],
        )
        .unwrap();

        let leaked = honest_proof.value_commitments()[0] - malicious_proof.value_commitments()[1];
        assert_eq!(
            baby_table().recover_bounded_complete(&leaked, LIMB_MODULUS),
            Some(12_345 - 200),
            "cross-slot Pedersen blinding reuse publicly reveals an honest-limb equation"
        );

        let case = Case::new(vec![
            Capsule::from_parts_for_test(honest_proof, honest_c),
            Capsule::from_parts_for_test(malicious_proof, malicious_c),
        ])
        .unwrap();
        assert!(matches!(
            case.verify(&target, &rpk, &[], &TestCtx),
            Err(Error::Verification(
                "case pieces have a cross-piece Pedersen commitment relation"
            ))
        ));
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn verify_rejects_cross_piece_pedersen_three_piece_relation_that_leaks_producer_limb() {
        // This is the higher-arity form of the Pedersen commitment leak. The
        // honest piece's slot-0 H-blinding is split across two malicious
        // pieces, so no same-slot pair cancels. The three-term public relation
        // Com_h,0 - Com_m1,0 - Com_m2,0 cancels H and exposes a bounded
        // equation containing the honest producer's limb.
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_08);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let honest = PrivateKey::from_scalar(Scalar::from(12_345u64));
        let malicious_a = private_key_from_limb_prefix(&[100]);
        let malicious_b = private_key_from_limb_prefix(&[200]);
        let target = commitment(&[&honest, &malicious_a, &malicious_b]);
        let blinding_a = random_nonzero_scalar_for_test(&mut rng);
        let blinding_b = random_nonzero_scalar_for_test(&mut rng);
        let honest_blinding = blinding_a + blinding_b;

        let (honest_proof, honest_c) = seal_with_prefix_value_blindings_for_test(
            honest.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &[honest_blinding],
        )
        .unwrap();
        let (malicious_a_proof, malicious_a_c) = seal_with_prefix_value_blindings_for_test(
            malicious_a.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &[blinding_a],
        )
        .unwrap();
        let (malicious_b_proof, malicious_b_c) = seal_with_prefix_value_blindings_for_test(
            malicious_b.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &[blinding_b],
        )
        .unwrap();

        let leaked = honest_proof.value_commitments()[0]
            - malicious_a_proof.value_commitments()[0]
            - malicious_b_proof.value_commitments()[0];
        assert_eq!(
            baby_table().recover_bounded_complete(&leaked, LIMB_MODULUS),
            Some(12_345 - 100 - 200),
            "three-piece Pedersen blinding cancellation publicly reveals an honest-limb equation"
        );

        let case = Case::new(vec![
            Capsule::from_parts_for_test(honest_proof, honest_c),
            Capsule::from_parts_for_test(malicious_a_proof, malicious_a_c),
            Capsule::from_parts_for_test(malicious_b_proof, malicious_b_c),
        ])
        .unwrap();
        assert!(matches!(
            case.verify(&target, &rpk, &[], &TestCtx),
            Err(Error::Verification(
                "case pieces have a cross-piece Pedersen commitment relation"
            ))
        ));
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn verify_rejects_cross_piece_pedersen_four_piece_relation_that_leaks_producer_limb() {
        // Same leak, one arity higher: the honest slot-0 blinding is split
        // across three malicious pieces. Support-three screens see nonzero H
        // residuals in every projection, while the four-term public equation
        // strips H completely.
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_09);
        let recipient = private_key(&mut rng);
        let rpk = recipient.public_key();
        let honest = PrivateKey::from_scalar(Scalar::from(12_345u64));
        let malicious_a = private_key_from_limb_prefix(&[100]);
        let malicious_b = private_key_from_limb_prefix(&[200]);
        let malicious_c = private_key_from_limb_prefix(&[300]);
        let target = commitment(&[&honest, &malicious_a, &malicious_b, &malicious_c]);
        let blinding_a = random_nonzero_scalar_for_test(&mut rng);
        let blinding_b = random_nonzero_scalar_for_test(&mut rng);
        let blinding_c = random_nonzero_scalar_for_test(&mut rng);
        let honest_blinding = blinding_a + blinding_b + blinding_c;

        let (honest_proof, honest_c) = seal_with_prefix_value_blindings_for_test(
            honest.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &[honest_blinding],
        )
        .unwrap();
        let (malicious_a_proof, malicious_a_c) = seal_with_prefix_value_blindings_for_test(
            malicious_a.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &[blinding_a],
        )
        .unwrap();
        let (malicious_b_proof, malicious_b_c) = seal_with_prefix_value_blindings_for_test(
            malicious_b.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &[blinding_b],
        )
        .unwrap();
        let (malicious_c_proof, malicious_c_c) = seal_with_prefix_value_blindings_for_test(
            malicious_c.scalar(),
            &rpk.point(),
            &[],
            &TestCtx,
            &mut rng,
            &[blinding_c],
        )
        .unwrap();

        let leaked = honest_proof.value_commitments()[0]
            - malicious_a_proof.value_commitments()[0]
            - malicious_b_proof.value_commitments()[0]
            - malicious_c_proof.value_commitments()[0];
        assert_eq!(
            baby_table().recover_bounded_complete(&leaked, LIMB_MODULUS),
            Some(12_345 - 100 - 200 - 300),
            "four-piece Pedersen blinding cancellation publicly reveals an honest-limb equation"
        );

        let case = Case::new(vec![
            Capsule::from_parts_for_test(honest_proof, honest_c),
            Capsule::from_parts_for_test(malicious_a_proof, malicious_a_c),
            Capsule::from_parts_for_test(malicious_b_proof, malicious_b_c),
            Capsule::from_parts_for_test(malicious_c_proof, malicious_c_c),
        ])
        .unwrap();
        assert!(matches!(
            case.verify(&target, &rpk, &[], &TestCtx),
            Err(Error::Verification(
                "case pieces have a cross-piece Pedersen commitment relation"
            ))
        ));
    }

    #[test]
    fn verify_rejects_mismatched_recipient() {
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_03);
        let recipient = private_key(&mut rng);
        let wrong = private_key(&mut rng);
        let access = private_key(&mut rng);
        let rpk = recipient.public_key();
        let apk = access.public_key();
        let s1 = private_key(&mut rng);
        let m = commitment(&[&s1]);
        let case = Case::new(vec![piece_capsule(&s1, &rpk, &apk)]).unwrap();
        // Verify against a different recipient: each piece's π fails.
        assert!(
            case.verify(&m, &wrong.public_key(), &[apk], &TestCtx)
                .is_err()
        );
    }

    #[test]
    #[ignore = "manual perf baseline; run with --release --ignored --nocapture"]
    fn max_piece_verify_latency() {
        use std::time::Instant;

        const PIECES: usize = 6;
        let mut rng = StdRng::seed_from_u64(0xCA_5E_70_01);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let rpk = recipient.public_key();
        let apk = access.public_key();

        // Production cold cost: a fresh recipient process builds the public
        // recovery-key BSGS table on first touch, then verifies. Sealing the
        // fixture below would warm the table, so force its cold build FIRST and
        // time it; the cold recovery cost a fresh process pays is build + the
        // (now warm) verify. Gating the warm verify alone hides the table's
        // build/memory cost, which is exactly what the 2^22 bump exploited.
        let build_start = Instant::now();
        let _ = crate::bsgs::is_public_recovery_key_scalar_multiple(&rpk.point());
        let build_ms = build_start.elapsed().as_secs_f64() * 1e3;

        let pieces = (0..PIECES)
            .map(|_| private_key(&mut rng))
            .collect::<Vec<_>>();
        let piece_refs = pieces.iter().collect::<Vec<_>>();
        let expected = commitment(&piece_refs);
        let case = Case::new(
            pieces
                .iter()
                .map(|piece| piece_capsule(piece, &rpk, &apk))
                .collect(),
        )
        .unwrap();

        let verify_start = Instant::now();
        let verified = case.verify(&expected, &rpk, &[apk], &TestCtx).unwrap();
        let verify_ms = verify_start.elapsed().as_secs_f64() * 1e3;
        assert_eq!(verified.pieces.len(), PIECES);

        let cold_ms = build_ms + verify_ms;
        println!(
            "case_verify pieces={PIECES} build_ms={build_ms:.3} verify_ms={verify_ms:.3} \
             single_core_floor_ms={cold_ms:.3}"
        );
    }

    #[test]
    fn equal_piece_split_opens() {
        // Two pieces with the SAME secret (same commitment C, but distinct fresh
        // segment masks E_j) — a legitimate (if astronomically rare) split.
        // Partials route by the DLEQ bases (E_j), not C, so each piece opens
        // independently and the sum is 2s. (Confirms there is no need to reject
        // duplicate commitments — they do not cross-route.)
        let mut rng = StdRng::seed_from_u64(0xCA_5E_01_04);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let rpk = recipient.public_key();
        let apk = access.public_key();
        let s = private_key(&mut rng);
        let m = commitment(&[&s, &s]);
        let case = Case::new(vec![
            piece_capsule(&s, &rpk, &apk),
            piece_capsule(&s, &rpk, &apk),
        ])
        .unwrap();
        let vcase = case.verify(&m, &rpk, &[apk], &TestCtx).unwrap();
        let partials = vcase.contribute(&access).unwrap();
        let recovered = vcase.unseal(&recipient, &partials).unwrap();
        assert_eq!(recovered.public_key(), m, "equal-piece split opens to 2s");
    }

    #[test]
    fn new_rejects_empty() {
        assert!(matches!(
            Case::new(Vec::new()),
            Err(Error::DegenerateInput(_))
        ));
    }
}
