//! Batched multi-statement Chaum–Pedersen DLEQ for the `contribute` partial.
//!
//! An authorizer's `Partial` must convince the recipient that **one** secret
//! scalar `w` simultaneously relates `(G, W)` and every segment pair
//! `(E_j, W_j)` — i.e. `W = w·G` and `W_j = w·E_j` for all limbs `j`. Without
//! this, an authorizer could publish a `W` that passes the per-gate
//! `Σ wᵢ·G == Y_accessₖ` aggregate while supplying segment points `W_j` that do
//! **not** equal `w·E_j`, corrupting the recipient's mask-strip with no
//! detectable error until the final `m·G == C` recheck (a silent denial, and a
//! partial-decryption oracle on attacker-chosen points if left unbound).
//!
//! The proof is the standard batched DLEQ: one shared nonce `k`, one
//! announcement `A_i = k·B_i` per base `B_i`, a single Fiat–Shamir challenge
//! `c` over the whole statement, and **one shared response** `z = k + c·w`. The
//! verifier checks `z·B_i == A_i + c·P_i` on every leg; because `z` and `c` are
//! shared across legs, two accepting transcripts collapse to a single common
//! `w = (z−z')/(c−c')` for *all* bases (special soundness), which is exactly the
//! "one `w`" claim.
//!
//! Domain + binding. The transcript is seeded with the crate domain (via
//! [`Transcript::new`]) and then a distinct DLEQ sub-domain tag, so a
//! contribution challenge can never be confused with a seal-`π` challenge. The
//! caller threads its own `binding` bytes (the capsule core-hash ‖ gate tag ‖
//! context) into the transcript, so a `Partial` is bound to exactly one
//! capsule, gate, and context and cannot be replayed elsewhere. The nonce `k`
//! is derived from fresh entropy, `w`, and the complete DLEQ statement, then
//! zeroized; pure RNG-output repetition therefore does not repeat `k` across
//! different challenges. Direct Schnorr nonce reuse across two challenges would
//! leak `w`.
//!
//! **Binding contract (consumer obligation).** This module binds *whatever
//! bytes it is handed* — it cannot enforce that `binding` actually encodes the
//! intended `(core-hash, gate, context)`. The `contribute` consumer MUST build
//! `binding` from a **single canonical, nonempty, unambiguously-framed** encoder
//! (length-prefixed/versioned components, never a raw concatenation of
//! variable-length fields). An empty, shared, or ambiguous `binding` lets two
//! proofs over the same `(bases, images)` be replayed across distinct semantic
//! contexts — a replay the algebra alone does not prevent.

use crate::error::Error;
use crate::transcript::{Transcript, length_prefix};
use k256::FieldBytes;
#[cfg(test)]
use k256::elliptic_curve::Field;
use k256::elliptic_curve::bigint::U256;
use k256::elliptic_curve::ops::Reduce;
use k256::{ProjectivePoint, Scalar};
use rand_core::{CryptoRng, RngCore};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// DLEQ sub-domain tag, absorbed first so a contribution challenge is
/// domain-separated from the seal-`π` challenge (which shares the seeded crate
/// domain). Bump on any wire/transcript change.
const DLEQ_DOMAIN: &[u8] = b"ve-capsule.contribute-dleq.v1";

/// Nonce-derivation sub-domain. The proof challenge stays on
/// [`DLEQ_DOMAIN`] v1; this is an internal prover hardening tag and does not
/// change verifier inputs or wire layout.
const DLEQ_NONCE_DOMAIN: &[u8] = b"ve-capsule.contribute-dleq.nonce.v1";

/// A batched multi-statement DLEQ proof: one announcement per base and a single
/// shared response. Verified against the same `(bases, images, binding)` the
/// prover used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchDleqProof {
    /// `A_i = k·B_i`, one per base, in base order.
    announcements: Vec<ProjectivePoint>,
    /// The shared response `z = k + c·w`.
    z: Scalar,
}

/// Sample a uniform nonzero scalar for tests and non-secret witnesses. A zero
/// nonce `k` would make every announcement the identity and collapse the
/// response to `z = c·w`, leaking `w = z/c`; the negligibly likely zero draw is
/// rejected.
#[cfg(test)]
fn random_nonzero_scalar<R: RngCore + CryptoRng>(rng: &mut R) -> Scalar {
    loop {
        let k = Scalar::random(&mut *rng);
        if !bool::from(k.is_zero()) {
            return k;
        }
    }
}

fn reduce_be_to_scalar(bytes: [u8; 32]) -> Scalar {
    let mut repr = FieldBytes::default();
    repr.copy_from_slice(&bytes);
    <Scalar as Reduce<U256>>::reduce_bytes(&repr)
}

fn absorb_nonce_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(length_prefix(bytes.len()));
    hasher.update(bytes);
}

fn absorb_nonce_points(hasher: &mut Sha256, points: &[ProjectivePoint]) {
    let count = u32::try_from(points.len()).unwrap_or(u32::MAX);
    hasher.update(count.to_be_bytes());
    for point in points {
        absorb_nonce_field(hasher, &crate::codec::encode_point(point));
    }
}

/// Derive the Schnorr/Chaum-Pedersen nonce from fresh entropy and the full
/// statement, keyed by the prover scalar. This preserves randomized proofs
/// under healthy entropy, but if an embedded RNG or forked process repeats its
/// byte stream, a different capsule/gate/context/bases statement still gets a
/// different `k` instead of exposing `w` through two responses.
fn statement_bound_nonce<R: RngCore + CryptoRng>(
    w: &Scalar,
    bases: &[ProjectivePoint],
    images: &[ProjectivePoint],
    binding: &[u8],
    rng: &mut R,
) -> Zeroizing<Scalar> {
    let secret = Zeroizing::new(<[u8; 32]>::from(w.to_bytes()));
    loop {
        let mut seed = Zeroizing::new([0u8; 32]);
        rng.fill_bytes(&mut *seed);

        for counter in 0u8..=u8::MAX {
            let mut hasher = Sha256::new();
            absorb_nonce_field(&mut hasher, DLEQ_NONCE_DOMAIN);
            absorb_nonce_field(&mut hasher, DLEQ_DOMAIN);
            absorb_nonce_field(&mut hasher, &*seed);
            absorb_nonce_field(&mut hasher, &*secret);
            absorb_nonce_field(&mut hasher, binding);
            absorb_nonce_points(&mut hasher, bases);
            absorb_nonce_points(&mut hasher, images);
            absorb_nonce_field(&mut hasher, &[counter]);

            let candidate = reduce_be_to_scalar(hasher.finalize().into());
            if !bool::from(candidate.is_zero()) {
                return Zeroizing::new(candidate);
            }
        }
    }
}

/// Absorb the full DLEQ statement and squeeze the challenge `c`. Built
/// identically by [`BatchDleqProof::prove`] and [`BatchDleqProof::verify`] from
/// the same inputs, so the two cannot diverge. The absorption order is pinned
/// normatively in `docs/design/ec-segve-soundness.md` §6; this function
/// implements that schedule.
fn challenge(
    bases: &[ProjectivePoint],
    images: &[ProjectivePoint],
    announcements: &[ProjectivePoint],
    binding: &[u8],
) -> Scalar {
    let mut t = Transcript::new();
    t.absorb_bytes(DLEQ_DOMAIN);
    t.absorb_bytes(binding);
    t.absorb_list_len(bases.len());
    for b in bases {
        t.absorb_point(b);
    }
    t.absorb_list_len(images.len());
    for p in images {
        t.absorb_point(p);
    }
    t.absorb_list_len(announcements.len());
    for a in announcements {
        t.absorb_point(a);
    }
    t.challenge()
}

/// Reject an empty base list or any identity base. An identity base `B_i = O`
/// forces `A_i = P_i = O` and makes that leg trivially satisfiable for any `w`,
/// so it must never enter a statement whose soundness rests on a common `w`.
fn check_bases(bases: &[ProjectivePoint]) -> Result<(), Error> {
    if bases.is_empty() {
        return Err(Error::DegenerateInput("DLEQ statement has no bases"));
    }
    if bases.iter().any(|b| b == &ProjectivePoint::IDENTITY) {
        return Err(Error::DegenerateInput("DLEQ base is the identity"));
    }
    Ok(())
}

impl BatchDleqProof {
    /// Reassemble a proof from its canonical parts — the per-base announcements
    /// and the shared response `z`. Crate-internal, for the [`Partial`] wire
    /// decoder ([`crate::opening::Partial::from_canonical_bytes`]); the bytes are
    /// re-verified against `(bases, images, binding)` by [`Self::verify`], so this
    /// constructor asserts nothing about the parts beyond their canonical decode.
    ///
    /// [`Partial`]: crate::opening::Partial
    pub(crate) const fn from_parts(announcements: Vec<ProjectivePoint>, z: Scalar) -> Self {
        Self { announcements, z }
    }

    /// The per-base announcements `{A_i = k·B_i}`, in base order — the wire fields
    /// the [`Partial`] codec serializes.
    ///
    /// [`Partial`]: crate::opening::Partial
    pub(crate) fn announcements(&self) -> &[ProjectivePoint] {
        &self.announcements
    }

    /// The shared response scalar `z = k + c·w` — the wire field the [`Partial`]
    /// codec serializes.
    ///
    /// [`Partial`]: crate::opening::Partial
    pub(crate) const fn response(&self) -> &Scalar {
        &self.z
    }

    /// Prove that one secret `w` relates every `bases[i]` to its image
    /// `w·bases[i]`, binding the proof to `binding`. Returns the canonical
    /// images `{w·bases[i]}` (so the caller publishes exactly what was proved)
    /// alongside the proof.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `bases` is empty or contains the identity.
    pub fn prove<R: RngCore + CryptoRng>(
        w: &Scalar,
        bases: &[ProjectivePoint],
        binding: &[u8],
        rng: &mut R,
    ) -> Result<(Vec<ProjectivePoint>, Self), Error> {
        check_bases(bases)?;

        let images: Vec<ProjectivePoint> = bases.iter().map(|b| *b * w).collect();
        let k = statement_bound_nonce(w, bases, &images, binding, rng);
        let announcements: Vec<ProjectivePoint> = bases.iter().map(|b| *b * *k).collect();

        let c = challenge(bases, &images, &announcements, binding);
        let z = *k + c * w;

        Ok((images, Self { announcements, z }))
    }

    /// Verify the proof against `(bases, images, binding)`: re-derive the
    /// challenge and check `z·B_i == A_i + c·P_i` on every leg.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `bases` is empty or contains the identity;
    /// [`Error::Verification`] on a shape mismatch or any failed leg.
    pub fn verify(
        &self,
        bases: &[ProjectivePoint],
        images: &[ProjectivePoint],
        binding: &[u8],
    ) -> Result<(), Error> {
        check_bases(bases)?;
        if images.len() != bases.len() || self.announcements.len() != bases.len() {
            return Err(Error::Verification("DLEQ statement shape mismatch"));
        }

        let c = challenge(bases, images, &self.announcements, binding);
        for ((base, image), announcement) in bases.iter().zip(images).zip(&self.announcements) {
            if *base * self.z != *announcement + *image * c {
                return Err(Error::Verification(
                    "DLEQ leg does not satisfy z·B = A + c·P",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// `L+1` distinct nonzero bases: `G` then `L` "segment" points `E_j = ρ_j·G`.
    fn sample_bases(rng: &mut StdRng, count: usize) -> Vec<ProjectivePoint> {
        (0..count)
            .map(|i| {
                if i == 0 {
                    ProjectivePoint::GENERATOR
                } else {
                    ProjectivePoint::GENERATOR * random_nonzero_scalar(rng)
                }
            })
            .collect()
    }

    #[test]
    fn prove_verify_round_trip_multi_base() {
        let mut rng = StdRng::seed_from_u64(0xD1_E9_00_01);
        for count in [1usize, 2, 5, 17] {
            let bases = sample_bases(&mut rng, count);
            let w = random_nonzero_scalar(&mut rng);
            let binding = b"core-hash|gate|ctx";
            let (images, proof) = BatchDleqProof::prove(&w, &bases, binding, &mut rng).unwrap();
            for (b, p) in bases.iter().zip(&images) {
                assert_eq!(*b * w, *p);
            }
            assert!(proof.verify(&bases, &images, binding).is_ok());
        }
    }

    #[test]
    fn rejects_wrong_binding() {
        let mut rng = StdRng::seed_from_u64(0xD1_E9_00_02);
        let bases = sample_bases(&mut rng, 4);
        let w = random_nonzero_scalar(&mut rng);
        let (images, proof) = BatchDleqProof::prove(&w, &bases, b"binding-A", &mut rng).unwrap();
        assert!(matches!(
            proof.verify(&bases, &images, b"binding-B"),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn rejects_tampered_announcement() {
        let mut rng = StdRng::seed_from_u64(0xD1_E9_00_03);
        let bases = sample_bases(&mut rng, 4);
        let w = random_nonzero_scalar(&mut rng);
        let (images, mut proof) = BatchDleqProof::prove(&w, &bases, b"b", &mut rng).unwrap();
        proof.announcements[1] += ProjectivePoint::GENERATOR;
        assert!(proof.verify(&bases, &images, b"b").is_err());
    }

    #[test]
    fn rejects_tampered_response() {
        let mut rng = StdRng::seed_from_u64(0xD1_E9_00_04);
        let bases = sample_bases(&mut rng, 4);
        let w = random_nonzero_scalar(&mut rng);
        let (images, mut proof) = BatchDleqProof::prove(&w, &bases, b"b", &mut rng).unwrap();
        proof.z += Scalar::ONE;
        assert!(proof.verify(&bases, &images, b"b").is_err());
    }

    #[test]
    fn rejects_tampered_image() {
        let mut rng = StdRng::seed_from_u64(0xD1_E9_00_05);
        let bases = sample_bases(&mut rng, 4);
        let w = random_nonzero_scalar(&mut rng);
        let (mut images, proof) = BatchDleqProof::prove(&w, &bases, b"b", &mut rng).unwrap();
        images[2] += ProjectivePoint::GENERATOR;
        assert!(proof.verify(&bases, &images, b"b").is_err());
    }

    #[test]
    fn rejects_split_witness_image() {
        // The practical face of special soundness: an honest proof for witness
        // `w` cannot be re-pointed at a statement whose leg 3 uses a *different*
        // witness (`W_3 = w'·E_3`, `w' ≠ w`). We forge that one image and verify
        // the honest proof's algebra against it; the shared `(c, z)` cannot
        // satisfy a leg whose image is off the common-`w` line, so it rejects.
        let mut rng = StdRng::seed_from_u64(0xD1_E9_00_06);
        let bases = sample_bases(&mut rng, 5);
        let w = random_nonzero_scalar(&mut rng);
        let w_other = random_nonzero_scalar(&mut rng);
        let (mut images, proof) = BatchDleqProof::prove(&w, &bases, b"b", &mut rng).unwrap();
        images[3] = bases[3] * w_other;
        assert!(proof.verify(&bases, &images, b"b").is_err());
    }

    #[test]
    fn fresh_nonce_per_proof() {
        let mut rng = StdRng::seed_from_u64(0xD1_E9_00_07);
        let bases = sample_bases(&mut rng, 3);
        let w = random_nonzero_scalar(&mut rng);
        let (_i1, p1) = BatchDleqProof::prove(&w, &bases, b"b", &mut rng).unwrap();
        let (_i2, p2) = BatchDleqProof::prove(&w, &bases, b"b", &mut rng).unwrap();
        assert_ne!(p1.announcements, p2.announcements);
    }

    struct RepeatingRng {
        byte: u8,
    }

    impl RngCore for RepeatingRng {
        fn next_u32(&mut self) -> u32 {
            u32::from_le_bytes([self.byte; 4])
        }

        fn next_u64(&mut self) -> u64 {
            u64::from_le_bytes([self.byte; 8])
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.fill(self.byte);
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    impl CryptoRng for RepeatingRng {}

    #[test]
    fn repeated_rng_output_does_not_reuse_contribution_nonce() {
        let mut rng = StdRng::seed_from_u64(0xD1_E9_00_0A);
        let bases = sample_bases(&mut rng, 4);
        let w = random_nonzero_scalar(&mut rng);

        let mut rng1 = RepeatingRng { byte: 0x42 };
        let (images1, proof1) = BatchDleqProof::prove(&w, &bases, b"binding-A", &mut rng1).unwrap();
        let mut rng2 = RepeatingRng { byte: 0x42 };
        let (images2, proof2) = BatchDleqProof::prove(&w, &bases, b"binding-B", &mut rng2).unwrap();

        assert!(proof1.verify(&bases, &images1, b"binding-A").is_ok());
        assert!(proof2.verify(&bases, &images2, b"binding-B").is_ok());
        assert_ne!(
            proof1.announcements, proof2.announcements,
            "repeated RNG bytes must not repeat the DLEQ nonce across bindings"
        );

        let c1 = challenge(&bases, &images1, proof1.announcements(), b"binding-A");
        let c2 = challenge(&bases, &images2, proof2.announcements(), b"binding-B");
        let denom = c1 - c2;
        assert!(
            !bool::from(denom.is_zero()),
            "distinct bindings should produce distinct DLEQ challenges"
        );
        let denom_inv = Option::<Scalar>::from(denom.invert()).unwrap();
        let extracted = (proof1.z - proof2.z) * denom_inv;
        assert_ne!(
            extracted, w,
            "nonce reuse across two challenges exposes the prover scalar"
        );
    }

    #[test]
    fn rejects_empty_and_identity_bases() {
        let mut rng = StdRng::seed_from_u64(0xD1_E9_00_08);
        let w = random_nonzero_scalar(&mut rng);
        assert!(matches!(
            BatchDleqProof::prove(&w, &[], b"b", &mut rng),
            Err(Error::DegenerateInput(_))
        ));
        let bad = vec![ProjectivePoint::GENERATOR, ProjectivePoint::IDENTITY];
        assert!(matches!(
            BatchDleqProof::prove(&w, &bad, b"b", &mut rng),
            Err(Error::DegenerateInput(_))
        ));
    }

    #[test]
    fn rejects_shape_mismatch() {
        let mut rng = StdRng::seed_from_u64(0xD1_E9_00_09);
        let bases = sample_bases(&mut rng, 4);
        let w = random_nonzero_scalar(&mut rng);
        let (images, proof) = BatchDleqProof::prove(&w, &bases, b"b", &mut rng).unwrap();
        assert!(matches!(
            proof.verify(&bases, &images[..3], b"b"),
            Err(Error::Verification(_))
        ));
    }
}
