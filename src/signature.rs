//! First-class signature backing for a capsule's opening core.
//!
//! A capsule is verified by its proofs **or** by a quorum signature over its
//! canonical statement — two mechanisms establishing the same fact ("this core is
//! authentic and its masks belong to `C`"), both yielding the same
//! [`VerifiedCapsule`](crate::VerifiedCapsule). Stripping discards the proofs and
//! relies on the signature: the quorum verified `π` (which welded the masks to
//! `C`) at provisioning, signed the core's statement, and verifying that signature
//! transfers the weld — closing the partial-decryption oracle that a bare stripped
//! core would otherwise open on `contribute`.
//!
//! The crate verifies the signature itself, so the `contribute` gate is
//! cryptographic, not a convention. The verifying key is **caller-supplied**: the
//! crate answers "is this signature valid under *this* key" (math); the caller
//! owns "*this* key is the legitimate quorum" (trust). There is deliberately no
//! caller-supplied-verifier escape hatch — the crate promotes only on a scheme it
//! verifies, so a future non-verifiable scheme is a real new tradeoff, not a plug-in.
//!
//! The one built-in scheme is BIP-340 Schnorr, matched to the quorum's
//! FROST(secp256k1, SHA-256)-TR ciphersuite: a FROST threshold signature under the
//! group key is an ordinary BIP-340 signature, so one verifier handles both FROST
//! and a plain single-key BIP-340 signer on that suite.

use crate::capsule::FrozenContext;
use crate::codec::{POINT_LEN, decode_point, encode_point};
use crate::composite::OpeningBinding;
use crate::context::Context;
use crate::elgamal::{LimbCiphertext, encode_limb};
use crate::error::Error;
use crate::params::Params;
use crate::transcript::{length_prefix, push_framed};
use k256::elliptic_curve::PrimeField;
use k256::elliptic_curve::bigint::U256;
use k256::elliptic_curve::ops::Reduce;
use k256::{FieldBytes, ProjectivePoint, Scalar};
use sha2::{Digest, Sha256};

/// Signing-layer domain separator for the attestation statement. The quorum's
/// group key also signs session tokens and authorizations, so a bare digest under
/// that key is a cross-protocol substitution target; this binds the statement to
/// the ve-capsule attestation role. Bump on any statement-layout change.
const ATTESTATION_DOMAIN: &[u8] = b"ve-capsule.attestation.v1";

/// Domain separator for the Case (additively-split secret) attestation statement.
/// Distinct from the single-capsule [`ATTESTATION_DOMAIN`] so a single-piece Case
/// signature can never be replayed as a single-capsule one (or vice versa).
const CASE_ATTESTATION_DOMAIN: &[u8] = b"ve-capsule.case-attestation.v1";

/// BIP-340 challenge tag, matching frost-secp256k1-tr's `H2`
/// (`tagged_hash("BIP0340/challenge")`).
const BIP340_CHALLENGE_TAG: &[u8] = b"BIP0340/challenge";

/// The signature scheme of a quorum attestation — secp256k1 only.
///
/// FROST emits a BIP-340 [`Schnorr`](Scheme::Schnorr) signature under the group
/// key. ECDSA (and any other scheme) is added together with its in-crate verifier;
/// there is deliberately no way to admit a scheme the crate does not verify.
/// `#[non_exhaustive]`: the scheme set is open — downstream matchers must not
/// assume Schnorr is the last variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Scheme {
    /// BIP-340 Schnorr on secp256k1 (covers FROST(secp256k1, SHA-256)-TR).
    Schnorr,
}

/// A quorum signature over a capsule's canonical attestation statement
/// (the crate-internal `attestation_statement`).
///
/// Tagged with its [`Scheme`]; the verifying key is supplied separately at
/// [`verify_signed`](crate::BoundCapsule::verify_signed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    scheme: Scheme,
    bytes: Vec<u8>,
}

impl Signature {
    /// A BIP-340 Schnorr signature from its 64-byte `R_x ‖ s` encoding (the form
    /// `frost-secp256k1-tr` serializes).
    #[must_use]
    pub fn schnorr(bytes: [u8; 64]) -> Self {
        Self {
            scheme: Scheme::Schnorr,
            bytes: bytes.to_vec(),
        }
    }

    /// The signature scheme.
    #[must_use]
    pub const fn scheme(&self) -> Scheme {
        self.scheme
    }

    /// The raw signature bytes (scheme-dependent encoding).
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// How a [`VerifiedCapsule`](crate::VerifiedCapsule) was established.
///
/// By its self-contained seal proof (trustless) or by a quorum signature
/// (delegated trust in the verifying-key holder). The opening operations are
/// identical; a consumer that needs trustless provenance can require
/// [`Proof`](Backing::Proof).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// Established by verifying the self-contained seal proof `π`.
    Proof,
    /// Established by verifying a quorum signature over the canonical statement.
    Signature,
}

/// The core identity digest: `SHA-256` over the core sub-range (`C ‖ masks`) using
/// the same encoders the wire uses. Shared by [`StrippedCapsule::digest`](crate::StrippedCapsule::digest)
/// and the [`attestation_statement`], so the bytes signed at provisioning and the
/// bytes rechecked at recovery are derived identically.
pub fn core_digest(elgamal: &[LimbCiphertext], c: &ProjectivePoint) -> [u8; 32] {
    let mut buf = Vec::with_capacity(POINT_LEN + elgamal.len() * 2 * POINT_LEN);
    buf.extend_from_slice(&encode_point(c));
    for ct in elgamal {
        encode_limb(ct, &mut buf);
    }
    Sha256::digest(&buf).into()
}

/// Build the canonical statement the quorum signs / the verifier checks:
///
/// `domain ‖ digest(core) ‖ recipient ‖ g* ‖ Y* ‖ ctx.domain ‖ ctx.binding ‖ params_id`
///
/// every component length-prefixed (no concatenation ambiguity). `digest(core)`
/// pins `C` and every mask — the substitution that kills the base-substitution
/// oracle; `recipient ‖ g* ‖ Y*` pin who and which gates (no roster replay);
/// `ctx.*` restore the replay protection decryption alone lacks (the group epoch
/// rides in `ctx.binding`, the framework's responsibility); `params_id` forks the
/// statement on any parameter change. Built once here so the producer
/// ([`attestation_message`](crate::VerifiedCapsule::attestation_message)) and the
/// checker ([`verify_signed`](crate::BoundCapsule::verify_signed)) cannot drift.
///
/// # Errors
///
/// [`Error::DegenerateInput`] if the (frozen) context's `binding_bytes` fails.
pub fn attestation_statement(
    core_digest: &[u8; 32],
    binding: &OpeningBinding,
    ctx: &FrozenContext,
) -> Result<Vec<u8>, Error> {
    let ctx_binding = ctx
        .binding_bytes()
        .map_err(|_| Error::DegenerateInput("context binding_bytes failed"))?;
    let mut out = Vec::new();
    push_framed(&mut out, ATTESTATION_DOMAIN);
    push_framed(&mut out, core_digest);
    push_framed(&mut out, &encode_point(&binding.recipient));
    push_framed(&mut out, &binding.g_star);
    push_framed(&mut out, &encode_point(&binding.y_star));
    push_framed(&mut out, ctx.domain().as_bytes());
    push_framed(&mut out, &ctx_binding);
    push_framed(&mut out, &Params::FROZEN.id());
    Ok(out)
}

/// Build the canonical statement for a [`Case`](crate::Case) — the additive split
/// `s = Σ σⱼ`:
///
/// `domain ‖ count ‖ {digest(coreⱼ)}↑ ‖ M ‖ recipient ‖ g* ‖ Y* ‖ ctx.domain ‖ ctx.binding ‖ params_id`
///
/// `{digest(coreⱼ)}↑` is the per-piece core digests **sorted ascending**, so the
/// statement is canonical regardless of the order the pieces happen to sit in (the
/// Case sums commitments, an order-free operation; binding a sorted list keeps
/// signer and verifier agreeing without imposing a piece order). The certified
/// target `M = Σ Mⱼ` is bound directly — the same fact [`Case::verify`] checks —
/// alongside the shared `recipient ‖ g* ‖ Y*` and `ctx`/`params`, blocking
/// cross-recipient / cross-epoch / cross-parameter replay. Built once here so the
/// producer ([`attestation_message`](crate::VerifiedCase::attestation_message)) and
/// the checker ([`verify_signed`](crate::BoundCase::verify_signed)) cannot drift.
///
/// # Errors
///
/// [`Error::DegenerateInput`] if the (frozen) context's `binding_bytes` fails.
pub fn case_attestation_statement(
    piece_digests: &[[u8; 32]],
    commitment: &ProjectivePoint,
    binding: &OpeningBinding,
    ctx: &FrozenContext,
) -> Result<Vec<u8>, Error> {
    let ctx_binding = ctx
        .binding_bytes()
        .map_err(|_| Error::DegenerateInput("context binding_bytes failed"))?;
    let mut sorted = piece_digests.to_vec();
    sorted.sort_unstable();
    let mut out = Vec::new();
    push_framed(&mut out, CASE_ATTESTATION_DOMAIN);
    out.extend_from_slice(&length_prefix(sorted.len()));
    for digest in &sorted {
        push_framed(&mut out, digest);
    }
    push_framed(&mut out, &encode_point(commitment));
    push_framed(&mut out, &encode_point(&binding.recipient));
    push_framed(&mut out, &binding.g_star);
    push_framed(&mut out, &encode_point(&binding.y_star));
    push_framed(&mut out, ctx.domain().as_bytes());
    push_framed(&mut out, &ctx_binding);
    push_framed(&mut out, &Params::FROZEN.id());
    Ok(out)
}

/// Reject a Case whose pieces include two byte-identical opening cores.
///
/// Honest sealing never produces a duplicate — each seal draws fresh `ElGamal`
/// randomness, so even two equal secrets get distinct masks and distinct digests
/// (the legitimate equal-secret split). A duplicate can only come from a
/// misbehaving dealer, and it is a gated-opening footgun: identical cores share
/// their DLEQ bases, so one authorizer's `Partial` verifies against every copy and
/// the gate bucket overcounts — a gated Case with duplicates verifies (and could be
/// signed) yet cannot be opened by its own `contribute` output. Screening here,
/// alongside the mask/key degeneracy gates, keeps the gated path openable.
///
/// # Errors
///
/// [`Error::Verification`] if any two piece digests are equal.
pub fn reject_duplicate_cores(piece_digests: &[[u8; 32]]) -> Result<(), Error> {
    let mut sorted = piece_digests.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::Verification("case contains duplicate piece cores"));
    }
    Ok(())
}

/// The 32-byte message a quorum actually signs: `SHA-256` over a canonical
/// attestation statement ([`attestation_statement`] / [`case_attestation_statement`]).
///
/// The statement can be long (its context binding may reach 64 `KiB`), but the
/// framework's FROST round signs a fixed 32-byte digest carried by a typed signing
/// intent — so the producer reduces the statement to this digest before signing and
/// [`verify_signed`](crate::BoundCapsule::verify_signed) reduces it the same way
/// before checking. The full statement stays the auditable artifact; this is what
/// the signature covers.
#[must_use]
pub fn attestation_digest(statement: &[u8]) -> [u8; 32] {
    Sha256::digest(statement).into()
}

/// Verify `sig` over `msg` against the caller-supplied verifying key, dispatching
/// on the signature scheme. The crate does the verification, so the promotion gate
/// is cryptographic.
///
/// # Errors
///
/// [`Error::Verification`] if the signature is malformed or does not verify.
pub fn verify_signature(
    sig: &Signature,
    verifying_key: &[u8; 32],
    msg: &[u8],
) -> Result<(), Error> {
    match sig.scheme {
        Scheme::Schnorr => {
            let bytes: &[u8; 64] = sig
                .bytes
                .as_slice()
                .try_into()
                .map_err(|_| Error::Verification("schnorr signature must be 64 bytes"))?;
            verify_schnorr_bip340(verifying_key, msg, bytes)
        }
    }
}

/// Reduce 32 big-endian bytes to a secp256k1 scalar (mod `n`).
fn reduce_be_to_scalar(bytes: [u8; 32]) -> Scalar {
    let mut repr = FieldBytes::default();
    repr.copy_from_slice(&bytes);
    <Scalar as Reduce<U256>>::reduce_bytes(&repr)
}

/// The BIP-340 challenge `e = int(SHA256t("BIP0340/challenge", r ‖ P_x ‖ m)) mod n`,
/// using the double-SHA256 tagged-hash construction frost-secp256k1-tr's `H2` uses.
fn bip340_challenge(r: &[u8], p_x: &[u8; 32], m: &[u8]) -> Scalar {
    let tag = Sha256::digest(BIP340_CHALLENGE_TAG);
    let mut h = Sha256::new();
    h.update(tag);
    h.update(tag);
    h.update(r);
    h.update(p_x);
    h.update(m);
    let digest: [u8; 32] = h.finalize().into();
    reduce_be_to_scalar(digest)
}

/// Verify a BIP-340 Schnorr signature, matching FROST(secp256k1, SHA-256)-TR
/// bit-for-bit. `verifying_key` is the 32-byte x-only group key; it lifts to its
/// even-`Y` point (SEC1 `0x02 ‖ x`, BIP-340 `lift_x`). The signature is `r ‖ s`
/// (`r = R_x`, `s < n`). Check `R = s·G − e·P` is non-identity, even-`Y`, with
/// `x(R) == r`.
///
/// # Errors
///
/// [`Error::Verification`] on a non-canonical `s` (`≥ n`), an invalid verifying key
/// (off-curve / `x ≥ p`), or a signature that does not satisfy the BIP-340 equation.
fn verify_schnorr_bip340(
    verifying_key: &[u8; 32],
    msg: &[u8],
    sig: &[u8; 64],
) -> Result<(), Error> {
    // Lift the x-only key to its even-Y point (SEC1 0x02 ‖ x); decode_point rejects
    // off-curve / x ≥ p / identity.
    let mut vk_sec1 = [0u8; POINT_LEN];
    vk_sec1[0] = 0x02;
    vk_sec1[1..].copy_from_slice(verifying_key);
    let p = decode_point(&vk_sec1)?;

    // s must be a canonical scalar (< n); r is compared against x(R) below, so an
    // r ≥ p simply cannot match (x(R) < p) and needs no separate range check.
    let (r, s_bytes) = sig.split_at(32);
    let mut s_repr = FieldBytes::default();
    s_repr.copy_from_slice(s_bytes);
    let s = Option::<Scalar>::from(Scalar::from_repr(s_repr)).ok_or(Error::Verification(
        "schnorr signature s is not canonical (>= n)",
    ))?;

    let e = bip340_challenge(r, verifying_key, msg);
    let r_point = ProjectivePoint::GENERATOR * s - p * e;

    // R must be non-identity (identity encodes to 0x00…), even-Y (tag 0x02), and
    // x(R) == r. encode_point gives 0x02/0x03 ‖ x for a real point, 33 zeros for O.
    let r_enc = encode_point(&r_point);
    if r_enc[0] != 0x02 || r_enc[1..] != *r {
        return Err(Error::Verification("schnorr signature does not verify"));
    }
    Ok(())
}

/// Real-signer test fixtures: produce signatures from the quorum's actual
/// FROST(secp256k1)-TR ciphersuite so the crate's verifier is tested against the
/// real thing, not a re-implementation. Shared by the [`signature`] verifier tests
/// and the [`crate::stripped`] end-to-end tests; test-only.
#[cfg(test)]
pub mod frost_test_support {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use std::collections::BTreeMap;

    pub type Id = frost_secp256k1_tr::Identifier;
    pub type KeyPackage = frost_secp256k1_tr::keys::KeyPackage;
    pub type PublicKeyPackage = frost_secp256k1_tr::keys::PublicKeyPackage;

    /// Trusted-dealer keygen for a `threshold`-of-`total` group.
    pub fn keygen(threshold: u16, total: u16) -> (BTreeMap<Id, KeyPackage>, PublicKeyPackage) {
        let ids: Vec<Id> = (1..=total).map(|i| Id::try_from(i).unwrap()).collect();
        let (shares, pubkeys) = frost_secp256k1_tr::keys::generate_with_dealer(
            total,
            threshold,
            frost_secp256k1_tr::keys::IdentifierList::Custom(&ids),
            rand::rngs::OsRng,
        )
        .unwrap();
        let packages = shares
            .into_iter()
            .map(|(id, share)| (id, share.try_into().unwrap()))
            .collect();
        (packages, pubkeys)
    }

    /// The 32-byte x-only group key — what the crate's verifier takes.
    pub fn group_xonly(pubkeys: &PublicKeyPackage) -> [u8; 32] {
        let sec1 = pubkeys.verifying_key().serialize().unwrap();
        sec1[1..33].try_into().unwrap()
    }

    /// A 2-of-`n` FROST signature over `msg`, serialized BIP-340 (64 B `R_x ‖ s`).
    pub fn sign(
        packages: &BTreeMap<Id, KeyPackage>,
        pubkeys: &PublicKeyPackage,
        msg: &[u8],
    ) -> [u8; 64] {
        use frost_secp256k1_tr::Ciphersuite as _;
        let mut rng = rand::rngs::OsRng;
        let signers: Vec<&KeyPackage> = packages.values().take(2).collect();
        let mut nonces = BTreeMap::new();
        let mut commitments = BTreeMap::new();
        for kp in &signers {
            let (n, c) = frost_secp256k1_tr::round1::commit(kp.signing_share(), &mut rng);
            nonces.insert(*kp.identifier(), n);
            commitments.insert(*kp.identifier(), c);
        }
        let pkg = frost_secp256k1_tr::SigningPackage::new(commitments, msg);
        let mut shares = BTreeMap::new();
        for kp in &signers {
            let n = nonces.get(kp.identifier()).unwrap();
            shares.insert(
                *kp.identifier(),
                frost_core::round2::sign(&pkg, n, kp).unwrap(),
            );
        }
        let sig = frost_core::aggregate(&pkg, &shares, pubkeys).unwrap();
        frost_secp256k1_tr::Secp256K1Sha256TR::serialize_signature(&sig)
            .unwrap()
            .try_into()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::frost_test_support::{group_xonly, keygen, sign};
    use super::*;

    // Ground-truth interop: produce an actual FROST threshold signature and
    // confirm the hand-rolled BIP-340 verifier accepts it (and rejects tampering),
    // so the verifier matches the signer bit-for-bit rather than by argument.

    #[test]
    fn verifies_real_frost_signature() {
        let (packages, pubkeys) = keygen(2, 3);
        let vk = group_xonly(&pubkeys);
        // An arbitrary-length message, like the canonical attestation statement.
        let msg = b"ve-capsule attestation statement bytes of arbitrary length";
        let sig = sign(&packages, &pubkeys, msg);
        assert!(
            verify_schnorr_bip340(&vk, msg, &sig).is_ok(),
            "the hand-rolled BIP-340 verifier must accept a real frost-secp256k1-tr signature"
        );
    }

    #[test]
    fn rejects_tampered_message_and_wrong_key() {
        let (packages, pubkeys) = keygen(2, 3);
        let vk = group_xonly(&pubkeys);
        let msg = b"the signed message";
        let sig = sign(&packages, &pubkeys, msg);

        // Tampered message.
        assert!(verify_schnorr_bip340(&vk, b"a different message", &sig).is_err());
        // Wrong key (a fresh group).
        let (_p2, pubkeys2) = keygen(2, 3);
        assert!(verify_schnorr_bip340(&group_xonly(&pubkeys2), msg, &sig).is_err());
        // Flipped signature byte.
        let mut bad = sig;
        bad[0] ^= 0x01;
        assert!(verify_schnorr_bip340(&vk, msg, &bad).is_err());
        // Flipped s byte (last).
        let mut bad_s = sig;
        bad_s[63] ^= 0x01;
        assert!(verify_schnorr_bip340(&vk, msg, &bad_s).is_err());
    }

    #[test]
    fn rejects_non_canonical_s() {
        // s = n (the group order) is non-canonical (≥ n) and must be rejected.
        let (packages, pubkeys) = keygen(2, 3);
        let vk = group_xonly(&pubkeys);
        let mut sig = sign(&packages, &pubkeys, b"m");
        // secp256k1 group order n, big-endian, into the s half.
        let n_be: [u8; 32] = [
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFE, 0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B, 0xBF, 0xD2, 0x5E, 0x8C,
            0xD0, 0x36, 0x41, 0x41,
        ];
        sig[32..].copy_from_slice(&n_be);
        assert!(matches!(
            verify_schnorr_bip340(&vk, b"m", &sig),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn reject_duplicate_cores_detects_repeats() {
        let a = [1u8; 32];
        let b = [2u8; 32];
        assert!(reject_duplicate_cores(&[]).is_ok());
        assert!(reject_duplicate_cores(&[a]).is_ok());
        assert!(reject_duplicate_cores(&[a, b]).is_ok());
        assert!(matches!(
            reject_duplicate_cores(&[a, b, a]),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn verify_signature_dispatch_matches_raw() {
        let (packages, pubkeys) = keygen(2, 3);
        let vk = group_xonly(&pubkeys);
        let msg = b"dispatch test";
        let raw = sign(&packages, &pubkeys, msg);
        let sig = Signature::schnorr(raw);
        assert_eq!(sig.scheme(), Scheme::Schnorr);
        assert_eq!(sig.bytes(), raw.as_slice());
        assert!(verify_signature(&sig, &vk, msg).is_ok());
        // Wrong-length bytes are rejected, not panicked.
        let short = Signature {
            scheme: Scheme::Schnorr,
            bytes: vec![0u8; 63],
        };
        assert!(verify_signature(&short, &vk, msg).is_err());
    }
}
