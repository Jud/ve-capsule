//! The public capsule API: seal a scalar to a recipient behind optional access
//! gates, confirm a capsule against an authorization, and open it.
//!
//! This is the blob-in / blob-out surface over the [`crate::composite`] seal and
//! the [`crate::opening`] contribute/unseal layers. The flow is:
//!
//! 1. `Capsule::builder(m, recipient, ctx).access_key(g)…seal()` → an opaque
//!    [`Capsule`].
//! 2. `capsule.verify(expected_pubkey, expected_recipient, expected_access_keys,
//!    ctx)` confirms the capsule against the authorization and returns a
//!    [`VerifiedCapsule`] capability token — the **only** place `contribute` and
//!    `unseal` exist, so neither runs on an unconfirmed capsule.
//! 3. An authorizer `vc.contribute(key)` → a [`Partial`] (or
//!    `vc.contribute_for_gate(key, gate)` toward an aggregate gate); the
//!    recipient `vc.unseal(recipient_secret, &partials)` → the recovered scalar.
//!
//! `verify` runs the (dominant-cost) seal-proof `π` once; `VerifiedCapsule`
//! carries the confirmed state so `unseal` opens **without** re-verifying `π` —
//! the latency path for constrained recovery.

use crate::assembly::Proof;
use crate::codec::{POINT_LEN, decode_point, encode_point};
use crate::composite::{self, OpeningBinding};
use crate::context::Context;
use crate::error::Error;
use crate::generators::g;
use crate::opening::{self, CapsuleRef, Partial};
use crate::signature::{self, Backing};
use crate::stripped::StrippedCapsule;
use k256::elliptic_curve::PrimeField;
use k256::{FieldBytes, ProjectivePoint, Scalar};
use rand_core::OsRng;
use std::borrow::Cow;
use zeroize::Zeroizing;

/// Wire magic for a canonical capsule blob; the kind+version envelope.
const WIRE_MAGIC: &[u8] = b"ve-capsule.cap.v1";

/// Wire format version. Bump only after an incompatible released format exists.
const WIRE_VERSION: u8 = 1;

/// A secret secp256k1 scalar — the sealed `m`, a recipient or access key, or one
/// participant's contribution. Zeroizes on drop; its public partner is
/// `scalar·G`.
#[derive(zeroize::Zeroize, zeroize::ZeroizeOnDrop)]
pub struct PrivateKey {
    scalar: Zeroizing<Scalar>,
}

impl PrivateKey {
    /// Instantiate from 32 big-endian secret bytes. Rejects a non-canonical
    /// (`≥ n`) or zero scalar — a party holds its own secret and derives its
    /// [`PublicKey`] with [`PrivateKey::public_key`].
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `bytes` is not 32 bytes, encodes `≥ n`, or
    /// is zero.
    pub fn from_secret(bytes: &[u8]) -> Result<Self, Error> {
        let array: [u8; 32] = bytes
            .try_into()
            .map_err(|_| Error::DegenerateInput("private key must be 32 bytes"))?;
        let repr = FieldBytes::from(array);
        let scalar = Option::<Scalar>::from(Scalar::from_repr(repr)).ok_or(
            Error::DegenerateInput("private key is not a canonical scalar"),
        )?;
        if bool::from(scalar.is_zero()) {
            return Err(Error::DegenerateInput("private key is zero"));
        }
        Ok(Self {
            scalar: Zeroizing::new(scalar),
        })
    }

    /// The public partner `scalar·G`. For the sealed `m`, this is exactly what
    /// [`Capsule::verify`] confirms as `expected_pubkey`.
    #[must_use]
    pub fn public_key(&self) -> PublicKey {
        PublicKey {
            point: g() * *self.scalar,
        }
    }

    /// Export the secret as 32 big-endian bytes — the exact inverse of
    /// [`PrivateKey::from_secret`].
    ///
    /// This is the **storage / recovery boundary**: the sanctioned way a
    /// recovered secret — the `s = Σ σⱼ` that
    /// [`VerifiedCase::unseal`](crate::VerifiedCase::unseal) returns — leaves the
    /// crate, so a caller can hand it to its own keystore or key type (e.g. a
    /// FROST signing share). The returned buffer zeroizes on drop; the caller is
    /// responsible for not widening its lifetime.
    #[must_use]
    pub fn to_secret_bytes(&self) -> Zeroizing<[u8; 32]> {
        Zeroizing::new(self.scalar.to_bytes().into())
    }

    /// Borrow the inner scalar (crate-internal — never leaves the crate).
    pub(crate) fn scalar(&self) -> &Scalar {
        &self.scalar
    }

    /// Wrap a recovered scalar as a `PrivateKey` (crate-internal; the unseal
    /// result). Zero is allowed here: an opened `m` may legitimately be zero.
    pub(crate) fn from_scalar(scalar: Scalar) -> Self {
        Self {
            scalar: Zeroizing::new(scalar),
        }
    }
}

/// A secp256k1 public point: a recipient key, an access key (gate), or the
/// public partner of a [`PrivateKey`]. Non-identity by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicKey {
    point: ProjectivePoint,
}

impl PublicKey {
    /// Parse from canonical 33-byte SEC1. Rejects the identity and any
    /// off-curve / non-canonical encoding.
    ///
    /// # Errors
    ///
    /// [`Error::PointDecode`] on a malformed encoding; [`Error::DegenerateInput`]
    /// if the point is the identity.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let point = decode_point(bytes)?;
        if point == ProjectivePoint::IDENTITY {
            return Err(Error::DegenerateInput("public key is the identity"));
        }
        Ok(Self { point })
    }

    /// Canonical 33-byte SEC1 encoding.
    #[must_use]
    pub fn to_canonical_bytes(&self) -> [u8; POINT_LEN] {
        encode_point(&self.point)
    }

    pub(crate) const fn point(&self) -> ProjectivePoint {
        self.point
    }
}

/// An opaque, immutable sealed capsule: the ec-segve proof and its commitment
/// `C = m·G`. Never holds plaintext; produced only by [`CapsuleBuilder::seal`].
pub struct Capsule {
    proof: Proof,
    c: ProjectivePoint,
}

impl Capsule {
    /// Build a capsule from raw proof parts for adversarial verifier tests.
    #[cfg(test)]
    pub(crate) const fn from_parts_for_test(proof: Proof, c: ProjectivePoint) -> Self {
        Self { proof, c }
    }

    /// Start sealing `m` to `recipient`, bound to `ctx`. Access-key gates are
    /// added on the returned [`CapsuleBuilder`].
    #[must_use]
    pub const fn builder<'a, C: Context + ?Sized>(
        m: &'a PrivateKey,
        recipient: &'a PublicKey,
        ctx: &'a C,
    ) -> CapsuleBuilder<'a, C> {
        CapsuleBuilder {
            m,
            recipient,
            ctx,
            gates: Vec::new(),
        }
    }

    /// Confirm the capsule against the authorization: the seal proof `π` under
    /// `ctx`, the **exact** access-key set (via the sealed gate commitment), the
    /// recipient, and that the committed secret is `expected_pubkey`
    /// (`C == expected_pubkey`, the §6.2 informed-consent gate). Returns the
    /// [`VerifiedCapsule`] token — the only path to `contribute`/`unseal`.
    ///
    /// # Errors
    ///
    /// [`Error::Verification`] if `π` fails, the commitment is not
    /// `expected_pubkey`, or the recipient/gate set does not match;
    /// [`Error::DegenerateInput`] on a degenerate expected input.
    pub fn verify<C: Context + ?Sized>(
        &self,
        expected_pubkey: &PublicKey,
        expected_recipient: &PublicKey,
        expected_access_keys: &[PublicKey],
        ctx: &C,
    ) -> Result<VerifiedCapsule<'_>, Error> {
        // Freeze the caller's context ONCE, then verify π and open against that
        // snapshot — so a mutable/nondeterministic `Context` cannot bind a
        // different context to the open than the one π was checked under.
        let frozen = FrozenContext::capture(ctx)?;
        let access: Vec<ProjectivePoint> =
            expected_access_keys.iter().map(PublicKey::point).collect();
        let binding = composite::verify_with_binding(
            &self.proof,
            &self.c,
            &expected_recipient.point,
            &access,
            &frozen,
        )?;
        if self.c != expected_pubkey.point {
            return Err(Error::Verification(
                "capsule commitment does not match expected public key",
            ));
        }
        Ok(VerifiedCapsule {
            core: self.as_capsule_ref(),
            binding,
            ctx: frozen,
            backing: Backing::Proof,
        })
    }

    /// Confirm an **ungated** capsule — one sealed with no access-key gates.
    /// Convenience for [`verify`](Self::verify) with an empty access set: it
    /// checks the seal proof `π` under `ctx`, the recipient, and that the
    /// committed secret is `expected_pubkey`. It still **rejects** a capsule that
    /// actually carries gates (the gate set must match exactly), so it cannot
    /// wave a gated capsule through as if it were open.
    ///
    /// # Errors
    ///
    /// As [`verify`](Self::verify).
    pub fn verify_ungated<C: Context + ?Sized>(
        &self,
        expected_pubkey: &PublicKey,
        expected_recipient: &PublicKey,
        ctx: &C,
    ) -> Result<VerifiedCapsule<'_>, Error> {
        self.verify(expected_pubkey, expected_recipient, &[], ctx)
    }

    /// The commitment `C = m·G` (crate-internal — a [`Case`](crate::Case) sums
    /// these to check completeness).
    pub(crate) const fn commitment(&self) -> ProjectivePoint {
        self.c
    }

    /// The sealed proof `π` (crate-internal). The one operation that re-runs π —
    /// [`composite::verify_bound`] — needs the full proof; opening needs only the
    /// core view ([`Self::as_capsule_ref`]).
    pub(crate) const fn proof(&self) -> &Proof {
        &self.proof
    }

    /// A borrowed view of the sealed opening core (crate-internal).
    pub(crate) fn as_capsule_ref(&self) -> CapsuleRef<'_> {
        CapsuleRef {
            elgamal: self.proof.elgamal(),
            c: self.c,
        }
    }

    /// Canonical wire bytes: `magic ‖ version ‖ C ‖ proof`. Deterministic — the
    /// same capsule always encodes identically (re-encode equality). Store /
    /// transport this opaque blob (I3); do not introspect it.
    #[must_use]
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let proof = self.proof.to_canonical_bytes();
        let mut out = Vec::with_capacity(WIRE_MAGIC.len() + 1 + POINT_LEN + proof.len());
        out.extend_from_slice(WIRE_MAGIC);
        out.push(WIRE_VERSION);
        out.extend_from_slice(&encode_point(&self.c));
        out.extend_from_slice(&proof);
        out
    }

    /// Parse a capsule from canonical wire bytes — the only deserialization door
    /// (there is no derived `Deserialize`, so a `Capsule` cannot be fabricated
    /// from arbitrary bytes bypassing validation). Validates the framing, decodes
    /// `C` and every proof point/scalar strictly (off-curve / non-canonical
    /// rejected), and enforces **re-encode equality** (the input must be the
    /// canonical encoding). `π` validity stays in [`Capsule::verify`].
    ///
    /// # Errors
    ///
    /// [`Error::PointDecode`] on bad magic/version, a malformed point/scalar, a
    /// length mismatch, or non-canonical input (re-encode mismatch);
    /// [`Error::DegenerateInput`] if a segment mask decodes to the identity.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let rest = bytes
            .strip_prefix(WIRE_MAGIC)
            .ok_or(Error::PointDecode("capsule: bad magic"))?;
        let (&version, rest) = rest
            .split_first()
            .ok_or(Error::PointDecode("capsule: truncated header"))?;
        if version != WIRE_VERSION {
            return Err(Error::PointDecode("capsule: unsupported version"));
        }
        let (c_bytes, proof_bytes) = rest
            .split_at_checked(POINT_LEN)
            .ok_or(Error::PointDecode("capsule: truncated commitment"))?;
        let c = decode_point(c_bytes)?;
        let proof = Proof::from_canonical_bytes(proof_bytes)?;
        let capsule = Self { proof, c };
        // Re-encode equality: the canonical decode is a bijection on canonical
        // inputs, so a mismatch means the input was non-canonical.
        if capsule.to_canonical_bytes() != bytes {
            return Err(Error::PointDecode("capsule: non-canonical encoding"));
        }
        Ok(capsule)
    }
}

/// Accumulates the optional access-key gates, then seals. Borrows the plaintext
/// (the caller's [`PrivateKey`] owns it); `seal` never mutates the caller's `m`.
pub struct CapsuleBuilder<'a, C: Context + ?Sized> {
    m: &'a PrivateKey,
    recipient: &'a PublicKey,
    ctx: &'a C,
    gates: Vec<ProjectivePoint>,
}

impl<C: Context + ?Sized> CapsuleBuilder<'_, C> {
    /// Add one access-key gate (repeatable; **all** added gates are required — an
    /// AND).
    #[must_use]
    pub fn access_key(mut self, key: &PublicKey) -> Self {
        self.gates.push(key.point);
        self
    }

    /// Add several access-key gates at once.
    #[must_use]
    pub fn access_keys(mut self, keys: &[PublicKey]) -> Self {
        self.gates.extend(keys.iter().map(PublicKey::point));
        self
    }

    /// Seal: coefficiented `Y*` over recipient + gates, gate commitment `g*`,
    /// EC-ElGamal + `π`. No gates ⇒ a plain single-recipient capsule.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] on a degenerate recipient/gate/aggregate or
    /// composite key; otherwise a sub-proof error.
    pub fn seal(self) -> Result<Capsule, Error> {
        let (proof, c) = composite::seal(
            self.m.scalar(),
            &self.recipient.point,
            &self.gates,
            self.ctx,
            &mut OsRng,
        )?;
        Ok(Capsule { proof, c })
    }
}

/// A snapshot of the caller's [`Context`] taken at `verify`, so the token binds
/// the *exact* context `π` was checked against — not a live `&C` that a mutable
/// or nondeterministic impl could change between verify and open. `domain` is
/// already `&'static`; only `binding_bytes` can vary, so it is captured once.
pub struct FrozenContext {
    domain: &'static str,
    binding: Vec<u8>,
}

impl FrozenContext {
    /// Capture a one-shot snapshot of `ctx`. Used by both `Capsule::verify` and
    /// `Case::verify` so the opening side cannot drift from what `π` was bound to.
    pub fn capture<C: Context + ?Sized>(ctx: &C) -> Result<Self, Error> {
        Ok(Self {
            domain: ctx.domain(),
            binding: ctx
                .binding_bytes()
                .map_err(|_| Error::DegenerateInput("context binding_bytes failed"))?
                .into_owned(),
        })
    }
}

impl Context for FrozenContext {
    fn domain(&self) -> &'static str {
        self.domain
    }

    fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
        Ok(Cow::Borrowed(&self.binding))
    }
}

/// A capsule confirmed against an authorization (`verify` passed).
///
/// The only place `contribute`/`unseal` exist, so neither runs on an unconfirmed
/// capsule. Carries the confirmed opening binding **and a frozen snapshot of the
/// verified context**, so `unseal` opens **without** re-verifying `π` yet can
/// never drift to a different context than the one `π` bound.
pub struct VerifiedCapsule<'a> {
    core: CapsuleRef<'a>,
    binding: OpeningBinding,
    ctx: FrozenContext,
    backing: Backing,
}

impl<'a> VerifiedCapsule<'a> {
    /// Assemble a verified token from already-verified parts (crate-internal). The
    /// signature path ([`BoundCapsule::verify_signed`](crate::BoundCapsule::verify_signed))
    /// builds the token here with [`Backing::Signature`]; the proof path uses the
    /// inherent constructor in [`Capsule::verify`] with [`Backing::Proof`].
    pub(crate) const fn from_parts(
        core: CapsuleRef<'a>,
        binding: OpeningBinding,
        ctx: FrozenContext,
        backing: Backing,
    ) -> Self {
        Self {
            core,
            binding,
            ctx,
            backing,
        }
    }
}

impl VerifiedCapsule<'_> {
    const fn capsule_ref(&self) -> CapsuleRef<'_> {
        self.core
    }

    /// How this token was established — proof (trustless) or signature (delegated
    /// to the verifying-key holder). The opening operations are identical; a
    /// consumer that needs trustless provenance can require [`Backing::Proof`].
    #[must_use]
    pub const fn backing(&self) -> Backing {
        self.backing
    }

    /// The canonical statement the quorum signs at provisioning to back this core
    /// with a signature: `domain ‖ digest(core) ‖ recipient ‖ g* ‖ Y* ‖ ctx ‖
    /// params`. The framework reduces these bytes to a 32-byte digest (via a typed
    /// signing intent), runs its FROST round over that digest, and attaches the
    /// resulting signature to the stripped artifact;
    /// [`BoundCapsule::verify_signed`](crate::BoundCapsule::verify_signed) later
    /// re-derives the identical bytes through the same builder and reduces them the
    /// same way, so signer and verifier agree by construction.
    ///
    /// **Contract:** the signature must cover `SHA-256` of the returned bytes, not
    /// the bytes themselves — that 32-byte digest is what
    /// [`BoundCapsule::verify_signed`](crate::BoundCapsule::verify_signed) checks
    /// and what the framework's typed signing intent sets as its `message_hash`.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if the verified context's `binding_bytes` fails.
    pub fn attestation_message(&self) -> Result<Vec<u8>, Error> {
        let digest = signature::core_digest(self.core.elgamal, &self.core.c);
        signature::attestation_statement(&digest, &self.binding, &self.ctx)
    }

    /// An authorizer's contribution toward opening, using `key` as a self-held
    /// access key: the gate is `key.public_key()`. Use this when the contributor
    /// holds the whole access key the capsule is gated on.
    ///
    /// For a threshold authorizer (a "key no one holds", where `key` is one
    /// participant's additive piece of an aggregate gate), use
    /// [`contribute_for_gate`](Self::contribute_for_gate) and pass the aggregate
    /// gate. If the contributor's own `key.public_key()` is *also* separately
    /// listed as a gate, this short form contributes to that own-key gate rather
    /// than the aggregate: it fails closed (the aggregate bucket stays
    /// unsatisfied) but is an intent trap, so threshold participants should always
    /// use `contribute_for_gate`.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `key.public_key()` is not in the access set
    /// or the context binding fails; otherwise a DLEQ error.
    pub fn contribute(&self, key: &PrivateKey) -> Result<Partial, Error> {
        self.contribute_for_gate(key, &key.public_key())
    }

    /// An authorizer's contribution for an explicit `gate`, using `key` (the
    /// scalar behind that gate, or one participant's additive piece of it). The
    /// recipient accepts a gate's bucket only when the contributed points sum to
    /// it (`Σ W == gate`), so several participants can each pass their piece with
    /// the same aggregate `gate`. Rejects a `gate` not in the capsule's confirmed
    /// access-key set.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `gate` is not in the access set or the
    /// context binding fails; otherwise a DLEQ error.
    pub fn contribute_for_gate(
        &self,
        key: &PrivateKey,
        gate: &PublicKey,
    ) -> Result<Partial, Error> {
        opening::contribute(
            self.capsule_ref(),
            &self.binding,
            &gate.point,
            key.scalar(),
            &self.ctx,
            &mut OsRng,
        )
    }

    /// The recipient's terminal op: open from `recipient`'s secret plus the
    /// authorizers' `partials`. Skips the seal-proof re-verify (this token proves
    /// `π` already passed under the frozen context), so it pays only the partial
    /// DLEQs + strip + BSGS.
    ///
    /// # Errors
    ///
    /// [`Error::Verification`] if a gate bucket is not qualifying, a limb is
    /// unrecoverable, or the recovered scalar's commitment is not `C`.
    pub fn unseal(
        &self,
        recipient: &PrivateKey,
        partials: &[Partial],
    ) -> Result<PrivateKey, Error> {
        let m = opening::unseal_verified(
            self.capsule_ref(),
            &self.binding,
            recipient.scalar(),
            partials,
            &self.ctx,
        )?;
        Ok(PrivateKey::from_scalar(m))
    }

    /// Strip the seal proof, keeping only the opening core (`C` + masks) for
    /// compact storage. Only a [`VerifiedCapsule`] can be stripped — the honest
    /// encoder — so the core carried forward is one whose `π` already passed.
    /// Recover it later via [`StrippedCapsule::bind`] + the self-securing
    /// [`BoundCapsule::unseal`](crate::BoundCapsule::unseal).
    #[must_use]
    pub fn strip(&self) -> StrippedCapsule {
        StrippedCapsule::from_core(self.core.elgamal.to_vec(), self.core.c)
    }
}

// ── Mirror verb-traits (§3.0): `key.verb(target)` reads as `target.verb(key)` ──
//
// Each is a one-line forward to the inherent method (the canonical impl), so the
// actor (`PrivateKey` — the only type that *does* anything) drives a second
// call-style without a second implementation.

/// Seal `self` (the plaintext scalar) — the mirror of [`CapsuleBuilder::seal`].
pub trait Seal {
    /// Seal to `recipient` behind `access_keys`, bound to `ctx`.
    ///
    /// # Errors
    ///
    /// As [`CapsuleBuilder::seal`].
    fn seal<C: Context + ?Sized>(
        &self,
        recipient: &PublicKey,
        access_keys: &[PublicKey],
        ctx: &C,
    ) -> Result<Capsule, Error>;
}

/// Contribute toward opening `target` (a verified capsule or case): the mirror
/// of the token's `contribute` / `contribute_for_gate`.
pub trait Contribute<T> {
    /// The contribution artifact (`Partial`, or `Vec<Partial>` for a case).
    type Output;
    /// Contribute using `self` as a self-held access key (gate = `self.public_key()`).
    fn contribute(&self, target: &T) -> Self::Output;
    /// Contribute toward an explicit `gate` (the threshold/share path).
    fn contribute_for_gate(&self, target: &T, gate: &PublicKey) -> Self::Output;
}

/// Open `target` (a verified capsule or case) — the mirror of the token's
/// `unseal`.
pub trait Unseal<T> {
    /// The recovered secret.
    type Output;
    /// Open `target` with `self` (the recipient secret) and `partials`.
    fn unseal(&self, target: &T, partials: &[Partial]) -> Self::Output;
}

impl Seal for PrivateKey {
    fn seal<C: Context + ?Sized>(
        &self,
        recipient: &PublicKey,
        access_keys: &[PublicKey],
        ctx: &C,
    ) -> Result<Capsule, Error> {
        Capsule::builder(self, recipient, ctx)
            .access_keys(access_keys)
            .seal()
    }
}

impl Contribute<VerifiedCapsule<'_>> for PrivateKey {
    type Output = Result<Partial, Error>;
    fn contribute(&self, target: &VerifiedCapsule<'_>) -> Self::Output {
        target.contribute(self)
    }
    fn contribute_for_gate(&self, target: &VerifiedCapsule<'_>, gate: &PublicKey) -> Self::Output {
        target.contribute_for_gate(self, gate)
    }
}

impl Unseal<VerifiedCapsule<'_>> for PrivateKey {
    type Output = Result<Self, Error>;
    fn unseal(&self, target: &VerifiedCapsule<'_>, partials: &[Partial]) -> Self::Output {
        target.unseal(self, partials)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::borrow::Cow;

    struct TestCtx;
    impl Context for TestCtx {
        fn domain(&self) -> &'static str {
            "ve-capsule.capsule-test"
        }
        fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
            Ok(Cow::Borrowed(b"capsule-api-binding"))
        }
    }

    /// The aggregate "key no one holds": Σ participants' public points, as a
    /// `PublicKey` — the gate shape the threshold tests seal against.
    fn aggregate_key(shares: &[&PrivateKey]) -> PublicKey {
        let sum = shares.iter().fold(ProjectivePoint::IDENTITY, |acc, s| {
            acc + s.public_key().point()
        });
        PublicKey::from_canonical_bytes(&encode_point(&sum)).unwrap()
    }

    /// A `PrivateKey` from a deterministic nonzero 32-byte secret.
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

    #[test]
    fn to_secret_bytes_is_the_inverse_of_from_secret() {
        // A canonical nonzero scalar < n round-trips byte-for-byte; this is the
        // storage/recovery boundary the recover path leans on.
        let secret = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
            0x1d, 0x1e, 0x1f, 0x20,
        ];
        let key = PrivateKey::from_secret(&secret).unwrap();
        assert_eq!(*key.to_secret_bytes(), secret);
        // And the exported bytes reconstruct the same public key.
        let reimported = PrivateKey::from_secret(&*key.to_secret_bytes()).unwrap();
        assert_eq!(reimported.public_key(), key.public_key());
    }

    #[test]
    fn mirror_verb_traits_match_inherent_calls() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_11_11);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let m = private_key(&mut rng);
        let (rpk, apk, mpk) = (recipient.public_key(), access.public_key(), m.public_key());

        // m.seal(..) ≡ Capsule::builder(&m, ..).seal()
        let capsule = m.seal(&rpk, &[apk], &TestCtx).unwrap();
        let vc = capsule.verify(&mpk, &rpk, &[apk], &TestCtx).unwrap();
        // access.contribute(&vc) ≡ vc.contribute(&access)
        let partial = access.contribute(&vc).unwrap();
        // recipient.unseal(&vc, ..) ≡ vc.unseal(&recipient, ..)
        let recovered = recipient.unseal(&vc, &[partial]).unwrap();
        assert_eq!(recovered.public_key(), mpk);
    }

    #[test]
    fn single_gate_round_trip_through_public_api() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_01_01);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let m = private_key(&mut rng);

        let capsule = Capsule::builder(&m, &recipient.public_key(), &TestCtx)
            .access_key(&access.public_key())
            .seal()
            .unwrap();

        let vc = capsule
            .verify(
                &m.public_key(),
                &recipient.public_key(),
                &[access.public_key()],
                &TestCtx,
            )
            .unwrap();

        let partial = vc.contribute(&access).unwrap();
        let recovered = vc.unseal(&recipient, &[partial]).unwrap();
        assert_eq!(
            recovered.public_key(),
            m.public_key(),
            "recovered scalar matches the sealed m"
        );
    }

    #[test]
    fn ungated_round_trip_through_public_api() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_01_02);
        let recipient = private_key(&mut rng);
        let m = private_key(&mut rng);
        let capsule = Capsule::builder(&m, &recipient.public_key(), &TestCtx)
            .seal()
            .unwrap();
        let vc = capsule
            .verify(&m.public_key(), &recipient.public_key(), &[], &TestCtx)
            .unwrap();
        let recovered = vc.unseal(&recipient, &[]).unwrap();
        assert_eq!(recovered.public_key(), m.public_key());
    }

    #[test]
    fn verify_ungated_accepts_ungated_and_rejects_gated() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_01_05);
        let recipient = private_key(&mut rng);
        let m = private_key(&mut rng);

        // An ungated capsule verifies through the convenience method.
        let ungated = Capsule::builder(&m, &recipient.public_key(), &TestCtx)
            .seal()
            .unwrap();
        assert!(
            ungated
                .verify_ungated(&m.public_key(), &recipient.public_key(), &TestCtx)
                .is_ok(),
            "verify_ungated must accept a capsule sealed with no gates"
        );

        // A gated capsule must NOT pass verify_ungated — it asserts no gates.
        let access = private_key(&mut rng);
        let gated = Capsule::builder(&m, &recipient.public_key(), &TestCtx)
            .access_key(&access.public_key())
            .seal()
            .unwrap();
        assert!(
            gated
                .verify_ungated(&m.public_key(), &recipient.public_key(), &TestCtx)
                .is_err(),
            "verify_ungated must reject a capsule that actually carries gates"
        );
    }

    #[test]
    fn verify_rejects_wrong_expected_pubkey() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_01_03);
        let recipient = private_key(&mut rng);
        let m = private_key(&mut rng);
        let other = private_key(&mut rng);
        let capsule = Capsule::builder(&m, &recipient.public_key(), &TestCtx)
            .seal()
            .unwrap();
        assert!(
            capsule
                .verify(&other.public_key(), &recipient.public_key(), &[], &TestCtx)
                .is_err(),
            "C == expected_pubkey gate must reject a mismatched secret"
        );
    }

    #[test]
    fn verify_rejects_wrong_access_set() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_01_04);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let wrong = private_key(&mut rng);
        let m = private_key(&mut rng);
        let capsule = Capsule::builder(&m, &recipient.public_key(), &TestCtx)
            .access_key(&access.public_key())
            .seal()
            .unwrap();
        assert!(
            capsule
                .verify(
                    &m.public_key(),
                    &recipient.public_key(),
                    &[wrong.public_key()],
                    &TestCtx,
                )
                .is_err(),
            "g* gate must reject a different access-key set"
        );
    }

    #[test]
    fn private_key_rejects_zero_and_bad_length() {
        assert!(PrivateKey::from_secret(&[0u8; 32]).is_err());
        assert!(PrivateKey::from_secret(&[1u8; 31]).is_err());
    }

    #[test]
    fn private_key_is_explicitly_zeroizable() {
        static_assertions::assert_impl_all!(
            PrivateKey: zeroize::Zeroize, zeroize::ZeroizeOnDrop
        );
        static_assertions::assert_not_impl_any!(PrivateKey: Clone, Copy, core::fmt::Debug);
    }

    /// Recipient-path latency through the public API: `verify` (the one
    /// expensive `π`) + token `unseal` (which skips the re-verify). Confirms the
    /// token halves the recipient cost vs a self-verifying open. Run with
    /// `cargo test -p ve-capsule --release -- --ignored --nocapture
    /// capsule::tests::token_recipient_latency`.
    #[test]
    #[ignore = "manual perf baseline; run with --release --ignored --nocapture"]
    fn token_recipient_latency() {
        use std::time::Instant;
        let mut rng = StdRng::seed_from_u64(0xCA_75_5E_ED);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let m = private_key(&mut rng);
        let rpk = recipient.public_key();
        let apk = access.public_key();
        let mpk = m.public_key();

        // Warm the lazy tables.
        {
            let cap = Capsule::builder(&m, &rpk, &TestCtx)
                .access_key(&apk)
                .seal()
                .unwrap();
            let vc = cap.verify(&mpk, &rpk, &[apk], &TestCtx).unwrap();
            let p = vc.contribute(&access).unwrap();
            vc.unseal(&recipient, &[p]).unwrap();
        }

        let capsule = Capsule::builder(&m, &rpk, &TestCtx)
            .access_key(&apk)
            .seal()
            .unwrap();
        let partial = capsule
            .verify(&mpk, &rpk, &[apk], &TestCtx)
            .unwrap()
            .contribute(&access)
            .unwrap();

        let t = Instant::now();
        let vc = capsule.verify(&mpk, &rpk, &[apk], &TestCtx).unwrap();
        let verify_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let recovered = vc.unseal(&recipient, &[partial]).unwrap();
        let unseal_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(recovered.public_key(), mpk);

        println!(
            "ve-capsule recipient path (local, token): verify={verify_ms:.1}ms \
             token-unseal={unseal_ms:.1}ms total={:.1}ms (unseal skips re-verify)",
            verify_ms + unseal_ms
        );
    }

    #[test]
    fn capsule_wire_round_trip_then_open() {
        // Seal → bytes → parse → verify → unseal: the I3 blob persistence path.
        let mut rng = StdRng::seed_from_u64(0xCA_75_0F_01);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let m = private_key(&mut rng);
        let (rpk, apk, mpk) = (recipient.public_key(), access.public_key(), m.public_key());

        let capsule = Capsule::builder(&m, &rpk, &TestCtx)
            .access_key(&apk)
            .seal()
            .unwrap();
        let bytes = capsule.to_canonical_bytes();
        assert_eq!(
            Capsule::from_canonical_bytes(&bytes)
                .unwrap()
                .to_canonical_bytes(),
            bytes,
            "re-encode equality"
        );

        let parsed = Capsule::from_canonical_bytes(&bytes).unwrap();
        let vc = parsed.verify(&mpk, &rpk, &[apk], &TestCtx).unwrap();
        let partial = vc.contribute(&access).unwrap();
        let recovered = vc.unseal(&recipient, &[partial]).unwrap();
        assert_eq!(recovered.public_key(), mpk, "opens after a wire round trip");
    }

    #[test]
    fn capsule_from_bytes_rejects_tamper_and_framing() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_0B_AD);
        let recipient = private_key(&mut rng);
        let m = private_key(&mut rng);
        let capsule = Capsule::builder(&m, &recipient.public_key(), &TestCtx)
            .seal()
            .unwrap();
        let bytes = capsule.to_canonical_bytes();
        // bad magic
        assert!(Capsule::from_canonical_bytes(b"not-a-capsule").is_err());
        // truncated
        assert!(Capsule::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());
        // trailing byte
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(Capsule::from_canonical_bytes(&trailing).is_err());
        // flipped byte inside the proof body (likely a non-canonical/invalid point)
        let mut flipped = bytes;
        let last = flipped.len() - 1;
        flipped[last] ^= 0x01;
        // either decode fails, or it parses but no longer re-encode-equals / verifies
        if let Ok(parsed) = Capsule::from_canonical_bytes(&flipped) {
            assert!(
                parsed
                    .verify(&m.public_key(), &recipient.public_key(), &[], &TestCtx)
                    .is_err()
            );
        }
    }

    #[test]
    fn public_key_round_trips_canonical_bytes() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_01_05);
        let pk = private_key(&mut rng).public_key();
        let bytes = pk.to_canonical_bytes();
        let pk2 = PublicKey::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(pk, pk2);
        assert!(PublicKey::from_canonical_bytes(&[0u8; POINT_LEN]).is_err());
    }

    // ── contribute: self-gate convenience vs explicit contribute_for_gate ──

    /// The short `contribute(key)` opens a single-holder gated capsule and tags
    /// the partial with the holder's own key (gate = `key.public_key()`), exactly
    /// as the explicit form with that gate.
    #[test]
    fn self_gate_contribute_opens_and_tags_own_key() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_C0_01);
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
        assert_eq!(
            partial.gate(),
            apk.point(),
            "self-gate tags the partial with key.public_key()"
        );
        assert_eq!(vc.unseal(&recipient, &[partial]).unwrap().public_key(), mpk);

        // Equivalent to the explicit form with gate = key.public_key().
        let explicit = vc.contribute_for_gate(&access, &apk).unwrap();
        assert_eq!(explicit.gate(), apk.point());
        assert_eq!(
            vc.unseal(&recipient, &[explicit]).unwrap().public_key(),
            mpk
        );
    }

    /// The short form rejects a key whose public half is not a listed gate,
    /// before building any partial: a misused threshold share fails fast.
    #[test]
    fn self_gate_contribute_rejects_unlisted_key() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_C0_02);
        let recipient = private_key(&mut rng);
        let access = private_key(&mut rng);
        let stranger = private_key(&mut rng);
        let m = private_key(&mut rng);
        let (rpk, apk, mpk) = (recipient.public_key(), access.public_key(), m.public_key());

        let capsule = Capsule::builder(&m, &rpk, &TestCtx)
            .access_key(&apk)
            .seal()
            .unwrap();
        let vc = capsule.verify(&mpk, &rpk, &[apk], &TestCtx).unwrap();

        assert!(
            matches!(vc.contribute(&stranger), Err(Error::DegenerateInput(_))),
            "self-gate must reject a key not in the access set"
        );
    }

    /// A "key no one holds": the gate is an aggregate `Y = (s1 + s2)·G`, and two
    /// participants each contribute their additive piece toward that one gate via
    /// `contribute_for_gate`. The recipient's per-gate `Σ W == Y` check sums the
    /// pieces and the capsule opens. This is the case the explicit gate exists for
    /// (`gate != key.public_key()`); one share alone leaves the bucket short.
    #[test]
    fn contribute_for_gate_opens_threshold_aggregate() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_C0_03);
        let recipient = private_key(&mut rng);
        let s1 = private_key(&mut rng);
        let s2 = private_key(&mut rng);
        let m = private_key(&mut rng);
        let (rpk, mpk) = (recipient.public_key(), m.public_key());

        let aggregate = aggregate_key(&[&s1, &s2]);

        let capsule = Capsule::builder(&m, &rpk, &TestCtx)
            .access_key(&aggregate)
            .seal()
            .unwrap();
        let vc = capsule.verify(&mpk, &rpk, &[aggregate], &TestCtx).unwrap();

        let p1 = vc.contribute_for_gate(&s1, &aggregate).unwrap();
        let p2 = vc.contribute_for_gate(&s2, &aggregate).unwrap();
        assert_eq!(p1.gate(), aggregate.point());
        assert_eq!(p2.gate(), aggregate.point());

        let recovered = vc.unseal(&recipient, &[p1, p2]).unwrap();
        assert_eq!(
            recovered.public_key(),
            mpk,
            "the two shares sum to the aggregate gate and open the capsule"
        );

        let only_one = vc.contribute_for_gate(&s1, &aggregate).unwrap();
        assert!(
            matches!(
                vc.unseal(&recipient, &[only_one]),
                Err(Error::Verification(msg)) if msg.contains("not qualifying")
            ),
            "one share alone leaves the aggregate bucket short: the per-gate \
             Σ W == Y_access check rejects it"
        );
    }

    /// Intent-trap guard: if a participant's OWN key is also a separately listed
    /// gate, the short form contributes to that own-key gate, not the aggregate.
    /// It is fail-closed for the aggregate (its bucket stays short), which is why
    /// threshold participants must use `contribute_for_gate`.
    #[test]
    fn self_gate_short_form_targets_own_key_not_aggregate() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_C0_04);
        let recipient = private_key(&mut rng);
        let s1 = private_key(&mut rng);
        let s2 = private_key(&mut rng);
        let m = private_key(&mut rng);
        let (rpk, mpk) = (recipient.public_key(), m.public_key());

        let aggregate = aggregate_key(&[&s1, &s2]);

        // The capsule lists BOTH the aggregate AND s1's own key as gates.
        let capsule = Capsule::builder(&m, &rpk, &TestCtx)
            .access_key(&aggregate)
            .access_key(&s1.public_key())
            .seal()
            .unwrap();
        let vc = capsule
            .verify(&mpk, &rpk, &[aggregate, s1.public_key()], &TestCtx)
            .unwrap();

        // s1 reaching for the short form tags its own-key gate, not the aggregate.
        let p = vc.contribute(&s1).unwrap();
        assert_eq!(
            p.gate(),
            s1.public_key().point(),
            "short form targets the own-key gate"
        );
        assert!(
            matches!(
                vc.unseal(&recipient, &[p]),
                Err(Error::Verification(msg)) if msg.contains("not qualifying")
            ),
            "the still-empty aggregate bucket fails the per-gate Σ W == Y_access check"
        );
    }
}
