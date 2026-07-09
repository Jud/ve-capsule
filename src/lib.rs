#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! `ve-capsule` — opinionated secp256k1-only verifiable encryption.
//!
//! Encrypt one secp256k1 scalar to a **fixed recipient**, optionally gated behind
//! a qualifying quorum of **access keys** — entirely on the native curve. Because
//! nothing detours through a second curve, a class group, or an RSA modulus, it
//! reuses the secp256k1 key types and signers your stack already runs — a gated
//! recovery is authorized by an ordinary secp256k1 quorum signature. The soundness
//! and Fiat–Shamir absorption contract lives in
//! `docs/design/ec-segve-soundness.md` in the sudo monorepo and ships as
//! `docs/ec-segve-soundness.md` in standalone release snapshots.
//!
//! The public API is blob-in / blob-out: [`Capsule::builder`] seals;
//! [`Capsule::verify`] confirms a capsule against an authorization and yields a
//! [`VerifiedCapsule`] capability token; an authorizer
//! [`VerifiedCapsule::contribute`]s a [`Partial`] and the recipient
//! [`VerifiedCapsule::unseal`]s. [`Case`] verifies and opens an additively-split
//! secret. The proof algebra is intentionally crate-private so callers cannot
//! bypass the consent-gated surface.
//!
//! For storage, recovery does not need the proof. After verification,
//! [`VerifiedCase::strip`] drops it to a compact opening core. A compact
//! core opens recipient-only and **self-securing** (recovery re-anchors on the
//! certified commitment `M`, so a tampered core fails closed), or a quorum
//! [`Signature`] stands in for the proof via `verify_signed` — BIP-340 Schnorr
//! today, an open [`Scheme`] set, with room for others. The signature verifies
//! under the caller-supplied x-only key; Taproot key-path FROST signatures use
//! the tweaked output key. The signature gates the `contribute` surface: a
//! stripped core has no proof, so it authenticates the core before any
//! authorizer contributes, lest an authorizer be duped into partial-decrypting
//! an attacker-fabricated capsule (a static-DH oracle the seal proof otherwise
//! closes). Recipient-only recovery contributes nothing and needs no signature.
//!
//! Recipient and access public keys are untrusted inputs. Gated capsules use
//! deterministic per-key aggregation coefficients and reject publicly enumerable
//! key components, but integrations should still possession-certify and
//! enrollment-bind the keys before presenting them as authorization material.

mod assembly;
mod batch_affine;
mod bsgs;
mod context;
// The public capsule API (Capsule / VerifiedCapsule / PrivateKey / PublicKey).
mod capsule;
// The Case: the verifiable opening of one additively-split secret.
mod carry;
mod case;
mod codec;
// The coefficiented composite-key seal (Y*) + gate commitment g*.
// Public capsule/case APIs call into this module; tests also exercise internals.
#[allow(dead_code)]
mod composite;
// The batched DLEQ that each `contribute` Partial carries.
mod dleq;
mod elgamal;
pub mod error;
mod generators;
// The recovery unseal hint: a compact whole-scalar contribution shadow openable
// recipient-only (one ECDH + hash + subtract) for compact recovery.
mod hint;
mod limbs;
mod linking;
mod msm;
// The BP++ weighted norm linear argument (compression core).
mod norm_arg;
// The aggregated BP++ reciprocal range proof at the frozen capsule shape.
mod range_circuit;
// The opening layer: authorizer `contribute` → non-destructive `Partial`.
// `unseal` (the self-verifying wrapper) is used only by tests; production opens
// through a verified-capsule token via `unseal_verified` — hence `allow(dead_code)`.
#[allow(dead_code)]
mod opening;
mod parallel;
pub mod params;
mod pedersen;
// The blob-in/blob-out provisioning seam: seal pieces + assemble a recovery payload.
mod provision;
// The blob-in/blob-out recovery seam: open a secret from a recovery payload.
mod recover;
// First-class signature backing: BIP-340 verify + the canonical attestation statement.
mod signature;
// The canonical compact-recovery-payload wire schema.
mod compact_payload;
// The proof-stripped opening core for compact recovery storage.
mod stripped;
mod transcript;

pub use capsule::{
    Capsule, CapsuleBuilder, Contribute, PrivateKey, PublicKey, Seal, Unseal, VerifiedCapsule,
};
pub use case::{Case, VerifiedCase};
pub use compact_payload::CompactRecoveryPayload;
pub use context::Context;
pub use error::Error;
pub use hint::{
    AuthorizerContribution, GateQuorum, GatedBinding, GatedPiece, HINT_LEN, PinnedHintVerifier,
    RecoveryHint, hint_attestation_message, public_gate_sum,
};
pub use opening::Partial;
pub use params::Params;
pub use provision::{
    assemble_recipient_recovery_payload, seal_recovery_hint, validate_recovery_hint,
    validate_recovery_hints_against_capsules,
};
pub use recover::{RecoveryContext, recover_recipient_secret_from_payload};
pub use signature::{Backing, Scheme, Signature};
pub use stripped::{BoundCapsule, BoundCase, StrippedCapsule, StrippedCase};
