//! Caller-supplied, domain-separated context bound into the seal/verify
//! transcript.
//!
//! `Context` is **caller-owned**: it supplies the ceremony/session domain and a
//! deterministic binding payload. It does not provide cross-protocol separation
//! — that comes from the crate's own transcript domains. The payload MUST be
//! deterministic (a producer and verifier reconstruct byte-identical
//! transcripts); a snapshot is frozen at `verify` so the open cannot drift.

use std::borrow::Cow;

use crate::Error;

/// Domain-separated context the caller binds into a `seal` / `verify` / `unseal`
/// transcript (replay + ceremony binding).
pub trait Context {
    /// Non-empty, compile-time-pinned domain string. The crate rejects an empty
    /// domain at the seal/verify boundary.
    fn domain(&self) -> &'static str;

    /// Deterministic binding payload, canonically encoded so a producer and
    /// verifier reconstruct byte-identical transcripts.
    ///
    /// # Errors
    ///
    /// The caller's error if canonicalizing the payload fails.
    fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, Error>;
}
