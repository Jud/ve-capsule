//! Error type for ec-segve-v1.

use core::fmt;

/// Errors produced by the ec-segve secp256k1 primitives.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A SEC1 point could not be decoded: wrong length, non-canonical tag,
    /// off-curve or out-of-field `x`, or a non-canonical identity encoding.
    PointDecode(&'static str),
    /// A degenerate input that would void a security property was supplied:
    /// e.g. an identity recovery key (`sk = 0`) or an identity `ElGamal` mask
    /// (`r = 0`), either of which exposes the encrypted limb.
    DegenerateInput(&'static str),
    /// A non-interactive proof failed to verify: a structural shape mismatch,
    /// a failed algebraic check (recomposition, ring sum, or branch equation),
    /// or a challenge the responses were not built for.
    Verification(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PointDecode(detail) => write!(f, "SEC1 point decode failed: {detail}"),
            Self::DegenerateInput(detail) => write!(f, "degenerate input rejected: {detail}"),
            Self::Verification(detail) => write!(f, "proof verification failed: {detail}"),
        }
    }
}

impl std::error::Error for Error {}
