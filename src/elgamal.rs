//! Per-limb curve-`ElGamal` encryption to the recovery public key.
//!
//! Each limb value `v ∈ [0, 2^ℓ)` is encrypted as the pair
//! `E = r·G`, `D = v·G + r·pk` under fresh randomness `r`, where `pk = sk·G`
//! is the recovery public key. The holder of `sk` recovers the limb point via
//! `D − sk·E = v·G` and then runs BSGS ([`crate::bsgs`]) to extract `v`.
//!
//! Two degenerate inputs are rejected because they void secrecy: an identity
//! recovery key (`sk = 0`, so `D = v·G` is a public plaintext, audit
//! SA-2026-324) and an identity mask (`r = 0`, so `E = identity` and again
//! `D = v·G`). The encryptor samples a nonzero `r`; a decryptor rejects any
//! ciphertext whose `E` is the identity.

use crate::codec::encode_point;
use crate::error::Error;
use k256::elliptic_curve::Field;
use k256::{ProjectivePoint, Scalar};
use rand_core::{CryptoRng, RngCore};

/// Detail string for the identity-mask gate, shared by every stage that
/// enforces it (wire decode / seal / verify / decrypt) so the sites cannot
/// drift.
pub const IDENTITY_MASK_DETAIL: &str = "ElGamal mask E is the identity";

/// A per-limb curve-`ElGamal` ciphertext.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LimbCiphertext {
    /// The mask commitment `E = r·G`.
    pub e: ProjectivePoint,
    /// The masked limb point `D = v·G + r·pk`.
    pub d: ProjectivePoint,
}

/// Sample a uniform nonzero scalar. `r = 0` would make `E` the identity and
/// leak the limb, so the (negligibly likely) zero draw is rejected.
fn random_nonzero_scalar<R: RngCore + CryptoRng>(rng: &mut R) -> Scalar {
    loop {
        let r = Scalar::random(&mut *rng);
        if !bool::from(r.is_zero()) {
            return r;
        }
    }
}

impl LimbCiphertext {
    /// Encrypt a limb value `v` to `pk`, returning the ciphertext and the
    /// encryption randomness `r` (the prover's witness for the linking proof).
    ///
    /// # Errors
    ///
    /// Returns [`Error::DegenerateInput`] if `pk` is the identity (`sk = 0`),
    /// which would make the ciphertext a public plaintext.
    pub fn encrypt<R: RngCore + CryptoRng>(
        limb: u32,
        pk: &ProjectivePoint,
        rng: &mut R,
    ) -> Result<(Self, Scalar), Error> {
        if pk == &ProjectivePoint::IDENTITY {
            return Err(Error::DegenerateInput(
                "recovery public key is the identity (sk=0)",
            ));
        }
        let r = random_nonzero_scalar(rng);
        let v = Scalar::from(u64::from(limb));
        let e = ProjectivePoint::GENERATOR * r;
        let d = ProjectivePoint::GENERATOR * v + *pk * r;
        Ok((Self { e, d }, r))
    }

    /// Decrypt to the limb point `v·G = D − sk·E`. The caller runs BSGS on the
    /// result to recover the integer `v`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DegenerateInput`] if `E` is the identity (`r = 0`), an
    /// unmasked ciphertext that must never be accepted.
    ///
    /// Single-secret decrypt, used by [`crate::assembly::open`] and its tests;
    /// the production recipient strips masks in the [`crate::opening`] layer.
    #[allow(dead_code)]
    pub fn decrypt_point(&self, sk: &Scalar) -> Result<ProjectivePoint, Error> {
        if self.e == ProjectivePoint::IDENTITY {
            return Err(Error::DegenerateInput(IDENTITY_MASK_DETAIL));
        }
        Ok(self.d - self.e * sk)
    }
}

/// Append one limb ciphertext `(E_k, D_k)` to `out` as two canonical 33-byte
/// SEC1 points, `E_k` then `D_k`. The single limb encoder shared by the full
/// proof ([`crate::assembly::Proof`]) and the stripped core
/// ([`crate::stripped::StrippedCapsule`]), so the two wire layouts cannot drift.
pub fn encode_limb(ct: &LimbCiphertext, out: &mut Vec<u8>) {
    out.extend_from_slice(&encode_point(&ct.e));
    out.extend_from_slice(&encode_point(&ct.d));
}

/// The soundness-critical identity-mask gate: reject a segment mask `E_k = O`
/// (`r_k = 0`), which makes `D_k = v_k·G` a public plaintext limb (soundness-doc
/// §4.4). The single gate shared by every limb decoder (full proof and stripped
/// core), applied after the strict [`crate::codec::decode_point`] and before the
/// limb's `D_k` is read, so it takes precedence over a later truncation.
///
/// # Errors
///
/// [`Error::DegenerateInput`] if `e` is the identity.
pub fn reject_identity_mask(e: &ProjectivePoint) -> Result<(), Error> {
    if e == &ProjectivePoint::IDENTITY {
        Err(Error::DegenerateInput(IDENTITY_MASK_DETAIL))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::bsgs::BabyTable;
    use crate::params::Params;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const PARAMS: Params = Params::FROZEN;

    fn keypair(rng: &mut StdRng) -> (Scalar, ProjectivePoint) {
        let sk = random_nonzero_scalar(rng);
        (sk, ProjectivePoint::GENERATOR * sk)
    }

    #[test]
    fn roundtrip_recovers_limb() {
        let table = BabyTable::new();
        let mut rng = StdRng::seed_from_u64(0xE6_C1_2A_03);
        let (sk, pk) = keypair(&mut rng);
        let max = u32::try_from(PARAMS.limb_modulus() - 1).unwrap();
        for limb in [0u32, 1, 2, max, 0x00AB_CDEF & max] {
            let (ct, _r) = LimbCiphertext::encrypt(limb, &pk, &mut rng).unwrap();
            let point = ct.decrypt_point(&sk).unwrap();
            assert_eq!(table.recover(&point), Some(limb), "limb={limb}");
        }
    }

    #[test]
    fn roundtrip_random_limbs() {
        let table = BabyTable::new();
        let mut rng = StdRng::seed_from_u64(0xE6_C1_2A_04);
        let (sk, pk) = keypair(&mut rng);
        for _ in 0..64 {
            let limb = u32::try_from(rng.next_u64() % PARAMS.limb_modulus()).unwrap();
            let (ct, _r) = LimbCiphertext::encrypt(limb, &pk, &mut rng).unwrap();
            assert_eq!(table.recover(&ct.decrypt_point(&sk).unwrap()), Some(limb));
        }
    }

    #[test]
    fn fresh_randomness_per_encryption() {
        // Two encryptions of the same limb must differ (semantic security):
        // distinct r ⇒ distinct E.
        let mut rng = StdRng::seed_from_u64(0xE6_C1_2A_05);
        let (_sk, pk) = keypair(&mut rng);
        let (a, _) = LimbCiphertext::encrypt(7, &pk, &mut rng).unwrap();
        let (b, _) = LimbCiphertext::encrypt(7, &pk, &mut rng).unwrap();
        assert_ne!(a.e, b.e);
        assert_ne!(a.d, b.d);
    }

    #[test]
    fn rejects_identity_public_key() {
        let mut rng = StdRng::seed_from_u64(0xE6_C1_2A_06);
        let err = LimbCiphertext::encrypt(1, &ProjectivePoint::IDENTITY, &mut rng);
        assert!(matches!(err, Err(Error::DegenerateInput(_))));
    }

    #[test]
    fn rejects_identity_mask_on_decrypt() {
        let mut rng = StdRng::seed_from_u64(0xE6_C1_2A_07);
        let (sk, pk) = keypair(&mut rng);
        let (ct, _) = LimbCiphertext::encrypt(3, &pk, &mut rng).unwrap();
        let forged = LimbCiphertext {
            e: ProjectivePoint::IDENTITY,
            d: ct.d,
        };
        assert!(matches!(
            forged.decrypt_point(&sk),
            Err(Error::DegenerateInput(_))
        ));
    }
}
