//! Strict, identity-capable SEC1 codec for secp256k1 points.
//!
//! ec-segve publishes identity-capable commitments (a zero limb's `P_k`, a
//! zero piece's `C_i`), so the codec MUST round-trip the identity point —
//! but with EXACTLY ONE canonical encoding. Wire bytes are NEVER routed
//! through `generic_ec::Point::from_bytes`: its 0.4.5 decoder collapses any
//! all-zero buffer of any length to the identity, giving one point multiple
//! wire representations and thus distinct Fiat–Shamir challenges for the same
//! point (audit SA-2026-333). Here, length and tag are checked BEFORE any
//! decode, and the identity has the single canonical form of 33 zero bytes.
//!
//! The canonical encodings are:
//! - **identity** → 33 zero bytes (`[0u8; 33]`), and nothing else decodes to
//!   the identity;
//! - **every other point** → 33-byte SEC1 *compressed* (`0x02`/`0x03` tag).
//!
//! Decoding additionally enforces on-curve membership and `x < p` (via
//! [`AffinePoint::from_encoded_point`], which fails closed on either), so a
//! valid-tag buffer with an off-curve or out-of-field `x` is rejected.

use crate::error::Error;
use k256::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use k256::{AffinePoint, EncodedPoint, ProjectivePoint};

/// Byte length of every canonical encoding: 33-byte SEC1 compressed, and the
/// 33-zero-byte identity.
pub const POINT_LEN: usize = 33;

/// The single canonical identity encoding: 33 zero bytes.
const IDENTITY_ENCODING: [u8; POINT_LEN] = [0u8; POINT_LEN];

/// Encode a point to its canonical 33-byte form.
///
/// The identity maps to 33 zero bytes; every other point to SEC1 compressed.
#[must_use]
pub fn encode_point(point: &ProjectivePoint) -> [u8; POINT_LEN] {
    if point == &ProjectivePoint::IDENTITY {
        return IDENTITY_ENCODING;
    }
    let encoded = point.to_affine().to_encoded_point(true);
    let mut out = [0u8; POINT_LEN];
    out.copy_from_slice(encoded.as_bytes());
    out
}

/// Encode an **already-affine** point to its canonical 33-byte form, with no
/// field inversion. Identical output to [`encode_point`] (the identity maps to
/// 33 zero bytes), but takes an [`AffinePoint`] so a caller that has
/// batch-normalized many points pays one inversion for the whole batch instead
/// of one per point — the BSGS giant-walk hot path.
#[must_use]
pub fn encode_affine_point(point: &AffinePoint) -> [u8; POINT_LEN] {
    let encoded = point.to_encoded_point(true);
    if encoded.is_identity() {
        return IDENTITY_ENCODING;
    }
    let mut out = [0u8; POINT_LEN];
    out.copy_from_slice(encoded.as_bytes());
    out
}

/// Decode a point from its canonical 33-byte form.
///
/// Rejects any input that is not exactly 33 bytes, any all-zero buffer of a
/// length other than 33, a non-`0x02`/`0x03` tag on a non-identity point, and
/// any off-curve or `x >= p` point. The identity is accepted only as 33 zero
/// bytes.
///
/// # Errors
///
/// Returns [`Error::PointDecode`] on any non-canonical or invalid encoding.
pub fn decode_point(bytes: &[u8]) -> Result<ProjectivePoint, Error> {
    if bytes.len() != POINT_LEN {
        return Err(Error::PointDecode("SEC1 point must be exactly 33 bytes"));
    }
    if bytes.iter().all(|&b| b == 0) {
        return Ok(ProjectivePoint::IDENTITY);
    }
    match bytes.first() {
        Some(0x02 | 0x03) => {}
        _ => {
            return Err(Error::PointDecode(
                "non-canonical SEC1 tag: identity is 33 zero bytes, points are compressed 0x02/0x03",
            ));
        }
    }
    let encoded = EncodedPoint::from_bytes(bytes)
        .map_err(|_| Error::PointDecode("malformed SEC1 compressed encoding"))?;
    let affine: Option<AffinePoint> = AffinePoint::from_encoded_point(&encoded).into();
    let affine = affine.ok_or(Error::PointDecode(
        "point is not on the secp256k1 curve or x >= p",
    ))?;
    let point = ProjectivePoint::from(affine);
    if point == ProjectivePoint::IDENTITY {
        return Err(Error::PointDecode(
            "identity must use the canonical 33-zero-byte encoding",
        ));
    }
    Ok(point)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;

    fn sample_point() -> ProjectivePoint {
        // A deterministic non-identity, non-generator point.
        ProjectivePoint::GENERATOR + ProjectivePoint::GENERATOR
    }

    #[test]
    fn roundtrip_generator() {
        let g = ProjectivePoint::GENERATOR;
        let enc = encode_point(&g);
        assert_eq!(enc.len(), POINT_LEN);
        assert!(enc[0] == 0x02 || enc[0] == 0x03);
        assert_eq!(decode_point(&enc).unwrap(), g);
    }

    #[test]
    fn roundtrip_arbitrary_point() {
        let p = sample_point();
        assert_eq!(decode_point(&encode_point(&p)).unwrap(), p);
    }

    #[test]
    fn roundtrip_identity() {
        let id = ProjectivePoint::IDENTITY;
        let enc = encode_point(&id);
        assert_eq!(enc, [0u8; POINT_LEN]);
        assert_eq!(decode_point(&enc).unwrap(), id);
    }

    #[test]
    fn only_canonical_identity_decodes_to_identity() {
        // 33 zero bytes is the ONLY identity encoding (SA-2026-333 defense):
        assert_eq!(decode_point(&[0u8; 33]).unwrap(), ProjectivePoint::IDENTITY);
        // all-zero buffers of any other length are rejected, not collapsed.
        for len in [0usize, 1, 32, 34, 65] {
            assert!(decode_point(&vec![0u8; len]).is_err());
        }
    }

    #[test]
    fn rejects_wrong_length() {
        let g = encode_point(&ProjectivePoint::GENERATOR);
        assert!(decode_point(&g[..32]).is_err());
        let mut long = g.to_vec();
        long.push(0);
        assert!(decode_point(&long).is_err());
    }

    #[test]
    fn rejects_non_compressed_tag() {
        let g = encode_point(&ProjectivePoint::GENERATOR);
        for tag in [0x00u8, 0x01, 0x04, 0x05, 0x06, 0x07, 0xFF] {
            let mut buf = g;
            buf[0] = tag;
            assert!(decode_point(&buf).is_err(), "tag {tag:#x} must reject");
        }
    }

    #[test]
    fn rejects_off_curve_and_out_of_field() {
        // tag 0x02 with x = 0: not an on-curve x.
        let mut zero_x = [0u8; 33];
        zero_x[0] = 0x02;
        assert!(decode_point(&zero_x).is_err());
        // tag 0x02 with x = all-0xFF: x >= field modulus p.
        let mut big_x = [0xFFu8; 33];
        big_x[0] = 0x02;
        assert!(decode_point(&big_x).is_err());
    }
}
