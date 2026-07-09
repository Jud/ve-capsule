//! Local multi-scalar helpers over secp256k1 points.

use k256::elliptic_curve::ops::{
    LinearCombination as _, LinearCombinationExt as _, MulByGenerator as _,
};
use k256::{ProjectivePoint, Scalar};

/// Compute `Σ points[i] * scalars[i]`, using zip-shortest semantics.
pub fn msm(points: &[ProjectivePoint], scalars: &[Scalar]) -> ProjectivePoint {
    let len = points.len().min(scalars.len());
    match len {
        0 => ProjectivePoint::IDENTITY,
        1 => points[0] * scalars[0],
        2 => ProjectivePoint::lincomb(&points[0], &scalars[0], &points[1], &scalars[1]),
        _ => {
            let mut terms = Vec::with_capacity(len);
            for (&point, &scalar) in points.iter().zip(scalars.iter()) {
                if point != ProjectivePoint::IDENTITY && !bool::from(scalar.is_zero()) {
                    terms.push((point, scalar));
                }
            }
            ProjectivePoint::lincomb_ext(terms.as_slice())
        }
    }
}

/// Compute `Σ points[i] * scalars[i]` for public verifier data.
///
/// This is variable-time in the scalar windows and must not be used with secret
/// witness scalars. Verifier-side BP++ challenges and proof scalars are public,
/// so a bucket MSM avoids hundreds of independent variable-base multiplications.
pub fn msm_vartime_public(points: &[ProjectivePoint], scalars: &[Scalar]) -> ProjectivePoint {
    let len = points.len().min(scalars.len());
    if len < 16 {
        return msm(points, scalars);
    }

    let window_bits = if len >= 128 {
        6usize
    } else if len >= 32 {
        5usize
    } else {
        4usize
    };
    let window_count = 256usize.div_ceil(window_bits);
    let bucket_count = (1usize << window_bits) - 1;
    let mut terms = Vec::with_capacity(len);
    for (&point, scalar) in points.iter().zip(scalars.iter()) {
        if point != ProjectivePoint::IDENTITY && !bool::from(scalar.is_zero()) {
            terms.push((point, scalar.to_bytes()));
        }
    }
    if terms.is_empty() {
        return ProjectivePoint::IDENTITY;
    }

    let mut acc = ProjectivePoint::IDENTITY;
    let mut buckets = vec![ProjectivePoint::IDENTITY; bucket_count];
    for window in (0..window_count).rev() {
        if window != window_count - 1 {
            for _ in 0..window_bits {
                acc = acc.double();
            }
        }
        buckets.fill(ProjectivePoint::IDENTITY);
        let bit_offset = window * window_bits;
        for (point, scalar_bytes) in &terms {
            let digit = scalar_window(scalar_bytes.as_ref(), bit_offset, window_bits);
            if digit != 0 {
                buckets[digit - 1] += point;
            }
        }

        let mut running = ProjectivePoint::IDENTITY;
        for bucket in buckets.iter().rev() {
            running += bucket;
            acc += running;
        }
    }
    acc
}

fn scalar_window(bytes: &[u8; 32], bit_offset: usize, width: usize) -> usize {
    let mut out = 0usize;
    for bit in 0..width {
        let scalar_bit = bit_offset + bit;
        if scalar_bit >= 256 {
            break;
        }
        let byte = bytes[31 - scalar_bit / 8];
        let value = (byte >> (scalar_bit % 8)) & 1;
        out |= usize::from(value) << bit;
    }
    out
}

/// Compute `a * a_scalar + b * b_scalar`.
pub fn lincomb2(
    a: &ProjectivePoint,
    a_scalar: &Scalar,
    b: &ProjectivePoint,
    b_scalar: &Scalar,
) -> ProjectivePoint {
    ProjectivePoint::lincomb(a, a_scalar, b, b_scalar)
}

/// Compute `scalar * G` with k256's fixed-base backend.
pub fn generator_mul(scalar: &Scalar) -> ProjectivePoint {
    ProjectivePoint::mul_by_generator(scalar)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use k256::elliptic_curve::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn naive(points: &[ProjectivePoint], scalars: &[Scalar]) -> ProjectivePoint {
        points
            .iter()
            .zip(scalars.iter())
            .fold(ProjectivePoint::IDENTITY, |acc, (point, scalar)| {
                acc + *point * scalar
            })
    }

    #[test]
    fn matches_naive_msm() {
        let mut rng = StdRng::seed_from_u64(0x5155_7001);
        for len in [0usize, 1, 2, 3, 8, 32, 142, 256] {
            let scalars: Vec<_> = (0..len).map(|_| Scalar::random(&mut rng)).collect();
            let points: Vec<_> = scalars.iter().map(generator_mul).collect();
            assert_eq!(msm(&points, &scalars), naive(&points, &scalars));
        }
    }

    #[test]
    fn matches_naive_zip_shortest() {
        let mut rng = StdRng::seed_from_u64(0x5155_7002);
        let points: Vec<_> = (0..5)
            .map(|_| generator_mul(&Scalar::random(&mut rng)))
            .collect();
        let scalars: Vec<_> = (0..3).map(|_| Scalar::random(&mut rng)).collect();
        assert_eq!(msm(&points, &scalars), naive(&points, &scalars));
    }

    #[test]
    fn public_vartime_msm_matches_naive() {
        let mut rng = StdRng::seed_from_u64(0x5155_7003);
        for len in [16usize, 31, 32, 64, 142, 256] {
            let scalars: Vec<_> = (0..len).map(|_| Scalar::random(&mut rng)).collect();
            let points: Vec<_> = (0..len)
                .map(|_| generator_mul(&Scalar::random(&mut rng)))
                .collect();
            assert_eq!(
                msm_vartime_public(&points, &scalars),
                naive(&points, &scalars)
            );
        }
    }
}
