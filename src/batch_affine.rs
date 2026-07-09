//! Batched affine point addition for the cold screening scans.
//!
//! The opening-binding key screens compute hundreds of thousands of point
//! sums whose only consumer is an x-coordinate membership probe. The generic
//! projective pipeline (complete mixed addition plus batch normalization)
//! spends ~17 field multiplications per sum; the affine chord rule with one
//! Montgomery batch inversion per set spends ~5 and never touches the
//! y-coordinate of the result. That difference is what keeps a cold scan
//! inside the interactive-latency budget on constrained single cores, where the
//! scoped-thread fan-out in `composite` buys little.
//!
//! The kernel is total over valid inputs: chord (`x₁ ≠ x₂`), tangent
//! (`Q = P`, doubling), and cancellation (`Q = −P`, identity sum) are all
//! handled in field arithmetic, so callers never route a partial case back
//! through projective code. Outputs are differentially pinned against
//! [`ProjectivePoint`] addition in the tests.

use k256::elliptic_curve::point::{AffineCoordinates, BatchNormalize};
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{AffinePoint, FieldElement, ProjectivePoint};

#[derive(Clone, Copy)]
enum BatchAddKind {
    Chord,
    Tangent,
    Identity,
}

#[derive(Clone, Copy)]
enum BatchSignedStrideKind {
    Chord,
    FirstIdentitySecondTangent,
    FirstTangentSecondIdentity,
}

/// Reusable scratch for [`batch_add_x_keys_visit`]. The cold relation scans call
/// the affine chord kernel once per giant row; reusing these buffers avoids
/// allocating kind, denominator, inverse, and output vectors for every row.
#[derive(Default)]
pub struct BatchAddScratch {
    kinds: Vec<BatchAddKind>,
    signed_stride_kinds: Vec<BatchSignedStrideKind>,
    denominators: Vec<FieldElement>,
    prefix_products: Vec<FieldElement>,
    inverses: Vec<FieldElement>,
}

impl BatchAddScratch {
    /// Build scratch sized for a batch of `capacity` additions.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            kinds: Vec::with_capacity(capacity),
            signed_stride_kinds: Vec::with_capacity(capacity),
            denominators: Vec::with_capacity(capacity),
            prefix_products: Vec::with_capacity(capacity),
            inverses: Vec::with_capacity(capacity),
        }
    }
}

/// An affine secp256k1 point with directly usable field coordinates, fully
/// normalized (magnitude 1, canonical). Never the identity: construction
/// fails on it, so the chord/tangent formulas below are total.
#[derive(Clone, Copy)]
pub struct FePoint {
    x: FieldElement,
    y: FieldElement,
}

impl FePoint {
    /// Convert a non-identity affine point; `None` for the identity (which
    /// has no affine coordinates). The coordinate bytes of a normalized
    /// point are always canonical, so conversion never fails for a real
    /// point.
    pub fn from_affine(point: &AffinePoint) -> Option<Self> {
        let encoded = point.to_encoded_point(false);
        let x = encoded
            .x()
            .and_then(|bytes| Option::<FieldElement>::from(FieldElement::from_bytes(bytes)))?;
        let y = encoded
            .y()
            .and_then(|bytes| Option::<FieldElement>::from(FieldElement::from_bytes(bytes)))?;
        Some(Self { x, y })
    }

    /// Convert a non-identity projective point (one field inversion).
    pub fn from_projective(point: &ProjectivePoint) -> Option<Self> {
        Self::from_affine(&point.to_affine())
    }

    /// The point's negation `(x, −y)`.
    pub fn negated(&self) -> Self {
        Self {
            x: self.x,
            y: self.y.negate(1).normalize(),
        }
    }

    /// The x-coordinate membership key.
    pub fn x_key(&self) -> [u8; 32] {
        self.x.to_bytes().into()
    }

    /// The crate's canonical compressed point encoding.
    pub fn encoded_point(&self) -> [u8; 33] {
        let mut encoded = [0u8; 33];
        encoded[0] = 0x02 + self.y.normalize().is_odd().unwrap_u8();
        encoded[1..].copy_from_slice(&self.x.to_bytes());
        encoded
    }
}

/// The x-key of each point, `None` for the identity (which has no affine x).
/// One Montgomery batch inversion serves the whole slice, replacing a
/// per-point `from_projective` field inversion when many points share a scan.
pub fn batch_x_keys(points: &[ProjectivePoint]) -> Vec<Option<[u8; 32]>> {
    ProjectivePoint::batch_normalize(points)
        .iter()
        .zip(points)
        .map(|(affine, point)| (point != &ProjectivePoint::IDENTITY).then(|| affine.x().into()))
        .collect()
}

/// x-keys of `lhs + rhs` for every pair, or `None` where the sum is the
/// identity. One Montgomery batch inversion serves the whole slice; the
/// y-coordinate of the sums is never computed.
pub fn batch_add_x_keys(pairs: &[(FePoint, FePoint)]) -> Vec<Option<[u8; 32]>> {
    let mut out = Vec::with_capacity(pairs.len());
    let mut scratch = BatchAddScratch::with_capacity(pairs.len());
    batch_add_x_keys_visit(pairs, &mut scratch, |key| {
        out.push(key);
        false
    });
    out
}

/// Visit the x-key of `lhs + rhs` for every pair, returning `true` as soon as
/// `visit` does. This is the allocation-free hot form for relation scanners that
/// only need to know whether any row hit exists.
pub fn batch_add_x_keys_visit(
    pairs: &[(FePoint, FePoint)],
    scratch: &mut BatchAddScratch,
    mut visit: impl FnMut(Option<[u8; 32]>) -> bool,
) -> bool {
    scratch.kinds.clear();
    scratch.denominators.clear();
    for (lhs, rhs) in pairs {
        let dx = rhs.x - lhs.x;
        if !bool::from(dx.normalizes_to_zero()) {
            scratch.kinds.push(BatchAddKind::Chord);
            scratch.denominators.push(dx);
        } else if bool::from((lhs.y + rhs.y).normalizes_to_zero()) {
            // rhs = −lhs: the sum is the identity. The placeholder keeps the
            // batch-inversion product non-zero.
            scratch.kinds.push(BatchAddKind::Identity);
            scratch.denominators.push(FieldElement::ONE);
        } else {
            // rhs = lhs: tangent rule; 2y ≠ 0 because the curve has no
            // two-torsion.
            scratch.kinds.push(BatchAddKind::Tangent);
            scratch.denominators.push(lhs.y.double());
        }
    }
    batch_invert_into(
        &scratch.denominators,
        &mut scratch.prefix_products,
        &mut scratch.inverses,
    );

    for (((lhs, rhs), kind), inverse) in pairs
        .iter()
        .zip(scratch.kinds.iter())
        .zip(scratch.inverses.iter())
    {
        let key = match kind {
            BatchAddKind::Identity => None,
            BatchAddKind::Chord => {
                let lambda = (rhs.y - lhs.y).mul(inverse);
                let x_sum = lambda.square() - lhs.x - rhs.x;
                Some(x_sum.to_bytes().into())
            }
            BatchAddKind::Tangent => {
                let lambda = lhs.x.square().mul_single(3).mul(inverse);
                // Subtract x twice rather than subtracting double(): `Sub`
                // negates its rhs at magnitude 1, and double() is magnitude 2.
                let x_sum = lambda.square() - lhs.x - lhs.x;
                Some(x_sum.to_bytes().into())
            }
        };
        if visit(key) {
            return true;
        }
    }
    false
}

/// Visit `direction - stride` and `direction + stride` x-keys for each
/// direction, where `negated_stride == -stride`. The two signed rows share the
/// same affine denominator, so one batch inversion yields both probes.
pub fn batch_add_signed_stride_x_keys_visit(
    directions: &[FePoint],
    negated_stride: FePoint,
    scratch: &mut BatchAddScratch,
    mut visit: impl FnMut(Option<[u8; 32]>, Option<[u8; 32]>) -> bool,
) -> bool {
    scratch.signed_stride_kinds.clear();
    scratch.denominators.clear();
    for lhs in directions {
        let dx = negated_stride.x - lhs.x;
        if !bool::from(dx.normalizes_to_zero()) {
            scratch
                .signed_stride_kinds
                .push(BatchSignedStrideKind::Chord);
            scratch.denominators.push(dx);
        } else if bool::from((lhs.y + negated_stride.y).normalizes_to_zero()) {
            scratch
                .signed_stride_kinds
                .push(BatchSignedStrideKind::FirstIdentitySecondTangent);
            scratch.denominators.push(lhs.y.double());
        } else {
            scratch
                .signed_stride_kinds
                .push(BatchSignedStrideKind::FirstTangentSecondIdentity);
            scratch.denominators.push(lhs.y.double());
        }
    }
    batch_invert_into(
        &scratch.denominators,
        &mut scratch.prefix_products,
        &mut scratch.inverses,
    );

    let positive_stride = negated_stride.negated();
    for ((lhs, kind), inverse) in directions
        .iter()
        .zip(scratch.signed_stride_kinds.iter())
        .zip(scratch.inverses.iter())
    {
        let keys = match kind {
            BatchSignedStrideKind::Chord => {
                let lambda_minus = (negated_stride.y - lhs.y).mul(inverse);
                let x_minus = lambda_minus.square() - lhs.x - negated_stride.x;
                let lambda_plus = (positive_stride.y - lhs.y).mul(inverse);
                let x_plus = lambda_plus.square() - lhs.x - positive_stride.x;
                (
                    Some(x_minus.to_bytes().into()),
                    Some(x_plus.to_bytes().into()),
                )
            }
            BatchSignedStrideKind::FirstIdentitySecondTangent => {
                (None, Some(tangent_x_key(lhs, inverse)))
            }
            BatchSignedStrideKind::FirstTangentSecondIdentity => {
                (Some(tangent_x_key(lhs, inverse)), None)
            }
        };
        if visit(keys.0, keys.1) {
            return true;
        }
    }
    false
}

fn tangent_x_key(point: &FePoint, inverse: &FieldElement) -> [u8; 32] {
    let lambda = point.x.square().mul_single(3).mul(inverse);
    let x_sum = lambda.square() - point.x - point.x;
    x_sum.to_bytes().into()
}

/// Visit the full affine sum `lhs + rhs` for every pair, returning `true` as
/// soon as `visit` does. Relation scans use this to generate candidate
/// combinations directly in affine form, avoiding a detour through projective
/// additions followed by batch normalization.
pub fn batch_add_points_visit(
    pairs: &[(FePoint, FePoint)],
    scratch: &mut BatchAddScratch,
    mut visit: impl FnMut(Option<FePoint>) -> bool,
) -> bool {
    scratch.kinds.clear();
    scratch.denominators.clear();
    for (lhs, rhs) in pairs {
        let dx = rhs.x - lhs.x;
        if !bool::from(dx.normalizes_to_zero()) {
            scratch.kinds.push(BatchAddKind::Chord);
            scratch.denominators.push(dx);
        } else if bool::from((lhs.y + rhs.y).normalizes_to_zero()) {
            scratch.kinds.push(BatchAddKind::Identity);
            scratch.denominators.push(FieldElement::ONE);
        } else {
            scratch.kinds.push(BatchAddKind::Tangent);
            scratch.denominators.push(lhs.y.double());
        }
    }
    batch_invert_into(
        &scratch.denominators,
        &mut scratch.prefix_products,
        &mut scratch.inverses,
    );

    for (((lhs, rhs), kind), inverse) in pairs
        .iter()
        .zip(scratch.kinds.iter())
        .zip(scratch.inverses.iter())
    {
        let point = match kind {
            BatchAddKind::Identity => None,
            BatchAddKind::Chord => {
                let lambda = (rhs.y - lhs.y).mul(inverse);
                let x_sum = (lambda.square() - lhs.x - rhs.x).normalize();
                let y_sum = lambda.mul(&(lhs.x - x_sum)) - lhs.y;
                Some(FePoint {
                    x: x_sum,
                    y: y_sum.normalize(),
                })
            }
            BatchAddKind::Tangent => {
                let lambda = lhs.x.square().mul_single(3).mul(inverse);
                let x_sum = (lambda.square() - lhs.x - lhs.x).normalize();
                let y_sum = lambda.mul(&(lhs.x - x_sum)) - lhs.y;
                Some(FePoint {
                    x: x_sum,
                    y: y_sum.normalize(),
                })
            }
        };
        if visit(point) {
            return true;
        }
    }
    false
}

/// Montgomery batch inversion: three multiplications per element plus one
/// shared field inversion. Inputs must be non-zero — the kernel's
/// denominators are non-zero by construction (chord `dx ≠ 0`, tangent
/// `2y ≠ 0`, identity placeholder `1`).
#[cfg(test)]
fn batch_invert(elements: &[FieldElement]) -> Vec<FieldElement> {
    let mut prefix_products = Vec::new();
    let mut inverses = Vec::new();
    batch_invert_into(elements, &mut prefix_products, &mut inverses);
    inverses
}

fn batch_invert_into(
    elements: &[FieldElement],
    prefix_products: &mut Vec<FieldElement>,
    inverses: &mut Vec<FieldElement>,
) {
    prefix_products.clear();
    prefix_products.reserve(elements.len());
    let mut product = FieldElement::ONE;
    for element in elements {
        product = product.mul(element);
        prefix_products.push(product);
    }
    let inverted = product.invert();
    // The product of non-zero field elements is non-zero; a failure here
    // means a zero denominator reached the chord/tangent rule, and the scan
    // must not continue on corrupted math.
    assert!(
        bool::from(inverted.is_some()),
        "zero denominator in batch inversion"
    );
    let mut running = inverted.unwrap_or(FieldElement::ONE);
    inverses.clear();
    inverses.resize(elements.len(), FieldElement::ONE);
    for index in (0..elements.len()).rev() {
        inverses[index] = if index == 0 {
            running
        } else {
            running.mul(&prefix_products[index - 1])
        };
        running = running.mul(&elements[index]);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;
    use k256::Scalar;
    use k256::elliptic_curve::Field;
    use k256::elliptic_curve::point::AffineCoordinates;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn fe(point: &ProjectivePoint) -> FePoint {
        FePoint::from_projective(point).unwrap()
    }

    /// The oracle: x-key of `lhs + rhs` via k256's complete projective
    /// addition, `None` for an identity sum.
    fn reference_x_key(lhs: &ProjectivePoint, rhs: &ProjectivePoint) -> Option<[u8; 32]> {
        let sum = *lhs + *rhs;
        (sum != ProjectivePoint::IDENTITY).then(|| sum.to_affine().x().into())
    }

    #[test]
    fn batch_x_keys_matches_per_point_and_skips_identity() {
        let mut rng = StdRng::seed_from_u64(0x0BA7_C4C0);
        let g = ProjectivePoint::GENERATOR;
        let mut points = Vec::new();
        for _ in 0..64 {
            points.push(g * Scalar::random(&mut rng));
        }
        // Identity planted at the ends and the middle: it must map to None
        // and not disturb its neighbours' keys.
        points.insert(0, ProjectivePoint::IDENTITY);
        points.insert(32, ProjectivePoint::IDENTITY);
        points.push(ProjectivePoint::IDENTITY);

        let keys = batch_x_keys(&points);
        assert_eq!(keys.len(), points.len());
        for (point, key) in points.iter().zip(keys) {
            if point == &ProjectivePoint::IDENTITY {
                assert_eq!(key, None);
            } else {
                let reference: [u8; 32] = point.to_affine().x().into();
                assert_eq!(key, Some(reference));
            }
        }
    }

    #[test]
    fn matches_projective_addition_on_random_and_degenerate_pairs() {
        let mut rng = StdRng::seed_from_u64(0x0BA7_C4AF);
        let g = ProjectivePoint::GENERATOR;
        let mut pairs = Vec::new();
        for _ in 0..256 {
            let lhs = g * Scalar::random(&mut rng);
            let rhs = g * Scalar::random(&mut rng);
            pairs.push((lhs, rhs));
        }
        // Planted degenerate and near-degenerate cases: doubling, identity
        // cancellation, negated chords, and small related multiples (the
        // shapes rationally related roster keys produce in the pair-sum
        // scan).
        let p = g * Scalar::random(&mut rng);
        let q = g * Scalar::random(&mut rng);
        pairs.push((p, p));
        pairs.push((p, -p));
        pairs.push((-p, p));
        pairs.push((p, q + q));
        pairs.push((-p, -p));
        for v in 1..=8u64 {
            let small = g * Scalar::from(v);
            pairs.push((small, small));
            pairs.push((small, -small));
            pairs.push((small, g * Scalar::from(v + 1)));
        }

        let fe_pairs: Vec<(FePoint, FePoint)> =
            pairs.iter().map(|(lhs, rhs)| (fe(lhs), fe(rhs))).collect();
        let keys = batch_add_x_keys(&fe_pairs);
        assert_eq!(keys.len(), pairs.len());
        for ((lhs, rhs), key) in pairs.iter().zip(keys) {
            assert_eq!(key, reference_x_key(lhs, rhs));
        }
    }

    #[test]
    fn full_batch_add_matches_projective_addition() {
        let mut rng = StdRng::seed_from_u64(0x0BA7_C4B3);
        let g = ProjectivePoint::GENERATOR;
        let mut pairs = Vec::new();
        for _ in 0..256 {
            let lhs = g * Scalar::random(&mut rng);
            let rhs = g * Scalar::random(&mut rng);
            pairs.push((lhs, rhs));
        }
        let p = g * Scalar::random(&mut rng);
        let q = g * Scalar::random(&mut rng);
        pairs.extend([(p, p), (p, -p), (-p, p), (p, q), (-p, -q)]);

        let fe_pairs: Vec<(FePoint, FePoint)> =
            pairs.iter().map(|(lhs, rhs)| (fe(lhs), fe(rhs))).collect();
        let mut scratch = BatchAddScratch::with_capacity(fe_pairs.len());
        let mut sums = Vec::new();
        batch_add_points_visit(&fe_pairs, &mut scratch, |sum| {
            sums.push(sum);
            false
        });

        for ((lhs, rhs), sum) in pairs.iter().zip(sums) {
            let reference = *lhs + *rhs;
            match sum {
                Some(sum) => {
                    let reference = fe(&reference);
                    assert_eq!(sum.x_key(), reference.x_key());
                    assert_eq!(sum.y.to_bytes(), reference.y.to_bytes());
                }
                None => assert_eq!(reference, ProjectivePoint::IDENTITY),
            }
        }
    }

    #[test]
    fn signed_stride_batch_matches_projective_signed_rows() {
        let mut rng = StdRng::seed_from_u64(0x0BA7_C4B4);
        let g = ProjectivePoint::GENERATOR;
        let stride = g * Scalar::from(987_654_321u64);
        let negated_stride = -stride;
        let mut directions = Vec::new();
        for _ in 0..256 {
            directions.push(g * Scalar::random(&mut rng));
        }
        directions.push(stride);
        directions.push(negated_stride);
        directions.push(g * Scalar::from(1u64));

        let fe_directions = directions.iter().map(fe).collect::<Vec<_>>();
        let negated_stride_fe = fe(&negated_stride);
        let mut scratch = BatchAddScratch::with_capacity(fe_directions.len());
        let mut keys = Vec::new();
        batch_add_signed_stride_x_keys_visit(
            &fe_directions,
            negated_stride_fe,
            &mut scratch,
            |minus_key, plus_key| {
                keys.push((minus_key, plus_key));
                false
            },
        );

        for (direction, (minus_key, plus_key)) in directions.iter().zip(keys) {
            assert_eq!(minus_key, reference_x_key(direction, &negated_stride));
            assert_eq!(plus_key, reference_x_key(&(-*direction), &negated_stride));
        }
    }

    #[test]
    fn negated_matches_projective_negation() {
        let mut rng = StdRng::seed_from_u64(0x0BA7_C4B0);
        for _ in 0..32 {
            let p = ProjectivePoint::GENERATOR * Scalar::random(&mut rng);
            let negated = fe(&p).negated();
            let reference = fe(&(-p));
            // x is negation-invariant, so the y comparison is the assertion
            // that actually pins the negation (codex review P2).
            assert_eq!(negated.x_key(), reference.x_key());
            assert_eq!(negated.y.to_bytes(), reference.y.to_bytes());
            assert_eq!(fe(&p).x_key(), reference.x_key());
        }
    }

    #[test]
    fn negated_feeds_subtraction_exactly() {
        // The consumers build `T − a·S` as `(T, negated(a·S))` kernel pairs;
        // pin that full path (including the y-negation the x-key cannot see)
        // against projective subtraction.
        let mut rng = StdRng::seed_from_u64(0x0BA7_C4B2);
        for _ in 0..32 {
            let t = ProjectivePoint::GENERATOR * Scalar::random(&mut rng);
            let s = ProjectivePoint::GENERATOR * Scalar::random(&mut rng);
            let keys = batch_add_x_keys(&[(fe(&t), fe(&s).negated())]);
            assert_eq!(keys, vec![reference_x_key(&t, &(-s))]);
        }
        // And the cancellation through negated(): T − T is the identity.
        let t = ProjectivePoint::GENERATOR * Scalar::random(&mut rng);
        assert_eq!(batch_add_x_keys(&[(fe(&t), fe(&t).negated())]), vec![None]);
    }

    #[test]
    fn from_affine_rejects_identity() {
        assert!(FePoint::from_affine(&AffinePoint::IDENTITY).is_none());
        assert!(FePoint::from_projective(&ProjectivePoint::IDENTITY).is_none());
    }

    #[test]
    fn batch_invert_matches_field_inversion() {
        let mut rng = StdRng::seed_from_u64(0x0BA7_C4B1);
        let elements: Vec<FieldElement> = (0..65)
            .map(|_| {
                let p = ProjectivePoint::GENERATOR * Scalar::random(&mut rng);
                fe(&p).x
            })
            .collect();
        let inverses = batch_invert(&elements);
        for (element, inverse) in elements.iter().zip(inverses) {
            assert_eq!(
                element.mul(&inverse).normalize(),
                FieldElement::ONE.normalize()
            );
        }
    }

    #[test]
    fn empty_batch_is_fine() {
        assert!(batch_add_x_keys(&[]).is_empty());
    }
}
