//! Weighted norm linear argument (WNLA) — the BP++ compression core (§4.1).
//!
//! Proves knowledge of vectors `l⃗`, `n⃗` opening a commitment
//! `C = v·G + ⟨h⃗, l⃗⟩ + ⟨g⃗, n⃗⟩` with `v = |n⃗|²_μ + ⟨c⃗, l⃗⟩`, where
//! `|n⃗|²_μ = Σ_i μ^{i+1}·n_i²` is the μ-weighted norm. Each fold round the
//! prover sends two cross-term points `(X, R)`, receives a challenge `γ`, and
//! both sides halve every vector (even/odd split); after the frozen number of
//! rounds the residual `l⃗`, `n⃗` ship in the clear and the verifier checks the
//! commitment equation directly. The protocol is [BP++ §4] as implemented —
//! deterministically, with no prover randomness — so it is differentially
//! testable against the `bp-pp` oracle (same math, different transcripts).
//!
//! **Challenge injection.** The fold challenges come through
//! [`FoldChallenges`], not from a transcript owned here: production wires the
//! §2 ratchet (absorb `(i, X_i, R_i)` → squeeze `gamma ‖ BE16(i)`, §3 item
//! 22); the differential harness replays the oracle's Merlin transcript. The
//! Fiat–Shamir discipline therefore lives at the call site, where the
//! soundness spec pins it.
//!
//! Verifier obligations enforced here (soundness doc §4.1): fixed shape (the
//! fold count and residual lengths are determined by the generator lengths —
//! never by the proof), identity flight points rejected, and the base-case
//! commitment equation. Scalar/point wire canonicality is the decode door's
//! job (§7), upstream of this module.

#![allow(clippy::many_single_char_names)]

use crate::error::Error;
use crate::msm::{generator_mul, lincomb2, msm, msm_vartime_public};
use k256::elliptic_curve::ops::Invert;
use k256::{ProjectivePoint, Scalar};
use zeroize::Zeroize;

/// Source of the per-round fold challenges `γ_i`.
///
/// `round` is 1-based and caller-counted (never prover-supplied); the running
/// `commitment` and the current vector lengths are passed so a replaying
/// implementation can mirror an oracle transcript that absorbs them.
pub trait FoldChallenges {
    // The argument list mirrors exactly what a replayed oracle transcript
    // absorbs per round; bundling them into a struct would only obscure that.
    #[allow(clippy::too_many_arguments)]
    fn gamma(
        &mut self,
        round: u16,
        commitment: &ProjectivePoint,
        x: &ProjectivePoint,
        r: &ProjectivePoint,
        l_len: usize,
        n_len: usize,
    ) -> Scalar;
}

/// The public parameters of one WNLA instance: the scalar base `G`, the two
/// generator vectors, the public linear coefficients `c⃗`, and the weights
/// `(ρ, μ)` with `μ = ρ²` at the top level.
pub struct NormArg {
    pub g: ProjectivePoint,
    pub g_vec: Vec<ProjectivePoint>,
    pub h_vec: Vec<ProjectivePoint>,
    pub c: Vec<Scalar>,
    pub rho: Scalar,
    pub mu: Scalar,
}

/// A WNLA proof: the per-round cross terms in round order, then the residual
/// clear vectors. All counts are functions of the instance shape — the wire
/// layer serializes this with zero length fields.
#[derive(Clone, Debug)]
pub struct NormProof {
    pub x: Vec<ProjectivePoint>,
    pub r: Vec<ProjectivePoint>,
    pub l: Vec<Scalar>,
    pub n: Vec<Scalar>,
}

/// `⟨a⃗, b⃗⟩` with zero-extension of the shorter vector (the steady-state
/// length-1 fold side makes one side shorter by design).
fn inner(a: &[Scalar], b: &[Scalar]) -> Scalar {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `Σ_i w^{i+1}·a_i·b_i` — the weighted inner product; `|n⃗|²_w` is
/// `weighted_inner(n, n, w)`.
fn weighted_inner(a: &[Scalar], b: &[Scalar], w: &Scalar) -> Scalar {
    let mut exp = Scalar::ONE;
    let mut acc = Scalar::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        exp *= w;
        acc += *x * y * exp;
    }
    acc
}

/// Even/odd split: `([a_0, a_2, …], [a_1, a_3, …])`.
fn split<T: Copy>(v: &[T]) -> (Vec<T>, Vec<T>) {
    let even = v.iter().copied().step_by(2).collect();
    let odd = v.iter().copied().skip(1).step_by(2).collect();
    (even, odd)
}

/// `a⃗ + y·b⃗`, zero-extending the shorter side.
fn fold_scalars(a: &[Scalar], b: &[Scalar], y: &Scalar) -> Vec<Scalar> {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| {
            let x = a.get(i).copied().unwrap_or(Scalar::ZERO);
            let z = b.get(i).copied().unwrap_or(Scalar::ZERO);
            x + z * y
        })
        .collect()
}

/// `s·a⃗ + y·b⃗` over points, zero-extending the shorter side.
fn fold_points(
    a: &[ProjectivePoint],
    b: &[ProjectivePoint],
    s: &Scalar,
    y: &Scalar,
) -> Vec<ProjectivePoint> {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| {
            let x = a.get(i).copied().unwrap_or(ProjectivePoint::IDENTITY);
            let z = b.get(i).copied().unwrap_or(ProjectivePoint::IDENTITY);
            lincomb2(&x, s, &z, y)
        })
        .collect()
}

/// The post-split length of a folded vector: `ceil(len / 2)` (a length-1 side
/// stays length 1; only an even split shortens).
const fn folded_len(len: usize) -> usize {
    len.div_ceil(2)
}

/// The fold-round count and residual `(l, n)` lengths for an instance shape,
/// mirroring the prover's termination rule `l_len + n_len < 6`. This is what
/// makes the proof fixed-shape: the verifier derives the expected counts from
/// the frozen generators, never from the wire.
pub const fn expected_shape(h_len: usize, g_len: usize) -> (usize, usize, usize) {
    let (mut l_len, mut n_len, mut rounds) = (h_len, g_len, 0usize);
    while l_len + n_len >= 6 {
        l_len = folded_len(l_len);
        n_len = folded_len(n_len);
        rounds += 1;
    }
    (rounds, l_len, n_len)
}

impl NormArg {
    /// `C = v·G + ⟨h⃗, l⃗⟩ + ⟨g⃗, n⃗⟩` with `v = |n⃗|²_μ + ⟨c⃗, l⃗⟩`.
    pub fn commit(&self, l: &[Scalar], n: &[Scalar]) -> ProjectivePoint {
        let v = inner(&self.c, l) + weighted_inner(n, n, &self.mu);
        self.commit_with_msm(l, n, v, msm)
    }

    fn commit_public(&self, l: &[Scalar], n: &[Scalar]) -> ProjectivePoint {
        let v = inner(&self.c, l) + weighted_inner(n, n, &self.mu);
        self.commit_with_msm(l, n, v, msm_vartime_public)
    }

    fn commit_with_msm(
        &self,
        l: &[Scalar],
        n: &[Scalar],
        v: Scalar,
        msm_impl: fn(&[ProjectivePoint], &[Scalar]) -> ProjectivePoint,
    ) -> ProjectivePoint {
        let g_v = if self.g == ProjectivePoint::GENERATOR {
            generator_mul(&v)
        } else {
            self.g * v
        };
        g_v + msm_impl(&self.h_vec, l) + msm_impl(&self.g_vec, n)
    }

    /// Prove knowledge of `(l⃗, n⃗)` opening `commitment`. Deterministic — the
    /// WNLA carries no prover randomness; zero-knowledge is the caller's
    /// blinding flights (§4.1), not this argument's.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if the witness lengths do not match the
    /// generator vectors (the caller pads to the frozen shape first).
    pub fn prove<C: FoldChallenges>(
        &self,
        commitment: &ProjectivePoint,
        challenges: &mut C,
        l: Vec<Scalar>,
        n: Vec<Scalar>,
    ) -> Result<NormProof, Error> {
        if l.len() != self.h_vec.len() || n.len() != self.g_vec.len() {
            return Err(Error::DegenerateInput("norm witness shape mismatch"));
        }

        let mut com = *commitment;
        let mut g_vec = self.g_vec.clone();
        let mut h_vec = self.h_vec.clone();
        let mut c = self.c.clone();
        let mut rho = self.rho;
        let mut mu = self.mu;
        let (mut l, mut n) = (l, n);

        let mut xs = Vec::new();
        let mut rs = Vec::new();
        let mut round: u16 = 0;

        while l.len() + n.len() >= 6 {
            round += 1;
            let rho_inv = rho.invert_vartime().unwrap_or(Scalar::ZERO);
            let mu2 = mu * mu;

            let (c0, c1) = split(&c);
            let (mut l0, mut l1) = split(&l);
            let (mut n0, mut n1) = split(&n);
            let (g0, g1) = split(&g_vec);
            let (h0, h1) = split(&h_vec);

            let two = Scalar::from(2u64);
            let vx =
                weighted_inner(&n0, &n1, &mu2) * rho_inv * two + inner(&c0, &l1) + inner(&c1, &l0);
            let vr = weighted_inner(&n1, &n1, &mu2) + inner(&c1, &l1);

            let x = (if self.g == ProjectivePoint::GENERATOR {
                generator_mul(&vx)
            } else {
                self.g * vx
            }) + msm(&h0, &l1)
                + msm(&h1, &l0)
                + msm(&g0, &n1.iter().map(|v| *v * rho).collect::<Vec<_>>())
                + msm(&g1, &n0.iter().map(|v| *v * rho_inv).collect::<Vec<_>>());
            let r = (if self.g == ProjectivePoint::GENERATOR {
                generator_mul(&vr)
            } else {
                self.g * vr
            }) + msm(&h1, &l1)
                + msm(&g1, &n1);

            let y = challenges.gamma(round, &com, &x, &r, l.len(), n.len());

            h_vec = fold_points(&h0, &h1, &Scalar::ONE, &y);
            g_vec = fold_points(&g0, &g1, &rho, &y);
            c = fold_scalars(&c0, &c1, &y);
            // The folded witness replaces the previous round's (both are
            // secret); scrub the outgoing halves and the replaced vectors.
            let mut next_l = fold_scalars(&l0, &l1, &y);
            let mut n0_scaled: Vec<Scalar> = n0.iter().map(|v| *v * rho_inv).collect();
            let mut next_n = fold_scalars(&n0_scaled, &n1, &y);
            n0_scaled.zeroize();
            std::mem::swap(&mut l, &mut next_l);
            std::mem::swap(&mut n, &mut next_n);
            next_l.zeroize();
            next_n.zeroize();
            l0.zeroize();
            l1.zeroize();
            n0.zeroize();
            n1.zeroize();
            com = com + x * y + r * (y * y - Scalar::ONE);

            rho = mu;
            mu = mu2;
            xs.push(x);
            rs.push(r);
        }

        Ok(NormProof { x: xs, r: rs, l, n })
    }

    /// Verify a WNLA proof against `commitment`.
    ///
    /// # Errors
    ///
    /// [`Error::Verification`] on a shape mismatch (fold count or residual
    /// lengths differ from the instance-derived expectation), an identity
    /// flight point, or a failed base-case commitment equation.
    pub fn verify<C: FoldChallenges>(
        &self,
        commitment: &ProjectivePoint,
        challenges: &mut C,
        proof: &NormProof,
    ) -> Result<(), Error> {
        let (rounds, l_len, n_len) = expected_shape(self.h_vec.len(), self.g_vec.len());
        if proof.x.len() != rounds || proof.r.len() != rounds {
            return Err(Error::Verification("norm fold count mismatch"));
        }
        if proof.l.len() != l_len || proof.n.len() != n_len {
            return Err(Error::Verification("norm residual length mismatch"));
        }
        for p in proof.x.iter().chain(proof.r.iter()) {
            if p == &ProjectivePoint::IDENTITY {
                return Err(Error::Verification("norm identity flight point"));
            }
        }

        let mut com = *commitment;
        let mut g_vec = self.g_vec.clone();
        let mut h_vec = self.h_vec.clone();
        let mut c = self.c.clone();
        let mut rho = self.rho;
        let mut mu = self.mu;
        let (mut l_len, mut n_len) = (self.h_vec.len(), self.g_vec.len());

        for (i, (x, r)) in proof.x.iter().zip(proof.r.iter()).enumerate() {
            let round = u16::try_from(i + 1).unwrap_or(u16::MAX);
            let y = challenges.gamma(round, &com, x, r, l_len, n_len);

            let (c0, c1) = split(&c);
            let (g0, g1) = split(&g_vec);
            let (h0, h1) = split(&h_vec);

            h_vec = fold_points(&h0, &h1, &Scalar::ONE, &y);
            g_vec = fold_points(&g0, &g1, &rho, &y);
            c = fold_scalars(&c0, &c1, &y);
            com = com + *x * y + *r * (y * y - Scalar::ONE);

            rho = mu;
            mu *= mu;
            l_len = folded_len(l_len);
            n_len = folded_len(n_len);
        }

        let base = Self {
            g: self.g,
            g_vec,
            h_vec,
            c,
            rho,
            mu,
        };
        if base.commit_public(&proof.l, &proof.n) == com {
            Ok(())
        } else {
            Err(Error::Verification("norm base equation failed"))
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::*;
    use crate::generators::{g, gvec, hvec};
    use crate::transcript::Transcript;
    use k256::elliptic_curve::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// The production challenge source: the §2 ratchet, absorbing
    /// `(i, X_i, R_i)` per §3 item 23 and squeezing `gamma ‖ BE16(i)`.
    struct RatchetChallenges(Transcript);

    impl FoldChallenges for RatchetChallenges {
        fn gamma(
            &mut self,
            round: u16,
            _commitment: &ProjectivePoint,
            x: &ProjectivePoint,
            r: &ProjectivePoint,
            _l_len: usize,
            _n_len: usize,
        ) -> Scalar {
            self.0.absorb_u16(round);
            self.0.absorb_point(x);
            self.0.absorb_point(r);
            let mut label = b"gamma".to_vec();
            label.extend_from_slice(&round.to_be_bytes());
            self.0.squeeze(&label)
        }
    }

    fn ratchet() -> RatchetChallenges {
        RatchetChallenges(Transcript::new())
    }

    fn instance(h_len: u16, g_len: u16, seed: u64) -> (NormArg, Vec<Scalar>, Vec<Scalar>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let rho = Scalar::random(&mut rng);
        let arg = NormArg {
            g: g(),
            g_vec: gvec(g_len),
            h_vec: hvec(h_len),
            c: (0..h_len).map(|_| Scalar::random(&mut rng)).collect(),
            rho,
            mu: rho * rho,
        };
        let l = (0..h_len).map(|_| Scalar::random(&mut rng)).collect();
        let n = (0..g_len).map(|_| Scalar::random(&mut rng)).collect();
        (arg, l, n)
    }

    #[test]
    fn roundtrip_across_shapes() {
        // Covers: base-case-only (1+4 < 6 → zero folds), one fold, the
        // production-like power-of-two shapes, and the steady-state
        // length-1 l side.
        for (h_len, g_len) in [(1, 4), (4, 4), (4, 8), (16, 64), (1, 8), (2, 32)] {
            let (arg, l, n) = instance(h_len, g_len, 0xB9_00 + u64::from(h_len * g_len));
            let com = arg.commit(&l, &n);
            let proof = arg.prove(&com, &mut ratchet(), l, n).unwrap();
            let (rounds, l_len, n_len) = expected_shape(usize::from(h_len), usize::from(g_len));
            assert_eq!(proof.x.len(), rounds, "shape ({h_len},{g_len})");
            assert_eq!((proof.l.len(), proof.n.len()), (l_len, n_len));
            arg.verify(&com, &mut ratchet(), &proof)
                .unwrap_or_else(|e| panic!("shape ({h_len},{g_len}): {e}"));
        }
    }

    #[test]
    fn wrong_commitment_rejected() {
        let (arg, l, n) = instance(4, 8, 0xB9_01);
        let com = arg.commit(&l, &n);
        let proof = arg.prove(&com, &mut ratchet(), l, n).unwrap();
        let wrong = com + g();
        assert_eq!(
            arg.verify(&wrong, &mut ratchet(), &proof),
            Err(Error::Verification("norm base equation failed"))
        );
    }

    #[test]
    fn tampered_flight_point_rejected() {
        let (arg, l, n) = instance(4, 8, 0xB9_02);
        let com = arg.commit(&l, &n);
        let mut proof = arg.prove(&com, &mut ratchet(), l, n).unwrap();
        proof.x[0] += g();
        assert!(arg.verify(&com, &mut ratchet(), &proof).is_err());
    }

    #[test]
    fn identity_flight_point_rejected() {
        let (arg, l, n) = instance(4, 8, 0xB9_03);
        let com = arg.commit(&l, &n);
        let mut proof = arg.prove(&com, &mut ratchet(), l, n).unwrap();
        proof.r[1] = ProjectivePoint::IDENTITY;
        assert_eq!(
            arg.verify(&com, &mut ratchet(), &proof),
            Err(Error::Verification("norm identity flight point"))
        );
    }

    #[test]
    fn truncated_rounds_rejected() {
        let (arg, l, n) = instance(4, 8, 0xB9_04);
        let com = arg.commit(&l, &n);
        let mut proof = arg.prove(&com, &mut ratchet(), l, n).unwrap();
        proof.x.pop();
        proof.r.pop();
        assert_eq!(
            arg.verify(&com, &mut ratchet(), &proof),
            Err(Error::Verification("norm fold count mismatch"))
        );
    }

    #[test]
    fn padded_residual_rejected() {
        let (arg, l, n) = instance(4, 8, 0xB9_05);
        let com = arg.commit(&l, &n);
        let mut proof = arg.prove(&com, &mut ratchet(), l, n).unwrap();
        proof.n.push(Scalar::ZERO);
        assert_eq!(
            arg.verify(&com, &mut ratchet(), &proof),
            Err(Error::Verification("norm residual length mismatch"))
        );
    }

    #[test]
    fn witness_shape_mismatch_rejected_at_prove() {
        let (arg, l, mut n) = instance(4, 8, 0xB9_06);
        n.pop();
        let com = arg.commit(&l, &n);
        assert_eq!(
            arg.prove(&com, &mut ratchet(), l, n).err(),
            Some(Error::DegenerateInput("norm witness shape mismatch"))
        );
    }

    #[test]
    fn challenge_source_divergence_rejected() {
        // A verifier whose transcript saw different earlier absorptions
        // derives different gammas — the proof must not verify.
        let (arg, l, n) = instance(4, 8, 0xB9_07);
        let com = arg.commit(&l, &n);
        let proof = arg.prove(&com, &mut ratchet(), l, n).unwrap();
        let mut diverged = ratchet();
        diverged.0.absorb_u8(0xFF);
        assert!(arg.verify(&com, &mut diverged, &proof).is_err());
    }

    /// Differential oracle: replays the `bp-pp` crate's Merlin transcript so
    /// our deterministic fold math runs under the oracle's exact challenges —
    /// proofs then cross-verify in both directions.
    struct MerlinChallenges(merlin::Transcript);

    impl MerlinChallenges {
        fn new() -> Self {
            Self(merlin::Transcript::new(b"wnla differential"))
        }
    }

    impl FoldChallenges for MerlinChallenges {
        fn gamma(
            &mut self,
            _round: u16,
            commitment: &ProjectivePoint,
            x: &ProjectivePoint,
            r: &ProjectivePoint,
            l_len: usize,
            n_len: usize,
        ) -> Scalar {
            // Byte-for-byte the absorption sequence of bp-pp's wnla.rs.
            bp_pp::transcript::app_point(b"wnla_com", commitment, &mut self.0);
            bp_pp::transcript::app_point(b"wnla_x", x, &mut self.0);
            bp_pp::transcript::app_point(b"wnla_r", r, &mut self.0);
            self.0.append_u64(b"l.sz", l_len as u64);
            self.0.append_u64(b"n.sz", n_len as u64);
            bp_pp::transcript::get_challenge(b"wnla_challenge", &mut self.0)
        }
    }

    fn bp_pp_instance(arg: &NormArg) -> bp_pp::wnla::WeightNormLinearArgument {
        bp_pp::wnla::WeightNormLinearArgument {
            g: arg.g,
            g_vec: arg.g_vec.clone(),
            h_vec: arg.h_vec.clone(),
            c: arg.c.clone(),
            rho: arg.rho,
            mu: arg.mu,
        }
    }

    #[test]
    fn differential_ours_proves_bp_pp_verifies() {
        for (h_len, g_len) in [(4, 8), (8, 16), (16, 64)] {
            let (arg, l, n) = instance(h_len, g_len, 0xD1_00 + u64::from(h_len));
            let com = arg.commit(&l, &n);
            let proof = arg.prove(&com, &mut MerlinChallenges::new(), l, n).unwrap();
            // bp-pp stores fold rounds innermost-first; ours is round order.
            let oracle_proof = bp_pp::wnla::Proof {
                r: proof.r.iter().rev().copied().collect(),
                x: proof.x.iter().rev().copied().collect(),
                l: proof.l.clone(),
                n: proof.n.clone(),
            };
            let mut t = merlin::Transcript::new(b"wnla differential");
            assert!(
                bp_pp_instance(&arg).verify(&com, &mut t, oracle_proof),
                "bp-pp rejected our proof at shape ({h_len},{g_len})"
            );
        }
    }

    #[test]
    fn differential_bp_pp_proves_ours_verifies() {
        for (h_len, g_len) in [(4, 8), (8, 16), (16, 64)] {
            let (arg, l, n) = instance(h_len, g_len, 0xD2_00 + u64::from(h_len));
            let com = arg.commit(&l, &n);
            let mut t = merlin::Transcript::new(b"wnla differential");
            let oracle_proof = bp_pp_instance(&arg).prove(&com, &mut t, l.clone(), n.clone());
            let proof = NormProof {
                x: oracle_proof.x.iter().rev().copied().collect(),
                r: oracle_proof.r.iter().rev().copied().collect(),
                l: oracle_proof.l,
                n: oracle_proof.n,
            };
            arg.verify(&com, &mut MerlinChallenges::new(), &proof)
                .unwrap_or_else(|e| panic!("we rejected bp-pp's proof at ({h_len},{g_len}): {e}"));
        }
    }

    #[test]
    fn commitment_matches_bp_pp() {
        let (arg, l, n) = instance(8, 16, 0xD3_00);
        assert_eq!(arg.commit(&l, &n), bp_pp_instance(&arg).commit(&l, &n));
    }
}
