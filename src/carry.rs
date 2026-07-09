//! Exact-integer carry chain proving `m ≤ n − 1` (§4.2).
//!
//! The range proof (§4.1) bounds each limb `v_k ∈ [0, 2^ℓ)`, but `L·ℓ = 264 >
//! 256` over-covers a scalar — so the recomposed integer `M = Σ_k v_k·2^{ℓk}`
//! could exceed `n − 1` (the `m' = n + x` wraparound: two distinct integers
//! with the same `m·G`). This module forecloses that by proving, over the
//! additively-homomorphic Pedersen commitments, the schoolbook-addition
//! identity `m + m̄ = n − 1` for the complement `m̄ = n − 1 − m`.
//!
//! For each limb the equation `v_k + v̄_k + c_{k−1} = (n−1)_k + c_k·2^ℓ` holds
//! with boolean carries `c_k ∈ {0,1}`, `c_{-1} = c_{L−1} = 0`. Two proof
//! ingredients enforce it:
//! - **boolean carries:** `ComC_k = c_k·G + g_k·H` for `k ∈ [0, L−1)`; each
//!   `c_k ∈ {0,1}` is proven by the §4.1 BP++ aggregate's base-2 digit group
//!   (the carry commitments are statement values of the range circuit — this
//!   module returns the [`CarryWitness`] the range prover consumes);
//! - **per-limb `H`-residual Schnorr:** the verifier forms the public point
//!   `R_k = Com_k + Com̄_k + ComC_{k−1} − (n−1)_k·G − 2^ℓ·ComC_k` and the prover
//!   shows `R_k ∈ ⟨H⟩` (a pure `H`-multiple). Since a nonzero `G`-coefficient
//!   would express `log_G(H)` (breaking Pedersen binding), each limb equation
//!   holds mod `n`; the bounded ranges then force it over `Z`, and the
//!   telescoped sum collapses to `M + M̄ = n − 1` with `M, M̄ ≥ 0`, so
//!   `M ≤ n − 1 < n`.
//!
//! The range proofs on the complement limbs `v̄_k` are load-bearing: without
//! `v̄_k ≥ 0` the bounding argument fails. They are part of the same BP++
//! aggregate; this module proves only the carry relation among the public
//! commitment points.
//!
//! The residual sigma runs under the shared final challenge `x` (§2), so the
//! API is two-phase: [`carry_commit`] emits the carry commitments (statement
//! item 18) and residual announcements (item 24) plus a secret
//! [`CarryProverState`]; [`CarryProverState::respond`] closes it once `x` is
//! squeezed.

use crate::error::Error;
use crate::generators::{g, h};
use crate::limbs::{LIMB_COUNT, LIMB_MODULUS, decompose};
use crate::pedersen::Commitment;
use k256::elliptic_curve::Field;
use k256::{ProjectivePoint, Scalar};
use rand_core::{CryptoRng, RngCore};
use zeroize::Zeroizing;

/// Public phase-1 output of the carry chain: the carry commitments (statement
/// item 18, absorbed before ANY challenge) and the residual announcements
/// (item 24, absorbed before the final squeeze).
#[derive(Clone, Debug)]
pub struct CarryCommitment {
    /// `ComC_k = c_k·G + g_k·H` for `k ∈ [0, L−1)` (the committed boolean
    /// carries; `c_{-1}` and `c_{L−1}` are the constant zero carries).
    pub carry_commitments: Vec<ProjectivePoint>,
    /// `H`-residual Schnorr announcements `A^R_k = ρ_k·H` for `k ∈ [0, L)`.
    pub residual_announcements: Vec<ProjectivePoint>,
}

/// The carry bits and blindings, returned to the caller so the BP++ range
/// circuit can include `ComC_k` in its base-2 digit group (the same openings
/// feed both proofs — one source, no drift). Secret; zeroized on drop.
#[derive(zeroize::Zeroize, zeroize::ZeroizeOnDrop)]
pub struct CarryWitness {
    /// `c_k ∈ {0, 1}` for `k ∈ [0, L−1)`.
    pub bits: Vec<u32>,
    /// `g_k` — the blinding of `ComC_k`.
    pub blindings: Vec<Scalar>,
}

/// Secret phase-1 state retained to compute responses once `x` is known: the
/// residual Schnorr nonces and witnesses, zeroized on drop.
#[derive(zeroize::Zeroize, zeroize::ZeroizeOnDrop)]
pub struct CarryProverState {
    residual_nonces: Zeroizing<Vec<Scalar>>,
    residual_witnesses: Zeroizing<Vec<Scalar>>,
}

/// Phase-2 output: the residual Schnorr responses.
#[derive(Clone, Debug)]
pub struct CarryResponses {
    /// Residual Schnorr responses `z^R_k = ρ_k + x·w_k` for `k ∈ [0, L)`.
    pub residual_responses: Vec<Scalar>,
}

/// Phase 1. Build the carry-chain proof for `m` against the complement
/// `m̄ = n − 1 − m`.
///
/// `value_blindings[k]` / `complement_blindings[k]` are the `s_k` / `s̄_k` that
/// mint `Com_k` / `Com̄_k`; they are the residual Schnorr's `H`-witness
/// material and so must match the commitments [`carry_verify`] receives.
/// Returns the public [`CarryCommitment`], the secret [`CarryProverState`],
/// and the [`CarryWitness`] for the BP++ base-2 digit group.
///
/// # Errors
///
/// Returns [`Error::DegenerateInput`] if a blinding slice is not `L` long, or
/// if `m`'s limbs and the complement do not satisfy `m + m̄ = n − 1` (an
/// internal invariant that always holds for the derived complement).
pub fn carry_commit<R: RngCore + CryptoRng>(
    m: &Scalar,
    value_blindings: &[Scalar],
    complement_blindings: &[Scalar],
    rng: &mut R,
) -> Result<(CarryCommitment, CarryProverState, CarryWitness), Error> {
    if value_blindings.len() != LIMB_COUNT || complement_blindings.len() != LIMB_COUNT {
        return Err(Error::DegenerateInput("carry chain: blinding count != L"));
    }

    let v = Zeroizing::new(decompose(m));
    let m_bar = Zeroizing::new(-Scalar::ONE - *m); // n − 1 − m, in [0, n−1]
    let v_bar = Zeroizing::new(decompose(&m_bar));
    let nm1 = decompose(&(-Scalar::ONE)); // (n−1)_k

    let two_l = Scalar::from(LIMB_MODULUS);

    let mut carry_commitments = Vec::with_capacity(LIMB_COUNT - 1);
    let mut carry_bits = Zeroizing::new(Vec::with_capacity(LIMB_COUNT - 1));
    // g_k carry blindings — secret (ComC_k − g_k·H reveals the carry), zeroized.
    let mut carry_blindings = Zeroizing::new(Vec::with_capacity(LIMB_COUNT - 1));

    // Schoolbook addition v + v̄ with carries; the honest complement makes it
    // telescope with a zero carry out of the top limb.
    let mut c_prev: u64 = 0; // c_{-1}
    for k in 0..LIMB_COUNT {
        let sum = u64::from(v[k]) + u64::from(v_bar[k]) + c_prev;
        if u64::from(nm1[k]) != sum % LIMB_MODULUS {
            return Err(Error::DegenerateInput(
                "carry chain: limb sum inconsistent with n-1",
            ));
        }
        let c_cur = sum / LIMB_MODULUS; // ∈ {0,1}
        if k < LIMB_COUNT - 1 {
            let blind = Scalar::random(&mut *rng);
            let comc = Commitment::point_of(Scalar::from(c_cur), blind);
            carry_commitments.push(comc);
            carry_bits.push(u32::try_from(c_cur).unwrap_or(u32::MAX));
            carry_blindings.push(blind);
        } else if c_cur != 0 {
            return Err(Error::DegenerateInput("carry chain: nonzero final carry"));
        }
        c_prev = c_cur;
    }

    // Residual Schnorrs: R_k = w_k·H with w_k = s_k + s̄_k + g_{k−1} − 2^ℓ·g_k
    // (g_{-1} = g_{L−1} = 0).
    let mut residual_announcements = Vec::with_capacity(LIMB_COUNT);
    let mut residual_nonces = Vec::with_capacity(LIMB_COUNT);
    let mut residual_witnesses = Vec::with_capacity(LIMB_COUNT);
    for k in 0..LIMB_COUNT {
        let g_in = if k == 0 {
            Scalar::ZERO
        } else {
            carry_blindings[k - 1]
        };
        let g_out = carry_blindings.get(k).copied().unwrap_or(Scalar::ZERO);
        let w_k = value_blindings[k] + complement_blindings[k] + g_in - two_l * g_out;
        let rho = Scalar::random(&mut *rng);
        residual_announcements.push(h() * rho);
        residual_nonces.push(rho);
        residual_witnesses.push(w_k);
    }

    Ok((
        CarryCommitment {
            carry_commitments,
            residual_announcements,
        },
        CarryProverState {
            residual_nonces: Zeroizing::new(residual_nonces),
            residual_witnesses: Zeroizing::new(residual_witnesses),
        },
        CarryWitness {
            bits: std::mem::take(&mut carry_bits),
            blindings: std::mem::take(&mut carry_blindings),
        },
    ))
}

impl CarryProverState {
    /// Phase 2. Compute the residual responses `z^R_k = ρ_k + x·w_k` under
    /// the shared final challenge `x`.
    #[must_use]
    pub fn respond(self, x: Scalar) -> CarryResponses {
        let residual_responses = self
            .residual_nonces
            .iter()
            .zip(self.residual_witnesses.iter())
            .map(|(rho, w)| *rho + x * w)
            .collect();
        CarryResponses { residual_responses }
    }
}

/// Verify the carry chain against the limb commitments `Com_k`
/// (`value_commits`) and `Com̄_k` (`complement_commits`) and the shared
/// challenge `x`.
///
/// Checks every residual
/// `z^R_k·H == A^R_k + x·R_k` with
/// `R_k = Com_k + Com̄_k + ComC_{k−1} − (n−1)_k·G − 2^ℓ·ComC_k`
/// (`ComC_{-1} = ComC_{L−1} = O`). Together with the BP++ aggregate — which
/// range-bounds both limb sets AND proves each carry boolean — this proves
/// `M = Σ_k v_k·2^{ℓk} ≤ n − 1`.
///
/// Operates on already-canonical [`Scalar`]s; a future wire/decode layer must
/// reject non-canonical responses (`< n`, soundness-doc §1) before calling.
///
/// # Errors
///
/// Returns [`Error::Verification`] on a shape mismatch or a failed residual
/// Schnorr.
pub fn carry_verify(
    value_commits: &[ProjectivePoint],
    complement_commits: &[ProjectivePoint],
    carry: &CarryCommitment,
    resp: &CarryResponses,
    x: Scalar,
) -> Result<(), Error> {
    if value_commits.len() != LIMB_COUNT || complement_commits.len() != LIMB_COUNT {
        return Err(Error::Verification("carry chain: commitment count != L"));
    }
    if carry.carry_commitments.len() != LIMB_COUNT - 1
        || carry.residual_announcements.len() != LIMB_COUNT
        || resp.residual_responses.len() != LIMB_COUNT
    {
        return Err(Error::Verification("carry chain: proof shape mismatch"));
    }

    let nm1 = decompose(&(-Scalar::ONE));
    let two_l = Scalar::from(LIMB_MODULUS);

    // Each limb equation holds (R_k is a pure H-multiple). Booleanity of the
    // committed carries is the BP++ aggregate's base-2 group (§4.1).
    for k in 0..LIMB_COUNT {
        let comc_in = if k == 0 {
            ProjectivePoint::IDENTITY
        } else {
            carry.carry_commitments[k - 1]
        };
        let comc_out = carry
            .carry_commitments
            .get(k)
            .copied()
            .unwrap_or(ProjectivePoint::IDENTITY);
        let nm1_k = Scalar::from(u64::from(nm1[k]));
        let r_k =
            value_commits[k] + complement_commits[k] + comc_in - g() * nm1_k - comc_out * two_l;
        if h() * resp.residual_responses[k] != carry.residual_announcements[k] + r_k * x {
            return Err(Error::Verification("carry residual Schnorr failed"));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;
    use crate::transcript::Transcript;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// A representative shared challenge bound to the carry commitments and
    /// residual announcements (soundness-doc items 18, 24).
    fn challenge_for(c: &CarryCommitment) -> Scalar {
        let mut t = Transcript::new();
        t.absorb_list_len(c.carry_commitments.len());
        for p in &c.carry_commitments {
            t.absorb_point(p);
        }
        for a in &c.residual_announcements {
            t.absorb_point(a);
        }
        t.challenge()
    }

    /// Build `Com_k`/`Com̄_k` for `m` and its complement under fresh blindings,
    /// returning the public commitment points plus the blindings (which the
    /// prover needs).
    fn commitments(
        m: &Scalar,
        rng: &mut StdRng,
    ) -> (
        Vec<ProjectivePoint>,
        Vec<ProjectivePoint>,
        Vec<Scalar>,
        Vec<Scalar>,
    ) {
        let v = decompose(m);
        let v_bar = decompose(&(-Scalar::ONE - *m));
        let s: Vec<Scalar> = (0..LIMB_COUNT).map(|_| Scalar::random(&mut *rng)).collect();
        let s_bar: Vec<Scalar> = (0..LIMB_COUNT).map(|_| Scalar::random(&mut *rng)).collect();
        let value_commits = (0..LIMB_COUNT)
            .map(|k| Commitment::point_of(Scalar::from(u64::from(v[k])), s[k]))
            .collect();
        let complement_commits = (0..LIMB_COUNT)
            .map(|k| Commitment::point_of(Scalar::from(u64::from(v_bar[k])), s_bar[k]))
            .collect();
        (value_commits, complement_commits, s, s_bar)
    }

    fn n_minus_one() -> Scalar {
        -Scalar::ONE
    }

    #[test]
    fn honest_chain_verifies() {
        let mut rng = StdRng::seed_from_u64(0x0C_A2_19_01);
        let values = [
            Scalar::ZERO,
            Scalar::ONE,
            n_minus_one(),
            Scalar::from(1_000_000u64),
            Scalar::random(&mut rng),
        ];
        for m in values {
            let (vc, cc, s, s_bar) = commitments(&m, &mut rng);
            let (carry, state, _witness) = carry_commit(&m, &s, &s_bar, &mut rng).unwrap();
            let x = challenge_for(&carry);
            let resp = state.respond(x);
            assert!(
                carry_verify(&vc, &cc, &carry, &resp, x).is_ok(),
                "honest chain failed to verify"
            );
        }
    }

    #[test]
    fn random_scalars_verify() {
        let mut rng = StdRng::seed_from_u64(0x0C_A2_19_02);
        for _ in 0..48 {
            let m = Scalar::random(&mut rng);
            let (vc, cc, s, s_bar) = commitments(&m, &mut rng);
            let (carry, state, _witness) = carry_commit(&m, &s, &s_bar, &mut rng).unwrap();
            let x = challenge_for(&carry);
            let resp = state.respond(x);
            assert!(carry_verify(&vc, &cc, &carry, &resp, x).is_ok());
        }
    }

    #[test]
    fn wrong_challenge_rejected() {
        let mut rng = StdRng::seed_from_u64(0x0C_A2_19_03);
        let m = Scalar::from(42u64);
        let (vc, cc, s, s_bar) = commitments(&m, &mut rng);
        let (carry, state, _witness) = carry_commit(&m, &s, &s_bar, &mut rng).unwrap();
        let x = challenge_for(&carry);
        let resp = state.respond(x);
        assert!(matches!(
            carry_verify(&vc, &cc, &carry, &resp, x + Scalar::ONE),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn prover_state_is_explicitly_zeroizable() {
        static_assertions::assert_impl_all!(
            CarryProverState: zeroize::Zeroize, zeroize::ZeroizeOnDrop
        );
        static_assertions::assert_not_impl_any!(CarryProverState: Clone, Copy);
    }

    #[test]
    fn tampered_residual_response_rejected() {
        let mut rng = StdRng::seed_from_u64(0x0C_A2_19_04);
        let m = Scalar::from(777u64);
        let (vc, cc, s, s_bar) = commitments(&m, &mut rng);
        let (carry, state, _witness) = carry_commit(&m, &s, &s_bar, &mut rng).unwrap();
        let x = challenge_for(&carry);
        let mut resp = state.respond(x);
        resp.residual_responses[0] += Scalar::ONE;
        assert!(matches!(
            carry_verify(&vc, &cc, &carry, &resp, x),
            Err(Error::Verification("carry residual Schnorr failed"))
        ));
    }

    #[test]
    fn boolean_valid_but_flipped_carry_rejected() {
        // A flipped carry bit is still boolean — the BP++ base-2 group would
        // happily range-bound it. The residual layer is what must reject:
        // with c'_k = 1 − c_k the limb equation's G-component is ±2^ℓ, so
        // R_k ∉ ⟨H⟩ and no response can satisfy the Schnorr.
        let mut rng = StdRng::seed_from_u64(0x0C_A2_19_0A);
        let m = Scalar::from(0xFEED_F00D_u64);
        let (vc, cc, s, s_bar) = commitments(&m, &mut rng);
        let (mut carry, state, witness) = carry_commit(&m, &s, &s_bar, &mut rng).unwrap();
        let delta = if witness.bits[0] == 0 { g() } else { -g() };
        carry.carry_commitments[0] += delta;
        let x = challenge_for(&carry);
        let resp = state.respond(x);
        assert!(matches!(
            carry_verify(&vc, &cc, &carry, &resp, x),
            Err(Error::Verification("carry residual Schnorr failed"))
        ));
    }

    #[test]
    fn wraparound_complement_rejected() {
        // The m' = n + x defense in unit form: an honest carry proof for m,
        // verified against complement commitments for a DIFFERENT m̄' (here the
        // complement of m+1, off by one). The limb equations no longer hold, so
        // some R_k gains a G-component (R_k ∉ ⟨H⟩) and its residual Schnorr —
        // built for the honest complement — fails. This is exactly what stops a
        // coalition from certifying an over-n M with a mismatched complement.
        let mut rng = StdRng::seed_from_u64(0x0C_A2_19_06);
        let m = Scalar::from(123_456u64);
        let (vc, _cc_honest, s, s_bar) = commitments(&m, &mut rng);
        let (carry, state, _witness) = carry_commit(&m, &s, &s_bar, &mut rng).unwrap();
        let x = challenge_for(&carry);
        let resp = state.respond(x);

        // Complement points for m+1 (so m + m̄' = n − 2 ≠ n − 1), reusing the
        // same blindings s_bar so only the committed values are inconsistent.
        let v_bad = decompose(&(-Scalar::ONE - (m + Scalar::ONE)));
        let cc_bad: Vec<ProjectivePoint> = (0..LIMB_COUNT)
            .map(|k| Commitment::point_of(Scalar::from(u64::from(v_bad[k])), s_bar[k]))
            .collect();

        assert!(matches!(
            carry_verify(&vc, &cc_bad, &carry, &resp, x),
            Err(Error::Verification("carry residual Schnorr failed"))
        ));
    }

    #[test]
    fn wrong_blinding_count_rejected_at_commit() {
        let mut rng = StdRng::seed_from_u64(0x0C_A2_19_07);
        let m = Scalar::from(5u64);
        let short = vec![Scalar::ONE; LIMB_COUNT - 1];
        let full = vec![Scalar::ONE; LIMB_COUNT];
        // Both arms of the length guard reject.
        assert!(matches!(
            carry_commit(&m, &short, &full, &mut rng),
            Err(Error::DegenerateInput(_))
        ));
        assert!(matches!(
            carry_commit(&m, &full, &short, &mut rng),
            Err(Error::DegenerateInput(_))
        ));
    }
}
