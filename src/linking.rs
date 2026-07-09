//! Shared-response multi-representation linking sigma (§4.3).
//!
//! The range proof (§4.1) and carry chain (§4.2) constrain the values
//! committed in `Com_k`, but nothing yet ties those to the actual ciphertexts
//! `(E_k, D_k)` or to the published target `C = m·G`. This sigma proves a
//! single consistent witness `{v_k, r_k, s_k}_k` such that simultaneously:
//! - (a) `E_k = r_k·G` and `D_k = v_k·G + r_k·pk` (the limb is correctly
//!   ElGamal-encrypted — and `E_k` is a real `r_k·G`, foreclosing the
//!   malformed-`E` decryption-failure ransom);
//! - (b) `Com_k = v_k·G + s_k·H` (the same `v_k` the range/carry proofs bound);
//! - (c) `(Σ_k 2^{ℓk}·v_k)·G = C` (the limbs recompose to the escrowed scalar).
//!
//! **Independent Schnorr legs under one challenge would NOT bind the same
//! `v_k`** across `D_k`, `Com_k`, and `C` — each would only prove its own
//! opening. The binding comes from a Maurer ([Mau09]) generalized Σ-protocol
//! that **reuses the same response scalar `z_{v,k}` for `v_k`** (and `z_{r,k}`
//! for `r_k`) in every leg it appears in. Extraction from two transcripts
//! sharing announcements with `x ≠ x'` then yields one `{v_k, r_k, s_k}`
//! satisfying (a), (b), (c) with the *same* `v_k`/`r_k` everywhere; in
//! particular leg (a) gives `E_k = r_k·G` exactly, so `D_k − sk·E_k = v_k·G`.
//!
//! Like the other sub-proofs this runs under one shared challenge `x` (§2):
//! [`linking_commit`] emits the announcements (soundness-doc item 24) plus a
//! secret [`LinkingProverState`], and [`LinkingProverState::respond`] produces
//! the responses once `x` is squeezed.

use crate::elgamal::LimbCiphertext;
use crate::error::Error;
use crate::generators::{g, h};
use crate::limbs::{LIMB_COUNT, limb_weights};
use crate::transcript::length_prefix;
#[cfg(test)]
use k256::elliptic_curve::Field;
use k256::elliptic_curve::bigint::U256;
use k256::elliptic_curve::ops::Reduce;
use k256::{FieldBytes, ProjectivePoint, Scalar};
use rand_core::{CryptoRng, RngCore};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// Internal nonce-derivation domain. The verifier transcript and wire format
/// stay at v1; this only hardens the prover's hidden sigma nonces against RNG
/// byte-stream repeats.
const LINKING_NONCE_DOMAIN: &[u8] = b"ve-capsule.linking-sigma.nonce.v1";

/// Public phase-1 announcements (soundness-doc item 24), absorbed before the
/// challenge is squeezed.
#[derive(Clone, Debug)]
#[allow(clippy::struct_field_names)]
pub struct LinkingCommitment {
    /// `A^E_k = β_k·G` (leg a, binds `r_k` in `E_k`).
    pub a_e: Vec<ProjectivePoint>,
    /// `A^D_k = α_k·G + β_k·pk` (leg a, `D_k`).
    pub a_d: Vec<ProjectivePoint>,
    /// `A^{Com}_k = α_k·G + γ_k·H` (leg b).
    pub a_com: Vec<ProjectivePoint>,
    /// `A^C = (Σ_k 2^{ℓk}·α_k)·G` — one aggregate over all limbs (leg c).
    pub a_c: ProjectivePoint,
}

/// Secret phase-1 state: the per-limb nonces `α_k`/`β_k`/`γ_k` (for
/// `v_k`/`r_k`/`s_k`) and the witnesses, retained to compute responses.
///
/// Zeroized on drop: after `respond` publishes the responses, a leaked nonce
/// would recover its witness (`v_k = (z_{v,k} − α_k)/x`), so the nonces and
/// witnesses must not linger in memory. Each field is [`Zeroizing`] (the same
/// per-secret-field mechanism the other prover states use).
#[derive(zeroize::Zeroize, zeroize::ZeroizeOnDrop)]
pub struct LinkingProverState {
    alpha: Zeroizing<Vec<Scalar>>,
    beta: Zeroizing<Vec<Scalar>>,
    gamma: Zeroizing<Vec<Scalar>>,
    v: Zeroizing<Vec<Scalar>>,
    r: Zeroizing<Vec<Scalar>>,
    s: Zeroizing<Vec<Scalar>>,
}

/// Phase-2 output: the shared response scalars. `z_v[k]` is reused across the
/// `D_k`, `Com_k`, and weighted-`C` checks (binding one `v_k`); `z_r[k]` across
/// the `E_k` and `D_k` checks (binding one `r_k`).
#[derive(Clone, Debug)]
#[allow(clippy::struct_field_names)]
pub struct LinkingResponses {
    /// `z_{v,k} = α_k + x·v_k`.
    pub z_v: Vec<Scalar>,
    /// `z_{r,k} = β_k + x·r_k`.
    pub z_r: Vec<Scalar>,
    /// `z_{s,k} = γ_k + x·s_k`.
    pub z_s: Vec<Scalar>,
}

/// The public statement a linking proof is verified against: the per-limb
/// ciphertexts `(E_k, D_k)` and commitments `Com_k`, the target `C = m·G`, and
/// the recovery key `pk`.
///
/// Bundling these keeps [`linking_verify`] from carrying loose parallel point
/// slices, and makes a `commitments`/ciphertext mix-up a type error rather than
/// a silent verification failure.
pub struct LinkingStatement<'a> {
    /// Per-limb `ElGamal` ciphertexts `(E_k, D_k)`.
    pub ciphertexts: &'a [LimbCiphertext],
    /// Per-limb value commitments `Com_k`.
    pub commitments: &'a [ProjectivePoint],
    /// The target commitment `C = m·G`.
    pub target: &'a ProjectivePoint,
    /// The recovery public key `pk`.
    pub pk: &'a ProjectivePoint,
}

fn reduce_be_to_scalar(bytes: [u8; 32]) -> Scalar {
    let mut repr = FieldBytes::default();
    repr.copy_from_slice(&bytes);
    <Scalar as Reduce<U256>>::reduce_bytes(&repr)
}

fn absorb_nonce_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(length_prefix(bytes.len()));
    hasher.update(bytes);
}

/// Derive a nonzero per-limb Schnorr nonce from fresh entropy, the relevant
/// witness, and a caller-built binding to every public item that can affect the
/// final `sigma.x` challenge. Repeated raw RNG output therefore does not repeat
/// `α`/`β`/`γ` across different statements or contexts.
fn statement_bound_nonce(
    seed: &[u8; 32],
    kind: &[u8],
    index: usize,
    witness: &Scalar,
    nonce_binding: &[u8],
) -> Scalar {
    let witness = Zeroizing::new(<[u8; 32]>::from(witness.to_bytes()));
    let index = u32::try_from(index).unwrap_or(u32::MAX).to_be_bytes();

    for counter in 0u8..=u8::MAX {
        let mut hasher = Sha256::new();
        absorb_nonce_field(&mut hasher, LINKING_NONCE_DOMAIN);
        absorb_nonce_field(&mut hasher, nonce_binding);
        absorb_nonce_field(&mut hasher, seed);
        absorb_nonce_field(&mut hasher, kind);
        absorb_nonce_field(&mut hasher, &index);
        absorb_nonce_field(&mut hasher, &*witness);
        absorb_nonce_field(&mut hasher, &[counter]);

        let candidate = reduce_be_to_scalar(hasher.finalize().into());
        if !bool::from(candidate.is_zero()) {
            return candidate;
        }
    }
    unreachable!("linking nonce derivation exhausted 256 nonzero retries")
}

/// Phase 1. Build the linking announcements for the witnesses `{v_k, r_k, s_k}`
/// under recovery key `pk`.
///
/// `v[k]` is the limb value (as a scalar), `r[k]` the `ElGamal` randomness of
/// `D_k`/`E_k`, `s[k]` the blinding of `Com_k`. `nonce_binding` must commit to
/// the capsule context and all public proof material that can affect the final
/// `sigma.x` challenge. Returns the public [`LinkingCommitment`] to absorb and
/// the secret [`LinkingProverState`].
///
/// # Errors
///
/// Returns [`Error::DegenerateInput`] if any witness slice is not `L` long.
pub fn linking_commit<R: RngCore + CryptoRng>(
    v: &[Scalar],
    r: &[Scalar],
    s: &[Scalar],
    pk: &ProjectivePoint,
    nonce_binding: &[u8],
    rng: &mut R,
) -> Result<(LinkingCommitment, LinkingProverState), Error> {
    if v.len() != LIMB_COUNT || r.len() != LIMB_COUNT || s.len() != LIMB_COUNT {
        return Err(Error::DegenerateInput("linking sigma: witness count != L"));
    }

    let weights = limb_weights();
    let mut alpha = Vec::with_capacity(LIMB_COUNT);
    let mut beta = Vec::with_capacity(LIMB_COUNT);
    let mut gamma = Vec::with_capacity(LIMB_COUNT);
    let mut a_e = Vec::with_capacity(LIMB_COUNT);
    let mut a_d = Vec::with_capacity(LIMB_COUNT);
    let mut a_com = Vec::with_capacity(LIMB_COUNT);
    let mut a_c = ProjectivePoint::IDENTITY;

    let mut seed = Zeroizing::new([0u8; 32]);
    rng.fill_bytes(&mut *seed);

    for (k, weight) in weights.iter().enumerate() {
        let al = statement_bound_nonce(&seed, b"alpha", k, &v[k], nonce_binding);
        let be = statement_bound_nonce(&seed, b"beta", k, &r[k], nonce_binding);
        let ga = statement_bound_nonce(&seed, b"gamma", k, &s[k], nonce_binding);
        a_e.push(g() * be);
        a_d.push(g() * al + *pk * be);
        a_com.push(g() * al + h() * ga);
        a_c += g() * (*weight * al);
        alpha.push(al);
        beta.push(be);
        gamma.push(ga);
    }

    Ok((
        LinkingCommitment {
            a_e,
            a_d,
            a_com,
            a_c,
        },
        LinkingProverState {
            alpha: Zeroizing::new(alpha),
            beta: Zeroizing::new(beta),
            gamma: Zeroizing::new(gamma),
            v: Zeroizing::new(v.to_vec()),
            r: Zeroizing::new(r.to_vec()),
            s: Zeroizing::new(s.to_vec()),
        },
    ))
}

impl LinkingProverState {
    /// Phase 2. Compute the shared responses under the challenge `x`:
    /// `z_{v,k}=α_k+x·v_k`, `z_{r,k}=β_k+x·r_k`, `z_{s,k}=γ_k+x·s_k`.
    ///
    /// Takes `self` by value (unlike the ring states' `&mut self`): `respond`
    /// reads the witnesses but moves nothing out, so dropping `self` here
    /// zeroizes every `Zeroizing` field in place. If a future change makes this
    /// move a field out, switch to the `&mut self` + `mem::take` pattern the
    /// ring states use, or the moved-from storage keeps stale secret bytes.
    #[must_use]
    pub fn respond(self, x: Scalar) -> LinkingResponses {
        let combine = |nonce: &[Scalar], witness: &[Scalar]| -> Vec<Scalar> {
            nonce
                .iter()
                .zip(witness.iter())
                .map(|(n, w)| *n + x * w)
                .collect()
        };
        LinkingResponses {
            z_v: combine(&self.alpha, &self.v),
            z_r: combine(&self.beta, &self.r),
            z_s: combine(&self.gamma, &self.s),
        }
    }
}

/// Verify the linking sigma against the public per-limb ciphertexts/commitments,
/// the target `C`, the recovery key `pk`, and the shared challenge `x`.
///
/// Per limb checks (the shared `z_v`/`z_r` are what force one `v_k`/`r_k`):
/// `z_r·G == A^E + x·E`, `z_v·G + z_r·pk == A^D + x·D`,
/// `z_v·G + z_s·H == A^{Com} + x·Com`; and once
/// `(Σ_k 2^{ℓk}·z_{v,k})·G == A^C + x·C`.
///
/// Operates on already-canonical [`Scalar`]s; a future wire/decode layer must
/// reject non-canonical responses (`< n`, soundness-doc §1) before calling.
///
/// # Errors
///
/// Returns [`Error::Verification`] on a shape mismatch or any failed leg.
pub fn linking_verify(
    statement: &LinkingStatement,
    commit: &LinkingCommitment,
    resp: &LinkingResponses,
    x: Scalar,
) -> Result<(), Error> {
    if statement.ciphertexts.len() != LIMB_COUNT
        || statement.commitments.len() != LIMB_COUNT
        || commit.a_e.len() != LIMB_COUNT
        || commit.a_d.len() != LIMB_COUNT
        || commit.a_com.len() != LIMB_COUNT
        || resp.z_v.len() != LIMB_COUNT
        || resp.z_r.len() != LIMB_COUNT
        || resp.z_s.len() != LIMB_COUNT
    {
        return Err(Error::Verification("linking sigma: shape mismatch"));
    }

    let weights = limb_weights();
    let mut weighted_zv = ProjectivePoint::IDENTITY;
    for (k, weight) in weights.iter().enumerate() {
        // leg a (E_k): binds r_k as the discrete log of E_k.
        if g() * resp.z_r[k] != commit.a_e[k] + statement.ciphertexts[k].e * x {
            return Err(Error::Verification("linking sigma: E_k leg failed"));
        }
        // leg a (D_k): same z_r[k] and z_v[k] tie r_k and v_k into D_k.
        if g() * resp.z_v[k] + *statement.pk * resp.z_r[k]
            != commit.a_d[k] + statement.ciphertexts[k].d * x
        {
            return Err(Error::Verification("linking sigma: D_k leg failed"));
        }
        // leg b (Com_k): same z_v[k] ties v_k into Com_k.
        if g() * resp.z_v[k] + h() * resp.z_s[k] != commit.a_com[k] + statement.commitments[k] * x {
            return Err(Error::Verification("linking sigma: Com_k leg failed"));
        }
        weighted_zv += g() * (*weight * resp.z_v[k]);
    }

    // leg c (C): the same z_v[k], weighted, recompose to the target.
    if weighted_zv != commit.a_c + *statement.target * x {
        return Err(Error::Verification("linking sigma: C leg failed"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::many_single_char_names
    )]

    use super::*;
    use crate::limbs::LIMB_MODULUS;
    use crate::pedersen::Commitment;
    use crate::transcript::Transcript;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const TEST_NONCE_BINDING: &[u8] = b"linking-test-nonce-binding";

    /// A consistent statement: random limb witnesses, the `ElGamal` ciphertexts
    /// and commitments they induce, and the recomposed target `C`.
    struct Statement {
        v: Vec<Scalar>,
        r: Vec<Scalar>,
        s: Vec<Scalar>,
        pk: ProjectivePoint,
        ciphertexts: Vec<LimbCiphertext>,
        com: Vec<ProjectivePoint>,
        c: ProjectivePoint,
    }

    fn statement(rng: &mut StdRng) -> Statement {
        let sk = Scalar::random(&mut *rng);
        let pk = g() * sk;
        let weights = limb_weights();
        let mut v = Vec::new();
        let mut r = Vec::new();
        let mut s = Vec::new();
        let mut ciphertexts = Vec::new();
        let mut com = Vec::new();
        let mut c = ProjectivePoint::IDENTITY;
        for weight in &weights {
            // limb value in [0, 2^ℓ)
            let vk = Scalar::from(rng.next_u64() % LIMB_MODULUS);
            let rk = Scalar::random(&mut *rng);
            let sk_blind = Scalar::random(&mut *rng);
            ciphertexts.push(LimbCiphertext {
                e: g() * rk,
                d: g() * vk + pk * rk,
            });
            com.push(Commitment::point_of(vk, sk_blind));
            c += g() * (*weight * vk);
            v.push(vk);
            r.push(rk);
            s.push(sk_blind);
        }
        Statement {
            v,
            r,
            s,
            pk,
            ciphertexts,
            com,
            c,
        }
    }

    impl Statement {
        fn stmt(&self) -> LinkingStatement<'_> {
            LinkingStatement {
                ciphertexts: &self.ciphertexts,
                commitments: &self.com,
                target: &self.c,
                pk: &self.pk,
            }
        }
    }

    fn challenge_for(commit: &LinkingCommitment) -> Scalar {
        let mut t = Transcript::new();
        for p in &commit.a_e {
            t.absorb_point(p);
        }
        for p in &commit.a_d {
            t.absorb_point(p);
        }
        for p in &commit.a_com {
            t.absorb_point(p);
        }
        t.absorb_point(&commit.a_c);
        t.challenge()
    }

    use rand::RngCore;

    struct RepeatingRng {
        byte: u8,
    }

    impl RngCore for RepeatingRng {
        fn next_u32(&mut self) -> u32 {
            u32::from_le_bytes([self.byte; 4])
        }

        fn next_u64(&mut self) -> u64 {
            u64::from_le_bytes([self.byte; 8])
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.fill(self.byte);
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    impl rand_core::CryptoRng for RepeatingRng {}

    #[test]
    fn honest_linking_verifies() {
        let mut rng = StdRng::seed_from_u64(0x11_44_C0_01);
        for _ in 0..16 {
            let st = statement(&mut rng);
            let (commit, state) =
                linking_commit(&st.v, &st.r, &st.s, &st.pk, TEST_NONCE_BINDING, &mut rng).unwrap();
            let x = challenge_for(&commit);
            let resp = state.respond(x);
            assert!(linking_verify(&st.stmt(), &commit, &resp, x).is_ok());
        }
    }

    #[test]
    fn repeated_rng_output_does_not_reuse_linking_value_nonce() {
        let mut rng = StdRng::seed_from_u64(0x11_44_C0_09);
        let st = statement(&mut rng);

        let mut rng_a = RepeatingRng { byte: 0x42 };
        let (_commit_a, state_a) = linking_commit(
            &st.v,
            &st.r,
            &st.s,
            &st.pk,
            b"linking-test-context-a",
            &mut rng_a,
        )
        .unwrap();
        let x_a = Scalar::from(17u64);
        let resp_a = state_a.respond(x_a);

        let mut rng_b = RepeatingRng { byte: 0x42 };
        let (_commit_b, state_b) = linking_commit(
            &st.v,
            &st.r,
            &st.s,
            &st.pk,
            b"linking-test-context-b",
            &mut rng_b,
        )
        .unwrap();
        let x_b = Scalar::from(29u64);
        let resp_b = state_b.respond(x_b);

        let recovered =
            (resp_a.z_v[0] - resp_b.z_v[0]) * Option::<Scalar>::from((x_a - x_b).invert()).unwrap();
        assert_ne!(
            recovered, st.v[0],
            "reused linking alpha reveals the value limb across distinct challenges"
        );
    }

    #[test]
    fn wrong_challenge_rejected() {
        let mut rng = StdRng::seed_from_u64(0x11_44_C0_02);
        let st = statement(&mut rng);
        let (commit, state) =
            linking_commit(&st.v, &st.r, &st.s, &st.pk, TEST_NONCE_BINDING, &mut rng).unwrap();
        let x = challenge_for(&commit);
        let resp = state.respond(x);
        assert!(matches!(
            linking_verify(&st.stmt(), &commit, &resp, x + Scalar::ONE),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn prover_state_is_explicitly_zeroizable() {
        static_assertions::assert_impl_all!(
            LinkingProverState: zeroize::Zeroize, zeroize::ZeroizeOnDrop
        );
        static_assertions::assert_not_impl_any!(LinkingProverState: Clone, Copy);
    }

    #[test]
    fn malformed_e_rejected() {
        // The ransom: publish E_k = (r_k+δ)·G while D_k stays v_k·G + r_k·pk, so
        // a decryptor computes D_k − sk·E_k = (v_k − δ·sk)·G ≠ v_k·G. The E_k leg
        // (shared z_r) cannot verify against the tampered E_k.
        let mut rng = StdRng::seed_from_u64(0x11_44_C0_03);
        let mut st = statement(&mut rng);
        let (commit, state) =
            linking_commit(&st.v, &st.r, &st.s, &st.pk, TEST_NONCE_BINDING, &mut rng).unwrap();
        let x = challenge_for(&commit);
        let resp = state.respond(x);
        st.ciphertexts[0].e += g(); // E_0 := (r_0 + 1)·G
        assert!(matches!(
            linking_verify(&st.stmt(), &commit, &resp, x),
            Err(Error::Verification("linking sigma: E_k leg failed"))
        ));
    }

    #[test]
    fn wrong_target_c_rejected() {
        // A package whose limbs recompose to m' ≠ m: C is replaced by m'·G, so
        // the weighted-z_v leg (leg c) fails — the wrong-scalar poison guard.
        let mut rng = StdRng::seed_from_u64(0x11_44_C0_04);
        let st = statement(&mut rng);
        let (commit, state) =
            linking_commit(&st.v, &st.r, &st.s, &st.pk, TEST_NONCE_BINDING, &mut rng).unwrap();
        let x = challenge_for(&commit);
        let resp = state.respond(x);
        let wrong_c = st.c + g(); // m' = m + 1
        assert!(matches!(
            linking_verify(
                &LinkingStatement {
                    ciphertexts: &st.ciphertexts,
                    commitments: &st.com,
                    target: &wrong_c,
                    pk: &st.pk,
                },
                &commit,
                &resp,
                x,
            ),
            Err(Error::Verification("linking sigma: C leg failed"))
        ));
    }

    #[test]
    fn mismatched_value_across_legs_rejected() {
        // Com_k opens to a different value than D_k encrypts: the shared z_v[k]
        // cannot satisfy both the D_k and Com_k legs at once.
        let mut rng = StdRng::seed_from_u64(0x11_44_C0_05);
        let mut st = statement(&mut rng);
        let (commit, state) =
            linking_commit(&st.v, &st.r, &st.s, &st.pk, TEST_NONCE_BINDING, &mut rng).unwrap();
        let x = challenge_for(&commit);
        let resp = state.respond(x);
        // Re-commit limb 0 to v_0 + 1 under the same blinding.
        st.com[0] = Commitment::point_of(st.v[0] + Scalar::ONE, st.s[0]);
        assert!(matches!(
            linking_verify(&st.stmt(), &commit, &resp, x),
            Err(Error::Verification("linking sigma: Com_k leg failed"))
        ));
    }

    #[test]
    fn tampered_response_rejected() {
        let mut rng = StdRng::seed_from_u64(0x11_44_C0_06);
        let st = statement(&mut rng);
        let (commit, state) =
            linking_commit(&st.v, &st.r, &st.s, &st.pk, TEST_NONCE_BINDING, &mut rng).unwrap();
        let x = challenge_for(&commit);
        let mut resp = state.respond(x);
        resp.z_v[0] += Scalar::ONE;
        assert!(matches!(
            linking_verify(&st.stmt(), &commit, &resp, x),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn wrong_witness_count_rejected_at_commit() {
        let mut rng = StdRng::seed_from_u64(0x11_44_C0_07);
        let full = vec![Scalar::ONE; LIMB_COUNT];
        let short = vec![Scalar::ONE; LIMB_COUNT - 1];
        let pk = g();
        assert!(matches!(
            linking_commit(&short, &full, &full, &pk, TEST_NONCE_BINDING, &mut rng),
            Err(Error::DegenerateInput(_))
        ));
        assert!(matches!(
            linking_commit(&full, &short, &full, &pk, TEST_NONCE_BINDING, &mut rng),
            Err(Error::DegenerateInput(_))
        ));
        assert!(matches!(
            linking_commit(&full, &full, &short, &pk, TEST_NONCE_BINDING, &mut rng),
            Err(Error::DegenerateInput(_))
        ));
    }

    #[test]
    fn verify_shape_mismatch_rejected() {
        let mut rng = StdRng::seed_from_u64(0x11_44_C0_08);
        let st = statement(&mut rng);
        let (commit, state) =
            linking_commit(&st.v, &st.r, &st.s, &st.pk, TEST_NONCE_BINDING, &mut rng).unwrap();
        let x = challenge_for(&commit);
        let resp = state.respond(x);
        // A short ciphertext slice trips the verifier's own shape guard.
        assert!(matches!(
            linking_verify(
                &LinkingStatement {
                    ciphertexts: &st.ciphertexts[..LIMB_COUNT - 1],
                    commitments: &st.com,
                    target: &st.c,
                    pk: &st.pk,
                },
                &commit,
                &resp,
                x,
            ),
            Err(Error::Verification("linking sigma: shape mismatch"))
        ));
    }
}
