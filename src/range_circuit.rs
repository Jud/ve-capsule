//! Aggregated BP++ reciprocal range proof at the frozen capsule shape (§4.1).
//!
//! One proof binds all `k = 32` committed values — 22 limbs in `[0, 2^24)`
//! (6 base-16 digits each) and 10 carries in `[0, 2)` — to the shared
//! Pedersen commitments `V_w = val_w·G + blind_w·H`. The construction is the
//! reciprocal-form arithmetic circuit protocol of [BP++ §5.3, §6.3] in the
//! shared-multiplicity layout of §6.4.2, normative per the soundness doc
//! (eprint 2022/510 rev. 2023-07-17 + the CypherStack-corrected lemmas).
//!
//! Flights (soundness doc §3 items 19–22): `C_L` (digits in the norm slot,
//! the 16 multiplicities in the linear slot) and `C_O` (blinding) → `α` →
//! `C_R` (reciprocals `1/(α + d)`) → `ρ, λ, β, δ` → `C_S` → `τ` → the
//! weighted norm linear argument over the folded claim. Digits and
//! multiplicities are bound BEFORE `α` — the Lemma-1 precondition.
//!
//! The constraint system is never materialized as matrices: at this frozen
//! shape every `c⃗`-vector entry has a closed form (digit-recomposition rows,
//! zero-forcing rows for the unused value slots, and the two per-base pole
//! rows), computed identically by prover and verifier from `(α, λ, μ)`.
//!
//! The blinding error-term vector `r⃗_S` is not transcribed from the paper:
//! it is derived from the identity it must satisfy — `g(T) = f̂(T)` with no
//! `T³` term — by collecting the non-`T³` coefficients of
//! `⟨ĉ_r(T), δr⃗_O + Tr⃗_L + T²r⃗_R + T³r⃗_V⟩` and solving slot-wise (each
//! `r⃗_S` slot feeds exactly one power through `T^{-1}·ĉ_r`). The honest
//! `T³` coefficient of `f̂` vanishing IS Eq. 34; the prover asserts it.

#![allow(clippy::similar_names)]

use crate::error::Error;
use crate::generators::{g, generators_digest, gvec, hvec};
use crate::msm::{generator_mul, msm, msm_vartime_public};
use crate::norm_arg::{FoldChallenges, NormArg, NormProof};
use crate::transcript::Transcript;
use k256::elliptic_curve::Field;
use k256::elliptic_curve::ops::Invert;
use k256::{ProjectivePoint, Scalar};
use rand_core::{CryptoRng, RngCore};
use zeroize::{Zeroize, Zeroizing};

/// Committed values: 11 value limbs ‖ 11 complement limbs ‖ 10 carries.
pub const K: usize = 32;
/// Limb-valued entries (the first 22 of the frozen order).
pub const LIMB_VALUES: usize = 22;
/// Carry entries (the last 10).
pub const CARRIES: usize = 10;
/// Digits per limb (base 16, `16^6 = 2^24`).
pub const DIGITS_PER_LIMB: usize = 6;
/// Value-vector width: slot 0 = the value, slots 1–15 forced zero; the 16
/// shared multiplicities fill `l⃗_L` exactly.
pub const N_V: usize = 16;
/// Total digits = poles: `22·6 + 10`.
pub const N_P: usize = LIMB_VALUES * DIGITS_PER_LIMB + CARRIES;
/// Multiplication gates (one reciprocal product per digit) = n-side width.
pub const N_M: usize = N_P;
/// Shared multiplicities: 15 base-16 symbols (1..=15) + 1 base-2 symbol (1).
pub const N_O: usize = 16;
/// Linear rows: `k·N_v` v-aligned rows + 2 pole rows.
pub const N_L_ROWS: usize = K * N_V + 2;
/// l-side width before padding: `N_v + 7` (the padded width is [`H_PAD`]).
#[allow(dead_code)] // documents the §4.1 shape table; the code uses H_PAD
pub const L_WIDTH: usize = N_V + 7;
/// Frozen padded l-side width (h⃗ generator count).
pub const H_PAD: usize = 32;
/// Frozen padded n-side width (g⃗ generator count).
pub const G_PAD: usize = 256;
/// Frozen WNLA fold-round count (asserted against the derivation by test).
pub const FOLD_ROUNDS: usize = 6;
/// Frozen residual `l⃗` length.
pub const RESIDUAL_L: usize = 1;
/// Frozen residual `n⃗` length.
pub const RESIDUAL_N: usize = 4;

/// The frozen generator vectors, derived once (RFC 9380 per element) and
/// reused by every prove/verify; the digest is the §3 item-13a statement
/// field, KAT-pinned in the tests.
fn frozen_generators() -> &'static (Vec<ProjectivePoint>, Vec<ProjectivePoint>, [u8; 32]) {
    static GENS: std::sync::LazyLock<(Vec<ProjectivePoint>, Vec<ProjectivePoint>, [u8; 32])> =
        std::sync::LazyLock::new(|| {
            let g_vec = gvec(u16::try_from(G_PAD).unwrap_or(u16::MAX));
            let h_vec = hvec(u16::try_from(H_PAD).unwrap_or(u16::MAX));
            let digest = generators_digest(&g_vec, &h_vec);
            (g_vec, h_vec, digest)
        });
    &GENS
}

/// The 32-byte generators digest absorbed as soundness-doc §3 item 13a.
#[must_use]
pub fn frozen_generators_digest() -> [u8; 32] {
    frozen_generators().2
}

const BASE16_DIGITS: usize = LIMB_VALUES * DIGITS_PER_LIMB;

/// The wire artifact: 4 flight points + the norm-argument folds
/// (16 points + 5 scalars = 688 B at the frozen shape).
#[derive(Clone, Debug)]
pub struct RangeProof {
    pub c_l: ProjectivePoint,
    pub c_o: ProjectivePoint,
    pub c_r: ProjectivePoint,
    pub c_s: ProjectivePoint,
    pub folds: NormProof,
}

/// Challenge source for the circuit flights (soundness doc §3 items 19–21);
/// fold challenges are [`FoldChallenges`]. Production wires the §2 ratchet;
/// tests may inject.
pub trait CircuitChallenges: FoldChallenges {
    /// Absorb `(C_L, C_O)` → squeeze `alpha`.
    fn alpha(&mut self, c_l: &ProjectivePoint, c_o: &ProjectivePoint) -> Scalar;
    /// Absorb `C_R` → squeeze `rho, lambda, beta, delta` (sequential).
    fn rho_lambda_beta_delta(&mut self, c_r: &ProjectivePoint) -> [Scalar; 4];
    /// Absorb `C_S` → squeeze `tau`.
    fn tau(&mut self, c_s: &ProjectivePoint) -> Scalar;
}

/// The production challenge wiring: the §2 ratchet over the capsule's master
/// transcript, absorbing the BP++ flights per soundness-doc §3 items 19–22.
pub struct TranscriptChallenges<'a>(pub &'a mut Transcript);

impl FoldChallenges for TranscriptChallenges<'_> {
    fn gamma(
        &mut self,
        round: u16,
        _commitment: &ProjectivePoint,
        x: &ProjectivePoint,
        r: &ProjectivePoint,
        _l_len: usize,
        _n_len: usize,
    ) -> Scalar {
        // Item 22: absorb (i, X_i, R_i) → squeeze gamma ‖ BE16(i). The round
        // index is this verifier-side counter, never prover-supplied; the
        // running commitment and lengths are statement-derived (frozen shape)
        // and already bound, so they are not re-absorbed.
        self.0.absorb_u16(round);
        self.0.absorb_point(x);
        self.0.absorb_point(r);
        let mut label = b"gamma".to_vec();
        label.extend_from_slice(&round.to_be_bytes());
        self.0.squeeze(&label)
    }
}

impl CircuitChallenges for TranscriptChallenges<'_> {
    fn alpha(&mut self, c_l: &ProjectivePoint, c_o: &ProjectivePoint) -> Scalar {
        // Item 19: C_L then C_O — digits and multiplicities bound before α.
        self.0.absorb_point(c_l);
        self.0.absorb_point(c_o);
        self.0.squeeze(b"alpha")
    }

    fn rho_lambda_beta_delta(&mut self, c_r: &ProjectivePoint) -> [Scalar; 4] {
        // Item 20: C_R, then the four sequential ratcheted squeezes.
        self.0.absorb_point(c_r);
        [
            self.0.squeeze(b"rho"),
            self.0.squeeze(b"lambda"),
            self.0.squeeze(b"beta"),
            self.0.squeeze(b"delta"),
        ]
    }

    fn tau(&mut self, c_s: &ProjectivePoint) -> Scalar {
        // Item 21.
        self.0.absorb_point(c_s);
        self.0.squeeze(b"tau")
    }
}

/// Prover witness: the 32 committed `(value, blinding)` pairs in frozen
/// order. Limb values must lie in `[0, 2^24)`, carries in `[0, 2)`. Secret;
/// zeroized on drop.
#[derive(zeroize::Zeroize, zeroize::ZeroizeOnDrop)]
pub struct RangeWitness {
    pub values: [u32; K],
    pub blindings: [Scalar; K],
}

/// The base-16 digits of limb `value`, little-endian, exactly 6.
fn limb_digits(value: u32) -> [u64; DIGITS_PER_LIMB] {
    core::array::from_fn(|j| (u64::from(value) >> (4 * j)) & 0xF)
}

/// The frozen digit vector `w⃗_D`: 132 base-16 limb digits (value-major),
/// then the 10 carry digits.
fn digit_vector(values: &[u32; K]) -> Vec<u64> {
    let mut d = Vec::with_capacity(N_P);
    for &v in &values[..LIMB_VALUES] {
        d.extend_from_slice(&limb_digits(v));
    }
    for &c in &values[LIMB_VALUES..] {
        d.push(u64::from(c));
    }
    d
}

/// The 16 shared multiplicities: `m⃗[t] = #{base-16 digits == t+1}` for
/// `t ∈ [0, 15)`, then `m⃗[15] = #{carries == 1}`. Zero-symbol
/// multiplicities are implicit (Eq. 75's `X`-pole term).
fn multiplicities(digits: &[u64]) -> [u64; N_O] {
    let mut m = [0u64; N_O];
    for &d in &digits[..BASE16_DIGITS] {
        if d > 0 {
            m[usize::try_from(d).unwrap_or(usize::MAX) - 1] += 1;
        }
    }
    for &c in &digits[BASE16_DIGITS..] {
        if c > 0 {
            m[N_O - 1] += 1;
        }
    }
    m
}

/// Scalar powers `(1, x, x², …, x^{n-1})`.
fn powers(x: &Scalar, n: usize) -> Vec<Scalar> {
    let mut acc = Scalar::ONE;
    (0..n)
        .map(|_| {
            let v = acc;
            acc *= x;
            v
        })
        .collect()
}

/// A sparse polynomial in `T` with scalar coefficients, keyed by power
/// (range `[-2, 7]` suffices for the whole protocol). Zeroized on drop: the
/// prover's instances carry blinding-dependent coefficients (`f̂`, `REST`),
/// and scrubbing the public constraint instances too is harmless.
#[derive(Clone, Default)]
struct Poly {
    coeffs: std::collections::BTreeMap<i32, Scalar>,
}

impl Drop for Poly {
    fn drop(&mut self) {
        for v in self.coeffs.values_mut() {
            v.zeroize();
        }
    }
}

impl Poly {
    fn add(&mut self, power: i32, value: Scalar) {
        *self.coeffs.entry(power).or_insert(Scalar::ZERO) += value;
    }

    fn coeff(&self, power: i32) -> Scalar {
        self.coeffs.get(&power).copied().unwrap_or(Scalar::ZERO)
    }

    fn eval(&self, at: &Scalar, at_inv: &Scalar) -> Scalar {
        let mut acc = Scalar::ZERO;
        for (&p, &c) in &self.coeffs {
            let mut term = c;
            let (base, reps) = if p >= 0 { (at, p) } else { (at_inv, -p) };
            for _ in 0..reps {
                term *= base;
            }
            acc += term;
        }
        acc
    }
}

/// A vector-valued sparse polynomial: per-power scalar vectors of width `w`.
/// Zeroized on drop — the prover's instances hold witness vectors (`l̂`, `n⃗`).
struct VecPoly {
    width: usize,
    terms: Vec<(i32, Vec<Scalar>)>,
}

impl Drop for VecPoly {
    fn drop(&mut self) {
        for (_, v) in &mut self.terms {
            v.zeroize();
        }
    }
}

impl VecPoly {
    const fn new(width: usize) -> Self {
        Self {
            width,
            terms: Vec::new(),
        }
    }

    fn add(&mut self, power: i32, vec: Vec<Scalar>) {
        debug_assert!(vec.len() <= self.width);
        self.terms.push((power, vec));
    }

    fn eval(&self, at: &Scalar, at_inv: &Scalar) -> Vec<Scalar> {
        let mut acc = vec![Scalar::ZERO; self.width];
        for (p, vec) in &self.terms {
            let mut scale = Scalar::ONE;
            let (base, reps) = if *p >= 0 { (at, *p) } else { (at_inv, -*p) };
            for _ in 0..reps {
                scale *= base;
            }
            for (slot, v) in vec.iter().enumerate() {
                acc[slot] += *v * scale;
            }
        }
        acc
    }

    /// `⟨self, other⟩_w` as a polynomial in `T`: the weighted inner product
    /// of every pair of terms, powers added. `weight = ONE` gives the plain
    /// inner product.
    fn inner(&self, other: &Self, weight: &Scalar) -> Poly {
        let mut out = Poly::default();
        for (pa, va) in &self.terms {
            for (pb, vb) in &other.terms {
                let mut exp = *weight;
                let mut acc = Scalar::ZERO;
                let len = va.len().max(vb.len());
                for i in 0..len {
                    let a = va.get(i).copied().unwrap_or(Scalar::ZERO);
                    let b = vb.get(i).copied().unwrap_or(Scalar::ZERO);
                    acc += a * b * exp;
                    if *weight != Scalar::ONE {
                        exp *= weight;
                    }
                }
                out.add(pa + pb, acc);
            }
        }
        out
    }
}

/// The public constraint vectors at challenge point `(α, λ, μ)` — closed
/// forms of `c⃗_{n,L}`, `c⃗_{n,R}`, `c⃗_{l,L}`, `⟨λ⃗, a⃗_l⟩`, `⟨μ⃗, a⃗_m⟩`
/// for the frozen shape. Computed identically by prover and verifier.
struct Constraints {
    c_n_l: Vec<Scalar>,
    c_n_r: Vec<Scalar>,
    c_l_l: Vec<Scalar>,
    lambda_a_l: Scalar,
    mu_a_m: Scalar,
}

/// `Error::Verification` if any public denominator `α + s` (symbols
/// `s ∈ {0..15}`) vanishes — the pole-handling obligation.
fn constraints(alpha: &Scalar, lambda: &Scalar, mu: &Scalar) -> Result<Constraints, Error> {
    // Public denominators: α (zero symbol) and α + s for s ∈ [1, 16).
    let mut denom_inv = Vec::with_capacity(16);
    for s in 0..16u64 {
        let d = *alpha + Scalar::from(s);
        let inv = Option::<Scalar>::from(d.invert()).ok_or(Error::Verification(
            "range pole challenge hit a symbol denominator",
        ))?;
        denom_inv.push(inv);
    }

    let lambda_pows = powers(lambda, N_L_ROWS);
    let mu_pows = powers(mu, N_M + 1);
    let mu_inv = Option::<Scalar>::from(mu.invert())
        .ok_or(Error::Verification("range mu challenge is zero"))?;
    let mu_inv_pows = powers(&mu_inv, N_M + 1);

    // c_n,L[i]: the digit-recomposition row of digit i's value, scaled
    // mu^{-(i+1)}. Row 16·w carries coefficient −weight(i) at digit i.
    let mut c_n_l = Vec::with_capacity(N_M);
    // c_n,R[i]: pole-row coefficient (−λ^{pole(i)}) scaled mu^{-(i+1)},
    // plus the mult-gate coefficient −α.
    let mut c_n_r = Vec::with_capacity(N_M);
    let pole16 = lambda_pows[K * N_V];
    let pole2 = lambda_pows[K * N_V + 1];
    for i in 0..N_M {
        let (value_index, weight, pole) = if i < BASE16_DIGITS {
            let w = i / DIGITS_PER_LIMB;
            let j = i % DIGITS_PER_LIMB;
            (w, Scalar::from(1u64 << (4 * j)), pole16)
        } else {
            (LIMB_VALUES + (i - BASE16_DIGITS), Scalar::ONE, pole2)
        };
        let row_lambda = lambda_pows[N_V * value_index];
        c_n_l.push(row_lambda * weight * mu_inv_pows[i + 1]);
        c_n_r.push(pole * mu_inv_pows[i + 1] + alpha);
    }

    // c_l,L[t]: the multiplicity-t coefficient of the E'-matching
    // derivation — the zero pole minus the symbol pole:
    // λ^{pole} · (1/α − 1/(α+sym)).
    let mut c_l_l = Vec::with_capacity(N_O);
    for t in 0..N_O {
        let (pole, sym_inv) = if t < N_O - 1 {
            (pole16, denom_inv[t + 1])
        } else {
            (pole2, denom_inv[1])
        };
        c_l_l.push(pole * (denom_inv[0] - sym_inv));
    }

    // ⟨λ⃗, a⃗_l⟩: the pole rows' constants n̂_b/α.
    let lambda_a_l = (pole16 * Scalar::from(BASE16_DIGITS as u64)
        + pole2 * Scalar::from(CARRIES as u64))
        * denom_inv[0];
    // ⟨μ⃗, a⃗_m⟩ = Σ_{i=1..N_M} μ^i (a⃗_m = 1⃗).
    let mut mu_a_m = Scalar::ZERO;
    for p in mu_pows.iter().skip(1) {
        mu_a_m += p;
    }

    Ok(Constraints {
        c_n_l,
        c_n_r,
        c_l_l,
        lambda_a_l,
        mu_a_m,
    })
}

/// `ĉ_r(T)` slot powers: slot s of the blinding region feeds `T^{CR_POWS[s]}`
/// with coefficient β (slot 0 feeds `T⁰` with coefficient 1).
const CR_POWS: [i32; 8] = [0, -1, 1, 2, 3, 5, 6, 7];

/// `p⃗_n(T) = T²c⃗_{n,L} + Tc⃗_{n,R}` (`c⃗_{n,O} = 0` at this shape).
fn p_n_poly(cons: &Constraints) -> VecPoly {
    let mut p = VecPoly::new(N_M);
    p.add(2, cons.c_n_l.clone());
    p.add(1, cons.c_n_r.clone());
    p
}

/// `p_s(T) = |p⃗_n(T)|²_μ + 2(⟨λ⃗,a⃗_l⟩ + ⟨μ⃗,a⃗_m⟩)T³`.
fn p_s_poly(cons: &Constraints, mu: &Scalar) -> Poly {
    let p_n = p_n_poly(cons);
    let mut p_s = p_n.inner(&p_n, mu);
    p_s.add(3, (cons.lambda_a_l + cons.mu_a_m).double());
    p_s
}

/// `ĉ_l(T) = 2T²c⃗_{l,L} − e_{N_v}(λ)_{1:}` (negative tail — it pairs the
/// zero-forcing rows; the `c⃗_{l,R}`/`c⃗_{l,O}` terms
/// vanish at this shape; `f_m = 0`, `f_l = 1`).
fn c_l_poly(cons: &Constraints, lambda: &Scalar) -> VecPoly {
    let mut c = VecPoly::new(N_V);
    c.add(
        2,
        cons.c_l_l
            .iter()
            .map(k256::elliptic_curve::Field::double)
            .collect(),
    );
    // e(λ)_{1:} zero-extended into the N_v-wide vector (last slot unused).
    let mut tail = powers(lambda, N_V);
    tail.remove(0);
    c.add(0, tail.iter().map(|v| -*v).collect());
    c
}

/// The l-side constraint vector for the WNLA at `τ`:
/// `c⃗(τ) = ĉ_r(τ)_{1:} ‖ ĉ_l(τ)`, zero-padded to [`H_PAD`].
fn wnla_c(cons: &Constraints, lambda: &Scalar, beta: &Scalar, tau: &Scalar) -> Vec<Scalar> {
    let tau_inv = tau.invert_vartime().unwrap_or(Scalar::ZERO);
    let mut c = Vec::with_capacity(H_PAD);
    for &p in &CR_POWS[1..] {
        let mut term = *beta;
        let (base, reps) = if p >= 0 { (tau, p) } else { (&tau_inv, -p) };
        for _ in 0..reps {
            term *= base;
        }
        c.push(term);
    }
    c.extend(c_l_poly(cons, lambda).eval(tau, &tau_inv));
    c.resize(H_PAD, Scalar::ZERO);
    c
}

/// The verifier-side commitment
/// `C(τ) = p_s(τ)G + ⟨p⃗_n(τ), g⃗⟩ + τ^{-1}C_S + δC_O + τC_L + τ²C_R + τ³V̂`.
#[allow(clippy::too_many_arguments)]
fn folded_commitment(
    cons: &Constraints,
    proof_points: (
        &ProjectivePoint,
        &ProjectivePoint,
        &ProjectivePoint,
        &ProjectivePoint,
    ),
    v_hat: &ProjectivePoint,
    g_vec: &[ProjectivePoint],
    mu: &Scalar,
    delta: &Scalar,
    tau: &Scalar,
) -> ProjectivePoint {
    let (c_l, c_o, c_r, c_s) = proof_points;
    let tau_inv = tau.invert_vartime().unwrap_or(Scalar::ZERO);
    let tau2 = tau * tau;
    let tau3 = tau2 * tau;
    let p_n = p_n_poly(cons).eval(tau, &tau_inv);
    let p_s = p_s_poly(cons, mu).eval(tau, &tau_inv);
    generator_mul(&p_s)
        + msm_vartime_public(g_vec, &p_n)
        + *c_s * tau_inv
        + *c_o * delta
        + *c_l * tau
        + *c_r * tau2
        + *v_hat * tau3
}

/// `V̂ = 2Σ_i λ^{N_v·i} V_i` (`f_l = 1`, `f_m = 0`).
fn v_hat_point(commitments: &[ProjectivePoint; K], lambda: &Scalar) -> ProjectivePoint {
    let lambda_pows = powers(lambda, K * N_V);
    let scalars: Vec<_> = (0..K).map(|i| lambda_pows[N_V * i]).collect();
    msm_vartime_public(commitments, &scalars).double()
}

/// Prove the aggregated range statement for `commitments` (which MUST open
/// as `values[w]·G + blindings[w]·H` — the caller seals them that way).
///
/// # Errors
///
/// [`Error::DegenerateInput`] on an out-of-range witness value;
/// [`Error::Verification`] on a vanishing pole or challenge denominator
/// (negligible for honest transcripts — there is nothing sound to resample
/// in Fiat–Shamir, so the prover errors out).
#[allow(clippy::too_many_lines)]
pub fn prove<C: CircuitChallenges, R: RngCore + CryptoRng>(
    commitments: &[ProjectivePoint; K],
    witness: &RangeWitness,
    challenges: &mut C,
    rng: &mut R,
) -> Result<RangeProof, Error> {
    for (w, &v) in witness.values.iter().enumerate() {
        let limit = if w < LIMB_VALUES { 1u64 << 24 } else { 2 };
        if u64::from(v) >= limit {
            return Err(Error::DegenerateInput("range witness value out of range"));
        }
    }

    // The digit expansion IS the secret (it reassembles to m's limbs);
    // zeroized on drop, as are the scalar forms handed to the prover body.
    let digits = Zeroizing::new(digit_vector(&witness.values));
    let mults = multiplicities(&digits);
    let digit_scalars: Zeroizing<Vec<Scalar>> =
        Zeroizing::new(digits.iter().map(|&d| Scalar::from(d)).collect());
    let mult_scalars: Zeroizing<Vec<Scalar>> =
        Zeroizing::new(mults.iter().map(|&m| Scalar::from(m)).collect());
    prove_inner(
        commitments,
        witness,
        &digit_scalars,
        &mult_scalars,
        challenges,
        rng,
        true,
    )
}

/// Adversarial-prover surface for the test suite: prove with INJECTED digit
/// and multiplicity vectors, skipping the honesty assertions — the strongest
/// thing a malicious prover can do within the protocol's message format. The
/// verifier must reject everything this produces from a false witness.
#[cfg(test)]
pub fn prove_with_witness_vectors<C: CircuitChallenges, R: RngCore + CryptoRng>(
    commitments: &[ProjectivePoint; K],
    witness: &RangeWitness,
    digit_scalars: &[Scalar],
    mult_scalars: &[Scalar],
    challenges: &mut C,
    rng: &mut R,
) -> Result<RangeProof, Error> {
    PAD_MASS.with(|m| m.set(None));
    prove_inner(
        commitments,
        witness,
        digit_scalars,
        mult_scalars,
        challenges,
        rng,
        false,
    )
}

#[cfg(test)]
thread_local! {
    /// Test-only padded-slot mass injection: `(l-slot, mass)` added to `C_S`
    /// and the folded witness — the §4.1 "padding region" demonstration.
    static PAD_MASS: std::cell::Cell<Option<(usize, Scalar)>> =
        const { std::cell::Cell::new(None) };
}

/// Prove with extra mass on a PADDED l-slot, self-consistently committed in
/// `C_S` (the only in-format way to occupy the padding region). The §4.1
/// benign-padding argument predicts this VERIFIES: the constraint vector is
/// zero on padded slots, so the mass is just unusual blinding.
#[cfg(test)]
pub fn prove_with_padded_mass<C: CircuitChallenges, R: RngCore + CryptoRng>(
    commitments: &[ProjectivePoint; K],
    witness: &RangeWitness,
    slot: usize,
    mass: Scalar,
    challenges: &mut C,
    rng: &mut R,
) -> Result<RangeProof, Error> {
    assert!((L_WIDTH..H_PAD).contains(&slot), "slot must be padding");
    let digits = digit_vector(&witness.values);
    let mults = multiplicities(&digits);
    let digit_scalars: Vec<Scalar> = digits.iter().map(|&d| Scalar::from(d)).collect();
    let mult_scalars: Vec<Scalar> = mults.iter().map(|&m| Scalar::from(m)).collect();
    PAD_MASS.with(|m| m.set(Some((slot, mass))));
    let out = prove_inner(
        commitments,
        witness,
        &digit_scalars,
        &mult_scalars,
        challenges,
        rng,
        false,
    );
    PAD_MASS.with(|m| m.set(None));
    out
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn prove_inner<C: CircuitChallenges, R: RngCore + CryptoRng>(
    commitments: &[ProjectivePoint; K],
    witness: &RangeWitness,
    digit_scalars: &[Scalar],
    mult_scalars: &[Scalar],
    challenges: &mut C,
    rng: &mut R,
    honest: bool,
) -> Result<RangeProof, Error> {
    let (g_vec, h_vec, _) = frozen_generators();

    // Blinding vectors with the paper's zero patterns (§5.3 CommitOL/CommitR;
    // the zero slots are exactly the T³ and >T⁶ feed-through positions).
    let rnd = |rng: &mut R| Scalar::random(&mut *rng);
    let r_o = Zeroizing::new([
        rnd(rng),
        rnd(rng),
        rnd(rng),
        rnd(rng),
        Scalar::ZERO,
        rnd(rng),
        rnd(rng),
        Scalar::ZERO,
    ]);
    let r_l = Zeroizing::new([
        rnd(rng),
        rnd(rng),
        rnd(rng),
        Scalar::ZERO,
        rnd(rng),
        rnd(rng),
        Scalar::ZERO,
        Scalar::ZERO,
    ]);
    let r_r = Zeroizing::new([
        rnd(rng),
        rnd(rng),
        Scalar::ZERO,
        rnd(rng),
        rnd(rng),
        Scalar::ZERO,
        Scalar::ZERO,
        Scalar::ZERO,
    ]);

    // l_L = the multiplicities; l_O / l_R / n_O are empty at this layout.
    let l_l = Zeroizing::new(mult_scalars.to_vec());
    let n_l = Zeroizing::new(digit_scalars.to_vec());

    // C_X = r_{X,0}G + ⟨r_{X,1:} ‖ l_X, h⃗⟩ + ⟨n_X, g⃗⟩.
    let commit_flight = |r: &[Scalar; 8], l: &[Scalar], n: &[Scalar]| {
        let mut lh: Vec<Scalar> = r[1..].to_vec();
        lh.extend_from_slice(l);
        generator_mul(&r[0]) + msm(h_vec, &lh) + msm(g_vec, n)
    };
    let c_l_pt = commit_flight(&r_l, &l_l, &n_l);
    let c_o_pt = commit_flight(&r_o, &[], &[]);

    let alpha = challenges.alpha(&c_l_pt, &c_o_pt);

    // Reciprocals r_i = 1/(α + d_i); a vanishing denominator is a protocol
    // error (negligible — α is bound to the committed digits).
    let mut recips = Zeroizing::new(Vec::with_capacity(N_M));
    for d in digit_scalars {
        let den = alpha + d;
        let inv = Option::<Scalar>::from(den.invert())
            .ok_or(Error::Verification("range prover hit a reciprocal pole"))?;
        recips.push(inv);
    }
    let n_r = recips;
    let c_r_pt = commit_flight(&r_r, &[], &n_r);

    let [rho, lambda, beta, delta] = challenges.rho_lambda_beta_delta(&c_r_pt);
    let mu = rho * rho;
    let cons = constraints(&alpha, &lambda, &mu)?;

    // Random blinding vectors for the S flight.
    let l_s: Zeroizing<Vec<Scalar>> = Zeroizing::new((0..N_V).map(|_| rnd(rng)).collect());
    let n_s: Zeroizing<Vec<Scalar>> = Zeroizing::new((0..N_M).map(|_| rnd(rng)).collect());

    // v̂ = 2Σ λ^{N_v i} v_{i,0}; r_V[1] = 2Σ λ^{N_v i} s_{V,i}.
    let lambda_pows = powers(&lambda, K * N_V);
    let mut v_hat = Scalar::ZERO;
    let mut s_hat = Scalar::ZERO;
    for i in 0..K {
        v_hat += lambda_pows[N_V * i] * Scalar::from(witness.values[i]);
        s_hat += lambda_pows[N_V * i] * witness.blindings[i];
    }
    v_hat = v_hat.double();
    s_hat = s_hat.double();
    let mut r_v = Zeroizing::new([Scalar::ZERO; 8]);
    r_v[1] = s_hat;
    s_hat.zeroize();

    // l̂(T) = T^{-1}l⃗_S + Tl⃗_L (l_O, l_R, v-tails all zero here);
    // n⃗(T) = T^{-1}n⃗_S + Tn⃗_L + T²n⃗_R + p⃗_n(T).
    let mut l_hat = VecPoly::new(N_V);
    l_hat.add(-1, l_s.to_vec());
    l_hat.add(1, l_l.to_vec());
    let mut n_poly = VecPoly::new(N_M);
    n_poly.add(-1, n_s.to_vec());
    n_poly.add(1, n_l.to_vec());
    n_poly.add(2, n_r.to_vec());
    n_poly.add(2, cons.c_n_l.clone());
    n_poly.add(1, cons.c_n_r.clone());

    // f̂(T) = p_s(T) + v̂T³ − ⟨ĉ_l(T), l̂(T)⟩ − |n⃗(T)|²_μ.
    let mut f_hat = p_s_poly(&cons, &mu);
    f_hat.add(3, v_hat);
    let c_l_t = c_l_poly(&cons, &lambda);
    let minus = c_l_t.inner(&l_hat, &Scalar::ONE);
    for (&p, &v) in &minus.coeffs {
        f_hat.add(p, -v);
    }
    let norm = n_poly.inner(&n_poly, &mu);
    for (&p, &v) in &norm.coeffs {
        f_hat.add(p, -v);
    }
    // Eq. 34: the value term vanishes for an honest witness. A failure here
    // is an implementation bug, not an input error. (The adversarial test
    // path proves false witnesses deliberately; its proofs must then fail
    // verification — that asymmetry is the soundness being tested.)
    debug_assert!(
        !honest || f_hat.coeff(3) == Scalar::ZERO,
        "BP++ value term must vanish"
    );

    // REST(T) = ⟨ĉ_r(T), δr⃗_O + Tr⃗_L + T²r⃗_R + T³r⃗_V⟩ — the blinding
    // feed-through; r⃗_S then cancels it against f̂ slot-wise.
    let beta_inv = Option::<Scalar>::from(beta.invert())
        .ok_or(Error::Verification("range beta challenge is zero"))?;
    let mut rest = Poly::default();
    for slot in 0..8 {
        let coeff = if slot == 0 { -Scalar::ONE } else { beta };
        let base_pow = CR_POWS[slot];
        for (shift, vec) in [(0, &r_o), (1, &r_l), (2, &r_r), (3, &r_v)] {
            let scale = if shift == 0 {
                delta * vec[slot]
            } else {
                vec[slot]
            };
            rest.add(base_pow + shift, coeff * scale);
        }
    }
    let mut r_s = Zeroizing::new([Scalar::ZERO; 8]);
    for slot in 0..8 {
        let p = CR_POWS[slot];
        // r_S slot feeds T^{p-1}·(coeff): solve coeff·r_S[slot] = f̂ − REST.
        let target = f_hat.coeff(p - 1) - rest.coeff(p - 1);
        r_s[slot] = if slot == 0 {
            -target
        } else {
            beta_inv * target
        };
    }
    debug_assert!(!honest || f_hat.coeff(3) - rest.coeff(3) == Scalar::ZERO);

    #[allow(unused_mut)]
    let mut c_s_pt = commit_flight(&r_s, &l_s, &n_s);
    #[cfg(test)]
    if let Some((slot, mass)) = PAD_MASS.with(std::cell::Cell::get) {
        c_s_pt += h_vec[slot] * mass;
    }
    let tau = challenges.tau(&c_s_pt);
    let tau_inv = Option::<Scalar>::from(tau.invert())
        .ok_or(Error::Verification("range tau challenge is zero"))?;

    // The WNLA witness at τ: l(τ) = r(τ)_{1:} ‖ l̂(τ), n(τ); zero-padded.
    let mut r_poly = VecPoly::new(8);
    r_poly.add(-1, r_s.to_vec());
    r_poly.add(0, r_o.iter().map(|v| *v * delta).collect());
    r_poly.add(1, r_l.to_vec());
    r_poly.add(2, r_r.to_vec());
    r_poly.add(3, r_v.to_vec());
    let r_at = Zeroizing::new(r_poly.eval(&tau, &tau_inv));

    let mut l_final: Vec<Scalar> = r_at[1..].to_vec();
    l_final.extend(l_hat.eval(&tau, &tau_inv));
    l_final.resize(H_PAD, Scalar::ZERO);
    #[cfg(test)]
    if let Some((slot, mass)) = PAD_MASS.with(std::cell::Cell::get) {
        l_final[slot] += tau_inv * mass;
    }
    let mut n_final = n_poly.eval(&tau, &tau_inv);
    n_final.resize(G_PAD, Scalar::ZERO);

    let c_vec = wnla_c(&cons, &lambda, &beta, &tau);
    let arg = NormArg {
        g: g(),
        g_vec: g_vec.clone(),
        h_vec: h_vec.clone(),
        c: c_vec,
        rho,
        mu,
    };
    let v_hat_pt = v_hat_point(commitments, &lambda);
    let commitment = folded_commitment(
        &cons,
        (&c_l_pt, &c_o_pt, &c_r_pt, &c_s_pt),
        &v_hat_pt,
        &arg.g_vec,
        &mu,
        &delta,
        &tau,
    );
    debug_assert!(
        !honest || commitment == arg.commit(&l_final, &n_final),
        "BP++ folded commitment must open to the folded witness"
    );
    let folds = arg.prove(&commitment, challenges, l_final, n_final)?;

    Ok(RangeProof {
        c_l: c_l_pt,
        c_o: c_o_pt,
        c_r: c_r_pt,
        c_s: c_s_pt,
        folds,
    })
}

/// Verify an aggregated range proof against the 32 statement commitments.
///
/// # Errors
///
/// [`Error::Verification`] on an identity flight point, a vanishing pole or
/// challenge denominator, or a failed norm-argument check.
pub fn verify<C: CircuitChallenges>(
    commitments: &[ProjectivePoint; K],
    proof: &RangeProof,
    challenges: &mut C,
) -> Result<(), Error> {
    for p in [&proof.c_l, &proof.c_o, &proof.c_r, &proof.c_s] {
        if p == &ProjectivePoint::IDENTITY {
            return Err(Error::Verification("range identity flight point"));
        }
    }

    let alpha = challenges.alpha(&proof.c_l, &proof.c_o);
    let [rho, lambda, beta, delta] = challenges.rho_lambda_beta_delta(&proof.c_r);
    let mu = rho * rho;
    let cons = constraints(&alpha, &lambda, &mu)?;
    let tau = challenges.tau(&proof.c_s);
    if bool::from(tau.is_zero()) || bool::from(beta.is_zero()) {
        return Err(Error::Verification("range challenge is zero"));
    }

    let (g_vec, h_vec, _) = frozen_generators();
    let c_vec = wnla_c(&cons, &lambda, &beta, &tau);
    let v_hat_pt = v_hat_point(commitments, &lambda);
    let commitment = folded_commitment(
        &cons,
        (&proof.c_l, &proof.c_o, &proof.c_r, &proof.c_s),
        &v_hat_pt,
        g_vec,
        &mu,
        &delta,
        &tau,
    );
    let arg = NormArg {
        g: g(),
        g_vec: g_vec.clone(),
        h_vec: h_vec.clone(),
        c: c_vec,
        rho,
        mu,
    };
    arg.verify(&commitment, challenges, &proof.folds)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::*;
    use crate::generators::h;
    use crate::transcript::Transcript;
    use k256::elliptic_curve::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    struct Ratchet(Transcript);

    impl FoldChallenges for Ratchet {
        fn gamma(
            &mut self,
            round: u16,
            commitment: &ProjectivePoint,
            x: &ProjectivePoint,
            r: &ProjectivePoint,
            l_len: usize,
            n_len: usize,
        ) -> Scalar {
            TranscriptChallenges(&mut self.0).gamma(round, commitment, x, r, l_len, n_len)
        }
    }

    impl CircuitChallenges for Ratchet {
        fn alpha(&mut self, c_l: &ProjectivePoint, c_o: &ProjectivePoint) -> Scalar {
            TranscriptChallenges(&mut self.0).alpha(c_l, c_o)
        }

        fn rho_lambda_beta_delta(&mut self, c_r: &ProjectivePoint) -> [Scalar; 4] {
            TranscriptChallenges(&mut self.0).rho_lambda_beta_delta(c_r)
        }

        fn tau(&mut self, c_s: &ProjectivePoint) -> Scalar {
            TranscriptChallenges(&mut self.0).tau(c_s)
        }
    }

    fn ratchet() -> Ratchet {
        Ratchet(Transcript::new())
    }

    // ===================== adversarial suite (unit 5) =====================

    /// The strongest in-format attack: prove with injected digit/multiplicity
    /// vectors against HONEST commitments, then run the honest verifier.
    fn assert_false_witness_rejected(
        label: &str,
        mutate: impl FnOnce(&mut Vec<Scalar>, &mut Vec<Scalar>),
        seed: u64,
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let w = witness(seed ^ 0xFFFF);
        let v = commitments_for(&w);
        let digits = super::digit_vector(&w.values);
        let mults = super::multiplicities(&digits);
        let mut digit_scalars: Vec<Scalar> = digits.iter().map(|&d| Scalar::from(d)).collect();
        let mut mult_scalars: Vec<Scalar> = mults.iter().map(|&m| Scalar::from(m)).collect();
        mutate(&mut digit_scalars, &mut mult_scalars);
        let proof = prove_with_witness_vectors(
            &v,
            &w,
            &digit_scalars,
            &mult_scalars,
            &mut ratchet(),
            &mut rng,
        )
        .unwrap();
        assert!(
            verify(&v, &proof, &mut ratchet()).is_err(),
            "{label}: false witness was accepted"
        );
    }

    #[test]
    fn out_of_set_digit_rejected() {
        // The core reciprocal-soundness test: re-encode a limb with a digit
        // outside [0, 16) while keeping the recomposed value identical
        // (d_0 += 16, d_1 -= 1), so every linear row still holds — only the
        // set-membership (pole) equation can catch it.
        assert_false_witness_rejected(
            "out-of-set digit",
            |digits, _| {
                digits[0] += Scalar::from(16u64);
                digits[1] -= Scalar::ONE;
            },
            0xAD_01,
        );

        // Discriminator: at an arbitrary challenge point the mutated digit
        // vector satisfies every recomposition row exactly, while the pole
        // equation residual is nonzero — the rejection above is the pole
        // equation's, not a linear-row side effect.
        let w = witness(0xAD_01 ^ 0xFFFF);
        let digits = super::digit_vector(&w.values);
        let mults = super::multiplicities(&digits);
        let mut ds: Vec<Scalar> = digits.iter().map(|&d| Scalar::from(d)).collect();
        ds[0] += Scalar::from(16u64);
        ds[1] -= Scalar::ONE;
        for (i, &val) in w.values.iter().take(LIMB_VALUES).enumerate() {
            let recomposed: Scalar = (0..DIGITS_PER_LIMB)
                .map(|j| ds[DIGITS_PER_LIMB * i + j] * Scalar::from(1u64 << (4 * j)))
                .sum();
            assert_eq!(recomposed, Scalar::from(val), "recomposition row {i}");
        }
        let alpha = Scalar::from(0x1234_5678u64);
        let lhs: Scalar = ds[..BASE16_DIGITS]
            .iter()
            .map(|d| (alpha + d).invert().unwrap())
            .sum();
        let total: Scalar = mults.iter().take(N_O - 1).map(|&m| Scalar::from(m)).sum();
        let rhs: Scalar = (Scalar::from(BASE16_DIGITS as u64) - total) * alpha.invert().unwrap()
            + (0..N_O - 1)
                .map(|t| {
                    Scalar::from(mults[t]) * (alpha + Scalar::from(t as u64 + 1)).invert().unwrap()
                })
                .sum::<Scalar>();
        assert_ne!(lhs, rhs, "pole equation must be the violated constraint");
    }

    #[test]
    fn non_boolean_carry_rejected_by_pole_equation() {
        // A carry committed to 2 with a CONSISTENT digit (recomposition row
        // holds: digit == committed value) — only the base-2 pole equation
        // can reject it, exactly the booleanity that bounds §4.2's window
        // argument. prove() refuses such a witness; the injected-vector
        // surface is the malicious prover.
        let mut rng = StdRng::seed_from_u64(0xAD_30);
        let mut w = witness(0xAD_31);
        w.values[LIMB_VALUES] = 2; // commitment honestly opens to 2
        let v = commitments_for(&w);
        let digits = super::digit_vector(&w.values);
        let mults = super::multiplicities(&digits);
        let digit_scalars: Vec<Scalar> = digits.iter().map(|&d| Scalar::from(d)).collect();
        let mult_scalars: Vec<Scalar> = mults.iter().map(|&m| Scalar::from(m)).collect();
        let proof = prove_with_witness_vectors(
            &v,
            &w,
            &digit_scalars,
            &mult_scalars,
            &mut ratchet(),
            &mut rng,
        )
        .unwrap();
        assert!(
            verify(&v, &proof, &mut ratchet()).is_err(),
            "non-boolean carry was accepted"
        );
    }

    #[test]
    fn multiplicity_count_off_by_one_rejected() {
        assert_false_witness_rejected(
            "multiplicity +1",
            |_, mults| mults[3] += Scalar::ONE,
            0xAD_02,
        );
    }

    #[test]
    fn field_wrapped_multiplicity_rejected() {
        // A "negative" multiplicity (n − 1 ≡ −1): the rational identity is
        // over formal counts; a wrapped count must not satisfy it.
        assert_false_witness_rejected(
            "wrapped multiplicity",
            |_, mults| mults[5] -= Scalar::ONE.double(),
            0xAD_03,
        );
    }

    #[test]
    fn base_partition_swap_rejected() {
        // Move one count from a base-16 symbol slot into the base-2 slot:
        // totals stay plausible, but each group's pole equation breaks.
        assert_false_witness_rejected(
            "base-partition swap",
            |_, mults| {
                mults[0] -= Scalar::ONE;
                mults[N_O - 1] += Scalar::ONE;
            },
            0xAD_04,
        );
    }

    #[test]
    fn zero_implied_count_mismatch_rejected() {
        // Claim one more zero digit than exists by shrinking a nonzero
        // symbol's count: the n̂/α (implicit-zero) term no longer balances.
        assert_false_witness_rejected(
            "zero-implied count",
            |_, mults| mults[1] -= Scalar::ONE,
            0xAD_05,
        );
    }

    #[test]
    fn pole_challenge_rejected() {
        // The verifier obligation: a challenge α hitting a symbol denominator
        // (α = −s) is rejected, not inverted-through or panicked on.
        let lambda = Scalar::from(7u64);
        let mu = Scalar::from(9u64);
        for s in 0..16u64 {
            let alpha = -Scalar::from(s);
            assert_eq!(
                super::constraints(&alpha, &lambda, &mu).err(),
                Some(Error::Verification(
                    "range pole challenge hit a symbol denominator"
                )),
                "alpha = -{s}"
            );
        }
    }

    /// A challenge source that skips one absorption (Frozen-Heart) or swaps
    /// squeeze labels — the implementation-bug classes the spec pins.
    struct Broken {
        t: Transcript,
        mode: &'static str,
        skip_round: u16,
    }

    impl FoldChallenges for Broken {
        fn gamma(
            &mut self,
            round: u16,
            _c: &ProjectivePoint,
            x: &ProjectivePoint,
            r: &ProjectivePoint,
            _l: usize,
            _n: usize,
        ) -> Scalar {
            match self.mode {
                "skip-fold" if round == self.skip_round => {}
                "skip-round-index" => {
                    self.t.absorb_point(x);
                    self.t.absorb_point(r);
                }
                _ => {
                    self.t.absorb_u16(round);
                    self.t.absorb_point(x);
                    self.t.absorb_point(r);
                }
            }
            if self.mode == "const-gamma-label" {
                self.t.squeeze(b"gamma")
            } else {
                let mut label = b"gamma".to_vec();
                label.extend_from_slice(&round.to_be_bytes());
                self.t.squeeze(&label)
            }
        }
    }

    impl CircuitChallenges for Broken {
        fn alpha(&mut self, c_l: &ProjectivePoint, c_o: &ProjectivePoint) -> Scalar {
            match self.mode {
                "skip-cl" => self.t.absorb_point(c_o),
                "skip-co" => self.t.absorb_point(c_l),
                _ => {
                    self.t.absorb_point(c_l);
                    self.t.absorb_point(c_o);
                }
            }
            self.t.squeeze(b"alpha")
        }

        fn rho_lambda_beta_delta(&mut self, c_r: &ProjectivePoint) -> [Scalar; 4] {
            if self.mode != "skip-cr" {
                self.t.absorb_point(c_r);
            }
            if self.mode == "swap-rho-lambda" {
                let lambda = self.t.squeeze(b"lambda");
                let rho = self.t.squeeze(b"rho");
                [
                    rho,
                    lambda,
                    self.t.squeeze(b"beta"),
                    self.t.squeeze(b"delta"),
                ]
            } else if self.mode == "fixed-delta" {
                // The 2022-draft soundness hole, re-created: δ is a constant
                // instead of a transcript squeeze.
                [
                    self.t.squeeze(b"rho"),
                    self.t.squeeze(b"lambda"),
                    self.t.squeeze(b"beta"),
                    Scalar::from(42u64),
                ]
            } else {
                [
                    self.t.squeeze(b"rho"),
                    self.t.squeeze(b"lambda"),
                    self.t.squeeze(b"beta"),
                    self.t.squeeze(b"delta"),
                ]
            }
        }

        fn tau(&mut self, c_s: &ProjectivePoint) -> Scalar {
            if self.mode != "skip-cs" {
                self.t.absorb_point(c_s);
            }
            self.t.squeeze(b"tau")
        }
    }

    #[test]
    fn frozen_heart_and_label_discipline_matrix() {
        // A prover whose transcript discipline is broken in any single place
        // (omitted absorption, swapped labels, constant δ) produces proofs
        // the honest verifier rejects — each absorption and each label is
        // load-bearing. This is the δ-omission regression (mode
        // "fixed-delta") plus the per-challenge Frozen-Heart matrix.
        let cases: Vec<(&'static str, u16)> = [
            "skip-cl",
            "skip-co",
            "skip-cr",
            "skip-cs",
            "swap-rho-lambda",
            "fixed-delta",
            "skip-round-index",
            "const-gamma-label",
        ]
        .into_iter()
        .map(|m| (m, 0))
        .chain((1..=u16::try_from(FOLD_ROUNDS).unwrap()).map(|i| ("skip-fold", i)))
        .collect();
        for (mode, skip_round) in cases {
            let mut rng = StdRng::seed_from_u64(0xAD_10);
            let w = witness(0xAD_11);
            let v = commitments_for(&w);
            let mut broken = Broken {
                t: Transcript::new(),
                mode,
                skip_round,
            };
            let proof = prove(&v, &w, &mut broken, &mut rng).unwrap();
            assert!(
                verify(&v, &proof, &mut ratchet()).is_err(),
                "{mode}({skip_round}): broken-discipline proof was accepted"
            );
        }
    }

    #[test]
    fn padded_slot_mass_is_benign_blinding() {
        // The §4.1 padding argument, demonstrated: in-format mass on a padded
        // l-slot (committed in C_S, opened in the fold witness) VERIFIES —
        // the constraint vector is zero there, so it is just unusual
        // blinding; nothing about the proven statement changes. The wire
        // itself has no padding coordinates (fixed residual lengths).
        let mut rng = StdRng::seed_from_u64(0xAD_20);
        let w = witness(0xAD_21);
        let v = commitments_for(&w);
        let proof = prove_with_padded_mass(
            &v,
            &w,
            H_PAD - 2,
            Scalar::from(0xBEEF_u64),
            &mut ratchet(),
            &mut rng,
        )
        .unwrap();
        verify(&v, &proof, &mut ratchet()).unwrap();
    }

    #[test]
    fn frozen_generators_digest_golden() {
        // Pins the exact frozen generator set (g(256) + h(32), item 13a):
        // any drift in the DSTs, counts, derivation, or digest framing must
        // update this vector deliberately.
        let hex = frozen_generators_digest()
            .iter()
            .fold(String::new(), |mut s, b| {
                use std::fmt::Write as _;
                let _ = write!(s, "{b:02x}");
                s
            });
        assert_eq!(
            hex,
            "2ef4f1c69287203c8da10886f4c4d0d617dd942a9c16833288da68e221d54ddc"
        );
    }

    #[test]
    fn frozen_wire_consts_match_derivation() {
        let (rounds, l_len, n_len) = crate::norm_arg::expected_shape(H_PAD, G_PAD);
        assert_eq!(rounds, FOLD_ROUNDS);
        assert_eq!(l_len, RESIDUAL_L);
        assert_eq!(n_len, RESIDUAL_N);
    }

    fn witness(seed: u64) -> RangeWitness {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut values = [0u32; K];
        for v in values.iter_mut().take(LIMB_VALUES) {
            *v = rand::Rng::r#gen::<u32>(&mut rng) & 0x00FF_FFFF;
        }
        for v in values.iter_mut().skip(LIMB_VALUES) {
            *v = u32::from(rand::Rng::r#gen::<bool>(&mut rng));
        }
        let blindings = core::array::from_fn(|_| Scalar::random(&mut rng));
        RangeWitness { values, blindings }
    }

    fn commitments_for(w: &RangeWitness) -> [ProjectivePoint; K] {
        core::array::from_fn(|i| g() * Scalar::from(w.values[i]) + h() * w.blindings[i])
    }

    #[test]
    fn honest_roundtrip_at_frozen_shape() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_01);
        let w = witness(0xCA_75_02);
        let v = commitments_for(&w);
        let proof = prove(&v, &w, &mut ratchet(), &mut rng).unwrap();
        verify(&v, &proof, &mut ratchet()).unwrap();
        // The frozen wire shape: 4 flight points + 6 fold rounds + 1/4 residuals.
        assert_eq!(proof.folds.x.len(), 6);
        assert_eq!(proof.folds.l.len(), 1);
        assert_eq!(proof.folds.n.len(), 4);
    }

    #[test]
    fn boundary_values_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_03);
        let mut w = witness(0xCA_75_04);
        w.values[0] = 0;
        w.values[1] = (1 << 24) - 1;
        w.values[LIMB_VALUES] = 0;
        w.values[LIMB_VALUES + 1] = 1;
        let v = commitments_for(&w);
        let proof = prove(&v, &w, &mut ratchet(), &mut rng).unwrap();
        verify(&v, &proof, &mut ratchet()).unwrap();
    }

    #[test]
    fn out_of_range_witness_rejected_at_prove() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_05);
        let mut w = witness(0xCA_75_06);
        w.values[0] = 1 << 24;
        let v = commitments_for(&w);
        assert_eq!(
            prove(&v, &w, &mut ratchet(), &mut rng).err(),
            Some(Error::DegenerateInput("range witness value out of range"))
        );
        let mut w2 = witness(0xCA_75_07);
        w2.values[LIMB_VALUES] = 2;
        let v2 = commitments_for(&w2);
        assert_eq!(
            prove(&v2, &w2, &mut ratchet(), &mut rng).err(),
            Some(Error::DegenerateInput("range witness value out of range"))
        );
    }

    #[test]
    fn wrong_commitment_rejected() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_08);
        let w = witness(0xCA_75_09);
        let mut v = commitments_for(&w);
        let proof = prove(&v, &w, &mut ratchet(), &mut rng).unwrap();
        v[3] += g();
        assert!(verify(&v, &proof, &mut ratchet()).is_err());
    }

    #[test]
    fn tampered_flight_rejected() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_0A);
        let w = witness(0xCA_75_0B);
        let v = commitments_for(&w);
        let base = prove(&v, &w, &mut ratchet(), &mut rng).unwrap();
        for tamper in 0..4 {
            let mut p = base.clone();
            match tamper {
                0 => p.c_l += g(),
                1 => p.c_o += g(),
                2 => p.c_r += g(),
                _ => p.c_s += g(),
            }
            assert!(verify(&v, &p, &mut ratchet()).is_err(), "tamper {tamper}");
        }
    }

    #[test]
    fn identity_flight_rejected() {
        let mut rng = StdRng::seed_from_u64(0xCA_75_0C);
        let w = witness(0xCA_75_0D);
        let v = commitments_for(&w);
        let mut p = prove(&v, &w, &mut ratchet(), &mut rng).unwrap();
        p.c_o = ProjectivePoint::IDENTITY;
        assert_eq!(
            verify(&v, &p, &mut ratchet()).err(),
            Some(Error::Verification("range identity flight point"))
        );
    }

    #[test]
    fn transcript_divergence_rejected() {
        // A verifier whose transcript differs (e.g. statement absorbed
        // differently upstream) derives different challenges.
        let mut rng = StdRng::seed_from_u64(0xCA_75_0E);
        let w = witness(0xCA_75_0F);
        let v = commitments_for(&w);
        let proof = prove(&v, &w, &mut ratchet(), &mut rng).unwrap();
        let mut diverged = ratchet();
        diverged.0.absorb_u8(0xFF);
        assert!(verify(&v, &proof, &mut diverged).is_err());
    }

    #[test]
    fn proofs_are_randomized() {
        // Same witness, fresh blinding flights: distinct proofs (SHVZK
        // blinding is live), both verifying.
        let mut rng = StdRng::seed_from_u64(0xCA_75_10);
        let w = witness(0xCA_75_11);
        let v = commitments_for(&w);
        let p1 = prove(&v, &w, &mut ratchet(), &mut rng).unwrap();
        let p2 = prove(&v, &w, &mut ratchet(), &mut rng).unwrap();
        assert_ne!(p1.c_s, p2.c_s);
        verify(&v, &p1, &mut ratchet()).unwrap();
        verify(&v, &p2, &mut ratchet()).unwrap();
    }
}
