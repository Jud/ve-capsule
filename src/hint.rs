//! The recovery **unseal hint** — a compact, self-securing, whole-scalar shadow of
//! one additive recovery contribution, openable **recipient-only** with one ECDH +
//! one hash + one subtraction (no BSGS), for compact recovery (design
//! `ve-capsule-recovery.md` §4).
//!
//! At seal the producer knows its contribution scalar `s_h`, a **fresh** ephemeral
//! `r`, and the **public** recipient/aggregate key `Y*`. It forms the DH point from
//! public data alone and masks the contribution:
//!
//! ```text
//! E*   = r·G                      r ←$ Z_n* — FRESH per piece, never a limb E_j (§7)
//! z    = r·Y*                              (= x_rcpt·E*, the mask the open reconstructs)
//! mask = H2S( LP(DOM) ‖ LP(SEC1(E*)) ‖ LP(SEC1(Y*)) ‖ LP(SEC1(VS))
//!             ‖ LP(ctx) ‖ LP(epoch) ‖ LP(BE32(idx)) ‖ LP(x(z)) )   ∈ Z_n
//! ct   = s_h + mask  (mod n)
//! hint = (E*, ct)                                                  (65 B / piece)
//! ```
//!
//! **Open (recipient-only):** `z = x_rcpt·E*`, `mask = H2S(…)`, `s_h = ct − mask`;
//! combine with caller-provided weights; accept iff `s·G == VS`. The compact-payload
//! path seals caller-prepared additive contributions and therefore uses unit weights.
//! The recheck against the **certified** `VS` is self-securing: a tampered hint
//! shifts `s` by `δ≠0` and fails closed (§10), never yielding a wrong secret. Only
//! `x(z)` is absorbed, so x-only ECDH on constrained devices reproduces the mask
//! bit-for-bit (§8).
//!
//! Whole-scalar and fresh-`E*` are normative: per-limb hints leak (~2²⁴ brute force),
//! and an `E*` reused from a limb ephemeral lets a full-core holder brute-force a limb
//! and confirm via `s·G == VS` (§7). [`RecoveryHint::aliases_any`] is the verifier-side
//! freshness check.
//!
//! **Gated open (§5, §8).** When recovery is strict-AND gated, `Y*` is the composite
//! `a_r·Y_rcpt + Σ_k a_k·Y_k` (coefficients and roster from `composite`). Each
//! authorizer returns a [`AuthorizerContribution`] `(w·G, w·E*)` + a DLEQ over
//! `[G, E*]`; per gate the `W` parts must sum to `Y_k` (strict-AND). An untrusted
//! coordinator runs the DLEQs + strict-AND ([`public_gate_sum`]) and hands the
//! recipient the public sum `S_gate = Σ_k a_k·(Σ_bucket w·E*)`; the recipient
//! finishes with one `sP + Q`: `z = a_r·(x_rcpt·E*) + S_gate = r·Y*`. The gated public entry is
//! [`PinnedHintVerifier::verify_and_recover_gated`] — it verifies the quorum's §8
//! hint-binding signature over the exact pieces, then combines and self-checks
//! `s·G == VS` (the raw single-piece open and the combine are crate-internal, so the
//! attestation cannot be skipped). A short or tampered quorum yields the wrong `z` and
//! fails the check — never a wrong secret.

use crate::codec::{POINT_LEN, decode_point, encode_point};
use crate::composite::{self, OpeningBinding};
use crate::dleq::BatchDleqProof;
use crate::error::Error;
use crate::generators::g;
use crate::params::Params;
use crate::signature::{self, Signature};
use crate::transcript::{length_prefix, push_framed};
use k256::elliptic_curve::Field;
use k256::elliptic_curve::PrimeField;
use k256::elliptic_curve::bigint::U256;
use k256::elliptic_curve::ops::Reduce;
use k256::elliptic_curve::point::AffineCoordinates;
use k256::{FieldBytes, ProjectivePoint, Scalar};
use rand_core::{CryptoRng, RngCore};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// Domain tag for the hint mask. Bump on any wire/derivation change.
const HINT_DOMAIN: &[u8] = b"ve-capsule.recovery-hint.v1";

/// Domain tag for an authorizer's gated-contribution DLEQ binding. The binding is
/// framed with the piece's `E*`, the composite `Y*`, the certified `VS`,
/// `ctx`/`epoch`, the roster commitment `g_star`, `idx`, and the gate key, so a
/// contribution is bound to exactly one (piece, gate, roster, recovery context)
/// and cannot be replayed onto another gate, piece, or roster. Bump on any wire
/// change.
const GATED_DLEQ_DOMAIN: &[u8] = b"ve-capsule.recovery-hint.gated-dleq.v1";

/// Byte length of a canonical scalar (`ct`): 32-byte big-endian, `< n`.
const SCALAR_LEN: usize = 32;

/// Canonical wire length of one hint: `E*` (33 B SEC1) ‖ `ct` (32 B) = 65 B.
pub const HINT_LEN: usize = POINT_LEN + SCALAR_LEN;

/// The recovery-level fields every piece's hint binds.
///
/// `Y*`, `VS`, `ctx`, and `epoch` are shared across all pieces of one recovery, so a
/// caller builds one `HintBinding` and reuses it for each piece (the per-piece `idx`
/// is passed separately). Grouping them also keeps the seal/open call sites legible.
#[derive(Clone, Copy)]
pub struct HintBinding<'a> {
    /// Recipient/aggregate key `Y*` (`= Y_rcpt` recipient-only; a composite if gated).
    pub y_star: &'a ProjectivePoint,
    /// The certified target point `VS = s·G` (never the payload's).
    pub vs: &'a ProjectivePoint,
    /// Context domain separator.
    pub ctx: &'a [u8],
    /// Recovery epoch identifier.
    pub epoch: &'a [u8],
}

/// Generous upper bound on the `ctx` / `epoch` domain separators. Bounding them
/// keeps the framing injective under the saturating 4-byte length prefix
/// (`transcript::length_prefix`) and keeps the mask preimage buffer fixed-size;
/// real values are tens of bytes.
const MAX_DOMAIN_SEP_LEN: usize = 256;

impl HintBinding<'_> {
    /// Validate the binding for both seal and open: the recipient/aggregate key and
    /// the certified target point must be non-identity, and the `ctx`/`epoch`
    /// domain separators must be within [`MAX_DOMAIN_SEP_LEN`].
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] on an identity `y_star`/`vs` or an over-long
    /// `ctx`/`epoch`.
    fn validate(&self) -> Result<(), Error> {
        if self.y_star == &ProjectivePoint::IDENTITY {
            return Err(Error::DegenerateInput(
                "recovery-hint recipient key Y* is the identity",
            ));
        }
        if self.vs == &ProjectivePoint::IDENTITY {
            return Err(Error::DegenerateInput(
                "recovery-hint certified target VS is the identity",
            ));
        }
        if self.ctx.len() > MAX_DOMAIN_SEP_LEN || self.epoch.len() > MAX_DOMAIN_SEP_LEN {
            return Err(Error::DegenerateInput(
                "recovery-hint ctx/epoch exceeds the maximum domain-separator length",
            ));
        }
        Ok(())
    }
}

/// SHA-256 → reduce a 32-byte big-endian digest into `Z_n`, the crate's
/// Fiat–Shamir convention (see `signature.rs`, `aggregate.rs`).
///
/// The reduction of a uniform 256-bit value mod the secp256k1 order has bias
/// `≤ 2⁻¹²⁷` from uniform — negligible, and absorbed into the random-oracle term of
/// the simulatability argument (design §11; resolves R3 for the 32-byte reduce).
fn reduce_be_to_scalar(digest: [u8; 32]) -> Scalar {
    let mut fb = FieldBytes::default();
    fb.copy_from_slice(&digest);
    <Scalar as Reduce<U256>>::reduce_bytes(&fb)
}

/// Sample a uniform nonzero ephemeral. `r = 0` makes `E*` the identity and exposes
/// the contribution (`ct = s_h`), so the negligibly likely zero draw is rejected.
fn nonzero_scalar<R: RngCore + CryptoRng>(rng: &mut R) -> Zeroizing<Scalar> {
    loop {
        let r = Scalar::random(&mut *rng);
        if !bool::from(r.is_zero()) {
            return Zeroizing::new(r);
        }
    }
}

/// The hint mask, built identically at seal and open so the two cannot diverge.
///
/// Computes `H2S(LP(DOM) ‖ LP(SEC1(E*)) ‖ LP(SEC1(Y*)) ‖ LP(SEC1(VS)) ‖ LP(ctx) ‖
/// LP(epoch) ‖ LP(BE32(idx)) ‖ LP(x(z)))`. `z` is the Diffie–Hellman point — `r·Y*`
/// at seal, `x_rcpt·E*` (or the gated sum) at open — and only its x-coordinate is
/// absorbed. Binding `SEC1(E*)` kills the `±E*` x-only alias (§7); binding `SEC1(VS)`
/// + `epoch` makes the mask target/epoch-specific (§10).
///
/// # Errors
///
/// [`Error::DegenerateInput`] if `z` is the identity (an exposed mask).
fn hint_mask(
    e_star: &ProjectivePoint,
    binding: &HintBinding<'_>,
    idx: u32,
    z: &ProjectivePoint,
) -> Result<Scalar, Error> {
    if z == &ProjectivePoint::IDENTITY {
        return Err(Error::DegenerateInput(
            "recovery-hint DH point is the identity",
        ));
    }
    let z_x = z.to_affine().x();
    let mut framed = Vec::new();
    push_framed(&mut framed, HINT_DOMAIN);
    push_framed(&mut framed, &encode_point(e_star));
    push_framed(&mut framed, &encode_point(binding.y_star));
    push_framed(&mut framed, &encode_point(binding.vs));
    push_framed(&mut framed, binding.ctx);
    push_framed(&mut framed, binding.epoch);
    push_framed(&mut framed, &idx.to_be_bytes());
    push_framed(&mut framed, &z_x);
    Ok(reduce_be_to_scalar(Sha256::digest(&framed).into()))
}

/// One piece's recovery hint: the fresh ephemeral `E* = r·G` and the masked
/// contribution `ct = s_h + mask`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryHint {
    e_star: ProjectivePoint,
    ct: Scalar,
}

impl RecoveryHint {
    /// Mint a hint for one additive contribution with a **caller-supplied** ephemeral
    /// `r`. Internal seam shared by [`RecoveryHint::seal`] and the cross-language KAT
    /// (which pins `r`).
    fn mint_with_ephemeral(
        contribution: &Scalar,
        binding: &HintBinding<'_>,
        idx: u32,
        r: &Scalar,
    ) -> Result<Self, Error> {
        binding.validate()?;
        if bool::from(r.is_zero()) {
            return Err(Error::DegenerateInput("recovery-hint ephemeral r is zero"));
        }
        let e_star = g() * r;
        let z = *binding.y_star * r;
        let mask = hint_mask(&e_star, binding, idx, &z)?;
        Ok(Self {
            e_star,
            ct: *contribution + mask,
        })
    }

    /// Seal a hint for one additive contribution under piece index `idx`, drawing a
    /// **fresh** ephemeral. The producer needs only its own contribution and the
    /// public [`HintBinding`] (recipient/aggregate key + certified `VS` +
    /// `ctx`/`epoch`).
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `binding.y_star` or `binding.vs` is the identity.
    pub(crate) fn seal<R: RngCore + CryptoRng>(
        contribution: &Scalar,
        binding: &HintBinding<'_>,
        idx: u32,
        rng: &mut R,
    ) -> Result<Self, Error> {
        let r = nonzero_scalar(rng);
        Self::mint_with_ephemeral(contribution, binding, idx, &r)
    }

    /// Open this piece **recipient-only** with the recipient secret `x_rcpt`
    /// (`Y* = x_rcpt·G`), reconstructing the masking DH point `z = x_rcpt·E*`.
    ///
    /// Returns the raw piece scalar (zeroizing) **before any self-check**, so it is
    /// crate-internal: the public boundary is [`recover_recipient_only`], which
    /// combines the pieces and authenticates the result via `s·G == VS`. A `pub`
    /// pre-check single-piece open would, for a one-piece recovery, hand out the full
    /// recovered secret without the certified-`VS` gate.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `binding` is degenerate (identity `y_star`/`vs`,
    /// over-long `ctx`/`epoch`) or the reconstructed `z` is the identity.
    pub(crate) fn open_piece(
        &self,
        x_rcpt: &Scalar,
        binding: &HintBinding<'_>,
        idx: u32,
    ) -> Result<Zeroizing<Scalar>, Error> {
        binding.validate()?;
        let z = self.e_star * x_rcpt;
        let mask = hint_mask(&self.e_star, binding, idx, &z)?;
        Ok(Zeroizing::new(self.ct - mask))
    }

    /// The fresh ephemeral `E* = r·G`.
    #[must_use]
    pub const fn e_star(&self) -> ProjectivePoint {
        self.e_star
    }

    /// The §7 fresh-`E*` invariant check: `true` iff this hint's `E*` aliases any of
    /// the capsule's limb ephemerals `E_j` as **`E* == E_j` or `E* == -E_j`**.
    ///
    /// Both must be rejected: the mask absorbs only `x(z)` and `x(P) == x(-P)`, so
    /// `E* == -E_j` reconstructs the *same* masking x-coordinate as `E* == E_j` — a
    /// full-core holder could then brute-force limb `v_j` and confirm via
    /// `s·G == VS`. This mirrors the ±E ElGamal-mask screen in `assembly.rs`. The
    /// stronger small-coefficient / G-offset relation screen
    /// (`assembly::degenerate_elgamal_mask`) is applied at capsule integration, when
    /// `E*` joins the limb set under a single screen.
    #[must_use]
    pub fn aliases_any(&self, limb_ephemerals: &[ProjectivePoint]) -> bool {
        let neg_e_star = -self.e_star;
        limb_ephemerals
            .iter()
            .any(|e_j| *e_j == self.e_star || *e_j == neg_e_star)
    }

    /// Canonical wire bytes: `E*` (33 B SEC1) ‖ `ct` (32 B big-endian).
    #[must_use]
    pub fn to_canonical_bytes(&self) -> [u8; HINT_LEN] {
        let mut out = [0u8; HINT_LEN];
        out[..POINT_LEN].copy_from_slice(&encode_point(&self.e_star));
        out[POINT_LEN..].copy_from_slice(&self.ct.to_bytes());
        out
    }

    /// Parse a hint from canonical bytes: strict SEC1 decode for `E*` (identity
    /// rejected) and a canonical (`< n`) `ct`.
    ///
    /// # Errors
    ///
    /// [`Error::PointDecode`] on truncation, a malformed `E*`, trailing bytes, or a
    /// non-canonical `ct`; [`Error::DegenerateInput`] if `E*` is the identity.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let (e_bytes, ct_bytes) = bytes
            .split_at_checked(POINT_LEN)
            .ok_or(Error::PointDecode("recovery hint: truncated"))?;
        let e_star = decode_point(e_bytes)?;
        if e_star == ProjectivePoint::IDENTITY {
            return Err(Error::DegenerateInput(
                "recovery-hint ephemeral E* is the identity",
            ));
        }
        if ct_bytes.len() != SCALAR_LEN {
            return Err(Error::PointDecode("recovery hint: trailing bytes"));
        }
        let mut fb = FieldBytes::default();
        fb.copy_from_slice(ct_bytes);
        let ct = Option::<Scalar>::from(Scalar::from_repr(fb))
            .ok_or(Error::PointDecode("recovery hint: non-canonical ct (>= n)"))?;
        Ok(Self { e_star, ct })
    }
}

/// Recover the combined secret `s = Σ λ_h s_h`, recipient-only and self-securing.
///
/// Opens each `(idx, hint)` piece with `x_rcpt`, combines under caller-provided
/// weights `λ`, and authenticates the result against the **certified** verification
/// share `binding.vs` (never the payload's — design §10). `n`-of-`n` additive
/// recovery is the degenerate case `λ ≡ 1`.
///
/// The piece set is validated as canonical: non-empty, one weight per piece, and
/// **strictly increasing** indices — rejecting duplicate and out-of-order pieces
/// (defense-in-depth; a missing or poisoned piece additionally fails the `s·G == VS`
/// check).
///
/// # Errors
///
/// [`Error::Verification`] on a malformed piece set or if the combined secret does
/// not reconstruct `binding.vs`; otherwise a per-piece open error.
pub fn recover_recipient_only(
    pieces: &[(u32, RecoveryHint)],
    weights: &[Scalar],
    x_rcpt: &Scalar,
    binding: &HintBinding<'_>,
) -> Result<Zeroizing<Scalar>, Error> {
    if pieces.is_empty() {
        return Err(Error::Verification("recovery-hint piece set is empty"));
    }
    if pieces.len() != weights.len() {
        return Err(Error::Verification(
            "recovery-hint piece count does not match weight count",
        ));
    }
    // Canonical order: strictly increasing indices reject duplicates and reordering.
    for window in pieces.windows(2) {
        if window[1].0 <= window[0].0 {
            return Err(Error::Verification(
                "recovery-hint pieces are not in strictly increasing index order",
            ));
        }
    }
    let mut s = Zeroizing::new(Scalar::ZERO);
    for ((idx, hint), weight) in pieces.iter().zip(weights) {
        let piece = hint.open_piece(x_rcpt, binding, *idx)?;
        *s += *weight * *piece;
    }
    if g() * *s == *binding.vs {
        Ok(s)
    } else {
        Err(Error::Verification(
            "recovered secret does not reconstruct the certified target VS",
        ))
    }
}

// ---------------------------------------------------------------------------
// Gated open (strict-AND, MuSig-weighted authorizers) — design §5, §8.
// ---------------------------------------------------------------------------

/// The gate roster + recovery context a gated open binds.
///
/// The composite recovery key `Y* = a_r·Y_rcpt + Σ_k a_k·Y_k` and the aggregation
/// coefficients are *derived* from `(recipient, access_keys)` via
/// `composite::opening_binding`, not supplied directly, so the open side cannot be
/// handed a `Y*` inconsistent with the roster. An empty `access_keys` is the ungated
/// (recipient-only) degenerate case (`Y* = Y_rcpt`, `a_r = 1`); prefer
/// `recover_recipient_only` there.
#[derive(Clone, Copy)]
pub struct GatedBinding<'a> {
    /// The recipient key `Y_rcpt` (the recipient recovery key), `= x_rcpt·G`.
    pub recipient: &'a ProjectivePoint,
    /// The access-gate roster `{Y_k}` — each gate's combined (sub-quorum) key.
    pub access_keys: &'a [ProjectivePoint],
    /// The certified target point `VS = s·G` (never the payload's — design §10).
    pub vs: &'a ProjectivePoint,
    /// Context domain separator.
    pub ctx: &'a [u8],
    /// Recovery epoch identifier.
    pub epoch: &'a [u8],
}

/// One authorizer's gated partial.
///
/// The single-point pair `(W, W*) = (w·G, w·E*)` and a `BatchDleqProof` over the
/// two bases `[G, E*]` proving **one** `w` relates both. The DLEQ is what stops an
/// authorizer from publishing a `W` that passes the per-gate strict-AND while
/// supplying a `W*` that is not `w·E*` (a silent mask-corruption / partial-decryption
/// oracle — see `dleq` module docs).
#[derive(Clone, Debug)]
pub struct AuthorizerContribution {
    /// `W = w·G`.
    w_g: ProjectivePoint,
    /// `W* = w·E*`.
    w_estar: ProjectivePoint,
    /// Batched DLEQ over `[G, E*]` proving a common `w`.
    dleq: BatchDleqProof,
}

impl AuthorizerContribution {
    /// Produce a contribution to `gate` for the piece whose ephemeral is `e_star`,
    /// with the authorizer's secret contribution `w` (`gate`'s combined secret is
    /// `Σ w` over its sub-quorum). The DLEQ is bound to exactly this (piece, gate,
    /// recovery) via `GATED_DLEQ_DOMAIN`, so it cannot be replayed.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `gate` is not in the derived roster or the
    /// binding is degenerate; otherwise a DLEQ proving error.
    pub fn contribute<R: RngCore + CryptoRng>(
        w: &Scalar,
        e_star: &ProjectivePoint,
        binding: &GatedBinding<'_>,
        idx: u32,
        gate: &ProjectivePoint,
        rng: &mut R,
    ) -> Result<Self, Error> {
        let opening = composite::opening_binding(binding.recipient, binding.access_keys)?;
        if !opening.gates.iter().any(|listed| listed == gate) {
            return Err(Error::DegenerateInput(
                "recovery-hint contribution gate is not in the roster",
            ));
        }
        let hint_binding = HintBinding {
            y_star: &opening.y_star,
            vs: binding.vs,
            ctx: binding.ctx,
            epoch: binding.epoch,
        };
        hint_binding.validate()?;
        let dleq_binding = gated_dleq_binding(e_star, &hint_binding, &opening.g_star, idx, gate);
        let bases = [g(), *e_star];
        let (images, dleq) = BatchDleqProof::prove(w, &bases, &dleq_binding, rng)?;
        let (w_g, w_estar) = match images.as_slice() {
            [first, second] => (*first, *second),
            _ => {
                return Err(Error::Verification(
                    "recovery-hint contribution: unexpected DLEQ image count",
                ));
            }
        };
        Ok(Self { w_g, w_estar, dleq })
    }

    /// Verify the DLEQ against `[G, E*]` and `dleq_binding`, rejecting identity
    /// images (a `w = 0` phantom authorizer) up front.
    fn verify(&self, e_star: &ProjectivePoint, dleq_binding: &[u8]) -> Result<(), Error> {
        if self.w_g == ProjectivePoint::IDENTITY || self.w_estar == ProjectivePoint::IDENTITY {
            return Err(Error::DegenerateInput(
                "recovery-hint contribution image is the identity",
            ));
        }
        let bases = [g(), *e_star];
        let images = [self.w_g, self.w_estar];
        self.dleq.verify(&bases, &images, dleq_binding)
    }
}

/// One gate's sub-quorum: the gate key `Y_k` it satisfies and the authorizer
/// contributions whose `W` parts must sum to `Y_k` (strict-AND).
#[derive(Clone, Debug)]
pub struct GateQuorum {
    gate: ProjectivePoint,
    contributions: Vec<AuthorizerContribution>,
}

impl GateQuorum {
    /// Bundle the contributions satisfying `gate`.
    #[must_use]
    pub const fn new(gate: ProjectivePoint, contributions: Vec<AuthorizerContribution>) -> Self {
        Self {
            gate,
            contributions,
        }
    }
}

/// One piece of a multi-piece gated recovery: the piece index, its hint, and the
/// per-gate quorums opening *that* piece (each piece has its own `E*`, so each needs
/// its own contributions).
#[derive(Clone, Copy)]
pub struct GatedPiece<'a> {
    /// The piece index.
    pub idx: u32,
    /// The piece's recovery hint.
    pub hint: &'a RecoveryHint,
    /// One [`GateQuorum`] per roster gate, opening this piece.
    pub quorums: &'a [GateQuorum],
}

/// The canonical DLEQ binding for one (piece `E*`, gate) under a recovery context.
/// Built identically by `AuthorizerContribution::contribute` and
/// `AuthorizerContribution::verify`, so a proof is bound to exactly one gate and
/// piece and cannot be replayed.
///
/// Binds the roster commitment `g_star` (design §5's `g*`) alongside the composite
/// `Y*`. `Y*` already folds the roster through its aggregation coefficients, but a
/// single aggregate point is not a transparent roster commitment, so binding
/// `g_star` pins the exact canonical gate set against a (theoretical) cross-roster
/// `Y*` collision. The *complete* recovery authorization — binding `(E*, ct)` +
/// roster + certified `VS` under a pinned verifier — is the separate hint-binding
/// signature (design §8, unit 1c); this is the per-contribution algebraic
/// anti-replay.
fn gated_dleq_binding(
    e_star: &ProjectivePoint,
    binding: &HintBinding<'_>,
    g_star: &[u8; 32],
    idx: u32,
    gate: &ProjectivePoint,
) -> Vec<u8> {
    let mut out = Vec::new();
    push_framed(&mut out, GATED_DLEQ_DOMAIN);
    push_framed(&mut out, &encode_point(e_star));
    push_framed(&mut out, &encode_point(binding.y_star));
    push_framed(&mut out, &encode_point(binding.vs));
    push_framed(&mut out, binding.ctx);
    push_framed(&mut out, binding.epoch);
    push_framed(&mut out, g_star);
    push_framed(&mut out, &idx.to_be_bytes());
    push_framed(&mut out, &encode_point(gate));
    out
}

/// The coordinator-side **public** computation: verify every authorizer DLEQ,
/// enforce the per-gate strict-AND `Σ W == Y_k`, and accumulate the weighted gate sum
/// `S_gate = Σ_k a_k·(Σ_bucket W*)`. Touches only public data — no recipient
/// secret. The `quorums` must biject with the derived roster gates.
fn gate_sum_inner(
    e_star: &ProjectivePoint,
    hint_binding: &HintBinding<'_>,
    opening: &OpeningBinding,
    idx: u32,
    quorums: &[GateQuorum],
) -> Result<ProjectivePoint, Error> {
    if quorums.len() != opening.gates.len() {
        return Err(Error::Verification(
            "recovery-hint gated open: quorum count does not match the gate roster",
        ));
    }
    // Reject duplicate quorum gates so the gate ↔ quorum mapping is a bijection.
    for (i, quorum) in quorums.iter().enumerate() {
        if quorums
            .iter()
            .skip(i + 1)
            .any(|other| other.gate == quorum.gate)
        {
            return Err(Error::Verification(
                "recovery-hint gated open: duplicate quorum gate",
            ));
        }
    }
    let mut s_gate = ProjectivePoint::IDENTITY;
    for (gate, a_k) in opening.gates.iter().zip(&opening.gate_weights) {
        let quorum =
            quorums
                .iter()
                .find(|quorum| &quorum.gate == gate)
                .ok_or(Error::Verification(
                    "recovery-hint gated open: no quorum for a roster gate",
                ))?;
        if quorum.contributions.is_empty() {
            return Err(Error::Verification(
                "recovery-hint gated open: empty gate quorum",
            ));
        }
        let dleq_binding = gated_dleq_binding(e_star, hint_binding, &opening.g_star, idx, gate);
        let mut bucket_g = ProjectivePoint::IDENTITY;
        let mut bucket_estar = ProjectivePoint::IDENTITY;
        for (j, contribution) in quorum.contributions.iter().enumerate() {
            contribution.verify(e_star, &dleq_binding)?;
            if quorum
                .contributions
                .iter()
                .take(j)
                .any(|prev| prev.w_g == contribution.w_g)
            {
                return Err(Error::Verification(
                    "recovery-hint gated open: duplicate contribution in a gate quorum",
                ));
            }
            bucket_g += contribution.w_g;
            bucket_estar += contribution.w_estar;
        }
        if bucket_g != *gate {
            return Err(Error::Verification(
                "recovery-hint gated open: gate quorum does not reconstruct the gate key",
            ));
        }
        s_gate += bucket_estar * *a_k;
    }
    Ok(s_gate)
}

/// Verify a gated quorum and return the public weighted gate sum
/// `S_gate = Σ_k a_k·(Σ_bucket w·E*)` for the piece `e_star` under `binding`.
///
/// This is the untrusted coordinator's whole job: it runs the DLEQs and the
/// strict-AND and hands `S_gate` (a public point) to the recipient, which finishes
/// the open with one `sP + Q` using only its own `x_rcpt` (design §8). The final
/// `s·G == VS` self-check (`recover_gated`) is what makes a tampered or short
/// quorum fail closed, so the coordinator's verification here is early/attributable
/// detection, not the soundness backstop.
///
/// # Errors
///
/// [`Error::DegenerateInput`] on a degenerate roster/binding; [`Error::Verification`]
/// on a DLEQ failure, a quorum/gate mismatch, or a gate whose quorum does not sum to
/// the gate key.
pub fn public_gate_sum(
    e_star: &ProjectivePoint,
    binding: &GatedBinding<'_>,
    idx: u32,
    quorums: &[GateQuorum],
) -> Result<ProjectivePoint, Error> {
    let opening = composite::opening_binding(binding.recipient, binding.access_keys)?;
    let hint_binding = HintBinding {
        y_star: &opening.y_star,
        vs: binding.vs,
        ctx: binding.ctx,
        epoch: binding.epoch,
    };
    hint_binding.validate()?;
    gate_sum_inner(e_star, &hint_binding, &opening, idx, quorums)
}

impl RecoveryHint {
    /// Open this piece behind a strict-AND gate quorum, reconstructing
    /// `z = a_r·(x_rcpt·E*) + Σ_k a_k·(Σ_bucket w·E*) = r·Y*` and stripping the mask.
    ///
    /// Verifies every authorizer DLEQ and enforces the per-gate strict-AND before
    /// computing the mask, then returns the raw piece scalar **before any self-check**
    /// — so it is crate-internal, like [`RecoveryHint::open_piece`]. The public
    /// boundary is [`recover_gated`], which combines the pieces and authenticates the
    /// result via `s·G == VS` (a single piece is not self-securing on its own, and for
    /// a one-piece share the raw value is the full secret).
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] on a degenerate roster/binding; [`Error::Verification`]
    /// on a DLEQ failure, a quorum/gate mismatch, a short/wrong quorum, or an exposed
    /// (identity) reconstructed `z`.
    pub(crate) fn open_piece_gated(
        &self,
        x_rcpt: &Scalar,
        binding: &GatedBinding<'_>,
        idx: u32,
        quorums: &[GateQuorum],
    ) -> Result<Zeroizing<Scalar>, Error> {
        let opening = composite::opening_binding(binding.recipient, binding.access_keys)?;
        let hint_binding = HintBinding {
            y_star: &opening.y_star,
            vs: binding.vs,
            ctx: binding.ctx,
            epoch: binding.epoch,
        };
        hint_binding.validate()?;
        let s_gate = gate_sum_inner(&self.e_star, &hint_binding, &opening, idx, quorums)?;
        let recipient_term = (self.e_star * x_rcpt) * opening.recipient_weight;
        let z = recipient_term + s_gate;
        let mask = hint_mask(&self.e_star, &hint_binding, idx, &z)?;
        Ok(Zeroizing::new(self.ct - mask))
    }
}

/// Recover the combined secret `s = Σ λ_h s_h` behind a strict-AND gate
/// quorum, self-securing against the **certified** `binding.vs`.
///
/// Opens each [`GatedPiece`] with `x_rcpt` and its per-gate quorums, combines under
/// caller-provided weights `λ`, and accepts iff the result reconstructs `binding.vs`.
/// A short, wrong, or tampered quorum yields a wrong `z` and fails the check —
/// never a wrong secret (design §10). The piece set is validated canonical
/// (non-empty, one weight per piece, strictly increasing indices).
///
/// Crate-internal: the gated path's public entry is
/// [`PinnedHintVerifier::verify_and_recover_gated`], which verifies the quorum's
/// hint-binding signature over the exact pieces under the pinned binding **before**
/// recovering with that same binding — so the §8 attestation cannot be skipped or
/// split from the secret's use. Recipient-only recovery uses [`recover_recipient_only`]
/// (no signature; self-securing).
///
/// # Errors
///
/// [`Error::Verification`] on a malformed piece set or a failed self-check;
/// [`Error::DegenerateInput`] on a degenerate roster/binding; otherwise a per-piece
/// gated-open error.
fn recover_gated(
    pieces: &[GatedPiece<'_>],
    weights: &[Scalar],
    x_rcpt: &Scalar,
    binding: &GatedBinding<'_>,
) -> Result<Zeroizing<Scalar>, Error> {
    if pieces.is_empty() {
        return Err(Error::Verification(
            "recovery-hint gated piece set is empty",
        ));
    }
    if pieces.len() != weights.len() {
        return Err(Error::Verification(
            "recovery-hint gated piece count does not match weight count",
        ));
    }
    for window in pieces.windows(2) {
        if window[1].idx <= window[0].idx {
            return Err(Error::Verification(
                "recovery-hint gated pieces are not in strictly increasing index order",
            ));
        }
    }
    let mut s = Zeroizing::new(Scalar::ZERO);
    for (piece, weight) in pieces.iter().zip(weights) {
        let opened = piece
            .hint
            .open_piece_gated(x_rcpt, binding, piece.idx, piece.quorums)?;
        *s += *weight * *opened;
    }
    if g() * *s == *binding.vs {
        Ok(s)
    } else {
        Err(Error::Verification(
            "recovered secret does not reconstruct the certified target VS",
        ))
    }
}

// ---------------------------------------------------------------------------
// Hint-binding attestation (quorum signature over the hint + roster) — §8, R5.
// ---------------------------------------------------------------------------

/// Domain separator for the recovery-hint attestation statement. Distinct from the
/// capsule / Case / aggregate attestation domains (in `signature.rs`) so a hint
/// signature can never be confused with — or replayed as — one of those. Bump on any
/// statement-layout change.
const RECOVERY_HINT_ATTESTATION_DOMAIN: &[u8] = b"ve-capsule.recovery-hint-attestation.v1";

/// Build the canonical statement a quorum signs to attest a recovery-hint set:
///
/// `domain ‖ count ‖ {BE32(idx) ‖ SEC1(E*) ‖ ct}↑ ‖ VS ‖ recipient ‖ g* ‖ Y*
///  ‖ ctx ‖ epoch ‖ params_id`
///
/// Per-piece records are sorted ascending (canonical regardless of presented order;
/// duplicate `idx` rejected). Binding `(E*, ct)` per piece is exactly what the 1b
/// per-contribution DLEQ binding cannot cover — it pins the masked contribution
/// bytes so an untrusted coordinator cannot substitute a piece — while
/// `VS ‖ recipient ‖ g* ‖ Y*` pin the certified target, recipient, and exact
/// roster (no cross-context / cross-roster replay), and `ctx ‖ epoch ‖ params_id`
/// fork the statement per recovery epoch and parameter set. Built once here so the
/// provisioner ([`hint_attestation_message`]) and the verifier
/// ([`PinnedHintVerifier::verify`]) cannot drift.
///
/// # Errors
///
/// [`Error::Verification`] on an empty piece set or a duplicate piece index.
fn hint_attestation_statement(
    pieces: &[(u32, &RecoveryHint)],
    binding: &HintBinding<'_>,
    opening: &OpeningBinding,
) -> Result<Vec<u8>, Error> {
    if pieces.is_empty() {
        return Err(Error::Verification(
            "recovery-hint attestation: empty piece set",
        ));
    }
    let mut sorted: Vec<(u32, &RecoveryHint)> = pieces.to_vec();
    sorted.sort_unstable_by_key(|(idx, _)| *idx);
    for window in sorted.windows(2) {
        if window[0].0 == window[1].0 {
            return Err(Error::Verification(
                "recovery-hint attestation: duplicate piece index",
            ));
        }
    }
    let mut out = Vec::new();
    push_framed(&mut out, RECOVERY_HINT_ATTESTATION_DOMAIN);
    out.extend_from_slice(&length_prefix(sorted.len()));
    for (idx, hint) in &sorted {
        let mut record = Vec::with_capacity(4 + HINT_LEN);
        record.extend_from_slice(&idx.to_be_bytes());
        record.extend_from_slice(&hint.to_canonical_bytes());
        push_framed(&mut out, &record);
    }
    push_framed(&mut out, &encode_point(binding.vs));
    push_framed(&mut out, &encode_point(&opening.recipient));
    push_framed(&mut out, &opening.g_star);
    push_framed(&mut out, &encode_point(&opening.y_star));
    push_framed(&mut out, binding.ctx);
    push_framed(&mut out, binding.epoch);
    push_framed(&mut out, &Params::FROZEN.id());
    Ok(out)
}

/// The 32-byte digest a quorum signs to attest a recovery-hint set.
///
/// Reduced from the canonical `hint_attestation_statement`. The provisioner feeds
/// this to its quorum signing round; the [`PinnedHintVerifier`] rebuilds and rechecks
/// it.
///
/// # Errors
///
/// [`Error::DegenerateInput`] on a degenerate roster/binding; [`Error::Verification`]
/// on an empty or duplicate-index piece set.
pub fn hint_attestation_message(
    pieces: &[(u32, &RecoveryHint)],
    binding: &GatedBinding<'_>,
) -> Result<[u8; 32], Error> {
    let opening = composite::opening_binding(binding.recipient, binding.access_keys)?;
    let hint_binding = HintBinding {
        y_star: &opening.y_star,
        vs: binding.vs,
        ctx: binding.ctx,
        epoch: binding.epoch,
    };
    hint_binding.validate()?;
    let statement = hint_attestation_statement(pieces, &hint_binding, &opening)?;
    Ok(signature::attestation_digest(&statement))
}

/// The pinned hint-attestation verifier (design §8).
///
/// The quorum verifying key and the recovery context (recipient, roster, certified
/// `VS`, `ctx`, `epoch`) are **pinned by the recipient verifier**, never
/// coordinator-supplied. At recovery the untrusted coordinator presents only the
/// hint pieces and the quorum signature; the verifier rebuilds the statement from
/// its **pinned** fields plus the presented pieces, so a coordinator that tampers with a piece, the
/// certified `VS`, the recipient, the roster, or the context yields a statement the
/// quorum never signed. This is what lets the recipient trust the coordinator-supplied
/// `S_gate` in the gated open (§8): the pieces' `(E*, ct)` and the certified `VS`
/// are authenticated before any secret is used, and the gated open's `s·G == VS`
/// self-check uses the same pinned `VS`.
#[derive(Clone, Copy)]
pub struct PinnedHintVerifier<'a> {
    /// The quorum's 32-byte x-only verifying key, pinned.
    pub quorum_key: &'a [u8; 32],
    /// The pinned recovery context (recipient, roster, certified `VS`, `ctx`, `epoch`).
    pub binding: GatedBinding<'a>,
}

impl PinnedHintVerifier<'_> {
    /// Verify the quorum `signature` over the presented `pieces` against the pinned
    /// context. The pieces' `(E*, ct)` are bound by the signature; every other field
    /// is taken from the pinned binding, so the coordinator cannot substitute the
    /// certified `VS`, the recipient, the roster, or the context.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] on a degenerate pinned roster/binding;
    /// [`Error::Verification`] on an empty/duplicate piece set or a signature that does
    /// not verify against the pinned quorum key and the rebuilt statement.
    pub fn verify(
        &self,
        pieces: &[(u32, &RecoveryHint)],
        signature: &Signature,
    ) -> Result<(), Error> {
        let opening = composite::opening_binding(self.binding.recipient, self.binding.access_keys)?;
        let hint_binding = HintBinding {
            y_star: &opening.y_star,
            vs: self.binding.vs,
            ctx: self.binding.ctx,
            epoch: self.binding.epoch,
        };
        hint_binding.validate()?;
        let statement = hint_attestation_statement(pieces, &hint_binding, &opening)?;
        let digest = signature::attestation_digest(&statement);
        signature::verify_signature(signature, self.quorum_key, &digest)
    }

    /// Atomically verify the quorum `signature` over `pieces` **and** recover the
    /// combined secret — the gated path's only public entry, so the §8 attestation
    /// can never be split from the secret's use.
    ///
    /// The same `pieces` (the [`GatedPiece`] hints) are authenticated by
    /// [`Self::verify`] and then opened by `recover_gated` under **this verifier's
    /// pinned binding** — a coordinator cannot authenticate one set and recover another,
    /// nor substitute the binding (the certified `VS`, recipient, roster, or context)
    /// that both the signature check and the final `s·G == VS` self-check use.
    /// Verification runs first, so a bad signature fails closed before any secret is
    /// opened.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] on a degenerate pinned roster/binding;
    /// [`Error::Verification`] if the attestation does not verify, the piece set is
    /// malformed, or the recovered secret fails the certified-`VS` self-check.
    pub fn verify_and_recover_gated(
        &self,
        pieces: &[GatedPiece<'_>],
        signature: &Signature,
        weights: &[Scalar],
        x_rcpt: &Scalar,
    ) -> Result<Zeroizing<Scalar>, Error> {
        let attestation_pieces: Vec<(u32, &RecoveryHint)> =
            pieces.iter().map(|piece| (piece.idx, piece.hint)).collect();
        self.verify(&attestation_pieces, signature)?;
        recover_gated(pieces, weights, x_rcpt, &self.binding)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

    use super::*;
    use crate::signature::frost_test_support::{group_xonly, keygen, sign};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const CTX: &[u8] = b"ve-capsule.kat.ctx";
    const EPOCH: &[u8] = b"ve-capsule.kat.epoch.0001";

    fn recipient_for(x_rcpt: &Scalar) -> ProjectivePoint {
        g() * x_rcpt
    }

    /// A binding owning its points for a test, with [`binding`](Self::binding) to
    /// borrow the [`HintBinding`] view the API takes.
    struct TestBinding {
        y_star: ProjectivePoint,
        vs: ProjectivePoint,
    }

    impl TestBinding {
        fn new(x_rcpt: &Scalar, vs: ProjectivePoint) -> Self {
            Self {
                y_star: recipient_for(x_rcpt),
                vs,
            }
        }

        fn binding(&self) -> HintBinding<'_> {
            HintBinding {
                y_star: &self.y_star,
                vs: &self.vs,
                ctx: CTX,
                epoch: EPOCH,
            }
        }
    }

    fn from_hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn to_hex(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            out.push(char::from_digit(u32::from(b >> 4), 16).unwrap());
            out.push(char::from_digit(u32::from(b & 0x0f), 16).unwrap());
        }
        out
    }

    fn scalar_from_hex(s: &str) -> Scalar {
        let bytes = from_hex(s);
        let mut fb = FieldBytes::default();
        fb.copy_from_slice(&bytes);
        Option::<Scalar>::from(Scalar::from_repr(fb)).unwrap()
    }

    #[test]
    fn single_piece_round_trip() {
        let mut rng = StdRng::seed_from_u64(0x5EC0_0101);
        let x_rcpt = Scalar::random(&mut rng);
        let s = Scalar::random(&mut rng);
        let tb = TestBinding::new(&x_rcpt, g() * s);
        let b = tb.binding();

        let hint = RecoveryHint::seal(&s, &b, 1, &mut rng).unwrap();
        let recovered = hint.open_piece(&x_rcpt, &b, 1).unwrap();
        assert_eq!(*recovered, s);
        let combined = recover_recipient_only(&[(1, hint)], &[Scalar::ONE], &x_rcpt, &b).unwrap();
        assert_eq!(*combined, s);
    }

    #[test]
    fn wire_round_trip_and_fixed_length() {
        let mut rng = StdRng::seed_from_u64(0x5EC0_0102);
        let x_rcpt = Scalar::random(&mut rng);
        let s = Scalar::random(&mut rng);
        let tb = TestBinding::new(&x_rcpt, g() * s);
        let b = tb.binding();
        let hint = RecoveryHint::seal(&s, &b, 7, &mut rng).unwrap();

        let bytes = hint.to_canonical_bytes();
        assert_eq!(bytes.len(), HINT_LEN);
        assert_eq!(HINT_LEN, 65);
        let decoded = RecoveryHint::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded, hint);
        assert_eq!(*decoded.open_piece(&x_rcpt, &b, 7).unwrap(), s);
    }

    #[test]
    fn weighted_multi_piece_recovers() {
        let mut rng = StdRng::seed_from_u64(0x5EC0_0103);
        let x_rcpt = Scalar::random(&mut rng);
        let s1 = Scalar::random(&mut rng);
        let s2 = Scalar::random(&mut rng);
        let weights = [Scalar::from(3u64), Scalar::from(5u64)];
        let s = weights[0] * s1 + weights[1] * s2;
        let tb = TestBinding::new(&x_rcpt, g() * s);
        let b = tb.binding();

        let h1 = RecoveryHint::seal(&s1, &b, 1, &mut rng).unwrap();
        let h2 = RecoveryHint::seal(&s2, &b, 2, &mut rng).unwrap();
        let recovered = recover_recipient_only(&[(1, h1), (2, h2)], &weights, &x_rcpt, &b).unwrap();
        assert_eq!(*recovered, s);
    }

    #[test]
    fn tampered_ct_fails_closed_against_vs() {
        let mut rng = StdRng::seed_from_u64(0x5EC0_0104);
        let x_rcpt = Scalar::random(&mut rng);
        let s = Scalar::random(&mut rng);
        let tb = TestBinding::new(&x_rcpt, g() * s);
        let b = tb.binding();
        let mut hint = RecoveryHint::seal(&s, &b, 1, &mut rng).unwrap();
        hint.ct += Scalar::ONE;
        assert!(matches!(
            recover_recipient_only(&[(1, hint)], &[Scalar::ONE], &x_rcpt, &b),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn wrong_recipient_secret_fails_closed() {
        let mut rng = StdRng::seed_from_u64(0x5EC0_0105);
        let x_rcpt = Scalar::random(&mut rng);
        let s = Scalar::random(&mut rng);
        let tb = TestBinding::new(&x_rcpt, g() * s);
        let b = tb.binding();
        let hint = RecoveryHint::seal(&s, &b, 1, &mut rng).unwrap();
        let wrong = Scalar::random(&mut rng);
        assert!(matches!(
            recover_recipient_only(&[(1, hint)], &[Scalar::ONE], &wrong, &b),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn context_or_epoch_mismatch_fails_closed() {
        let mut rng = StdRng::seed_from_u64(0x5EC0_0106);
        let x_rcpt = Scalar::random(&mut rng);
        let s = Scalar::random(&mut rng);
        let tb = TestBinding::new(&x_rcpt, g() * s);
        let hint = RecoveryHint::seal(&s, &tb.binding(), 1, &mut rng).unwrap();

        // Wrong context and wrong epoch each re-derive a different mask -> wrong secret.
        let wrong_ctx = HintBinding {
            ctx: b"other.ctx",
            ..tb.binding()
        };
        let wrong_epoch = HintBinding {
            epoch: b"other.epoch",
            ..tb.binding()
        };
        assert_ne!(*hint.open_piece(&x_rcpt, &wrong_ctx, 1).unwrap(), s);
        assert_ne!(*hint.open_piece(&x_rcpt, &wrong_epoch, 1).unwrap(), s);
    }

    #[test]
    fn duplicate_and_out_of_order_pieces_rejected() {
        let mut rng = StdRng::seed_from_u64(0x5EC0_0107);
        let x_rcpt = Scalar::random(&mut rng);
        let s1 = Scalar::random(&mut rng);
        let s2 = Scalar::random(&mut rng);
        let weights = [Scalar::ONE, Scalar::ONE];
        let tb = TestBinding::new(&x_rcpt, g() * (s1 + s2));
        let b = tb.binding();
        let h1 = RecoveryHint::seal(&s1, &b, 1, &mut rng).unwrap();
        let h2 = RecoveryHint::seal(&s2, &b, 2, &mut rng).unwrap();

        // Out of order (2, 1).
        assert!(matches!(
            recover_recipient_only(&[(2, h2.clone()), (1, h1.clone())], &weights, &x_rcpt, &b),
            Err(Error::Verification(_))
        ));
        // Duplicate index (1, 1).
        assert!(matches!(
            recover_recipient_only(&[(1, h1), (1, h2)], &weights, &x_rcpt, &b),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn missing_piece_fails_closed() {
        // A 2-piece recovery opened with only one piece (and that subset's weight) does
        // not reconstruct VS -> fail-closed.
        let mut rng = StdRng::seed_from_u64(0x5EC0_0108);
        let x_rcpt = Scalar::random(&mut rng);
        let s1 = Scalar::random(&mut rng);
        let s2 = Scalar::random(&mut rng);
        let tb = TestBinding::new(&x_rcpt, g() * (s1 + s2));
        let b = tb.binding();
        let h1 = RecoveryHint::seal(&s1, &b, 1, &mut rng).unwrap();
        let _h2 = RecoveryHint::seal(&s2, &b, 2, &mut rng).unwrap();
        assert!(matches!(
            recover_recipient_only(&[(1, h1)], &[Scalar::ONE], &x_rcpt, &b),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn poisoned_piece_fails_closed() {
        // One good piece, one piece sealed to a DIFFERENT scalar than VS expects.
        let mut rng = StdRng::seed_from_u64(0x5EC0_0109);
        let x_rcpt = Scalar::random(&mut rng);
        let s1 = Scalar::random(&mut rng);
        let s2 = Scalar::random(&mut rng);
        let weights = [Scalar::ONE, Scalar::ONE];
        let tb = TestBinding::new(&x_rcpt, g() * (s1 + s2));
        let b = tb.binding();
        let h1 = RecoveryHint::seal(&s1, &b, 1, &mut rng).unwrap();
        let poison = Scalar::random(&mut rng);
        let h2 = RecoveryHint::seal(&poison, &b, 2, &mut rng).unwrap();
        assert!(matches!(
            recover_recipient_only(&[(1, h1), (2, h2)], &weights, &x_rcpt, &b),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn fresh_ephemeral_alias_detected() {
        // §7: a hint whose E* equals a limb ephemeral E_j must be detectable.
        let mut rng = StdRng::seed_from_u64(0x5EC0_010A);
        let x_rcpt = Scalar::random(&mut rng);
        let s = Scalar::random(&mut rng);
        let tb = TestBinding::new(&x_rcpt, g() * s);
        let hint = RecoveryHint::seal(&s, &tb.binding(), 1, &mut rng).unwrap();

        let unrelated = g() * Scalar::random(&mut rng);
        assert!(!hint.aliases_any(&[unrelated]));
        // Exact alias E* == E_j.
        assert!(hint.aliases_any(&[unrelated, hint.e_star()]));
        // Inverse alias E* == -E_j: same x(z) under the x-only mask, must also trip.
        assert!(hint.aliases_any(&[unrelated, -hint.e_star()]));
    }

    #[test]
    fn open_side_binding_validation() {
        // open_piece / recover validate the binding too (untrusted-coordinator
        // path), not just seal: identity y_star/vs and over-long ctx/epoch are
        // rejected.
        let mut rng = StdRng::seed_from_u64(0x5EC0_010C);
        let x_rcpt = Scalar::random(&mut rng);
        let s = Scalar::random(&mut rng);
        let tb = TestBinding::new(&x_rcpt, g() * s);
        let hint = RecoveryHint::seal(&s, &tb.binding(), 1, &mut rng).unwrap();

        let id = ProjectivePoint::IDENTITY;
        let bad_vs = HintBinding {
            vs: &id,
            ..tb.binding()
        };
        assert!(matches!(
            hint.open_piece(&x_rcpt, &bad_vs, 1),
            Err(Error::DegenerateInput(_))
        ));
        let oversized = vec![0u8; MAX_DOMAIN_SEP_LEN + 1];
        let bad_ctx = HintBinding {
            ctx: &oversized,
            ..tb.binding()
        };
        assert!(matches!(
            hint.open_piece(&x_rcpt, &bad_ctx, 1),
            Err(Error::DegenerateInput(_))
        ));
        // And at seal.
        assert!(matches!(
            RecoveryHint::seal(&s, &bad_ctx, 1, &mut rng),
            Err(Error::DegenerateInput(_))
        ));
    }

    #[test]
    fn degenerate_inputs_rejected() {
        let mut rng = StdRng::seed_from_u64(0x5EC0_010B);
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let x_rcpt = Scalar::random(&mut rng);
        let y_star = recipient_for(&x_rcpt);

        // Identity recipient key / VS at seal.
        let id = ProjectivePoint::IDENTITY;
        let bad_y = HintBinding {
            y_star: &id,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let bad_vs = HintBinding {
            y_star: &y_star,
            vs: &id,
            ctx: CTX,
            epoch: EPOCH,
        };
        assert!(matches!(
            RecoveryHint::seal(&s, &bad_y, 1, &mut rng),
            Err(Error::DegenerateInput(_))
        ));
        assert!(matches!(
            RecoveryHint::seal(&s, &bad_vs, 1, &mut rng),
            Err(Error::DegenerateInput(_))
        ));
        // Identity E* on decode (33 zero bytes is the canonical identity).
        let mut id_e = [0u8; HINT_LEN];
        id_e[POINT_LEN..].copy_from_slice(&Scalar::ONE.to_bytes());
        assert!(matches!(
            RecoveryHint::from_canonical_bytes(&id_e),
            Err(Error::DegenerateInput(_))
        ));
        // Truncated buffer.
        assert!(matches!(
            RecoveryHint::from_canonical_bytes(&[0u8; 10]),
            Err(Error::PointDecode(_))
        ));
    }

    /// Cross-language known-answer vector pinning the wire format and mask derivation
    /// (design §13 KAT). Any independent implementation must reproduce
    /// these bytes exactly.
    #[test]
    fn cross_language_kat() {
        let s = scalar_from_hex("1111111111111111111111111111111111111111111111111111111111111111");
        let x_rcpt =
            scalar_from_hex("2222222222222222222222222222222222222222222222222222222222222222");
        let r = scalar_from_hex("3333333333333333333333333333333333333333333333333333333333333333");
        let idx = 7u32;
        let y_star = g() * x_rcpt;
        let vs = g() * s; // single unit-weight contribution -> combined VS = s·G.
        let binding = HintBinding {
            y_star: &y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };

        let hint = RecoveryHint::mint_with_ephemeral(&s, &binding, idx, &r).unwrap();
        let bytes = hint.to_canonical_bytes();
        let z = y_star * r;
        let mask = hint_mask(&hint.e_star, &binding, idx, &z).unwrap();

        // The exact framed mask preimage a reimplementer must hash; assert it reduces
        // to the same mask, then pin it.
        let mut preimage = Vec::new();
        push_framed(&mut preimage, HINT_DOMAIN);
        push_framed(&mut preimage, &encode_point(&hint.e_star));
        push_framed(&mut preimage, &encode_point(&y_star));
        push_framed(&mut preimage, &encode_point(&vs));
        push_framed(&mut preimage, CTX);
        push_framed(&mut preimage, EPOCH);
        push_framed(&mut preimage, &idx.to_be_bytes());
        push_framed(&mut preimage, &z.to_affine().x());
        assert_eq!(reduce_be_to_scalar(Sha256::digest(&preimage).into()), mask);
        assert_eq!(
            to_hex(&preimage),
            "0000001b76652d63617073756c652e7265636f766572792d68696e742e763100000021023c72addb4fdf09af94f0c94d7fe92a386a7e70cf8a1d85916386bb2535c7b1b10000002102466d7fcae563e5cb09a0d1870bb580344804617879a14949cf22285f1bae3f2700000021034f355bdcb7cc0af728ef3cceb9615d90684bb5b2ca5f859ab0f0b704075871aa0000001276652d63617073756c652e6b61742e6374780000001976652d63617073756c652e6b61742e65706f63682e303030310000000400000007000000209110f8760a37d96052e3dcaf14862a147654f49f722cf213568ccef1eca2ec71",
            "mask preimage"
        );

        // Frozen vectors — any independent implementation must match these.
        assert_eq!(
            to_hex(&bytes[..POINT_LEN]),
            "023c72addb4fdf09af94f0c94d7fe92a386a7e70cf8a1d85916386bb2535c7b1b1",
            "E*"
        );
        assert_eq!(
            to_hex(&z.to_affine().x()),
            "9110f8760a37d96052e3dcaf14862a147654f49f722cf213568ccef1eca2ec71",
            "x(z)"
        );
        assert_eq!(
            to_hex(&mask.to_bytes()),
            "b05c3d7a3d14ee17940cce247b9be24d59f92a585fc523c8e0a3002afcf7c1ff",
            "mask"
        );
        assert_eq!(
            to_hex(&bytes[POINT_LEN..]),
            "c16d4e8b4e25ff28a51ddf358cacf35e6b0a3b6970d634d9f1b4113c0e08d310",
            "ct"
        );

        let recovered =
            recover_recipient_only(&[(idx, hint)], &[Scalar::ONE], &x_rcpt, &binding).unwrap();
        assert_eq!(*recovered, s);
    }

    // --- Gated open (§5, §8) ---

    /// A `(secret, secret·G)` pair with a full-width random scalar, so no key is
    /// publicly enumerable (the composite-key screen rejects small multiples).
    fn keypair(rng: &mut StdRng) -> (Scalar, ProjectivePoint) {
        let x = Scalar::random(rng);
        (x, g() * x)
    }

    #[test]
    fn gated_single_gate_round_trip() {
        let mut rng = StdRng::seed_from_u64(0x6A7E_0001);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let (x_a, gate_a) = keypair(&mut rng);
        let access = [gate_a];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 1u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();

        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let contribution = AuthorizerContribution::contribute(
            &x_a,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        let quorums = [GateQuorum::new(gate_a, vec![contribution])];

        let opened = hint
            .open_piece_gated(&x_rcpt, &gated, idx, &quorums)
            .unwrap();
        assert_eq!(*opened, s);

        let piece = GatedPiece {
            idx,
            hint: &hint,
            quorums: &quorums,
        };
        let recovered = recover_gated(&[piece], &[Scalar::ONE], &x_rcpt, &gated).unwrap();
        assert_eq!(*recovered, s);
    }

    #[test]
    fn gated_two_gate_and_round_trip() {
        let mut rng = StdRng::seed_from_u64(0x6A7E_0002);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let (x_a, gate_a) = keypair(&mut rng);
        let (x_b, gate_b) = keypair(&mut rng);
        let access = [gate_a, gate_b];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 4u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let c_a = AuthorizerContribution::contribute(
            &x_a,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        let c_b = AuthorizerContribution::contribute(
            &x_b,
            &hint.e_star(),
            &gated,
            idx,
            &gate_b,
            &mut rng,
        )
        .unwrap();
        let quorums = [
            GateQuorum::new(gate_a, vec![c_a.clone()]),
            GateQuorum::new(gate_b, vec![c_b.clone()]),
        ];
        assert_eq!(
            *hint
                .open_piece_gated(&x_rcpt, &gated, idx, &quorums)
                .unwrap(),
            s
        );

        // Quorum list order does not matter (gates are matched by point).
        let swapped = [
            GateQuorum::new(gate_b, vec![c_b]),
            GateQuorum::new(gate_a, vec![c_a]),
        ];
        assert_eq!(
            *hint
                .open_piece_gated(&x_rcpt, &gated, idx, &swapped)
                .unwrap(),
            s
        );
    }

    #[test]
    fn gated_bucket_threshold_round_trip() {
        // One gate Y_k = (w1 + w2)·G satisfied by a two-authorizer sub-quorum.
        let mut rng = StdRng::seed_from_u64(0x6A7E_0003);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let w1 = Scalar::random(&mut rng);
        let w2 = Scalar::random(&mut rng);
        let gate = g() * (w1 + w2);
        let access = [gate];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 2u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let c1 =
            AuthorizerContribution::contribute(&w1, &hint.e_star(), &gated, idx, &gate, &mut rng)
                .unwrap();
        let c2 =
            AuthorizerContribution::contribute(&w2, &hint.e_star(), &gated, idx, &gate, &mut rng)
                .unwrap();
        let quorums = [GateQuorum::new(gate, vec![c1, c2])];
        assert_eq!(
            *hint
                .open_piece_gated(&x_rcpt, &gated, idx, &quorums)
                .unwrap(),
            s
        );
    }

    #[test]
    fn gated_short_quorum_fails_closed() {
        let mut rng = StdRng::seed_from_u64(0x6A7E_0004);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let (x_a, gate_a) = keypair(&mut rng);
        let (_x_b, gate_b) = keypair(&mut rng);
        let access = [gate_a, gate_b];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 6u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };

        // Only gate_a's quorum -> count mismatch with the 2-gate roster.
        let c_a = AuthorizerContribution::contribute(
            &x_a,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        let short = [GateQuorum::new(gate_a, vec![c_a])];
        assert!(matches!(
            hint.open_piece_gated(&x_rcpt, &gated, idx, &short),
            Err(Error::Verification(_))
        ));

        // Both gates present but gate_b's bucket is empty.
        let c_a2 = AuthorizerContribution::contribute(
            &x_a,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        let with_empty = [
            GateQuorum::new(gate_a, vec![c_a2]),
            GateQuorum::new(gate_b, vec![]),
        ];
        assert!(matches!(
            hint.open_piece_gated(&x_rcpt, &gated, idx, &with_empty),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn gated_wrong_sum_quorum_fails_closed() {
        // A consistent (DLEQ-valid) contribution whose w ≠ dlog(gate): the strict-AND
        // Σ W == Y_k catches it before the mask is even formed.
        let mut rng = StdRng::seed_from_u64(0x6A7E_0005);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let (_x_a, gate_a) = keypair(&mut rng);
        let access = [gate_a];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 7u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let wrong_w = Scalar::random(&mut rng);
        let bad = AuthorizerContribution::contribute(
            &wrong_w,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        let quorums = [GateQuorum::new(gate_a, vec![bad])];
        assert!(matches!(
            hint.open_piece_gated(&x_rcpt, &gated, idx, &quorums),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn gated_tampered_w_estar_caught_by_dleq() {
        // Flip W* so it is no longer w·E*. The DLEQ over [G, E*] rejects it even
        // though the W (= w·G) part still satisfies the strict-AND.
        let mut rng = StdRng::seed_from_u64(0x6A7E_0006);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let (x_a, gate_a) = keypair(&mut rng);
        let access = [gate_a];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 8u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let mut c = AuthorizerContribution::contribute(
            &x_a,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        c.w_estar += g();
        let quorums = [GateQuorum::new(gate_a, vec![c])];
        assert!(matches!(
            hint.open_piece_gated(&x_rcpt, &gated, idx, &quorums),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn gated_contribution_replayed_across_gates_rejected() {
        // A contribution bound to gate A, presented for gate B, fails its DLEQ
        // (the gate is in the binding).
        let mut rng = StdRng::seed_from_u64(0x6A7E_0007);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let (x_a, gate_a) = keypair(&mut rng);
        let (x_b, gate_b) = keypair(&mut rng);
        let access = [gate_a, gate_b];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 9u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let c_a = AuthorizerContribution::contribute(
            &x_a,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        let c_b = AuthorizerContribution::contribute(
            &x_b,
            &hint.e_star(),
            &gated,
            idx,
            &gate_b,
            &mut rng,
        )
        .unwrap();
        // Present each contribution under the *other* gate.
        let swapped = [
            GateQuorum::new(gate_a, vec![c_b]),
            GateQuorum::new(gate_b, vec![c_a]),
        ];
        assert!(matches!(
            hint.open_piece_gated(&x_rcpt, &gated, idx, &swapped),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn gated_empty_roster_matches_recipient_only() {
        // An empty roster is the recipient-only degenerate case: Y* == Y_rcpt and the
        // gated open reduces bit-for-bit to the recipient-only open.
        let mut rng = StdRng::seed_from_u64(0x6A7E_0008);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let access: [ProjectivePoint; 0] = [];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        assert_eq!(opening.y_star, recipient);
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 3u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let gated_open = hint.open_piece_gated(&x_rcpt, &gated, idx, &[]).unwrap();
        let recipient_only = hint.open_piece(&x_rcpt, &seal_binding, idx).unwrap();
        assert_eq!(*gated_open, *recipient_only);
        assert_eq!(*gated_open, s);
    }

    #[test]
    fn coordinator_handles_public_only() {
        // The untrusted coordinator runs the DLEQs + strict-AND and produces the
        // public S_gate; the recipient finishes with only its own x_rcpt.
        let mut rng = StdRng::seed_from_u64(0x6A7E_0009);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let (x_a, gate_a) = keypair(&mut rng);
        let access = [gate_a];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 5u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let c_a = AuthorizerContribution::contribute(
            &x_a,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        let quorums = [GateQuorum::new(gate_a, vec![c_a])];

        // Host: public only.
        let s_gate = public_gate_sum(&hint.e_star(), &gated, idx, &quorums).unwrap();
        // Device: the lone secret is x_rcpt.
        let recipient_term = (hint.e_star() * x_rcpt) * opening.recipient_weight;
        let z = recipient_term + s_gate;
        let mask = hint_mask(&hint.e_star(), &seal_binding, idx, &z).unwrap();
        assert_eq!(hint.ct - mask, s);

        // The all-in-one path agrees.
        assert_eq!(
            *hint
                .open_piece_gated(&x_rcpt, &gated, idx, &quorums)
                .unwrap(),
            s
        );
    }

    #[test]
    fn gated_wrong_recipient_secret_fails_closed() {
        // A valid quorum but the wrong recipient secret: recover_gated's s·G == VS
        // self-check fails closed.
        let mut rng = StdRng::seed_from_u64(0x6A7E_000A);
        let (_x_rcpt, recipient) = keypair(&mut rng);
        let (x_a, gate_a) = keypair(&mut rng);
        let access = [gate_a];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 1u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let c_a = AuthorizerContribution::contribute(
            &x_a,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        let quorums = [GateQuorum::new(gate_a, vec![c_a])];
        let piece = GatedPiece {
            idx,
            hint: &hint,
            quorums: &quorums,
        };
        let wrong = Scalar::random(&mut rng);
        assert!(matches!(
            recover_gated(&[piece], &[Scalar::ONE], &wrong, &gated),
            Err(Error::Verification(_))
        ));
    }

    // --- Hint-binding attestation (§8, R5) ---

    /// A single-gate gated recovery fixture: returns its `(recipient, access roster,
    /// certified VS, piece-1 hint)`. Callers borrow the owned values to build a
    /// [`GatedBinding`].
    fn single_gate_case(
        rng: &mut StdRng,
    ) -> (
        ProjectivePoint,
        [ProjectivePoint; 1],
        ProjectivePoint,
        RecoveryHint,
    ) {
        let recipient = g() * Scalar::random(&mut *rng);
        let access = [g() * Scalar::random(&mut *rng)];
        let s = Scalar::random(&mut *rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let hint = RecoveryHint::seal(&s, &seal_binding, 1, rng).unwrap();
        (recipient, access, vs, hint)
    }

    #[test]
    fn hint_attestation_round_trip() {
        let mut rng = StdRng::seed_from_u64(0x517E_0001);
        let (recipient, access, vs, hint) = single_gate_case(&mut rng);
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let pieces = [(1u32, &hint)];
        let msg = hint_attestation_message(&pieces, &gated).unwrap();

        let (packages, pubkeys) = keygen(2, 3);
        let qk = group_xonly(&pubkeys);
        let sig = Signature::schnorr(sign(&packages, &pubkeys, &msg));

        let verifier = PinnedHintVerifier {
            quorum_key: &qk,
            binding: gated,
        };
        assert!(verifier.verify(&pieces, &sig).is_ok());
    }

    #[test]
    fn hint_attestation_rejects_tampered_piece() {
        let mut rng = StdRng::seed_from_u64(0x517E_0002);
        let (recipient, access, vs, hint) = single_gate_case(&mut rng);
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let pieces = [(1u32, &hint)];
        let msg = hint_attestation_message(&pieces, &gated).unwrap();
        let (packages, pubkeys) = keygen(2, 3);
        let qk = group_xonly(&pubkeys);
        let sig = Signature::schnorr(sign(&packages, &pubkeys, &msg));

        // Flip the masked contribution of the presented piece -> statement differs -> reject.
        let mut tampered = hint.clone();
        tampered.ct += Scalar::ONE;
        let tampered_pieces = [(1u32, &tampered)];
        let verifier = PinnedHintVerifier {
            quorum_key: &qk,
            binding: gated,
        };
        assert!(matches!(
            verifier.verify(&tampered_pieces, &sig),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn hint_attestation_rejects_wrong_quorum_key() {
        let mut rng = StdRng::seed_from_u64(0x517E_0003);
        let (recipient, access, vs, hint) = single_gate_case(&mut rng);
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let pieces = [(1u32, &hint)];
        let msg = hint_attestation_message(&pieces, &gated).unwrap();
        let (packages, pubkeys) = keygen(2, 3);
        let sig = Signature::schnorr(sign(&packages, &pubkeys, &msg));

        // A different group's key never signed this statement.
        let (_p2, pubkeys2) = keygen(2, 3);
        let wrong_qk = group_xonly(&pubkeys2);
        let verifier = PinnedHintVerifier {
            quorum_key: &wrong_qk,
            binding: gated,
        };
        assert!(matches!(
            verifier.verify(&pieces, &sig),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn hint_attestation_binds_certified_vs() {
        // A verifier pinned to a different VS than the quorum signed rebuilds a
        // different statement and rejects — the coordinator cannot substitute the
        // certified VS.
        let mut rng = StdRng::seed_from_u64(0x517E_0004);
        let (recipient, access, vs, hint) = single_gate_case(&mut rng);
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let pieces = [(1u32, &hint)];
        let msg = hint_attestation_message(&pieces, &gated).unwrap();
        let (packages, pubkeys) = keygen(2, 3);
        let qk = group_xonly(&pubkeys);
        let sig = Signature::schnorr(sign(&packages, &pubkeys, &msg));

        let other_vs = g() * Scalar::random(&mut rng);
        let gated2 = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &other_vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let verifier = PinnedHintVerifier {
            quorum_key: &qk,
            binding: gated2,
        };
        assert!(matches!(
            verifier.verify(&pieces, &sig),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn hint_attestation_binds_piece_set() {
        // Adding a piece not covered by the signature changes the statement -> reject.
        let mut rng = StdRng::seed_from_u64(0x517E_0005);
        let (recipient, access, vs, hint) = single_gate_case(&mut rng);
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let pieces = [(1u32, &hint)];
        let msg = hint_attestation_message(&pieces, &gated).unwrap();
        let (packages, pubkeys) = keygen(2, 3);
        let qk = group_xonly(&pubkeys);
        let sig = Signature::schnorr(sign(&packages, &pubkeys, &msg));
        let verifier = PinnedHintVerifier {
            quorum_key: &qk,
            binding: gated,
        };

        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let extra_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let extra =
            RecoveryHint::seal(&Scalar::random(&mut rng), &extra_binding, 2, &mut rng).unwrap();
        let two = [(1u32, &hint), (2u32, &extra)];
        assert!(matches!(
            verifier.verify(&two, &sig),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn hint_attestation_binds_roster() {
        // A verifier pinned to a different roster rebuilds different g*/Y* -> reject.
        let mut rng = StdRng::seed_from_u64(0x517E_0006);
        let (recipient, access, vs, hint) = single_gate_case(&mut rng);
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let pieces = [(1u32, &hint)];
        let msg = hint_attestation_message(&pieces, &gated).unwrap();
        let (packages, pubkeys) = keygen(2, 3);
        let qk = group_xonly(&pubkeys);
        let sig = Signature::schnorr(sign(&packages, &pubkeys, &msg));

        let other_access = [g() * Scalar::random(&mut rng)];
        let gated2 = GatedBinding {
            recipient: &recipient,
            access_keys: &other_access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let verifier = PinnedHintVerifier {
            quorum_key: &qk,
            binding: gated2,
        };
        assert!(matches!(
            verifier.verify(&pieces, &sig),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn hint_attestation_rejects_duplicate_index() {
        let mut rng = StdRng::seed_from_u64(0x517E_0007);
        let (recipient, access, vs, hint) = single_gate_case(&mut rng);
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let dup = [(1u32, &hint), (1u32, &hint)];
        assert!(matches!(
            hint_attestation_message(&dup, &gated),
            Err(Error::Verification(_))
        ));
    }

    // --- Atomic verified gated recovery (1b ↔ 1c seam) ---

    #[test]
    fn verified_recovery_round_trip() {
        // Full Phase-1 e2e: sealed hint (1a) + gate quorum (1b) + quorum attestation
        // (1c), recovered atomically through the pinned verifier.
        let mut rng = StdRng::seed_from_u64(0x5EA1_0001);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let (x_a, gate_a) = keypair(&mut rng);
        let access = [gate_a];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 1u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let c_a = AuthorizerContribution::contribute(
            &x_a,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        let quorums = [GateQuorum::new(gate_a, vec![c_a])];

        let attest_pieces = [(idx, &hint)];
        let msg = hint_attestation_message(&attest_pieces, &gated).unwrap();
        let (packages, pubkeys) = keygen(2, 3);
        let qk = group_xonly(&pubkeys);
        let sig = Signature::schnorr(sign(&packages, &pubkeys, &msg));

        let verifier = PinnedHintVerifier {
            quorum_key: &qk,
            binding: gated,
        };
        let gated_pieces = [GatedPiece {
            idx,
            hint: &hint,
            quorums: &quorums,
        }];
        let recovered = verifier
            .verify_and_recover_gated(&gated_pieces, &sig, &[Scalar::ONE], &x_rcpt)
            .unwrap();
        assert_eq!(*recovered, s);
    }

    #[test]
    fn verified_recovery_rejects_tampered_piece() {
        let mut rng = StdRng::seed_from_u64(0x5EA1_0002);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let (x_a, gate_a) = keypair(&mut rng);
        let access = [gate_a];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 1u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let c_a = AuthorizerContribution::contribute(
            &x_a,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        let quorums = [GateQuorum::new(gate_a, vec![c_a])];
        let attest_pieces = [(idx, &hint)];
        let msg = hint_attestation_message(&attest_pieces, &gated).unwrap();
        let (packages, pubkeys) = keygen(2, 3);
        let qk = group_xonly(&pubkeys);
        let sig = Signature::schnorr(sign(&packages, &pubkeys, &msg));
        let verifier = PinnedHintVerifier {
            quorum_key: &qk,
            binding: gated,
        };

        // ct tamper (E* unchanged, so the quorum still matches) is caught by the
        // attestation before any secret is opened.
        let mut bad = hint.clone();
        bad.ct += Scalar::ONE;
        let bad_pieces = [GatedPiece {
            idx,
            hint: &bad,
            quorums: &quorums,
        }];
        assert!(matches!(
            verifier.verify_and_recover_gated(&bad_pieces, &sig, &[Scalar::ONE], &x_rcpt),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn verified_recovery_rejects_wrong_pinned_vs() {
        let mut rng = StdRng::seed_from_u64(0x5EA1_0003);
        let (x_rcpt, recipient) = keypair(&mut rng);
        let (x_a, gate_a) = keypair(&mut rng);
        let access = [gate_a];
        let s = Scalar::random(&mut rng);
        let vs = g() * s;
        let opening = composite::opening_binding(&recipient, &access).unwrap();
        let seal_binding = HintBinding {
            y_star: &opening.y_star,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let idx = 1u32;
        let hint = RecoveryHint::seal(&s, &seal_binding, idx, &mut rng).unwrap();
        let gated = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let c_a = AuthorizerContribution::contribute(
            &x_a,
            &hint.e_star(),
            &gated,
            idx,
            &gate_a,
            &mut rng,
        )
        .unwrap();
        let quorums = [GateQuorum::new(gate_a, vec![c_a])];
        let attest_pieces = [(idx, &hint)];
        let msg = hint_attestation_message(&attest_pieces, &gated).unwrap();
        let (packages, pubkeys) = keygen(2, 3);
        let qk = group_xonly(&pubkeys);
        let sig = Signature::schnorr(sign(&packages, &pubkeys, &msg));

        // A verifier pinned to a different VS than the quorum signed: the rebuilt
        // statement differs, so verification fails before recovery.
        let other_vs = g() * Scalar::random(&mut rng);
        let gated2 = GatedBinding {
            recipient: &recipient,
            access_keys: &access,
            vs: &other_vs,
            ctx: CTX,
            epoch: EPOCH,
        };
        let verifier = PinnedHintVerifier {
            quorum_key: &qk,
            binding: gated2,
        };
        let gated_pieces = [GatedPiece {
            idx,
            hint: &hint,
            quorums: &quorums,
        }];
        assert!(matches!(
            verifier.verify_and_recover_gated(&gated_pieces, &sig, &[Scalar::ONE], &x_rcpt),
            Err(Error::Verification(_))
        ));
    }
}
