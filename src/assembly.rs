//! M1 assembly: `seal` → `verify` → `open` over the full proof system.
//!
//! Ties the sub-proofs into one non-interactive package on one linear
//! multi-squeeze transcript (§2). For a scalar `m ∈ [0, n)` and recovery key
//! `pk`, [`seal`] segments `m` into `L` limbs, and per limb publishes a
//! curve-`ElGamal` ciphertext `(E_k, D_k)` and Pedersen commitments `Com_k` /
//! `Com̄_k` (for `m̄ = n−1−m`); the carry commitments `ComC_k` complete the
//! statement. One aggregated BP++ range proof (§4.1) then bounds every limb
//! and proves every carry boolean (items 19–22, challenges
//! `α; ρ, λ, β, δ; τ; γ_i`); the carry residual Schnorrs (§4.2) and linking
//! sigma (§4.3) close under the final squeeze `x` (items 24–25 → `sigma.x`).
//! The absorption order is normative (soundness-doc §3) and shared by `seal`
//! and `verify` through [`statement_transcript`] / [`sigma_challenge`] — they
//! cannot diverge.
//!
//! [`verify`] takes the commitment `C` as the proof statement; the `C == T`
//! target binding is enforced one layer up by [`crate::Capsule::verify`] (this
//! proof-only check is its building block). [`open`] decrypts each limb
//! (`D_k − sk·E_k = v_k·G`, then BSGS), recomposes `m`, and rechecks `m·G == C`.

use crate::batch_affine::{BatchAddScratch, FePoint, batch_add_points_visit};
use crate::bsgs::{
    any_signed_fe_g_multiple_below, any_signed_g_multiple_below, baby_table,
    is_public_recovery_key_scalar_multiple, is_signed_g_multiple_below,
};
use crate::carry::{self, CarryCommitment, CarryResponses};
use crate::codec::{POINT_LEN, decode_point, encode_affine_point, encode_point};
use crate::context::Context;
use crate::elgamal::{IDENTITY_MASK_DETAIL, LimbCiphertext, encode_limb, reject_identity_mask};
use crate::error::Error;
use crate::generators::{g, h};
use crate::limbs::{LIMB_COUNT, LIMB_MODULUS, decompose, limb_weights, recompose};
use crate::linking::{self, LinkingCommitment, LinkingResponses};
use crate::norm_arg::NormProof;
use crate::parallel::{parallel_any, parallel_map_indexed, worker_count};
use crate::params::Params;
use crate::pedersen::Commitment;
use crate::range_circuit::{
    self, FOLD_ROUNDS, RESIDUAL_L, RESIDUAL_N, RangeProof, RangeWitness, TranscriptChallenges,
};
use crate::transcript::{Transcript, push_framed};
use k256::elliptic_curve::PrimeField;
use k256::elliptic_curve::point::{AffineCoordinates, BatchNormalize};
use k256::{FieldBytes, ProjectivePoint, Scalar};
use rand_core::{CryptoRng, RngCore};
use std::borrow::Cow;
use std::sync::{Mutex, MutexGuard, OnceLock};
use zeroize::Zeroizing;

/// Frozen segmentation tuple this module is built against.
const PARAMS: Params = Params::FROZEN;

/// Statement version absorbed as transcript item 2 (bump on any wire change).
const STATEMENT_VERSION: u8 = 1;

/// Challenge width absorbed as transcript item 3 (pins the §2 full-32-byte
/// reduce decision into the challenge).
const CHALLENGE_WIDTH: u16 = 256;

/// Signed-multiple window for the recovery-key enumerability checks. A small
/// `pk = c·G` with `|c|` in this window has a *public* discrete log, so every
/// verifier could decrypt the package (secrecy collapse); a small `pk = c·H`
/// (NUMS `H`, unknown dlog) instead yields a package that verifies but is
/// recoverable by no signer (availability). Both are rejected.
const RECOVERY_KEY_ENUMERABILITY_BOUND: u16 = 4096;

/// Public `G` multiples just above the limb domain are still public nonces.
/// The limb BSGS table intentionally stops at `2^ell - 1`, so this window
/// closes the boundary tail of the recovery-key enumerability check.
const PUBLIC_G_BOUNDARY_WINDOW_BOUND: u16 = RECOVERY_KEY_ENUMERABILITY_BOUND;

/// Coefficient window for `ElGamal` mask relation hardening. Coefficients in this
/// window catch low-support public linear equations between segment masks before
/// they leak bounded limb relations through matching `D`-point combinations.
const MASK_RELATION_COEFFICIENT_BOUND: u16 = 2;

/// Maximum Case pieces admitted to exhaustive cross-piece mask relation
/// screening. The global support-six unit scan streams three-term halves, so it
/// shares the constrained-device cap used by Pedersen subset screening.
///
/// Also the piece-count cap for the recovery-hint cross-scheme screen
/// ([`crate::provision::validate_recovery_hints_against_capsules`]), which
/// flattens to two pieces and so cannot rely on the piece guard inside the scan.
pub const CROSS_PIECE_MASK_RELATION_PIECE_BOUND: usize = 6;

/// Maximum proof-backed Case pieces admitted to exhaustive cross-piece
/// Pedersen relation screening. Unlike the `ElGamal` mask detector, this scanner
/// must test bounded public-`G` offsets for every signed same-slot subset, so it
/// has its own constrained-device cap and fails closed above it.
const CROSS_PIECE_PEDERSEN_RELATION_PIECE_BOUND: usize = 6;

/// Cross-piece coefficient-two relation support scanned with pair-vs-pair
/// meet-in-the-middle. Support five would stream coefficient-two triples and is
/// deliberately outside the constrained-device cold path.
const CROSS_PIECE_SMALL_COEFF_RELATION_SUPPORT_BOUND: usize = 4;

/// Unit-coefficient support scanned globally across Case pieces. Support six is
/// the largest profile that still uses three-term halves; support seven would
/// require streaming four-term halves on the constrained-device cold path.
const CROSS_PIECE_UNIT_RELATION_SUPPORT_BOUND: usize = 6;

/// Largest half-combination this module streams for the cross-piece mask scan.
const CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND: usize = 3;

/// Pedersen limb commitments should not be publicly openable as small `G`
/// multiples. Values and complements are `< 2^ell`; carry commitments are
/// smaller, so the limb bound covers every statement commitment slot.
const PEDERSEN_COMMITMENT_PUBLIC_G_BOUND: u64 = LIMB_MODULUS;

/// Positive public-key screening cache size. Successful checks are safe to reuse
/// for stable recipient/access/composite keys; invalid points are never cached.
const PUBLIC_KEY_SCREENING_CACHE_CAP: usize = 256;

/// Maximum caller context domain bytes accepted by the EC transcript.
const MAX_CONTEXT_DOMAIN_BYTES: usize = 256;

/// Maximum caller binding payload bytes accepted by the EC transcript.
const MAX_CONTEXT_BINDING_BYTES: usize = 64 * 1024;

/// The secp256k1 group order `n`, 32-byte big-endian (transcript item 8). Fixed
/// public constant; a `Params` swap that changed the curve would fork the
/// challenge here. Verified against `−1 = n−1` by `order_constant_is_n`.
const N_BE: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B, 0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41,
];

/// A complete ec-segve recovery package: the public ciphertexts, commitments,
/// and the three sub-proofs, all under one shared challenge.
#[derive(Clone, Debug)]
pub struct Proof {
    /// Per-limb curve-`ElGamal` ciphertexts `(E_k, D_k)`.
    elgamal: Vec<LimbCiphertext>,
    /// Value limb commitments `Com_k` (minted by the value range proofs).
    value_commitments: Vec<ProjectivePoint>,
    /// Complement limb commitments `Com̄_k`.
    complement_commitments: Vec<ProjectivePoint>,
    /// The aggregated BP++ range proof bounding every limb (value and
    /// complement) and proving every carry boolean (§4.1).
    range: RangeProof,
    /// The carry chain proving `m + m̄ = n − 1` (so `m ≤ n − 1`).
    carry: CarryCommitment,
    carry_resp: CarryResponses,
    /// The linking sigma binding the limbs to `(E_k, D_k)`, `Com_k`, and `C`.
    linking: LinkingCommitment,
    linking_resp: LinkingResponses,
}

fn push_point(out: &mut Vec<u8>, p: &ProjectivePoint) {
    out.extend_from_slice(&encode_point(p));
}

fn push_scalar(out: &mut Vec<u8>, s: &Scalar) {
    out.extend_from_slice(&s.to_bytes());
}

impl Proof {
    /// The per-limb curve-`ElGamal` ciphertexts `(E_k, D_k)` — the capsule's
    /// opening core. The opening layer builds a [`crate::opening::CapsuleRef`]
    /// from this slice plus `C`; the rest of the proof is not read to open, so a
    /// proof-stripped capsule presents the same core view.
    pub(crate) fn elgamal(&self) -> &[LimbCiphertext] {
        &self.elgamal
    }

    /// Public value-limb Pedersen commitments `Com_k = v_k*G + s_k*H`.
    #[cfg(test)]
    pub(crate) fn value_commitments(&self) -> &[ProjectivePoint] {
        &self.value_commitments
    }

    /// Canonical wire bytes of the proof: every point as 33-byte SEC1, every
    /// scalar as 32-byte big-endian, in a fixed field order. Encode-only and
    /// length-fixed for the frozen params; the matching decoder belongs to the
    /// framework wire layer. Does not include `C` (the statement) — the backend
    /// ciphertext prepends it.
    #[must_use]
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for ct in &self.elgamal {
            encode_limb(ct, &mut out);
        }
        for p in &self.value_commitments {
            push_point(&mut out, p);
        }
        for p in &self.complement_commitments {
            push_point(&mut out, p);
        }
        for p in [
            &self.range.c_l,
            &self.range.c_o,
            &self.range.c_r,
            &self.range.c_s,
        ] {
            push_point(&mut out, p);
        }
        for p in self.range.folds.x.iter().chain(self.range.folds.r.iter()) {
            push_point(&mut out, p);
        }
        for sc in self.range.folds.l.iter().chain(self.range.folds.n.iter()) {
            push_scalar(&mut out, sc);
        }
        for p in &self.carry.carry_commitments {
            push_point(&mut out, p);
        }
        for p in &self.carry.residual_announcements {
            push_point(&mut out, p);
        }
        for s in &self.carry_resp.residual_responses {
            push_scalar(&mut out, s);
        }
        for legs in [&self.linking.a_e, &self.linking.a_d, &self.linking.a_com] {
            for p in legs {
                push_point(&mut out, p);
            }
        }
        push_point(&mut out, &self.linking.a_c);
        for legs in [
            &self.linking_resp.z_v,
            &self.linking_resp.z_r,
            &self.linking_resp.z_s,
        ] {
            for s in legs {
                push_scalar(&mut out, s);
            }
        }
        out
    }

    /// Parse a proof from the canonical bytes produced by [`Proof::to_canonical_bytes`].
    ///
    /// The encoding is fixed-layout (every count derives from the frozen params),
    /// so this reads exactly that layout: each point via the strict
    /// [`decode_point`] (rejecting off-curve / non-canonical SEC1) and each scalar
    /// as 32 big-endian bytes rejected unless canonical (`< n`) — the
    /// soundness-doc §1 obligation against `e+n`/`z+n` malleability. Trailing or
    /// short input is rejected.
    ///
    /// # Errors
    ///
    /// [`Error::PointDecode`] on a malformed point, a non-canonical scalar, or a
    /// length mismatch (short/trailing bytes). [`Error::DegenerateInput`] if a
    /// segment mask `E_k` decodes to the identity (`r_k = 0`) — that gate fires
    /// at the first degenerate limb, taking precedence over a later length
    /// error in the same input.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let limbs = LIMB_COUNT;
        let mut reader = Reader::new(bytes);

        let mut elgamal = Vec::with_capacity(limbs);
        for _ in 0..limbs {
            // Identity-mask gate at the decode boundary (soundness-doc §4.4
            // "Retained mask gates"): E_k = O means r_k = 0, so D_k = v_k·G is a
            // public plaintext limb. Checked before reading D_k so it takes
            // precedence over a later truncation; the shared gate is also applied
            // by the stripped-core decoder, so the soundness check has one home.
            let mask = reader.point()?;
            reject_identity_mask(&mask)?;
            let masked = reader.point()?;
            elgamal.push(LimbCiphertext { e: mask, d: masked });
        }
        let value_commitments = reader.points(limbs)?;
        let complement_commitments = reader.points(limbs)?;
        // The BP++ artifact is fixed-shape (soundness-doc §7: zero length
        // fields — every count below is a frozen constant).
        let range = RangeProof {
            c_l: reader.point()?,
            c_o: reader.point()?,
            c_r: reader.point()?,
            c_s: reader.point()?,
            folds: NormProof {
                x: reader.points(FOLD_ROUNDS)?,
                r: reader.points(FOLD_ROUNDS)?,
                l: reader.scalars(RESIDUAL_L)?,
                n: reader.scalars(RESIDUAL_N)?,
            },
        };
        let carry = CarryCommitment {
            carry_commitments: reader.points(limbs - 1)?,
            residual_announcements: reader.points(limbs)?,
        };
        let carry_resp = CarryResponses {
            residual_responses: reader.scalars(limbs)?,
        };
        let linking = LinkingCommitment {
            a_e: reader.points(limbs)?,
            a_d: reader.points(limbs)?,
            a_com: reader.points(limbs)?,
            a_c: reader.point()?,
        };
        let linking_resp = LinkingResponses {
            z_v: reader.scalars(limbs)?,
            z_r: reader.scalars(limbs)?,
            z_s: reader.scalars(limbs)?,
        };
        reader.finish()?;

        Ok(Self {
            elgamal,
            value_commitments,
            complement_commitments,
            range,
            carry,
            carry_resp,
            linking,
            linking_resp,
        })
    }
}

/// Width of a canonical scalar on the wire (32-byte big-endian).
const SCALAR_LEN: usize = 32;

/// A forward cursor over canonical proof bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or(Error::PointDecode("proof bytes truncated"))?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn point(&mut self) -> Result<ProjectivePoint, Error> {
        decode_point(self.take(POINT_LEN)?)
    }

    fn points(&mut self, n: usize) -> Result<Vec<ProjectivePoint>, Error> {
        (0..n).map(|_| self.point()).collect()
    }

    fn scalar(&mut self) -> Result<Scalar, Error> {
        let mut repr = FieldBytes::default();
        repr.copy_from_slice(self.take(SCALAR_LEN)?);
        Option::from(Scalar::from_repr(repr))
            .ok_or(Error::PointDecode("non-canonical scalar (>= n)"))
    }

    fn scalars(&mut self, n: usize) -> Result<Vec<Scalar>, Error> {
        (0..n).map(|_| self.scalar()).collect()
    }

    const fn finish(self) -> Result<(), Error> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::PointDecode("trailing bytes after proof"))
        }
    }
}

/// Borrowed view of the statement the transcript absorbs before any challenge
/// (soundness-doc §3 items 1–18). Built identically by `seal` (from the fresh
/// commitments) and `verify` (from the `Proof`), so the shared
/// [`statement_transcript`] cannot diverge between them.
struct TranscriptInputs<'a> {
    c: &'a ProjectivePoint,
    pk: &'a ProjectivePoint,
    elgamal: &'a [LimbCiphertext],
    value_commitments: &'a [ProjectivePoint],
    complement_commitments: &'a [ProjectivePoint],
    carry_commitments: &'a [ProjectivePoint],
}

pub fn degenerate_elgamal_mask(ciphertexts: &[LimbCiphertext]) -> Option<&'static str> {
    if let Some(detail) = degenerate_elgamal_mask_basic(ciphertexts) {
        return Some(detail);
    }
    if has_pairwise_elgamal_mask_scalar_relation(ciphertexts) {
        return Some("ElGamal masks have a public scalar relation");
    }
    // The small-coefficient scan runs the same relation engine at coefficient
    // bound 2, whose signed enumeration strictly contains the bound-1 signed
    // subsets, so it subsumes a separate signed-subset (bound-1) pass.
    if has_small_coefficient_elgamal_mask_relation(ciphertexts) {
        return Some("ElGamal masks have a small-coefficient relation");
    }
    if has_elgamal_mask_public_g_offset_relation(ciphertexts) {
        return Some("ElGamal masks have a public G-offset relation");
    }
    None
}

fn degenerate_elgamal_mask_basic(ciphertexts: &[LimbCiphertext]) -> Option<&'static str> {
    let mut seen = std::collections::HashSet::with_capacity(ciphertexts.len());
    for lhs in ciphertexts {
        // E_k = identity ⇒ r_k = 0 ⇒ D_k = v_k·G is a public plaintext limb.
        if lhs.e == ProjectivePoint::IDENTITY {
            return Some(IDENTITY_MASK_DETAIL);
        }
        // E_i = ±E_j cancels the recovery-key term in D_i ∓ D_j and leaks a
        // bounded limb relation to every observer, even though the proof algebra
        // can still be internally valid.
        let encoded = encode_point(&lhs.e);
        let inverted = encode_point(&(-lhs.e));
        if seen.contains(&encoded) {
            return Some("ElGamal mask repeats a previous mask");
        }
        if seen.contains(&inverted) {
            return Some("ElGamal mask inverts a previous mask");
        }
        seen.insert(encoded);
    }
    None
}

#[derive(Clone, Copy)]
struct MaskOwners {
    first_piece: usize,
    spans_multiple_pieces: bool,
}

#[derive(Clone, Copy)]
struct MaskXOwners {
    first_mask_index: usize,
    first_piece: usize,
    has_other_mask: bool,
    spans_multiple_pieces: bool,
}

impl MaskOwners {
    const fn new(first_piece: usize) -> Self {
        Self {
            first_piece,
            spans_multiple_pieces: false,
        }
    }

    const fn has_piece_other_than(self, piece: usize) -> bool {
        self.first_piece != piece || self.spans_multiple_pieces
    }

    const fn observe_piece(&mut self, piece: usize) {
        if self.first_piece != piece {
            self.spans_multiple_pieces = true;
        }
    }
}

impl MaskXOwners {
    const fn new(mask_index: usize, piece_index: usize) -> Self {
        Self {
            first_mask_index: mask_index,
            first_piece: piece_index,
            has_other_mask: false,
            spans_multiple_pieces: false,
        }
    }

    const fn has_mask_other_than(self, mask_index: usize) -> bool {
        self.first_mask_index != mask_index || self.has_other_mask
    }

    const fn has_piece_other_than(self, piece: usize) -> bool {
        self.first_piece != piece || self.spans_multiple_pieces
    }

    const fn observe(&mut self, mask_index: usize, piece: usize) {
        if self.first_mask_index != mask_index {
            self.has_other_mask = true;
        }
        if self.first_piece != piece {
            self.spans_multiple_pieces = true;
        }
    }
}

#[derive(Clone, Copy)]
struct IndexedMask {
    mask_index: usize,
    piece_index: usize,
    point: ProjectivePoint,
}

#[derive(Clone, Copy)]
struct AffineIndexedMask {
    mask_index: usize,
    piece_index: usize,
    point: FePoint,
}

#[derive(Clone, Copy)]
struct AffineMaskBuildCombination {
    point: Option<FePoint>,
    combination: MaskCombinationMeta,
    next_term_index: usize,
}

#[derive(Clone, Copy)]
struct IndexedPedersenCommitment {
    piece_index: usize,
    point: ProjectivePoint,
    value_bound: u64,
}

#[derive(Clone, Copy)]
struct MaskCombination {
    point: ProjectivePoint,
    encoded: [u8; POINT_LEN],
    mask_indices: [usize; CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND],
    piece_indices: [usize; CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND],
    support: usize,
}

#[allow(clippy::large_enum_variant)]
enum MaskCombinationBucket {
    One(MaskCombination),
    Many(Vec<MaskCombination>),
}

type MaskCombinationMap = std::collections::HashMap<[u8; POINT_LEN], MaskCombinationBucket>;

#[derive(Clone, Copy)]
struct MaskCombinationMeta {
    encoded: [u8; POINT_LEN],
    mask_indices: [usize; CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND],
    piece_indices: [usize; CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND],
    support: usize,
}

#[allow(clippy::large_enum_variant)]
enum MaskCombinationMetaBucket {
    One(MaskCombinationMeta),
    Many(Vec<MaskCombinationMeta>),
}

type MaskCombinationMetaMap = std::collections::HashMap<[u8; POINT_LEN], MaskCombinationMetaBucket>;

impl MaskCombination {
    const EMPTY_INDEX: usize = usize::MAX;

    const fn empty() -> Self {
        Self {
            point: ProjectivePoint::IDENTITY,
            encoded: [0u8; POINT_LEN],
            mask_indices: [Self::EMPTY_INDEX; CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND],
            piece_indices: [Self::EMPTY_INDEX; CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND],
            support: 0,
        }
    }

    fn extend(self, term: &IndexedMask) -> Option<Self> {
        let mut next = self.extend_indices(term.mask_index, term.piece_index)?;
        next.point += term.point;
        Some(next)
    }

    fn extend_indices(self, mask_index: usize, piece_index: usize) -> Option<Self> {
        if self.support == CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND
            || self.contains_mask(mask_index)
        {
            return None;
        }
        let mut next = self;
        next.mask_indices[self.support] = mask_index;
        next.piece_indices[self.support] = piece_index;
        next.support += 1;
        Some(next)
    }

    fn contains_mask(&self, mask_index: usize) -> bool {
        self.mask_indices[..self.support].contains(&mask_index)
    }

    fn disjoint_from(&self, other: &Self) -> bool {
        self.mask_indices[..self.support]
            .iter()
            .all(|mask_index| !other.contains_mask(*mask_index))
    }

    fn spans_multiple_pieces_with(&self, other: &Self) -> bool {
        let Some(&first_piece) = self.piece_indices[..self.support]
            .iter()
            .chain(other.piece_indices[..other.support].iter())
            .find(|&&piece_index| piece_index != Self::EMPTY_INDEX)
        else {
            return false;
        };
        self.piece_indices[..self.support]
            .iter()
            .chain(other.piece_indices[..other.support].iter())
            .any(|&piece_index| piece_index != Self::EMPTY_INDEX && piece_index != first_piece)
    }
}

impl MaskCombinationBucket {
    const fn one(combination: MaskCombination) -> Self {
        Self::One(combination)
    }

    fn push(&mut self, combination: MaskCombination) {
        match self {
            Self::One(existing) => {
                *self = Self::Many(vec![*existing, combination]);
            }
            Self::Many(combinations) => combinations.push(combination),
        }
    }

    fn any(&self, mut predicate: impl FnMut(&MaskCombination) -> bool) -> bool {
        match self {
            Self::One(combination) => predicate(combination),
            Self::Many(combinations) => combinations.iter().any(predicate),
        }
    }
}

impl MaskCombinationMeta {
    const EMPTY_INDEX: usize = usize::MAX;

    const fn empty() -> Self {
        Self {
            encoded: [0u8; POINT_LEN],
            mask_indices: [Self::EMPTY_INDEX; CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND],
            piece_indices: [Self::EMPTY_INDEX; CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND],
            support: 0,
        }
    }

    fn extend_indices(self, mask_index: usize, piece_index: usize) -> Option<Self> {
        if self.support == CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND
            || self.contains_mask(mask_index)
        {
            return None;
        }
        let mut next = self;
        next.mask_indices[self.support] = mask_index;
        next.piece_indices[self.support] = piece_index;
        next.support += 1;
        Some(next)
    }

    fn contains_mask(&self, mask_index: usize) -> bool {
        self.mask_indices[..self.support].contains(&mask_index)
    }

    fn disjoint_from(&self, other: &Self) -> bool {
        self.mask_indices[..self.support]
            .iter()
            .all(|mask_index| !other.contains_mask(*mask_index))
    }

    fn spans_multiple_pieces_with(&self, other: &Self) -> bool {
        let Some(&first_piece) = self.piece_indices[..self.support]
            .iter()
            .chain(other.piece_indices[..other.support].iter())
            .find(|&&piece_index| piece_index != Self::EMPTY_INDEX)
        else {
            return false;
        };
        self.piece_indices[..self.support]
            .iter()
            .chain(other.piece_indices[..other.support].iter())
            .any(|&piece_index| piece_index != Self::EMPTY_INDEX && piece_index != first_piece)
    }
}

impl MaskCombinationMetaBucket {
    const fn one(combination: MaskCombinationMeta) -> Self {
        Self::One(combination)
    }

    fn push(&mut self, combination: MaskCombinationMeta) {
        match self {
            Self::One(existing) => {
                *self = Self::Many(vec![*existing, combination]);
            }
            Self::Many(combinations) => combinations.push(combination),
        }
    }

    fn any(&self, mut predicate: impl FnMut(&MaskCombinationMeta) -> bool) -> bool {
        match self {
            Self::One(combination) => predicate(combination),
            Self::Many(combinations) => combinations.iter().any(predicate),
        }
    }
}

fn reject_oversized_cross_piece_relation_scan(piece_count: usize) -> Option<&'static str> {
    (piece_count > CROSS_PIECE_MASK_RELATION_PIECE_BOUND)
        .then_some("case has too many pieces for cross-piece mask relation scan")
}

fn reject_oversized_cross_piece_pedersen_relation_scan(piece_count: usize) -> Option<&'static str> {
    (piece_count > CROSS_PIECE_PEDERSEN_RELATION_PIECE_BOUND)
        .then_some("case has too many pieces for cross-piece Pedersen relation scan")
}

fn encoded_projective_batch(points: &[ProjectivePoint]) -> Vec<[u8; POINT_LEN]> {
    let mut out = vec![encode_point(&ProjectivePoint::IDENTITY); points.len()];
    let mut non_identity = Vec::with_capacity(points.len());
    let mut indices = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        if point != &ProjectivePoint::IDENTITY {
            non_identity.push(*point);
            indices.push(index);
        }
    }
    if non_identity.is_empty() {
        // `batch_normalize` inverts the product of z-coordinates and would
        // unwrap a `None` on an empty (or all-identity) batch; the identity
        // placeholders are already in place.
        return out;
    }
    for (index, affine) in indices
        .into_iter()
        .zip(ProjectivePoint::batch_normalize(non_identity.as_slice()))
    {
        out[index] = encode_affine_point(&affine);
    }
    out
}

fn owner_x_key(encoded: &[u8; POINT_LEN]) -> [u8; 32] {
    let mut key = [0u8; 32];
    key.copy_from_slice(&encoded[1..]);
    key
}

pub fn cross_piece_elgamal_mask_relation<'a, I>(pieces: I) -> Option<&'static str>
where
    I: IntoIterator<Item = &'a [LimbCiphertext]>,
{
    const DETAIL: &str = "case pieces have a cross-piece ElGamal mask relation";

    let mut owners = std::collections::HashMap::<[u8; POINT_LEN], MaskOwners>::new();
    let mut masks = Vec::new();
    for (piece_index, piece) in pieces.into_iter().enumerate() {
        let piece_count = piece_index + 1;
        if let Some(detail) = reject_oversized_cross_piece_relation_scan(piece_count) {
            return Some(detail);
        }
        for ct in piece {
            let encoded = encode_point(&ct.e);
            match owners.get_mut(&encoded) {
                Some(owner) => {
                    if owner.has_piece_other_than(piece_index) {
                        return Some(DETAIL);
                    }
                    owner.observe_piece(piece_index);
                    if owner.has_piece_other_than(piece_index) {
                        return Some(DETAIL);
                    }
                }
                None => {
                    owners.insert(encoded, MaskOwners::new(piece_index));
                }
            }
            let indexed_mask = IndexedMask {
                mask_index: masks.len(),
                piece_index,
                point: ct.e,
            };
            masks.push(indexed_mask);
        }
    }
    if has_public_scalar_mask_relation(&masks, RECOVERY_KEY_ENUMERABILITY_BOUND, true) {
        return Some(DETAIL);
    }
    if has_cross_piece_pair_public_g_offset_relation(&masks) {
        return Some(DETAIL);
    }
    if has_cross_piece_mask_relation_in_support_range(
        &masks,
        1,
        CROSS_PIECE_SMALL_COEFF_RELATION_SUPPORT_BOUND + 1,
        CROSS_PIECE_UNIT_RELATION_SUPPORT_BOUND,
    ) || has_cross_piece_mask_relation_with_support(
        &masks,
        MASK_RELATION_COEFFICIENT_BOUND,
        CROSS_PIECE_SMALL_COEFF_RELATION_SUPPORT_BOUND,
    ) {
        return Some(DETAIL);
    }
    None
}

fn signed_mask_multiples(masks: &[IndexedMask], coefficient_bound: u16) -> Vec<IndexedMask> {
    let mut multiples = Vec::with_capacity(masks.len() * usize::from(coefficient_bound) * 2);
    for mask in masks {
        let mut multiple = mask.point;
        for _ in 1..=coefficient_bound {
            multiples.push(IndexedMask {
                mask_index: mask.mask_index,
                piece_index: mask.piece_index,
                point: multiple,
            });
            multiples.push(IndexedMask {
                mask_index: mask.mask_index,
                piece_index: mask.piece_index,
                point: -multiple,
            });
            multiple += mask.point;
        }
    }
    multiples
}

/// Recurse over size-`exact_support` ordered subsets of `terms` from index
/// `start`, extending `current`; `visit` is called on each complete subset and
/// may stop the walk early by returning `false`. Single source for the
/// sequential base case and the per-first-term parallel traversal below.
fn walk_mask_combinations_exact(
    terms: &[IndexedMask],
    exact_support: usize,
    start: usize,
    current: MaskCombination,
    visit: &mut impl FnMut(MaskCombination) -> bool,
) -> bool {
    if current.support == exact_support {
        return visit(current);
    }
    for term_idx in start..terms.len() {
        if let Some(next) = current.extend(&terms[term_idx]) {
            if !walk_mask_combinations_exact(terms, exact_support, term_idx + 1, next, visit) {
                return false;
            }
        }
    }
    true
}

/// Every size-`exact_support` ordered subset of `terms`, each with its summed
/// point and canonical encoding. The size-`exact_support` ordered subsets
/// partition exactly by their minimum term-index, so each first term is an
/// independent subtree built on its own worker; the per-worker results
/// concatenate in first-term order, byte-identical (same subsets, same order,
/// same per-point encodings) to a sequential DFS.
fn collect_mask_combinations_exact(
    terms: &[IndexedMask],
    exact_support: usize,
) -> Vec<MaskCombination> {
    if exact_support == 0 || exact_support > CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND {
        return Vec::new();
    }
    if worker_count(terms.len()) <= 1 {
        let mut combinations = Vec::with_capacity(mask_combination_capacity(terms, exact_support));
        walk_mask_combinations_exact(
            terms,
            exact_support,
            0,
            MaskCombination::empty(),
            &mut |combination| {
                combinations.push(combination);
                true
            },
        );
        fill_combination_encodings(&mut combinations);
        return combinations;
    }
    let per_first_term = parallel_map_indexed(terms.len(), |first_idx| {
        let Some(first) = MaskCombination::empty().extend(&terms[first_idx]) else {
            return Vec::new();
        };
        let mut combinations = Vec::new();
        walk_mask_combinations_exact(
            terms,
            exact_support,
            first_idx + 1,
            first,
            &mut |combination| {
                combinations.push(combination);
                true
            },
        );
        fill_combination_encodings(&mut combinations);
        combinations
    });
    per_first_term.into_iter().flatten().collect()
}

fn mask_combination_capacity(terms: &[IndexedMask], exact_support: usize) -> usize {
    let mask_count = terms
        .iter()
        .map(|term| term.mask_index)
        .max()
        .map_or(0usize, |max_index| max_index.saturating_add(1));
    combination_capacity(mask_count, terms.len(), exact_support)
}

fn affine_mask_combination_capacity(terms: &[AffineIndexedMask], exact_support: usize) -> usize {
    let mask_count = terms
        .iter()
        .map(|term| term.mask_index)
        .max()
        .map_or(0usize, |max_index| max_index.saturating_add(1));
    combination_capacity(mask_count, terms.len(), exact_support)
}

fn combination_capacity(mask_count: usize, term_count: usize, exact_support: usize) -> usize {
    if mask_count == 0 || exact_support > mask_count {
        return 0;
    }
    let alternatives_per_mask = term_count.div_ceil(mask_count);
    let exact_support = u32::try_from(exact_support).unwrap_or(u32::MAX);
    binomial(mask_count, exact_support)
        .saturating_mul(alternatives_per_mask.saturating_pow(exact_support))
}

fn binomial(n: usize, k: u32) -> usize {
    let k = usize::try_from(k).unwrap_or(usize::MAX);
    let k = k.min(n.saturating_sub(k));
    let mut value = 1usize;
    for i in 0..k {
        value = value.saturating_mul(n - i) / (i + 1);
    }
    value
}

/// Fill each combination's canonical `encoded` from its `point`, in one
/// batched normalization over the slice.
fn fill_combination_encodings(combinations: &mut [MaskCombination]) {
    let points = combinations
        .iter()
        .map(|combination| combination.point)
        .collect::<Vec<_>>();
    for (combination, encoded) in combinations
        .iter_mut()
        .zip(encoded_projective_batch(&points))
    {
        combination.encoded = encoded;
    }
}

const fn negate_encoded_point(mut encoded: [u8; POINT_LEN]) -> [u8; POINT_LEN] {
    if encoded[0] != 0 {
        encoded[0] ^= 1;
    }
    encoded
}

fn relation_exists_between_halves(
    left_sums: &MaskCombinationMap,
    right_combinations: &[MaskCombination],
) -> bool {
    // Existential probe: does any right combination sum to the negation of a
    // left combination (a vanishing cross-piece signed subset)? Order-free, so
    // it fans out; every chunk negates its own combinations' encodings and
    // probes the read-only left map.
    parallel_any(right_combinations, |chunk| {
        chunk.iter().any(|right| {
            let target = negate_encoded_point(right.encoded);
            let Some(left_candidates) = left_sums.get(&target) else {
                return false;
            };
            left_candidates.any(|left| {
                left.support + right.support >= 2
                    && left.disjoint_from(right)
                    && left.spans_multiple_pieces_with(right)
            })
        })
    })
}

const fn identity_encoding() -> [u8; POINT_LEN] {
    [0u8; POINT_LEN]
}

fn signed_affine_mask_terms(masks: &[IndexedMask]) -> Option<Vec<AffineIndexedMask>> {
    let points = masks.iter().map(|mask| mask.point).collect::<Vec<_>>();
    let affine = ProjectivePoint::batch_normalize(points.as_slice());
    let mut terms = Vec::with_capacity(masks.len() * 2);
    for (mask, point) in masks.iter().zip(affine.iter()) {
        let point = FePoint::from_affine(point)?;
        terms.push(AffineIndexedMask {
            mask_index: mask.mask_index,
            piece_index: mask.piece_index,
            point,
        });
        terms.push(AffineIndexedMask {
            mask_index: mask.mask_index,
            piece_index: mask.piece_index,
            point: point.negated(),
        });
    }
    Some(terms)
}

fn collect_unit_affine_mask_pairs(terms: &[AffineIndexedMask]) -> Vec<AffineMaskBuildCombination> {
    let mut pairs = Vec::with_capacity(affine_mask_combination_capacity(terms, 2));
    let mut metadata = Vec::with_capacity(pairs.capacity());
    for (lhs_idx, lhs) in terms.iter().enumerate() {
        for (rhs_idx, rhs) in terms.iter().enumerate().skip(lhs_idx + 1) {
            let Some(combination) = MaskCombinationMeta::empty()
                .extend_indices(lhs.mask_index, lhs.piece_index)
                .and_then(|combination| {
                    combination.extend_indices(rhs.mask_index, rhs.piece_index)
                })
            else {
                continue;
            };
            pairs.push((lhs.point, rhs.point));
            metadata.push((rhs_idx + 1, combination));
        }
    }

    let mut out = Vec::with_capacity(pairs.len());
    let mut scratch = BatchAddScratch::with_capacity(pairs.len());
    let mut index = 0usize;
    let _ = batch_add_points_visit(&pairs, &mut scratch, |point| {
        let (next_term_index, mut combination) = metadata[index];
        index += 1;
        combination.encoded = point.map_or_else(identity_encoding, |point| point.encoded_point());
        out.push(AffineMaskBuildCombination {
            point,
            combination,
            next_term_index,
        });
        false
    });
    out
}

fn collect_unit_affine_mask_triples(
    terms: &[AffineIndexedMask],
    pairs: &[AffineMaskBuildCombination],
) -> Vec<MaskCombinationMeta> {
    let mut add_pairs = Vec::with_capacity(affine_mask_combination_capacity(terms, 3));
    let mut metadata = Vec::with_capacity(add_pairs.capacity());
    let mut out = Vec::with_capacity(add_pairs.capacity());
    for pair in pairs {
        for term in &terms[pair.next_term_index..] {
            let Some(combination) = pair
                .combination
                .extend_indices(term.mask_index, term.piece_index)
            else {
                continue;
            };
            if let Some(point) = pair.point {
                add_pairs.push((point, term.point));
                metadata.push(combination);
            } else {
                let mut combination = combination;
                combination.encoded = term.point.encoded_point();
                out.push(combination);
            }
        }
    }

    let mut scratch = BatchAddScratch::with_capacity(add_pairs.len());
    let mut index = 0usize;
    let _ = batch_add_points_visit(&add_pairs, &mut scratch, |point| {
        let mut combination = metadata[index];
        index += 1;
        combination.encoded = point.map_or_else(identity_encoding, |point| point.encoded_point());
        out.push(combination);
        false
    });
    out
}

fn combination_map(combinations: &[MaskCombination]) -> MaskCombinationMap {
    let mut sums = MaskCombinationMap::with_capacity(combinations.len());
    for combination in combinations {
        match sums.entry(combination.encoded) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().push(*combination);
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(MaskCombinationBucket::one(*combination));
            }
        }
    }
    sums
}

fn compact_combination_map(combinations: &[MaskCombinationMeta]) -> MaskCombinationMetaMap {
    let mut sums = MaskCombinationMetaMap::with_capacity(combinations.len());
    for combination in combinations {
        match sums.entry(combination.encoded) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().push(*combination);
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(MaskCombinationMetaBucket::one(*combination));
            }
        }
    }
    sums
}

fn compact_relation_exists_between_halves(
    left_sums: &MaskCombinationMetaMap,
    right_combinations: &[MaskCombinationMeta],
) -> bool {
    parallel_any(right_combinations, |chunk| {
        chunk.iter().any(|right| {
            let target = negate_encoded_point(right.encoded);
            let Some(left_candidates) = left_sums.get(&target) else {
                return false;
            };
            left_candidates.any(|left| {
                left.support + right.support >= 2
                    && left.disjoint_from(right)
                    && left.spans_multiple_pieces_with(right)
            })
        })
    })
}

fn has_unit_cross_piece_mask_relation_support_five_or_six(masks: &[IndexedMask]) -> bool {
    let Some(terms) = signed_affine_mask_terms(masks) else {
        return true;
    };
    let pairs = collect_unit_affine_mask_pairs(&terms);
    let pair_combinations = pairs
        .iter()
        .map(|pair| pair.combination)
        .collect::<Vec<_>>();
    let triples = collect_unit_affine_mask_triples(&terms, &pairs);
    let left_sums = compact_combination_map(&triples);
    compact_relation_exists_between_halves(&left_sums, &pair_combinations)
        || compact_relation_exists_between_halves(&left_sums, &triples)
}

fn has_cross_piece_mask_relation_with_support(
    masks: &[IndexedMask],
    coefficient_bound: u16,
    support_bound: usize,
) -> bool {
    has_cross_piece_mask_relation_in_support_range(masks, coefficient_bound, 2, support_bound)
}

fn has_cross_piece_mask_relation_in_support_range(
    masks: &[IndexedMask],
    coefficient_bound: u16,
    min_support: usize,
    support_bound: usize,
) -> bool {
    if support_bound > CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND * 2 {
        return true;
    }
    let min_support = min_support.max(2);
    if min_support > support_bound {
        return false;
    }
    if coefficient_bound == 1
        && min_support == CROSS_PIECE_SMALL_COEFF_RELATION_SUPPORT_BOUND + 1
        && support_bound == CROSS_PIECE_UNIT_RELATION_SUPPORT_BOUND
    {
        return has_unit_cross_piece_mask_relation_support_five_or_six(masks);
    }

    let signed_multiples = signed_mask_multiples(masks, coefficient_bound);
    let max_half_support = support_bound
        .div_ceil(2)
        .min(CROSS_PIECE_MASK_RELATION_HALF_SUPPORT_BOUND);
    let mut combinations_by_support = Vec::with_capacity(max_half_support + 1);
    combinations_by_support.push(Vec::new());
    for exact_support in 1..=max_half_support {
        combinations_by_support.push(collect_mask_combinations_exact(
            &signed_multiples,
            exact_support,
        ));
    }

    let mut left_sums_by_support = (0..=max_half_support)
        .map(|_| None)
        .collect::<Vec<Option<MaskCombinationMap>>>();
    for support in min_support..=support_bound {
        let left_support = support.div_ceil(2);
        let right_support = support - left_support;
        if left_sums_by_support[left_support].is_none() {
            left_sums_by_support[left_support] =
                Some(combination_map(&combinations_by_support[left_support]));
        }
        let Some(left_sums) = left_sums_by_support[left_support].as_ref() else {
            return true;
        };
        if relation_exists_between_halves(left_sums, &combinations_by_support[right_support]) {
            return true;
        }
    }
    false
}

const fn pedersen_statement_value_bound(slot: usize) -> u64 {
    if slot < LIMB_COUNT * 2 {
        LIMB_MODULUS
    } else {
        2
    }
}

fn collect_cross_piece_public_g_offset_candidates(
    slot: &[(usize, ProjectivePoint)],
    value_bound: u64,
    candidates: &mut std::collections::BTreeMap<u64, Vec<ProjectivePoint>>,
) {
    fn walk(
        slot: &[(usize, ProjectivePoint)],
        point: ProjectivePoint,
        support: usize,
        value_bound: u64,
        candidates: &mut std::collections::BTreeMap<u64, Vec<ProjectivePoint>>,
    ) {
        let Some(((_, term), rest)) = slot.split_first() else {
            if support >= 2 {
                let bound = value_bound.saturating_mul(u64::try_from(support).unwrap_or(u64::MAX));
                candidates.entry(bound).or_default().push(point);
            }
            return;
        };

        walk(rest, point, support, value_bound, candidates);
        walk(rest, point + *term, support + 1, value_bound, candidates);
        if support > 0 {
            walk(rest, point - *term, support + 1, value_bound, candidates);
        }
    }

    walk(slot, ProjectivePoint::IDENTITY, 0, value_bound, candidates);
}

fn has_public_g_offset_candidate(
    candidates: std::collections::BTreeMap<u64, Vec<ProjectivePoint>>,
) -> bool {
    candidates
        .into_iter()
        .any(|(bound, points)| !points.is_empty() && any_signed_g_multiple_below(&points, bound))
}

fn mask_x_owners(masks: &[IndexedMask]) -> std::collections::HashMap<[u8; 32], MaskXOwners> {
    let mut owners = std::collections::HashMap::with_capacity(masks.len());
    for mask in masks {
        let encoded = encode_point(&mask.point);
        owners
            .entry(owner_x_key(&encoded))
            .and_modify(|owner: &mut MaskXOwners| {
                owner.observe(mask.mask_index, mask.piece_index);
            })
            .or_insert_with(|| MaskXOwners::new(mask.mask_index, mask.piece_index));
    }
    owners
}

fn has_public_scalar_mask_relation(
    masks: &[IndexedMask],
    scalar_bound: u16,
    require_cross_piece: bool,
) -> bool {
    let owners = mask_x_owners(masks);
    // The per-mask scalar ladders are independent; fan them out, each worker
    // reusing its own candidate buffer and probing the read-only owners map.
    parallel_any(masks, |chunk| {
        let mut candidates = Vec::with_capacity(usize::from(scalar_bound));
        chunk.iter().any(|mask| {
            candidates.clear();
            let mut multiple = ProjectivePoint::IDENTITY;
            for _ in 1..=scalar_bound {
                multiple += mask.point;
                candidates.push(multiple);
            }
            ProjectivePoint::batch_normalize(candidates.as_slice())
                .iter()
                .any(|affine| {
                    let x_key: [u8; 32] = affine.x().into();
                    owners.get(&x_key).is_some_and(|owner| {
                        if require_cross_piece {
                            owner.has_piece_other_than(mask.piece_index)
                        } else {
                            owner.has_mask_other_than(mask.mask_index)
                        }
                    })
                })
        })
    })
}

fn has_pairwise_elgamal_mask_scalar_relation(ciphertexts: &[LimbCiphertext]) -> bool {
    let masks = ciphertexts
        .iter()
        .enumerate()
        .map(|(mask_index, ct)| IndexedMask {
            mask_index,
            piece_index: 0,
            point: ct.e,
        })
        .collect::<Vec<_>>();
    has_public_scalar_mask_relation_small_target_bsgs(&masks, RECOVERY_KEY_ENUMERABILITY_BOUND)
}

fn ceil_sqrt(value: u16) -> u64 {
    let value = u64::from(value);
    let mut root = 1u64;
    while root.saturating_mul(root) < value {
        root += 1;
    }
    root
}

fn public_scalar_bsgs_hit(
    giant_index: u64,
    baby_index: u64,
    baby_steps: u64,
    scalar_bound: u16,
) -> bool {
    let center = giant_index.saturating_mul(baby_steps);
    let plus = center.saturating_add(baby_index);
    let minus = center.abs_diff(baby_index);
    let bound = u64::from(scalar_bound);
    (plus != 0 && plus <= bound) || (minus != 0 && minus <= bound)
}

fn has_public_scalar_mask_relation_small_target_bsgs(
    masks: &[IndexedMask],
    scalar_bound: u16,
) -> bool {
    if scalar_bound == 0 || masks.len() < 2 {
        return false;
    }

    let baby_steps = ceil_sqrt(scalar_bound);
    let giant_steps = u64::from(scalar_bound).div_ceil(baby_steps);
    let baby_cap = usize::try_from(baby_steps).unwrap_or(0);
    let giant_cap = masks
        .len()
        .saturating_sub(1)
        .saturating_mul(usize::try_from((giant_steps + 1) * 2).unwrap_or(0));
    let mut baby_points = Vec::with_capacity(baby_cap);
    let mut baby_entries = Vec::<([u8; 32], u64)>::with_capacity(baby_cap);
    let mut giant_points = Vec::with_capacity(giant_cap);
    let mut giant_indices = Vec::with_capacity(giant_cap);

    for (base_index, base) in masks.iter().enumerate() {
        baby_points.clear();
        baby_entries.clear();
        let mut baby = base.point;
        for _ in 1..=baby_steps {
            baby_points.push(baby);
            baby += base.point;
        }
        for (offset, affine) in ProjectivePoint::batch_normalize(baby_points.as_slice())
            .iter()
            .enumerate()
        {
            baby_entries.push((
                affine.x().into(),
                u64::try_from(offset + 1).unwrap_or(u64::MAX),
            ));
        }
        baby_entries.sort_unstable_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));

        let stride = baby_points
            .last()
            .copied()
            .unwrap_or(ProjectivePoint::IDENTITY);
        giant_points.clear();
        giant_indices.clear();
        for (target_index, target) in masks.iter().enumerate() {
            if target_index == base_index {
                continue;
            }
            let mut minus_walk = target.point;
            let mut plus_walk = target.point;
            for giant_index in 0..=giant_steps {
                let center = giant_index.saturating_mul(baby_steps);
                if minus_walk == ProjectivePoint::IDENTITY
                    && center != 0
                    && center <= u64::from(scalar_bound)
                {
                    return true;
                }
                if plus_walk == ProjectivePoint::IDENTITY
                    && center != 0
                    && center <= u64::from(scalar_bound)
                {
                    return true;
                }
                if minus_walk != ProjectivePoint::IDENTITY {
                    giant_points.push(minus_walk);
                    giant_indices.push(giant_index);
                }
                if plus_walk != ProjectivePoint::IDENTITY && giant_index != 0 {
                    giant_points.push(plus_walk);
                    giant_indices.push(giant_index);
                }
                minus_walk -= stride;
                plus_walk += stride;
            }
        }

        for (affine, giant_index) in ProjectivePoint::batch_normalize(giant_points.as_slice())
            .iter()
            .zip(giant_indices.iter().copied())
        {
            let x_key: [u8; 32] = affine.x().into();
            if let Ok(hit) = baby_entries.binary_search_by(|(baby_x, _)| baby_x.cmp(&x_key)) {
                let baby_index = baby_entries[hit].1;
                if public_scalar_bsgs_hit(giant_index, baby_index, baby_steps, scalar_bound) {
                    return true;
                }
            }
        }
    }
    false
}

fn bounded_mask_sums(
    points: &[ProjectivePoint],
    coefficient_bound: u16,
) -> Vec<(ProjectivePoint, usize)> {
    let mut sums = vec![(ProjectivePoint::IDENTITY, 0)];
    for point in points {
        let existing = sums.clone();
        let mut multiple = ProjectivePoint::IDENTITY;
        for _ in 1..=coefficient_bound {
            multiple += *point;
            for term in [multiple, -multiple] {
                for &(sum, support) in &existing {
                    sums.push((sum + term, support + 1));
                }
            }
        }
    }
    sums
}

fn has_elgamal_mask_relation_with_coefficient_bound(
    ciphertexts: &[LimbCiphertext],
    coefficient_bound: u16,
) -> bool {
    let masks = ciphertexts.iter().map(|ct| ct.e).collect::<Vec<_>>();
    has_point_relation_with_coefficient_bound(&masks, coefficient_bound)
}

fn has_point_relation_with_coefficient_bound(
    points: &[ProjectivePoint],
    coefficient_bound: u16,
) -> bool {
    let (left, right) = points.split_at(points.len() / 2);
    let mut left_sums = std::collections::HashMap::<[u8; POINT_LEN], usize>::new();
    let left_combinations = bounded_mask_sums(left, coefficient_bound);
    let left_points = left_combinations
        .iter()
        .map(|(sum, _support)| *sum)
        .collect::<Vec<_>>();
    for ((_, support), encoded) in left_combinations
        .iter()
        .zip(encoded_projective_batch(&left_points))
    {
        left_sums
            .entry(encoded)
            .and_modify(|stored| *stored = (*stored).max(*support))
            .or_insert(*support);
    }

    let right_combinations = bounded_mask_sums(right, coefficient_bound);
    let right_points = right_combinations
        .iter()
        .map(|(sum, _support)| -*sum)
        .collect::<Vec<_>>();
    for ((_, right_support), target) in right_combinations
        .iter()
        .zip(encoded_projective_batch(&right_points))
    {
        let Some(left_support) = left_sums.get(&target) else {
            continue;
        };
        if left_support + *right_support >= 2 {
            return true;
        }
    }
    false
}

fn affine_relation_multiples(
    points: &[(usize, ProjectivePoint)],
    coefficient_bound: u16,
) -> Option<Vec<Vec<FePoint>>> {
    let mut projective = Vec::with_capacity(points.len() * usize::from(coefficient_bound));
    for &(_, point) in points {
        let mut multiple = ProjectivePoint::IDENTITY;
        for _ in 1..=coefficient_bound {
            multiple += point;
            projective.push(multiple);
        }
    }
    let affine = ProjectivePoint::batch_normalize(projective.as_slice());
    let mut affine = affine.iter();
    let mut multiples = Vec::with_capacity(points.len());
    for _ in points {
        let mut row = Vec::with_capacity(usize::from(coefficient_bound));
        for _ in 1..=coefficient_bound {
            row.push(FePoint::from_affine(affine.next()?)?);
        }
        multiples.push(row);
    }
    Some(multiples)
}

fn has_pair_public_g_offset_relation(
    points: &[(usize, ProjectivePoint)],
    coefficient_bound: u16,
    require_cross_piece: bool,
) -> bool {
    let Some(multiples) = affine_relation_multiples(points, coefficient_bound) else {
        return true;
    };
    let per_pair = usize::from(coefficient_bound).pow(2) * 2;
    let mut pairs = Vec::with_capacity(points.len().saturating_mul(points.len()) * per_pair / 2);
    let mut bounds = Vec::with_capacity(pairs.capacity());
    for (lhs_idx, (lhs_piece, _)) in points.iter().enumerate() {
        for (rhs_idx, (rhs_piece, _)) in points.iter().enumerate().skip(lhs_idx + 1) {
            if require_cross_piece && lhs_piece == rhs_piece {
                continue;
            }
            for (lhs_offset, &lhs) in multiples[lhs_idx].iter().enumerate() {
                let lhs_coeff = u16::try_from(lhs_offset + 1).unwrap_or(u16::MAX);
                for (rhs_offset, &rhs) in multiples[rhs_idx].iter().enumerate() {
                    let rhs_coeff = u16::try_from(rhs_offset + 1).unwrap_or(u16::MAX);
                    pairs.push((lhs, rhs));
                    bounds.push(mask_public_g_offset_bound(lhs_coeff, rhs_coeff, false));
                    pairs.push((lhs, rhs.negated()));
                    bounds.push(mask_public_g_offset_bound(lhs_coeff, rhs_coeff, true));
                }
            }
        }
    }

    let mut candidates_by_bound = std::collections::BTreeMap::<u64, Vec<FePoint>>::new();
    let mut candidate_index = 0usize;
    let mut scratch = BatchAddScratch::with_capacity(pairs.len());
    if batch_add_points_visit(&pairs, &mut scratch, |candidate| {
        let bound = bounds[candidate_index];
        candidate_index += 1;
        let Some(candidate) = candidate else {
            return true;
        };
        candidates_by_bound
            .entry(bound)
            .or_default()
            .push(candidate);
        false
    }) {
        return true;
    }
    candidates_by_bound
        .into_iter()
        .any(|(bound, candidates)| any_signed_fe_g_multiple_below(&candidates, bound))
}

fn mask_public_g_offset_bound(lhs_coeff: u16, rhs_coeff: u16, difference: bool) -> u64 {
    let coefficient_bound = if difference {
        lhs_coeff.max(rhs_coeff)
    } else {
        lhs_coeff.saturating_add(rhs_coeff)
    };
    u64::from(coefficient_bound).saturating_mul(LIMB_MODULUS)
}

fn has_elgamal_mask_public_g_offset_relation(ciphertexts: &[LimbCiphertext]) -> bool {
    if ciphertexts
        .iter()
        .any(|ct| is_signed_g_multiple_below(&ct.e, LIMB_MODULUS))
    {
        return true;
    }

    let points = ciphertexts
        .iter()
        .map(|ct| (0usize, ct.e))
        .collect::<Vec<_>>();
    has_pair_public_g_offset_relation(&points, MASK_RELATION_COEFFICIENT_BOUND, false)
}

fn has_cross_piece_pair_public_g_offset_relation(masks: &[IndexedMask]) -> bool {
    let points = masks
        .iter()
        .map(|mask| (mask.piece_index, mask.point))
        .collect::<Vec<_>>();
    has_pair_public_g_offset_relation(&points, MASK_RELATION_COEFFICIENT_BOUND, true)
}

fn has_small_coefficient_elgamal_mask_relation(ciphertexts: &[LimbCiphertext]) -> bool {
    has_elgamal_mask_relation_with_coefficient_bound(ciphertexts, MASK_RELATION_COEFFICIENT_BOUND)
}

fn pedersen_statement_commitments(proof: &Proof) -> Option<[ProjectivePoint; range_circuit::K]> {
    (proof.value_commitments.len() == LIMB_COUNT
        && proof.complement_commitments.len() == LIMB_COUNT
        && proof.carry.carry_commitments.len() == LIMB_COUNT - 1)
        .then(|| {
            range_statement(
                &proof.value_commitments,
                &proof.complement_commitments,
                &proof.carry.carry_commitments,
            )
        })
}

fn degenerate_pedersen_commitment(proof: &Proof) -> Option<&'static str> {
    let commitments = pedersen_statement_commitments(proof)?;
    commitments
        .iter()
        .any(|commitment| {
            is_signed_g_multiple_below(commitment, PEDERSEN_COMMITMENT_PUBLIC_G_BOUND)
        })
        .then_some("Pedersen commitment has a public opening")
}

pub fn cross_piece_pedersen_commitment_relation<'a, I>(proofs: I) -> Option<&'static str>
where
    I: IntoIterator<Item = &'a Proof>,
{
    const DETAIL: &str = "case pieces have a cross-piece Pedersen commitment relation";

    let mut slots = vec![Vec::<(usize, ProjectivePoint)>::new(); range_circuit::K];
    let mut all_commitments = Vec::new();
    for (piece_index, proof) in proofs.into_iter().enumerate() {
        let piece_count = piece_index + 1;
        if let Some(detail) = reject_oversized_cross_piece_pedersen_relation_scan(piece_count) {
            return Some(detail);
        }
        let Some(commitments) = pedersen_statement_commitments(proof) else {
            return Some("case proof Pedersen commitment shape mismatch");
        };
        for (slot_index, (slot, commitment)) in slots.iter_mut().zip(commitments).enumerate() {
            slot.push((piece_index, commitment));
            all_commitments.push(IndexedPedersenCommitment {
                piece_index,
                point: commitment,
                value_bound: pedersen_statement_value_bound(slot_index),
            });
        }
    }

    if has_cross_piece_pair_pedersen_public_g_offset_relation(&all_commitments) {
        return Some(DETAIL);
    }

    let mut candidates = std::collections::BTreeMap::<u64, Vec<ProjectivePoint>>::new();
    for (slot_index, slot) in slots.into_iter().enumerate() {
        collect_cross_piece_public_g_offset_candidates(
            &slot,
            pedersen_statement_value_bound(slot_index),
            &mut candidates,
        );
    }
    if has_public_g_offset_candidate(candidates) {
        return Some(DETAIL);
    }
    None
}

fn has_cross_piece_pair_pedersen_public_g_offset_relation(
    commitments: &[IndexedPedersenCommitment],
) -> bool {
    let points = commitments
        .iter()
        .map(|commitment| (commitment.piece_index, commitment.point))
        .collect::<Vec<_>>();
    let Some(multiples) = affine_relation_multiples(&points, 1) else {
        return true;
    };

    let mut pairs = Vec::new();
    let mut bounds = Vec::new();
    for (lhs_idx, lhs) in commitments.iter().enumerate() {
        for (rhs_idx, rhs) in commitments.iter().enumerate().skip(lhs_idx + 1) {
            if lhs.piece_index == rhs.piece_index {
                continue;
            }
            let lhs_point = multiples[lhs_idx][0];
            let rhs_point = multiples[rhs_idx][0];
            pairs.push((lhs_point, rhs_point));
            bounds.push(lhs.value_bound.saturating_add(rhs.value_bound));
            pairs.push((lhs_point, rhs_point.negated()));
            bounds.push(lhs.value_bound.max(rhs.value_bound));
        }
    }

    let mut candidates_by_bound = std::collections::BTreeMap::<u64, Vec<FePoint>>::new();
    let mut candidate_index = 0usize;
    let mut scratch = BatchAddScratch::with_capacity(pairs.len());
    if batch_add_points_visit(&pairs, &mut scratch, |candidate| {
        let bound = bounds[candidate_index];
        candidate_index += 1;
        let Some(candidate) = candidate else {
            return true;
        };
        candidates_by_bound
            .entry(bound)
            .or_default()
            .push(candidate);
        false
    }) {
        return true;
    }
    candidates_by_bound
        .into_iter()
        .any(|(bound, candidates)| any_signed_fe_g_multiple_below(&candidates, bound))
}

pub fn reject_degenerate_recovery_key(
    pk: &ProjectivePoint,
    error: fn(&'static str) -> Error,
) -> Result<(), Error> {
    if pk == &ProjectivePoint::IDENTITY {
        return Err(error("recovery public key is the identity"));
    }
    let encoded = encode_point(pk);
    if cached_public_key_screening(&encoded) {
        return Ok(());
    }
    if is_publicly_enumerable_signed_g_multiple(pk) {
        return Err(error("recovery public key is publicly enumerable"));
    }
    if small_signed_h_relation_table().contains(pk) {
        return Err(error(
            "recovery public key is a public NUMS-generator multiple",
        ));
    }
    cache_public_key_screening(encoded);
    Ok(())
}

pub fn reject_publicly_enumerable_key_component(
    point: &ProjectivePoint,
    g_multiple_message: &'static str,
    nums_multiple_message: &'static str,
    error: fn(&'static str) -> Error,
) -> Result<(), Error> {
    let encoded = encode_point(point);
    if cached_public_key_screening(&encoded) {
        return Ok(());
    }
    if is_publicly_enumerable_signed_g_multiple(point) {
        return Err(error(g_multiple_message));
    }
    if small_signed_h_relation_table().contains(point) {
        return Err(error(nums_multiple_message));
    }
    cache_public_key_screening(encoded);
    Ok(())
}

type PublicKeyScreeningCache = std::collections::HashSet<[u8; POINT_LEN]>;

fn public_key_screening_cache() -> &'static Mutex<PublicKeyScreeningCache> {
    static CACHE: OnceLock<Mutex<PublicKeyScreeningCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(std::collections::HashSet::with_capacity(
            PUBLIC_KEY_SCREENING_CACHE_CAP,
        ))
    })
}

fn lock_public_key_screening_cache() -> MutexGuard<'static, PublicKeyScreeningCache> {
    match public_key_screening_cache().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn cached_public_key_screening(encoded: &[u8; POINT_LEN]) -> bool {
    lock_public_key_screening_cache().contains(encoded)
}

fn cache_public_key_screening(encoded: [u8; POINT_LEN]) {
    let mut cache = lock_public_key_screening_cache();
    if cache.len() < PUBLIC_KEY_SCREENING_CACHE_CAP {
        cache.insert(encoded);
    }
}

fn is_publicly_enumerable_signed_g_multiple(point: &ProjectivePoint) -> bool {
    // The signed public recovery-key scalar screen (|v| < 2^32) strictly
    // subsumes the limb-window BSGS walk (|v| < 2^ℓ) this check used to run
    // alongside it. The boundary-window table is also inside that range
    // (2^ℓ + window < 2^32) but stays as an O(1) independent gate so the
    // boundary intent survives any future re-tuning of the scalar bound.
    is_public_recovery_key_scalar_multiple(point) || public_g_boundary_table().contains(point)
}

struct SmallSignedPointTable {
    points: std::collections::HashSet<[u8; crate::codec::POINT_LEN]>,
}

struct PublicGBoundaryTable {
    points: std::collections::HashSet<[u8; crate::codec::POINT_LEN]>,
}

fn public_g_boundary_table() -> &'static PublicGBoundaryTable {
    static TABLE: std::sync::OnceLock<PublicGBoundaryTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(PublicGBoundaryTable::new)
}

fn small_signed_h_relation_table() -> &'static SmallSignedPointTable {
    static TABLE: std::sync::OnceLock<SmallSignedPointTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| SmallSignedPointTable::new(h(), RECOVERY_KEY_ENUMERABILITY_BOUND))
}

impl SmallSignedPointTable {
    fn new(generator: ProjectivePoint, bound: u16) -> Self {
        let mut points = std::collections::HashSet::with_capacity(usize::from(bound) * 2);
        let mut multiple = ProjectivePoint::IDENTITY;
        for _ in 0..bound {
            multiple += generator;
            points.insert(encode_point(&multiple));
            points.insert(encode_point(&(-multiple)));
        }
        Self { points }
    }

    fn contains(&self, point: &ProjectivePoint) -> bool {
        self.points.contains(&encode_point(point))
    }
}

impl PublicGBoundaryTable {
    fn new() -> Self {
        let mut points = std::collections::HashSet::with_capacity(
            usize::from(PUBLIC_G_BOUNDARY_WINDOW_BOUND) * 2,
        );
        let mut multiple = g() * Scalar::from(PARAMS.limb_modulus());
        for _ in 0..=PUBLIC_G_BOUNDARY_WINDOW_BOUND {
            points.insert(encode_point(&multiple));
            points.insert(encode_point(&(-multiple)));
            multiple += g();
        }
        Self { points }
    }

    fn contains(&self, point: &ProjectivePoint) -> bool {
        self.points.contains(&encode_point(point))
    }
}

fn verify_sigma_algebra(
    proof: &Proof,
    c: &ProjectivePoint,
    pk: &ProjectivePoint,
    x: Scalar,
) -> Result<(), Error> {
    carry::carry_verify(
        &proof.value_commitments,
        &proof.complement_commitments,
        &proof.carry,
        &proof.carry_resp,
        x,
    )?;

    linking::linking_verify(
        &linking::LinkingStatement {
            ciphertexts: &proof.elgamal,
            commitments: &proof.value_commitments,
            target: c,
            pk,
        },
        &proof.linking,
        &proof.linking_resp,
        x,
    )
}

/// The 32 statement commitments of the BP++ aggregate, in frozen order:
/// value limbs ‖ complement limbs ‖ carries (soundness-doc §4.1).
fn range_statement(
    value: &[ProjectivePoint],
    complement: &[ProjectivePoint],
    carries: &[ProjectivePoint],
) -> [ProjectivePoint; range_circuit::K] {
    let mut out = [ProjectivePoint::IDENTITY; range_circuit::K];
    for (slot, p) in out
        .iter_mut()
        .zip(value.iter().chain(complement.iter()).chain(carries.iter()))
    {
        *slot = *p;
    }
    out
}

/// Validate a caller context's transcript limits and return its binding bytes:
/// the domain must be nonempty and ≤ [`MAX_CONTEXT_DOMAIN_BYTES`], and the
/// binding ≤ [`MAX_CONTEXT_BINDING_BYTES`]. Shared by [`challenge`] (which then
/// absorbs the validated bytes as transcript items 9–10) and the stripped-path
/// [`crate::stripped::StrippedCapsule::bind`], which re-runs the context gates π
/// would otherwise enforce.
///
/// # Errors
///
/// [`Error::DegenerateInput`] on an empty/oversized domain, an oversized binding,
/// or a context whose `binding_bytes` fails.
pub fn validate_context_limits<C: Context + ?Sized>(context: &C) -> Result<Cow<'_, [u8]>, Error> {
    let domain = context.domain();
    if domain.is_empty() {
        return Err(Error::DegenerateInput("context domain is empty"));
    }
    if domain.len() > MAX_CONTEXT_DOMAIN_BYTES {
        return Err(Error::DegenerateInput("context domain is too large"));
    }
    let binding = context
        .binding_bytes()
        .map_err(|_| Error::DegenerateInput("context binding_bytes failed"))?;
    if binding.len() > MAX_CONTEXT_BINDING_BYTES {
        return Err(Error::DegenerateInput("context binding is too large"));
    }
    Ok(binding)
}

/// Absorb the statement — items 1–18 of the normative order (soundness-doc
/// §3) — and return the running transcript. Item 1 (`protocol_id`) is the
/// domain seed inside [`Transcript::new`]; `2^ℓ` is not absorbed (it is
/// `d^D`); `T` is not absorbed (bound externally by the framework `C == T`
/// gate). The BP++ segment (items 19–22) then squeezes its challenges on this
/// same transcript, and [`sigma_challenge`] closes it.
fn statement_transcript_with_binding(
    inp: &TranscriptInputs,
    context_domain: &str,
    binding: &[u8],
) -> Transcript {
    let mut t = Transcript::new(); // 1: protocol_id (DOMAIN seed)
    t.absorb_u8(STATEMENT_VERSION); // 2
    t.absorb_u16(CHALLENGE_WIDTH); // 3
    t.absorb_u16(u16::try_from(PARAMS.limb_bits).unwrap_or(u16::MAX)); // 4: ℓ
    t.absorb_u16(u16::try_from(PARAMS.limb_count).unwrap_or(u16::MAX)); // 5: L
    t.absorb_u16(u16::try_from(PARAMS.digit_base).unwrap_or(u16::MAX)); // 6: d
    t.absorb_u16(u16::try_from(PARAMS.digits_per_limb).unwrap_or(u16::MAX)); // 7: D
    t.absorb_bytes(&N_BE); // 8: n
    t.absorb_bytes(context_domain.as_bytes()); // 9: context domain
    t.absorb_bytes(binding); // 10: context binding
    t.absorb_point(inp.c); // 11: C
    t.absorb_point(inp.pk); // 12: pk
    t.absorb_point(&h()); // 13: H (local recompute)
    t.absorb_bytes(&range_circuit::frozen_generators_digest()); // 13a: g⃗/h⃗ digest

    // 14: limb weights 2^{ℓk}
    let weights = limb_weights();
    t.absorb_list_len(weights.len());
    for w in &weights {
        t.absorb_scalar(w);
    }
    // 15: ElGamal list (E_k ‖ D_k)
    t.absorb_list_len(inp.elgamal.len());
    for ct in inp.elgamal {
        t.absorb_point(&ct.e);
        t.absorb_point(&ct.d);
    }
    // 16: Com_k
    t.absorb_list_len(inp.value_commitments.len());
    for com in inp.value_commitments {
        t.absorb_point(com);
    }
    // 17: Com̄_k
    t.absorb_list_len(inp.complement_commitments.len());
    for com in inp.complement_commitments {
        t.absorb_point(com);
    }
    // 18: carry commitments ComC_k — the full statement is now bound before
    // any challenge exists.
    t.absorb_list_len(inp.carry_commitments.len());
    for cc in inp.carry_commitments {
        t.absorb_point(cc);
    }
    t
}

fn statement_transcript<C: Context + ?Sized>(
    inp: &TranscriptInputs,
    context: &C,
) -> Result<Transcript, Error> {
    let binding = validate_context_limits(context)?;
    Ok(statement_transcript_with_binding(
        inp,
        context.domain(),
        binding.as_ref(),
    ))
}

fn push_binding_count(out: &mut Vec<u8>, count: usize) {
    push_framed(out, &u32::try_from(count).unwrap_or(u32::MAX).to_be_bytes());
}

fn push_binding_point(out: &mut Vec<u8>, point: &ProjectivePoint) {
    push_framed(out, &encode_point(point));
}

fn push_binding_scalar(out: &mut Vec<u8>, scalar: &Scalar) {
    push_framed(out, &scalar.to_bytes());
}

fn push_binding_point_list(out: &mut Vec<u8>, points: &[ProjectivePoint]) {
    push_binding_count(out, points.len());
    for point in points {
        push_binding_point(out, point);
    }
}

fn push_binding_scalar_list(out: &mut Vec<u8>, scalars: &[Scalar]) {
    push_binding_count(out, scalars.len());
    for scalar in scalars {
        push_binding_scalar(out, scalar);
    }
}

/// Build the internal prover nonce binding for the linking sigma. This mirrors
/// every public field that can influence `sigma.x` before linking's own
/// announcements are generated, so an RNG byte-stream repeat across a different
/// statement/context/range/carry transcript cannot repeat `α`/`β`/`γ`.
fn linking_nonce_binding(
    inp: &TranscriptInputs,
    context_domain: &str,
    binding: &[u8],
    carry: &CarryCommitment,
    range: &RangeProof,
) -> Vec<u8> {
    let mut out = Vec::new();
    push_framed(&mut out, b"ve-capsule.linking-sigma.nonce-binding.v1");
    push_framed(&mut out, &[STATEMENT_VERSION]);
    push_framed(&mut out, &CHALLENGE_WIDTH.to_be_bytes());
    push_framed(
        &mut out,
        &u16::try_from(PARAMS.limb_bits)
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    push_framed(
        &mut out,
        &u16::try_from(PARAMS.limb_count)
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    push_framed(
        &mut out,
        &u16::try_from(PARAMS.digit_base)
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    push_framed(
        &mut out,
        &u16::try_from(PARAMS.digits_per_limb)
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    push_framed(&mut out, &N_BE);
    push_framed(&mut out, context_domain.as_bytes());
    push_framed(&mut out, binding);
    push_binding_point(&mut out, inp.c);
    push_binding_point(&mut out, inp.pk);
    push_binding_point(&mut out, &h());
    push_framed(&mut out, &range_circuit::frozen_generators_digest());

    let weights = limb_weights();
    push_binding_scalar_list(&mut out, &weights);
    push_binding_count(&mut out, inp.elgamal.len());
    for ct in inp.elgamal {
        push_binding_point(&mut out, &ct.e);
        push_binding_point(&mut out, &ct.d);
    }
    push_binding_point_list(&mut out, inp.value_commitments);
    push_binding_point_list(&mut out, inp.complement_commitments);
    push_binding_point_list(&mut out, inp.carry_commitments);
    push_binding_point_list(&mut out, &carry.residual_announcements);

    push_binding_point(&mut out, &range.c_l);
    push_binding_point(&mut out, &range.c_o);
    push_binding_point(&mut out, &range.c_r);
    push_binding_point(&mut out, &range.c_s);
    push_binding_point_list(&mut out, &range.folds.x);
    push_binding_point_list(&mut out, &range.folds.r);
    push_binding_scalar_list(&mut out, &range.folds.l);
    push_binding_scalar_list(&mut out, &range.folds.n);

    out
}

/// Absorb the sigma announcements (items 24–25) and squeeze the final
/// challenge `x` (label `sigma.x`), consuming the transcript — the type-level
/// final-squeeze rule (§2).
fn sigma_challenge(
    mut t: Transcript,
    carry: &CarryCommitment,
    linking: &LinkingCommitment,
) -> Scalar {
    // 24: carry residual Schnorr announcements
    t.absorb_list_len(carry.residual_announcements.len());
    for a in &carry.residual_announcements {
        t.absorb_point(a);
    }
    // 25: linking announcements — A^E, A^D, A^Com (each L), then the aggregate A^C
    t.absorb_list_len(linking.a_e.len());
    for a in &linking.a_e {
        t.absorb_point(a);
    }
    t.absorb_list_len(linking.a_d.len());
    for a in &linking.a_d {
        t.absorb_point(a);
    }
    t.absorb_list_len(linking.a_com.len());
    for a in &linking.a_com {
        t.absorb_point(a);
    }
    t.absorb_point(&linking.a_c);

    t.finalize(b"sigma.x")
}

/// Seal `m` into a recovery package for `pk`, returning the [`Proof`] and the
/// commitment `C = m·G`.
///
/// # Errors
///
/// Returns [`Error::DegenerateInput`] if `pk` is the identity, or if a
/// sub-proof rejects its input (e.g. a context binding failure).
pub fn seal<R: RngCore + CryptoRng, C: Context + ?Sized>(
    m: &Scalar,
    pk: &ProjectivePoint,
    context: &C,
    rng: &mut R,
) -> Result<(Proof, ProjectivePoint), Error> {
    reject_degenerate_recovery_key(pk, Error::DegenerateInput)?;

    seal_inner(m, pk, context, rng, |limb, pk, rng| {
        LimbCiphertext::encrypt(limb, pk, rng)
    })
}

/// The seal body, parameterized over the `ElGamal` encryptor so tests can
/// inject controlled mask scalars while production uses the honest one.
fn seal_inner<R: RngCore + CryptoRng, C: Context + ?Sized, E>(
    m: &Scalar,
    pk: &ProjectivePoint,
    context: &C,
    rng: &mut R,
    encrypt: E,
) -> Result<(Proof, ProjectivePoint), Error>
where
    E: FnMut(u32, &ProjectivePoint, &mut R) -> Result<(LimbCiphertext, Scalar), Error>,
{
    let (proof, c) = seal_inner_ungated(m, pk, context, rng, encrypt, |_, _, rng| {
        use k256::elliptic_curve::Field as _;
        Scalar::random(rng)
    })?;
    if let Some(detail) = degenerate_elgamal_mask(proof.elgamal()) {
        return Err(Error::DegenerateInput(detail));
    }
    if let Some(detail) = degenerate_pedersen_commitment(&proof) {
        return Err(Error::DegenerateInput(detail));
    }
    Ok((proof, c))
}

/// [`seal_inner`] without the degenerate-mask seal gate — the shared body.
/// Tests use it directly to build packages whose masks only `verify` rejects.
#[allow(clippy::many_single_char_names)] // m, v, c, t, x: the protocol's symbols
fn seal_inner_ungated<R: RngCore + CryptoRng, C: Context + ?Sized, E>(
    m: &Scalar,
    pk: &ProjectivePoint,
    context: &C,
    rng: &mut R,
    mut encrypt: E,
    mut commitment_blinding: impl FnMut(usize, bool, &mut R) -> Scalar,
) -> Result<(Proof, ProjectivePoint), Error>
where
    E: FnMut(u32, &ProjectivePoint, &mut R) -> Result<(LimbCiphertext, Scalar), Error>,
{
    let context_binding = validate_context_limits(context)?;
    let context_domain = context.domain();

    // Secret witness material — the limbs of m / m̄ and the per-limb scalars
    // handed to the proofs — is zeroized on drop.
    let v = Zeroizing::new(decompose(m));
    let v_bar = Zeroizing::new(decompose(&(-Scalar::ONE - *m)));

    let mut elgamal = Vec::with_capacity(LIMB_COUNT);
    let mut value_commitments = Vec::with_capacity(LIMB_COUNT);
    let mut complement_commitments = Vec::with_capacity(LIMB_COUNT);
    let mut v_scalars = Zeroizing::new(Vec::with_capacity(LIMB_COUNT));
    let mut r_scalars = Zeroizing::new(Vec::with_capacity(LIMB_COUNT));
    let mut s_scalars = Zeroizing::new(Vec::with_capacity(LIMB_COUNT));
    let mut s_bar_scalars = Zeroizing::new(Vec::with_capacity(LIMB_COUNT));

    for k in 0..LIMB_COUNT {
        let com_k = Commitment::with_blinding(
            Scalar::from(u64::from(v[k])),
            commitment_blinding(k, false, rng),
        );
        let com_bar_k = Commitment::with_blinding(
            Scalar::from(u64::from(v_bar[k])),
            commitment_blinding(k, true, rng),
        );
        let (ct, r_k) = encrypt(v[k], pk, rng)?;

        v_scalars.push(com_k.value);
        s_scalars.push(com_k.blinding);
        s_bar_scalars.push(com_bar_k.blinding);
        r_scalars.push(r_k);
        value_commitments.push(com_k.point);
        complement_commitments.push(com_bar_k.point);
        elgamal.push(ct);
    }

    let (carry, carry_state, carry_witness) =
        carry::carry_commit(m, &s_scalars, &s_bar_scalars, rng)?;
    let c = g() * m;
    let transcript_inputs = TranscriptInputs {
        c: &c,
        pk,
        elgamal: &elgamal,
        value_commitments: &value_commitments,
        complement_commitments: &complement_commitments,
        carry_commitments: &carry.carry_commitments,
    };

    // Items 1–18: the full statement, before any challenge.
    let mut t = statement_transcript_with_binding(
        &transcript_inputs,
        context_domain,
        context_binding.as_ref(),
    );

    // Items 19–22: the aggregated BP++ range proof, multi-squeeze.
    let mut values = [0u32; range_circuit::K];
    let mut blindings = [Scalar::ZERO; range_circuit::K];
    for k in 0..LIMB_COUNT {
        values[k] = v[k];
        values[LIMB_COUNT + k] = v_bar[k];
        blindings[k] = s_scalars[k];
        blindings[LIMB_COUNT + k] = s_bar_scalars[k];
    }
    for (k, (&bit, &blind)) in carry_witness
        .bits
        .iter()
        .zip(carry_witness.blindings.iter())
        .enumerate()
    {
        values[2 * LIMB_COUNT + k] = bit;
        blindings[2 * LIMB_COUNT + k] = blind;
    }
    let witness = RangeWitness { values, blindings };
    let statement = range_statement(
        &value_commitments,
        &complement_commitments,
        &carry.carry_commitments,
    );
    let range = range_circuit::prove(&statement, &witness, &mut TranscriptChallenges(&mut t), rng)?;
    let nonce_binding = linking_nonce_binding(
        &transcript_inputs,
        context_domain,
        context_binding.as_ref(),
        &carry,
        &range,
    );
    let (linking, linking_state) =
        linking::linking_commit(&v_scalars, &r_scalars, &s_scalars, pk, &nonce_binding, rng)?;

    // Items 24–25 → the final squeeze.
    let x = sigma_challenge(t, &carry, &linking);

    Ok((
        Proof {
            elgamal,
            value_commitments,
            complement_commitments,
            range,
            carry,
            carry_resp: carry_state.respond(x),
            linking,
            linking_resp: linking_state.respond(x),
        },
        c,
    ))
}

#[cfg(test)]
pub fn seal_with_prefix_mask_scalars_for_test<R, C>(
    m: &Scalar,
    pk: &ProjectivePoint,
    context: &C,
    rng: &mut R,
    prefix: &[Scalar],
) -> Result<(Proof, ProjectivePoint), Error>
where
    R: RngCore + CryptoRng,
    C: Context + ?Sized,
{
    let mut k = 0usize;
    seal_inner_ungated(
        m,
        pk,
        context,
        rng,
        |limb, pk, rng| {
            let r_k = prefix.get(k).copied().unwrap_or_else(|| {
                use k256::elliptic_curve::Field as _;
                loop {
                    let scalar = Scalar::random(&mut *rng);
                    if !bool::from(scalar.is_zero()) {
                        break scalar;
                    }
                }
            });
            k += 1;
            let limb = Scalar::from(u64::from(limb));
            Ok((
                LimbCiphertext {
                    e: g() * r_k,
                    d: g() * limb + *pk * r_k,
                },
                r_k,
            ))
        },
        |_, _, rng| {
            use k256::elliptic_curve::Field as _;
            Scalar::random(rng)
        },
    )
}

#[cfg(test)]
pub fn seal_with_prefix_value_blindings_for_test<R, C>(
    m: &Scalar,
    pk: &ProjectivePoint,
    context: &C,
    rng: &mut R,
    prefix: &[Scalar],
) -> Result<(Proof, ProjectivePoint), Error>
where
    R: RngCore + CryptoRng,
    C: Context + ?Sized,
{
    seal_inner_ungated(
        m,
        pk,
        context,
        rng,
        LimbCiphertext::encrypt,
        |k, is_complement, rng| {
            use k256::elliptic_curve::Field as _;

            if !is_complement {
                if let Some(blinding) = prefix.get(k) {
                    return *blinding;
                }
            }
            Scalar::random(rng)
        },
    )
}

/// Verify a proof against the commitment `C`, recovery key `pk`, and
/// caller `context`.
///
/// Re-runs the whole transcript schedule and every gate: the aggregated BP++
/// range proof (limbs + carry booleanity), the carry residuals, and the
/// linking sigma.
///
/// # Errors
///
/// Returns [`Error::Verification`] on an identity `pk`, a shape mismatch, or any
/// failed sub-proof.
pub fn verify<C: Context + ?Sized>(
    proof: &Proof,
    c: &ProjectivePoint,
    pk: &ProjectivePoint,
    context: &C,
) -> Result<(), Error> {
    verify_with_mask_profile(proof, c, pk, context, MaskScreening::Full)
}

/// Verify one proof inside a proof-backed Case.
///
/// Case verification runs the expensive cross-piece relation scans after every
/// piece proof is algebraically valid. A same-piece low-arity mask relation can
/// only reveal that sealer's own additive piece, which is outside the Case
/// producer-to-producer privacy model; the cross-piece scans below remain the
/// guard against one producer learning another producer's private contribution.
pub fn verify_case_piece<C: Context + ?Sized>(
    proof: &Proof,
    c: &ProjectivePoint,
    pk: &ProjectivePoint,
    context: &C,
) -> Result<(), Error> {
    verify_with_mask_profile(proof, c, pk, context, MaskScreening::CasePiece)
}

#[derive(Clone, Copy)]
enum MaskScreening {
    Full,
    CasePiece,
}

fn verify_with_mask_profile<C: Context + ?Sized>(
    proof: &Proof,
    c: &ProjectivePoint,
    pk: &ProjectivePoint,
    context: &C,
    mask_screening: MaskScreening,
) -> Result<(), Error> {
    reject_degenerate_recovery_key(pk, Error::Verification)?;
    if proof.elgamal.len() != LIMB_COUNT
        || proof.value_commitments.len() != LIMB_COUNT
        || proof.complement_commitments.len() != LIMB_COUNT
        || proof.carry.carry_commitments.len() != LIMB_COUNT - 1
    {
        return Err(Error::Verification("proof shape mismatch"));
    }
    // Structural mask gates (identity, duplicate/inverse, low-coefficient
    // relations — soundness-doc §4.4 "Retained mask gates"): reject before
    // transcript/proof work.
    let mask_detail = match mask_screening {
        MaskScreening::Full => degenerate_elgamal_mask(&proof.elgamal),
        MaskScreening::CasePiece => degenerate_elgamal_mask_basic(&proof.elgamal),
    };
    if let Some(detail) = mask_detail {
        return Err(Error::Verification(detail));
    }
    if let Some(detail) = degenerate_pedersen_commitment(proof) {
        return Err(Error::Verification(detail));
    }

    let mut t = statement_transcript(
        &TranscriptInputs {
            c,
            pk,
            elgamal: &proof.elgamal,
            value_commitments: &proof.value_commitments,
            complement_commitments: &proof.complement_commitments,
            carry_commitments: &proof.carry.carry_commitments,
        },
        context,
    )?;

    let statement = range_statement(
        &proof.value_commitments,
        &proof.complement_commitments,
        &proof.carry.carry_commitments,
    );
    range_circuit::verify(&statement, &proof.range, &mut TranscriptChallenges(&mut t))?;

    let x = sigma_challenge(t, &proof.carry, &proof.linking);
    verify_sigma_algebra(proof, c, pk, x)?;

    Ok(())
}

/// Recover `m` from a package using the recovery secret key `sk` (with
/// `pk = sk·G`): decrypt each limb to `v_k·G`, BSGS to `v_k`, recompose, and
/// recheck `m·G == C` as defense-in-depth.
///
/// # Errors
///
/// Returns [`Error::Verification`] on a shape mismatch or an opening that does
/// not reconstruct a scalar whose commitment is `C`. Returns
/// [`Error::DegenerateInput`] if a ciphertext mask is the identity.
///
/// The single-secret open primitive: the production recipient opens via the
/// contribution [`crate::opening`] layer (mask-strip, no `sk` for `Y*`), but the
/// seal/verify round-trip tests and `composite`'s sum-secret check exercise this.
#[allow(dead_code)]
pub fn open(proof: &Proof, c: &ProjectivePoint, sk: &Scalar) -> Result<Scalar, Error> {
    if proof.elgamal.len() != LIMB_COUNT {
        return Err(Error::Verification("proof shape mismatch"));
    }
    let table = baby_table();
    let mut limbs = [0u32; LIMB_COUNT];
    let mut limb_missing = false;
    for (k, ct) in proof.elgamal.iter().enumerate() {
        let point = ct.decrypt_point(sk)?;
        match table.recover_bounded_complete(&point, LIMB_MODULUS) {
            Some(recovered) => limbs[k] = recovered,
            None => limb_missing = true,
        }
    }
    let m = recompose(&limbs);
    if limb_missing || &(g() * m) != c {
        return Err(Error::Verification("opening failed"));
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;
    use crate::bsgs::BabyTable;
    use k256::elliptic_curve::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::borrow::Cow;

    struct TestContext;
    impl Context for TestContext {
        fn domain(&self) -> &'static str {
            "ve-capsule.ec-segve.test"
        }
        fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
            Ok(Cow::Borrowed(b"test-binding-payload"))
        }
    }

    fn keypair(rng: &mut StdRng) -> (Scalar, ProjectivePoint) {
        let sk = Scalar::random(rng);
        (sk, g() * sk)
    }

    fn random_nonzero_scalar_for_test(rng: &mut StdRng) -> Scalar {
        loop {
            let scalar = Scalar::random(&mut *rng);
            if !bool::from(scalar.is_zero()) {
                return scalar;
            }
        }
    }

    fn seal_with_prefix_mask_scalars(
        m: &Scalar,
        pk: &ProjectivePoint,
        context: &TestContext,
        rng: &mut StdRng,
        prefix: &[Scalar],
    ) -> Result<(Proof, ProjectivePoint), Error> {
        // The honest seal body with injected mask scalars: prefix[k] where
        // provided, fresh nonzero randomness elsewhere. Note seal_inner runs
        // the degenerate-mask gates, which these tests rely on exercising at
        // VERIFY; so inject through a gate-free encryptor and skip the seal
        // gate by sealing each prefix case manually below.
        let mut k = 0usize;
        seal_inner_ungated(
            m,
            pk,
            context,
            rng,
            |limb, pk, rng| {
                let r_k = prefix
                    .get(k)
                    .copied()
                    .unwrap_or_else(|| random_nonzero_scalar_for_test(rng));
                k += 1;
                let limb = Scalar::from(u64::from(limb));
                Ok((
                    LimbCiphertext {
                        e: g() * r_k,
                        d: g() * limb + *pk * r_k,
                    },
                    r_k,
                ))
            },
            |_, _, rng| {
                use k256::elliptic_curve::Field as _;
                Scalar::random(rng)
            },
        )
    }

    fn seal_with_prefix_value_blindings(
        m: &Scalar,
        pk: &ProjectivePoint,
        context: &TestContext,
        rng: &mut StdRng,
        prefix: &[Scalar],
    ) -> Result<(Proof, ProjectivePoint), Error> {
        seal_inner_ungated(
            m,
            pk,
            context,
            rng,
            LimbCiphertext::encrypt,
            |k, is_complement, rng| {
                use k256::elliptic_curve::Field as _;

                if !is_complement {
                    if let Some(blinding) = prefix.get(k) {
                        return *blinding;
                    }
                }
                Scalar::random(rng)
            },
        )
    }

    #[test]
    fn order_constant_is_n() {
        // N_BE must be exactly the group order: n − 1 = −1, and only the last
        // byte differs by one (n ends …41, n−1 ends …40, no borrow).
        let nm1 = (-Scalar::ONE).to_bytes();
        assert_eq!(N_BE[..31], nm1[..31]);
        assert_eq!(nm1[31], 0x40);
        assert_eq!(N_BE[31], 0x41);
    }

    #[test]
    fn seal_verify_open_round_trip() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_01);
        let ctx = TestContext;
        let max = -Scalar::ONE; // n − 1
        for m in [
            Scalar::ZERO,
            Scalar::ONE,
            Scalar::from(0x00AB_CDEF_1234_5678u64),
            max,
            Scalar::random(&mut rng),
        ] {
            let (sk, pk) = keypair(&mut rng);
            let (proof, c) = seal(&m, &pk, &ctx, &mut rng).unwrap();
            assert!(verify(&proof, &c, &pk, &ctx).is_ok(), "verify failed");
            assert_eq!(open(&proof, &c, &sk).unwrap(), m, "open mismatch");
        }
    }

    #[test]
    fn identity_mask_rejected() {
        // Poison: an E_k = identity (r_k = 0) makes D_k = v_k·G a public
        // plaintext. verify must reject it outright (not defer to open).
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_09);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let (mut proof, c) = seal(&Scalar::from(64u64), &pk, &ctx, &mut rng).unwrap();
        proof.elgamal[0].e = ProjectivePoint::IDENTITY;
        assert!(matches!(
            verify(&proof, &c, &pk, &ctx),
            Err(Error::Verification(IDENTITY_MASK_DETAIL))
        ));
    }

    #[test]
    fn duplicate_elgamal_mask_rejected() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_0A);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let r = random_nonzero_scalar_for_test(&mut rng);
        let (proof, c) =
            seal_with_prefix_mask_scalars(&Scalar::from(64u64), &pk, &ctx, &mut rng, &[r, r])
                .unwrap();
        assert!(matches!(
            verify(&proof, &c, &pk, &ctx),
            Err(Error::Verification("ElGamal mask repeats a previous mask"))
        ));
    }

    #[test]
    fn inverse_elgamal_mask_rejected() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_0B);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let r = random_nonzero_scalar_for_test(&mut rng);
        let (proof, c) =
            seal_with_prefix_mask_scalars(&Scalar::from(64u64), &pk, &ctx, &mut rng, &[r, -r])
                .unwrap();
        assert!(matches!(
            verify(&proof, &c, &pk, &ctx),
            Err(Error::Verification("ElGamal mask inverts a previous mask"))
        ));
    }

    #[test]
    fn signed_subset_elgamal_mask_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_0C);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let r0 = random_nonzero_scalar_for_test(&mut rng);
        let r1 = random_nonzero_scalar_for_test(&mut rng);
        let r2 = r0 + r1;
        let (proof, c) =
            seal_with_prefix_mask_scalars(&Scalar::from(64u64), &pk, &ctx, &mut rng, &[r0, r1, r2])
                .unwrap();
        // E_2 = E_0 + E_1 is a coefficient-1 signed subset; the coefficient-2
        // small-coefficient scan subsumes it, so it is the screen that now
        // rejects the bundle.
        assert!(matches!(
            verify(&proof, &c, &pk, &ctx),
            Err(Error::Verification(
                "ElGamal masks have a small-coefficient relation"
            ))
        ));
    }

    #[test]
    fn small_coefficient_elgamal_mask_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_0D);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let r0 = random_nonzero_scalar_for_test(&mut rng);
        let r1 = random_nonzero_scalar_for_test(&mut rng);
        let r2 = r0 + r0 + r1;
        let (proof, c) =
            seal_with_prefix_mask_scalars(&Scalar::from(64u64), &pk, &ctx, &mut rng, &[r0, r1, r2])
                .unwrap();
        assert!(matches!(
            verify(&proof, &c, &pk, &ctx),
            Err(Error::Verification(
                "ElGamal masks have a small-coefficient relation"
            ))
        ));
    }

    #[test]
    fn pairwise_scalar_elgamal_mask_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_0E);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let r0 = random_nonzero_scalar_for_test(&mut rng);
        let r1 = r0 * Scalar::from(3u64);
        let (proof, c) =
            seal_with_prefix_mask_scalars(&Scalar::from(64u64), &pk, &ctx, &mut rng, &[r1, r0])
                .unwrap();

        let leaked_relation = proof.elgamal[0].d - proof.elgamal[1].d * Scalar::from(3u64);
        assert_eq!(leaked_relation, g() * Scalar::from(64u64));
        assert!(matches!(
            verify(&proof, &c, &pk, &ctx),
            Err(Error::Verification(
                "ElGamal masks have a public scalar relation"
            ))
        ));
    }

    #[test]
    fn public_g_offset_elgamal_mask_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_22);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let r0 = random_nonzero_scalar_for_test(&mut rng);
        let r1 = Scalar::ONE - r0;
        let (proof, c) =
            seal_with_prefix_mask_scalars(&Scalar::from(64u64), &pk, &ctx, &mut rng, &[r0, r1])
                .unwrap();

        let leaked_relation = proof.elgamal[0].d + proof.elgamal[1].d - pk;
        assert_eq!(leaked_relation, g() * Scalar::from(64u64));
        assert!(matches!(
            verify(&proof, &c, &pk, &ctx),
            Err(Error::Verification(
                "ElGamal masks have a public G-offset relation"
            ))
        ));
    }

    #[test]
    fn publicly_openable_pedersen_commitment_rejected() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_1B);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let (proof, c) = seal_with_prefix_value_blindings(
            &Scalar::from(64u64),
            &pk,
            &ctx,
            &mut rng,
            &[Scalar::ZERO],
        )
        .unwrap();

        assert!(matches!(
            verify(&proof, &c, &pk, &ctx),
            Err(Error::Verification(
                "Pedersen commitment has a public opening"
            ))
        ));
    }

    #[test]
    fn cross_piece_pedersen_relation_rejects_over_max_piece_profile() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_1C);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let (proof, _c) = seal(&Scalar::from(64u64), &pk, &ctx, &mut rng).unwrap();
        let proofs = vec![proof; CROSS_PIECE_PEDERSEN_RELATION_PIECE_BOUND + 1];

        assert_eq!(
            cross_piece_pedersen_commitment_relation(proofs.iter()),
            Some("case has too many pieces for cross-piece Pedersen relation scan")
        );
    }

    #[test]
    fn cross_piece_pedersen_relation_rejects_max_admitted_subset_relation() {
        const PIECES: usize = CROSS_PIECE_PEDERSEN_RELATION_PIECE_BOUND;

        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_1E);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let malicious_blindings = (1..PIECES)
            .map(|_| random_nonzero_scalar_for_test(&mut rng))
            .collect::<Vec<_>>();
        let honest_blinding = malicious_blindings
            .iter()
            .copied()
            .fold(Scalar::ZERO, |acc, blinding| acc + blinding);
        let (honest_proof, _c) = seal_with_prefix_value_blindings(
            &Scalar::from(12_345u64),
            &pk,
            &ctx,
            &mut rng,
            &[honest_blinding],
        )
        .unwrap();
        let mut proofs = vec![honest_proof];
        for (idx, blinding) in malicious_blindings.iter().enumerate() {
            let value = Scalar::from(100u64 * (u64::try_from(idx).unwrap() + 1));
            let (proof, _c) =
                seal_with_prefix_value_blindings(&value, &pk, &ctx, &mut rng, &[*blinding])
                    .unwrap();
            proofs.push(proof);
        }

        let leaked = proofs[1..]
            .iter()
            .fold(proofs[0].value_commitments[0], |acc, proof| {
                acc - proof.value_commitments[0]
            });
        assert_eq!(
            BabyTable::new().recover_bounded_complete(&leaked, LIMB_MODULUS),
            Some(12_345 - 100 - 200 - 300 - 400 - 500),
            "max-admitted Pedersen subset cancellation publicly reveals an honest-limb equation"
        );
        assert_eq!(
            cross_piece_pedersen_commitment_relation(proofs.iter()),
            Some("case pieces have a cross-piece Pedersen commitment relation")
        );
    }

    #[test]
    fn cross_piece_pedersen_relation_rejects_cross_slot_pair_relation() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_23);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let reused_blinding = random_nonzero_scalar_for_test(&mut rng);
        let unrelated_blinding = random_nonzero_scalar_for_test(&mut rng);
        let (honest_proof, _c) = seal_with_prefix_value_blindings(
            &Scalar::from(12_345u64),
            &pk,
            &ctx,
            &mut rng,
            &[reused_blinding],
        )
        .unwrap();
        let malicious_value = recompose(&{
            let mut limbs = [0u32; LIMB_COUNT];
            limbs[0] = 100;
            limbs[1] = 200;
            limbs
        });
        let (malicious_proof, _c) = seal_with_prefix_value_blindings(
            &malicious_value,
            &pk,
            &ctx,
            &mut rng,
            &[unrelated_blinding, reused_blinding],
        )
        .unwrap();

        let leaked = honest_proof.value_commitments[0] - malicious_proof.value_commitments[1];
        assert_eq!(
            BabyTable::new().recover_bounded_complete(&leaked, LIMB_MODULUS),
            Some(12_345 - 200),
            "cross-slot Pedersen blinding reuse publicly reveals an honest-limb equation"
        );
        assert_eq!(
            cross_piece_pedersen_commitment_relation([&honest_proof, &malicious_proof]),
            Some("case pieces have a cross-piece Pedersen commitment relation")
        );
    }

    #[test]
    #[ignore = "manual perf baseline; run with --release --ignored --nocapture"]
    fn cross_piece_pedersen_relation_max_profile_latency() {
        use std::time::Instant;

        const PIECES: usize = CROSS_PIECE_PEDERSEN_RELATION_PIECE_BOUND;
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_1D);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let mut proofs = Vec::with_capacity(PIECES);
        for piece in 0..PIECES {
            let (proof, _c) = seal(
                &Scalar::from(u64::try_from(64 + piece).unwrap()),
                &pk,
                &ctx,
                &mut rng,
            )
            .unwrap();
            proofs.push(proof);
        }

        let start = Instant::now();
        let relation = cross_piece_pedersen_commitment_relation(proofs.iter());
        let total_ms = start.elapsed().as_secs_f64() * 1e3;
        println!(
            "cross_piece_pedersen_relation pieces={PIECES} slots={} total_ms={total_ms:.3}",
            range_circuit::K
        );
        assert_eq!(relation, None);
    }

    #[test]
    #[ignore = "manual perf baseline; run with --release --ignored --nocapture"]
    fn cross_piece_pedersen_relation_phase_latency() {
        use std::time::Instant;

        const PIECES: usize = CROSS_PIECE_PEDERSEN_RELATION_PIECE_BOUND;
        let mut rng = StdRng::seed_from_u64(0x5E_A1_70_03);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let mut slots = vec![Vec::<(usize, ProjectivePoint)>::new(); range_circuit::K];
        let mut all_commitments = Vec::new();
        for piece_index in 0..PIECES {
            let (proof, _c) = seal(
                &Scalar::from(u64::try_from(64 + piece_index).unwrap()),
                &pk,
                &ctx,
                &mut rng,
            )
            .unwrap();
            let commitments = pedersen_statement_commitments(&proof).unwrap();
            for (slot_index, (slot, commitment)) in slots.iter_mut().zip(commitments).enumerate() {
                slot.push((piece_index, commitment));
                all_commitments.push(IndexedPedersenCommitment {
                    piece_index,
                    point: commitment,
                    value_bound: pedersen_statement_value_bound(slot_index),
                });
            }
        }

        let start = Instant::now();
        assert!(!has_cross_piece_pair_pedersen_public_g_offset_relation(
            &all_commitments
        ));
        let pair_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        let mut candidates = std::collections::BTreeMap::<u64, Vec<ProjectivePoint>>::new();
        for (slot_index, slot) in slots.into_iter().enumerate() {
            collect_cross_piece_public_g_offset_candidates(
                &slot,
                pedersen_statement_value_bound(slot_index),
                &mut candidates,
            );
        }
        assert!(!has_public_g_offset_candidate(candidates));
        let subset_ms = start.elapsed().as_secs_f64() * 1e3;

        println!("cross_piece_pedersen_phases pair={pair_ms:.3} subset={subset_ms:.3}");
    }

    #[test]
    #[ignore = "manual perf baseline; run with --release --ignored --nocapture"]
    fn cross_piece_mask_relation_max_profile_latency() {
        use std::time::Instant;

        const PIECES: usize = CROSS_PIECE_MASK_RELATION_PIECE_BOUND;
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_19);
        let mut pieces = Vec::with_capacity(PIECES);
        for _ in 0..PIECES {
            let mut piece = Vec::with_capacity(LIMB_COUNT);
            for _ in 0..LIMB_COUNT {
                piece.push(LimbCiphertext {
                    e: g() * random_nonzero_scalar_for_test(&mut rng),
                    d: g() * random_nonzero_scalar_for_test(&mut rng),
                });
            }
            assert_eq!(degenerate_elgamal_mask(&piece), None);
            pieces.push(piece);
        }

        let start = Instant::now();
        let relation = cross_piece_elgamal_mask_relation(pieces.iter().map(Vec::as_slice));
        let total_ms = start.elapsed().as_secs_f64() * 1e3;
        println!(
            "cross_piece_mask_relation pieces={PIECES} masks={} total_ms={total_ms:.3}",
            PIECES * LIMB_COUNT
        );
        assert_eq!(relation, None);
    }

    #[test]
    #[ignore = "manual perf baseline; run with --release --ignored --nocapture"]
    fn cross_piece_mask_relation_phase_latency() {
        use std::time::Instant;

        const PIECES: usize = CROSS_PIECE_MASK_RELATION_PIECE_BOUND;
        let mut rng = StdRng::seed_from_u64(0x5E_A1_70_02);
        let mut masks = Vec::with_capacity(PIECES * LIMB_COUNT);
        for piece_index in 0..PIECES {
            for _ in 0..LIMB_COUNT {
                masks.push(IndexedMask {
                    mask_index: masks.len(),
                    piece_index,
                    point: g() * random_nonzero_scalar_for_test(&mut rng),
                });
            }
        }

        let start = Instant::now();
        assert!(!has_public_scalar_mask_relation(
            &masks,
            RECOVERY_KEY_ENUMERABILITY_BOUND,
            true,
        ));
        let scalar_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        assert!(!has_cross_piece_pair_public_g_offset_relation(&masks));
        let pair_public_g_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        assert!(!has_cross_piece_mask_relation_in_support_range(
            &masks,
            1,
            CROSS_PIECE_SMALL_COEFF_RELATION_SUPPORT_BOUND + 1,
            CROSS_PIECE_UNIT_RELATION_SUPPORT_BOUND,
        ));
        let unit_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        assert!(!has_cross_piece_mask_relation_with_support(
            &masks,
            MASK_RELATION_COEFFICIENT_BOUND,
            CROSS_PIECE_SMALL_COEFF_RELATION_SUPPORT_BOUND,
        ));
        let coeff_ms = start.elapsed().as_secs_f64() * 1e3;

        println!(
            "cross_piece_mask_phases scalar={scalar_ms:.3} pair_public_g={pair_public_g_ms:.3} \
             unit={unit_ms:.3} coeff={coeff_ms:.3}"
        );
    }

    #[test]
    fn cross_piece_mask_relation_rejects_over_max_piece_profile() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_1A);
        let pieces = (0..=CROSS_PIECE_MASK_RELATION_PIECE_BOUND)
            .map(|_| {
                vec![LimbCiphertext {
                    e: g() * random_nonzero_scalar_for_test(&mut rng),
                    d: g() * random_nonzero_scalar_for_test(&mut rng),
                }]
            })
            .collect::<Vec<_>>();

        assert_eq!(
            cross_piece_elgamal_mask_relation(pieces.iter().map(Vec::as_slice)),
            Some("case has too many pieces for cross-piece mask relation scan")
        );
    }

    #[test]
    fn publicly_enumerable_recovery_key_rejected_structurally() {
        // If the recipient public key has a publicly recoverable discrete log,
        // every verifier can act as the offline recipient. The proof remains
        // internally valid, but the ElGamal layer no longer hides any limb.
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_20);
        let ctx = TestContext;
        let public_sk = Scalar::from(7u64);
        let pk = g() * public_sk;
        let m = Scalar::from(0x00AB_CDEFu64);
        let table = BabyTable::new();
        let (proof, c) = seal_with_prefix_mask_scalars(&m, &pk, &ctx, &mut rng, &[]).unwrap();

        assert_eq!(table.recover(&pk), Some(7));
        assert_eq!(open(&proof, &c, &public_sk).unwrap(), m);
        assert_eq!(
            seal(&m, &pk, &ctx, &mut rng).unwrap_err(),
            Error::DegenerateInput("recovery public key is publicly enumerable")
        );
        assert_eq!(
            verify(&proof, &c, &pk, &ctx),
            Err(Error::Verification(
                "recovery public key is publicly enumerable"
            ))
        );
    }

    #[test]
    fn limb_boundary_recovery_key_rejected_structurally() {
        // The limb BSGS table intentionally excludes 2^ell for plaintext
        // recovery, but the same boundary scalar is still a public recovery
        // key. If accepted, every verifier can decrypt the package.
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_2D);
        let ctx = TestContext;
        let public_sk = Scalar::from(PARAMS.limb_modulus());
        let pk = g() * public_sk;
        let m = Scalar::from(0x00AB_CDEFu64);
        let table = BabyTable::new();
        let (proof, c) = seal_with_prefix_mask_scalars(&m, &pk, &ctx, &mut rng, &[]).unwrap();

        assert_eq!(table.recover(&pk), None);
        assert_eq!(open(&proof, &c, &public_sk).unwrap(), m);
        assert_eq!(
            verify(&proof, &c, &pk, &ctx),
            Err(Error::Verification(
                "recovery public key is publicly enumerable"
            ))
        );
        assert_eq!(
            seal(&m, &pk, &ctx, &mut rng).unwrap_err(),
            Error::DegenerateInput("recovery public key is publicly enumerable")
        );
    }

    #[test]
    fn nums_generator_recovery_key_rejected_structurally() {
        // H is deliberately a NUMS point with unknown log_G(H). Accepting it as
        // a recovery key would produce packages that verify but cannot be
        // opened by any known secp256k1 private key.
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_2A);
        let ctx = TestContext;
        let pk = h();
        let m = Scalar::from(0x00AB_CDEFu64);
        let (proof, c) = seal_with_prefix_mask_scalars(&m, &pk, &ctx, &mut rng, &[]).unwrap();

        assert_eq!(
            seal(&m, &pk, &ctx, &mut rng).unwrap_err(),
            Error::DegenerateInput("recovery public key is a public NUMS-generator multiple")
        );
        assert_eq!(
            verify(&proof, &c, &pk, &ctx),
            Err(Error::Verification(
                "recovery public key is a public NUMS-generator multiple"
            ))
        );
    }

    /// Seal a baseline package, apply `corrupt`, and assert `verify` rejects —
    /// confirming every gate is actually wired (a single-field corruption of
    /// any component must be caught, not silently accepted).
    fn assert_corruption_rejected(label: &str, corrupt: impl FnOnce(&mut Proof)) {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_FF_01);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let (mut proof, c) = seal(&Scalar::from(0xBEEFu64), &pk, &ctx, &mut rng).unwrap();
        assert!(
            verify(&proof, &c, &pk, &ctx).is_ok(),
            "{label}: baseline must verify"
        );
        corrupt(&mut proof);
        assert!(
            matches!(verify(&proof, &c, &pk, &ctx), Err(Error::Verification(_))),
            "{label}: corruption was not rejected"
        );
    }

    #[test]
    fn tamper_matrix_every_component_rejected() {
        // One corruption per component type across all four sub-proofs. Absorbed
        // items (commitments/announcements) fork the challenge; responses fail
        // their leg under the honest challenge — either way verify must reject.
        assert_corruption_rejected("E_k", |p| p.elgamal[0].e += g());
        assert_corruption_rejected("D_k", |p| p.elgamal[0].d += g());
        assert_corruption_rejected("Com_k", |p| p.value_commitments[0] += g());
        assert_corruption_rejected("Com̄_k", |p| p.complement_commitments[0] += g());
        assert_corruption_rejected("BP++ C_L", |p| p.range.c_l += g());
        assert_corruption_rejected("BP++ C_O", |p| p.range.c_o += g());
        assert_corruption_rejected("BP++ C_R", |p| p.range.c_r += g());
        assert_corruption_rejected("BP++ C_S", |p| p.range.c_s += g());
        assert_corruption_rejected("BP++ fold X", |p| p.range.folds.x[0] += g());
        assert_corruption_rejected("BP++ fold R", |p| p.range.folds.r[3] += g());
        assert_corruption_rejected("BP++ residual l", |p| {
            p.range.folds.l[0] += Scalar::ONE;
        });
        assert_corruption_rejected("BP++ residual n", |p| {
            p.range.folds.n[2] += Scalar::ONE;
        });
        assert_corruption_rejected("carry commitment", |p| p.carry.carry_commitments[0] += g());
        assert_corruption_rejected("carry residual announcement", |p| {
            p.carry.residual_announcements[0] += g();
        });
        assert_corruption_rejected("carry residual response", |p| {
            p.carry_resp.residual_responses[0] += Scalar::ONE;
        });
        assert_corruption_rejected("linking A_E", |p| p.linking.a_e[0] += g());
        assert_corruption_rejected("linking A_D", |p| p.linking.a_d[0] += g());
        assert_corruption_rejected("linking A_Com", |p| p.linking.a_com[0] += g());
        assert_corruption_rejected("linking A_C", |p| p.linking.a_c += g());
        assert_corruption_rejected("linking z_v", |p| p.linking_resp.z_v[0] += Scalar::ONE);
        assert_corruption_rejected("linking z_r", |p| p.linking_resp.z_r[0] += Scalar::ONE);
        assert_corruption_rejected("linking z_s", |p| p.linking_resp.z_s[0] += Scalar::ONE);
    }

    #[test]
    fn wrong_target_rejected() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_05);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let (proof, c) = seal(&Scalar::from(123u64), &pk, &ctx, &mut rng).unwrap();
        assert!(matches!(
            verify(&proof, &(c + g()), &pk, &ctx),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn wrong_context_rejected() {
        // A different binding context re-derives a different challenge.
        struct OtherContext;
        impl Context for OtherContext {
            fn domain(&self) -> &'static str {
                "ve-capsule.ec-segve.test"
            }
            fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
                Ok(Cow::Borrowed(b"different-binding"))
            }
        }
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_06);
        let (_sk, pk) = keypair(&mut rng);
        let (proof, c) = seal(&Scalar::from(55u64), &pk, &TestContext, &mut rng).unwrap();
        assert!(matches!(
            verify(&proof, &c, &pk, &OtherContext),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn open_wrong_key_rejected() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_07);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let (proof, c) = seal(&Scalar::from(31337u64), &pk, &ctx, &mut rng).unwrap();
        let wrong_sk = Scalar::random(&mut rng);
        assert!(matches!(
            open(&proof, &c, &wrong_sk),
            Err(Error::Verification(_))
        ));
    }

    #[test]
    fn seal_rejects_identity_pk() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_08);
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &ProjectivePoint::IDENTITY,
                &TestContext,
                &mut rng
            ),
            Err(Error::DegenerateInput(_))
        ));
    }

    #[test]
    fn empty_context_domain_rejected() {
        struct EmptyDomain;
        impl Context for EmptyDomain {
            fn domain(&self) -> &'static str {
                ""
            }
            fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
                Ok(Cow::Borrowed(b"binding"))
            }
        }

        let mut rng = StdRng::seed_from_u64(0x5E_A1_00_0A);
        let (_sk, pk) = keypair(&mut rng);
        assert!(matches!(
            seal(&Scalar::from(1u64), &pk, &EmptyDomain, &mut rng),
            Err(Error::DegenerateInput(_))
        ));
    }

    #[test]
    fn proof_canonical_round_trip() {
        // Decode mirrors encode exactly: re-encode-equality (canonical layout)
        // and the decoded proof still verifies.
        let mut rng = StdRng::seed_from_u64(0x5E_A1_DE_C0);
        let ctx = TestContext;
        for m in [Scalar::ZERO, Scalar::from(0x00AB_CDEFu64), -Scalar::ONE] {
            let (_sk, pk) = keypair(&mut rng);
            let (proof, c) = seal(&m, &pk, &ctx, &mut rng).unwrap();
            let bytes = proof.to_canonical_bytes();
            let decoded = Proof::from_canonical_bytes(&bytes).unwrap();
            assert_eq!(decoded.to_canonical_bytes(), bytes, "re-encode equality");
            assert!(verify(&decoded, &c, &pk, &ctx).is_ok(), "decoded verifies");
        }
    }

    #[test]
    fn from_canonical_bytes_rejects_identity_mask() {
        // E_0 occupies the first 33 bytes of the canonical layout; all-zero is
        // the canonical identity encoding. The decode boundary must reject it
        // (soundness-doc §4.4 mask gate), not defer to verify.
        let mut rng = StdRng::seed_from_u64(0x5E_A1_DE_C2);
        let (_sk, pk) = keypair(&mut rng);
        let (proof, _c) = seal(&Scalar::from(7u64), &pk, &TestContext, &mut rng).unwrap();
        let mut bytes = proof.to_canonical_bytes();
        bytes[..POINT_LEN].fill(0);
        assert!(matches!(
            Proof::from_canonical_bytes(&bytes),
            Err(Error::DegenerateInput(IDENTITY_MASK_DETAIL))
        ));
        // Precedence pin: 33 zero bytes is both truncated AND an identity E_0;
        // the mask gate fires before the length error.
        assert!(matches!(
            Proof::from_canonical_bytes(&[0u8; POINT_LEN]),
            Err(Error::DegenerateInput(IDENTITY_MASK_DETAIL))
        ));
    }

    #[test]
    fn bppp_wire_noncanonical_scalar_rejected() {
        // The first BP++ residual scalar sits after the ElGamal pairs, the
        // limb commitments, the four flights, and the twelve fold points.
        // Overwrite it with a 32-byte value >= n: the decode door must
        // reject (soundness-doc §1 — no z+n wire malleability).
        let mut rng = StdRng::seed_from_u64(0x5E_A1_0B_01);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let (proof, _c) = seal(&Scalar::from(99u64), &pk, &ctx, &mut rng).unwrap();
        let mut bytes = proof.to_canonical_bytes();
        let first_scalar = (2 * LIMB_COUNT + 2 * LIMB_COUNT + 4 + 2 * FOLD_ROUNDS) * POINT_LEN;
        bytes[first_scalar..first_scalar + SCALAR_LEN].fill(0xFF);
        assert_eq!(
            Proof::from_canonical_bytes(&bytes).err(),
            Some(Error::PointDecode("non-canonical scalar (>= n)"))
        );
    }

    #[test]
    fn bppp_wire_corrupted_flight_point_rejected() {
        // A non-canonical SEC1 tag in the BP++ flight region is rejected at
        // decode, before any verification work.
        let mut rng = StdRng::seed_from_u64(0x5E_A1_0B_02);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let (proof, _c) = seal(&Scalar::from(7u64), &pk, &ctx, &mut rng).unwrap();
        let mut bytes = proof.to_canonical_bytes();
        let flight_offset = (2 * LIMB_COUNT + 2 * LIMB_COUNT) * POINT_LEN;
        bytes[flight_offset] = 0x04; // uncompressed tag: never canonical here
        assert!(matches!(
            Proof::from_canonical_bytes(&bytes),
            Err(Error::PointDecode(_))
        ));
    }

    #[test]
    fn proof_wire_length_is_pinned() {
        // The fixed-shape layout, in bytes: 115 points (22 ElGamal + 22 limb
        // commitments + 16 BP++ + 21 carry + 34 linking) and 49 scalars
        // (5 BP++ residuals + 11 carry + 33 linking) = 5,363.
        let mut rng = StdRng::seed_from_u64(0x5E_A1_0B_04);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let (proof, _c) = seal(&Scalar::from(1u64), &pk, &ctx, &mut rng).unwrap();
        assert_eq!(
            proof.to_canonical_bytes().len(),
            115 * POINT_LEN + 49 * SCALAR_LEN
        );
        assert_eq!(proof.to_canonical_bytes().len(), 5363);
    }

    #[test]
    #[ignore = "manual measurement: cargo test -p ve-capsule --release -- --ignored measure_bppp"]
    fn measure_bppp_sizes_and_timing() {
        use std::time::Instant;
        let mut rng = StdRng::seed_from_u64(42);
        let ctx = TestContext;
        let (sk, pk) = keypair(&mut rng);
        let m = Scalar::random(&mut rng);

        let t0 = Instant::now();
        let (proof, c) = seal(&m, &pk, &ctx, &mut rng).unwrap();
        let seal_t = t0.elapsed();

        let bytes = proof.to_canonical_bytes();

        let t1 = Instant::now();
        verify(&proof, &c, &pk, &ctx).unwrap();
        let verify_t = t1.elapsed();

        let t2 = Instant::now();
        assert_eq!(open(&proof, &c, &sk).unwrap(), m);
        let open_t = t2.elapsed();

        eprintln!("proof wire bytes: {}", bytes.len());
        eprintln!("seal: {seal_t:?}, verify: {verify_t:?}, open: {open_t:?}");
    }

    #[test]
    #[ignore = "manual perf baseline; run with --release --ignored --nocapture"]
    fn verify_phase_latency() {
        use std::time::Instant;

        let mut rng = StdRng::seed_from_u64(0x5E_A1_70_01);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let (proof, c) = seal(&Scalar::from(0x00C0_FFEEu64), &pk, &ctx, &mut rng).unwrap();

        let start = Instant::now();
        reject_degenerate_recovery_key(&pk, Error::Verification).unwrap();
        let key_screen_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        let mut seen = std::collections::HashSet::with_capacity(proof.elgamal.len());
        for lhs in &proof.elgamal {
            assert_ne!(lhs.e, ProjectivePoint::IDENTITY);
            assert!(seen.insert(encode_point(&lhs.e)));
            assert!(!seen.contains(&encode_point(&(-lhs.e))));
        }
        let mask_basic_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        assert!(!has_pairwise_elgamal_mask_scalar_relation(&proof.elgamal));
        let mask_scalar_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        assert!(!has_small_coefficient_elgamal_mask_relation(&proof.elgamal));
        let mask_coeff_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        assert!(!has_elgamal_mask_public_g_offset_relation(&proof.elgamal));
        let mask_public_g_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        assert_eq!(degenerate_pedersen_commitment(&proof), None);
        let pedersen_public_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        assert_eq!(degenerate_elgamal_mask(&proof.elgamal), None);
        assert_eq!(degenerate_pedersen_commitment(&proof), None);
        let structural_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        let mut transcript = statement_transcript(
            &TranscriptInputs {
                c: &c,
                pk: &pk,
                elgamal: &proof.elgamal,
                value_commitments: &proof.value_commitments,
                complement_commitments: &proof.complement_commitments,
                carry_commitments: &proof.carry.carry_commitments,
            },
            &ctx,
        )
        .unwrap();
        let transcript_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        let statement = range_statement(
            &proof.value_commitments,
            &proof.complement_commitments,
            &proof.carry.carry_commitments,
        );
        range_circuit::verify(
            &statement,
            &proof.range,
            &mut TranscriptChallenges(&mut transcript),
        )
        .unwrap();
        let range_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        let x = sigma_challenge(transcript, &proof.carry, &proof.linking);
        let challenge_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        verify_sigma_algebra(&proof, &c, &pk, x).unwrap();
        let sigma_ms = start.elapsed().as_secs_f64() * 1e3;

        println!(
            "verify_phases key={key_screen_ms:.3} structural={structural_ms:.3} \
             mask_basic={mask_basic_ms:.3} \
             mask_scalar={mask_scalar_ms:.3} mask_coeff={mask_coeff_ms:.3} \
             mask_public_g={mask_public_g_ms:.3} pedersen_public={pedersen_public_ms:.3} \
             transcript={transcript_ms:.3} range={range_ms:.3} challenge={challenge_ms:.3} \
             sigma={sigma_ms:.3}"
        );
    }

    #[test]
    fn proof_wire_round_trips_reencode_equal() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_0B_03);
        let ctx = TestContext;
        let (_sk, pk) = keypair(&mut rng);
        let (proof, c) = seal(&Scalar::from(0x00C0_FFEEu64), &pk, &ctx, &mut rng).unwrap();
        let bytes = proof.to_canonical_bytes();
        let decoded = Proof::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_canonical_bytes(), bytes);
        assert!(verify(&decoded, &c, &pk, &ctx).is_ok());
    }

    #[test]
    fn from_canonical_bytes_rejects_truncated_and_trailing() {
        let mut rng = StdRng::seed_from_u64(0x5E_A1_DE_C1);
        let (_sk, pk) = keypair(&mut rng);
        let (proof, _c) = seal(&Scalar::from(7u64), &pk, &TestContext, &mut rng).unwrap();
        let bytes = proof.to_canonical_bytes();
        assert!(Proof::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(Proof::from_canonical_bytes(&trailing).is_err());
    }
}
