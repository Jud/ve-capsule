//! Non-destructive authorizer contribution: a [`Partial`] toward opening a
//! capsule.
//!
//! An authorizer holding the scalar `w` behind a gate (one access key, or one
//! participant's contribution) `contribute`s a `Partial` that
//! carries, for the gate it opens: the public point `W = w·G`, the per-segment
//! decryptions `{W_j = w·E_j}` (where `E_j` is segment `j`'s `ElGamal` mask), and
//! a batched DLEQ ([`crate::dleq`]) proving one `w` relates `(G, W)` and every
//! `(E_j, W_j)`. The capsule itself is never touched — a `Partial` is a
//! standalone artifact the recipient gathers and sums.
//!
//! A `Partial` releases a necessary partial decryption: the
//! recipient provably cannot finish without `{w·E_j}` (computing `x_access·E_j`
//! from `(Y_access, E_j)` is CDH-hard), yet it does **not** leak `w` (the share
//! is reusable across unlimited capsules). The DLEQ is bound to a canonical
//! `(capsule core ‖ gate ‖ context)` binding, so a `Partial` cannot be replayed
//! against a different capsule, gate, or context.
//!
//! `contribute` runs only against a capsule whose `(recipient, access_keys)` the
//! caller has verified (the [`OpeningBinding`] is derived from those), and
//! rejects a `gate` that is not in the capsule's canonical access-key set.

use crate::assembly::Proof;
use crate::bsgs::baby_table;
use crate::codec::{POINT_LEN, decode_point, encode_point};
use crate::composite::OpeningBinding;
use crate::context::Context;
use crate::dleq::BatchDleqProof;
use crate::elgamal::LimbCiphertext;
use crate::error::Error;
use crate::generators::g;
use crate::limbs::{LIMB_COUNT, LIMB_MODULUS, recompose};
use crate::signature;
use crate::transcript::push_framed;
use k256::elliptic_curve::PrimeField;
use k256::{FieldBytes, ProjectivePoint, Scalar};
use rand_core::{CryptoRng, RngCore};
use zeroize::Zeroizing;

/// Domain tag for the canonical contribution binding (capsule core ‖ gate ‖
/// context) the `Partial` DLEQ is bound to. Bump on any wire change.
const CONTRIBUTE_BINDING_DOMAIN: &[u8] = b"ve-capsule.contribute-binding.v1";

/// Wire magic for a serialized [`Partial`]; distinct from the capsule, case, and
/// aggregate magics so the decode doors stay unmistakable.
const PARTIAL_WIRE_MAGIC: &[u8] = b"ve-capsule.partial.v1";

/// `Partial` wire version. Bump only after an incompatible released format.
const PARTIAL_WIRE_VERSION: u8 = 1;

/// Byte length of a canonical scalar (`z`): 32-byte big-endian, rejected on
/// decode unless `< n` (the soundness-doc §6 / §1 canonicality obligation).
const SCALAR_LEN: usize = 32;

/// A borrowed view of a sealed capsule's **opening core** — the per-limb
/// `ElGamal` ciphertexts `(E_k, D_k)` and the commitment `C = m·G`. This is
/// everything `contribute`/`unseal` read; the seal proof `π` is not part of the
/// core, so a proof-stripped capsule presents the identical view.
#[derive(Clone, Copy)]
pub struct CapsuleRef<'a> {
    /// The per-limb `ElGamal` ciphertexts `(E_k, D_k)`, in limb order.
    pub elgamal: &'a [LimbCiphertext],
    /// The commitment `C = m·G`.
    pub c: ProjectivePoint,
}

impl CapsuleRef<'_> {
    /// Number of segments `L` (one per limb).
    const fn segment_count(&self) -> usize {
        self.elgamal.len()
    }

    /// The per-segment `ElGamal` masks `E_k = r_k·G`, in limb order — the DLEQ
    /// bases (alongside `G`) and the points the recipient strips with `w·E_k`.
    fn segment_masks(&self) -> Vec<ProjectivePoint> {
        self.elgamal.iter().map(|ct| ct.e).collect()
    }

    /// The per-segment masked points `D_k = v_k·G + r_k·Y*`, in limb order — the
    /// recipient subtracts the summed mask `x*·E_k` from each to recover `v_k·G`.
    fn segment_ciphertexts(&self) -> Vec<ProjectivePoint> {
        self.elgamal.iter().map(|ct| ct.d).collect()
    }
}

/// An authorizer's non-destructive contribution toward opening one capsule.
///
/// For one gate: the gate tag, `W = w·G`, the per-segment partials
/// `{W_j = w·E_j}`, and the batched DLEQ binding them to a single `w` and to the
/// capsule core, gate, and context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partial {
    /// The access key this partial opens (`Y_accessₖ`); the recipient buckets
    /// partials by this tag and checks `Σ W == Y_accessₖ` per gate.
    gate: ProjectivePoint,
    /// `W = w·G` — the public point of the contributed scalar.
    w_g: ProjectivePoint,
    /// `{W_j = w·E_j}` in segment order — the mask strips the recipient sums.
    masks: Vec<ProjectivePoint>,
    /// The batched DLEQ proving one `w` relates `(G, W)` and every `(E_j, W_j)`.
    dleq: BatchDleqProof,
}

/// The canonical contribution binding
/// `(domain ‖ digest(core) ‖ Y* ‖ g* ‖ gate ‖ ctx)`, every field
/// length-prefixed (normative layout: soundness doc §6 item 3).
/// Built identically by `contribute` and [`Partial::verify`], so the DLEQ
/// challenge cannot diverge; it pins the `Partial` to exactly this capsule core,
/// gate, and context. The digest covers `C` and every `(E_j, D_j)`; binding only
/// `C` plus the DLEQ bases would leave a producer partial reusable against a
/// tampered stripped core with the same masks and different ciphertext limbs.
fn contribution_binding<C: Context + ?Sized>(
    capsule: CapsuleRef<'_>,
    binding: &OpeningBinding,
    gate: &ProjectivePoint,
    ctx: &C,
) -> Result<Vec<u8>, Error> {
    let ctx_binding = ctx
        .binding_bytes()
        .map_err(|_| Error::DegenerateInput("context binding_bytes failed"))?;
    let core_digest = signature::core_digest(capsule.elgamal, &capsule.c);
    let mut out = Vec::new();
    push_framed(&mut out, CONTRIBUTE_BINDING_DOMAIN);
    push_framed(&mut out, &core_digest);
    push_framed(&mut out, &encode_point(&binding.y_star));
    push_framed(&mut out, &binding.g_star);
    push_framed(&mut out, &encode_point(gate));
    push_framed(&mut out, ctx.domain().as_bytes());
    push_framed(&mut out, &ctx_binding);
    Ok(out)
}

/// The DLEQ bases `[G, E_0, …, E_{L-1}]` for a capsule: the generator followed
/// by every segment mask. The images a `Partial` carries are `w` times these.
fn dleq_bases(capsule: CapsuleRef<'_>) -> Vec<ProjectivePoint> {
    let mut bases = Vec::with_capacity(capsule.segment_count() + 1);
    bases.push(g());
    bases.extend(capsule.segment_masks());
    bases
}

/// Reject a `gate` that is not in the capsule's canonical access-key set — an
/// authorizer must not contribute toward a gate the capsule is not bound to.
fn require_listed_gate(binding: &OpeningBinding, gate: &ProjectivePoint) -> Result<(), Error> {
    if binding.gates.iter().any(|listed| listed == gate) {
        Ok(())
    } else {
        Err(Error::DegenerateInput(
            "gate is not in the capsule's access-key set",
        ))
    }
}

/// Contribute a non-destructive [`Partial`] for `gate`, using scalar `w`. The
/// capsule is identified by its opening core (the masks `{E_k}` and `C`);
/// `binding` is the [`OpeningBinding`] derived from the verified
/// `(recipient, access_keys)`.
///
/// # Errors
///
/// [`Error::DegenerateInput`] if `gate` is not in the access-key set or the
/// context binding fails; otherwise a DLEQ construction error.
pub fn contribute<C: Context + ?Sized, R: RngCore + CryptoRng>(
    capsule: CapsuleRef<'_>,
    binding: &OpeningBinding,
    gate: &ProjectivePoint,
    w: &Scalar,
    ctx: &C,
    rng: &mut R,
) -> Result<Partial, Error> {
    require_listed_gate(binding, gate)?;
    let bases = dleq_bases(capsule);
    let bind = contribution_binding(capsule, binding, gate, ctx)?;
    let (images, dleq) = BatchDleqProof::prove(w, &bases, &bind, rng)?;
    // images[0] = w·G; images[1..] = w·E_j, in segment order.
    let w_g = images[0];
    let masks = images[1..].to_vec();
    Ok(Partial {
        gate: *gate,
        w_g,
        masks,
        dleq,
    })
}

impl Partial {
    /// The gate this partial opens (`Y_accessₖ`).
    #[must_use]
    pub const fn gate(&self) -> ProjectivePoint {
        self.gate
    }

    /// `W = w·G` — summed per gate to check `Σ W == Y_accessₖ`.
    #[must_use]
    pub const fn w_g(&self) -> ProjectivePoint {
        self.w_g
    }

    /// The per-segment masks `{W_j = w·E_j}`.
    #[must_use]
    pub fn masks(&self) -> &[ProjectivePoint] {
        &self.masks
    }

    /// Canonical wire bytes:
    /// `magic ‖ version ‖ gate ‖ W ‖ {W_j}×L ‖ {A_i}×(L+1) ‖ z`.
    ///
    /// Fixed-length for the frozen params (`L = LIMB_COUNT`): every point is a
    /// 33-byte canonical SEC1 encoding, `z` is 32 big-endian bytes. Deterministic
    /// (the decoder enforces re-encode equality). A `Partial` an authorizer
    /// `contribute`s for a real (frozen-param) capsule always has exactly `L`
    /// masks and `L+1` announcements; the encoder rejects a malformed `Partial`
    /// (one built from a non-`LIMB_COUNT` `CapsuleRef`) rather than emit bytes
    /// the fixed-layout decoder could not round-trip.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if the mask count is not `LIMB_COUNT` or the
    /// announcement count is not `LIMB_COUNT + 1`.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, Error> {
        let announcements = self.dleq.announcements();
        if self.masks.len() != LIMB_COUNT || announcements.len() != LIMB_COUNT + 1 {
            return Err(Error::DegenerateInput(
                "partial has a non-canonical segment count",
            ));
        }
        let mut out = Vec::with_capacity(
            PARTIAL_WIRE_MAGIC.len()
                + 1
                + POINT_LEN * (2 + LIMB_COUNT + LIMB_COUNT + 1)
                + SCALAR_LEN,
        );
        out.extend_from_slice(PARTIAL_WIRE_MAGIC);
        out.push(PARTIAL_WIRE_VERSION);
        out.extend_from_slice(&encode_point(&self.gate));
        out.extend_from_slice(&encode_point(&self.w_g));
        for mask in &self.masks {
            out.extend_from_slice(&encode_point(mask));
        }
        for announcement in announcements {
            out.extend_from_slice(&encode_point(announcement));
        }
        out.extend_from_slice(&self.dleq.response().to_bytes());
        Ok(out)
    }

    /// Parse a `Partial` from canonical wire bytes — the only deserialization
    /// door. Validates framing, decodes `gate`, `W`, the `L` masks, the `L+1`
    /// announcements (each via the strict `decode_point`), and the response `z`
    /// (rejected unless `< n`), then enforces re-encode equality. This is the
    /// soundness-doc §6 obligation: enforce the §1 canonicality rules (`z < n`,
    /// strict point decode) *before* verification, exactly as the seal proof's
    /// decoder does.
    ///
    /// A decoded `Partial` is **unauthenticated** — the decoder asserts framing
    /// and canonicality only. Authenticity comes from
    /// [`Partial::verify`] against the capsule core, binding, and context, which
    /// every opening path (`open_core`) runs before a partial is summed, so a
    /// forged or tampered decoded partial fails verification and is skipped.
    ///
    /// # Errors
    ///
    /// [`Error::PointDecode`] on bad magic/version, a malformed/short/trailing
    /// field, a non-canonical point or scalar (`z ≥ n`), or non-canonical framing.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let rest = bytes
            .strip_prefix(PARTIAL_WIRE_MAGIC)
            .ok_or(Error::PointDecode("partial: bad magic"))?;
        let (&version, mut rest) = rest
            .split_first()
            .ok_or(Error::PointDecode("partial: truncated header"))?;
        if version != PARTIAL_WIRE_VERSION {
            return Err(Error::PointDecode("partial: unsupported version"));
        }
        let take_point = |rest: &mut &[u8]| -> Result<ProjectivePoint, Error> {
            let (head, tail) = rest
                .split_at_checked(POINT_LEN)
                .ok_or(Error::PointDecode("partial: truncated point"))?;
            *rest = tail;
            decode_point(head)
        };
        let gate = take_point(&mut rest)?;
        let w_g = take_point(&mut rest)?;
        let mut masks = Vec::with_capacity(LIMB_COUNT);
        for _ in 0..LIMB_COUNT {
            masks.push(take_point(&mut rest)?);
        }
        let mut announcements = Vec::with_capacity(LIMB_COUNT + 1);
        for _ in 0..=LIMB_COUNT {
            announcements.push(take_point(&mut rest)?);
        }
        let (z_bytes, rest) = rest
            .split_at_checked(SCALAR_LEN)
            .ok_or(Error::PointDecode("partial: truncated response"))?;
        let mut z_repr = FieldBytes::default();
        z_repr.copy_from_slice(z_bytes);
        let z = Option::<Scalar>::from(Scalar::from_repr(z_repr)).ok_or(Error::PointDecode(
            "partial: non-canonical response (z >= n)",
        ))?;
        if !rest.is_empty() {
            return Err(Error::PointDecode("partial: trailing bytes"));
        }
        let partial = Self {
            gate,
            w_g,
            masks,
            dleq: BatchDleqProof::from_parts(announcements, z),
        };
        // Belt-and-braces: both failure paths here are unreachable today (the
        // loops above built exactly the counts the encoder requires, and every
        // accepted decode re-encodes to its input bytes), but a future refactor
        // must surface as the documented PointDecode, not leak DegenerateInput.
        let reencoded = partial
            .to_canonical_bytes()
            .map_err(|_| Error::PointDecode("partial: non-canonical encoding"))?;
        if reencoded != bytes {
            return Err(Error::PointDecode("partial: non-canonical encoding"));
        }
        Ok(partial)
    }

    /// Verify this partial against the capsule core `(C, E_j, D_j)`, its
    /// `binding`, and the context: the gate is listed, the segment count matches,
    /// and the DLEQ proves a single `w` relates `(G, W)` and every `(E_j, W_j)`.
    ///
    /// # Errors
    ///
    /// [`Error::DegenerateInput`] if `gate` is not listed or the context binding
    /// fails; [`Error::Verification`] on a segment-count mismatch or a failed
    /// DLEQ.
    pub fn verify<C: Context + ?Sized>(
        &self,
        capsule: CapsuleRef<'_>,
        binding: &OpeningBinding,
        ctx: &C,
    ) -> Result<(), Error> {
        require_listed_gate(binding, &self.gate)?;
        if self.masks.len() != capsule.segment_count() {
            return Err(Error::Verification("partial segment-count mismatch"));
        }
        let bases = dleq_bases(capsule);
        let mut images = Vec::with_capacity(self.masks.len() + 1);
        images.push(self.w_g);
        images.extend_from_slice(&self.masks);
        let bind = contribution_binding(capsule, binding, &self.gate, ctx)?;
        self.dleq.verify(&bases, &images, &bind)
    }
}

/// The recipient's terminal op: open the capsule from its own secret plus the
/// authorizers' `partials`, recovering the sealed scalar `m`.
///
/// Steps: verify and deduplicate every partial; enforce **strict AND** — each
/// expected gate's bucket must satisfy `Σ W == Y_accessₖ`; then per segment
/// subtract the total access mask `Σ_partials W_j`
/// and the recipient's own `x_rcpt·E_j` from `D_j` to get `v_j·G`, BSGS to
/// `v_j`, recompose, and recheck `m·G == C`. A gate one short (or a wrong
/// recipient secret) yields no accepted secret — the bucket check or final
/// opening check fails closed.
///
/// A wrong or irrelevant `Partial` is harmless — it fails verification and is
/// skipped, never summed. An exactly re-submitted `Partial` (the same artifact)
/// is an idempotent no-op (deduplicated by full equality); two *distinct*
/// artifacts for one participant (a re-run of `contribute`, fresh DLEQ entropy)
/// both count and fail-close the bucket — per-participant dedup needs a
/// participant id, which lives in the caller's orchestration layer, not this ID-less
/// primitive.
///
/// `unseal` is **self-verifying**: it re-runs the capsule's seal proof `π` under
/// `ctx` via [`crate::composite::verify_bound`] before decrypting, so opening is
/// context-bound even for an ungated capsule with no partials.
///
/// # Errors
///
/// [`Error::Verification`] if the capsule's `π` fails under `ctx`, a gate bucket
/// is not qualifying, or the opening does not reconstruct a scalar whose
/// commitment is `C`.
pub fn unseal<C: Context + ?Sized>(
    proof: &Proof,
    c: ProjectivePoint,
    binding: &OpeningBinding,
    recipient_secret: &Scalar,
    partials: &[Partial],
    ctx: &C,
) -> Result<Scalar, Error> {
    // Self-verify: re-run the capsule's seal proof under ctx (context-binds the
    // open even when ungated), then open. This is the one open path that needs
    // the full `Proof` (to re-run π); a verified-capsule token calls
    // `unseal_verified` directly off the core view to skip this second π verify
    // on the latency budget — the proof verify is the dominant local cost (~150 ms).
    crate::composite::verify_bound(proof, &c, binding, ctx)?;
    unseal_verified(
        CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        },
        binding,
        recipient_secret,
        partials,
        ctx,
    )
}

/// The opening core **without** re-verifying the capsule's seal proof. Callable
/// only once the caller has already confirmed `π` under `ctx` (a verified-capsule
/// token); [`unseal`] is the self-verifying wrapper. Still verifies every
/// `Partial`'s DLEQ, enforces strict-AND, strips, BSGS, and rechecks `m·G == C`.
///
/// # Errors
///
/// As [`unseal`], minus the seal-proof recheck (the caller owns that).
pub fn unseal_verified<C: Context + ?Sized>(
    capsule: CapsuleRef<'_>,
    binding: &OpeningBinding,
    recipient_secret: &Scalar,
    partials: &[Partial],
    ctx: &C,
) -> Result<Scalar, Error> {
    open_core(
        capsule,
        binding,
        recipient_secret,
        partials,
        ctx,
        LIMB_MODULUS,
    )
}

/// The single-capsule opening front door: verify+dedup partials, enforce
/// strict-AND gate buckets, sum the weighted access masks, then delegate to
/// [`open_core_with_access_masks`] to strip each segment, BSGS-recover its limb
/// **under `max_limb_exclusive`**, recompose, and recheck `m·G == C`.
/// [`unseal_verified`] calls this with the single-capsule bound `2^ℓ`; the wider
/// summed-limb bound `H·2^ℓ` (a summed limb `Σₕ vₖ⁽ʰ⁾` lands in `[0, H·2^ℓ)`) is
/// used by the stripped-case path, which calls [`open_core_with_access_masks`]
/// directly — both share that one recover kernel.
///
/// # Errors
///
/// As [`unseal_verified`].
pub fn open_core<C: Context + ?Sized>(
    capsule: CapsuleRef<'_>,
    binding: &OpeningBinding,
    recipient_secret: &Scalar,
    partials: &[Partial],
    ctx: &C,
    max_limb_exclusive: u64,
) -> Result<Scalar, Error> {
    if capsule.segment_count() != LIMB_COUNT {
        return Err(Error::Verification("proof segment-count mismatch"));
    }

    // Verify each partial; drop ones that fail (a wrong/irrelevant partial is
    // harmless), then deduplicate by full structural equality — an exactly
    // re-submitted artifact is a no-op. We do NOT dedup by `(gate, W)`: two
    // distinct participants could (astronomically rarely) share `W`, and both
    // must count; collapsing them would drop a valid share.
    let mut verified: Vec<&Partial> = Vec::new();
    for partial in partials {
        if partial.verify(capsule, binding, ctx).is_err() {
            continue;
        }
        if verified.contains(&partial) {
            continue;
        }
        verified.push(partial);
    }

    // Strict AND: every expected gate's bucket sums to its key. A flat aggregate
    // would let a cross-group surplus cover another group's shortfall.
    for gate in &binding.gates {
        let mut w_sum = ProjectivePoint::IDENTITY;
        for partial in &verified {
            if partial.gate() == *gate {
                w_sum += partial.w_g();
            }
        }
        if w_sum != *gate {
            return Err(Error::Verification(
                "access gate not qualifying: Σ W ≠ Y_access",
            ));
        }
    }

    let mut weighted_access_masks = [ProjectivePoint::IDENTITY; LIMB_COUNT];
    // Strip and recover each limb: D_j − (Σ_partials W_j + x_rcpt·E_j) = v_j·G.
    for (j, access_mask) in weighted_access_masks.iter_mut().enumerate() {
        for partial in &verified {
            let weight = binding
                .gate_weight(&partial.gate())
                .ok_or(Error::Verification("partial gate weight missing"))?;
            *access_mask += partial.masks()[j] * weight;
        }
    }
    open_core_with_access_masks(
        capsule,
        binding,
        recipient_secret,
        &weighted_access_masks,
        max_limb_exclusive,
    )
}

/// Open a core after the caller has already verified any authorizer partials,
/// enforced the gate buckets, and supplied the weighted access mask per segment.
///
/// This is used by [`open_core`] directly and by stripped case opening, where
/// per-piece partials are verified against their original cores and then summed
/// before one aggregate opening. That aggregate path deliberately avoids
/// accepting/rejecting individual shifted piece limbs before the final aggregate
/// commitment check.
pub fn open_core_with_access_masks(
    capsule: CapsuleRef<'_>,
    binding: &OpeningBinding,
    recipient_secret: &Scalar,
    weighted_access_masks: &[ProjectivePoint; LIMB_COUNT],
    max_limb_exclusive: u64,
) -> Result<Scalar, Error> {
    if capsule.segment_count() != LIMB_COUNT {
        return Err(Error::Verification("proof segment-count mismatch"));
    }

    // Strip and recover each limb: D_j − (Σ_partials W_j + x_rcpt·E_j) = v_j·G.
    // The recovered limbs are the plaintext; zeroize them on drop. Do not return
    // a distinct per-limb BSGS failure: on unauthenticated stripped/aggregate
    // cores, that would let attacker-chosen D_j shifts reveal whether the shifted
    // plaintext limb stayed inside the search interval before the final C check.
    let segment_masks = capsule.segment_masks();
    let segment_ciphertexts = capsule.segment_ciphertexts();
    let table = baby_table();
    let mut limbs = Zeroizing::new([0u32; LIMB_COUNT]);
    let mut limb_missing = false;
    for (j, limb) in limbs.iter_mut().enumerate() {
        let total_mask = weighted_access_masks[j]
            + segment_masks[j] * (*recipient_secret * binding.recipient_weight);
        let limb_point = segment_ciphertexts[j] - total_mask;
        match table.recover_bounded_complete(&limb_point, max_limb_exclusive) {
            Some(recovered) => *limb = recovered,
            None => limb_missing = true,
        }
    }

    let m = recompose(&limbs);
    if limb_missing || g() * m != capsule.c {
        return Err(Error::Verification("opening failed"));
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::manual_let_else,
        clippy::panic
    )]

    use super::*;
    use crate::composite::{opening_binding, seal};
    use k256::elliptic_curve::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::borrow::Cow;

    struct TestCtx;
    impl Context for TestCtx {
        fn domain(&self) -> &'static str {
            "ve-capsule.opening-test"
        }
        fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
            Ok(Cow::Borrowed(b"opening-binding"))
        }
    }

    fn keypair(rng: &mut StdRng) -> (Scalar, ProjectivePoint) {
        let x = Scalar::random(rng);
        (x, g() * x)
    }

    /// Seal a one-gate capsule and return everything the opening layer needs.
    fn seal_one_gate(
        rng: &mut StdRng,
    ) -> (
        Proof,
        ProjectivePoint,
        ProjectivePoint,
        Scalar,
        OpeningBinding,
    ) {
        let (_x_r, recipient) = keypair(rng);
        let (x_a, access) = keypair(rng);
        let m = Scalar::from(0x00AB_CDEFu64);
        let (proof, c) = seal(&m, &recipient, &[access], &TestCtx, rng).unwrap();
        let binding = opening_binding(&recipient, &[access]).unwrap();
        (proof, c, access, x_a, binding)
    }

    #[test]
    fn contribute_produces_verifiable_partial() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_0001);
        let (proof, c, access, x_a, binding) = seal_one_gate(&mut rng);
        let partial = contribute(
            CapsuleRef {
                elgamal: proof.elgamal(),
                c,
            },
            &binding,
            &access,
            &x_a,
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        // W = w·G, and for a single authorizer w = x_a so W == the gate itself.
        assert_eq!(partial.w_g(), access);
        // W_j = w·E_j for every segment.
        for (mask, ct) in partial.masks().iter().zip(proof.elgamal()) {
            assert_eq!(*mask, ct.e * x_a);
        }
        assert!(
            partial
                .verify(
                    CapsuleRef {
                        elgamal: proof.elgamal(),
                        c,
                    },
                    &binding,
                    &TestCtx
                )
                .is_ok()
        );
    }

    #[test]
    fn rejects_gate_not_in_set() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_0002);
        let (proof, c, _access, x_a, binding) = seal_one_gate(&mut rng);
        let (_x_o, outsider) = keypair(&mut rng);
        assert!(matches!(
            contribute(
                CapsuleRef {
                    elgamal: proof.elgamal(),
                    c,
                },
                &binding,
                &outsider,
                &x_a,
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput(
                "gate is not in the capsule's access-key set"
            ))
        ));
    }

    #[test]
    fn rejects_tampered_mask() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_0003);
        let (proof, c, access, x_a, binding) = seal_one_gate(&mut rng);
        let mut partial = contribute(
            CapsuleRef {
                elgamal: proof.elgamal(),
                c,
            },
            &binding,
            &access,
            &x_a,
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        partial.masks[0] += g();
        assert!(
            partial
                .verify(
                    CapsuleRef {
                        elgamal: proof.elgamal(),
                        c,
                    },
                    &binding,
                    &TestCtx
                )
                .is_err()
        );
    }

    #[test]
    fn rejects_replay_against_different_context() {
        struct OtherCtx;
        impl Context for OtherCtx {
            fn domain(&self) -> &'static str {
                "ve-capsule.opening-test"
            }
            fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
                Ok(Cow::Borrowed(b"a-different-binding"))
            }
        }
        let mut rng = StdRng::seed_from_u64(0x0BEA_0004);
        let (proof, c, access, x_a, binding) = seal_one_gate(&mut rng);
        let partial = contribute(
            CapsuleRef {
                elgamal: proof.elgamal(),
                c,
            },
            &binding,
            &access,
            &x_a,
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        // Same capsule, same gate, different context ⇒ different binding ⇒ reject.
        assert!(
            partial
                .verify(
                    CapsuleRef {
                        elgamal: proof.elgamal(),
                        c,
                    },
                    &binding,
                    &OtherCtx
                )
                .is_err()
        );
    }

    #[test]
    fn rejects_replay_against_different_capsule() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_0005);
        let (proof, c, access, x_a, binding) = seal_one_gate(&mut rng);
        let partial = contribute(
            CapsuleRef {
                elgamal: proof.elgamal(),
                c,
            },
            &binding,
            &access,
            &x_a,
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        // A second capsule to the same gate has fresh E_j and a fresh C.
        let (proof2, c2) = seal(
            &Scalar::from(7u64),
            &binding.recipient,
            &[access],
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        let binding2 = OpeningBinding {
            y_star: binding.y_star,
            recipient: binding.recipient,
            recipient_weight: binding.recipient_weight,
            g_star: binding.g_star,
            gates: binding.gates.clone(),
            gate_weights: binding.gate_weights.clone(),
        };
        assert!(
            partial
                .verify(
                    CapsuleRef {
                        elgamal: proof2.elgamal(),
                        c: c2,
                    },
                    &binding2,
                    &TestCtx
                )
                .is_err()
        );
    }

    #[test]
    fn rejects_replay_against_tampered_ciphertext_limb() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_0006);
        let (proof, c, access, x_a, binding) = seal_one_gate(&mut rng);
        let partial = contribute(
            CapsuleRef {
                elgamal: proof.elgamal(),
                c,
            },
            &binding,
            &access,
            &x_a,
            &TestCtx,
            &mut rng,
        )
        .unwrap();

        let mut tampered_elgamal = proof.elgamal().to_vec();
        tampered_elgamal[0].d += g();

        assert!(
            partial
                .verify(
                    CapsuleRef {
                        elgamal: &tampered_elgamal,
                        c,
                    },
                    &binding,
                    &TestCtx
                )
                .is_err(),
            "a producer partial must be bound to the full opening core, not only C and E"
        );
    }

    #[test]
    fn tampered_ciphertext_limb_failures_are_not_distinguishable_by_error() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_0007);
        let (recipient_secret, recipient) = keypair(&mut rng);
        let m = Scalar::from(0x12_3456u64);
        let (proof, c) = seal(&m, &recipient, &[], &TestCtx, &mut rng).unwrap();
        let binding = opening_binding(&recipient, &[]).unwrap();
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        assert_eq!(
            unseal_verified(capsule, &binding, &recipient_secret, &[], &TestCtx).unwrap(),
            m
        );

        let limb0 = crate::limbs::decompose(&m)[0];
        assert!(u64::from(limb0) + 1 < LIMB_MODULUS);

        let mut still_in_range = proof.elgamal().to_vec();
        still_in_range[0].d += g();
        let still_in_range_err = match unseal_verified(
            CapsuleRef {
                elgamal: &still_in_range,
                c,
            },
            &binding,
            &recipient_secret,
            &[],
            &TestCtx,
        ) {
            Ok(_) => panic!("in-range tamper must fail the final commitment check"),
            Err(err) => err,
        };

        let mut bsgs_miss = proof.elgamal().to_vec();
        let outside_delta = LIMB_MODULUS - u64::from(limb0);
        bsgs_miss[0].d += g() * Scalar::from(outside_delta);
        let bsgs_miss_err = match unseal_verified(
            CapsuleRef {
                elgamal: &bsgs_miss,
                c,
            },
            &binding,
            &recipient_secret,
            &[],
            &TestCtx,
        ) {
            Ok(_) => panic!("out-of-range tamper must fail opening"),
            Err(err) => err,
        };

        assert_eq!(still_in_range_err.to_string(), bsgs_miss_err.to_string());
        assert!(
            still_in_range_err.to_string().contains("opening failed"),
            "{still_in_range_err}"
        );
    }

    // ── Partial wire codec: boundary (de)serialization ──────────────────────

    /// The byte offset of the first point region (`gate`) in canonical bytes.
    const POINT_REGION_START: usize = PARTIAL_WIRE_MAGIC.len() + 1;

    #[test]
    fn partial_wire_round_trip_and_framing() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_2001);
        let (proof, c, access, x_a, binding) = seal_one_gate(&mut rng);
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        let partial = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();

        let bytes = partial.to_canonical_bytes().unwrap();
        // magic ‖ ver(1) ‖ gate(33) ‖ W(33) ‖ L masks ‖ (L+1) announcements ‖ z(32).
        assert_eq!(
            bytes.len(),
            PARTIAL_WIRE_MAGIC.len()
                + 1
                + POINT_LEN * (2 + LIMB_COUNT + LIMB_COUNT + 1)
                + SCALAR_LEN
        );

        let decoded = Partial::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded, partial, "round-trips to the same Partial");
        assert_eq!(
            decoded.to_canonical_bytes().unwrap(),
            bytes,
            "re-encode equality"
        );
        // A decoded partial still verifies against the capsule it was built for.
        assert!(decoded.verify(capsule, &binding, &TestCtx).is_ok());

        // bad magic / bad version / truncated / trailing.
        assert!(Partial::from_canonical_bytes(b"not-a-partial").is_err());
        let mut bad_ver = bytes.clone();
        bad_ver[PARTIAL_WIRE_MAGIC.len()] = PARTIAL_WIRE_VERSION + 1;
        assert!(Partial::from_canonical_bytes(&bad_ver).is_err());
        assert!(Partial::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(Partial::from_canonical_bytes(&trailing).is_err());
    }

    /// The pinned canonical bytes of the deterministic KAT partial (hex).
    const PARTIAL_WIRE_KAT: &str = concat!(
        "76652d63617073756c652e7061727469616c2e76310103072a5c3e9bd60fe1151d2ada56e6753b9f4d1efa1a765eb5b45aec",
        "2cfa9a48ab03072a5c3e9bd60fe1151d2ada56e6753b9f4d1efa1a765eb5b45aec2cfa9a48ab02cd960f4e10a09741f822a1",
        "a92f5c0f1d8cf1993333162c9819a315d7b77cb044038269aad329ea43ad7ee03fe5ec2bb5d990fb0431eb07956ed078aba6",
        "f9a62db50200f3dfcb742405ec9dfdd0699fa02c058e9819460f76b29f36aa9606ce207d6902aba7fdd16a7bc3482cd501c5",
        "58ba59ba1d48ed1def4fc700523380b26d7fa9830393021ccbaf7d7c317a80cce6e758b59a48c5457fa993bebc724bae809d",
        "fc882c02a0d26d190387355e4790f4869cfe0cb7df8cffcc9afe5fe642933caba4cc3cbc03ac063926dd7bd7c816c343f545",
        "84bd22418fdf67a9fae88246d05de2a6ed86e602a6013fa24d86aca8da6fa58a2ec7c6b93d7b31706f3e85123be318603523",
        "ca9202e405193d53127f6cecc9146074a6136d56457f6b3b03940aa9b80899a48fe7c90302cdc1caf491fefe6d54968990f7",
        "e39986a2eaff6c2f1ed47bf2c31c92926f0202417dabe7efba48836ffffa77070314bbd34e33a4e607abfa6686bcfa8be472",
        "3a02a291b36fa3ca9236355abfd5244381726a1c50cdaf91e5e86dec86f7ffea75f00265373276f2d838ac9998dfa6657b6e",
        "4e1f2d1ce427b1681535285defaea37fc902baef0ee6b4d9ee7d97d79a8213da03117a70f320f412b3024b957dcbd2fdb650",
        "034fff2019b79694ed8569759308fb5ffab40c0c842e2f9857b2bb1fdb7182ee5402ff0a06a110910ca8cb4f77b37908e310",
        "613b4c8d7314f8c776a5683761df8586021de07bf6316c8664026e5874c3533098fe820efe27651ffaf02e44e07362de2102",
        "a150aa189b6035ba1c1a1c891519da563855cabc40a9a6b4e795560aefd013fa0350db41ca25fef07560cd69bb819f0daa30",
        "695fdd8884ae00b70901a794a2d7c4035dd72d3676ee8a4937fc1cec36a074eb29024002d83996a2919be7ad8009319d03f2",
        "7a437cffe098b152483eee0a16ce383e7d7214cd3928e42cdef0235ecd43fa03970a2d3aed75ea60b6c9ac8605aca1a74123",
        "7f2e37228f3799af419160b590530277627ff60ed0c2a8242cabbd24694b7f439b3dc97c9554e4a78147081b7192825ab6f5",
        "fe8d378e1adcc11f26a6366253efc5a273395f92d3e609a074a1829efc",
    );

    /// Known-answer test for the `Partial` wire format and DLEQ transcript: a
    /// fully deterministic partial must serialize to exactly these bytes.
    /// Round-trip tests cannot catch the encoder and decoder drifting *together*
    /// (a field reorder, a magic typo), and ordinary proof verification cannot
    /// catch an intentional challenge-binding change. This vector pins both the
    /// released layout and proof transcript bytes.
    #[test]
    fn partial_wire_known_answer() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_CA70);
        let (proof, c, access, x_a, binding) = seal_one_gate(&mut rng);
        let partial = contribute(
            CapsuleRef {
                elgamal: proof.elgamal(),
                c,
            },
            &binding,
            &access,
            &x_a,
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        let bytes = partial.to_canonical_bytes().unwrap();
        let hex = bytes.iter().fold(String::new(), |mut s, b| {
            use std::fmt::Write;
            let _ = write!(s, "{b:02x}");
            s
        });
        assert_eq!(hex, PARTIAL_WIRE_KAT, "Partial wire layout drifted");
        // And the pinned bytes still decode + round-trip.
        let decoded = Partial::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded, partial);
    }

    #[test]
    fn partial_wire_decoded_partial_unseals() {
        // The serialized shape: the authorizer's partial crosses the wire, and
        // the recipient opens with the decoded artifact.
        let mut rng = StdRng::seed_from_u64(0x0BEA_2002);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, access) = keypair(&mut rng);
        let m = Scalar::from(0x00DE_F123u64);
        let (proof, c) = seal(&m, &recipient, &[access], &TestCtx, &mut rng).unwrap();
        let binding = opening_binding(&recipient, &[access]).unwrap();
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        let partial = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();
        let wire = partial.to_canonical_bytes().unwrap();
        let decoded = Partial::from_canonical_bytes(&wire).unwrap();
        let recovered = unseal(&proof, c, &binding, &x_r, &[decoded], &TestCtx).unwrap();
        assert_eq!(recovered, m);
    }

    #[test]
    fn partial_wire_rejects_noncanonical_scalar() {
        // z ≥ n must be rejected (the §6 / §1 canonicality gate against z+n
        // malleability), matching the seal proof's scalar decoder.
        let mut rng = StdRng::seed_from_u64(0x0BEA_2003);
        let (proof, c, access, x_a, binding) = seal_one_gate(&mut rng);
        let partial = contribute(
            CapsuleRef {
                elgamal: proof.elgamal(),
                c,
            },
            &binding,
            &access,
            &x_a,
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        let mut bytes = partial.to_canonical_bytes().unwrap();
        // The response z is the trailing 32 bytes. All-0xFF is ≥ n ⇒ non-canonical.
        let z_start = bytes.len() - SCALAR_LEN;
        for b in &mut bytes[z_start..] {
            *b = 0xFF;
        }
        assert!(matches!(
            Partial::from_canonical_bytes(&bytes),
            Err(Error::PointDecode(_))
        ));
    }

    #[test]
    fn partial_wire_routes_every_point_through_strict_decode() {
        // The §6 decoder contract is strict point decode on every point. Corrupt
        // the SEC1 tag byte of one point in each region — gate, W, a mask, an
        // announcement — and confirm each is rejected, proving none bypass
        // codec::decode_point.
        let mut rng = StdRng::seed_from_u64(0x0BEA_2004);
        let (proof, c, access, x_a, binding) = seal_one_gate(&mut rng);
        let partial = contribute(
            CapsuleRef {
                elgamal: proof.elgamal(),
                c,
            },
            &binding,
            &access,
            &x_a,
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        let bytes = partial.to_canonical_bytes().unwrap();
        let gate = POINT_REGION_START;
        let w_g = gate + POINT_LEN;
        let first_mask = w_g + POINT_LEN;
        let first_announcement = first_mask + POINT_LEN * LIMB_COUNT;
        for tag_offset in [gate, w_g, first_mask, first_announcement] {
            let mut corrupt = bytes.clone();
            // 0x05 is not a valid SEC1 tag (only 0x02/0x03 for points here).
            corrupt[tag_offset] = 0x05;
            assert!(
                Partial::from_canonical_bytes(&corrupt).is_err(),
                "a malformed point at offset {tag_offset} must reject"
            );
        }
    }

    #[test]
    fn partial_wire_decoded_then_tampered_fails_verify() {
        // The decoder does not authenticate: re-point one mask to a different valid
        // point, re-encode, decode (canonical OK), and confirm verify rejects it —
        // verify, not the codec, is the soundness gate.
        let mut rng = StdRng::seed_from_u64(0x0BEA_2005);
        let (proof, c, access, x_a, binding) = seal_one_gate(&mut rng);
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        let mut partial = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();
        partial.masks[0] += g();
        let bytes = partial.to_canonical_bytes().unwrap();
        let decoded = Partial::from_canonical_bytes(&bytes).unwrap();
        assert!(
            decoded.verify(capsule, &binding, &TestCtx).is_err(),
            "a canonically-encoded but tampered partial fails DLEQ verify"
        );
    }

    #[test]
    fn partial_wire_encoder_rejects_malformed_segment_count() {
        // The fixed-layout decoder reads exactly LIMB_COUNT masks; the encoder must
        // refuse a Partial whose counts do not match (constructible via a non-frozen
        // CapsuleRef) rather than emit bytes it cannot round-trip.
        let mut rng = StdRng::seed_from_u64(0x0BEA_2006);
        let (proof, c, access, x_a, binding) = seal_one_gate(&mut rng);
        let good = contribute(
            CapsuleRef {
                elgamal: proof.elgamal(),
                c,
            },
            &binding,
            &access,
            &x_a,
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        // Wrong mask count.
        let mut short_masks = good.clone();
        short_masks.masks.pop();
        assert!(matches!(
            short_masks.to_canonical_bytes(),
            Err(Error::DegenerateInput(_))
        ));
        // Wrong announcement count (the other guarded branch).
        let mut announcements = good.dleq.announcements().to_vec();
        announcements.pop();
        let short_announcements = Partial {
            gate: good.gate,
            w_g: good.w_g,
            masks: good.masks.clone(),
            dleq: BatchDleqProof::from_parts(announcements, *good.dleq.response()),
        };
        assert!(matches!(
            short_announcements.to_canonical_bytes(),
            Err(Error::DegenerateInput(_))
        ));
    }

    // ── unseal: the recipient's terminal op (the round-trip milestone) ──────

    #[test]
    fn tracer_round_trip_single_gate() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_1001);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, access) = keypair(&mut rng);
        let m = Scalar::from(0x00CA_FE12u64);
        let (proof, c) = seal(&m, &recipient, &[access], &TestCtx, &mut rng).unwrap();
        let binding = opening_binding(&recipient, &[access]).unwrap();
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        let partial = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();
        let recovered = unseal(&proof, c, &binding, &x_r, &[partial], &TestCtx).unwrap();
        assert_eq!(recovered, m);
    }

    #[test]
    fn and_round_trip_two_gates() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_1002);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, a) = keypair(&mut rng);
        let (x_b, b) = keypair(&mut rng);
        let m = Scalar::from(0xDEAD_BEEFu64);
        let (proof, c) = seal(&m, &recipient, &[a, b], &TestCtx, &mut rng).unwrap();
        let binding = opening_binding(&recipient, &[a, b]).unwrap();
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        let pa = contribute(capsule, &binding, &a, &x_a, &TestCtx, &mut rng).unwrap();
        let pb = contribute(capsule, &binding, &b, &x_b, &TestCtx, &mut rng).unwrap();
        let recovered = unseal(&proof, c, &binding, &x_r, &[pa, pb], &TestCtx).unwrap();
        assert_eq!(recovered, m);
    }

    #[test]
    fn ungated_round_trip_recipient_only() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_1003);
        let (x_r, recipient) = keypair(&mut rng);
        let m = Scalar::from(0x1234u64);
        let (proof, c) = seal(&m, &recipient, &[], &TestCtx, &mut rng).unwrap();
        let binding = opening_binding(&recipient, &[]).unwrap();
        let recovered = unseal(&proof, c, &binding, &x_r, &[], &TestCtx).unwrap();
        assert_eq!(recovered, m);
    }

    #[test]
    fn sub_threshold_gate_not_qualifying() {
        // Two gates, but only one bucket filled: the other gate's Σ W ≠ Y_access.
        let mut rng = StdRng::seed_from_u64(0x0BEA_1004);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, a) = keypair(&mut rng);
        let (_x_b, b) = keypair(&mut rng);
        let (proof, c) =
            seal(&Scalar::from(9u64), &recipient, &[a, b], &TestCtx, &mut rng).unwrap();
        let binding = opening_binding(&recipient, &[a, b]).unwrap();
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        let pa = contribute(capsule, &binding, &a, &x_a, &TestCtx, &mut rng).unwrap();
        assert!(matches!(
            unseal(&proof, c, &binding, &x_r, &[pa], &TestCtx),
            Err(Error::Verification(
                "access gate not qualifying: Σ W ≠ Y_access"
            ))
        ));
    }

    #[test]
    fn wrong_recipient_secret_fails() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_1005);
        let (_x_r, recipient) = keypair(&mut rng);
        let (x_a, access) = keypair(&mut rng);
        let (proof, c) = seal(
            &Scalar::from(11u64),
            &recipient,
            &[access],
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        let binding = opening_binding(&recipient, &[access]).unwrap();
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        let partial = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();
        let wrong = Scalar::random(&mut rng);
        assert!(unseal(&proof, c, &binding, &wrong, &[partial], &TestCtx).is_err());
    }

    #[test]
    fn duplicate_partial_is_idempotent() {
        // A re-submitted Partial is an idempotent share, not double-counted.
        let mut rng = StdRng::seed_from_u64(0x0BEA_1006);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, access) = keypair(&mut rng);
        let m = Scalar::from(0x77u64);
        let (proof, c) = seal(&m, &recipient, &[access], &TestCtx, &mut rng).unwrap();
        let binding = opening_binding(&recipient, &[access]).unwrap();
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        let partial = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();
        let recovered = unseal(
            &proof,
            c,
            &binding,
            &x_r,
            &[partial.clone(), partial],
            &TestCtx,
        )
        .unwrap();
        assert_eq!(recovered, m);
    }

    #[test]
    fn re_run_contribute_double_counts_fail_closed() {
        // Two SEPARATE contribute calls for the same (gate, w) carry different
        // DLEQ entropy — distinct artifacts. The ID-less primitive sums both
        // (bucket → 2·W ≠ Y_access) and fails closed; per-participant dedup is
        // the caller's job. (Re-SENDING the same artifact is the idempotent no-op,
        // covered by `duplicate_partial_is_idempotent`.)
        let mut rng = StdRng::seed_from_u64(0x0BEA_1008);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, access) = keypair(&mut rng);
        let (proof, c) = seal(
            &Scalar::from(0x55u64),
            &recipient,
            &[access],
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        let binding = opening_binding(&recipient, &[access]).unwrap();
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        let p1 = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();
        let p2 = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();
        assert_ne!(
            p1, p2,
            "fresh DLEQ entropy makes the two structurally distinct"
        );
        assert!(matches!(
            unseal(&proof, c, &binding, &x_r, &[p1, p2], &TestCtx),
            Err(Error::Verification(
                "access gate not qualifying: Σ W ≠ Y_access"
            ))
        ));
    }

    #[test]
    fn irrelevant_partial_is_skipped_not_fatal() {
        // A qualifying set plus one junk (DLEQ-failing) partial still opens —
        // the junk fails verification and is skipped, not summed.
        let mut rng = StdRng::seed_from_u64(0x0BEA_1009);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, access) = keypair(&mut rng);
        let m = Scalar::from(0x42u64);
        let (proof, c) = seal(&m, &recipient, &[access], &TestCtx, &mut rng).unwrap();
        let binding = opening_binding(&recipient, &[access]).unwrap();
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        let good = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();
        let mut junk = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();
        junk.masks[0] += g();
        let recovered = unseal(&proof, c, &binding, &x_r, &[junk, good], &TestCtx).unwrap();
        assert_eq!(recovered, m);
    }

    #[test]
    fn tampered_partial_rejected_at_unseal() {
        let mut rng = StdRng::seed_from_u64(0x0BEA_1007);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, access) = keypair(&mut rng);
        let (proof, c) = seal(
            &Scalar::from(3u64),
            &recipient,
            &[access],
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        let binding = opening_binding(&recipient, &[access]).unwrap();
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };
        let mut partial = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();
        partial.masks[0] += g();
        assert!(unseal(&proof, c, &binding, &x_r, &[partial], &TestCtx).is_err());
    }

    /// Latency baseline for the single-gate round-trip phases, against the goal
    /// of a < 5 s constrained-device recovery. Run on a development machine with
    /// `cargo test -p ve-capsule --release -- --ignored --nocapture
    /// opening::tests::latency_baseline`. Treat local numbers as a floor, not the
    /// constrained-device budget.
    #[test]
    #[ignore = "manual perf baseline; run with --release --ignored --nocapture"]
    fn latency_baseline() {
        use std::time::Instant;
        let mut rng = StdRng::seed_from_u64(0x0BEA_5EED);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, access) = keypair(&mut rng);
        let m = Scalar::from(0x00CA_FE12u64);

        // Warm up the lazy tables (BabyTable + the NUMS/boundary tables build on
        // first use) so the timings below are steady-state, not one-time init.
        {
            let (wp, wc) = seal(&m, &recipient, &[access], &TestCtx, &mut rng).unwrap();
            let wb = opening_binding(&recipient, &[access]).unwrap();
            let wr = CapsuleRef {
                elgamal: wp.elgamal(),
                c: wc,
            };
            let wpart = contribute(wr, &wb, &access, &x_a, &TestCtx, &mut rng).unwrap();
            unseal(&wp, wc, &wb, &x_r, &[wpart], &TestCtx).unwrap();
        }

        let t = Instant::now();
        let (proof, c) = seal(&m, &recipient, &[access], &TestCtx, &mut rng).unwrap();
        let seal_ms = t.elapsed().as_secs_f64() * 1e3;

        let binding = opening_binding(&recipient, &[access]).unwrap();
        let capsule = CapsuleRef {
            elgamal: proof.elgamal(),
            c,
        };

        let t = Instant::now();
        crate::composite::verify(&proof, &c, &recipient, &[access], &TestCtx).unwrap();
        let verify_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let partial = contribute(capsule, &binding, &access, &x_a, &TestCtx, &mut rng).unwrap();
        let contribute_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let recovered = unseal(&proof, c, &binding, &x_r, &[partial], &TestCtx).unwrap();
        let unseal_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(recovered, m);

        println!(
            "ve-capsule single-gate round-trip (local): seal={seal_ms:.1}ms \
             verify={verify_ms:.1}ms contribute={contribute_ms:.1}ms \
             unseal={unseal_ms:.1}ms (unseal re-verifies; token path would skip)"
        );
    }
}
