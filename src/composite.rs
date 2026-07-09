//! Composite-key seal: seal one scalar to a coefficiented key aggregate,
//! pinning the exact gate list with a commitment `g*`.
//!
//! The capsule construction reuses the existing ec-segve seal proof `π`
//! verbatim — it just seals to the **composite** point `Y*` instead of a single
//! recovery key. With no access keys, `Y* = Y_rcpt`. With gates, the roster is
//! coefficiented as `Y* = a_rcpt·Y_rcpt + Σ a_k·Y_accessₖ`, where each
//! coefficient is deterministically derived from the canonical recipient +
//! sorted duplicate-free gate list. Opening therefore requires the recipient's
//! weighted secret and every access gate's weighted contribution, which is what
//! makes the gate an AND (recipient + all access groups). Per-segment fresh
//! randomness, range/carry/linking proofs, and BSGS are all unchanged — `Y*` is
//! just a public point to `assembly`.
//!
//! Two things must be bound that `Y*` alone does not capture:
//!
//! - **the recipient `Y_rcpt`** — pinned at seal so the quorum can never
//!   redirect the opening (invariant I1); and
//! - **the exact gate list `g*`** — a single aggregate point is not a transparent
//!   roster commitment, even though coefficients make the roster an input to
//!   aggregation. `g* = H(domain ‖ canonical_sorted_unique({Y_accessₖ}))` pins the
//!   list itself; it is a hash, so it is binding, not hiding (a verifier can
//!   only *confirm a guessed list*, never enumerate).
//!
//! Both are bound by threading them through the **context-binding seam** that
//! `assembly`'s Fiat–Shamir transcript already absorbs (item 10), so `π` binds
//! `(caller context, Y_rcpt, g*, Y*, C)` without touching the soundness-critical
//! transcript order in `assembly`. The gate list is canonicalized before `Y*`,
//! `g*`, and the aggregation coefficients are formed, so a capsule cannot
//! smuggle a duplicate gate. Raw and weighted access-key aggregates that cancel
//! to the identity are rejected.
//!
//! # Untrusted key inputs
//!
//! Recipient and access public keys are untrusted inputs. This layer cannot
//! attest that a party actually possesses the secret behind its key, so the
//! integration layer still SHOULD possession-certify and enrollment-bind every
//! key. The crate nevertheless fails closed on the algebraic key-substitution
//! classes it can see locally:
//!
//! - the access-key roster length is capped before relation-table work;
//! - identity components are rejected;
//! - publicly enumerable `G` multiples and small public `H`/NUMS multiples are
//!   rejected per component;
//! - bounded rational scalar, low-coefficient two-source, unit signed-subset,
//!   and low-target-coefficient signed-subset linear relations between
//!   mandatory components are rejected, so related holders cannot synthesize
//!   another recipient/access bucket or scalar from their own secrets; and
//! - gated capsules use MuSig-style deterministic coefficients, so an attacker
//!   cannot choose `Y_mal = X − (Y_rcpt + Σ Y_honest)` and make the aggregate
//!   secret `x` known by construction.

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::{borrow::Cow, collections::HashMap};

use crate::assembly::{self, Proof, reject_publicly_enumerable_key_component};
use crate::batch_affine::{FePoint, batch_add_x_keys, batch_x_keys};
use crate::codec::{POINT_LEN, encode_affine_point, encode_point};
use crate::context::Context;
use crate::error::Error;
use crate::parallel::parallel_map_indexed;
use crate::transcript::{length_prefix, push_framed};
use k256::elliptic_curve::bigint::U256;
use k256::elliptic_curve::ops::Reduce;
use k256::elliptic_curve::point::BatchNormalize;
use k256::{FieldBytes, ProjectivePoint, Scalar};
use rand_core::{CryptoRng, RngCore};
use sha2::{Digest, Sha256};

/// Domain tag for the gate-list commitment `g*`. Bump on any wire change.
const GATE_LIST_DOMAIN: &[u8] = b"ve-capsule.gate-list.v1";

/// Domain tag for the composite seal binding (recipient ‖ g*) folded into the
/// `π` context binding. Bump on any wire change.
const SEAL_BINDING_DOMAIN: &[u8] = b"ve-capsule.seal-binding.v1";

/// Domain tag for the coefficiented composite-key roster digest.
const AGGREGATION_LIST_DOMAIN: &[u8] = b"ve-capsule.key-aggregation-list.v1";

/// Domain tag for deterministic per-key aggregation coefficients.
const AGGREGATION_COEFFICIENT_DOMAIN: &[u8] = b"ve-capsule.key-aggregation-coeff.v1";

/// Signed relation window for mandatory key components. If components have a
/// small public scalar or low-coefficient linear relation, holders of related
/// secrets can synthesize another component's opening partial and collapse an
/// AND gate.
const COMPONENT_RELATION_BOUND: u16 = 4096;

/// Target-side coefficient window for three-component relations. The source
/// coefficients use [`COMPONENT_RELATION_BOUND`]; keeping this side smaller
/// catches low-coefficient target scaling without making every seal quadratic in
/// the full public-relation window.
const COMPONENT_TARGET_RELATION_BOUND: u16 = 64;

/// Fully mixed coefficient window for multi-source subset relations. Pair and
/// two-source target relations use larger specialized scans; this bound covers
/// the higher-arity mixed-coefficient class under the constrained-device roster cap.
const COMPONENT_MIXED_SUBSET_RELATION_BOUND: u16 = 8;

/// Batch size for mixed-relation probes. Keeps normalization amortized without
/// holding the full right-half × target-coefficient product in memory.
const MIXED_RELATION_PROBE_CHUNK: usize = 8192;

/// Maximum access-key gates in one capsule. Access public keys are untrusted
/// inputs and relation scanning is intentionally conservative, so the roster is
/// capped before any per-component tables are built. The cap is sized to keep a
/// cold full-relation scan viable on constrained devices while still covering
/// the highest-arity regression in this module.
const MAX_ACCESS_KEYS: usize = 5;

/// Positive validated-policy bindings cached per process. Policy rosters are
/// expected to be reused across capsules; caching only successful validations
/// avoids repeated BSGS/relation screening without changing the fail-closed path
/// for new or malformed untrusted keys.
const OPENING_BINDING_CACHE_CAP: usize = 64;

/// A gate with its canonical SEC1 encoding, used as the sort/uniqueness key.
struct CanonicalGate {
    encoded: [u8; POINT_LEN],
    point: ProjectivePoint,
}

struct CompositeKey {
    y_star: ProjectivePoint,
    recipient_weight: Scalar,
    gate_weights: Vec<Scalar>,
}

type OpeningBindingCache = Vec<([u8; 32], OpeningBinding)>;

struct ComponentKey {
    is_recipient: bool,
    point: ProjectivePoint,
    relation_table: ComponentRelationTable,
}

impl ComponentKey {
    fn new(is_recipient: bool, point: ProjectivePoint) -> Self {
        let relation_table = ComponentRelationTable::new(point);
        Self {
            is_recipient,
            point,
            relation_table,
        }
    }
}

/// A point's affine x-coordinate as a comparable key. A point and its negation
/// share an x-coordinate, so x-keys fold one sign of every signed-window
/// membership test for free: `x(Q) ∈ {x(a·P) : a ∈ [1, B]}` iff `Q = a·P` for
/// some `1 ≤ |a| ≤ B`.
type XKey = [u8; 32];

/// A sorted x-key set with a u64-prefix fast path: probes binary-search the
/// prefixes and touch the full keys only on a prefix hit.
struct XKeySet {
    prefixes: Vec<u64>,
    keys: Vec<XKey>,
}

impl XKeySet {
    fn from_keys(mut keys: Vec<XKey>) -> Self {
        keys.sort_unstable();
        // Lexicographic key order keeps the big-endian prefixes sorted.
        let prefixes = keys.iter().map(x_key_prefix).collect();
        Self { prefixes, keys }
    }

    fn contains(&self, key: &XKey) -> bool {
        self.prefixes.binary_search(&x_key_prefix(key)).is_ok()
            && self.keys.binary_search(key).is_ok()
    }
}

fn x_key_prefix(key: &XKey) -> u64 {
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&key[..8]);
    u64::from_be_bytes(prefix)
}

/// Per-component signed-window table: the positive multiples `a·P` for
/// `a ∈ [1, COMPONENT_RELATION_BOUND]` as field-coordinate affine points
/// (ready for the chord kernel), plus their x-key set, which covers the full
/// signed window because negative multiples share their positive
/// counterpart's x-coordinate.
struct ComponentRelationTable {
    multiples: Vec<FePoint>,
    x_set: XKeySet,
}

impl ComponentRelationTable {
    fn new(point: ProjectivePoint) -> Self {
        let bound = usize::from(COMPONENT_RELATION_BOUND);
        let mut projective = Vec::with_capacity(bound);
        let mut multiple = point;
        for _ in 1..=COMPONENT_RELATION_BOUND {
            projective.push(multiple);
            multiple += point;
        }
        let multiples: Option<Vec<FePoint>> =
            ProjectivePoint::batch_normalize(projective.as_slice())
                .iter()
                .map(FePoint::from_affine)
                .collect();
        // `a·P` for non-identity `P` on a prime-order curve is never the
        // identity; fail closed rather than scan over a short table.
        assert!(multiples.is_some(), "identity in window-table ladder");
        let multiples = multiples.unwrap_or_default();
        let x_set = XKeySet::from_keys(multiples.iter().map(FePoint::x_key).collect());
        Self { multiples, x_set }
    }

    /// True iff `a·P = ±b·Q` for
    /// `1 ≤ a,b ≤ COMPONENT_RELATION_BOUND`. This catches rational scalar
    /// relations such as `2·P = 3·Q`, not only integer multiples.
    fn has_signed_scalar_relation(&self, rhs: &Self) -> bool {
        self.multiples
            .iter()
            .any(|multiple| rhs.x_set.contains(&multiple.x_key()))
    }

    fn low_multiples(&self) -> &[FePoint] {
        &self.multiples[..usize::from(COMPONENT_TARGET_RELATION_BOUND)]
    }
}

fn has_direct_public_scalar_relation(lhs: &ProjectivePoint, rhs: &ProjectivePoint) -> bool {
    ComponentRelationTable::new(*lhs).has_signed_scalar_relation(&ComponentRelationTable::new(*rhs))
}

#[derive(Clone, Copy)]
enum ComponentRelationKind {
    Scalar,
    Linear,
}

#[derive(Clone, Copy)]
struct SignedSubsetState {
    point: ProjectivePoint,
    non_empty: bool,
    involves_recipient: bool,
}

fn encoded_projective_batch(points: &[ProjectivePoint]) -> Vec<[u8; POINT_LEN]> {
    ProjectivePoint::batch_normalize(points)
        .iter()
        .map(encode_affine_point)
        .collect()
}

/// Canonicalize a gate list: reject oversized rosters, sort by canonical SEC1
/// encoding, and reject duplicates. Empty in ⇒ empty out (an ungated capsule).
/// The result is the canonical set both `Y*` and `g*` are formed over, so
/// ordering can never change either and duplicate gates cannot collapse an AND
/// policy.
fn canonical_gates(access_keys: &[ProjectivePoint]) -> Result<Vec<CanonicalGate>, Error> {
    if access_keys.len() > MAX_ACCESS_KEYS {
        return Err(Error::DegenerateInput("too many access keys"));
    }
    let mut gates: Vec<CanonicalGate> = access_keys
        .iter()
        .map(|p| CanonicalGate {
            encoded: encode_point(p),
            point: *p,
        })
        .collect();
    gates.sort_unstable_by(|a, b| a.encoded.cmp(&b.encoded));
    if gates
        .windows(2)
        .any(|pair| pair[0].encoded == pair[1].encoded)
    {
        return Err(Error::DegenerateInput("duplicate access key"));
    }
    Ok(gates)
}

/// The gate-list commitment `g* = H(domain ‖ count ‖ sorted unique encodings)`.
/// Points are fixed-width SEC1, so the concatenation is unambiguous. An empty
/// list commits to the ungated marker (`count = 0`).
fn gate_commitment(canonical: &[CanonicalGate]) -> [u8; 32] {
    let mut buf = Vec::new();
    push_framed(&mut buf, GATE_LIST_DOMAIN);
    buf.extend_from_slice(&length_prefix(canonical.len()));
    for gate in canonical {
        buf.extend_from_slice(&gate.encoded);
    }
    Sha256::digest(&buf).into()
}

fn aggregation_list_digest(recipient: &ProjectivePoint, canonical: &[CanonicalGate]) -> [u8; 32] {
    let mut buf = Vec::new();
    push_framed(&mut buf, AGGREGATION_LIST_DOMAIN);
    push_framed(&mut buf, &encode_point(recipient));
    buf.extend_from_slice(&length_prefix(canonical.len()));
    for gate in canonical {
        push_framed(&mut buf, &gate.encoded);
    }
    Sha256::digest(&buf).into()
}

fn reduce_digest_to_scalar(digest: [u8; 32]) -> Scalar {
    let mut repr = FieldBytes::default();
    repr.copy_from_slice(&digest);
    <Scalar as Reduce<U256>>::reduce_bytes(&repr)
}

fn aggregation_coefficient(list_digest: &[u8; 32], role: &[u8], encoded: &[u8]) -> Scalar {
    for counter in 0u8..=u8::MAX {
        let mut buf = Vec::new();
        push_framed(&mut buf, AGGREGATION_COEFFICIENT_DOMAIN);
        push_framed(&mut buf, list_digest);
        push_framed(&mut buf, role);
        push_framed(&mut buf, encoded);
        push_framed(&mut buf, &[counter]);
        let coefficient = reduce_digest_to_scalar(Sha256::digest(&buf).into());
        if !bool::from(coefficient.is_zero()) {
            return coefficient;
        }
    }
    // Unreachable unless SHA-256-to-scalar yields zero 256 times in a row.
    Scalar::ONE
}

fn aggregation_weights(
    recipient: &ProjectivePoint,
    canonical: &[CanonicalGate],
) -> (Scalar, Vec<Scalar>) {
    if canonical.is_empty() {
        return (Scalar::ONE, Vec::new());
    }
    let list_digest = aggregation_list_digest(recipient, canonical);
    let recipient_weight =
        aggregation_coefficient(&list_digest, b"recipient", &encode_point(recipient));
    let gate_weights = canonical
        .iter()
        .map(|gate| aggregation_coefficient(&list_digest, b"gate", &gate.encoded))
        .collect();
    (recipient_weight, gate_weights)
}

fn has_public_scalar_relation(lhs: &ComponentKey, rhs: &ComponentKey) -> bool {
    lhs.relation_table
        .has_signed_scalar_relation(&rhs.relation_table)
}

/// X-keys of `{T + a·S, T − a·S : a ∈ [1, COMPONENT_RELATION_BOUND]}` for one
/// (source, target) component pair. Probing these against another component's
/// window table detects every relation `±T = a·S + b·R` with both source
/// coefficients in the signed window: the x-keys fold the global sign and the
/// `±a` branches fold the rest. The set depends only on (source, target), so
/// the pair scan shares it across every rhs component. Sums run through the
/// chord kernel; identity sums (only possible under a scalar relation the
/// earlier screen rejected) drop out exactly as they did under exact
/// encodings.
fn signed_offset_x_keys(target: &ProjectivePoint, source: &ComponentRelationTable) -> Vec<XKey> {
    let target_fe = FePoint::from_projective(target);
    // Components are screened non-identity before any relation scan; fail
    // closed rather than build an empty (never-matching) offset set.
    assert!(
        target_fe.is_some(),
        "identity component reached the offset-set builder"
    );
    let mut pairs = Vec::with_capacity(source.multiples.len() * 2);
    if let Some(target_fe) = target_fe {
        for multiple in &source.multiples {
            pairs.push((target_fe, *multiple));
            pairs.push((target_fe, multiple.negated()));
        }
    }
    batch_add_x_keys(&pairs).into_iter().flatten().collect()
}

/// Unit-target two-source screen for the component pair `(lhs_idx, rhs_idx)`:
/// detects `±T = a·lhs + b·rhs` with `1 ≤ |a|, |b| ≤ COMPONENT_RELATION_BOUND`
/// for every other component `T`, probing the shared per-(source, target)
/// offset sets against the rhs window table. Every (source, target) key this
/// can touch is pre-built by the caller; a missing entry would mean the
/// screen lost coverage, so indexing fails closed by panicking.
fn has_unit_target_two_source_relation(
    components: &[ComponentKey],
    lhs_idx: usize,
    rhs_idx: usize,
    offset_sets: &HashMap<(usize, usize), Vec<XKey>>,
) -> Option<bool> {
    let lhs = &components[lhs_idx];
    let rhs = &components[rhs_idx];
    let source_involves_recipient = lhs.is_recipient || rhs.is_recipient;
    let mut access_only_match = false;
    for (target_idx, target) in components.iter().enumerate() {
        if target_idx == lhs_idx || target_idx == rhs_idx {
            continue;
        }
        let offsets = &offset_sets[&(lhs_idx, target_idx)];
        if offsets
            .iter()
            .any(|key| rhs.relation_table.x_set.contains(key))
        {
            if source_involves_recipient || target.is_recipient {
                return Some(true);
            }
            access_only_match = true;
        }
    }
    access_only_match.then_some(false)
}

fn has_low_coefficient_three_component_relation(
    lhs: &ComponentKey,
    rhs: &ComponentKey,
    targets: &[(bool, &ComponentRelationTable)],
) -> Option<bool> {
    // Folded pair sums: a ∈ [1, bound] on the lhs side, b ∈ ±[1, bound] on
    // the rhs side. Negating a sum lands on the (−a, −b) sum, so the x-key
    // set covers the full signed coefficient square. The chord kernel
    // handles the degenerate diagonal (rationally related sources can make
    // `a·L = ±b·R`): doublings keep their exact x and identity sums drop,
    // both exactly as under the projective pipeline.
    let lhs_low = lhs.relation_table.low_multiples();
    let rhs_low = rhs.relation_table.low_multiples();
    let negated_rhs: Vec<FePoint> = rhs_low.iter().map(FePoint::negated).collect();
    let mut pairs = Vec::with_capacity(lhs_low.len() * rhs_low.len() * 2);
    for lhs_multiple in lhs_low {
        for (rhs_multiple, negated) in rhs_low.iter().zip(&negated_rhs) {
            pairs.push((*lhs_multiple, *rhs_multiple));
            pairs.push((*lhs_multiple, *negated));
        }
    }
    let sum_set = XKeySet::from_keys(batch_add_x_keys(&pairs).into_iter().flatten().collect());
    let source_involves_recipient = lhs.is_recipient || rhs.is_recipient;
    let mut access_only_match = false;
    for &(target_involves_recipient, target_table) in targets {
        let involves_recipient = source_involves_recipient || target_involves_recipient;
        // The target-side candidates k·T for k ∈ ±[1, target bound] are
        // exactly the target's low window-table multiples (x-keys fold the
        // sign), so no fresh point arithmetic is needed.
        for target_multiple in target_table.low_multiples() {
            if sum_set.contains(&target_multiple.x_key()) {
                if involves_recipient {
                    return Some(true);
                }
                access_only_match = true;
                break;
            }
        }
    }
    access_only_match.then_some(false)
}

const SIGNED_SUBSET_EMPTY: u8 = 0b001;
const SIGNED_SUBSET_ACCESS_ONLY: u8 = 0b010;
const SIGNED_SUBSET_WITH_RECIPIENT: u8 = 0b100;

fn ternary_state_count(width: usize) -> usize {
    (0..width).fold(1usize, |count, _| count * 3)
}

fn signed_subset_states(components: &[ComponentKey]) -> Vec<SignedSubsetState> {
    let mut states = vec![SignedSubsetState {
        point: ProjectivePoint::IDENTITY,
        non_empty: false,
        involves_recipient: false,
    }];
    for component in components {
        let mut next = Vec::with_capacity(states.len() * 3);
        for state in &states {
            next.push(*state);
            let involves_recipient = state.involves_recipient || component.is_recipient;
            next.push(SignedSubsetState {
                point: state.point + component.point,
                non_empty: true,
                involves_recipient,
            });
            next.push(SignedSubsetState {
                point: state.point - component.point,
                non_empty: true,
                involves_recipient,
            });
        }
        states = next;
    }
    states
}

fn signed_subset_states_except(
    components: &[ComponentKey],
    excluded_idx: usize,
) -> Vec<SignedSubsetState> {
    let mut states = vec![SignedSubsetState {
        point: ProjectivePoint::IDENTITY,
        non_empty: false,
        involves_recipient: false,
    }];
    for (component_idx, component) in components.iter().enumerate() {
        if component_idx == excluded_idx {
            continue;
        }
        let mut next = Vec::with_capacity(states.len() * 3);
        for state in &states {
            next.push(*state);
            let involves_recipient = state.involves_recipient || component.is_recipient;
            next.push(SignedSubsetState {
                point: state.point + component.point,
                non_empty: true,
                involves_recipient,
            });
            next.push(SignedSubsetState {
                point: state.point - component.point,
                non_empty: true,
                involves_recipient,
            });
        }
        states = next;
    }
    states
}

fn mixed_coefficient_subset_states(
    components: &[&ComponentKey],
    coefficient_bound: u16,
) -> Vec<SignedSubsetState> {
    let mut states = vec![SignedSubsetState {
        point: ProjectivePoint::IDENTITY,
        non_empty: false,
        involves_recipient: false,
    }];
    for component in components {
        let mut next = Vec::with_capacity(states.len() * (usize::from(coefficient_bound) * 2 + 1));
        for state in &states {
            next.push(*state);
            let involves_recipient = state.involves_recipient || component.is_recipient;
            let mut multiple = component.point;
            for _ in 1..=coefficient_bound {
                next.push(SignedSubsetState {
                    point: state.point + multiple,
                    non_empty: true,
                    involves_recipient,
                });
                next.push(SignedSubsetState {
                    point: state.point - multiple,
                    non_empty: true,
                    involves_recipient,
                });
                multiple += component.point;
            }
        }
        states = next;
    }
    states
}

const fn signed_subset_flag(state: &SignedSubsetState) -> u8 {
    if !state.non_empty {
        SIGNED_SUBSET_EMPTY
    } else if state.involves_recipient {
        SIGNED_SUBSET_WITH_RECIPIENT
    } else {
        SIGNED_SUBSET_ACCESS_ONLY
    }
}

fn has_unit_signed_subset_relation(components: &[ComponentKey]) -> Option<bool> {
    let split_at = components.len() / 2;
    let (left, right) = components.split_at(split_at);
    let mut left_sums: HashMap<[u8; POINT_LEN], u8> =
        HashMap::with_capacity(ternary_state_count(left.len()));
    for state in signed_subset_states(left) {
        let flags = left_sums.entry(encode_point(&state.point)).or_default();
        *flags |= signed_subset_flag(&state);
    }

    for right_state in signed_subset_states(right) {
        let needed = encode_point(&(-right_state.point));
        let Some(flags) = left_sums.get(&needed).copied() else {
            continue;
        };
        if !right_state.non_empty {
            if flags & SIGNED_SUBSET_ACCESS_ONLY != 0 {
                return Some(false);
            }
            if flags & SIGNED_SUBSET_WITH_RECIPIENT != 0 {
                return Some(true);
            }
        } else if right_state.involves_recipient {
            if flags != 0 {
                return Some(true);
            }
        } else {
            if flags & (SIGNED_SUBSET_EMPTY | SIGNED_SUBSET_ACCESS_ONLY) != 0 {
                return Some(false);
            }
            if flags & SIGNED_SUBSET_WITH_RECIPIENT != 0 {
                return Some(true);
            }
        }
    }
    None
}

fn has_low_coefficient_signed_subset_target_relation(components: &[ComponentKey]) -> Option<bool> {
    for (target_idx, target) in components.iter().enumerate() {
        let target_low = XKeySet::from_keys(
            target
                .relation_table
                .low_multiples()
                .iter()
                .map(FePoint::x_key)
                .collect(),
        );
        // Batch-normalize every non-empty subset point to a single field
        // inversion instead of one per `from_projective` probe. An identity
        // subset point has no x and is skipped, exactly as the per-point
        // `from_projective` `None` was skipped.
        let states = signed_subset_states_except(components, target_idx);
        let non_empty: Vec<&SignedSubsetState> =
            states.iter().filter(|state| state.non_empty).collect();
        let points: Vec<ProjectivePoint> = non_empty.iter().map(|state| state.point).collect();
        for (state, x_key) in non_empty.iter().zip(batch_x_keys(&points)) {
            let Some(x_key) = x_key else {
                continue;
            };
            if target_low.contains(&x_key) {
                return Some(target.is_recipient || state.involves_recipient);
            }
        }
    }
    None
}

const fn relation_from_half_flags(
    left_flags: u8,
    right_state: &SignedSubsetState,
    target_involves_recipient: bool,
) -> Option<bool> {
    let usable_left_flags = if right_state.non_empty {
        left_flags
    } else {
        left_flags & (SIGNED_SUBSET_ACCESS_ONLY | SIGNED_SUBSET_WITH_RECIPIENT)
    };
    if usable_left_flags == 0 {
        return None;
    }
    Some(
        target_involves_recipient
            || right_state.involves_recipient
            || usable_left_flags & SIGNED_SUBSET_WITH_RECIPIENT != 0,
    )
}

fn mixed_relation_probe_result(
    left_sums: &HashMap<[u8; POINT_LEN], u8>,
    probe_points: &[ProjectivePoint],
    probe_right_states: &[SignedSubsetState],
    target_involves_recipient: bool,
) -> Option<bool> {
    let probe_encodings = encoded_projective_batch(probe_points);
    for (encoded, right_state) in probe_encodings.iter().zip(probe_right_states) {
        let Some(left_flags) = left_sums.get(encoded).copied() else {
            continue;
        };
        if let Some(involves_recipient) =
            relation_from_half_flags(left_flags, right_state, target_involves_recipient)
        {
            return Some(involves_recipient);
        }
    }
    None
}

fn mixed_coefficient_relation_for_target(
    components: &[ComponentKey],
    target_idx: usize,
) -> Option<bool> {
    let target = &components[target_idx];
    let sources = components
        .iter()
        .enumerate()
        .filter_map(|(component_idx, component)| (component_idx != target_idx).then_some(component))
        .collect::<Vec<_>>();
    // Meet in the middle, but bias the split so the probe side (iterated
    // against every one of the `COMPONENT_MIXED_SUBSET_RELATION_BOUND` target
    // multiples) holds the *fewer* sources. The relation space is identical
    // for any partition — only the work isn't, and the probe side is the
    // multiplied one.
    let split_at = sources.len().div_ceil(2);
    let left_states = mixed_coefficient_subset_states(
        &sources[..split_at],
        COMPONENT_MIXED_SUBSET_RELATION_BOUND,
    );
    let left_points = left_states
        .iter()
        .map(|state| state.point)
        .collect::<Vec<_>>();
    let left_encodings = encoded_projective_batch(&left_points);
    let mut left_sums = HashMap::<[u8; POINT_LEN], u8>::with_capacity(left_states.len());
    for (state, encoded) in left_states.iter().zip(left_encodings) {
        let flags = left_sums.entry(encoded).or_default();
        *flags |= signed_subset_flag(state);
    }

    let right_states = mixed_coefficient_subset_states(
        &sources[split_at..],
        COMPONENT_MIXED_SUBSET_RELATION_BOUND,
    );
    let mut target_multiples =
        Vec::with_capacity(usize::from(COMPONENT_MIXED_SUBSET_RELATION_BOUND));
    let mut multiple = target.point;
    for _ in 1..=COMPONENT_MIXED_SUBSET_RELATION_BOUND {
        target_multiples.push(multiple);
        multiple += target.point;
    }

    let mut probe_points = Vec::with_capacity(MIXED_RELATION_PROBE_CHUNK);
    let mut probe_right_states = Vec::with_capacity(MIXED_RELATION_PROBE_CHUNK);
    for right_state in &right_states {
        for target_multiple in &target_multiples {
            probe_points.push(*target_multiple - right_state.point);
            probe_right_states.push(*right_state);
            if probe_points.len() == MIXED_RELATION_PROBE_CHUNK {
                if let Some(involves_recipient) = mixed_relation_probe_result(
                    &left_sums,
                    &probe_points,
                    &probe_right_states,
                    target.is_recipient,
                ) {
                    return Some(involves_recipient);
                }
                probe_points.clear();
                probe_right_states.clear();
            }
        }
    }
    mixed_relation_probe_result(
        &left_sums,
        &probe_points,
        &probe_right_states,
        target.is_recipient,
    )
}

fn has_mixed_coefficient_signed_subset_target_relation(
    components: &[ComponentKey],
) -> Option<bool> {
    let target_results = parallel_map_indexed(components.len(), |target_idx| {
        mixed_coefficient_relation_for_target(components, target_idx)
    });
    for result in target_results {
        if result.is_some() {
            return result;
        }
    }
    None
}

const fn relation_error(involves_recipient: bool, kind: ComponentRelationKind) -> Error {
    if involves_recipient {
        Error::DegenerateInput(match kind {
            ComponentRelationKind::Scalar => "recipient/access keys have a public scalar relation",
            ComponentRelationKind::Linear => "recipient/access keys have a public linear relation",
        })
    } else {
        Error::DegenerateInput(match kind {
            ComponentRelationKind::Scalar => "access keys have a public scalar relation",
            ComponentRelationKind::Linear => "access keys have a public linear relation",
        })
    }
}

fn has_pair_component_relation(
    components: &[ComponentKey],
    pair_indices: &[(usize, usize)],
    offset_sets: &HashMap<(usize, usize), Vec<XKey>>,
) -> Option<bool> {
    let pair_results = parallel_map_indexed(pair_indices.len(), |idx| {
        let (lhs_idx, rhs_idx) = pair_indices[idx];
        let unit_target =
            has_unit_target_two_source_relation(components, lhs_idx, rhs_idx, offset_sets);
        if unit_target.is_some() {
            // The sequential scan errors on the unit-target screen before the
            // low-coefficient screen ever runs for this pair.
            return (unit_target, None);
        }
        let targets = components
            .iter()
            .enumerate()
            .filter_map(|(target_idx, target)| {
                (target_idx != lhs_idx && target_idx != rhs_idx)
                    .then_some((target.is_recipient, &target.relation_table))
            })
            .collect::<Vec<_>>();
        let lhs = &components[lhs_idx];
        let rhs = &components[rhs_idx];
        (
            unit_target,
            has_low_coefficient_three_component_relation(lhs, rhs, &targets),
        )
    });
    for (unit_target, low_coefficient) in pair_results {
        if let Some(involves_recipient) = unit_target {
            return Some(involves_recipient);
        }
        if let Some(involves_recipient) = low_coefficient {
            return Some(involves_recipient);
        }
    }
    None
}

fn reject_component_relations(
    recipient: &ProjectivePoint,
    canonical: &[CanonicalGate],
) -> Result<(), Error> {
    if canonical.is_empty() {
        return Ok(());
    }
    if canonical.len() == 1 {
        if has_direct_public_scalar_relation(recipient, &canonical[0].point) {
            return Err(relation_error(true, ComponentRelationKind::Scalar));
        }
        return Ok(());
    }

    // Window tables are per-component and independent — build them across
    // worker threads.
    let component_points: Vec<(bool, ProjectivePoint)> = std::iter::once((true, *recipient))
        .chain(canonical.iter().map(|gate| (false, gate.point)))
        .collect();
    let components: Vec<ComponentKey> = parallel_map_indexed(component_points.len(), |idx| {
        let (is_recipient, point) = component_points[idx];
        ComponentKey::new(is_recipient, point)
    });

    for (i, lhs) in components.iter().enumerate() {
        for rhs in components.iter().skip(i + 1) {
            if has_public_scalar_relation(lhs, rhs) {
                return Err(relation_error(
                    lhs.is_recipient || rhs.is_recipient,
                    ComponentRelationKind::Scalar,
                ));
            }
        }
    }

    // The (source, target) offset sets and the per-pair screens are likewise
    // independent: build and probe them across worker threads, then inspect
    // the results in pair order so the error reported is exactly the one the
    // sequential scan would hit first.
    let pair_indices: Vec<(usize, usize)> = (0..components.len())
        .flat_map(|lhs_idx| (lhs_idx + 1..components.len()).map(move |rhs_idx| (lhs_idx, rhs_idx)))
        .collect();
    let mut offset_keys: Vec<(usize, usize)> = Vec::new();
    for &(lhs_idx, rhs_idx) in &pair_indices {
        for target_idx in 0..components.len() {
            let key = (lhs_idx, target_idx);
            if target_idx != lhs_idx && target_idx != rhs_idx && !offset_keys.contains(&key) {
                offset_keys.push(key);
            }
        }
    }
    let offset_values = parallel_map_indexed(offset_keys.len(), |idx| {
        let (source_idx, target_idx) = offset_keys[idx];
        signed_offset_x_keys(
            &components[target_idx].point,
            &components[source_idx].relation_table,
        )
    });
    let offset_sets: HashMap<(usize, usize), Vec<XKey>> =
        offset_keys.into_iter().zip(offset_values).collect();

    if let Some(involves_recipient) =
        has_pair_component_relation(&components, &pair_indices, &offset_sets)
    {
        return Err(relation_error(
            involves_recipient,
            ComponentRelationKind::Linear,
        ));
    }

    if let Some(involves_recipient) =
        has_mixed_coefficient_signed_subset_target_relation(&components)
    {
        return Err(relation_error(
            involves_recipient,
            ComponentRelationKind::Linear,
        ));
    }

    if let Some(involves_recipient) = has_low_coefficient_signed_subset_target_relation(&components)
    {
        return Err(relation_error(
            involves_recipient,
            ComponentRelationKind::Linear,
        ));
    }

    if let Some(involves_recipient) = has_unit_signed_subset_relation(&components) {
        return Err(relation_error(
            involves_recipient,
            ComponentRelationKind::Linear,
        ));
    }
    Ok(())
}

/// Form the composite key `Y*` over the canonical gate set, rejecting
/// per-component and aggregate degeneracy. Ungated capsules stay
/// `Y* = Y_rcpt`; gated capsules use deterministic per-key aggregation
/// coefficients:
///
/// `Y* = a_rcpt·Y_rcpt + Σ a_k·Y_accessₖ`.
///
/// This makes the access-key roster a hard input to the public-key aggregation
/// rather than raw additive material. `Y*` itself gets the full enumerable/NUMS
/// rejection inside [`assembly::seal`]/[`assembly::verify`]; here we reject the
/// identity recipient, any identity gate, and raw or weighted canceling access
/// aggregates.
///
/// # Errors
///
/// [`Error::DegenerateInput`] on an identity recipient/gate or a canceling
/// aggregate.
fn composite_key(
    recipient: &ProjectivePoint,
    canonical: &[CanonicalGate],
) -> Result<CompositeKey, Error> {
    if recipient == &ProjectivePoint::IDENTITY {
        return Err(Error::DegenerateInput("recipient key is the identity"));
    }
    // One screening task per component, evaluated across worker threads and
    // inspected in component order, so the first failure reported is the same
    // one the sequential scan would hit.
    let screen_results = parallel_map_indexed(canonical.len() + 1, |idx| {
        let Some(gate_idx) = idx.checked_sub(1) else {
            return reject_publicly_enumerable_key_component(
                recipient,
                "recipient key is publicly enumerable",
                "recipient key is a public NUMS-generator multiple",
                Error::DegenerateInput,
            );
        };
        let gate = &canonical[gate_idx];
        if gate.point == ProjectivePoint::IDENTITY {
            return Err(Error::DegenerateInput("access key is the identity"));
        }
        reject_publicly_enumerable_key_component(
            &gate.point,
            "access key is publicly enumerable",
            "access key is a public NUMS-generator multiple",
            Error::DegenerateInput,
        )
    });
    for result in screen_results {
        result?;
    }
    let mut access_sum = ProjectivePoint::IDENTITY;
    for gate in canonical {
        access_sum += gate.point;
    }
    if !canonical.is_empty() && access_sum == ProjectivePoint::IDENTITY {
        return Err(Error::DegenerateInput(
            "access-key list sums to the identity",
        ));
    }
    reject_component_relations(recipient, canonical)?;
    let (recipient_weight, gate_weights) = aggregation_weights(recipient, canonical);
    let mut weighted_access_sum = ProjectivePoint::IDENTITY;
    for (gate, weight) in canonical.iter().zip(&gate_weights) {
        weighted_access_sum += gate.point * *weight;
    }
    if !canonical.is_empty() && weighted_access_sum == ProjectivePoint::IDENTITY {
        return Err(Error::DegenerateInput(
            "weighted access-key list sums to the identity",
        ));
    }
    Ok(CompositeKey {
        y_star: *recipient * recipient_weight + weighted_access_sum,
        recipient_weight,
        gate_weights,
    })
}

/// Wraps the caller's [`Context`], folding the recipient encoding and `g*` into
/// the binding so `π` commits to them. Domain separation comes from
/// [`SEAL_BINDING_DOMAIN`]; the caller's own binding is framed inside.
struct SealContext<'a, C: Context + ?Sized> {
    inner: &'a C,
    recipient_encoded: [u8; POINT_LEN],
    g_star: [u8; 32],
}

impl<C: Context + ?Sized> Context for SealContext<'_, C> {
    fn domain(&self) -> &'static str {
        self.inner.domain()
    }

    fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
        let inner = self.inner.binding_bytes()?;
        let mut out = Vec::new();
        push_framed(&mut out, SEAL_BINDING_DOMAIN);
        push_framed(&mut out, &inner);
        push_framed(&mut out, &self.recipient_encoded);
        push_framed(&mut out, &self.g_star);
        Ok(Cow::Owned(out))
    }
}

fn opening_binding_from_canonical(
    recipient: &ProjectivePoint,
    canonical: &[CanonicalGate],
) -> Result<OpeningBinding, Error> {
    let cache_key = aggregation_list_digest(recipient, canonical);
    if let Some(binding) = cached_opening_binding(&cache_key) {
        return Ok(binding);
    }
    let composite = composite_key(recipient, canonical)?;
    let g_star = gate_commitment(canonical);
    let gates = canonical.iter().map(|gate| gate.point).collect();
    let binding = OpeningBinding {
        y_star: composite.y_star,
        recipient: *recipient,
        recipient_weight: composite.recipient_weight,
        g_star,
        gates,
        gate_weights: composite.gate_weights,
    };
    cache_opening_binding(cache_key, &binding);
    Ok(binding)
}

fn opening_binding_cache() -> &'static Mutex<OpeningBindingCache> {
    static CACHE: OnceLock<Mutex<OpeningBindingCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::with_capacity(OPENING_BINDING_CACHE_CAP)))
}

fn lock_opening_binding_cache() -> MutexGuard<'static, OpeningBindingCache> {
    match opening_binding_cache().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn cached_opening_binding(cache_key: &[u8; 32]) -> Option<OpeningBinding> {
    let cache = lock_opening_binding_cache();
    cache
        .iter()
        .find_map(|(stored_key, binding)| (stored_key == cache_key).then(|| binding.clone()))
}

fn cache_opening_binding(cache_key: [u8; 32], binding: &OpeningBinding) {
    let mut cache = lock_opening_binding_cache();
    if cache.iter().any(|(stored_key, _)| stored_key == &cache_key) {
        return;
    }
    if cache.len() < OPENING_BINDING_CACHE_CAP {
        cache.push((cache_key, binding.clone()));
    }
}

fn seal_context_from_binding<'a, C: Context + ?Sized>(
    binding: &OpeningBinding,
    ctx: &'a C,
) -> SealContext<'a, C> {
    SealContext {
        inner: ctx,
        recipient_encoded: encode_point(&binding.recipient),
        g_star: binding.g_star,
    }
}

/// Derive `(OpeningBinding, SealContext)` from `(recipient, access_keys, ctx)`.
/// The single place this is computed, so `seal` and `verify` cannot drift — the
/// binding that pins the recipient and exact gate list must be byte-identical on
/// both sides or the construction is unsound.
///
/// # Errors
///
/// [`Error::DegenerateInput`] on a degenerate recipient/gate/aggregate.
fn prepare<'a, C: Context + ?Sized>(
    recipient: &ProjectivePoint,
    access_keys: &[ProjectivePoint],
    ctx: &'a C,
) -> Result<(OpeningBinding, SealContext<'a, C>), Error> {
    let canonical = canonical_gates(access_keys)?;
    let binding = opening_binding_from_canonical(recipient, &canonical)?;
    let seal_ctx = seal_context_from_binding(&binding, ctx);
    Ok((binding, seal_ctx))
}

/// Seal `m` to the coefficiented composite key over `(recipient, access_keys)`,
/// bound to `ctx`. Returns the ec-segve [`Proof`] and the commitment `C = m·G`.
///
/// Recipient and access keys are untrusted input. This function rejects the
/// locally visible degenerate/publicly-enumerable key classes and uses
/// deterministic aggregation coefficients for gated capsules. The integration
/// layer should still possession-certify and enrollment-bind keys before
/// presenting them as authorization material.
///
/// # Errors
///
/// [`Error::DegenerateInput`] on a degenerate recipient/gate/aggregate or a
/// degenerate `Y*` (rejected by [`assembly::seal`]); otherwise any sub-proof
/// error.
pub fn seal<R: RngCore + CryptoRng, C: Context + ?Sized>(
    m: &Scalar,
    recipient: &ProjectivePoint,
    access_keys: &[ProjectivePoint],
    ctx: &C,
    rng: &mut R,
) -> Result<(Proof, ProjectivePoint), Error> {
    let (binding, seal_ctx) = prepare(recipient, access_keys, ctx)?;
    assembly::seal(m, &binding.y_star, &seal_ctx, rng)
}

#[cfg(test)]
pub fn seal_with_prefix_mask_scalars_for_test<R, C>(
    m: &Scalar,
    recipient: &ProjectivePoint,
    access_keys: &[ProjectivePoint],
    ctx: &C,
    rng: &mut R,
    prefix: &[Scalar],
) -> Result<(Proof, ProjectivePoint), Error>
where
    R: RngCore + CryptoRng,
    C: Context + ?Sized,
{
    let (binding, seal_ctx) = prepare(recipient, access_keys, ctx)?;
    assembly::seal_with_prefix_mask_scalars_for_test(m, &binding.y_star, &seal_ctx, rng, prefix)
}

#[cfg(test)]
pub fn seal_with_prefix_value_blindings_for_test<R, C>(
    m: &Scalar,
    recipient: &ProjectivePoint,
    access_keys: &[ProjectivePoint],
    ctx: &C,
    rng: &mut R,
    prefix: &[Scalar],
) -> Result<(Proof, ProjectivePoint), Error>
where
    R: RngCore + CryptoRng,
    C: Context + ?Sized,
{
    let (binding, seal_ctx) = prepare(recipient, access_keys, ctx)?;
    assembly::seal_with_prefix_value_blindings_for_test(m, &binding.y_star, &seal_ctx, rng, prefix)
}

/// Verify a composite-key package against the *expected* `(recipient,
/// access_keys, ctx)` and commitment `C`. Recomputes `Y*` and `g*` from the
/// expected inputs and re-runs `π`; a mismatched recipient or gate list (even
/// one whose sum collides with `Y*`) yields a different binding and fails.
///
/// # Errors
///
/// [`Error::DegenerateInput`] on a degenerate expected input;
/// [`Error::Verification`] on any failed proof gate.
pub fn verify<C: Context + ?Sized>(
    proof: &Proof,
    c: &ProjectivePoint,
    recipient: &ProjectivePoint,
    access_keys: &[ProjectivePoint],
    ctx: &C,
) -> Result<(), Error> {
    verify_with_binding(proof, c, recipient, access_keys, ctx).map(|_| ())
}

/// Verify a composite-key package and return the validated opening binding that
/// was used for verification. Public capsule/case callers use this to avoid
/// re-running the expensive key-screening pass after `π` verifies.
///
/// # Errors
///
/// [`Error::DegenerateInput`] on a degenerate expected input;
/// [`Error::Verification`] on any failed proof gate.
pub fn verify_with_binding<C: Context + ?Sized>(
    proof: &Proof,
    c: &ProjectivePoint,
    recipient: &ProjectivePoint,
    access_keys: &[ProjectivePoint],
    ctx: &C,
) -> Result<OpeningBinding, Error> {
    let (binding, seal_ctx) = prepare(recipient, access_keys, ctx)?;
    assembly::verify(proof, c, &binding.y_star, &seal_ctx)?;
    Ok(binding)
}

/// The capsule's opening-side binding material, derived once from the verified
/// `(recipient, access_keys)`: the coefficiented composite key `Y*`, the
/// recipient point, the gate commitment `g*`, the canonical gate points, and
/// their aggregation coefficients. The opening layer combines this with `C`
/// and the context to bind each `Partial`.
#[derive(Clone)]
pub struct OpeningBinding {
    /// The coefficiented composite key.
    pub y_star: ProjectivePoint,
    /// The verified recipient point.
    pub recipient: ProjectivePoint,
    /// Aggregation coefficient for the recipient.
    pub recipient_weight: Scalar,
    /// The gate commitment `g*` over the canonical gate set.
    pub g_star: [u8; 32],
    /// The canonical sorted duplicate-free gate points.
    pub gates: Vec<ProjectivePoint>,
    /// Aggregation coefficient for each canonical gate, in `gates` order.
    pub gate_weights: Vec<Scalar>,
}

impl OpeningBinding {
    /// Return the deterministic aggregation coefficient for a listed gate.
    pub fn gate_weight(&self, gate: &ProjectivePoint) -> Option<Scalar> {
        self.gates
            .iter()
            .zip(&self.gate_weights)
            .find_map(|(listed, weight)| (listed == gate).then_some(*weight))
    }
}

/// Derive the [`OpeningBinding`] for a capsule from its verified
/// `(recipient, access_keys)`. Same canonicalization + degeneracy rejection as
/// [`seal`]/[`verify`], so the opening side agrees with the sealed core.
///
/// # Errors
///
/// [`Error::DegenerateInput`] on a degenerate recipient/gate/aggregate.
pub fn opening_binding(
    recipient: &ProjectivePoint,
    access_keys: &[ProjectivePoint],
) -> Result<OpeningBinding, Error> {
    let canonical = canonical_gates(access_keys)?;
    opening_binding_from_canonical(recipient, &canonical)
}

/// Re-verify a capsule's seal proof `π` from an [`OpeningBinding`] and the
/// context — the *real* composite verification (it reuses the verified
/// recipient encoding and `g*` the seal bound), not a bare proof-against-`Y*`
/// check. The recipient-side terminal op re-runs this so opening is
/// context-bound even when there are no partials.
///
/// # Errors
///
/// [`Error::Verification`] on any failed proof gate (wrong context, malformed
/// capsule, mismatched binding).
pub fn verify_bound<C: Context + ?Sized>(
    proof: &Proof,
    c: &ProjectivePoint,
    binding: &OpeningBinding,
    ctx: &C,
) -> Result<(), Error> {
    let seal_ctx = seal_context_from_binding(binding, ctx);
    assembly::verify(proof, c, &binding.y_star, &seal_ctx)
}

/// Case-piece verification profile: algebraically verify one proof against an
/// already-derived opening binding, while leaving cross-piece relation secrecy
/// checks to [`crate::case::Case::verify`].
pub fn verify_case_piece_bound<C: Context + ?Sized>(
    proof: &Proof,
    c: &ProjectivePoint,
    binding: &OpeningBinding,
    ctx: &C,
) -> Result<(), Error> {
    let seal_ctx = seal_context_from_binding(binding, ctx);
    assembly::verify_case_piece(proof, c, &binding.y_star, &seal_ctx)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;
    use crate::generators::g;
    use k256::elliptic_curve::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    struct TestCtx;
    impl Context for TestCtx {
        fn domain(&self) -> &'static str {
            "ve-capsule.composite-test"
        }
        fn binding_bytes(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
            Ok(Cow::Borrowed(b"composite-binding"))
        }
    }

    /// A keypair `(x, x·G)` with `x` outside the public BSGS window (use a large
    /// random scalar so `Y*` is never publicly enumerable).
    fn keypair(rng: &mut StdRng) -> (Scalar, ProjectivePoint) {
        let x = Scalar::random(rng);
        (x, g() * x)
    }

    /// Open a composite package with the *summed* secret `x* = Σ participants`.
    /// This is the Unit-2 stand-in for the multi-contributor open flow:
    /// it proves `Y* = x*·G` opens the package.
    fn open_with_weighted_sum(
        proof: &Proof,
        c: &ProjectivePoint,
        recipient: &ProjectivePoint,
        recipient_secret: Scalar,
        access: &[(ProjectivePoint, Scalar)],
    ) -> Result<Scalar, Error> {
        let access_points: Vec<ProjectivePoint> = access.iter().map(|(point, _)| *point).collect();
        let binding = opening_binding(recipient, &access_points)?;
        let mut x_star = recipient_secret * binding.recipient_weight;
        for (gate, weight) in binding.gates.iter().zip(&binding.gate_weights) {
            let (_, secret) = access
                .iter()
                .find(|(point, _)| point == gate)
                .ok_or(Error::DegenerateInput("missing gate secret"))?;
            x_star += *secret * *weight;
        }
        assembly::open(proof, c, &x_star)
    }

    #[test]
    fn ungated_round_trip() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_01);
        let (x_r, recipient) = keypair(&mut rng);
        let m = Scalar::from(0x00AB_CDEFu64);
        let (proof, c) = seal(&m, &recipient, &[], &TestCtx, &mut rng).unwrap();
        assert!(verify(&proof, &c, &recipient, &[], &TestCtx).is_ok());
        assert_eq!(
            open_with_weighted_sum(&proof, &c, &recipient, x_r, &[]).unwrap(),
            m
        );
    }

    #[test]
    fn single_gate_round_trip() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_02);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, access) = keypair(&mut rng);
        let m = Scalar::from(0x1234_5678u64);
        let (proof, c) = seal(&m, &recipient, &[access], &TestCtx, &mut rng).unwrap();
        assert!(verify(&proof, &c, &recipient, &[access], &TestCtx).is_ok());
        assert_eq!(
            open_with_weighted_sum(&proof, &c, &recipient, x_r, &[(access, x_a)]).unwrap(),
            m
        );
    }

    #[test]
    fn multi_gate_and_round_trip() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_03);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, a) = keypair(&mut rng);
        let (x_b, b) = keypair(&mut rng);
        let m = Scalar::from(42u64);
        let (proof, c) = seal(&m, &recipient, &[a, b], &TestCtx, &mut rng).unwrap();
        assert!(verify(&proof, &c, &recipient, &[a, b], &TestCtx).is_ok());
        assert_eq!(
            open_with_weighted_sum(&proof, &c, &recipient, x_r, &[(a, x_a), (b, x_b)]).unwrap(),
            m
        );
    }

    #[test]
    fn gate_order_does_not_matter() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_04);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let (_x_b, b) = keypair(&mut rng);
        let (proof, c) =
            seal(&Scalar::from(7u64), &recipient, &[a, b], &TestCtx, &mut rng).unwrap();
        assert!(verify(&proof, &c, &recipient, &[b, a], &TestCtx).is_ok());
    }

    #[test]
    fn different_gate_list_rejected_even_when_raw_sum_collides() {
        // Seal to {a, b}. Build {a, c, d} with c + d = b. In the old raw
        // additive construction the aggregate would collide; the hardened
        // construction rejects because both the coefficiented aggregate and g*
        // are bound to the exact roster.
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_05);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_b_scalar, b) = keypair(&mut rng);
        let (_a_scalar, a) = keypair(&mut rng);
        let (proof, c) = seal(
            &Scalar::from(99u64),
            &recipient,
            &[a, b],
            &TestCtx,
            &mut rng,
        )
        .unwrap();
        // c_pt + d_pt = b, both random and distinct from a/b.
        let c_scalar = Scalar::random(&mut rng);
        let c_pt = g() * c_scalar;
        let d_pt = b - c_pt; // (b_scalar - c_scalar)·G
        assert_eq!(c_pt + d_pt, b);
        let substitute = [a, c_pt, d_pt];
        assert!(
            verify(&proof, &c, &recipient, &substitute, &TestCtx).is_err(),
            "a different gate list with the same raw sum must not verify"
        );
    }

    #[test]
    fn rogue_access_key_cannot_choose_known_composite_secret() {
        // Regression for the pre-fix raw additive aggregation bug: after seeing
        // the recipient key, a malicious gate could choose
        // Y_mal = X*G - Y_recipient for an attacker-known X. An honest seal to
        // (recipient, Y_mal) then targeted Y* = X*G and was decryptable by the
        // attacker alone.
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_0C);
        let (_x_r, recipient) = keypair(&mut rng);
        let attacker_secret = Scalar::random(&mut rng);
        let rogue_gate = g() * attacker_secret - recipient;
        let m = Scalar::from(0xFEED_FACEu64);

        let (proof, c) = seal(&m, &recipient, &[rogue_gate], &TestCtx, &mut rng).unwrap();
        assert!(verify(&proof, &c, &recipient, &[rogue_gate], &TestCtx).is_ok());
        assert!(
            assembly::open(&proof, &c, &attacker_secret).is_err(),
            "a participant-chosen access key must not make the composite secret attacker-known"
        );
    }

    #[test]
    fn wrong_recipient_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_06);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_w, wrong) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let (proof, c) = seal(&Scalar::from(5u64), &recipient, &[a], &TestCtx, &mut rng).unwrap();
        assert!(verify(&proof, &c, &wrong, &[a], &TestCtx).is_err());
    }

    #[test]
    fn duplicate_gate_is_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_07);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let m = Scalar::from(0xBEEFu64);
        assert!(matches!(
            seal(&m, &recipient, &[a, a], &TestCtx, &mut rng),
            Err(Error::DegenerateInput("duplicate access key"))
        ));

        let (proof, c) = seal(&m, &recipient, &[a], &TestCtx, &mut rng).unwrap();
        assert!(matches!(
            verify(&proof, &c, &recipient, &[a, a], &TestCtx),
            Err(Error::DegenerateInput("duplicate access key"))
        ));
    }

    #[test]
    fn too_many_access_keys_rejected_before_relation_scan() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_17);
        let access_keys = (0..=MAX_ACCESS_KEYS)
            .map(|_| keypair(&mut rng).1)
            .collect::<Vec<_>>();
        assert!(matches!(
            canonical_gates(&access_keys),
            Err(Error::DegenerateInput("too many access keys"))
        ));
    }

    #[test]
    #[ignore = "manual perf baseline; run with --release --ignored --nocapture"]
    fn max_gate_opening_binding_cold_latency() {
        use std::time::Instant;

        const ROSTERS: usize = 4;
        const ROSTERS_DENOMINATOR: f64 = 4.0;
        const GATES: usize = MAX_ACCESS_KEYS;
        let mut rng = StdRng::seed_from_u64(0x51_6E_ED_5E_ED);
        let mut rosters = Vec::with_capacity(ROSTERS);
        for _ in 0..ROSTERS {
            let (_x_r, recipient) = keypair(&mut rng);
            let access = (0..GATES).map(|_| keypair(&mut rng).1).collect::<Vec<_>>();
            rosters.push((recipient, access));
        }

        let (_x_w, warm_recipient) = keypair(&mut rng);
        let warm_access = (0..GATES).map(|_| keypair(&mut rng).1).collect::<Vec<_>>();
        opening_binding(&warm_recipient, &warm_access).unwrap();

        let start = Instant::now();
        for (recipient, access) in &rosters {
            opening_binding(recipient, access).unwrap();
        }
        let total_ms = start.elapsed().as_secs_f64() * 1e3;
        println!(
            "max_gate_opening_binding gates={GATES} rosters={ROSTERS} total_ms={total_ms:.3} avg_ms={:.3}",
            total_ms / ROSTERS_DENOMINATOR
        );
    }

    #[test]
    fn canceling_gate_list_rejected() {
        // a + (-a) = O makes the raw access aggregate identity. Reject at seal
        // even though the weighted aggregate would no longer cancel.
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_08);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let neg_a = -a;
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &recipient,
                &[a, neg_a],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput(
                "access-key list sums to the identity"
            ))
        ));
    }

    #[test]
    fn identity_recipient_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_09);
        let (_x_a, a) = keypair(&mut rng);
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &ProjectivePoint::IDENTITY,
                &[a],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput("recipient key is the identity"))
        ));
    }

    #[test]
    fn identity_gate_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_0A);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &recipient,
                &[a, ProjectivePoint::IDENTITY],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput("access key is the identity"))
        ));
    }

    #[test]
    fn publicly_enumerable_access_key_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_0D);
        let (_x_r, recipient) = keypair(&mut rng);
        let public_gate = g() * Scalar::from(7u64);
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &recipient,
                &[public_gate],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput("access key is publicly enumerable"))
        ));
    }

    #[test]
    fn publicly_enumerable_recipient_key_rejected_even_when_gated() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_0E);
        let public_recipient = g() * Scalar::from(9u64);
        let (_x_a, access) = keypair(&mut rng);
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &public_recipient,
                &[access],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput(
                "recipient key is publicly enumerable"
            ))
        ));
    }

    #[test]
    fn single_access_key_rational_scalar_recipient_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_28);
        let (x_a, access) = keypair(&mut rng);
        let inv_two: Scalar = Scalar::from(2u64).invert().into_option().unwrap();
        let recipient_secret = x_a * Scalar::from(3u64) * inv_two;
        let recipient = g() * recipient_secret;

        let accepted = opening_binding(&recipient, &[access]);
        if let Ok(binding) = &accepted {
            assert_eq!(
                recipient,
                access * Scalar::from(3u64) * inv_two,
                "relation sanity check"
            );
            assert_eq!(
                binding.recipient, recipient,
                "single-gate relation was accepted before the scalar-relation fix"
            );
        }

        assert!(matches!(
            accepted,
            Err(Error::DegenerateInput(
                "recipient/access keys have a public scalar relation"
            ))
        ));
    }

    #[test]
    fn access_key_public_multiple_of_recipient_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_0F);
        let (_x_r, recipient) = keypair(&mut rng);
        let related_access = recipient * Scalar::from(2u64);
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &recipient,
                &[related_access],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput(
                "recipient/access keys have a public scalar relation"
            ))
        ));
    }

    #[test]
    fn related_access_keys_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_10);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, access) = keypair(&mut rng);
        let related_access = access * Scalar::from(2u64);
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &recipient,
                &[access, related_access],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput(
                "access keys have a public scalar relation"
            ))
        ));
    }

    #[test]
    fn access_key_rational_scalar_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_29);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, access) = keypair(&mut rng);
        let inv_two: Scalar = Scalar::from(2u64).invert().into_option().unwrap();
        let related_access = access * Scalar::from(3u64) * inv_two;

        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &recipient,
                &[access, related_access],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput(
                "access keys have a public scalar relation"
            ))
        ));
    }

    #[test]
    fn access_key_public_sum_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_11);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let (_x_b, b) = keypair(&mut rng);
        let related_access = a + b;
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &recipient,
                &[a, b, related_access],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput(
                "access keys have a public linear relation"
            ))
        ));
    }

    #[test]
    fn access_key_mixed_coefficient_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_12);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let (_x_b, b) = keypair(&mut rng);
        let related_access = a * Scalar::from(2u64) + b;
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &recipient,
                &[a, b, related_access],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput(
                "access keys have a public linear relation"
            ))
        ));
    }

    #[test]
    fn access_key_two_high_coefficient_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_13);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let (_x_b, b) = keypair(&mut rng);
        let related_access = a * Scalar::from(2u64) + b * Scalar::from(3u64);
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &recipient,
                &[a, b, related_access],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput(
                "access keys have a public linear relation"
            ))
        ));
    }

    #[test]
    fn access_key_target_coefficient_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_14);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let (_x_b, b) = keypair(&mut rng);
        let inv_five: Scalar = Scalar::from(5u64).invert().into_option().unwrap();
        let related_access = (a * Scalar::from(2u64) + b * Scalar::from(3u64)) * inv_five;
        assert!(matches!(
            seal(
                &Scalar::from(1u64),
                &recipient,
                &[a, b, related_access],
                &TestCtx,
                &mut rng
            ),
            Err(Error::DegenerateInput(
                "access keys have a public linear relation"
            ))
        ));
    }

    #[test]
    fn access_key_three_source_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_18);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, a) = keypair(&mut rng);
        let (x_b, b) = keypair(&mut rng);
        let (x_c, c) = keypair(&mut rng);
        let related_access = a + b + c;
        let related_secret = x_a + x_b + x_c;
        let message = Scalar::from(0x485u64);
        let accepted = seal(
            &message,
            &recipient,
            &[a, b, c, related_access],
            &TestCtx,
            &mut rng,
        );

        if let Ok((proof, commitment)) = &accepted {
            assert_eq!(
                open_with_weighted_sum(
                    proof,
                    commitment,
                    &recipient,
                    x_r,
                    &[
                        (a, x_a),
                        (b, x_b),
                        (c, x_c),
                        (related_access, related_secret)
                    ],
                )
                .unwrap(),
                message,
                "three colluding access holders can synthesize the fourth gate"
            );
        }

        assert!(matches!(
            accepted,
            Err(Error::DegenerateInput(
                "access keys have a public linear relation"
            ))
        ));
    }

    #[test]
    fn access_key_four_source_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_19);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, a) = keypair(&mut rng);
        let (x_b, b) = keypair(&mut rng);
        let (x_c, c) = keypair(&mut rng);
        let (x_d, d) = keypair(&mut rng);
        let related_access = a + b + c + d;
        let related_secret = x_a + x_b + x_c + x_d;
        let message = Scalar::from(0x487u64);
        let accepted = seal(
            &message,
            &recipient,
            &[a, b, c, d, related_access],
            &TestCtx,
            &mut rng,
        );

        if let Ok((proof, commitment)) = &accepted {
            assert_eq!(
                open_with_weighted_sum(
                    proof,
                    commitment,
                    &recipient,
                    x_r,
                    &[
                        (a, x_a),
                        (b, x_b),
                        (c, x_c),
                        (d, x_d),
                        (related_access, related_secret)
                    ],
                )
                .unwrap(),
                message,
                "four colluding access holders can synthesize the fifth gate"
            );
        }

        assert!(matches!(
            accepted,
            Err(Error::DegenerateInput(
                "access keys have a public linear relation"
            ))
        ));
    }

    #[test]
    fn access_key_three_source_target_coefficient_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_23);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, a) = keypair(&mut rng);
        let (x_b, b) = keypair(&mut rng);
        let (x_c, c) = keypair(&mut rng);
        let inv_two: Scalar = Scalar::from(2u64).invert().into_option().unwrap();
        let related_access = (a + b + c) * inv_two;
        let related_secret = (x_a + x_b + x_c) * inv_two;
        let message = Scalar::from(0x493u64);
        assert_eq!(related_access, g() * related_secret);

        let accepted = seal(
            &message,
            &recipient,
            &[a, b, c, related_access],
            &TestCtx,
            &mut rng,
        );
        if let Ok((proof, commitment)) = &accepted {
            assert_eq!(
                open_with_weighted_sum(
                    proof,
                    commitment,
                    &recipient,
                    x_r,
                    &[
                        (a, x_a),
                        (b, x_b),
                        (c, x_c),
                        (related_access, related_secret)
                    ],
                )
                .unwrap(),
                message,
                "three colluding access holders recover the fourth gate secret"
            );
        }

        assert!(matches!(
            accepted,
            Err(Error::DegenerateInput(
                "access keys have a public linear relation"
            ))
        ));
    }

    #[test]
    fn access_key_three_source_mixed_coefficient_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_26);
        let (x_r, recipient) = keypair(&mut rng);
        let (x_a, a) = keypair(&mut rng);
        let (x_b, b) = keypair(&mut rng);
        let (x_c, c) = keypair(&mut rng);
        let inv_two: Scalar = Scalar::from(2u64).invert().into_option().unwrap();
        let related_access =
            (a * Scalar::from(3u64) + b * Scalar::from(5u64) + c * Scalar::from(7u64)) * inv_two;
        let related_secret =
            (x_a * Scalar::from(3u64) + x_b * Scalar::from(5u64) + x_c * Scalar::from(7u64))
                * inv_two;
        let message = Scalar::from(0x494u64);
        assert_eq!(related_access, g() * related_secret);

        let accepted = seal(
            &message,
            &recipient,
            &[a, b, c, related_access],
            &TestCtx,
            &mut rng,
        );
        if let Ok((proof, commitment)) = &accepted {
            assert_eq!(
                open_with_weighted_sum(
                    proof,
                    commitment,
                    &recipient,
                    x_r,
                    &[
                        (a, x_a),
                        (b, x_b),
                        (c, x_c),
                        (related_access, related_secret)
                    ],
                )
                .unwrap(),
                message,
                "mixed-coefficient source holders recover the fourth gate secret"
            );
        }

        assert!(matches!(
            accepted,
            Err(Error::DegenerateInput(
                "access keys have a public linear relation"
            ))
        ));
    }

    #[test]
    fn recipient_three_source_target_coefficient_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_24);
        let (x_a, a) = keypair(&mut rng);
        let (x_b, b) = keypair(&mut rng);
        let (x_c, c) = keypair(&mut rng);
        let recipient_secret =
            (x_a + x_b + x_c) * Scalar::from(2u64).invert().into_option().unwrap();
        let recipient = g() * recipient_secret;

        assert!(matches!(
            opening_binding(&recipient, &[a, b, c]),
            Err(Error::DegenerateInput(
                "recipient/access keys have a public linear relation"
            ))
        ));
    }

    #[test]
    fn recipient_max_source_target_coefficient_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_25);
        let sources = (0..MAX_ACCESS_KEYS)
            .map(|_| keypair(&mut rng))
            .collect::<Vec<_>>();
        let inv_two: Scalar = Scalar::from(2u64).invert().into_option().unwrap();
        let recipient_secret = sources
            .iter()
            .fold(Scalar::ZERO, |sum, (secret, _)| sum + secret)
            * inv_two;
        let recipient = g() * recipient_secret;
        let access = sources.iter().map(|(_, point)| *point).collect::<Vec<_>>();

        assert!(matches!(
            opening_binding(&recipient, &access),
            Err(Error::DegenerateInput(
                "recipient/access keys have a public linear relation"
            ))
        ));
    }

    #[test]
    fn recipient_max_source_mixed_coefficient_relation_rejected() {
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_27);
        let sources = (0..MAX_ACCESS_KEYS)
            .map(|_| keypair(&mut rng))
            .collect::<Vec<_>>();
        let coefficients = [2u64, 3, 5, 7, 8];
        let inv_two: Scalar = Scalar::from(2u64).invert().into_option().unwrap();
        let recipient_secret = sources
            .iter()
            .zip(coefficients)
            .fold(Scalar::ZERO, |sum, ((secret, _), coefficient)| {
                sum + *secret * Scalar::from(coefficient)
            })
            * inv_two;
        let recipient = g() * recipient_secret;
        let access = sources.iter().map(|(_, point)| *point).collect::<Vec<_>>();

        assert!(matches!(
            opening_binding(&recipient, &access),
            Err(Error::DegenerateInput(
                "recipient/access keys have a public linear relation"
            ))
        ));
    }

    #[test]
    fn scalar_relation_window_boundary() {
        // B = 4096·A sits exactly on the signed scalar window; 4097·A is the
        // first multiple outside it (and outside every other screen's reach).
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_20);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let inside = a * Scalar::from(u64::from(COMPONENT_RELATION_BOUND));
        assert!(matches!(
            opening_binding(&recipient, &[a, inside]),
            Err(Error::DegenerateInput(
                "access keys have a public scalar relation"
            ))
        ));
        let outside = a * Scalar::from(u64::from(COMPONENT_RELATION_BOUND) + 1);
        assert!(opening_binding(&recipient, &[a, outside]).is_ok());
    }

    #[test]
    fn unit_target_two_source_window_boundary() {
        // C = 4096·A + 4096·B sits exactly on the unit-target source window;
        // C = 4097·A + B needs a source coefficient outside it and no other
        // screen covers the trio.
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_21);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let (_x_b, b) = keypair(&mut rng);
        let bound = Scalar::from(u64::from(COMPONENT_RELATION_BOUND));
        let inside = a * bound + b * bound;
        assert!(matches!(
            opening_binding(&recipient, &[a, b, inside]),
            Err(Error::DegenerateInput(
                "access keys have a public linear relation"
            ))
        ));
        let outside = a * (bound + Scalar::ONE) + b;
        assert!(opening_binding(&recipient, &[a, b, outside]).is_ok());
    }

    #[test]
    fn low_coefficient_target_window_boundary() {
        // 64·C = 64·A + 63·B sits exactly on the low-coefficient target
        // window (the fractional coefficients keep it out of the unit-target
        // and scalar screens); 65·C = 64·A + 63·B is the first target
        // coefficient outside it.
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_22);
        let (_x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let (_x_b, b) = keypair(&mut rng);
        let target_bound = u64::from(COMPONENT_TARGET_RELATION_BOUND);
        let combination = a * Scalar::from(target_bound) + b * Scalar::from(target_bound - 1);
        let inv_bound: Scalar = Scalar::from(target_bound).invert().into_option().unwrap();
        let inside = combination * inv_bound;
        assert!(matches!(
            opening_binding(&recipient, &[a, b, inside]),
            Err(Error::DegenerateInput(
                "access keys have a public linear relation"
            ))
        ));
        let inv_past_bound: Scalar = Scalar::from(target_bound + 1)
            .invert()
            .into_option()
            .unwrap();
        let outside = combination * inv_past_bound;
        assert!(opening_binding(&recipient, &[a, b, outside]).is_ok());
    }

    #[test]
    fn wrong_sum_secret_does_not_open() {
        // Opening with the recipient secret alone (missing the access secret)
        // recovers nothing — all-or-nothing at the composite key.
        let mut rng = StdRng::seed_from_u64(0xC0_FF_EE_0B);
        let (x_r, recipient) = keypair(&mut rng);
        let (_x_a, a) = keypair(&mut rng);
        let (proof, c) = seal(&Scalar::from(8u64), &recipient, &[a], &TestCtx, &mut rng).unwrap();
        assert!(assembly::open(&proof, &c, &x_r).is_err());
    }
}
