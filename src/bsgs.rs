//! Baby-step giant-step recovery of a limb value from `v·G`.
//!
//! After `ElGamal` decryption a limb surfaces as the point `P = v·G` with
//! `v ∈ [0, 2^ℓ)`. Recovering the small scalar `v` from `P` is a bounded
//! discrete log, solved by baby-step giant-step in `O(2^{ℓ/2})` group ops:
//! precompute a table of `j·G` for `j ∈ [0, m)` (the baby steps), then walk
//! `P − i·(m·G)` for `i ∈ [0, m)` (the giant steps) until a baby step matches.
//! With `m = 2^{⌈ℓ/2⌉}` the two ranges cover all of `[0, 2^ℓ)`.
//!
//! The table is built once (it depends only on `G` and `ℓ`) and reused for
//! every limb of every piece — building it per call would dominate the
//! recovery cost. This is vartime by construction (a deliberate, documented
//! property of the recovery path; the opener already holds the secret key).

#[cfg(test)]
use crate::batch_affine::batch_add_x_keys_visit;
use crate::batch_affine::{
    BatchAddScratch, FePoint, batch_add_signed_stride_x_keys_visit, batch_add_x_keys,
};
use crate::codec::{POINT_LEN, encode_affine_point, encode_point};
use crate::parallel::parallel_any;
use crate::params::Params;
use k256::elliptic_curve::point::{AffineCoordinates, BatchNormalize};
use k256::{ProjectivePoint, Scalar};
use std::collections::HashMap;

/// Chunk size for the batched giant walk. One Montgomery batch-inversion covers
/// a whole chunk, so the per-step affine encoding (a field inversion in the
/// naive walk) collapses to ~one inversion per chunk — the dominant verify cost
/// before this (a recovery-key check did ~131k inversions per verify).
const GIANT_WALK_CHUNK: usize = 1024;

/// Number of signed-`G` query directions probed together by the public-key
/// BSGS table. The batch-affine kernel needs one inversion per row per chunk;
/// 16384 keeps memory modest while amortizing the row walk across relation scans.
const PUBLIC_G_QUERY_CHUNK: usize = 16_384;

/// Batched baby-step giant-step search shared by every BSGS table. Walks
/// `cur = point − i·stride` for `i ∈ [0, giant_steps)`, **batch-normalizing each
/// chunk** so the hot per-step affine encoding pays one inversion per chunk, not
/// one per step. On each table hit at baby index `j`, calls
/// `accept(i·baby_steps + j)` and returns its first `Some`; an identity step is
/// the `j = 0` hit. BSGS representation is unique over the covered range, so
/// there is at most one in-range hit: continuing past a hit whose `accept`
/// returns `None` (an out-of-range over-cover tail under odd `ℓ`) can therefore
/// never skip a real match, and the result is independent of chunk boundaries.
/// This matches the naive per-step walk exactly.
///
/// Production recovery goes through [`batched_giant_search_complete`]; this
/// early-exit variant survives as the test-side differential oracle for it.
#[cfg(test)]
fn batched_giant_search<R>(
    table: &HashMap<[u8; POINT_LEN], u64>,
    stride: &ProjectivePoint,
    point: &ProjectivePoint,
    giant_steps: u64,
    baby_steps: u64,
    accept: impl Fn(u64) -> Option<R>,
) -> Option<R> {
    let chunk_len = u64::try_from(GIANT_WALK_CHUNK).unwrap_or(u64::MAX);
    let mut cur = *point;
    let mut i: u64 = 0;
    while i < giant_steps {
        let chunk = chunk_len.min(giant_steps - i);
        let cap = usize::try_from(chunk).unwrap_or(GIANT_WALK_CHUNK);
        let mut projective = Vec::with_capacity(cap);
        let mut indices = Vec::with_capacity(cap);
        for k in 0..chunk {
            let gi = i + k;
            if cur == ProjectivePoint::IDENTITY {
                // The identity is the baby table's `j = 0` entry.
                if let Some(found) = accept(gi.saturating_mul(baby_steps)) {
                    return Some(found);
                }
            } else {
                projective.push(cur);
                indices.push(gi);
            }
            cur -= *stride;
        }
        // One field inversion for the whole chunk, then encode each affine point
        // with no further inversion. The empty-slice skip is a tidy guard (a
        // chunk is all-identity, hence empty, only under degenerate params); it
        // avoids a no-op `batch_normalize` call rather than fixing a fault.
        if !projective.is_empty() {
            let affine = ProjectivePoint::batch_normalize(projective.as_slice());
            for (a, &gi) in affine.iter().zip(&indices) {
                if let Some(j) = table.get(&encode_affine_point(a)) {
                    if let Some(found) = accept(gi.saturating_mul(baby_steps).saturating_add(*j)) {
                        return Some(found);
                    }
                }
            }
        }
        i += chunk;
    }
    None
}

/// Same search domain as [`batched_giant_search`], but it scans every giant row
/// and records the first accepted hit instead of returning as soon as a hit is
/// found. This is still a variable-time primitive (group ops, normalization, and
/// hash-table probes are not constant-time), but callers that process
/// attacker-tampered ciphertexts can avoid exposing whether a limb failed BSGS
/// or merely failed the final commitment check.
fn batched_giant_search_complete<R>(
    table: &HashMap<[u8; POINT_LEN], u64>,
    stride: &ProjectivePoint,
    point: &ProjectivePoint,
    giant_steps: u64,
    baby_steps: u64,
    accept: impl Fn(u64) -> Option<R>,
) -> Option<R> {
    let chunk_len = u64::try_from(GIANT_WALK_CHUNK).unwrap_or(u64::MAX);
    let mut cur = *point;
    let mut found = None;
    let mut i: u64 = 0;
    while i < giant_steps {
        let chunk = chunk_len.min(giant_steps - i);
        let cap = usize::try_from(chunk).unwrap_or(GIANT_WALK_CHUNK);
        let mut projective = Vec::with_capacity(cap);
        let mut indices = Vec::with_capacity(cap);
        for k in 0..chunk {
            let gi = i + k;
            if cur == ProjectivePoint::IDENTITY {
                if found.is_none() {
                    found = accept(gi.saturating_mul(baby_steps));
                }
            } else {
                projective.push(cur);
                indices.push(gi);
            }
            cur -= *stride;
        }
        if !projective.is_empty() {
            let affine = ProjectivePoint::batch_normalize(projective.as_slice());
            for (a, &gi) in affine.iter().zip(&indices) {
                if let Some(j) = table.get(&encode_affine_point(a)) {
                    let candidate = gi.saturating_mul(baby_steps).saturating_add(*j);
                    if found.is_none() {
                        found = accept(candidate);
                    }
                }
            }
        }
        i += chunk;
    }
    found
}

/// Frozen segmentation tuple this module is built against.
const PARAMS: Params = Params::FROZEN;

/// Baby-step count `m = 2^{⌈ℓ/2⌉}`; also the giant-step stride.
const BABY_STEPS: u64 = PARAMS.bsgs_table_size();

/// Public recovery-key scalars below `2^32` are cheaply enumerable and leak
/// every ciphertext limb to ceremony observers. This is intentionally wider
/// than the limb plaintext BSGS domain, which excludes `2^ell` by design.
const PUBLIC_RECOVERY_KEY_SCALAR_BOUND: u64 = 1u64 << 32;

/// Baby-step count `m` for the public recovery-key scalar table. The table keys
/// baby entries by exact x-coordinate, which folds `±j`, so one giant row covers
/// a `2m`-wide scalar window and the double-stride walk needs `bound / 2m` rows
/// per direction. Raising `m` halves the warm per-candidate rows but multiplies
/// the cold table build and its resident memory in lockstep: at `2^22` the
/// `HashMap<[u8;32], u32>` is ~4.2M entries (~390 MB resident, ~1.6 s cold
/// build on a desktop core, far worse on constrained cores) and only shaves ~35 ms off a
/// verify the batched `msm` already brought under the gate. The cold perf gate
/// measures the
/// build-plus-verify a fresh recovery actually pays, so it pins this constrained-device
/// point and rejects a silent memory/cold-build blow-up for a warm-only gain.
const PUBLIC_RECOVERY_KEY_BABY_STEPS: u64 = 1u64 << 19;

/// Cheap no-false-negative prefilter for public-`G` baby x-coordinates. Relation
/// scans almost never hit the baby table; a 26-bit prefix bitset rejects most
/// random x-coordinates before the exact 32-byte lookup.
const BABY_X_FILTER_BITS: usize = 26;
const BABY_X_FILTER_WORDS: usize = (1usize << BABY_X_FILTER_BITS) / u64::BITS as usize;

/// Precomputed baby-step table for limb discrete-log recovery.
///
/// Maps the canonical encoding of `j·G` to `j` for every `j ∈ [0, m)`. Build
/// once with [`BabyTable::new`] and reuse across all limbs.
pub struct BabyTable {
    table: HashMap<[u8; POINT_LEN], u64>,
    /// `m·G`, subtracted once per giant step.
    stride: ProjectivePoint,
}

impl BabyTable {
    /// Build the baby-step table: `j·G` for `j ∈ [0, BABY_STEPS)`.
    #[must_use]
    pub fn new() -> Self {
        // BABY_STEPS (2^12 for the frozen params) provably fits usize; fall
        // back to an unsized map rather than risk a truncating cast.
        let cap = usize::try_from(BABY_STEPS).unwrap_or(0);
        let mut table = HashMap::with_capacity(cap);
        let mut acc = ProjectivePoint::IDENTITY;
        for j in 0..BABY_STEPS {
            table.insert(encode_point(&acc), j);
            acc += ProjectivePoint::GENERATOR;
        }
        // After the loop `acc == BABY_STEPS·G`, exactly the giant-step stride.
        Self { table, stride: acc }
    }

    /// Recover `v ∈ [0, 2^ℓ)` from `point == v·G`, or `None` if no such `v`
    /// exists in range (the point is not a small multiple of `G`).
    ///
    /// Test-side oracle: production limb recovery uses the complete-scan
    /// variant; tests keep this early-exit walk as its differential check.
    #[cfg(test)]
    #[must_use]
    pub fn recover(&self, point: &ProjectivePoint) -> Option<u32> {
        self.recover_bounded(point, PARAMS.limb_modulus())
    }

    /// Recover `v ∈ [0, max_exclusive)` from `point == v·G`, or `None` if no such
    /// `v` exists in range. Generalizes [`recover`](Self::recover) to a caller-set
    /// upper bound for the homomorphic-aggregate open: summing `H` per-piece limbs
    /// lands a summed limb in `[0, H·2^ℓ)`, so it needs `H×` more giant rows than a
    /// single-capsule limb. The baby table (the `2^{⌈ℓ/2⌉}` precomputed `j·G`) is
    /// unchanged — only the giant walk lengthens — and BSGS uniqueness over the
    /// covered range still gives at most one in-range hit, so the last giant row's
    /// over-cover past `max_exclusive−1` is rejected by the `accept` filter. The
    /// caller bounds `max_exclusive ≤ 2^32` via the `u8` piece count
    /// (`H·2^ℓ ≤ 255·2^24 < 2^32`), so the recovered value fits the `u32` limb.
    ///
    /// Test-side oracle, like [`recover`](Self::recover).
    #[cfg(test)]
    #[must_use]
    pub fn recover_bounded(&self, point: &ProjectivePoint, max_exclusive: u64) -> Option<u32> {
        let giant_steps = max_exclusive.div_ceil(BABY_STEPS);
        batched_giant_search(
            &self.table,
            &self.stride,
            point,
            giant_steps,
            BABY_STEPS,
            |v| {
                u32::try_from(v)
                    .ok()
                    .filter(|&v| u64::from(v) < max_exclusive)
            },
        )
    }

    /// Recover `v ∈ [0, max_exclusive)` without returning early on the giant-step
    /// row that contains `v`.
    ///
    /// This deliberately trades recovery latency for a flatter failure surface at
    /// unauthenticated opening boundaries. It does not make BSGS constant-time;
    /// it only prevents callers from learning, through the public error shape or
    /// coarse hit-position timing, whether a tampered limb was outside the search
    /// interval or recovered but later failed the aggregate commitment check.
    #[must_use]
    pub fn recover_bounded_complete(
        &self,
        point: &ProjectivePoint,
        max_exclusive: u64,
    ) -> Option<u32> {
        let giant_steps = max_exclusive.div_ceil(BABY_STEPS);
        batched_giant_search_complete(
            &self.table,
            &self.stride,
            point,
            giant_steps,
            BABY_STEPS,
            |v| {
                u32::try_from(v)
                    .ok()
                    .filter(|&v| u64::from(v) < max_exclusive)
            },
        )
    }
}

impl Default for BabyTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-wide cached [`BabyTable`] — built once and shared across every
/// seal/verify/open (see the module note on why per-call builds are too costly).
pub fn baby_table() -> &'static BabyTable {
    static TABLE: std::sync::OnceLock<BabyTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(BabyTable::new)
}

/// X-folded baby table for the public recovery-key scalar screen.
///
/// Entries are `(x-prefix(j·G), j)` for `j ∈ [1, m]`, sorted by prefix; a
/// point and its negation share an x-coordinate, so one entry covers `±j`.
/// `j = 0` (the identity) has no affine x and is matched on the walk instead.
struct PublicRecoveryKeyTable {
    x_to_j: HashMap<[u8; 32], u32>,
    x_prefix_filter: BabyXPrefixFilter,
    /// `2m·G`. With x-folded matching a giant row at index `i` covers the
    /// scalar window `[2m·i − m, 2m·i + m]`, so double-stride rows tile the
    /// line gap-free with half the rows of the exact-match stride `m·G`.
    stride: ProjectivePoint,
    /// `−(i·2m·G)` for `i ∈ [1, bound/2m]`, ready for the chord kernel: the
    /// giant rows `direction − i·stride` become independent batched affine
    /// sums instead of a sequential projective walk.
    negated_strides: Vec<FePoint>,
}

struct BabyXPrefixFilter {
    words: Box<[u64]>,
}

impl BabyXPrefixFilter {
    fn new() -> Self {
        Self {
            words: vec![0u64; BABY_X_FILTER_WORDS].into_boxed_slice(),
        }
    }

    fn insert(&mut self, x: &[u8; 32]) {
        let index = baby_x_filter_index(x);
        self.words[index / u64::BITS as usize] |= 1u64 << (index % u64::BITS as usize);
    }

    fn may_contain(&self, x: &[u8; 32]) -> bool {
        let index = baby_x_filter_index(x);
        (self.words[index / u64::BITS as usize] & (1u64 << (index % u64::BITS as usize))) != 0
    }
}

fn baby_x_filter_index(x: &[u8; 32]) -> usize {
    (usize::from(x[0]) << 18)
        | (usize::from(x[1]) << 10)
        | (usize::from(x[2]) << 2)
        | (usize::from(x[3]) >> 6)
}

impl PublicRecoveryKeyTable {
    fn new() -> Self {
        const BUILD_CHUNK: usize = 1 << 16;
        let m = PUBLIC_RECOVERY_KEY_BABY_STEPS;
        let mut x_to_j = HashMap::with_capacity(usize::try_from(m).unwrap_or(0));
        let mut x_prefix_filter = BabyXPrefixFilter::new();
        let mut chunk: Vec<ProjectivePoint> = Vec::with_capacity(BUILD_CHUNK);
        // `acc` enters iteration `j` holding `j·G`; chunked batch
        // normalization keeps the build at one field inversion per chunk.
        let mut acc = ProjectivePoint::GENERATOR;
        let mut next_j: u32 = 1;
        for j in 1..=m {
            chunk.push(acc);
            acc += ProjectivePoint::GENERATOR;
            if chunk.len() == BUILD_CHUNK || j == m {
                for affine in ProjectivePoint::batch_normalize(chunk.as_slice()) {
                    let x: [u8; 32] = affine.x().into();
                    x_prefix_filter.insert(&x);
                    x_to_j.insert(x, next_j);
                    next_j = next_j.saturating_add(1);
                }
                chunk.clear();
            }
        }
        // After the loop `acc == (m + 1)·G`, so `2m·G = 2·(acc − G)`.
        let stride = (acc - ProjectivePoint::GENERATOR).double();

        // Static giant-row ladder for the frozen screen bound: i·stride for
        // i ∈ [1, rows], stored negated. i·stride for i in this range is
        // never the identity (its scalar is far below the group order).
        let rows = usize::try_from(PUBLIC_RECOVERY_KEY_SCALAR_BOUND.div_ceil(m.saturating_mul(2)))
            .unwrap_or(0);
        let mut ladder = Vec::with_capacity(rows);
        let mut row_acc = stride;
        for _ in 0..rows {
            ladder.push(row_acc);
            row_acc += stride;
        }
        let negated_strides: Option<Vec<FePoint>> =
            ProjectivePoint::batch_normalize(ladder.as_slice())
                .iter()
                .map(|affine| FePoint::from_affine(affine).map(|fe| fe.negated()))
                .collect();
        assert!(
            negated_strides.is_some(),
            "identity in the giant-row stride ladder"
        );
        let negated_strides = negated_strides.unwrap_or_default();

        Self {
            x_to_j,
            x_prefix_filter,
            stride,
            negated_strides,
        }
    }

    /// True when `point == v·G` for a signed `v` with `|v| < bound`.
    ///
    /// Probes `point` and `−point` against double-stride giant rows, computed
    /// as one chord-kernel batch per direction over the static negated stride
    /// ladder. An x-prefix hit at baby index `j` is confirmed against an
    /// exact recomputed `j·G` (full x, then y-parity of the exactly
    /// recomputed row point to pin the sign of the row offset), so prefix
    /// collisions can neither accept nor reject a key the exact-encoding
    /// predicate would not, and the recovered candidate `v = 2m·i ± j` is
    /// filtered against `bound` exactly.
    fn contains_signed_scalar_below(&self, point: &ProjectivePoint, bound: u64) -> bool {
        if point == &ProjectivePoint::IDENTITY {
            // The identity is `0·G`.
            return 0 < bound;
        }
        let row_width = PUBLIC_RECOVERY_KEY_BABY_STEPS.saturating_mul(2);
        let rows = bound.div_ceil(row_width);
        // The static ladder is sized for the frozen screen bound; a wider
        // walk would silently lose coverage, so fail closed.
        assert!(
            usize::try_from(rows).is_ok_and(|rows| rows <= self.negated_strides.len()),
            "giant-row ladder shorter than the requested screen bound"
        );
        for direction in [*point, -*point] {
            let Some(direction_fe) = FePoint::from_projective(&direction) else {
                // Unreachable (the identity returned above), but fail closed
                // on the row-0 probe rather than skip the direction.
                return true;
            };
            // Row 0: the walk point is the direction itself.
            if self.row_hit_below(&direction_fe.x_key(), &direction, 0, bound) {
                return true;
            }
            let pairs: Vec<(FePoint, FePoint)> = self
                .negated_strides
                .iter()
                .take(usize::try_from(rows).unwrap_or(usize::MAX))
                .map(|negated_stride| (direction_fe, *negated_stride))
                .collect();
            for (row, key) in batch_add_x_keys(&pairs).into_iter().enumerate() {
                let giant_index = row as u64 + 1;
                match key {
                    // direction − i·stride is the identity:
                    // direction == (2m·i)·G exactly.
                    None => {
                        if row_width.saturating_mul(giant_index) < bound {
                            return true;
                        }
                    }
                    Some(x_key) => {
                        if self.row_hit_below(&x_key, &direction, giant_index, bound) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// True when any point in `points` is `v·G` for `v` in `[0, bound)`.
    ///
    /// This is the batched positive-direction form of
    /// [`contains_signed_scalar_below`]. Callers that already enumerate both
    /// `P` and `-P` can use this to avoid probing every candidate twice.
    #[cfg(test)]
    fn contains_any_scalar_below(&self, points: &[ProjectivePoint], bound: u64) -> bool {
        if points.is_empty() {
            return false;
        }
        if points.contains(&ProjectivePoint::IDENTITY) {
            return 0 < bound;
        }
        let row_width = PUBLIC_RECOVERY_KEY_BABY_STEPS.saturating_mul(2);
        let rows = bound.div_ceil(row_width);
        assert!(
            usize::try_from(rows).is_ok_and(|rows| rows <= self.negated_strides.len()),
            "giant-row ladder shorter than the requested screen bound"
        );
        let rows_len = usize::try_from(rows).unwrap_or(usize::MAX);

        // The candidate sweep is a boolean "does any point hit" with no
        // ordering, so it fans out across worker threads — the lever the
        // recovery-path cross-piece scans (run on constrained cores too) need
        // beyond the per-point chord kernel. The table is read-only.
        parallel_any(points, |chunk| {
            self.screen_chunk_below(chunk, bound, row_width, rows_len)
        })
    }

    /// True when any point in `points` is `±v·G` for `v` in `[0, bound)`.
    ///
    /// Relation gates only care that a public linear combination lands in a
    /// bounded generator window, not which sign it lands on. This x-folded probe
    /// therefore carries each candidate once instead of explicitly testing both
    /// `P` and `-P`, and it can accept a full-x baby-table hit without the
    /// positive-direction y-parity disambiguation used by
    /// [`contains_any_scalar_below`].
    fn contains_any_signed_scalar_below(&self, points: &[ProjectivePoint], bound: u64) -> bool {
        if points.is_empty() {
            return false;
        }
        if points.contains(&ProjectivePoint::IDENTITY) {
            return 0 < bound;
        }
        let row_width = PUBLIC_RECOVERY_KEY_BABY_STEPS.saturating_mul(2);
        let rows = bound.div_ceil(row_width);
        assert!(
            usize::try_from(rows).is_ok_and(|rows| rows <= self.negated_strides.len()),
            "giant-row ladder shorter than the requested screen bound"
        );
        let rows_len = usize::try_from(rows).unwrap_or(usize::MAX);

        parallel_any(points, |chunk| {
            self.screen_signed_chunk_below(chunk, bound, row_width, rows_len)
        })
    }

    fn contains_any_signed_fe_scalar_below(&self, points: &[FePoint], bound: u64) -> bool {
        if points.is_empty() {
            return false;
        }
        let row_width = PUBLIC_RECOVERY_KEY_BABY_STEPS.saturating_mul(2);
        let rows = bound.div_ceil(row_width);
        assert!(
            usize::try_from(rows).is_ok_and(|rows| rows <= self.negated_strides.len()),
            "giant-row ladder shorter than the requested screen bound"
        );
        let rows_len = usize::try_from(rows).unwrap_or(usize::MAX);

        parallel_any(points, |chunk| {
            self.screen_signed_fe_chunk_below(chunk, bound, row_width, rows_len)
        })
    }

    /// Sequential screen of one point chunk: `true` if any point is a positive
    /// `G`-multiple below `bound`. Inner `PUBLIC_G_QUERY_CHUNK` batches keep
    /// one Montgomery inversion per chunk for the row sweep.
    #[cfg(test)]
    fn screen_chunk_below(
        &self,
        points: &[ProjectivePoint],
        bound: u64,
        row_width: u64,
        rows_len: usize,
    ) -> bool {
        for points_chunk in points.chunks(PUBLIC_G_QUERY_CHUNK) {
            let mut direction_projectives = Vec::with_capacity(points_chunk.len());
            for point in points_chunk {
                direction_projectives.push(*point);
            }
            let direction_affines =
                ProjectivePoint::batch_normalize(direction_projectives.as_slice());
            let directions = direction_projectives
                .into_iter()
                .zip(direction_affines.iter())
                .map(|(direction, affine)| {
                    FePoint::from_affine(affine)
                        .map(|direction_fe| (direction, direction_fe))
                        .ok_or(())
                })
                .collect::<Result<Vec<_>, _>>();
            let Ok(directions) = directions else {
                return true;
            };
            let chunk = directions.as_slice();
            for (direction, direction_fe) in chunk {
                if self.row_hit_below(&direction_fe.x_key(), direction, 0, bound) {
                    return true;
                }
            }

            let mut pairs = Vec::with_capacity(chunk.len());
            let mut scratch = BatchAddScratch::with_capacity(chunk.len());
            for (row, negated_stride) in self.negated_strides.iter().take(rows_len).enumerate() {
                pairs.clear();
                pairs.extend(
                    chunk
                        .iter()
                        .map(|&(_direction, direction_fe)| (direction_fe, *negated_stride)),
                );
                let giant_index = row as u64 + 1;
                let mut key_index = 0usize;
                if batch_add_x_keys_visit(&pairs, &mut scratch, |key| {
                    let (direction, _direction_fe) = chunk[key_index];
                    key_index += 1;
                    match key {
                        None => {
                            if row_width.saturating_mul(giant_index) < bound {
                                return true;
                            }
                        }
                        Some(x_key) => {
                            if self.row_hit_below(&x_key, &direction, giant_index, bound) {
                                return true;
                            }
                        }
                    }
                    false
                }) {
                    return true;
                }
            }
        }

        false
    }

    fn screen_signed_chunk_below(
        &self,
        points: &[ProjectivePoint],
        bound: u64,
        row_width: u64,
        rows_len: usize,
    ) -> bool {
        for points_chunk in points.chunks(PUBLIC_G_QUERY_CHUNK) {
            let direction_affines = ProjectivePoint::batch_normalize(points_chunk);
            let unsigned_directions = direction_affines
                .iter()
                .map(FePoint::from_affine)
                .collect::<Option<Vec<_>>>();
            let Some(unsigned_directions) = unsigned_directions else {
                return true;
            };
            for direction_fe in &unsigned_directions {
                if self.signed_x_hit_below(&direction_fe.x_key(), 0, bound) {
                    return true;
                }
            }

            let mut scratch = BatchAddScratch::with_capacity(unsigned_directions.len());
            for (row, negated_stride) in self.negated_strides.iter().take(rows_len).enumerate() {
                let giant_index = row as u64 + 1;
                if batch_add_signed_stride_x_keys_visit(
                    &unsigned_directions,
                    *negated_stride,
                    &mut scratch,
                    |minus_key, plus_key| {
                        for key in [minus_key, plus_key] {
                            match key {
                                None => {
                                    if row_width.saturating_mul(giant_index) < bound {
                                        return true;
                                    }
                                }
                                Some(x_key) => {
                                    if self.signed_x_hit_below(&x_key, giant_index, bound) {
                                        return true;
                                    }
                                }
                            }
                        }
                        false
                    },
                ) {
                    return true;
                }
            }
        }

        false
    }

    fn screen_signed_fe_chunk_below(
        &self,
        points: &[FePoint],
        bound: u64,
        row_width: u64,
        rows_len: usize,
    ) -> bool {
        for points_chunk in points.chunks(PUBLIC_G_QUERY_CHUNK) {
            for direction_fe in points_chunk {
                if self.signed_x_hit_below(&direction_fe.x_key(), 0, bound) {
                    return true;
                }
            }

            let mut scratch = BatchAddScratch::with_capacity(points_chunk.len());
            for (row, negated_stride) in self.negated_strides.iter().take(rows_len).enumerate() {
                let giant_index = row as u64 + 1;
                if batch_add_signed_stride_x_keys_visit(
                    points_chunk,
                    *negated_stride,
                    &mut scratch,
                    |minus_key, plus_key| {
                        for key in [minus_key, plus_key] {
                            match key {
                                None => {
                                    if row_width.saturating_mul(giant_index) < bound {
                                        return true;
                                    }
                                }
                                Some(x_key) => {
                                    if self.signed_x_hit_below(&x_key, giant_index, bound) {
                                        return true;
                                    }
                                }
                            }
                        }
                        false
                    },
                ) {
                    return true;
                }
            }
        }

        false
    }

    /// Exact row membership: `cur_x` is the x-key of the giant-walk point
    /// `direction − giant_index·stride`; return whether that point equals
    /// `±j·G` for a table entry `j` whose candidate scalar
    /// `2m·giant_index ± j` lies in `[0, bound)`. A negative candidate
    /// belongs to the opposite walk direction and is skipped here. The walk
    /// point's y-parity (needed only on a full-x match, i.e. a real hit or a
    /// ~2⁻⁴⁵ prefix collision) is taken from an exact projective recompute.
    fn row_hit_below(
        &self,
        cur_x: &[u8; 32],
        direction: &ProjectivePoint,
        giant_index: u64,
        bound: u64,
    ) -> bool {
        if !self.x_prefix_filter.may_contain(cur_x) {
            return false;
        }
        let Some(&j) = self.x_to_j.get(cur_x) else {
            return false;
        };
        let row_base = PUBLIC_RECOVERY_KEY_BABY_STEPS
            .saturating_mul(2)
            .saturating_mul(giant_index);
        let baby = (ProjectivePoint::GENERATOR * Scalar::from(u64::from(j))).to_affine();
        let cur = (*direction - self.stride * Scalar::from(giant_index)).to_affine();
        let candidate = if baby.y_is_odd().unwrap_u8() == cur.y_is_odd().unwrap_u8() {
            // cur == +j·G ⟹ direction == (row_base + j)·G.
            row_base.checked_add(u64::from(j))
        } else {
            // cur == −j·G ⟹ direction == (row_base − j)·G.
            row_base.checked_sub(u64::from(j))
        };
        if let Some(v) = candidate {
            return v < bound;
        }
        false
    }

    fn signed_x_hit_below(&self, cur_x: &[u8; 32], giant_index: u64, bound: u64) -> bool {
        if !self.x_prefix_filter.may_contain(cur_x) {
            return false;
        }
        let Some(&j) = self.x_to_j.get(cur_x) else {
            return false;
        };
        let row_base = PUBLIC_RECOVERY_KEY_BABY_STEPS
            .saturating_mul(2)
            .saturating_mul(giant_index);
        let j = u64::from(j);
        if giant_index == 0 {
            return j < bound;
        }
        if row_base.checked_add(j).is_some_and(|v| v < bound)
            || row_base.checked_sub(j).is_some_and(|v| v < bound)
        {
            return true;
        }
        false
    }
}

fn public_recovery_key_table() -> &'static PublicRecoveryKeyTable {
    static TABLE: std::sync::OnceLock<PublicRecoveryKeyTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(PublicRecoveryKeyTable::new)
}

/// Return `true` when `point` has a signed public recovery-key scalar below the
/// crate's low-scalar rejection bound.
#[must_use]
pub fn is_public_recovery_key_scalar_multiple(point: &ProjectivePoint) -> bool {
    is_signed_g_multiple_below(point, PUBLIC_RECOVERY_KEY_SCALAR_BOUND)
}

/// Return `true` when `point == v*G` for signed `v` with `|v| < bound`.
#[must_use]
pub fn is_signed_g_multiple_below(point: &ProjectivePoint, bound: u64) -> bool {
    public_recovery_key_table().contains_signed_scalar_below(point, bound)
}

/// Return `true` when any point in `points` has a non-negative public `G`
/// scalar below `bound`.
#[must_use]
#[cfg(test)]
pub fn any_g_multiple_below(points: &[ProjectivePoint], bound: u64) -> bool {
    public_recovery_key_table().contains_any_scalar_below(points, bound)
}

/// Return `true` when any point in `points` has a signed public `G` scalar below
/// `bound`.
#[must_use]
pub fn any_signed_g_multiple_below(points: &[ProjectivePoint], bound: u64) -> bool {
    public_recovery_key_table().contains_any_signed_scalar_below(points, bound)
}

/// Return `true` when any affine point in `points` has a signed public `G`
/// scalar below `bound`.
#[must_use]
pub fn any_signed_fe_g_multiple_below(points: &[FePoint], bound: u64) -> bool {
    public_recovery_key_table().contains_any_signed_fe_scalar_below(points, bound)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use k256::Scalar;

    fn point_for(v: u64) -> ProjectivePoint {
        ProjectivePoint::GENERATOR * Scalar::from(v)
    }

    /// `Some(v)` as the `u32` `recover` returns for an in-range `v`.
    fn expected(v: u64) -> Option<u32> {
        u32::try_from(v).ok()
    }

    #[test]
    fn recovers_boundary_values() {
        let t = BabyTable::new();
        for v in [
            0u64,
            1,
            2,
            BABY_STEPS - 1,
            BABY_STEPS,
            PARAMS.limb_modulus() - 1,
        ] {
            assert_eq!(t.recover(&point_for(v)), expected(v), "failed for v={v}");
        }
    }

    #[test]
    fn recovers_random_values() {
        use rand::rngs::StdRng;
        use rand::{RngCore, SeedableRng};
        let t = BabyTable::new();
        let mut rng = StdRng::seed_from_u64(0xB5_65_5E_02);
        for _ in 0..256 {
            let v = rng.next_u64() % PARAMS.limb_modulus();
            assert_eq!(t.recover(&point_for(v)), expected(v));
        }
    }

    #[test]
    fn complete_scan_matches_regular_bounded_recovery() {
        let t = BabyTable::new();
        for v in [0u64, 1, BABY_STEPS - 1, BABY_STEPS, 65_537] {
            let point = point_for(v);
            assert_eq!(
                t.recover_bounded_complete(&point, PARAMS.limb_modulus()),
                t.recover_bounded(&point, PARAMS.limb_modulus())
            );
        }
        let outside = point_for(PARAMS.limb_modulus());
        assert_eq!(
            t.recover_bounded_complete(&outside, PARAMS.limb_modulus()),
            None
        );
    }

    #[test]
    fn rejects_out_of_range() {
        let t = BabyTable::new();
        // v = 2^ℓ is one past the largest in-range limb: must not be recovered.
        assert_eq!(t.recover(&point_for(PARAMS.limb_modulus())), None);
        // A far out-of-range value likewise yields None.
        assert_eq!(t.recover(&point_for(PARAMS.limb_modulus() + 12345)), None);
    }

    #[test]
    fn public_recovery_key_table_catches_limb_boundary_and_tail() {
        assert!(is_public_recovery_key_scalar_multiple(&point_for(
            PARAMS.limb_modulus()
        )));
        assert!(is_public_recovery_key_scalar_multiple(&point_for(
            PARAMS.limb_modulus() + 1
        )));
        assert!(is_public_recovery_key_scalar_multiple(
            &(-point_for(PARAMS.limb_modulus()))
        ));
    }

    #[test]
    fn public_recovery_key_scalar_window_boundary() {
        // 2^32 − 1 is the largest enumerable scalar in either sign; 2^32 is
        // out, and 2^32 + 1 pins the row-boundary sign disambiguation (its
        // giant row contains an x-fold hit whose negative-offset candidate
        // is in range but whose true offset is not).
        let bound = PUBLIC_RECOVERY_KEY_SCALAR_BOUND;
        assert!(is_public_recovery_key_scalar_multiple(&point_for(
            bound - 1
        )));
        assert!(is_public_recovery_key_scalar_multiple(
            &(-point_for(bound - 1))
        ));
        assert!(!is_public_recovery_key_scalar_multiple(&point_for(bound)));
        assert!(!is_public_recovery_key_scalar_multiple(&point_for(
            bound + 1
        )));
        assert!(!is_public_recovery_key_scalar_multiple(
            &(-point_for(bound + 1))
        ));
    }

    #[test]
    fn positive_batch_public_g_probe_has_positive_semantics() {
        let below = point_for(PARAMS.limb_modulus() - 1);
        let negative_below = -below;
        let outside = point_for(PARAMS.limb_modulus());
        assert!(any_g_multiple_below(
            &[outside, below],
            PARAMS.limb_modulus()
        ));
        assert!(!any_g_multiple_below(
            &[outside, negative_below],
            PARAMS.limb_modulus()
        ));
    }

    #[test]
    fn signed_batch_public_g_probe_folds_negative_semantics() {
        let below = point_for(PARAMS.limb_modulus() - 1);
        let negative_below = -below;
        let outside = point_for(PARAMS.limb_modulus());
        assert!(any_signed_g_multiple_below(
            &[outside, below],
            PARAMS.limb_modulus()
        ));
        assert!(any_signed_g_multiple_below(
            &[outside, negative_below],
            PARAMS.limb_modulus()
        ));
        assert!(!any_signed_g_multiple_below(
            &[point_for(PARAMS.limb_modulus())],
            PARAMS.limb_modulus()
        ));
        assert!(any_signed_g_multiple_below(
            &[ProjectivePoint::IDENTITY],
            PARAMS.limb_modulus()
        ));
    }

    #[test]
    fn signed_affine_batch_public_g_probe_matches_projective_probe() {
        let below = point_for(PARAMS.limb_modulus() - 1);
        let negative_below = -below;
        let outside = point_for(PARAMS.limb_modulus());
        let points = [outside, below, negative_below];
        let affines = ProjectivePoint::batch_normalize(&points);
        let fe_points = affines
            .iter()
            .map(FePoint::from_affine)
            .collect::<Option<Vec<_>>>()
            .unwrap();
        assert_eq!(
            any_signed_fe_g_multiple_below(&fe_points, PARAMS.limb_modulus()),
            any_signed_g_multiple_below(&points, PARAMS.limb_modulus())
        );
    }

    #[test]
    fn public_recovery_key_table_fold_seams() {
        // The x-fold seams: the baby bound m, the row width 2m, and their
        // neighbors, in both signs, plus the identity (scalar zero).
        let m = PUBLIC_RECOVERY_KEY_BABY_STEPS;
        // 2^32 − 2m is the last in-range identity row of the giant walk.
        let last_identity_row = PUBLIC_RECOVERY_KEY_SCALAR_BOUND - 2 * m;
        for v in [
            1,
            m - 1,
            m,
            m + 1,
            2 * m - 1,
            2 * m,
            2 * m + 1,
            3 * m,
            last_identity_row,
        ] {
            assert!(
                is_public_recovery_key_scalar_multiple(&point_for(v)),
                "v={v}"
            );
            assert!(
                is_public_recovery_key_scalar_multiple(&(-point_for(v))),
                "-v={v}"
            );
        }
        assert!(is_public_recovery_key_scalar_multiple(
            &ProjectivePoint::IDENTITY
        ));
    }

    #[test]
    fn identity_recovers_zero() {
        let t = BabyTable::new();
        assert_eq!(t.recover(&ProjectivePoint::IDENTITY), Some(0));
    }
}
