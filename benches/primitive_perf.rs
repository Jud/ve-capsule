//! Parameter-freeze benchmark — BSGS + range-verify MSM benches that
//! freeze the ec-segve-v1 segmentation tuple (the limb width `ℓ`).
//!
//! Perf gate: the constrained-device recovery critical path is
//! `verify_MSM + L·BSGS`. If a candidate can't run that in ~1 s on an
//! reference desktop, that `ℓ` is too slow for the constrained target. Rule: freeze the
//! LARGEST `ℓ` whose critical path is comfortably under ~1 s (margin,
//! not a 0.99 s fit).
//!
//! API choice: raw `k256::{ProjectivePoint, Scalar}`. k256 0.13's group
//! API (`ProjectivePoint::GENERATOR`, `* Scalar`, `to_affine()` +
//! `to_bytes()` for a compressed key) is the cleanest path for a raw
//! bench. There is no batched multi-exp on the stable `k256` surface, so
//! the MSM is the honest unoptimized `Σ aᵢ·Pᵢ` an unbatched verifier
//! does — a fair representative of the verify cost.
//!
//! This is a `harness = false` bench: a plain timed loop with
//! `black_box` + `eprintln` so the numbers are read directly off stderr.
//! Run: `cargo bench -p ve-capsule`.

// Bench-only ergonomics: this binary is never on a shipped hot path.
#![allow(clippy::expect_used, clippy::cast_precision_loss)]

use core::hint::black_box;
use std::collections::HashMap;
use std::time::Instant;

use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{ProjectivePoint, Scalar};
use rand::SeedableRng;
use rand::rngs::StdRng;
use ve_capsule::Params;

/// Median of a small sample (sorted, middle element). Panics on empty
/// input — callers always pass a non-empty slice.
fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    samples[samples.len() / 2]
}

/// Warm `f` once (untimed), then time it `samples` times and return the
/// median wall-clock in milliseconds.
fn time_median_ms(samples: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm-up
    let mut ms: Vec<f64> = (0..samples)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    median(&mut ms)
}

/// Compressed SEC1 key bytes for a projective point, the natural BSGS
/// table key. A `Vec` (not a fixed `[u8; 33]`) because the identity
/// encodes to a single `0x00` byte, not 33 — the `j = 0` baby step is
/// the identity, so a fixed-length copy would panic.
fn compress(p: &ProjectivePoint) -> Vec<u8> {
    p.to_affine().to_encoded_point(true).as_bytes().to_vec()
}

/// Build the BSGS baby-step table: `j → j·G` for `j ∈ [0, m)`, keyed by
/// compressed bytes. Built ONCE per candidate and reused across limbs.
fn build_baby_table(m: u64) -> HashMap<Vec<u8>, u64> {
    let g = ProjectivePoint::GENERATOR;
    let mut table = HashMap::with_capacity(usize::try_from(m).expect("table size fits usize"));
    let mut acc = ProjectivePoint::IDENTITY;
    for j in 0..m {
        table.insert(compress(&acc), j);
        acc += g;
    }
    table
}

/// Worst-case BSGS giant search for a single limb: recover `v = 2^ℓ − 1`
/// (the largest value in `[0, 2^ℓ)`, forcing the maximum giant steps)
/// from `target = v·G`. Returns the recovered scalar so the caller can
/// `black_box` it and assert correctness.
fn bsgs_giant_search(target: ProjectivePoint, m: u64, table: &HashMap<Vec<u8>, u64>) -> u64 {
    // Giant step is `-m·G`; walk `target, target − m·G, target − 2m·G, …`
    // until a residue lands in the baby table. v = i·m + j.
    let g = ProjectivePoint::GENERATOR;
    let neg_m_g = -(g * Scalar::from(m));
    let mut cur = target;
    let mut i = 0u64;
    loop {
        if let Some(&j) = table.get(&compress(&cur)) {
            return i * m + j;
        }
        cur += neg_m_g;
        i += 1;
        // [0, 2^ℓ) needs at most ceil(2^ℓ / m) giant steps; m = 2^ceil(ℓ/2)
        // so this terminates well within m+1 iterations.
        assert!(i <= m, "BSGS giant search overran the search space");
    }
}

/// One unbatched multi-exponentiation `Σ aᵢ·Pᵢ` of the given width —
/// the representative range-verify MSM cost.
fn msm(scalars: &[Scalar], points: &[ProjectivePoint]) -> ProjectivePoint {
    let mut acc = ProjectivePoint::IDENTITY;
    for (a, p) in scalars.iter().zip(points.iter()) {
        acc += *p * *a;
    }
    acc
}

/// Random scalar/point vectors of the given width for the MSM bench.
fn random_msm_inputs(width: usize, rng: &mut StdRng) -> (Vec<Scalar>, Vec<ProjectivePoint>) {
    let g = ProjectivePoint::GENERATOR;
    let scalars: Vec<Scalar> = (0..width).map(|_| random_scalar(rng)).collect();
    // Points are random multiples of G (real verify points are arbitrary
    // group elements; a random multiple is a representative dense point).
    let points: Vec<ProjectivePoint> = (0..width).map(|_| g * random_scalar(rng)).collect();
    (scalars, points)
}

/// A uniformly random non-trivial scalar.
fn random_scalar(rng: &mut StdRng) -> Scalar {
    use k256::elliptic_curve::Field;
    Scalar::random(rng)
}

#[allow(clippy::too_many_lines)]
fn main() {
    // Fixed seed: the bench is about timing, not randomness; a fixed seed
    // keeps the worst-case BSGS target and MSM inputs reproducible.
    let mut rng = StdRng::seed_from_u64(0xEC_5E_61_E0);
    let g = ProjectivePoint::GENERATOR;

    eprintln!("ec-segve primitive_perf (chunk Wb) — k256 raw point/scalar");
    eprintln!("no batched multi-exp on k256 0.13 → MSM is unbatched Σ aᵢ·Pᵢ");
    eprintln!(
        "{:<10} {:>12} {:>14} {:>14} {:>12} {:>12} {:>16}",
        "cand", "bsgs_ms", "msm_1x_ms", "msm_4x_ms", "tbl_ms", "Lsearch_ms", "critical_ms"
    );

    for cand in Params::CANDIDATES {
        let l = cand.limb_bits;
        let big_l = u64::from(cand.limb_count);
        let m = cand.bsgs_table_size();

        // ---- BSGS: table built ONCE, giant search ×L (worst case) ----
        // Worst-case limb value v_k = 2^ℓ − 1, target = v_k·G.
        let v_k = (1u64 << l) - 1;
        let target = g * Scalar::from(v_k);

        // ---- BSGS: table built ONCE, giant search ×L (worst case) ----
        let table_build_ms = time_median_ms(3, || {
            black_box(build_baby_table(m));
        });
        let table = build_baby_table(m);
        // Correctness gate, run untimed so the assert never sits in the
        // measured window: the worst-case limb must round-trip.
        assert_eq!(
            bsgs_giant_search(target, m, &table),
            v_k,
            "BSGS must recover the worst-case limb"
        );
        let one_search_ms = time_median_ms(3, || {
            black_box(bsgs_giant_search(black_box(target), m, &table));
        });
        let l_search_ms = one_search_ms * big_l as f64;
        let bsgs_total_ms = table_build_ms + l_search_ms;

        // ---- Range-verify MSM: width W = 2·L·D·b, and 4× variant ----
        // Compute in usize directly to avoid a u64→usize truncation cast.
        let width_1x =
            2 * cand.limb_count as usize * cand.digits_per_limb as usize * cand.digit_base as usize;
        let width_4x = width_1x * 4;

        let (s1, p1) = random_msm_inputs(width_1x, &mut rng);
        let (s4, p4) = random_msm_inputs(width_4x, &mut rng);

        let msm_1x_ms = time_median_ms(3, || {
            black_box(msm(black_box(&s1), black_box(&p1)));
        });
        let msm_4x_ms = time_median_ms(3, || {
            black_box(msm(black_box(&s4), black_box(&p4)));
        });

        // ---- Critical path = verify_MSM (1×) + L·BSGS ----
        let critical_ms = msm_1x_ms + bsgs_total_ms;

        let name = match l {
            16 => "L16_D16",
            24 => "L24_D16",
            32 => "L32_D16",
            _ => "??",
        };
        eprintln!(
            "{name:<10} {bsgs_total_ms:>12.3} {msm_1x_ms:>14.3} {msm_4x_ms:>14.3} \
             {table_build_ms:>12.3} {l_search_ms:>12.3} {critical_ms:>16.3}"
        );
    }

    eprintln!("(prove-side cost is irrelevant to this freeze gate.)");
}
