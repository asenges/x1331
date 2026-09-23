use std::hint::black_box;
use std::time::Instant;

const DEPTHS: [usize; 18] = [
    1, 2, 3, 4, 5, 6, 8, 11, 16, 21, 32, 48, 64, 85, 128, 192, 256, 299,
];

const TARGET_OPERATIONS: u64 = 120_000_000;
const MIN_TRIALS: u64 = 250_000;
const MAX_TRIALS: u64 = 5_000_000;

#[inline(always)]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline(always)]
fn interfere(state: u8, accumulator: u64, level: usize) -> u64 {
    let s = state as u64;
    let layer = state.count_ones() as u64;

    accumulator
        .rotate_left(((s + level as u64) & 63) as u32)
        .wrapping_add(
            s.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(layer << (level & 31)),
        )
        ^ ((level as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03))
}

#[inline(never)]
fn navigate_deep(seed: u64, depth: usize) -> u64 {
    let mut rng = seed;
    let mut word = 0u64;
    let mut available_bits = 0usize;
    let mut acc = 0x1331_1331_1331_1331u64;

    for level in 0..depth {
        if available_bits < 3 {
            word = splitmix64(&mut rng);
            available_bits = 63;
        }

        let state = (word & 0b111) as u8;
        word >>= 3;
        available_bits -= 3;

        acc = interfere(state, acc, level);
    }

    black_box(acc)
}

fn trials_for_depth(depth: usize) -> u64 {
    let desired = TARGET_OPERATIONS / depth as u64;
    desired.clamp(MIN_TRIALS, MAX_TRIALS)
}

fn log10_space(depth: usize) -> f64 {
    depth as f64 * (8.0_f64).log10()
}

fn fit_linear(results: &[(f64, f64)]) -> (f64, f64, f64) {
    let n = results.len() as f64;

    let sum_x: f64 = results.iter().map(|x| x.0).sum();
    let sum_y: f64 = results.iter().map(|x| x.1).sum();
    let sum_xx: f64 = results.iter().map(|x| x.0 * x.0).sum();
    let sum_xy: f64 = results.iter().map(|x| x.0 * x.1).sum();

    let denom = n * sum_xx - sum_x * sum_x;

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    let mean_y = sum_y / n;

    let ss_tot: f64 = results
        .iter()
        .map(|x| {
            let d = x.1 - mean_y;
            d * d
        })
        .sum();

    let ss_res: f64 = results
        .iter()
        .map(|x| {
            let predicted = intercept + slope * x.0;
            let d = x.1 - predicted;
            d * d
        })
        .sum();

    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    (slope, intercept, r2)
}

fn main() {
    println!("============================================================");
    println!(" X1331 LIVE-09F.2/.3 — ULTRA-DEEP SCALING LAB");
    println!(" RAW 3-BIT FIGURES GENERATED AT EVERY LEVEL");
    println!(" No flat materialization of 8^d");
    println!("============================================================");
    println!();

    println!(
        "{:<6} {:>6} {:>14} {:>12} {:>12} {:>12} {:>12}",
        "DEPTH", "BITS", "LOG10(N)", "TRIALS", "TOTAL_MS", "NS/NAV", "NS/LEVEL"
    );

    let mut benchmark_seed = 0x1331_09F2_2026_0923u64;
    let mut results: Vec<(f64, f64)> = Vec::new();
    let mut checksum = 0u64;

    for depth in DEPTHS {
        let trials = trials_for_depth(depth);

        for _ in 0..10_000u64 {
            let seed = splitmix64(&mut benchmark_seed);
            checksum ^= navigate_deep(black_box(seed), black_box(depth));
        }

        let start = Instant::now();

        for _ in 0..trials {
            let seed = splitmix64(&mut benchmark_seed);
            checksum ^= navigate_deep(black_box(seed), black_box(depth));
        }

        let elapsed = start.elapsed();
        let ns = elapsed.as_nanos() as f64;
        let ns_per_nav = ns / trials as f64;
        let ns_per_level = ns_per_nav / depth as f64;

        results.push((depth as f64, ns_per_nav));

        println!(
            "{:<6} {:>6} {:>14.3} {:>12} {:>12.3} {:>12.3} {:>12.3}",
            depth,
            depth * 3,
            log10_space(depth),
            trials,
            elapsed.as_secs_f64() * 1000.0,
            ns_per_nav,
            ns_per_level
        );
    }

    let (slope, intercept, r2) = fit_linear(&results);

    let depth11_ns = results.iter().find(|x| x.0 == 11.0).map(|x| x.1).unwrap();

    let depth85_ns = results.iter().find(|x| x.0 == 85.0).map(|x| x.1).unwrap();

    let depth299_ns = results.iter().find(|x| x.0 == 299.0).map(|x| x.1).unwrap();

    println!();
    println!("============================================================");
    println!(" LIVE-09F ULTRA-DEEP FINAL");
    println!("============================================================");

    println!("linear_fit_ns = {:.9} * depth + {:.9}", slope, intercept);
    println!("linear_fit_r2={:.9}", r2);

    println!();
    println!("depth11_bits=33");
    println!("depth11_log10_space={:.6}", log10_space(11));
    println!("depth11_ns={:.6}", depth11_ns);

    println!();
    println!("depth85_bits=255");
    println!("depth85_log10_space={:.6}", log10_space(85));
    println!("depth85_ns={:.6}", depth85_ns);

    println!();
    println!("depth299_bits=897");
    println!("depth299_log10_space={:.6}", log10_space(299));
    println!("depth299_ns={:.6}", depth299_ns);

    println!();
    println!("depth85_vs_11_runtime={:.6}x", depth85_ns / depth11_ns);
    println!("depth299_vs_11_runtime={:.6}x", depth299_ns / depth11_ns);

    println!("checksum={checksum:016x}");

    println!();
    println!("Interpretation:");
    println!("  Every level consumes a generated 3-bit figure.");
    println!("  Runtime measures hierarchical traversal only.");
    println!("  The implicit possibility space is never materialized.");
    println!("  Linear T(depth) implies logarithmic traversal versus N=8^depth.");
    println!("  This is representational/traversal complexity.");
    println!("  It is NOT evidence of O(log N) arbitrary search.");
    println!("  It is NOT evidence of SHA prediction.");

    println!();

    if r2 >= 0.98 {
        println!("LIVE-09F ULTRA-DEEP SCALING CHECK: PASS");
    } else {
        println!("LIVE-09F ULTRA-DEEP SCALING CHECK: REVIEW");
    }

    black_box(checksum);
}
