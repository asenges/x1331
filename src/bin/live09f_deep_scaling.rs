use std::hint::black_box;
use std::time::Instant;

const MAX_DEPTH: usize = 11;
const TRIALS: u64 = 5_000_000;

#[derive(Clone, Copy)]
struct X1331Node {
    state: u8,
}

impl X1331Node {
    #[inline(always)]
    fn from_bits(value: u64, level: usize) -> Self {
        let shift = level * 3;
        Self {
            state: ((value >> shift) & 0b111) as u8,
        }
    }

    #[inline(always)]
    fn layer(self) -> u8 {
        self.state.count_ones() as u8
    }

    #[inline(always)]
    fn interfere(self, accumulator: u64, level: usize) -> u64 {
        let s = self.state as u64;
        let layer = self.layer() as u64;

        accumulator
            .rotate_left(((s + level as u64) & 63) as u32)
            .wrapping_add(
                s.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(layer << (level & 31)),
            )
            ^ ((level as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03))
    }
}

#[inline(never)]
fn navigate(value: u64, depth: usize) -> u64 {
    let mut acc = 0x1331_1331_1331_1331u64;

    for level in 0..depth {
        let node = X1331Node::from_bits(value, level);
        acc = node.interfere(acc, level);
    }

    acc
}

#[inline(always)]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    println!("============================================================");
    println!(" X1331 LIVE-09F — DEEP SCALING LAB");
    println!(" 8^d implicit possibility space");
    println!(" No flat materialization of N");
    println!(" Objective: depth 11 = 2^33 possibilities");
    println!("============================================================");
    println!("trials_per_depth={TRIALS}");
    println!();

    println!(
        "{:<5} {:>16} {:>6} {:>14} {:>14} {:>14}",
        "DEPTH", "N=8^d", "BITS", "TOTAL_MS", "NS/NAV", "NS/LEVEL"
    );

    let mut seed = 0x1331_09F0_2026_0923u64;
    let mut results = Vec::new();
    let mut global_checksum = 0u64;

    for depth in 1..=MAX_DEPTH {
        let bits = depth * 3;
        let n: u128 = 1u128 << bits;

        // Warm-up.
        let mut warm_checksum = 0u64;
        for _ in 0..100_000u64 {
            let value = splitmix64(&mut seed);
            warm_checksum ^= navigate(black_box(value), black_box(depth));
        }
        black_box(warm_checksum);

        let start = Instant::now();
        let mut checksum = 0u64;

        for _ in 0..TRIALS {
            let value = splitmix64(&mut seed);
            checksum ^= navigate(black_box(value), black_box(depth));
        }

        let elapsed = start.elapsed();
        let elapsed_ns = elapsed.as_nanos() as f64;
        let ns_per_nav = elapsed_ns / TRIALS as f64;
        let ns_per_level = ns_per_nav / depth as f64;

        global_checksum ^= checksum;

        println!(
            "{:<5} {:>16} {:>6} {:>14.3} {:>14.3} {:>14.3}",
            depth,
            n,
            bits,
            elapsed.as_secs_f64() * 1000.0,
            ns_per_nav,
            ns_per_level
        );

        results.push((depth as f64, n as f64, ns_per_nav));
    }

    // Fit T = a*d + b.
    let count = results.len() as f64;
    let sum_d: f64 = results.iter().map(|r| r.0).sum();
    let sum_t: f64 = results.iter().map(|r| r.2).sum();
    let sum_dd: f64 = results.iter().map(|r| r.0 * r.0).sum();
    let sum_dt: f64 = results.iter().map(|r| r.0 * r.2).sum();

    let denominator = count * sum_dd - sum_d * sum_d;
    let slope = (count * sum_dt - sum_d * sum_t) / denominator;
    let intercept = (sum_t - slope * sum_d) / count;

    let mean_t = sum_t / count;
    let ss_tot: f64 = results
        .iter()
        .map(|r| {
            let x = r.2 - mean_t;
            x * x
        })
        .sum();

    let ss_res: f64 = results
        .iter()
        .map(|r| {
            let predicted = intercept + slope * r.0;
            let x = r.2 - predicted;
            x * x
        })
        .sum();

    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    let depth11 = results.last().unwrap().2;
    let depth1 = results.first().unwrap().2;

    let n1 = 8.0_f64;
    let n11 = 8_589_934_592.0_f64;

    let space_growth = n11 / n1;
    let runtime_growth = depth11 / depth1;

    println!();
    println!("============================================================");
    println!(" LIVE-09F FINAL");
    println!("============================================================");
    println!("max_depth={MAX_DEPTH}");
    println!("max_bits={}", MAX_DEPTH * 3);
    println!("max_implicit_space=8589934592");
    println!("linear_fit_ns = {:.6} * depth + {:.6}", slope, intercept);
    println!("linear_fit_r2={:.9}", r2);
    println!("depth1_ns={:.6}", depth1);
    println!("depth11_ns={:.6}", depth11);
    println!("implicit_space_growth={:.3}x", space_growth);
    println!("runtime_growth={:.6}x", runtime_growth);
    println!("global_checksum={global_checksum:016x}");
    println!();

    println!("Interpretation:");
    println!("  N is an implicit addressable X1331 space.");
    println!("  The benchmark does NOT enumerate N possibilities.");
    println!("  It measures the cost of traversing d local 8-state levels.");
    println!("  Since N=8^d, linear runtime in d corresponds to");
    println!("  logarithmic traversal cost with respect to implicit N.");
    println!("  This does NOT prove arbitrary search can be solved in O(log N).");
    println!("  Correct branch selection requires predictive information.");
    println!();

    if r2 >= 0.95 && runtime_growth < 20.0 {
        println!("LIVE-09F SCALING CHECK: PASS");
    } else {
        println!("LIVE-09F SCALING CHECK: REVIEW");
    }

    black_box(global_checksum);
}
