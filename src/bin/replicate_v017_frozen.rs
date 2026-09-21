use csv::StringRecord;
use std::{
    error::Error,
    path::Path,
};

const INPUT: &str = "data/bitcoin-blocks.csv";

// Frozen discovery interval.
// NEVER count these rows as replication evidence.
const DISCOVERY_START: u64 = 890_000;
const DISCOVERY_END: u64 = 890_222;

// Frozen observation from v0.16.
const DISCOVERY_N: u64 = 223;
const DISCOVERY_BIT7_ONES: u64 = 72;

// Frozen hypothesis:
// H1: P(nonce bit 7 = 1) < 0.50
const NULL_P: f64 = 0.50;

#[derive(Debug)]
struct Sample {
    height: u64,
    nonce: u32,
    l9: u8,
}

fn parse_binary_state(s: &str) -> Result<u8, Box<dyn Error>> {
    Ok(u8::from_str_radix(s, 2)?)
}

fn load_samples() -> Result<Vec<Sample>, Box<dyn Error>> {
    if !Path::new(INPUT).exists() {
        return Err(format!("dataset not found: {}", INPUT).into());
    }

    let mut reader = csv::Reader::from_path(INPUT)?;
    let headers = reader.headers()?.clone();

    let height_idx = headers
        .iter()
        .position(|h| h == "height")
        .ok_or("missing height column")?;

    let nonce_idx = headers
        .iter()
        .position(|h| h == "nonce")
        .ok_or("missing nonce column")?;

    let l9_idx = headers
        .iter()
        .position(|h| h == "x1331_l9")
        .ok_or("missing x1331_l9 column")?;

    let verified_idx = headers
        .iter()
        .position(|h| h == "verified")
        .ok_or("missing verified column")?;

    let mut samples = Vec::new();

    for result in reader.records() {
        let record: StringRecord = result?;

        if record.get(verified_idx) != Some("true") {
            continue;
        }

        let height: u64 = record
            .get(height_idx)
            .ok_or("missing height")?
            .parse()?;

        let nonce: u32 = record
            .get(nonce_idx)
            .ok_or("missing nonce")?
            .parse()?;

        let l9 = parse_binary_state(
            record
                .get(l9_idx)
                .ok_or("missing L9 state")?,
        )?;

        samples.push(Sample {
            height,
            nonce,
            l9,
        });
    }

    samples.sort_by_key(|s| s.height);

    Ok(samples)
}

// Exact one-sided binomial lower-tail probability:
//
// P(X <= observed | n, p=0.5)
//
// Since p=0.5, calculate probabilities recursively starting
// from P(X=0)=2^-n.
//
// For large n where 2^-n underflows, fall back to a
// continuity-corrected normal approximation.
fn binomial_lower_tail(observed: u64, n: u64) -> f64 {
    if n == 0 {
        return 1.0;
    }

    if n <= 1023 {
        let mut probability = 2f64.powf(-(n as f64));
        let mut cumulative = probability;

        if observed == 0 {
            return cumulative;
        }

        for k in 0..observed {
            probability *=
                (n - k) as f64
                / (k + 1) as f64;

            cumulative += probability;
        }

        cumulative.min(1.0)
    } else {
        let mean = n as f64 * NULL_P;
        let variance = n as f64 * NULL_P * (1.0 - NULL_P);
        let sd = variance.sqrt();

        // continuity correction for P(X <= observed)
        let z =
            (observed as f64 + 0.5 - mean)
            / sd;

        normal_cdf(z)
    }
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / 2f64.sqrt()))
}

// Abramowitz & Stegun approximation.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * x);

    let y = 1.0
        - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1)
            * t)
            * (-x * x).exp();

    sign * y
}

fn bit7_is_one(nonce: u32) -> bool {
    ((nonce >> 7) & 1) == 1
}

fn l9_distribution(samples: &[&Sample]) -> [u64; 8] {
    let mut counts = [0u64; 8];

    for sample in samples {
        counts[sample.l9 as usize] += 1;
    }

    counts
}

fn entropy(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();

    if total == 0 {
        return 0.0;
    }

    let total = total as f64;

    counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

fn print_window(
    name: &str,
    start: u64,
    end: u64,
    samples: &[Sample],
) {
    let window: Vec<&Sample> = samples
        .iter()
        .filter(|s| s.height >= start && s.height <= end)
        .collect();

    println!();
    println!("{}", name);
    println!("{}", "=".repeat(name.len()));
    println!("Heights : {} .. {}", start, end);

    if window.is_empty() {
        println!("Status  : NO DATA YET");
        return;
    }

    let n = window.len() as u64;

    let first_height = window.first().unwrap().height;
    let last_height = window.last().unwrap().height;

    let ones = window
        .iter()
        .filter(|s| bit7_is_one(s.nonce))
        .count() as u64;

    let zeros = n - ones;

    let ones_pct =
        ones as f64 / n as f64 * 100.0;

    let zeros_pct =
        zeros as f64 / n as f64 * 100.0;

    let p_value =
        binomial_lower_tail(ones, n);

    let expected_ones = n as f64 * 0.5;

    let effect =
        ones_pct - 50.0;

    let l9 = l9_distribution(&window);
    let h = entropy(&l9);

    println!("Observed: {} .. {}", first_height, last_height);
    println!("N       : {}", n);

    if first_height != start || last_height != end {
        println!("Status  : PARTIAL WINDOW");
    } else if n != end - start + 1 {
        println!("Status  : WARNING - POSSIBLE HEIGHT GAPS");
    } else {
        println!("Status  : COMPLETE WINDOW");
    }

    println!();
    println!("FROZEN H1 TEST");
    println!("bit7=1 : {} ({:.4}%)", ones, ones_pct);
    println!("bit7=0 : {} ({:.4}%)", zeros, zeros_pct);
    println!("Expected ones under H0 : {:.3}", expected_ones);
    println!("Effect vs 50%          : {:+.4} percentage points", effect);
    println!("One-sided binomial p   : {:.10}", p_value);

    println!();
    println!("L9 DISTRIBUTION");

    for state in 0..8 {
        println!(
            "  {:03b} : {:4} ({:7.3}%)",
            state,
            l9[state],
            l9[state] as f64 / n as f64 * 100.0
        );
    }

    println!("L9 entropy: {:.6} / 3.000000", h);
}

fn print_combined_holdout(samples: &[Sample]) {
    let holdout: Vec<&Sample> = samples
        .iter()
        .filter(|s| s.height > DISCOVERY_END)
        .collect();

    println!();
    println!("COMBINED HOLDOUT");
    println!("================");

    if holdout.is_empty() {
        println!("No post-discovery data yet.");
        return;
    }

    let n = holdout.len() as u64;

    let ones = holdout
        .iter()
        .filter(|s| bit7_is_one(s.nonce))
        .count() as u64;

    let zeros = n - ones;

    let pct =
        ones as f64 / n as f64 * 100.0;

    let p =
        binomial_lower_tail(ones, n);

    println!(
        "Range     : {} .. {}",
        holdout.first().unwrap().height,
        holdout.last().unwrap().height
    );

    println!("N         : {}", n);
    println!("bit7 ones : {} ({:.4}%)", ones, pct);
    println!(
        "bit7 zeros: {} ({:.4}%)",
        zeros,
        zeros as f64 / n as f64 * 100.0
    );
    println!("H0        : 50.0000%");
    println!("Effect    : {:+.4} percentage points", pct - 50.0);
    println!("p(one-sided): {:.10}", p);

    println!();
    println!("IMPORTANT:");
    println!("Discovery rows 890000..890222 are excluded.");
}

fn verify_frozen_discovery(samples: &[Sample]) -> Result<(), Box<dyn Error>> {
    let discovery: Vec<&Sample> = samples
        .iter()
        .filter(|s| {
            s.height >= DISCOVERY_START
                && s.height <= DISCOVERY_END
        })
        .collect();

    if discovery.len() as u64 != DISCOVERY_N {
        return Err(
            format!(
                "frozen discovery count changed: expected {}, found {}",
                DISCOVERY_N,
                discovery.len()
            )
            .into()
        );
    }

    let ones = discovery
        .iter()
        .filter(|s| bit7_is_one(s.nonce))
        .count() as u64;

    if ones != DISCOVERY_BIT7_ONES {
        return Err(
            format!(
                "frozen discovery result changed: expected {} bit7 ones, found {}",
                DISCOVERY_BIT7_ONES,
                ones
            )
            .into()
        );
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let samples = load_samples()?;

    if samples.is_empty() {
        return Err("no verified samples found".into());
    }

    verify_frozen_discovery(&samples)?;

    println!("X1331 Runtime v0.17");
    println!("==================");
    println!("Frozen Historical Replication Lab");
    println!();

    println!("Dataset : {}", INPUT);
    println!("Rows    : {}", samples.len());
    println!();

    println!("FROZEN DISCOVERY");
    println!("================");
    println!(
        "Heights    : {} .. {}",
        DISCOVERY_START,
        DISCOVERY_END
    );
    println!("N          : {}", DISCOVERY_N);
    println!(
        "bit7 ones  : {} ({:.4}%)",
        DISCOVERY_BIT7_ONES,
        DISCOVERY_BIT7_ONES as f64
            / DISCOVERY_N as f64
            * 100.0
    );
    println!("H1         : P(bit7=1) < 0.50");
    println!("Role       : DISCOVERY ONLY");
    println!("Replication: EXCLUDED");

    // First partial replication interval deliberately starts
    // immediately after discovery.
    print_window(
        "REPLICATION A",
        890_223,
        890_999,
        &samples,
    );

    print_window(
        "REPLICATION B",
        891_000,
        891_999,
        &samples,
    );

    print_window(
        "REPLICATION C",
        892_000,
        892_999,
        &samples,
    );

    print_window(
        "REPLICATION D",
        893_000,
        893_999,
        &samples,
    );

    print_window(
        "REPLICATION E",
        894_000,
        894_999,
        &samples,
    );

    print_window(
        "REPLICATION F",
        895_000,
        895_999,
        &samples,
    );

    print_window(
        "REPLICATION G",
        896_000,
        896_999,
        &samples,
    );

    print_window(
        "REPLICATION H",
        897_000,
        897_999,
        &samples,
    );

    print_window(
        "REPLICATION I",
        898_000,
        898_999,
        &samples,
    );

    print_window(
        "REPLICATION J",
        899_000,
        899_999,
        &samples,
    );

    print_combined_holdout(&samples);

    println!();
    println!("DECISION RULE");
    println!("=============");
    println!("The hypothesis is frozen before replication data.");
    println!("Do not modify bit position, direction, or discovery interval");
    println!("after observing future blocks.");
    println!();
    println!("A replicated deviation would establish only a historical");
    println!("winning-nonce distribution effect. It would NOT by itself");
    println!("establish SHA-256 predictability or profitable mining.");

    Ok(())
}
