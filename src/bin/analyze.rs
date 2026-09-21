use csv::StringRecord;
use std::{
    error::Error,
    f64::consts::LN_2,
    path::Path,
};

const INPUT: &str = "data/bitcoin-blocks.csv";
const LEVELS: usize = 11;
const FULL_LEVELS: usize = 10;
const STATES: usize = 8;

#[derive(Debug)]
struct Sample {
    height: u64,
    nonce: u32,
    path: Vec<u8>,
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

    let verified_idx = headers
        .iter()
        .position(|h| h == "verified")
        .ok_or("missing verified column")?;

    let mut level_indices = Vec::new();

    for level in 1..=LEVELS {
        let name = format!("x1331_l{}", level);

        let idx = headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("missing column {}", name))?;

        level_indices.push(idx);
    }

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

        let mut path = Vec::with_capacity(LEVELS);

        for idx in &level_indices {
            let value = record
                .get(*idx)
                .ok_or("missing X1331 state")?;

            path.push(parse_binary_state(value)?);
        }

        samples.push(Sample {
            height,
            nonce,
            path,
        });
    }

    samples.sort_by_key(|s| s.height);

    Ok(samples)
}

fn entropy(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();

    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;

    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_f;
            -p * (p.ln() / LN_2)
        })
        .sum()
}

fn chi_square_uniform(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();

    if total == 0 || counts.is_empty() {
        return 0.0;
    }

    let expected = total as f64 / counts.len() as f64;

    counts
        .iter()
        .map(|&observed| {
            let diff = observed as f64 - expected;
            diff * diff / expected
        })
        .sum()
}

// Survival function for chi-square distribution.
// Uses regularized upper incomplete gamma Q(a,x).
fn chi_square_p_value(chi2: f64, df: f64) -> f64 {
    gamma_q(df / 2.0, chi2 / 2.0)
}

fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }

    if x == 0.0 {
        return 1.0;
    }

    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_continued_fraction(a, x)
    }
}

fn gamma_p_series(a: f64, x: f64) -> f64 {
    const ITMAX: usize = 1000;
    const EPS: f64 = 3.0e-14;

    let gln = ln_gamma(a);

    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;

    for _ in 0..ITMAX {
        ap += 1.0;
        del *= x / ap;
        sum += del;

        if del.abs() < sum.abs() * EPS {
            break;
        }
    }

    sum * (-x + a * x.ln() - gln).exp()
}

fn gamma_q_continued_fraction(a: f64, x: f64) -> f64 {
    const ITMAX: usize = 1000;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-300;

    let gln = ln_gamma(a);

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b.max(FPMIN);
    let mut h = d;

    for i in 1..=ITMAX {
        let an = -(i as f64) * ((i as f64) - a);

        b += 2.0;

        d = an * d + b;

        if d.abs() < FPMIN {
            d = FPMIN;
        }

        c = b + an / c;

        if c.abs() < FPMIN {
            c = FPMIN;
        }

        d = 1.0 / d;

        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPS {
            break;
        }
    }

    (-x + a * x.ln() - gln).exp() * h
}

// Lanczos approximation.
fn ln_gamma(z: f64) -> f64 {
    let coefficients = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }

    let z = z - 1.0;

    let mut x = 0.99999999999980993;

    for (i, c) in coefficients.iter().enumerate() {
        x += c / (z + i as f64 + 1.0);
    }

    let t = z + coefficients.len() as f64 - 0.5;

    0.5 * (2.0 * std::f64::consts::PI).ln()
        + (z + 0.5) * t.ln()
        - t
        + x.ln()
}

fn hamming_weight(state: u8) -> usize {
    state.count_ones() as usize
}

fn print_level_distribution(samples: &[Sample]) {
    println!("X1331 STATE DISTRIBUTIONS");
    println!("=========================");

    for level in 0..FULL_LEVELS {
        let mut counts = [0u64; STATES];

        for sample in samples {
            counts[sample.path[level] as usize] += 1;
        }

        let n = samples.len() as f64;
        let expected = n / 8.0;

        let chi2 = chi_square_uniform(&counts);
        let p = chi_square_p_value(chi2, 7.0);
        let h = entropy(&counts);

        println!();
        println!(
            "L{} | expected/state {:.3} | entropy {:.5}/3.00000 | chi2 {:.4} | p {:.6}",
            level + 1,
            expected,
            h,
            chi2,
            p
        );

        for state in 0..STATES {
            let pct = counts[state] as f64 / n * 100.0;

            println!(
                "  {:03b} : {:4}  {:7.3}%",
                state,
                counts[state],
                pct
            );
        }
    }

    // L11 only has the final two bits, therefore 4 states.
    let mut counts = [0u64; 4];

    for sample in samples {
        counts[sample.path[10] as usize] += 1;
    }

    let chi2 = chi_square_uniform(&counts);
    let p = chi_square_p_value(chi2, 3.0);
    let h = entropy(&counts);

    println!();
    println!(
        "L11 (terminal 2 bits) | expected/state {:.3} | entropy {:.5}/2.00000 | chi2 {:.4} | p {:.6}",
        samples.len() as f64 / 4.0,
        h,
        chi2,
        p
    );

    for state in 0..4 {
        println!(
            "  {:02b}  : {:4}  {:7.3}%",
            state,
            counts[state],
            counts[state] as f64 / samples.len() as f64 * 100.0
        );
    }
}

fn print_hamming_groups(samples: &[Sample]) {
    println!();
    println!("1 | 3 | 3 | 1 HAMMING GROUPS");
    println!("============================");

    let expected_pct = [12.5, 37.5, 37.5, 12.5];

    for level in 0..FULL_LEVELS {
        let mut groups = [0u64; 4];

        for sample in samples {
            let state = sample.path[level];
            groups[hamming_weight(state)] += 1;
        }

        println!();
        println!("L{}", level + 1);

        for weight in 0..4 {
            let pct =
                groups[weight] as f64
                    / samples.len() as f64
                    * 100.0;

            println!(
                "  weight {} | {:4} | {:7.3}% | expected {:5.1}%",
                weight,
                groups[weight],
                pct,
                expected_pct[weight]
            );
        }
    }
}

fn print_bit_distribution(samples: &[Sample]) {
    println!();
    println!("NONCE BIT DISTRIBUTION");
    println!("======================");

    let n = samples.len() as f64;

    let mut total_ones = 0u64;

    for bit in (0..32).rev() {
        let ones = samples
            .iter()
            .filter(|s| ((s.nonce >> bit) & 1) == 1)
            .count() as u64;

        total_ones += ones;

        let pct = ones as f64 / n * 100.0;

        println!(
            "bit {:02} | ones {:4} | {:7.3}%",
            bit,
            ones,
            pct
        );
    }

    let total_bits = samples.len() as u64 * 32;

    println!();
    println!(
        "Overall ones : {} / {} = {:.4}%",
        total_ones,
        total_bits,
        total_ones as f64 / total_bits as f64 * 100.0
    );
}

fn print_nonce_summary(samples: &[Sample]) {
    println!();
    println!("NONCE SUMMARY");
    println!("=============");

    let min = samples.iter().map(|s| s.nonce).min().unwrap();
    let max = samples.iter().map(|s| s.nonce).max().unwrap();

    let mean =
        samples
            .iter()
            .map(|s| s.nonce as f64)
            .sum::<f64>()
            / samples.len() as f64;

    let midpoint = u32::MAX as f64 / 2.0;

    println!("Minimum : {}", min);
    println!("Maximum : {}", max);
    println!("Mean    : {:.3}", mean);
    println!("Uniform midpoint expectation: {:.3}", midpoint);

    let lower_half =
        samples
            .iter()
            .filter(|s| s.nonce < 0x8000_0000)
            .count();

    println!(
        "Lower 32-bit half : {} ({:.3}%)",
        lower_half,
        lower_half as f64 / samples.len() as f64 * 100.0
    );

    println!(
        "Upper 32-bit half : {} ({:.3}%)",
        samples.len() - lower_half,
        (samples.len() - lower_half) as f64
            / samples.len() as f64
            * 100.0
    );
}

fn print_transitions(samples: &[Sample]) {
    println!();
    println!("L1 TEMPORAL TRANSITIONS");
    println!("=======================");

    if samples.len() < 2 {
        println!("Not enough samples.");
        return;
    }

    let mut matrix = [[0u64; STATES]; STATES];

    for pair in samples.windows(2) {
        let from = pair[0].path[0] as usize;
        let to = pair[1].path[0] as usize;

        matrix[from][to] += 1;
    }

    print!("from\\to");

    for state in 0..STATES {
        print!(" {:03b}", state);
    }

    println!();

    for from in 0..STATES {
        print!("{:03b}   ", from);

        for to in 0..STATES {
            print!(" {:3}", matrix[from][to]);
        }

        println!();
    }

    let same =
        samples
            .windows(2)
            .filter(|pair| pair[0].path[0] == pair[1].path[0])
            .count();

    let transitions = samples.len() - 1;

    println!();
    println!(
        "Same L1 state consecutively: {} / {} = {:.3}%",
        same,
        transitions,
        same as f64 / transitions as f64 * 100.0
    );

    println!("Uniform expectation ≈ 12.5%");
}

fn print_conditional_l2_given_l1(samples: &[Sample]) {
    println!();
    println!("CONDITIONAL DISTRIBUTION P(L2 | L1)");
    println!("===================================");

    let mut matrix = [[0u64; STATES]; STATES];

    for sample in samples {
        let l1 = sample.path[0] as usize;
        let l2 = sample.path[1] as usize;

        matrix[l1][l2] += 1;
    }

    for l1 in 0..STATES {
        let total: u64 = matrix[l1].iter().sum();

        println!();
        println!("L1={:03b} | n={}", l1, total);

        if total == 0 {
            continue;
        }

        for l2 in 0..STATES {
            println!(
                "  L2={:03b} : {:3} ({:7.3}%)",
                l2,
                matrix[l1][l2],
                matrix[l1][l2] as f64 / total as f64 * 100.0
            );
        }
    }
}

fn print_repeated_nonces(samples: &[Sample]) {
    use std::collections::HashMap;

    println!();
    println!("REPEATED NONCES");
    println!("===============");

    let mut counts: HashMap<u32, usize> = HashMap::new();

    for sample in samples {
        *counts.entry(sample.nonce).or_insert(0) += 1;
    }

    let mut repeated: Vec<(u32, usize)> =
        counts
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .collect();

    repeated.sort_by(|a, b| b.1.cmp(&a.1));

    if repeated.is_empty() {
        println!("No repeated nonce values.");
    } else {
        for (nonce, count) in repeated {
            println!("nonce {} repeated {} times", nonce, count);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let samples = load_samples()?;

    println!("X1331 Runtime v0.16");
    println!("==================");
    println!("Historical Winning Nonce Analyzer");
    println!("Offline statistical pilot");
    println!();

    if samples.is_empty() {
        return Err("no verified samples found".into());
    }

    println!("Dataset        : {}", INPUT);
    println!("Verified rows  : {}", samples.len());
    println!(
        "Height range   : {} .. {}",
        samples.first().unwrap().height,
        samples.last().unwrap().height
    );
    println!(
        "Expected L1/state under uniformity: {:.3}",
        samples.len() as f64 / 8.0
    );

    print_level_distribution(&samples);
    print_hamming_groups(&samples);
    print_nonce_summary(&samples);
    print_bit_distribution(&samples);
    print_transitions(&samples);
    print_conditional_l2_given_l1(&samples);
    print_repeated_nonces(&samples);

    println!();
    println!("INTERPRETATION RULE");
    println!("===================");
    println!("This is an exploratory pilot dataset.");
    println!("A deviation here is NOT evidence of SHA-256 predictability.");
    println!("Any candidate anomaly must be reproduced on a larger,");
    println!("chronologically separate dataset before modeling.");

    Ok(())
}
