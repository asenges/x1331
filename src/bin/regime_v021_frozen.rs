use csv::Reader;
use std::{
    error::Error,
    f64::consts::SQRT_2,
};

const INPUT: &str = "data/bitcoin-history-v020-frozen.csv";

const START: u64 = 700_000;
const END: u64 = 899_999;
const EXPECTED_ROWS: usize = 200_000;
const WINDOW: u64 = 10_000;

#[derive(Debug)]
struct Row {
    height: u64,
    nonce: u32,
    verified: bool,
}

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
            * t
            * (-x * x).exp());

    sign * y
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / SQRT_2))
}

fn wilson_interval(successes: u64, n: u64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }

    let z = 1.959963984540054;
    let nf = n as f64;
    let p = successes as f64 / nf;

    let denominator = 1.0 + z * z / nf;

    let center =
        (p + z * z / (2.0 * nf)) / denominator;

    let margin =
        z
            * ((p * (1.0 - p) / nf
                + z * z / (4.0 * nf * nf))
                .sqrt())
            / denominator;

    (center - margin, center + margin)
}

fn lower_tail_p(successes: u64, n: u64) -> f64 {
    if n == 0 {
        return 1.0;
    }

    let mean = n as f64 * 0.5;
    let sd = (n as f64 * 0.25).sqrt();

    // continuity correction
    let z = (successes as f64 + 0.5 - mean) / sd;

    normal_cdf(z)
}

fn bit(nonce: u32, bit: usize) -> u8 {
    ((nonce >> bit) & 1) as u8
}

fn x1331_state(nonce: u32, level: usize) -> usize {
    // Levels 1..10 consume bits:
    // 31..29, 28..26, ... 4..2
    let shift = 29 - level * 3;
    ((nonce >> shift) & 7) as usize
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 v0.21 Regime Mapping Laboratory");
    println!("=====================================");
    println!("Input       : {}", INPUT);
    println!("Range       : {} .. {}", START, END);
    println!("Rows        : {}", EXPECTED_ROWS);
    println!("Window      : {} blocks", WINDOW);
    println!();

    let mut reader = Reader::from_path(INPUT)?;

    let headers = reader.headers()?.clone();

    let height_idx = headers
        .iter()
        .position(|h| h == "height")
        .ok_or("height column missing")?;

    let nonce_idx = headers
        .iter()
        .position(|h| h == "nonce")
        .ok_or("nonce column missing")?;

    let verified_idx = headers
        .iter()
        .position(|h| h == "verified")
        .ok_or("verified column missing")?;

    let mut rows = Vec::<Row>::with_capacity(EXPECTED_ROWS);

    for record in reader.records() {
        let record = record?;

        let height: u64 = record[height_idx].parse()?;
        let nonce: u32 = record[nonce_idx].parse()?;

        let verified = matches!(
            record[verified_idx].to_ascii_lowercase().as_str(),
            "true" | "1" | "yes"
        );

        rows.push(Row {
            height,
            nonce,
            verified,
        });
    }

    println!("DATASET VALIDATION");
    println!("------------------");

    if rows.len() != EXPECTED_ROWS {
        return Err(
            format!(
                "expected {} rows, got {}",
                EXPECTED_ROWS,
                rows.len()
            )
            .into()
        );
    }

    if rows.first().map(|r| r.height) != Some(START) {
        return Err("unexpected first height".into());
    }

    if rows.last().map(|r| r.height) != Some(END) {
        return Err("unexpected final height".into());
    }

    for (i, row) in rows.iter().enumerate() {
        let expected = START + i as u64;

        if row.height != expected {
            return Err(
                format!(
                    "non-consecutive height: expected {}, got {}",
                    expected,
                    row.height
                )
                .into()
            );
        }

        if !row.verified {
            return Err(
                format!(
                    "row {} is not verified",
                    row.height
                )
                .into()
            );
        }
    }

    println!("Rows        : {}", rows.len());
    println!("First       : {}", rows[0].height);
    println!("Last        : {}", rows[rows.len() - 1].height);
    println!("Consecutive : YES");
    println!("Verified    : ALL");
    println!();

    println!("FROZEN PRIMARY HYPOTHESIS");
    println!("-------------------------");
    println!("H1: P(winning nonce bit7 = 1) < 0.50");
    println!("Discovery interval 890000..890222 is NOT redefined.");
    println!();

    println!("BIT7 — FIXED 10,000-BLOCK WINDOWS");
    println!("---------------------------------");
    println!(
        "{:<19} {:>7} {:>7} {:>10} {:>11} {:>23} {:>14}",
        "Window",
        "N",
        "ones",
        "ones%",
        "effect(pp)",
        "Wilson95%",
        "lower-p"
    );

    let mut start = START;

    while start <= END {
        let stop = start + WINDOW - 1;

        let slice: Vec<&Row> = rows
            .iter()
            .filter(|r| r.height >= start && r.height <= stop)
            .collect();

        let n = slice.len() as u64;

        let ones = slice
            .iter()
            .filter(|r| bit(r.nonce, 7) == 1)
            .count() as u64;

        let rate = ones as f64 / n as f64;
        let effect = (rate - 0.5) * 100.0;

        let (lo, hi) = wilson_interval(ones, n);
        let p = lower_tail_p(ones, n);

        println!(
            "{}..{} {:>7} {:>7} {:>9.4}% {:>+10.4} [{:>7.3}%, {:>7.3}%] {:>14.6e}",
            start,
            stop,
            n,
            ones,
            rate * 100.0,
            effect,
            lo * 100.0,
            hi * 100.0,
            p
        );

        start += WINDOW;
    }

    println!();

    // --------------------------------------------------------
    // PRE-890K CHARACTERIZATION
    // --------------------------------------------------------

    let pre: Vec<&Row> = rows
        .iter()
        .filter(|r| r.height <= 889_999)
        .collect();

    let pre_n = pre.len() as u64;

    let pre_ones = pre
        .iter()
        .filter(|r| bit(r.nonce, 7) == 1)
        .count() as u64;

    let pre_rate = pre_ones as f64 / pre_n as f64;
    let (pre_lo, pre_hi) = wilson_interval(pre_ones, pre_n);

    println!("PRE-890000 RETROSPECTIVE CHARACTERIZATION");
    println!("-----------------------------------------");
    println!("Range       : 700000 .. 889999");
    println!("N           : {}", pre_n);
    println!("bit7 ones   : {}", pre_ones);
    println!("bit7 ones % : {:.6}%", pre_rate * 100.0);
    println!(
        "Wilson 95%  : [{:.6}%, {:.6}%]",
        pre_lo * 100.0,
        pre_hi * 100.0
    );
    println!(
        "Effect      : {:+.6} percentage points vs 50%",
        (pre_rate - 0.5) * 100.0
    );
    println!(
        "NOTE        : retrospective characterization, not a pristine prospective holdout."
    );
    println!();

    // --------------------------------------------------------
    // KNOWN 890K INTEGRITY CHECK
    // --------------------------------------------------------

    let known: Vec<&Row> = rows
        .iter()
        .filter(|r| r.height >= 890_000)
        .collect();

    let known_ones = known
        .iter()
        .filter(|r| bit(r.nonce, 7) == 1)
        .count() as u64;

    println!("V0.19 CONSISTENCY CHECK");
    println!("-----------------------");
    println!("Range       : 890000 .. 899999");
    println!("N           : {}", known.len());
    println!("bit7 ones   : {}", known_ones);
    println!(
        "bit7 ones % : {:.4}%",
        known_ones as f64 / known.len() as f64 * 100.0
    );
    println!("Expected    : 3470 / 10000 = 34.7000%");

    if known.len() != 10_000 || known_ones != 3470 {
        return Err(
            format!(
                "v0.19 consistency check FAILED: got {}/{}",
                known_ones,
                known.len()
            )
            .into()
        );
    }

    println!("Integrity   : PASS");
    println!();

    // --------------------------------------------------------
    // ALL 32 NONCE BITS
    // --------------------------------------------------------

    println!("32-BIT NONCE MAP — FULL 700000..899999");
    println!("--------------------------------------");
    println!(
        "{:<6} {:>10} {:>12} {:>12}",
        "bit",
        "ones",
        "ones%",
        "effect(pp)"
    );

    for b in (0..32).rev() {
        let ones = rows
            .iter()
            .filter(|r| bit(r.nonce, b) == 1)
            .count();

        let rate = ones as f64 / rows.len() as f64;

        println!(
            "{:<6} {:>10} {:>11.5}% {:>+11.5}",
            b,
            ones,
            rate * 100.0,
            (rate - 0.5) * 100.0
        );
    }

    println!();

    // --------------------------------------------------------
    // 32 BITS × 20 WINDOWS
    // --------------------------------------------------------

    println!("32-BIT TEMPORAL MAP");
    println!("-------------------");
    println!("Each value = percentage of ones in fixed 10k window.");
    println!();

    print!("{:<6}", "bit");

    let mut ws = START;

    while ws <= END {
        print!(" {:>7}", ws / 1000);
        ws += WINDOW;
    }

    println!();

    for b in (0..32).rev() {
        print!("{:<6}", b);

        let mut ws = START;

        while ws <= END {
            let we = ws + WINDOW - 1;

            let mut n = 0usize;
            let mut ones = 0usize;

            for row in &rows {
                if row.height >= ws && row.height <= we {
                    n += 1;

                    if bit(row.nonce, b) == 1 {
                        ones += 1;
                    }
                }
            }

            let pct = ones as f64 / n as f64 * 100.0;

            print!(" {:>6.2}%", pct);

            ws += WINDOW;
        }

        println!();
    }

    println!();

    // --------------------------------------------------------
    // X1331 STATE DISTRIBUTION
    // --------------------------------------------------------

    println!("X1331 LEVEL DISTRIBUTIONS — FULL DATASET");
    println!("----------------------------------------");

    for level in 0..10 {
        let mut counts = [0u64; 8];

        for row in &rows {
            counts[x1331_state(row.nonce, level)] += 1;
        }

        println!(
            "L{:02} bits {:02}..{:02}",
            level + 1,
            31 - level * 3,
            29 - level * 3
        );

        for state in 0..8 {
            println!(
                "  {:03b}: {:>7}  {:>8.4}%",
                state,
                counts[state],
                counts[state] as f64
                    / rows.len() as f64
                    * 100.0
            );
        }
    }

    println!();
    println!("REGIME MAPPING COMPLETE");
    println!("=======================");
    println!("No ML training performed.");
    println!("No adaptive bit selection performed.");
    println!("H1 remained frozen.");
    println!("Historical distribution != SHA256 predictability.");

    Ok(())
}
