use csv::Reader;
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    time::Instant,
};

const INPUT: &str =
    "data/bitcoin-history-v020-frozen.csv";

const START: u64 = 700_000;
const END: u64 = 899_999;
const EXPECTED_ROWS: usize = 200_000;

const HEADER_STEP: usize = 100;
const EXPECTED_HEADERS: usize = 2_000;

const STATES: usize = 8;
const SAMPLES_PER_STATE: usize = 256;

const TOTAL_HASHES: usize =
    EXPECTED_HEADERS * STATES * SAMPLES_PER_STATE;

// Fixed before execution.
// Used only for deterministic reproducibility.
const RNG_SEED: u64 = 0x1331_2026_0921_A55A;

#[derive(Clone)]
struct BlockRow {
    height: u64,
    header: [u8; 80],
    verified: bool,
}

#[derive(Clone, Default)]
struct StateStats {
    n: u64,

    lz4: u64,
    lz8: u64,
    lz12: u64,
    lz16: u64,

    // Uniform [0,1)-like score constructed
    // from the first 64 bits of SHA256d.
    // Lower is a smaller hash prefix.
    sum_u64_norm: f64,
    sum_u64_norm_sq: f64,

    leading_zero_sum: u64,
}

#[derive(Clone)]
struct SplitStats {
    states: [StateStats; STATES],
}

impl SplitStats {
    fn new() -> Self {
        Self {
            states: std::array::from_fn(|_| {
                StateStats::default()
            }),
        }
    }
}

// ============================================================
// DETERMINISTIC PRNG
// ============================================================

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state =
            self.state.wrapping_add(
                0x9E3779B97F4A7C15,
            );

        let mut z = self.state;

        z = (z ^ (z >> 30))
            .wrapping_mul(
                0xBF58476D1CE4E5B9,
            );

        z = (z ^ (z >> 27))
            .wrapping_mul(
                0x94D049BB133111EB,
            );

        z ^ (z >> 31)
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

// ============================================================
// HASH
// ============================================================

fn sha256d(data: &[u8]) -> [u8; 32] {
    let first = Sha256::digest(data);
    let second = Sha256::digest(first);

    let mut out = [0u8; 32];
    out.copy_from_slice(&second);
    out
}

fn leading_zero_bits(hash: &[u8; 32]) -> u32 {
    let mut total = 0u32;

    for &byte in hash {
        if byte == 0 {
            total += 8;
        } else {
            total += byte.leading_zeros();
            break;
        }
    }

    total
}

// SHA digest bytes are treated here as a
// big-endian bit string for the continuous
// comparison. Under ideal SHA256 output this
// is uniform regardless of Bitcoin display
// endianness.
fn normalized_prefix(hash: &[u8; 32]) -> f64 {
    let x = u64::from_be_bytes([
        hash[0],
        hash[1],
        hash[2],
        hash[3],
        hash[4],
        hash[5],
        hash[6],
        hash[7],
    ]);

    x as f64 / (u64::MAX as f64 + 1.0)
}

// ============================================================
// NONCE CONSTRUCTION
// ============================================================

// L9 corresponds to nonce bits 7,6,5.
//
// Clear those three bits and inject the
// requested state. Every state receives
// exactly the same sample count.
fn force_l9_state(
    raw: u32,
    state: usize,
) -> u32 {
    let mask = !(0b111u32 << 5);

    (raw & mask)
        | ((state as u32 & 0b111) << 5)
}

fn l9_state(nonce: u32) -> usize {
    ((nonce >> 5) & 0b111) as usize
}

// ============================================================
// STATS
// ============================================================

fn update_stats(
    stats: &mut StateStats,
    hash: &[u8; 32],
) {
    let lz = leading_zero_bits(hash);
    let x = normalized_prefix(hash);

    stats.n += 1;

    if lz >= 4 {
        stats.lz4 += 1;
    }

    if lz >= 8 {
        stats.lz8 += 1;
    }

    if lz >= 12 {
        stats.lz12 += 1;
    }

    if lz >= 16 {
        stats.lz16 += 1;
    }

    stats.leading_zero_sum += lz as u64;

    stats.sum_u64_norm += x;
    stats.sum_u64_norm_sq += x * x;
}

fn pct(count: u64, n: u64) -> f64 {
    if n == 0 {
        0.0
    } else {
        count as f64 / n as f64 * 100.0
    }
}

fn mean(stats: &StateStats) -> f64 {
    stats.sum_u64_norm / stats.n as f64
}

fn variance(stats: &StateStats) -> f64 {
    let n = stats.n as f64;

    let m = stats.sum_u64_norm / n;

    let raw =
        stats.sum_u64_norm_sq / n
        - m * m;

    raw.max(0.0)
}

fn standard_error(stats: &StateStats) -> f64 {
    (variance(stats) / stats.n as f64).sqrt()
}

fn mean_lz(stats: &StateStats) -> f64 {
    stats.leading_zero_sum as f64
        / stats.n as f64
}

fn aggregate(
    split: &SplitStats,
) -> StateStats {
    let mut out = StateStats::default();

    for s in &split.states {
        out.n += s.n;

        out.lz4 += s.lz4;
        out.lz8 += s.lz8;
        out.lz12 += s.lz12;
        out.lz16 += s.lz16;

        out.sum_u64_norm +=
            s.sum_u64_norm;

        out.sum_u64_norm_sq +=
            s.sum_u64_norm_sq;

        out.leading_zero_sum +=
            s.leading_zero_sum;
    }

    out
}

fn print_split(
    name: &str,
    split: &SplitStats,
) {
    println!("{}", name);
    println!(
        "{}",
        "-".repeat(name.len())
    );

    println!(
        "{:<5} {:>10} {:>11} {:>11} {:>10} {:>10} {:>10} {:>10}",
        "L9",
        "N",
        "MeanHash",
        "SE",
        "MeanLZ",
        "LZ>=4",
        "LZ>=8",
        "LZ>=12"
    );

    for state in 0..STATES {
        let s = &split.states[state];

        println!(
            "{:03b}   {:>10} {:>11.8} {:>11.8} {:>10.6} {:>9.5}% {:>9.5}% {:>9.5}%",
            state,
            s.n,
            mean(s),
            standard_error(s),
            mean_lz(s),
            pct(s.lz4, s.n),
            pct(s.lz8, s.n),
            pct(s.lz12, s.n),
        );
    }

    println!();

    println!(
        "{:<5} {:>12}",
        "L9",
        "LZ>=16"
    );

    for state in 0..STATES {
        let s = &split.states[state];

        println!(
            "{:03b}   {:>11.6}%",
            state,
            pct(s.lz16, s.n)
        );
    }

    let all = aggregate(split);

    println!();
    println!(
        "Aggregate mean hash : {:.9}",
        mean(&all)
    );
    println!(
        "Aggregate mean LZ   : {:.9}",
        mean_lz(&all)
    );
    println!(
        "Aggregate LZ>=8     : {:.6}%",
        pct(all.lz8, all.n)
    );
    println!();
}

// ============================================================
// MAIN
// ============================================================

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 v0.25 Counterfactual SHA Laboratory");
    println!("=========================================");
    println!("Input              : {}", INPUT);
    println!(
        "Historical range   : {}..{}",
        START, END
    );
    println!(
        "Header step        : {}",
        HEADER_STEP
    );
    println!(
        "Headers sampled    : {}",
        EXPECTED_HEADERS
    );
    println!(
        "Samples/state/head : {}",
        SAMPLES_PER_STATE
    );
    println!(
        "SHA256d evaluations: {}",
        TOTAL_HASHES
    );
    println!(
        "PRNG seed          : 0x{:016X}",
        RNG_SEED
    );
    println!("L9                  : nonce bits 7..5");
    println!("Budget/state        : EQUAL");
    println!("Winning nonce       : NOT USED");
    println!();

    // ========================================================
    // LOAD
    // ========================================================

    let mut reader =
        Reader::from_path(INPUT)?;

    let headers =
        reader.headers()?.clone();

    let height_idx =
        headers
            .iter()
            .position(|h| h == "height")
            .ok_or("height missing")?;

    let header_idx =
        headers
            .iter()
            .position(|h| h == "header_hex")
            .ok_or("header_hex missing")?;

    let verified_idx =
        headers
            .iter()
            .position(|h| h == "verified")
            .ok_or("verified missing")?;

    let mut selected =
        Vec::<BlockRow>::with_capacity(
            EXPECTED_HEADERS
        );

    let mut row_index = 0usize;

    for result in reader.records() {
        let record = result?;

        let height: u64 =
            record[height_idx].parse()?;

        let verified = matches!(
            record[verified_idx]
                .to_ascii_lowercase()
                .as_str(),
            "true" | "1" | "yes"
        );

        if height != START + row_index as u64 {
            return Err(
                format!(
                    "non-consecutive dataset at row {}",
                    row_index
                )
                .into(),
            );
        }

        if !verified {
            return Err(
                format!(
                    "unverified block {}",
                    height
                )
                .into(),
            );
        }

        if row_index % HEADER_STEP == 0 {
            let bytes =
                hex::decode(&record[header_idx])?;

            if bytes.len() != 80 {
                return Err(
                    format!(
                        "header {} has {} bytes",
                        height,
                        bytes.len()
                    )
                    .into(),
                );
            }

            let mut header = [0u8; 80];
            header.copy_from_slice(&bytes);

            selected.push(BlockRow {
                height,
                header,
                verified,
            });
        }

        row_index += 1;
    }

    if row_index != EXPECTED_ROWS {
        return Err(
            format!(
                "expected {} rows, got {}",
                EXPECTED_ROWS,
                row_index
            )
            .into(),
        );
    }

    if selected.len() != EXPECTED_HEADERS {
        return Err(
            format!(
                "expected {} sampled headers, got {}",
                EXPECTED_HEADERS,
                selected.len()
            )
            .into(),
        );
    }

    if selected[0].height != START {
        return Err("first sampled height mismatch".into());
    }

    if selected.last().unwrap().height != 899_900 {
        return Err(
            "last sampled height mismatch".into()
        );
    }

    println!("DATASET VALIDATION");
    println!("------------------");
    println!("Rows            : {}", row_index);
    println!(
        "Sampled headers : {}",
        selected.len()
    );
    println!(
        "First header    : {}",
        selected[0].height
    );
    println!(
        "Last header     : {}",
        selected.last().unwrap().height
    );
    println!("Verified        : ALL");
    println!();

    // ========================================================
    // SPLIT
    //
    // First 1000 sampled headers = A
    // Last  1000 sampled headers = B
    //
    // We do not select an interesting state on A
    // and redefine a hypothesis. Both are reported.
    // ========================================================

    let mut split_a = SplitStats::new();
    let mut split_b = SplitStats::new();
    let mut combined = SplitStats::new();

    let mut rng =
        SplitMix64::new(RNG_SEED);

    let started = Instant::now();

    let mut hashes_done = 0usize;

    // ========================================================
    // COUNTERFACTUAL HASHING
    // ========================================================

    for (header_index, block) in
        selected.iter().enumerate()
    {
        if !block.verified {
            return Err(
                "internal verified failure".into()
            );
        }

        for state in 0..STATES {
            for _ in 0..SAMPLES_PER_STATE {
                let raw = rng.next_u32();

                let nonce =
                    force_l9_state(
                        raw,
                        state
                    );

                if l9_state(nonce) != state {
                    return Err(
                        "L9 construction failure".into()
                    );
                }

                let mut candidate =
                    block.header;

                candidate[76..80]
                    .copy_from_slice(
                        &nonce.to_le_bytes()
                    );

                let hash =
                    sha256d(&candidate);

                update_stats(
                    &mut combined.states[state],
                    &hash,
                );

                if header_index
                    < EXPECTED_HEADERS / 2
                {
                    update_stats(
                        &mut split_a.states[state],
                        &hash,
                    );
                } else {
                    update_stats(
                        &mut split_b.states[state],
                        &hash,
                    );
                }

                hashes_done += 1;
            }
        }

        if (header_index + 1) % 250 == 0 {
            println!(
                "Progress: {:>4}/{} headers  {:>8} SHA256d",
                header_index + 1,
                EXPECTED_HEADERS,
                hashes_done
            );
        }
    }

    let elapsed = started.elapsed();

    if hashes_done != TOTAL_HASHES {
        return Err(
            format!(
                "hash count mismatch {} != {}",
                hashes_done,
                TOTAL_HASHES
            )
            .into(),
        );
    }

    // Equal-budget integrity.
    let expected_per_state =
        (EXPECTED_HEADERS
            * SAMPLES_PER_STATE) as u64;

    for state in 0..STATES {
        if combined.states[state].n
            != expected_per_state
        {
            return Err(
                format!(
                    "state {} budget mismatch",
                    state
                )
                .into(),
            );
        }
    }

    println!();
    println!("HASHING COMPLETE");
    println!("----------------");
    println!(
        "SHA256d          : {}",
        hashes_done
    );
    println!(
        "Elapsed          : {:.3}s",
        elapsed.as_secs_f64()
    );
    println!(
        "SHA256d/sec      : {:.2}",
        hashes_done as f64
            / elapsed.as_secs_f64()
    );
    println!(
        "Per L9 state     : {}",
        expected_per_state
    );
    println!("Equal budget      : PASS");
    println!();

    // ========================================================
    // RESULTS
    // ========================================================

    print_split(
        "SPLIT A — heights 700000..799900",
        &split_a,
    );

    print_split(
        "SPLIT B — heights 800000..899900",
        &split_b,
    );

    print_split(
        "COMBINED — 2000 HEADERS",
        &combined,
    );

    // ========================================================
    // RANGE ACROSS STATES
    // ========================================================

    let mut min_mean = f64::INFINITY;
    let mut max_mean = f64::NEG_INFINITY;

    let mut min_state = 0usize;
    let mut max_state = 0usize;

    for state in 0..STATES {
        let m = mean(&combined.states[state]);

        if m < min_mean {
            min_mean = m;
            min_state = state;
        }

        if m > max_mean {
            max_mean = m;
            max_state = state;
        }
    }

    println!("STATE RANGE — CONTINUOUS HASH SCORE");
    println!("-----------------------------------");
    println!(
        "Lowest mean : {:03b} {:.9}",
        min_state,
        min_mean
    );
    println!(
        "Highest mean: {:03b} {:.9}",
        max_state,
        max_mean
    );
    println!(
        "Range       : {:.9}",
        max_mean - min_mean
    );
    println!();

    println!("EXPECTED IDEAL-SHA REFERENCE");
    println!("----------------------------");
    println!("Mean normalized prefix : 0.500000");
    println!("P(LZ>=4)               : 6.250000%");
    println!("P(LZ>=8)               : 0.390625%");
    println!("P(LZ>=12)              : 0.024414%");
    println!("P(LZ>=16)              : 0.001526%");
    println!("Mean leading zeros     : ~1.000000");
    println!();

    println!("INTERPRETATION");
    println!("--------------");
    println!("Each L9 state received exactly the");
    println!("same computational budget.");
    println!();
    println!("Therefore the historical frequency");
    println!("with which miners produced winning");
    println!("nonces in each L9 state cannot by");
    println!("itself create a difference here.");
    println!();
    println!("A state difference in this experiment");
    println!("must replicate across independent");
    println!("header splits before further study.");
    println!();
    println!("No state is promoted to a new");
    println!("hypothesis from this run alone.");
    println!();
    println!("This experiment still does NOT prove");
    println!("or disprove profitable mining.");
    println!("It tests SHA256d output geometry under");
    println!("controlled equal-budget nonce regions.");

    println!();
    println!("V0.25 COMPLETE");
    println!("==============");
    println!("Historical winning nonce : EXCLUDED");
    println!("Equal state budget        : YES");
    println!("Real Bitcoin headers      : YES");
    println!("Real SHA256d              : YES");
    println!("Adaptive selection        : NONE");
    println!("ML                         : NONE");

    Ok(())
}
