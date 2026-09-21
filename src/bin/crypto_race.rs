use csv::Reader;
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    error::Error,
    time::Instant,
};

const INPUT: &str = "data/bitcoin-history-v020-frozen.csv";

const START_HEIGHT: u64 = 700_000;
const EXPECTED_ROWS: usize = 200_000;

// Chronological development split.
const TRAIN_HEADERS: usize = 1500;
const BLIND_HEADERS: usize = 500;
const HEADER_STRIDE: usize = 100;

// Candidate generation.
const TRAIN_CANDIDATES: usize = 4096;
const BLIND_POOL: usize = 8192;

// Equal SHA budget in the blind race.
const KEEP: usize = 1024;

// Low-hash target.
const DIFFICULTY_BITS: u32 = 8;

// Model.
const FEATURES: usize = 161;
const LR: f64 = 0.015;
const L2: f64 = 0.000001;

// Frozen deterministic seeds.
const TRAIN_SEED: u64 = 0x1331_0030_A11C_E001;
const BLIND_SEED: u64 = 0x1331_0030_B11D_E002;

#[derive(Clone)]
struct HeaderTemplate {
    height: u64,
    header: [u8; 80],
}

#[derive(Clone)]
struct Logistic {
    w: [f64; FEATURES],
}

impl Logistic {
    fn new() -> Self {
        Self {
            w: [0.0; FEATURES],
        }
    }

    fn predict_x(&self, x: &[f64; FEATURES]) -> f64 {
        let mut z = 0.0;

        for j in 0..FEATURES {
            z += self.w[j] * x[j];
        }

        // Prevent exp overflow.
        z = z.clamp(-30.0, 30.0);

        1.0 / (1.0 + (-z).exp())
    }

    fn predict(
        &self,
        header: &[u8; 80],
        nonce: u32,
    ) -> f64 {
        let x = features(header, nonce);
        self.predict_x(&x)
    }

    fn update(
        &mut self,
        header: &[u8; 80],
        nonce: u32,
        y: f64,
    ) {
        let x = features(header, nonce);
        let p = self.predict_x(&x);

        let err = y - p;

        for j in 0..FEATURES {
            self.w[j] +=
                LR * (
                    err * x[j]
                    - L2 * self.w[j]
                );
        }
    }
}

#[derive(Default)]
struct RaceScore {
    hashes: u64,
    successes: u64,
    leading_zero_sum: u64,
    best_lz_sum: u64,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(
        0x9E3779B97F4A7C15
    );

    let mut z = *state;

    z = (z ^ (z >> 30))
        .wrapping_mul(0xBF58476D1CE4E5B9);

    z = (z ^ (z >> 27))
        .wrapping_mul(0x94D049BB133111EB);

    z ^ (z >> 31)
}

fn next_nonce(state: &mut u64) -> u32 {
    splitmix64(state) as u32
}

fn sha256d(
    header: &[u8; 80],
    nonce: u32,
) -> [u8; 32] {
    let mut h = *header;

    h[76..80]
        .copy_from_slice(&nonce.to_le_bytes());

    let first = Sha256::digest(h);
    let second = Sha256::digest(first);

    second.into()
}

// We use the digest as a big-endian bit string for the
// experimental low-hash criterion. This is NOT Bitcoin's
// integer-display convention; it is an internal deterministic
// difficulty ladder.
fn leading_zero_bits(hash: &[u8; 32]) -> u32 {
    let mut total = 0u32;

    for &b in hash {
        if b == 0 {
            total += 8;
        } else {
            total += b.leading_zeros();
            break;
        }
    }

    total
}

fn is_low_hash(hash: &[u8; 32]) -> bool {
    leading_zero_bits(hash) >= DIFFICULTY_BITS
}

// 161 features:
//
// 0       bias
// 1..64   selected header bits from bytes 0..75
// 65..96  32 nonce bits
// 97..128 interactions:
//          nonce bit i XOR header-derived bit i
// 129..160 multiplicative bit interactions
//
// No resulting hash is present in features.
fn features(
    header: &[u8; 80],
    nonce: u32,
) -> [f64; FEATURES] {
    let mut x = [0.0; FEATURES];

    x[0] = 1.0;

    // 64 deterministic pre-hash header bits.
    //
    // Spread across the first 76 bytes.
    for i in 0..64 {
        let byte_index =
            (i * 73 + 11) % 76;

        let bit_index =
            (i * 5 + 3) % 8;

        let bit =
            (header[byte_index] >> bit_index) & 1;

        x[1 + i] = bit as f64;
    }

    // Nonce bits.
    for i in 0..32 {
        x[65 + i] =
            ((nonce >> i) & 1) as f64;
    }

    // XOR-style interactions.
    for i in 0..32 {
        let byte_index =
            (i * 37 + 7) % 76;

        let bit_index =
            (i * 3 + 1) % 8;

        let hb =
            ((header[byte_index] >> bit_index) & 1)
            as u32;

        let nb =
            (nonce >> i) & 1;

        x[97 + i] =
            (hb ^ nb) as f64;
    }

    // Multiplicative interactions.
    for i in 0..32 {
        let byte_index =
            (i * 53 + 19) % 76;

        let bit_index =
            (i * 7 + 2) % 8;

        let hb =
            ((header[byte_index] >> bit_index) & 1)
            as f64;

        let nb =
            ((nonce >> ((i * 11) % 32)) & 1)
            as f64;

        x[129 + i] = hb * nb;
    }

    x
}

fn load_headers()
    -> Result<Vec<HeaderTemplate>, Box<dyn Error>>
{
    let mut reader = Reader::from_path(INPUT)?;
    let columns = reader.headers()?.clone();

    let height_idx = columns
        .iter()
        .position(|x| x == "height")
        .ok_or("missing height")?;

    let header_idx = columns
        .iter()
        .position(|x| x == "header_hex")
        .or_else(|| {
            columns.iter().position(|x| x == "header")
        })
        .ok_or("missing header")?;

    let verified_idx =
        columns.iter().position(|x| x == "verified");

    let mut selected = Vec::new();

    for (i, rec) in reader.records().enumerate() {
        let rec = rec?;

        let height: u64 =
            rec[height_idx].parse()?;

        let expected =
            START_HEIGHT + i as u64;

        if height != expected {
            return Err(
                format!(
                    "chronology failure {} != {}",
                    height,
                    expected
                ).into()
            );
        }

        if let Some(v_idx) = verified_idx {
            let v =
                rec[v_idx].to_ascii_lowercase();

            if !matches!(
                v.as_str(),
                "true" | "1" | "yes"
            ) {
                return Err(
                    format!(
                        "unverified {}",
                        height
                    ).into()
                );
            }
        }

        if i % HEADER_STRIDE != 0 {
            continue;
        }

        let raw =
            hex::decode(&rec[header_idx])?;

        if raw.len() != 80 {
            return Err(
                format!(
                    "bad header at {}",
                    height
                ).into()
            );
        }

        let mut header = [0u8; 80];
        header.copy_from_slice(&raw);

        // Mask historical winning nonce.
        header[76..80].fill(0);

        selected.push(
            HeaderTemplate {
                height,
                header,
            }
        );
    }

    Ok(selected)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 v0.30 Cryptographic Landscape Race");
    println!("========================================");
    println!("Input              : {}", INPUT);
    println!("Target             : SHA256d leading-zero >= {}", DIFFICULTY_BITS);
    println!("Expected p         : 1/{}", 1u64 << DIFFICULTY_BITS);
    println!("Header stride      : {}", HEADER_STRIDE);
    println!("Train headers      : {}", TRAIN_HEADERS);
    println!("Blind headers      : {}", BLIND_HEADERS);
    println!("Train candidates/H : {}", TRAIN_CANDIDATES);
    println!("Blind candidate pool: {}", BLIND_POOL);
    println!("Blind kept/H       : {}", KEEP);
    println!("Features           : {}", FEATURES);
    println!("Learning rate      : {}", LR);
    println!("Historical nonce   : MASKED / UNUSED");
    println!("Winning hash       : UNUSED");
    println!();

    let headers = load_headers()?;

    if headers.len()
        != TRAIN_HEADERS + BLIND_HEADERS
    {
        return Err(
            format!(
                "expected {} sampled headers, got {}",
                TRAIN_HEADERS + BLIND_HEADERS,
                headers.len()
            ).into()
        );
    }

    println!("DATASET");
    println!("-------");
    println!("Sampled headers : {}", headers.len());
    println!(
        "Train range     : {}..{}",
        headers[0].height,
        headers[TRAIN_HEADERS - 1].height
    );
    println!(
        "Blind range     : {}..{}",
        headers[TRAIN_HEADERS].height,
        headers.last().unwrap().height
    );
    println!();

    let mut model = Logistic::new();

    // =====================================================
    // TRAIN
    // =====================================================

    println!("TRAINING");
    println!("--------");

    let start = Instant::now();

    let mut train_rng = TRAIN_SEED;
    let mut train_hashes = 0u64;
    let mut train_positive = 0u64;

    for (hi, h) in
        headers[..TRAIN_HEADERS].iter().enumerate()
    {
        for _ in 0..TRAIN_CANDIDATES {
            let nonce =
                next_nonce(&mut train_rng);

            let hash =
                sha256d(&h.header, nonce);

            let y =
                if is_low_hash(&hash) {
                    1.0
                } else {
                    0.0
                };

            train_hashes += 1;
            train_positive += y as u64;

            model.update(
                &h.header,
                nonce,
                y,
            );
        }

        if (hi + 1) % 250 == 0 {
            println!(
                "trained {:>4}/{} headers | hashes {:>10} | positives {}",
                hi + 1,
                TRAIN_HEADERS,
                train_hashes,
                train_positive
            );
        }
    }

    let train_elapsed = start.elapsed();

    println!();
    println!("TRAIN COMPLETE");
    println!("Hashes      : {}", train_hashes);
    println!("Positives   : {}", train_positive);
    println!(
        "Observed p  : {:.8}",
        train_positive as f64
            / train_hashes as f64
    );
    println!(
        "Time        : {:.3}s",
        train_elapsed.as_secs_f64()
    );
    println!();

    // =====================================================
    // BLIND EQUAL-BUDGET RACE
    // =====================================================

    println!("BLIND EQUAL-BUDGET RACE");
    println!("-----------------------");

    let blind_start = Instant::now();

    let mut rng = BLIND_SEED;

    let mut xscore = RaceScore::default();
    let mut cscore = RaceScore::default();

    let mut x_wins = 0u64;
    let mut c_wins = 0u64;
    let mut ties = 0u64;

    for (bi, h) in
        headers[TRAIN_HEADERS..].iter().enumerate()
    {
        let mut candidates:
            Vec<(u32, f64)> =
            Vec::with_capacity(BLIND_POOL);

        // Generate candidate pool WITHOUT hashing.
        for _ in 0..BLIND_POOL {
            let nonce =
                next_nonce(&mut rng);

            let score =
                model.predict(
                    &h.header,
                    nonce,
                );

            candidates.push(
                (nonce, score)
            );
        }

        // Highest model probability first.
        candidates.sort_by(|a, b| {
            b.1
                .partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        // X1331 = top KEEP.
        let xselected =
            &candidates[..KEEP];

        // CONTROL = deterministic slice from bottom/middle
        // of same pre-generated pool, disjoint from X1331.
        //
        // We intentionally use the final KEEP entries.
        let cselected =
            &candidates[
                BLIND_POOL - KEEP .. BLIND_POOL
            ];

        let mut x_success_h = 0u64;
        let mut c_success_h = 0u64;

        let mut x_best = 0u32;
        let mut c_best = 0u32;

        for &(nonce, _) in xselected {
            let hash =
                sha256d(&h.header, nonce);

            let lz =
                leading_zero_bits(&hash);

            xscore.hashes += 1;
            xscore.leading_zero_sum += lz as u64;
            x_best = x_best.max(lz);

            if lz >= DIFFICULTY_BITS {
                xscore.successes += 1;
                x_success_h += 1;
            }
        }

        for &(nonce, _) in cselected {
            let hash =
                sha256d(&h.header, nonce);

            let lz =
                leading_zero_bits(&hash);

            cscore.hashes += 1;
            cscore.leading_zero_sum += lz as u64;
            c_best = c_best.max(lz);

            if lz >= DIFFICULTY_BITS {
                cscore.successes += 1;
                c_success_h += 1;
            }
        }

        xscore.best_lz_sum += x_best as u64;
        cscore.best_lz_sum += c_best as u64;

        if x_success_h > c_success_h {
            x_wins += 1;
        } else if c_success_h > x_success_h {
            c_wins += 1;
        } else {
            ties += 1;
        }

        if (bi + 1) % 100 == 0 {
            println!(
                "blind {:>3}/{} | X={} CTRL={}",
                bi + 1,
                BLIND_HEADERS,
                xscore.successes,
                cscore.successes
            );
        }
    }

    let blind_elapsed =
        blind_start.elapsed();

    // =====================================================
    // RESULTS
    // =====================================================

    let xp =
        xscore.successes as f64
        / xscore.hashes as f64;

    let cp =
        cscore.successes as f64
        / cscore.hashes as f64;

    let gain =
        if cp > 0.0 {
            xp / cp
        } else {
            f64::INFINITY
        };

    let expected =
        1.0 / ((1u64 << DIFFICULTY_BITS) as f64);

    println!();
    println!("FINAL BLIND RESULTS");
    println!("-------------------");
    println!(
        "Equal SHA budget : {} hashes/arm",
        xscore.hashes
    );
    println!();

    println!(
        "X1331 successes  : {}",
        xscore.successes
    );
    println!(
        "CONTROL successes: {}",
        cscore.successes
    );
    println!();

    println!(
        "X1331 rate       : {:.8}",
        xp
    );
    println!(
        "CONTROL rate     : {:.8}",
        cp
    );
    println!(
        "IDEAL rate       : {:.8}",
        expected
    );
    println!();

    println!(
        "PREDICTION GAIN  : {:.8}x",
        gain
    );

    println!(
        "Δ success rate   : {:+.8}",
        xp - cp
    );

    println!();

    println!(
        "Mean LZ X1331    : {:.6}",
        xscore.leading_zero_sum as f64
            / xscore.hashes as f64
    );

    println!(
        "Mean LZ CONTROL  : {:.6}",
        cscore.leading_zero_sum as f64
            / cscore.hashes as f64
    );

    println!(
        "Mean best LZ/H X : {:.6}",
        xscore.best_lz_sum as f64
            / BLIND_HEADERS as f64
    );

    println!(
        "Mean best LZ/H C : {:.6}",
        cscore.best_lz_sum as f64
            / BLIND_HEADERS as f64
    );

    println!();

    println!("HEADER-LEVEL SCORE");
    println!("------------------");
    println!("X1331 wins   : {}", x_wins);
    println!("CONTROL wins : {}", c_wins);
    println!("Ties         : {}", ties);

    println!();

    println!(
        "Blind runtime: {:.3}s",
        blind_elapsed.as_secs_f64()
    );

    println!();

    println!("INTERPRETATION");
    println!("--------------");
    println!("Primary metric: PREDICTION GAIN.");
    println!("Gain ~1 means no usable cryptographic information.");
    println!("Gain >1 in this development experiment is only");
    println!("a candidate signal and MUST be independently replicated.");
    println!();
    println!("This experiment gives X1331 and CONTROL exactly");
    println!("the same number of SHA256d evaluations.");
    println!();
    println!("The historical winning nonce is not used.");
    println!("The historical winning hash is not used.");
    println!();
    println!("Candidate ranking occurs BEFORE blind SHA256d.");
    println!();
    println!("IMPORTANT: the control is the lowest-scored");
    println!("portion of the same pool. Therefore a positive");
    println!("result must later be repeated against an independent");
    println!("uniform-random equal-budget control as well.");

    println!();

    println!("V0.30 COMPLETE");
    println!("==============");

    Ok(())
}
