use csv::Reader;
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    error::Error,
    time::Instant,
};

const INPUT: &str =
    "data/bitcoin-history-v020-frozen.csv";

const START: u64 = 700_000;
const EXPECTED_ROWS: usize = 200_000;

const HEADER_STEP: usize = 100;
const HEADERS: usize = 2_000;

const TRAIN_HEADERS: usize = 1_500;
const BLIND_HEADERS: usize = 500;

const STATES: usize = 8;
const SAMPLES: usize = 256;

const FEATURES: usize = 77; // bias + 76 header bytes
const EPOCHS: usize = 40;
const LR: f64 = 0.05;

// Frozen deterministic counterfactual generator.
const RNG_SEED: u64 = 0x1331_2026_0921_A55A;

#[derive(Clone)]
struct Row {
    height: u64,
    header: [u8; 80],
}

#[derive(Clone)]
struct Example {
    height: u64,
    x: [f64; FEATURES],

    // Number of SHA256d samples satisfying
    // LZ >= 8 in each L9 region.
    successes: [u32; STATES],

    // Empirical distribution of low-hash
    // events among the 8 regions.
    target: [f64; STATES],
}

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
                0x9E3779B97F4A7C15
            );

        let mut z = self.state;

        z = (z ^ (z >> 30))
            .wrapping_mul(
                0xBF58476D1CE4E5B9
            );

        z = (z ^ (z >> 27))
            .wrapping_mul(
                0x94D049BB133111EB
            );

        z ^ (z >> 31)
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

fn sha256d(data: &[u8]) -> [u8; 32] {
    let a = Sha256::digest(data);
    let b = Sha256::digest(a);

    let mut out = [0u8; 32];
    out.copy_from_slice(&b);
    out
}

fn leading_zero_bits(hash: &[u8; 32]) -> u32 {
    let mut n = 0;

    for &b in hash {
        if b == 0 {
            n += 8;
        } else {
            n += b.leading_zeros();
            break;
        }
    }

    n
}

fn force_state(raw: u32, state: usize) -> u32 {
    (raw & !(0b111u32 << 5))
        | ((state as u32) << 5)
}

fn features(header: &[u8; 80]) -> [f64; FEATURES] {
    let mut x = [0.0; FEATURES];

    x[0] = 1.0;

    // Critical leakage rule:
    // bytes 0..75 only.
    // Nonce bytes 76..79 are EXCLUDED.
    for i in 0..76 {
        x[i + 1] =
            header[i] as f64 / 255.0;
    }

    x
}

fn softmax(
    logits: &[f64; STATES],
) -> [f64; STATES] {
    let max =
        logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

    let mut p = [0.0; STATES];
    let mut sum = 0.0;

    for s in 0..STATES {
        p[s] = (logits[s] - max).exp();
        sum += p[s];
    }

    for s in 0..STATES {
        p[s] /= sum;
    }

    p
}

fn predict(
    weights: &[[f64; FEATURES]; STATES],
    x: &[f64; FEATURES],
) -> [f64; STATES] {
    let mut logits = [0.0; STATES];

    for s in 0..STATES {
        for j in 0..FEATURES {
            logits[s] += weights[s][j] * x[j];
        }
    }

    softmax(&logits)
}

fn rank(
    p: &[f64; STATES],
) -> [usize; STATES] {
    let mut r = [0,1,2,3,4,5,6,7];

    r.sort_by(|&a, &b| {
        p[b]
            .partial_cmp(&p[a])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    r
}

fn fmt_state(s: usize) -> String {
    format!("{:03b}", s)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 v0.26 Header -> Hash Landscape Laboratory");
    println!("===============================================");
    println!("Input             : {}", INPUT);
    println!("Headers           : {}", HEADERS);
    println!("Train             : {}", TRAIN_HEADERS);
    println!("Blind             : {}", BLIND_HEADERS);
    println!("Samples/state     : {}", SAMPLES);
    println!("Landscape target  : LZ >= 8");
    println!("States            : 8 (nonce bits 7..5)");
    println!("Features          : Header76 only");
    println!("Nonce feature     : EXCLUDED");
    println!("Hash feature      : EXCLUDED");
    println!("Epochs            : {}", EPOCHS);
    println!("Learning rate     : {:.4}", LR);
    println!("Future leakage    : NONE");
    println!();

    // ========================================================
    // LOAD HEADERS
    // ========================================================

    let mut reader = Reader::from_path(INPUT)?;
    let hdrs = reader.headers()?.clone();

    let height_idx =
        hdrs.iter()
            .position(|x| x == "height")
            .ok_or("height missing")?;

    let header_idx =
        hdrs.iter()
            .position(|x| x == "header_hex")
            .ok_or("header_hex missing")?;

    let verified_idx =
        hdrs.iter()
            .position(|x| x == "verified")
            .ok_or("verified missing")?;

    let mut rows = Vec::with_capacity(HEADERS);
    let mut row_index = 0usize;

    for result in reader.records() {
        let record = result?;

        let height: u64 =
            record[height_idx].parse()?;

        if height != START + row_index as u64 {
            return Err(
                format!("chronology failure at {}", height).into()
            );
        }

        let verified = matches!(
            record[verified_idx]
                .to_ascii_lowercase()
                .as_str(),
            "true" | "1" | "yes"
        );

        if !verified {
            return Err(
                format!("unverified {}", height).into()
            );
        }

        if row_index % HEADER_STEP == 0 {
            let raw =
                hex::decode(&record[header_idx])?;

            if raw.len() != 80 {
                return Err("header length != 80".into());
            }

            let mut header = [0u8; 80];
            header.copy_from_slice(&raw);

            rows.push(Row {
                height,
                header,
            });
        }

        row_index += 1;
    }

    if row_index != EXPECTED_ROWS {
        return Err(
            format!(
                "expected {} rows got {}",
                EXPECTED_ROWS,
                row_index
            ).into()
        );
    }

    if rows.len() != HEADERS {
        return Err(
            format!(
                "expected {} headers got {}",
                HEADERS,
                rows.len()
            ).into()
        );
    }

    println!("DATASET VALIDATION");
    println!("------------------");
    println!("Rows          : {}", row_index);
    println!("Headers       : {}", rows.len());
    println!(
        "Range         : {}..{}",
        rows[0].height,
        rows.last().unwrap().height
    );
    println!("Verified      : ALL");
    println!();

    // ========================================================
    // BUILD COUNTERFACTUAL LANDSCAPES
    // ========================================================

    println!("BUILDING LANDSCAPES");
    println!("-------------------");

    let started = Instant::now();
    let mut rng = SplitMix64::new(RNG_SEED);
    let mut examples = Vec::with_capacity(HEADERS);
    let mut hashes = 0u64;

    for (i, row) in rows.iter().enumerate() {
        let mut successes = [0u32; STATES];

        for state in 0..STATES {
            for _ in 0..SAMPLES {
                let nonce =
                    force_state(
                        rng.next_u32(),
                        state
                    );

                let mut candidate = row.header;

                candidate[76..80]
                    .copy_from_slice(
                        &nonce.to_le_bytes()
                    );

                let h = sha256d(&candidate);

                if leading_zero_bits(&h) >= 8 {
                    successes[state] += 1;
                }

                hashes += 1;
            }
        }

        let total_successes: u32 =
            successes.iter().sum();

        // With 2048 trials/header and p=1/256,
        // zero-event headers are possible.
        // For those, target is uniform and contributes
        // no directional information.
        let mut target = [1.0 / 8.0; STATES];

        if total_successes > 0 {
            for s in 0..STATES {
                target[s] =
                    successes[s] as f64
                    / total_successes as f64;
            }
        }

        examples.push(Example {
            height: row.height,
            x: features(&row.header),
            successes,
            target,
        });

        if (i + 1) % 250 == 0 {
            println!(
                "Progress {:>4}/{}  SHA256d {}",
                i + 1,
                HEADERS,
                hashes
            );
        }
    }

    let elapsed = started.elapsed();

    println!();
    println!("LANDSCAPES COMPLETE");
    println!("-------------------");
    println!("SHA256d      : {}", hashes);
    println!(
        "Elapsed      : {:.3}s",
        elapsed.as_secs_f64()
    );
    println!(
        "SHA256d/sec  : {:.2}",
        hashes as f64 / elapsed.as_secs_f64()
    );
    println!();

    // ========================================================
    // TRAIN
    // ========================================================

    let mut weights =
        [[0.0f64; FEATURES]; STATES];

    println!("TRAINING");
    println!("--------");

    for epoch in 0..EPOCHS {
        let mut loss = 0.0;

        for ex in &examples[..TRAIN_HEADERS] {
            let p = predict(&weights, &ex.x);

            for s in 0..STATES {
                let t = ex.target[s];

                if t > 0.0 {
                    loss -= t * p[s].max(1e-15).ln();
                }
            }

            for s in 0..STATES {
                let error =
                    p[s] - ex.target[s];

                for j in 0..FEATURES {
                    weights[s][j] -=
                        LR * error * ex.x[j];
                }
            }
        }

        if epoch == 0
            || (epoch + 1) % 5 == 0
        {
            println!(
                "epoch {:>2}/{:<2} loss {:.9}",
                epoch + 1,
                EPOCHS,
                loss / TRAIN_HEADERS as f64
            );
        }
    }

    println!();

    // ========================================================
    // BLIND EVALUATION
    // ========================================================

    println!("BLIND EVALUATION");
    println!("----------------");
    println!(
        "Range : {}..{}",
        examples[TRAIN_HEADERS].height,
        examples.last().unwrap().height
    );

    let mut total_events = 0u64;

    let mut model_hits = [0u64; 5];

    // Static ranking learned only from aggregate
    // TRAIN landscape.
    let mut train_events = [0u64; STATES];

    for ex in &examples[..TRAIN_HEADERS] {
        for s in 0..STATES {
            train_events[s] +=
                ex.successes[s] as u64;
        }
    }

    let train_total: u64 =
        train_events.iter().sum();

    let mut static_p = [0.0; STATES];

    if train_total == 0 {
        static_p = [1.0 / 8.0; STATES];
    } else {
        for s in 0..STATES {
            static_p[s] =
                train_events[s] as f64
                / train_total as f64;
        }
    }

    let static_rank = rank(&static_p);
    let mut static_hits = [0u64; 5];

    // Deterministic rotating control.
    // This is NOT used to train anything.
    let mut control_hits = [0u64; 5];

    let mut predicted_top1_counts =
        [0u64; STATES];

    for (blind_i, ex) in
        examples[TRAIN_HEADERS..]
            .iter()
            .enumerate()
    {
        let p = predict(&weights, &ex.x);
        let r = rank(&p);

        predicted_top1_counts[r[0]] += 1;

        let events: u64 =
            ex.successes
                .iter()
                .map(|&x| x as u64)
                .sum();

        total_events += events;

        for k in 1..=4 {
            for &s in &r[..k] {
                model_hits[k] +=
                    ex.successes[s] as u64;
            }

            for &s in &static_rank[..k] {
                static_hits[k] +=
                    ex.successes[s] as u64;
            }

            // Rotating state order so the control
            // does not always choose 000,001,...
            let start = blind_i % STATES;

            for offset in 0..k {
                let s =
                    (start + offset) % STATES;

                control_hits[k] +=
                    ex.successes[s] as u64;
            }
        }
    }

    println!("Blind low-hash events : {}", total_events);
    println!();

    println!("TRAIN STATIC PRIOR");
    println!("------------------");

    for &s in &static_rank {
        println!(
            "{}  events={}  p={:.6}%",
            fmt_state(s),
            train_events[s],
            static_p[s] * 100.0
        );
    }

    println!();

    println!("BLIND N -> K RESULTS");
    println!("--------------------");
    println!(
        "{:<3} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "K",
        "Random",
        "Static",
        "Observer",
        "GainRnd",
        "GainStat"
    );

    for k in 1..=4 {
        let random =
            k as f64 / STATES as f64;

        let static_recall =
            static_hits[k] as f64
            / total_events as f64;

        let model_recall =
            model_hits[k] as f64
            / total_events as f64;

        println!(
            "{:<3} {:>10.5}% {:>10.5}% {:>10.5}% {:>11.6} {:>11.6}",
            k,
            random * 100.0,
            static_recall * 100.0,
            model_recall * 100.0,
            model_recall / random,
            model_recall / static_recall,
        );
    }

    println!();

    println!("ROTATING CONTROL");
    println!("----------------");

    for k in 1..=4 {
        let recall =
            control_hits[k] as f64
            / total_events as f64;

        println!(
            "K={} recall={:.5}% gain={:.6}",
            k,
            recall * 100.0,
            recall
                / (k as f64 / STATES as f64)
        );
    }

    println!();

    println!("OBSERVER TOP1 DISTRIBUTION");
    println!("--------------------------");

    for s in 0..STATES {
        println!(
            "{}  {:>4}/{}  {:.2}%",
            fmt_state(s),
            predicted_top1_counts[s],
            BLIND_HEADERS,
            predicted_top1_counts[s] as f64
                / BLIND_HEADERS as f64
                * 100.0
        );
    }

    println!();

    println!("INTERPRETATION RULE");
    println!("-------------------");
    println!("The Observer is useful only if its");
    println!("blind performance improves over BOTH");
    println!("uniform/random expectation and the");
    println!("static TRAIN landscape.");
    println!();
    println!("A training-loss decrease is NOT");
    println!("evidence of predictive information.");
    println!();
    println!("No hyperparameter may be changed");
    println!("after this output and still call the");
    println!("result a replication of v0.26.");
    println!();

    println!("V0.26 COMPLETE");
    println!("==============");
    println!("Train headers       : {}", TRAIN_HEADERS);
    println!("Blind headers       : {}", BLIND_HEADERS);
    println!("Header nonce bytes  : EXCLUDED");
    println!("Candidate hashes    : LABEL ONLY");
    println!("Future leakage      : NONE");
    println!("Adaptive blind fit  : NONE");

    Ok(())
}
