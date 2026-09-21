use csv::Reader;
use std::{
    cmp::Ordering,
    error::Error,
};

const INPUT: &str = "data/bitcoin-history-v020-frozen.csv";

const START_HEIGHT: u64 = 700_000;
const EXPECTED_ROWS: usize = 200_000;

const WARMUP: usize = 20_000;

const STATES: usize = 8;
const FEATURES: usize = 33;

// Frozen before results.
const LEARNING_RATE: f64 = 0.010;
const L2: f64 = 0.00001;

#[derive(Clone)]
struct Block {
    height: u64,
    version: u32,
    target: usize,
}

#[derive(Clone)]
struct SoftmaxObserver {
    weights: [[f64; FEATURES]; STATES],
}

impl SoftmaxObserver {
    fn new() -> Self {
        Self {
            weights: [[0.0; FEATURES]; STATES],
        }
    }

    fn features(version: u32) -> [f64; FEATURES] {
        let mut x = [0.0; FEATURES];

        // Bias
        x[0] = 1.0;

        // Exact 32 version bits.
        for bit in 0..32 {
            x[bit + 1] =
                if ((version >> bit) & 1) == 1 {
                    1.0
                } else {
                    0.0
                };
        }

        x
    }

    fn predict_from_x(
        &self,
        x: &[f64; FEATURES],
    ) -> [f64; STATES] {
        let mut logits = [0.0; STATES];

        for s in 0..STATES {
            let mut z = 0.0;

            for j in 0..FEATURES {
                z += self.weights[s][j] * x[j];
            }

            logits[s] = z;
        }

        let max_logit = logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut p = [0.0; STATES];
        let mut sum = 0.0;

        for s in 0..STATES {
            p[s] = (logits[s] - max_logit).exp();
            sum += p[s];
        }

        for s in 0..STATES {
            p[s] /= sum;
        }

        p
    }

    fn predict(
        &self,
        version: u32,
    ) -> [f64; STATES] {
        let x = Self::features(version);
        self.predict_from_x(&x)
    }

    fn update(
        &mut self,
        version: u32,
        target: usize,
    ) {
        let x = Self::features(version);
        let p = self.predict_from_x(&x);

        for s in 0..STATES {
            let y =
                if s == target {
                    1.0
                } else {
                    0.0
                };

            let error = y - p[s];

            for j in 0..FEATURES {
                self.weights[s][j] +=
                    LEARNING_RATE
                    * (
                        error * x[j]
                        - L2 * self.weights[s][j]
                    );
            }
        }
    }
}

#[derive(Default, Clone)]
struct Score {
    n: u64,
    hits: [u64; 5],
    logloss: f64,
    brier: f64,
}

fn rank(p: &[f64; STATES]) -> [usize; STATES] {
    let mut r = [0, 1, 2, 3, 4, 5, 6, 7];

    r.sort_by(|&a, &b| {
        p[b]
            .partial_cmp(&p[a])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    r
}

fn score_prediction(
    score: &mut Score,
    p: &[f64; STATES],
    target: usize,
) {
    score.n += 1;

    let r = rank(p);

    for k in 1..=4 {
        if r[..k].contains(&target) {
            score.hits[k] += 1;
        }
    }

    score.logloss -=
        p[target].max(1e-15).ln();

    let mut brier = 0.0;

    for s in 0..STATES {
        let y =
            if s == target {
                1.0
            } else {
                0.0
            };

        let d = p[s] - y;
        brier += d * d;
    }

    score.brier += brier;
}

fn prior_prediction(
    counts: &[u64; STATES],
    total: u64,
) -> [f64; STATES] {
    let mut p = [1.0 / STATES as f64; STATES];

    if total > 0 {
        // Small smoothing to avoid zero probability.
        let denom =
            total as f64 + STATES as f64;

        for s in 0..STATES {
            p[s] =
                (counts[s] as f64 + 1.0)
                / denom;
        }
    }

    p
}

fn print_score(
    name: &str,
    score: &Score,
) {
    println!(
        "{:<18} {:>8.3}% {:>8.3}% {:>8.3}% {:>8.3}% {:>11.6} {:>11.6}",
        name,
        score.hits[1] as f64
            / score.n as f64 * 100.0,
        score.hits[2] as f64
            / score.n as f64 * 100.0,
        score.hits[3] as f64
            / score.n as f64 * 100.0,
        score.hits[4] as f64
            / score.n as f64 * 100.0,
        score.logloss / score.n as f64,
        score.brier / score.n as f64,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 v0.29 Version-Bit Observer");
    println!("================================");
    println!("Input              : {}", INPUT);
    println!("Rows               : {}", EXPECTED_ROWS);
    println!("Warm-up            : {}", WARMUP);
    println!(
        "Scored             : {}",
        EXPECTED_ROWS - WARMUP
    );
    println!("Features           : bias + 32 version bits");
    println!("Parameters         : {}", FEATURES * STATES);
    println!("Learning rate      : {:.6}", LEARNING_RATE);
    println!("L2                 : {:.8}", L2);
    println!("Target             : historical winning nonce L9");
    println!("Walk-forward       : YES");
    println!("Future leakage     : NONE");
    println!("Nonce as feature   : NO");
    println!("Winning hash input : NO");
    println!();

    // ========================================================
    // LOAD DATASET
    // ========================================================

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
        .ok_or("missing header_hex/header")?;

    let verified_idx =
        columns.iter().position(|x| x == "verified");

    let mut blocks =
        Vec::with_capacity(EXPECTED_ROWS);

    for (i, rec) in reader.records().enumerate() {
        let rec = rec?;

        let height: u64 =
            rec[height_idx].parse()?;

        let expected =
            START_HEIGHT + i as u64;

        if height != expected {
            return Err(
                format!(
                    "chronology failure: expected {}, got {}",
                    expected,
                    height
                )
                .into()
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
                        "unverified block {}",
                        height
                    )
                    .into()
                );
            }
        }

        let raw =
            hex::decode(&rec[header_idx])?;

        if raw.len() != 80 {
            return Err(
                format!(
                    "header length != 80 at {}",
                    height
                )
                .into()
            );
        }

        let version =
            u32::from_le_bytes(
                raw[0..4].try_into()?
            );

        let nonce =
            u32::from_le_bytes(
                raw[76..80].try_into()?
            );

        let target =
            ((nonce >> 5) & 0b111) as usize;

        blocks.push(Block {
            height,
            version,
            target,
        });
    }

    if blocks.len() != EXPECTED_ROWS {
        return Err(
            format!(
                "expected {} rows, got {}",
                EXPECTED_ROWS,
                blocks.len()
            )
            .into()
        );
    }

    println!("DATASET VALIDATION");
    println!("------------------");
    println!("Rows       : {}", blocks.len());
    println!(
        "Range      : {}..{}",
        blocks.first().unwrap().height,
        blocks.last().unwrap().height
    );
    println!("Chronology : PASS");
    println!();

    // ========================================================
    // OBSERVER
    // ========================================================

    let mut observer =
        SoftmaxObserver::new();

    let mut prior_counts =
        [0u64; STATES];

    let mut prior_total =
        0u64;

    let mut prior_score =
        Score::default();

    let mut observer_score =
        Score::default();

    // Per 20k scored segment diagnostics.
    let mut segment_prior =
        Score::default();

    let mut segment_observer =
        Score::default();

    let mut segment_start =
        0u64;

    println!("WALK-FORWARD");
    println!("------------");

    for (i, block) in blocks.iter().enumerate() {
        // -----------------------------------------------
        // Prediction BEFORE reveal/update.
        // -----------------------------------------------

        if i >= WARMUP {
            if segment_prior.n == 0 {
                segment_start =
                    block.height;
            }

            let pp =
                prior_prediction(
                    &prior_counts,
                    prior_total,
                );

            let op =
                observer.predict(
                    block.version
                );

            score_prediction(
                &mut prior_score,
                &pp,
                block.target,
            );

            score_prediction(
                &mut observer_score,
                &op,
                block.target,
            );

            score_prediction(
                &mut segment_prior,
                &pp,
                block.target,
            );

            score_prediction(
                &mut segment_observer,
                &op,
                block.target,
            );
        }

        // -----------------------------------------------
        // Reveal target only now.
        // -----------------------------------------------

        prior_counts[block.target] += 1;
        prior_total += 1;

        observer.update(
            block.version,
            block.target,
        );

        // -----------------------------------------------
        // Segment report every 20k scored observations.
        // -----------------------------------------------

        if segment_observer.n == 20_000 {
            let end =
                block.height;

            let prior_ll =
                segment_prior.logloss
                / segment_prior.n as f64;

            let obs_ll =
                segment_observer.logloss
                / segment_observer.n as f64;

            let prior_r1 =
                segment_prior.hits[1] as f64
                / segment_prior.n as f64;

            let obs_r1 =
                segment_observer.hits[1] as f64
                / segment_observer.n as f64;

            println!(
                "{}..{} | PRIOR R1 {:>6.2}% LL {:.6} | OBS R1 {:>6.2}% LL {:.6} | ΔLL {:+.6}",
                segment_start,
                end,
                prior_r1 * 100.0,
                prior_ll,
                obs_r1 * 100.0,
                obs_ll,
                obs_ll - prior_ll
            );

            segment_prior =
                Score::default();

            segment_observer =
                Score::default();
        }
    }

    // Last partial segment if any.
    if segment_observer.n > 0 {
        let end =
            blocks.last().unwrap().height;

        let prior_ll =
            segment_prior.logloss
            / segment_prior.n as f64;

        let obs_ll =
            segment_observer.logloss
            / segment_observer.n as f64;

        let prior_r1 =
            segment_prior.hits[1] as f64
            / segment_prior.n as f64;

        let obs_r1 =
            segment_observer.hits[1] as f64
            / segment_observer.n as f64;

        println!(
            "{}..{} | PRIOR R1 {:>6.2}% LL {:.6} | OBS R1 {:>6.2}% LL {:.6} | ΔLL {:+.6}",
            segment_start,
            end,
            prior_r1 * 100.0,
            prior_ll,
            obs_r1 * 100.0,
            obs_ll,
            obs_ll - prior_ll
        );
    }

    println!();

    // ========================================================
    // FINAL RESULTS
    // ========================================================

    println!("FINAL RESULTS");
    println!("-------------");

    println!(
        "{:<18} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11}",
        "Model",
        "R@1",
        "R@2",
        "R@3",
        "R@4",
        "LogLoss",
        "Brier"
    );

    print_score(
        "PRIOR",
        &prior_score,
    );

    print_score(
        "VERSION-BITS",
        &observer_score,
    );

    println!();

    // ========================================================
    // GAINS
    // ========================================================

    println!("GAIN VS PRIOR");
    println!("-------------");

    for k in 1..=4 {
        let prior =
            prior_score.hits[k] as f64
            / prior_score.n as f64;

        let obs =
            observer_score.hits[k] as f64
            / observer_score.n as f64;

        println!(
            "G@{} = {:.6}   ΔRecall = {:+.4} pp",
            k,
            obs / prior,
            (obs - prior) * 100.0
        );
    }

    let prior_ll =
        prior_score.logloss
        / prior_score.n as f64;

    let obs_ll =
        observer_score.logloss
        / observer_score.n as f64;

    let prior_brier =
        prior_score.brier
        / prior_score.n as f64;

    let obs_brier =
        observer_score.brier
        / observer_score.n as f64;

    println!();

    println!(
        "ΔLogLoss = {:+.8}",
        obs_ll - prior_ll
    );

    println!(
        "ΔBrier   = {:+.8}",
        obs_brier - prior_brier
    );

    println!();

    // ========================================================
    // LEARNED BIT MAGNITUDES
    // ========================================================

    println!("LEARNED VERSION-BIT MAGNITUDES");
    println!("------------------------------");
    println!("Descriptive only.");
    println!("Magnitude = RMS weight across 8 L9 states.");
    println!();

    let mut magnitudes:
        Vec<(usize, f64)> =
        Vec::new();

    for bit in 0..32 {
        let feature = bit + 1;

        let mut sum_sq = 0.0;

        for state in 0..STATES {
            let w =
                observer.weights[state][feature];

            sum_sq += w * w;
        }

        let rms =
            (sum_sq / STATES as f64).sqrt();

        magnitudes.push((bit, rms));
    }

    magnitudes.sort_by(|a, b| {
        b.1
            .partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
    });

    for (rank_i, (bit, magnitude))
        in magnitudes.iter().enumerate()
    {
        println!(
            "#{:02} version bit {:>2}  RMS={:.8}",
            rank_i + 1,
            bit,
            magnitude
        );
    }

    println!();

    // ========================================================
    // FINAL PRIOR
    // ========================================================

    println!("FINAL HISTORICAL L9 PRIOR");
    println!("-------------------------");

    for state in 0..STATES {
        println!(
            "{:03b} {:>7} {:>9.5}%",
            state,
            prior_counts[state],
            prior_counts[state] as f64
                / prior_total as f64
                * 100.0
        );
    }

    println!();

    println!("INTERPRETATION RULE");
    println!("-------------------");
    println!("Primary comparison is VERSION-BITS vs PRIOR.");
    println!();
    println!("Useful historical conditional information requires");
    println!("lower out-of-sample walk-forward LogLoss/Brier.");
    println!();
    println!("Top-K gains > 1 would additionally show that");
    println!("probability improvements alter collapse decisions.");
    println!();
    println!("Because v0.29 was designed after observing v0.27/v0.28,");
    println!("700000..899999 is DEVELOPMENT evidence.");
    println!("It is NOT a pristine confirmation dataset for v0.29.");
    println!();
    println!("A positive result must later be frozen and tested");
    println!("on newly acquired blocks > 899999.");
    println!();
    println!("This remains mining-process prediction.");
    println!("It does NOT establish SHA256d predictability.");

    println!();

    println!("V0.29 COMPLETE");
    println!("==============");
    println!("Future leakage       : NONE");
    println!("Winning nonce input  : NO");
    println!("Winning hash input   : NO");
    println!("Features             : version bits only");
    println!("Update               : one-pass online");

    Ok(())
}
