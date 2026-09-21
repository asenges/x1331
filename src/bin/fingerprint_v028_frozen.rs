use csv::Reader;
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    error::Error,
};

const INPUT: &str = "data/bitcoin-history-v020-frozen.csv";

const START_HEIGHT: u64 = 700_000;
const EXPECTED_ROWS: usize = 200_000;

const WARMUP: usize = 20_000;
const BUCKETS: usize = 4096;
const STATES: usize = 8;

// Laplace/Dirichlet smoothing.
const ALPHA: f64 = 8.0;

#[derive(Clone)]
struct Block {
    height: u64,
    header: [u8; 80],
    target: usize,
}

#[derive(Clone, Copy)]
enum Model {
    Prior,
    Version,
    Merkle,
    Time,
    PrevHash,
    VersionMerkle,
    VersionTime,
    All,
}

impl Model {
    fn name(self) -> &'static str {
        match self {
            Model::Prior => "PRIOR",
            Model::Version => "VERSION",
            Model::Merkle => "MERKLE",
            Model::Time => "TIME",
            Model::PrevHash => "PREVHASH",
            Model::VersionMerkle => "VERSION+MERKLE",
            Model::VersionTime => "VERSION+TIME",
            Model::All => "ALL",
        }
    }
}

const MODELS: [Model; 8] = [
    Model::Prior,
    Model::Version,
    Model::Merkle,
    Model::Time,
    Model::PrevHash,
    Model::VersionMerkle,
    Model::VersionTime,
    Model::All,
];

#[derive(Clone)]
struct BucketModel {
    counts: Vec<[u64; STATES]>,
    totals: Vec<u64>,
}

impl BucketModel {
    fn new() -> Self {
        Self {
            counts: vec![[0; STATES]; BUCKETS],
            totals: vec![0; BUCKETS],
        }
    }

    fn update(&mut self, bucket: usize, target: usize) {
        self.counts[bucket][target] += 1;
        self.totals[bucket] += 1;
    }

    fn predict(
        &self,
        bucket: usize,
        prior: &[u64; STATES],
        prior_total: u64,
    ) -> [f64; STATES] {
        let mut p = [0.0; STATES];

        let mut prior_p = [1.0 / STATES as f64; STATES];

        if prior_total > 0 {
            for s in 0..STATES {
                prior_p[s] =
                    prior[s] as f64 / prior_total as f64;
            }
        }

        let n = self.totals[bucket] as f64;

        for s in 0..STATES {
            p[s] =
                (self.counts[bucket][s] as f64
                    + ALPHA * prior_p[s])
                / (n + ALPHA);
        }

        p
    }
}

#[derive(Default, Clone)]
struct Score {
    n: u64,
    hits: [u64; 5],
    logloss: f64,
    brier: f64,
}

fn hash_bucket(parts: &[&[u8]]) -> usize {
    let mut h = Sha256::new();

    for part in parts {
        h.update(part);
    }

    let out = h.finalize();

    let x = u64::from_le_bytes(
        out[0..8].try_into().unwrap()
    );

    (x as usize) % BUCKETS
}

fn bucket_for(model: Model, h: &[u8; 80]) -> usize {
    match model {
        Model::Prior => 0,

        // version bytes
        Model::Version => {
            hash_bucket(&[&h[0..4]])
        }

        // merkle root bytes
        Model::Merkle => {
            hash_bucket(&[&h[36..68]])
        }

        // nTime
        Model::Time => {
            hash_bucket(&[&h[68..72]])
        }

        // previous block hash
        Model::PrevHash => {
            hash_bucket(&[&h[4..36]])
        }

        Model::VersionMerkle => {
            hash_bucket(&[
                &h[0..4],
                &h[36..68],
            ])
        }

        Model::VersionTime => {
            hash_bucket(&[
                &h[0..4],
                &h[68..72],
            ])
        }

        // IMPORTANT:
        // nonce bytes 76..79 are never included.
        Model::All => {
            hash_bucket(&[
                &h[0..4],   // version
                &h[4..36],  // prev hash
                &h[36..68], // merkle
                &h[68..72], // time
                &h[72..76], // nBits
            ])
        }
    }
}

fn prior_prediction(
    counts: &[u64; STATES],
    total: u64,
) -> [f64; STATES] {
    let mut p = [1.0 / STATES as f64; STATES];

    if total > 0 {
        for s in 0..STATES {
            p[s] = counts[s] as f64 / total as f64;
        }
    }

    p
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

fn add_score(
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

    score.logloss -= p[target].max(1e-15).ln();

    let mut brier = 0.0;

    for s in 0..STATES {
        let y = if s == target { 1.0 } else { 0.0 };
        let d = p[s] - y;
        brier += d * d;
    }

    score.brier += brier;
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 v0.28 Challenge Fingerprint Ablation");
    println!("==========================================");
    println!("Input             : {}", INPUT);
    println!("Rows              : {}", EXPECTED_ROWS);
    println!("Warm-up           : {}", WARMUP);
    println!("Scored            : {}", EXPECTED_ROWS - WARMUP);
    println!("Buckets/model     : {}", BUCKETS);
    println!("Smoothing alpha   : {:.3}", ALPHA);
    println!("Target            : historical winning nonce L9");
    println!("Target bits       : nonce bits 7..5");
    println!("Walk-forward      : YES");
    println!("Future leakage    : NONE");
    println!("Winning nonce input: EXCLUDED");
    println!("Winning hash input : EXCLUDED");
    println!("Purpose           : process fingerprint attribution");
    println!();

    // ========================================================
    // LOAD
    // ========================================================

    let mut reader = Reader::from_path(INPUT)?;
    let headers = reader.headers()?.clone();

    let height_idx = headers
        .iter()
        .position(|x| x == "height")
        .ok_or("missing height")?;

    let header_idx = headers
        .iter()
        .position(|x| x == "header_hex")
        .or_else(|| headers.iter().position(|x| x == "header"))
        .ok_or("missing header_hex/header")?;

    let verified_idx =
        headers.iter().position(|x| x == "verified");

    let mut blocks = Vec::with_capacity(EXPECTED_ROWS);

    for (i, rec) in reader.records().enumerate() {
        let rec = rec?;

        let height: u64 = rec[height_idx].parse()?;

        let expected = START_HEIGHT + i as u64;

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
            let v = rec[v_idx].to_ascii_lowercase();

            if !matches!(v.as_str(), "true" | "1" | "yes") {
                return Err(
                    format!("unverified {}", height).into()
                );
            }
        }

        let raw = hex::decode(&rec[header_idx])?;

        if raw.len() != 80 {
            return Err(
                format!("header length != 80 at {}", height).into()
            );
        }

        let mut header = [0u8; 80];
        header.copy_from_slice(&raw);

        let nonce = u32::from_le_bytes(
            header[76..80].try_into()?
        );

        let target = ((nonce >> 5) & 0b111) as usize;

        blocks.push(Block {
            height,
            header,
            target,
        });
    }

    if blocks.len() != EXPECTED_ROWS {
        return Err(
            format!(
                "expected {}, got {}",
                EXPECTED_ROWS,
                blocks.len()
            )
            .into()
        );
    }

    println!("DATASET VALIDATION");
    println!("------------------");
    println!("Rows        : {}", blocks.len());
    println!(
        "Range       : {}..{}",
        blocks.first().unwrap().height,
        blocks.last().unwrap().height
    );
    println!("Chronology  : PASS");
    println!("Header size : 80 bytes");
    println!();

    // ========================================================
    // MODELS
    // ========================================================

    let mut bucket_models: Vec<BucketModel> =
        MODELS.iter().map(|_| BucketModel::new()).collect();

    let mut scores: Vec<Score> =
        MODELS.iter().map(|_| Score::default()).collect();

    let mut prior_counts = [0u64; STATES];
    let mut prior_total = 0u64;

    // ========================================================
    // WALK FORWARD
    // ========================================================

    println!("WALK-FORWARD");
    println!("------------");

    for (i, block) in blocks.iter().enumerate() {
        let scoring = i >= WARMUP;

        // Prediction MUST occur before update.
        if scoring {
            for (m_idx, model) in MODELS.iter().enumerate() {
                let p = match model {
                    Model::Prior => {
                        prior_prediction(
                            &prior_counts,
                            prior_total,
                        )
                    }

                    _ => {
                        let bucket =
                            bucket_for(*model, &block.header);

                        bucket_models[m_idx].predict(
                            bucket,
                            &prior_counts,
                            prior_total,
                        )
                    }
                };

                add_score(
                    &mut scores[m_idx],
                    &p,
                    block.target,
                );
            }
        }

        // Only after scoring do we reveal/update target.
        prior_counts[block.target] += 1;
        prior_total += 1;

        for (m_idx, model) in MODELS.iter().enumerate() {
            if matches!(model, Model::Prior) {
                continue;
            }

            let bucket =
                bucket_for(*model, &block.header);

            bucket_models[m_idx].update(
                bucket,
                block.target,
            );
        }

        if (i + 1) % 20_000 == 0 {
            println!(
                "processed {:>6}/{} height {}",
                i + 1,
                EXPECTED_ROWS,
                block.height
            );
        }
    }

    println!();

    // ========================================================
    // RESULTS
    // ========================================================

    println!("MODEL RESULTS");
    println!("-------------");

    println!(
        "{:<16} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11}",
        "Model",
        "R@1",
        "R@2",
        "R@3",
        "R@4",
        "LogLoss",
        "Brier"
    );

    for (i, model) in MODELS.iter().enumerate() {
        let s = &scores[i];

        println!(
            "{:<16} {:>8.3}% {:>8.3}% {:>8.3}% {:>8.3}% {:>11.6} {:>11.6}",
            model.name(),
            s.hits[1] as f64 / s.n as f64 * 100.0,
            s.hits[2] as f64 / s.n as f64 * 100.0,
            s.hits[3] as f64 / s.n as f64 * 100.0,
            s.hits[4] as f64 / s.n as f64 * 100.0,
            s.logloss / s.n as f64,
            s.brier / s.n as f64,
        );
    }

    println!();

    // ========================================================
    // RELATIVE TO PRIOR
    // ========================================================

    let prior = &scores[0];

    println!("GAIN VS WALK-FORWARD PRIOR");
    println!("--------------------------");

    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Model",
        "G@1",
        "G@2",
        "G@3",
        "G@4",
        "ΔLogLoss"
    );

    for i in 1..MODELS.len() {
        let s = &scores[i];

        let mut gain = [0.0; 5];

        for k in 1..=4 {
            let a =
                s.hits[k] as f64 / s.n as f64;

            let b =
                prior.hits[k] as f64 / prior.n as f64;

            gain[k] = a / b;
        }

        let ll =
            s.logloss / s.n as f64;

        let prior_ll =
            prior.logloss / prior.n as f64;

        println!(
            "{:<16} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>+12.8}",
            MODELS[i].name(),
            gain[1],
            gain[2],
            gain[3],
            gain[4],
            ll - prior_ll
        );
    }

    println!();

    // ========================================================
    // OCCUPANCY
    // ========================================================

    println!("BUCKET OCCUPANCY");
    println!("----------------");

    for (i, model) in MODELS.iter().enumerate() {
        if matches!(model, Model::Prior) {
            continue;
        }

        let bm = &bucket_models[i];

        let used =
            bm.totals.iter().filter(|&&n| n > 0).count();

        let ge10 =
            bm.totals.iter().filter(|&&n| n >= 10).count();

        let ge100 =
            bm.totals.iter().filter(|&&n| n >= 100).count();

        let max =
            bm.totals.iter().copied().max().unwrap_or(0);

        println!(
            "{:<16} used={:>4}/{:<4} >=10={:>4} >=100={:>4} max={}",
            model.name(),
            used,
            BUCKETS,
            ge10,
            ge100,
            max
        );
    }

    println!();

    // ========================================================
    // FINAL PRIOR
    // ========================================================

    println!("FINAL HISTORICAL PRIOR");
    println!("----------------------");

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
    println!("A fingerprint family is informative only if");
    println!("it improves OUT-OF-SAMPLE walk-forward");
    println!("performance over the historical PRIOR.");
    println!();
    println!("R@K alone is insufficient.");
    println!("Lower LogLoss/Brier is stronger evidence");
    println!("that the conditional probabilities improved.");
    println!();
    println!("MERKLE/PREVHASH associations do NOT imply");
    println!("cryptographic SHA predictability.");
    println!("They may identify mining systems, epochs,");
    println!("pool construction, or other process effects.");
    println!();
    println!("This experiment predicts HISTORICAL winning");
    println!("nonce geometry, not counterfactual SHA success.");

    println!();

    println!("V0.28 COMPLETE");
    println!("==============");
    println!("Future leakage       : NONE");
    println!("Nonce as feature     : NO");
    println!("Winning hash feature : NO");
    println!("Adaptive tuning      : NONE");
    println!("Primary baseline     : walk-forward PRIOR");

    Ok(())
}
