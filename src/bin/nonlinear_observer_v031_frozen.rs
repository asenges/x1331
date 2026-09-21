use csv::Reader;
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    error::Error,
    time::Instant,
};

const INPUT: &str = "data/bitcoin-history-v020-frozen.csv";

const START_HEIGHT: u64 = 700_000;
const TRAIN_HEADERS: usize = 1500;
const BLIND_HEADERS: usize = 500;
const HEADER_STRIDE: usize = 100;

const TRAIN_CANDIDATES: usize = 4096;
const BLIND_POOL: usize = 8192;
const KEEP: usize = 1024;

const DIFFICULTY_BITS: u32 = 8;

const INPUTS: usize = 161;
const H1: usize = 128;
const H2: usize = 64;

const LR: f32 = 0.002;
const TRAIN_SEED: u64 = 0x1331_0031_A11C_E001;
const BLIND_SEED: u64 = 0x1331_0031_B11D_E002;
const WEIGHT_SEED: u64 = 0x1331_0031_CAFE_0001;
const RANDOM_SEED: u64 = 0x1331_0031_DEAD_0002;

#[derive(Clone)]
struct HeaderTemplate {
    height: u64,
    header: [u8; 80],
}

struct Net {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: f32,
}

#[derive(Default)]
struct Score {
    hashes: u64,
    successes: u64,
    lz_sum: u64,
    best_sum: u64,
}

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30))
        .wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27))
        .wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn rand01(s: &mut u64) -> f32 {
    let v = splitmix64(s) >> 40;
    v as f32 / ((1u64 << 24) as f32)
}

fn next_nonce(s: &mut u64) -> u32 {
    splitmix64(s) as u32
}

fn sha256d(header: &[u8; 80], nonce: u32) -> [u8; 32] {
    let mut h = *header;
    h[76..80].copy_from_slice(&nonce.to_le_bytes());
    let a = Sha256::digest(h);
    Sha256::digest(a).into()
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

fn features(header: &[u8; 80], nonce: u32) -> [f32; INPUTS] {
    let mut x = [0.0f32; INPUTS];
    x[0] = 1.0;

    for i in 0..64 {
        let bi = (i * 73 + 11) % 76;
        let bj = (i * 5 + 3) % 8;
        x[1 + i] = ((header[bi] >> bj) & 1) as f32;
    }

    for i in 0..32 {
        x[65 + i] = ((nonce >> i) & 1) as f32;
    }

    for i in 0..32 {
        let bi = (i * 37 + 7) % 76;
        let bj = (i * 3 + 1) % 8;
        let hb = ((header[bi] >> bj) & 1) as u32;
        let nb = (nonce >> i) & 1;
        x[97 + i] = (hb ^ nb) as f32;
    }

    for i in 0..32 {
        let bi = (i * 53 + 19) % 76;
        let bj = (i * 7 + 2) % 8;
        let hb = ((header[bi] >> bj) & 1) as f32;
        let nb = ((nonce >> ((i * 11) % 32)) & 1) as f32;
        x[129 + i] = hb * nb;
    }

    x
}

impl Net {
    fn new() -> Self {
        let mut rng = WEIGHT_SEED;

        let mut w1 = vec![0.0; INPUTS * H1];
        let mut w2 = vec![0.0; H1 * H2];
        let mut w3 = vec![0.0; H2];

        let s1 = (2.0 / INPUTS as f32).sqrt();
        let s2 = (2.0 / H1 as f32).sqrt();
        let s3 = (2.0 / H2 as f32).sqrt();

        for w in &mut w1 {
            *w = (rand01(&mut rng) * 2.0 - 1.0) * s1;
        }
        for w in &mut w2 {
            *w = (rand01(&mut rng) * 2.0 - 1.0) * s2;
        }
        for w in &mut w3 {
            *w = (rand01(&mut rng) * 2.0 - 1.0) * s3;
        }

        Self {
            w1,
            b1: vec![0.0; H1],
            w2,
            b2: vec![0.0; H2],
            w3,
            b3: -5.5, // roughly appropriate for p ~ 1/256
        }
    }

    fn forward(&self, x: &[f32; INPUTS])
        -> (Vec<f32>, Vec<f32>, f32)
    {
        let mut h1 = vec![0.0f32; H1];

        for j in 0..H1 {
            let mut z = self.b1[j];
            let off = j * INPUTS;

            for i in 0..INPUTS {
                z += self.w1[off + i] * x[i];
            }

            h1[j] = z.max(0.0);
        }

        let mut h2 = vec![0.0f32; H2];

        for k in 0..H2 {
            let mut z = self.b2[k];
            let off = k * H1;

            for j in 0..H1 {
                z += self.w2[off + j] * h1[j];
            }

            h2[k] = z.max(0.0);
        }

        let mut z = self.b3;
        for k in 0..H2 {
            z += self.w3[k] * h2[k];
        }

        z = z.clamp(-20.0, 20.0);
        let p = 1.0 / (1.0 + (-z).exp());

        (h1, h2, p)
    }

    fn predict(&self, header: &[u8; 80], nonce: u32) -> f32 {
        let x = features(header, nonce);
        self.forward(&x).2
    }

    fn train_one(&mut self, header: &[u8; 80], nonce: u32, y: f32) {
        let x = features(header, nonce);
        let (h1, h2, p) = self.forward(&x);

        // BCE + sigmoid derivative.
        let dz3 = p - y;

        let old_w3 = self.w3.clone();
        let old_w2 = self.w2.clone();

        for k in 0..H2 {
            self.w3[k] -= LR * dz3 * h2[k];
        }
        self.b3 -= LR * dz3;

        let mut dz2 = vec![0.0f32; H2];

        for k in 0..H2 {
            let dh = dz3 * old_w3[k];
            dz2[k] = if h2[k] > 0.0 { dh } else { 0.0 };
        }

        for k in 0..H2 {
            let off = k * H1;
            for j in 0..H1 {
                self.w2[off + j] -= LR * dz2[k] * h1[j];
            }
            self.b2[k] -= LR * dz2[k];
        }

        let mut dz1 = vec![0.0f32; H1];

        for j in 0..H1 {
            let mut dh = 0.0;
            for k in 0..H2 {
                dh += dz2[k] * old_w2[k * H1 + j];
            }
            dz1[j] = if h1[j] > 0.0 { dh } else { 0.0 };
        }

        for j in 0..H1 {
            let off = j * INPUTS;
            for i in 0..INPUTS {
                self.w1[off + i] -= LR * dz1[j] * x[i];
            }
            self.b1[j] -= LR * dz1[j];
        }
    }
}

fn load_headers() -> Result<Vec<HeaderTemplate>, Box<dyn Error>> {
    let mut r = Reader::from_path(INPUT)?;
    let cols = r.headers()?.clone();

    let hi = cols.iter().position(|x| x == "height")
        .ok_or("missing height")?;

    let hh = cols.iter()
        .position(|x| x == "header_hex")
        .or_else(|| cols.iter().position(|x| x == "header"))
        .ok_or("missing header")?;

    let vi = cols.iter().position(|x| x == "verified");

    let mut out = Vec::new();

    for (i, rec) in r.records().enumerate() {
        let rec = rec?;
        let height: u64 = rec[hi].parse()?;

        if height != START_HEIGHT + i as u64 {
            return Err(format!("chronology failure at {}", height).into());
        }

        if let Some(v) = vi {
            let s = rec[v].to_ascii_lowercase();
            if !matches!(s.as_str(), "true" | "1" | "yes") {
                return Err(format!("unverified {}", height).into());
            }
        }

        if i % HEADER_STRIDE != 0 {
            continue;
        }

        let raw = hex::decode(&rec[hh])?;

        if raw.len() != 80 {
            return Err(format!("bad header {}", height).into());
        }

        let mut header = [0u8; 80];
        header.copy_from_slice(&raw);

        // Critical: remove historical winning nonce.
        header[76..80].fill(0);

        out.push(HeaderTemplate { height, header });
    }

    Ok(out)
}

fn eval_arm(
    h: &HeaderTemplate,
    candidates: &[(u32, f32)],
    indices: &[usize],
    score: &mut Score,
) -> u64 {
    let mut successes = 0;
    let mut best = 0;

    for &idx in indices {
        let nonce = candidates[idx].0;
        let hash = sha256d(&h.header, nonce);
        let lz = leading_zero_bits(&hash);

        score.hashes += 1;
        score.lz_sum += lz as u64;
        best = best.max(lz);

        if lz >= DIFFICULTY_BITS {
            score.successes += 1;
            successes += 1;
        }
    }

    score.best_sum += best as u64;
    successes
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 v0.31 Nonlinear Cryptographic Observer");
    println!("============================================");
    println!("Network            : {} -> {} -> {} -> 1", INPUTS, H1, H2);
    println!("Target             : SHA256d LZ >= {}", DIFFICULTY_BITS);
    println!("Train headers      : {}", TRAIN_HEADERS);
    println!("Blind headers      : {}", BLIND_HEADERS);
    println!("Candidate pool/H   : {}", BLIND_POOL);
    println!("Keep/arm/H         : {}", KEEP);
    println!("Arms               : TOP / RANDOM / BOTTOM");
    println!("Historical nonce   : MASKED / UNUSED");
    println!("Winning hash       : UNUSED");
    println!("Status             : DEVELOPMENT");
    println!();

    let headers = load_headers()?;

    if headers.len() != TRAIN_HEADERS + BLIND_HEADERS {
        return Err(format!("expected 2000 headers, got {}", headers.len()).into());
    }

    println!("TRAIN {}..{}",
        headers[0].height,
        headers[TRAIN_HEADERS - 1].height
    );
    println!("BLIND {}..{}",
        headers[TRAIN_HEADERS].height,
        headers.last().unwrap().height
    );
    println!();

    let mut net = Net::new();
    let mut rng = TRAIN_SEED;

    let start = Instant::now();
    let mut train_hashes = 0u64;
    let mut positives = 0u64;

    for (n, h) in headers[..TRAIN_HEADERS].iter().enumerate() {
        for _ in 0..TRAIN_CANDIDATES {
            let nonce = next_nonce(&mut rng);
            let hash = sha256d(&h.header, nonce);
            let y = if leading_zero_bits(&hash) >= DIFFICULTY_BITS {
                positives += 1;
                1.0
            } else {
                0.0
            };

            net.train_one(&h.header, nonce, y);
            train_hashes += 1;
        }

        if (n + 1) % 250 == 0 {
            println!(
                "trained {:4}/{} | hashes {} | positives {}",
                n + 1, TRAIN_HEADERS, train_hashes, positives
            );
        }
    }

    println!();
    println!("TRAIN COMPLETE");
    println!("Hashes    : {}", train_hashes);
    println!("Positives : {}", positives);
    println!(
        "Rate      : {:.8}",
        positives as f64 / train_hashes as f64
    );
    println!("Time      : {:.3}s", start.elapsed().as_secs_f64());
    println!();

    let mut pool_rng = BLIND_SEED;
    let mut random_rng = RANDOM_SEED;

    let mut top = Score::default();
    let mut random = Score::default();
    let mut bottom = Score::default();

    let mut top_gt_random = 0;
    let mut random_gt_top = 0;
    let mut tr_ties = 0;

    let blind_start = Instant::now();

    for (bi, h) in headers[TRAIN_HEADERS..].iter().enumerate() {
        let mut candidates = Vec::with_capacity(BLIND_POOL);

        for _ in 0..BLIND_POOL {
            let nonce = next_nonce(&mut pool_rng);
            let p = net.predict(&h.header, nonce);
            candidates.push((nonce, p));
        }

        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let top_idx: Vec<usize> = (0..KEEP).collect();
        let bottom_idx: Vec<usize> =
            (BLIND_POOL - KEEP..BLIND_POOL).collect();

        // Random arm sampled without replacement from the
        // middle 75% so it is disjoint from TOP and BOTTOM.
        let middle_start = KEEP;
        let middle_end = BLIND_POOL - KEEP;

        let mut middle: Vec<usize> =
            (middle_start..middle_end).collect();

        // Partial Fisher-Yates.
        for i in 0..KEEP {
            let remaining = middle.len() - i;
            let j = i +
                (splitmix64(&mut random_rng) as usize % remaining);
            middle.swap(i, j);
        }

        let random_idx = &middle[..KEEP];

        let ts = eval_arm(h, &candidates, &top_idx, &mut top);
        let rs = eval_arm(h, &candidates, random_idx, &mut random);
        let _bs = eval_arm(h, &candidates, &bottom_idx, &mut bottom);

        if ts > rs {
            top_gt_random += 1;
        } else if rs > ts {
            random_gt_top += 1;
        } else {
            tr_ties += 1;
        }

        if (bi + 1) % 100 == 0 {
            println!(
                "blind {:3}/{} | TOP {} RANDOM {} BOTTOM {}",
                bi + 1,
                BLIND_HEADERS,
                top.successes,
                random.successes,
                bottom.successes
            );
        }
    }

    let rate = |s: &Score| -> f64 {
        s.successes as f64 / s.hashes as f64
    };

    let tr = rate(&top);
    let rr = rate(&random);
    let br = rate(&bottom);

    println!();
    println!("FINAL RESULTS");
    println!("-------------");
    println!("Hashes/arm : {}", top.hashes);
    println!();
    println!("TOP    : {:5}  rate {:.8}", top.successes, tr);
    println!("RANDOM : {:5}  rate {:.8}", random.successes, rr);
    println!("BOTTOM : {:5}  rate {:.8}", bottom.successes, br);
    println!();

    println!("GAIN TOP/RANDOM    : {:.8}x", tr / rr);
    println!("GAIN TOP/BOTTOM    : {:.8}x", tr / br);
    println!("GAIN RANDOM/BOTTOM : {:.8}x", rr / br);
    println!();

    println!(
        "Mean LZ TOP/RANDOM/BOTTOM : {:.6} / {:.6} / {:.6}",
        top.lz_sum as f64 / top.hashes as f64,
        random.lz_sum as f64 / random.hashes as f64,
        bottom.lz_sum as f64 / bottom.hashes as f64
    );

    println!(
        "Mean best LZ/header       : {:.6} / {:.6} / {:.6}",
        top.best_sum as f64 / BLIND_HEADERS as f64,
        random.best_sum as f64 / BLIND_HEADERS as f64,
        bottom.best_sum as f64 / BLIND_HEADERS as f64
    );

    println!();
    println!("TOP vs RANDOM header score");
    println!("TOP wins    : {}", top_gt_random);
    println!("RANDOM wins : {}", random_gt_top);
    println!("Ties        : {}", tr_ties);

    println!();
    println!("Blind time : {:.3}s", blind_start.elapsed().as_secs_f64());

    println!();
    println!("DECISION RULE");
    println!("-------------");
    println!("Desired ordering: TOP > RANDOM > BOTTOM.");
    println!("Primary mining metric: TOP/RANDOM.");
    println!("TOP/BOTTOM alone is insufficient.");
    println!("Gain approximately 1 means no usable signal.");
    println!();
    println!("Any positive result here remains DEVELOPMENT evidence");
    println!("because heights <= 899999 have already informed X1331.");
    println!("Independent confirmation requires untouched headers > 899999.");
    println!();
    println!("V0.31 COMPLETE");

    Ok(())
}
