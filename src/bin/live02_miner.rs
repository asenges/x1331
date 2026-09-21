use csv::Reader;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    error::Error,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    net::TcpStream,
    path::Path,
    time::{Duration, Instant},
};

const INPUT: &str = "data/bitcoin-history-v020-frozen.csv";
const OUTPUT: &str = "data/live02-audit.csv";
const LEARN_OUTPUT: &str = "data/live02-learn.csv";
const STATE_OUTPUT: &str = "data/live02-state.json";
const MODEL_OUTPUT: &str = "data/live02-observer.bin";

const POOL: &str = "sha256.poolbinance.com:443";
const WORKER: &str = "Cl0udB4ck0ff1c3.x1331";
const PASSWORD: &str = "x";

const START_HEIGHT: u64 = 700_000;
const TRAIN_HEADERS: usize = 1500;
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

    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);

    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);

    z ^ (z >> 31)
}

fn rand01(s: &mut u64) -> f32 {
    let v = splitmix64(s) >> 40;
    v as f32 / ((1u64 << 24) as f32)
}

fn next_nonce(s: &mut u64) -> u32 {
    splitmix64(s) as u32
}

fn sha256d_bytes(data: &[u8]) -> [u8; 32] {
    let first = Sha256::digest(data);
    Sha256::digest(first).into()
}

fn sha256d(header: &[u8; 80], nonce: u32) -> [u8; 32] {
    let mut h = *header;
    h[76..80].copy_from_slice(&nonce.to_le_bytes());

    sha256d_bytes(&h)
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
            b3: -5.5,
        }
    }

    fn forward(&self, x: &[f32; INPUTS]) -> (Vec<f32>, Vec<f32>, f32) {
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

fn load_training_headers() -> Result<Vec<HeaderTemplate>, Box<dyn Error>> {
    let mut r = Reader::from_path(INPUT)?;

    let cols = r.headers()?.clone();

    let hi = cols
        .iter()
        .position(|x| x == "height")
        .ok_or("missing height")?;

    let hh = cols
        .iter()
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

        if out.len() >= TRAIN_HEADERS {
            break;
        }

        let raw = hex::decode(&rec[hh])?;

        if raw.len() != 80 {
            return Err(format!("bad header {}", height).into());
        }

        let mut header = [0u8; 80];

        header.copy_from_slice(&raw);

        // Exact v0.31 rule:
        // historical winning nonce is unavailable
        // to the Observer.
        header[76..80].fill(0);

        out.push(HeaderTemplate { height, header });
    }

    if out.len() != TRAIN_HEADERS {
        return Err(format!(
            "expected {} training headers, got {}",
            TRAIN_HEADERS,
            out.len()
        )
        .into());
    }

    Ok(out)
}

fn train_frozen_observer() -> Result<Net, Box<dyn Error>> {
    println!("TRAINING FROZEN v0.31 OBSERVER");
    println!("------------------------------");

    let headers = load_training_headers()?;

    println!(
        "Training range : {}..{}",
        headers.first().unwrap().height,
        headers.last().unwrap().height
    );

    println!("Headers        : {}", TRAIN_HEADERS);

    println!("Candidates/H   : {}", TRAIN_CANDIDATES);

    println!("Target         : LZ >= {}", DIFFICULTY_BITS);

    println!();

    let mut net = Net::new();

    let mut rng = TRAIN_SEED;

    let start = Instant::now();

    let mut train_hashes = 0u64;

    let mut positives = 0u64;

    for (n, h) in headers.iter().enumerate() {
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
                n + 1,
                TRAIN_HEADERS,
                train_hashes,
                positives
            );
        }
    }

    println!();
    println!("TRAIN COMPLETE");
    println!("Hashes    : {}", train_hashes);
    println!("Positives : {}", positives);
    println!("Rate      : {:.8}", positives as f64 / train_hashes as f64);
    println!("Time      : {:.3}s", start.elapsed().as_secs_f64());

    println!();
    println!("BOOTSTRAP COMPLETE — ADAPTIVE PHASE NOT YET ARMED");
    println!();

    Ok(net)
}

fn send_json(stream: &mut TcpStream, value: Value) -> Result<(), Box<dyn Error>> {
    let mut line = serde_json::to_string(&value)?;

    line.push('\n');

    stream.write_all(line.as_bytes())?;

    stream.flush()?;

    Ok(())
}

fn stratum_prevhash_to_header(v: Vec<u8>) -> Result<[u8; 32], Box<dyn Error>> {
    if v.len() != 32 {
        return Err(format!("expected 32-byte prevhash, got {}", v.len()).into());
    }

    let mut out = [0u8; 32];

    for word in 0..8 {
        let i = word * 4;

        out[i] = v[i + 3];

        out[i + 1] = v[i + 2];

        out[i + 2] = v[i + 1];

        out[i + 3] = v[i];
    }

    Ok(out)
}

fn build_header(
    extranonce1: &str,
    extranonce2_size: usize,
    p: &[Value],
) -> Result<([u8; 80], String), Box<dyn Error>> {
    if p.len() < 9 {
        return Err(format!("unexpected mining.notify params: {}", p.len()).into());
    }

    let prevhash_hex = p[1].as_str().ok_or("prevhash")?;

    let coinb1 = p[2].as_str().ok_or("coinb1")?;

    let coinb2 = p[3].as_str().ok_or("coinb2")?;

    let branches = p[4].as_array().ok_or("merkle branches")?;

    let version_hex = p[5].as_str().ok_or("version")?;

    let nbits_hex = p[6].as_str().ok_or("nbits")?;

    let ntime_hex = p[7].as_str().ok_or("ntime")?;

    // Frozen LIVE-01 protocol:
    // exactly one header per mining.notify and
    // extranonce2 = zero.
    let extranonce2 = vec![0u8; extranonce2_size];

    let mut coinbase = Vec::new();

    coinbase.extend(hex::decode(coinb1)?);

    coinbase.extend(hex::decode(extranonce1)?);

    coinbase.extend(&extranonce2);

    coinbase.extend(hex::decode(coinb2)?);

    let mut merkle = sha256d_bytes(&coinbase);

    for branch in branches {
        let branch_hex = branch.as_str().ok_or("invalid merkle branch")?;

        let branch_bytes = hex::decode(branch_hex)?;

        if branch_bytes.len() != 32 {
            return Err("invalid merkle branch length".into());
        }

        let mut combined = Vec::with_capacity(64);

        combined.extend_from_slice(&merkle);

        combined.extend_from_slice(&branch_bytes);

        merkle = sha256d_bytes(&combined);
    }

    let version = hex::decode(version_hex)?;

    let prevhash = stratum_prevhash_to_header(hex::decode(prevhash_hex)?)?;

    let ntime = hex::decode(ntime_hex)?;

    let nbits = hex::decode(nbits_hex)?;

    if version.len() != 4 || ntime.len() != 4 || nbits.len() != 4 {
        return Err("version/ntime/nbits must each be 4 bytes".into());
    }

    let mut header = [0u8; 80];

    let mut v = version;

    v.reverse();

    header[0..4].copy_from_slice(&v);

    header[4..36].copy_from_slice(&prevhash);

    // v3 frozen rule:
    // folded binary Merkle digest enters
    // the mining header directly.
    header[36..68].copy_from_slice(&merkle);

    let mut t = ntime;

    t.reverse();

    header[68..72].copy_from_slice(&t);

    let mut b = nbits;

    b.reverse();

    header[72..76].copy_from_slice(&b);

    header[76..80].copy_from_slice(&0u32.to_le_bytes());

    Ok((header, hex::encode(extranonce2)))
}

#[derive(Clone, Copy)]
struct Experience {
    nonce: u32,
    probability: f32,
    lz: u32,
    success: bool,
}

fn observe_candidate(net: &Net, header: &[u8; 80], nonce: u32) -> Experience {
    /*
     * Prediction MUST happen before SHA256d.
     */
    let probability = net.predict(header, nonce);

    let hash = sha256d(header, nonce);

    let lz = leading_zero_bits(&hash);

    Experience {
        nonce,
        probability,
        lz,
        success: lz >= DIFFICULTY_BITS,
    }
}

fn learn_experiences(
    net: &mut Net,
    header: &[u8; 80],
    experiences: &[Experience],
) -> (u64, u64, u32) {
    let mut hashes = 0u64;

    let mut positives = 0u64;

    let mut best = 0u32;

    /*
     * SHA/results already exist before any update.
     * Therefore an update cannot affect the
     * observations from this same job.
     */
    for e in experiences {
        hashes += 1;

        if e.success {
            positives += 1;
        }

        best = best.max(e.lz);
    }

    for e in experiences {
        let y = if e.success { 1.0 } else { 0.0 };

        net.train_one(header, e.nonce, y);
    }

    (hashes, positives, best)
}

fn eval_indices(header: &[u8; 80], candidates: &[(u32, f32)], indices: &[usize]) -> Score {
    let mut score = Score::default();

    let mut best = 0u32;

    for &idx in indices {
        let nonce = candidates[idx].0;

        let hash = sha256d(header, nonce);

        let lz = leading_zero_bits(&hash);

        score.hashes += 1;

        score.lz_sum += lz as u64;

        best = best.max(lz);

        if lz >= DIFFICULTY_BITS {
            score.successes += 1;
        }
    }

    score.best_sum = best as u64;

    score
}

fn ensure_log() -> Result<File, Box<dyn Error>> {
    let exists = Path::new(OUTPUT).exists();

    let mut file = OpenOptions::new().create(true).append(true).open(OUTPUT)?;

    if !exists {
        writeln!(
            file,
            "sequence,job_id,header80,extranonce2,difficulty,clean_jobs,top_hashes,random_hashes,top_lz8,random_lz8,top_lz_sum,random_lz_sum,top_best_lz,random_best_lz,cumulative_top_hashes,cumulative_random_hashes,cumulative_top_lz8,cumulative_random_lz8"
        )?;

        file.flush()?;
    }

    Ok(file)
}

fn write_f32(file: &mut File, value: f32) -> Result<(), Box<dyn Error>> {
    file.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_f32(file: &mut File) -> Result<f32, Box<dyn Error>> {
    use std::io::Read;

    let mut b = [0u8; 4];

    file.read_exact(&mut b)?;

    Ok(f32::from_le_bytes(b))
}

fn save_model(net: &Net) -> Result<(), Box<dyn Error>> {
    let tmp = format!("{}.tmp", MODEL_OUTPUT);

    let mut f = File::create(&tmp)?;

    f.write_all(b"X1331L02")?;

    for &v in &net.w1 {
        write_f32(&mut f, v)?;
    }

    for &v in &net.b1 {
        write_f32(&mut f, v)?;
    }

    for &v in &net.w2 {
        write_f32(&mut f, v)?;
    }

    for &v in &net.b2 {
        write_f32(&mut f, v)?;
    }

    for &v in &net.w3 {
        write_f32(&mut f, v)?;
    }

    write_f32(&mut f, net.b3)?;

    f.flush()?;

    std::fs::rename(tmp, MODEL_OUTPUT)?;

    Ok(())
}

fn load_model() -> Result<Net, Box<dyn Error>> {
    use std::io::Read;

    let mut f = File::open(MODEL_OUTPUT)?;

    let mut magic = [0u8; 8];

    f.read_exact(&mut magic)?;

    if &magic != b"X1331L02" {
        return Err("invalid LIVE-02 model".into());
    }

    let mut net = Net {
        w1: vec![0.0; INPUTS * H1],
        b1: vec![0.0; H1],
        w2: vec![0.0; H1 * H2],
        b2: vec![0.0; H2],
        w3: vec![0.0; H2],
        b3: 0.0,
    };

    for v in &mut net.w1 {
        *v = read_f32(&mut f)?;
    }

    for v in &mut net.b1 {
        *v = read_f32(&mut f)?;
    }

    for v in &mut net.w2 {
        *v = read_f32(&mut f)?;
    }

    for v in &mut net.b2 {
        *v = read_f32(&mut f)?;
    }

    for v in &mut net.w3 {
        *v = read_f32(&mut f)?;
    }

    net.b3 = read_f32(&mut f)?;

    Ok(net)
}

fn write_live_state(
    sequence: u64,
    mode: &str,
    pool: &str,
    learn_jobs: u64,
    audit_jobs: u64,
    learn_samples: u64,
    top: &Score,
    random: &Score,
) -> Result<(), Box<dyn Error>> {
    let gain = if random.successes > 0 {
        top.successes as f64 / random.successes as f64
    } else {
        f64::NAN
    };

    let confidence = if audit_jobs < 50 {
        "COLLECTING"
    } else if gain >= 1.05 {
        "POSITIVE-UNCONFIRMED"
    } else if gain <= 0.95 {
        "NEGATIVE-UNCONFIRMED"
    } else {
        "NEUTRAL"
    };

    let tmp = format!("{}.tmp", STATE_OUTPUT);

    let mut f = File::create(&tmp)?;

    writeln!(f, "{{")?;

    writeln!(f, "  \"status\": \"running\",")?;

    writeln!(f, "  \"pool\": \"{}\",", pool)?;

    writeln!(f, "  \"mode\": \"{}\",", mode)?;

    writeln!(f, "  \"sequence\": {},", sequence)?;

    writeln!(f, "  \"learn_jobs\": {},", learn_jobs)?;

    writeln!(f, "  \"audit_jobs\": {},", audit_jobs)?;

    writeln!(f, "  \"learn_samples\": {},", learn_samples)?;

    writeln!(f, "  \"audit_top_hashes\": {},", top.hashes)?;

    writeln!(f, "  \"audit_random_hashes\": {},", random.hashes)?;

    writeln!(f, "  \"audit_top_successes\": {},", top.successes)?;

    writeln!(f, "  \"audit_random_successes\": {},", random.successes)?;

    writeln!(f, "  \"gain\": {:.9},", gain)?;

    writeln!(f, "  \"confidence\": \"{}\"", confidence)?;

    writeln!(f, "}}")?;

    f.flush()?;

    std::fs::rename(tmp, STATE_OUTPUT)?;

    Ok(())
}

/*
 * Bitcoin difficulty-1 target:
 *
 * 00000000ffff0000000000000000000000000000000000000000000000000000
 *
 * Stored here in big-endian numeric order.
 */
const DIFF1_TARGET: [u8; 32] = [
    0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/*
 * Exact division of a 256-bit big-endian integer
 * by an integer Stratum difficulty.
 *
 * Binance has so far assigned integer difficulties
 * (131072 and 32768 in our sessions).
 */
fn share_target_from_difficulty(difficulty: f64) -> Result<[u8; 32], Box<dyn Error>> {
    if !difficulty.is_finite() || difficulty < 1.0 {
        return Err(format!("invalid pool difficulty: {}", difficulty).into());
    }

    if difficulty.fract() != 0.0 {
        return Err(format!(
            "fractional Stratum difficulty {} is not yet supported safely",
            difficulty
        )
        .into());
    }

    if difficulty > u64::MAX as f64 {
        return Err("pool difficulty exceeds u64".into());
    }

    let divisor = difficulty as u64;

    if divisor == 0 {
        return Err("zero pool difficulty".into());
    }

    let mut target = [0u8; 32];

    let mut remainder = 0u128;

    for i in 0..32 {
        let current = (remainder << 8) | DIFF1_TARGET[i] as u128;

        target[i] = (current / divisor as u128) as u8;

        remainder = current % divisor as u128;
    }

    Ok(target)
}

/*
 * SHA256 libraries return the raw digest bytes.
 *
 * Bitcoin's uint256 PoW comparison interprets those
 * bytes as a little-endian integer.  The target above
 * is represented big-endian, therefore compare the
 * digest in reverse byte order.
 */
fn hash_meets_share_target(hash: &[u8; 32], target_be: &[u8; 32]) -> bool {
    for i in 0..32 {
        let hb = hash[31 - i];

        let tb = target_be[i];

        if hb < tb {
            return true;
        }

        if hb > tb {
            return false;
        }
    }

    true
}

fn nonce_submit_hex(nonce: u32) -> String {
    format!("{:08x}", nonce)
}

fn verify_share_target_math() -> Result<(), Box<dyn Error>> {
    let t1 = share_target_from_difficulty(1.0)?;

    if t1 != DIFF1_TARGET {
        return Err("difficulty=1 target self-test failed".into());
    }

    let t131072 = share_target_from_difficulty(131072.0)?;

    println!("SHARE TARGET SELF-TEST");

    println!("difficulty 1      : {}", hex::encode(t1));

    println!("difficulty 131072 : {}", hex::encode(t131072));

    println!("target math       : PASS");

    Ok(())
}

#[derive(Clone)]
struct LiveJob {
    job_id: String,
    header: [u8; 80],
    extranonce2: String,
    ntime: String,
    clean_jobs: bool,
    difficulty: f64,
    target: [u8; 32],
    nonce_cursor: u64,
    nonce_offset: u32,
}

#[derive(Default)]
struct Runtime {
    sequence: u64,
    learn_batches: u64,
    audit_batches: u64,
    learn_samples: u64,

    top: Score,
    random: Score,

    wins: u64,
    losses: u64,
    ties: u64,

    total_live_hashes: u64,

    shares_found: u64,
    shares_submitted: u64,
    shares_accepted: u64,
    shares_rejected: u64,
    shares_stale: u64,

    submit_id: u64,
    random_rng: u64,
}

fn load_runtime() -> Runtime {
    let mut r = Runtime {
        submit_id: 1000,
        random_rng: RANDOM_SEED,
        ..Runtime::default()
    };

    let text = match std::fs::read_to_string(STATE_OUTPUT) {
        Ok(v) => v,
        Err(_) => return r,
    };

    let v: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return r,
    };

    let u = |name: &str| -> u64 { v.get(name).and_then(Value::as_u64).unwrap_or(0) };

    r.sequence = u("sequence");
    r.learn_batches = u("learn_batches");
    r.audit_batches = u("audit_batches");
    r.learn_samples = u("learn_samples");

    r.top.hashes = u("audit_top_hashes");
    r.top.successes = u("audit_top_successes");
    r.top.lz_sum = u("audit_top_lz_sum");
    r.top.best_sum = u("audit_top_best_sum");

    r.random.hashes = u("audit_random_hashes");
    r.random.successes = u("audit_random_successes");
    r.random.lz_sum = u("audit_random_lz_sum");
    r.random.best_sum = u("audit_random_best_sum");

    r.wins = u("wins");
    r.losses = u("losses");
    r.ties = u("ties");

    r.total_live_hashes = u("total_live_hashes");

    r.shares_found = u("shares_found");
    r.shares_submitted = u("shares_submitted");
    r.shares_accepted = u("shares_accepted");
    r.shares_rejected = u("shares_rejected");
    r.shares_stale = u("shares_stale");

    r.submit_id = v.get("submit_id").and_then(Value::as_u64).unwrap_or(1000);

    r.random_rng = v
        .get("random_rng")
        .and_then(Value::as_u64)
        .unwrap_or(RANDOM_SEED);

    r
}

fn save_runtime(
    r: &Runtime,
    mode: &str,
    difficulty: Option<f64>,
    job_id: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let gain = if r.random.successes > 0 {
        r.top.successes as f64 / r.random.successes as f64
    } else {
        0.0
    };

    let state = json!({
        "status": "running",
        "pool": POOL,
        "worker": WORKER,
        "mode": mode,

        "sequence": r.sequence,
        "learn_batches": r.learn_batches,
        "audit_batches": r.audit_batches,
        "learn_samples": r.learn_samples,

        "audit_top_hashes": r.top.hashes,
        "audit_random_hashes": r.random.hashes,
        "audit_top_successes": r.top.successes,
        "audit_random_successes": r.random.successes,
        "audit_top_lz_sum": r.top.lz_sum,
        "audit_random_lz_sum": r.random.lz_sum,
        "audit_top_best_sum": r.top.best_sum,
        "audit_random_best_sum": r.random.best_sum,

        "wins": r.wins,
        "losses": r.losses,
        "ties": r.ties,

        "gain": gain,

        "total_live_hashes":
            r.total_live_hashes,

        "shares_found":
            r.shares_found,

        "shares_submitted":
            r.shares_submitted,

        "shares_accepted":
            r.shares_accepted,

        "shares_rejected":
            r.shares_rejected,

        "shares_stale":
            r.shares_stale,

        "submit_id":
            r.submit_id,

        "random_rng":
            r.random_rng,

        "difficulty":
            difficulty,

        "job_id":
            job_id
    });

    let tmp = format!("{}.tmp", STATE_OUTPUT);

    std::fs::write(&tmp, serde_json::to_vec_pretty(&state)?)?;

    std::fs::rename(tmp, STATE_OUTPUT)?;

    Ok(())
}

fn ensure_share_log() -> Result<File, Box<dyn Error>> {
    const SHARE_LOG: &str = "data/live02-shares.csv";

    let exists = Path::new(SHARE_LOG).exists();

    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(SHARE_LOG)?;

    if !exists {
        writeln!(
            f,
            "submit_id,job_id,extranonce2,ntime,nonce,difficulty,result"
        )?;

        f.flush()?;
    }

    Ok(f)
}

fn job_offset(job_id: &str, header: &[u8; 80]) -> u32 {
    let mut x = 0x1331_1331u32;

    for &b in job_id.as_bytes() {
        x = x.rotate_left(5) ^ b as u32;

        x = x.wrapping_mul(0x9E3779B1);
    }

    for &b in &header[..16] {
        x = x.rotate_left(3) ^ b as u32;
    }

    x
}

/*
 * Odd multiplier => bijection over u32.
 *
 * Therefore, while counter < 2^32,
 * a nonce cannot repeat within the job.
 */
fn job_nonce(job: &LiveJob, counter: u64) -> u32 {
    (counter as u32)
        .wrapping_mul(0x9E3779B1)
        .wrapping_add(job.nonce_offset)
}

fn append_learn_log(
    sequence: u64,
    job: &LiveJob,
    hashes: u64,
    positives: u64,
    best: u32,
) -> Result<(), Box<dyn Error>> {
    let exists = Path::new(LEARN_OUTPUT).exists();

    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(LEARN_OUTPUT)?;

    if !exists {
        writeln!(f, "sequence,job_id,hashes,positives,best_lz,difficulty")?;
    }

    writeln!(
        f,
        "{},{},{},{},{},{}",
        sequence, job.job_id, hashes, positives, best, job.difficulty
    )?;

    f.flush()?;

    Ok(())
}

fn add_score(dst: &mut Score, src: &Score) {
    dst.hashes += src.hashes;
    dst.successes += src.successes;
    dst.lz_sum += src.lz_sum;
    dst.best_sum += src.best_sum;
}

fn score_experience(score: &mut Score, e: &Experience) {
    score.hashes += 1;
    score.lz_sum += e.lz as u64;

    score.best_sum = score.best_sum.max(e.lz as u64);

    if e.success {
        score.successes += 1;
    }
}

fn submit_share(
    stream: &mut TcpStream,
    r: &mut Runtime,
    share_log: &mut File,
    job: &LiveJob,
    nonce: u32,
) -> Result<(), Box<dyn Error>> {
    r.submit_id += 1;

    let id = r.submit_id;

    let nonce_hex = nonce_submit_hex(nonce);

    send_json(
        stream,
        json!({
            "id": id,
            "method": "mining.submit",
            "params": [
                WORKER,
                job.job_id,
                job.extranonce2,
                job.ntime,
                nonce_hex
            ]
        }),
    )?;

    r.shares_submitted += 1;

    writeln!(
        share_log,
        "{},{},{},{},{},{},SUBMITTED",
        id, job.job_id, job.extranonce2, job.ntime, nonce_hex, job.difficulty
    )?;

    share_log.flush()?;

    println!(
        "SHARE SUBMITTED | id {} | job {} | nonce {} | diff {}",
        id, job.job_id, nonce_hex, job.difficulty
    );

    Ok(())
}

fn hash_selected(
    stream: &mut TcpStream,
    r: &mut Runtime,
    share_log: &mut File,
    net: &Net,
    job: &LiveJob,
    candidates: &[(u32, f32)],
    indices: &[usize],
) -> Result<Vec<Experience>, Box<dyn Error>> {
    let mut out = Vec::with_capacity(indices.len());

    for &idx in indices {
        let nonce = candidates[idx].0;

        /*
         * Prediction already happened before SHA.
         */
        let probability = candidates[idx].1;

        let hash = sha256d(&job.header, nonce);

        r.total_live_hashes += 1;

        let lz = leading_zero_bits(&hash);

        let e = Experience {
            nonce,
            probability,
            lz,
            success: lz >= DIFFICULTY_BITS,
        };

        if hash_meets_share_target(&hash, &job.target) {
            r.shares_found += 1;

            println!(
                "*** POOL SHARE FOUND *** job={} nonce={} raw_hash={} LZ={}",
                job.job_id,
                nonce_submit_hex(nonce),
                hex::encode(hash),
                lz
            );

            submit_share(stream, r, share_log, job, nonce)?;
        }

        out.push(e);
    }

    /*
     * Keep borrow of net semantically explicit:
     * predictions came from this frozen batch model.
     */
    let _ = net;

    Ok(out)
}

fn mine_batch(
    stream: &mut TcpStream,
    net: &mut Net,
    r: &mut Runtime,
    job: &mut LiveJob,
    audit_log: &mut File,
    share_log: &mut File,
) -> Result<(), Box<dyn Error>> {
    r.sequence += 1;

    let sequence = r.sequence;

    /*
     * 9 LEARN : 1 AUDIT.
     *
     * Every tenth batch is untouched prospective
     * evaluation.
     */
    let audit = sequence % 10 == 0;

    if job.nonce_cursor + BLIND_POOL as u64 > (1u64 << 32) {
        return Err("nonce space exhausted for current extranonce2".into());
    }

    let mut candidates = Vec::with_capacity(BLIND_POOL);

    /*
     * Generate 8192 unique candidates.
     * Prediction happens BEFORE any SHA256d.
     */
    for i in 0..BLIND_POOL {
        let counter = job.nonce_cursor + i as u64;

        let nonce = job_nonce(job, counter);

        let p = net.predict(&job.header, nonce);

        candidates.push((nonce, p));
    }

    job.nonce_cursor += BLIND_POOL as u64;

    candidates.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let top_idx: Vec<usize> = (0..KEEP).collect();

    /*
     * RANDOM is sampled without replacement from
     * the middle 75%, preserving the v0.31 control.
     */
    let mut middle: Vec<usize> = (KEEP..BLIND_POOL - KEEP).collect();

    for i in 0..KEEP {
        let remaining = middle.len() - i;

        let j = i + (splitmix64(&mut r.random_rng) as usize % remaining);

        middle.swap(i, j);
    }

    let random_idx = &middle[..KEEP];

    let top_exp = hash_selected(stream, r, share_log, net, job, &candidates, &top_idx)?;

    let random_exp = hash_selected(stream, r, share_log, net, job, &candidates, random_idx)?;

    if audit {
        r.audit_batches += 1;

        let mut ts = Score::default();

        let mut rs = Score::default();

        for e in &top_exp {
            score_experience(&mut ts, e);
        }

        for e in &random_exp {
            score_experience(&mut rs, e);
        }

        add_score(&mut r.top, &ts);

        add_score(&mut r.random, &rs);

        if ts.successes > rs.successes {
            r.wins += 1;
        } else if rs.successes > ts.successes {
            r.losses += 1;
        } else {
            r.ties += 1;
        }

        let gain = if r.random.successes > 0 {
            r.top.successes as f64 / r.random.successes as f64
        } else {
            0.0
        };

        writeln!(
            audit_log,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            sequence,
            job.job_id,
            hex::encode(job.header),
            job.extranonce2,
            job.difficulty,
            job.clean_jobs,
            ts.hashes,
            rs.hashes,
            ts.successes,
            rs.successes,
            ts.lz_sum,
            rs.lz_sum,
            ts.best_sum,
            rs.best_sum,
            r.top.hashes,
            r.random.hashes,
            r.top.successes,
            r.random.successes
        )?;

        audit_log.flush()?;

        println!(
            "AUDIT {:8} {} | TOP {:3} RANDOM {:3} | cumulative {} / {} | G {:.6}",
            sequence,
            job.job_id,
            ts.successes,
            rs.successes,
            r.top.successes,
            r.random.successes,
            gain
        );

        println!(
            "               W/L/T {} / {} / {} | live hashes {}",
            r.wins, r.losses, r.ties, r.total_live_hashes
        );
    } else {
        r.learn_batches += 1;

        /*
         * All SHA observations exist before the
         * first weight update.
         */
        let mut all = Vec::with_capacity(KEEP * 2);

        all.extend_from_slice(&top_exp);

        all.extend_from_slice(&random_exp);

        let mut positives = 0u64;

        let mut best = 0u32;

        for e in &all {
            if e.success {
                positives += 1;
            }

            best = best.max(e.lz);
        }

        /*
         * Learning begins only after every selected
         * result in this batch has been observed.
         */
        for e in &all {
            net.train_one(&job.header, e.nonce, if e.success { 1.0 } else { 0.0 });
        }

        r.learn_samples += all.len() as u64;

        append_learn_log(sequence, job, all.len() as u64, positives, best)?;

        save_model(net)?;

        println!(
            "LEARN {:8} {} | hashes {} | positives {} | rate {:.8} | best LZ {}",
            sequence,
            job.job_id,
            all.len(),
            positives,
            positives as f64 / all.len() as f64,
            best
        );
    }

    save_runtime(
        r,
        if audit { "AUDIT" } else { "LEARN" },
        Some(job.difficulty),
        Some(&job.job_id),
    )?;

    Ok(())
}

fn spawn_reader(stream: TcpStream) -> std::sync::mpsc::Receiver<Value> {
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let mut reader = BufReader::new(stream);

        loop {
            let mut line = String::new();

            match reader.read_line(&mut line) {
                Ok(0) => break,

                Ok(_) => match serde_json::from_str::<Value>(line.trim()) {
                    Ok(v) => {
                        if tx.send(v).is_err() {
                            break;
                        }
                    }

                    Err(e) => {
                        eprintln!("STRATUM invalid JSON: {}", e);
                    }
                },

                Err(e) => {
                    eprintln!("STRATUM reader: {}", e);
                    break;
                }
            }
        }
    });

    rx
}

fn handle_submit_response(
    msg: &Value,
    r: &mut Runtime,
    share_log: &mut File,
) -> Result<bool, Box<dyn Error>> {
    let id = match msg.get("id").and_then(Value::as_u64) {
        Some(v) if v >= 1001 => v,
        _ => return Ok(false),
    };

    let accepted = msg.get("result").and_then(Value::as_bool).unwrap_or(false);

    if accepted {
        r.shares_accepted += 1;

        writeln!(share_log, "{},,,,,,ACCEPTED", id)?;

        println!(
            "*** SHARE ACCEPTED *** id {} | accepted {}",
            id, r.shares_accepted
        );
    } else {
        r.shares_rejected += 1;

        let error = msg.get("error").cloned().unwrap_or(Value::Null);

        let stale = error
            .as_array()
            .and_then(|a| a.first())
            .and_then(Value::as_i64)
            == Some(21);

        if stale {
            r.shares_stale += 1;
        }

        writeln!(share_log, "{},,,,,,REJECTED:{}", id, error)?;

        println!("*** SHARE REJECTED *** id {} | {}", id, error);
    }

    share_log.flush()?;

    Ok(true)
}

fn run_session(net: &mut Net, r: &mut Runtime) -> Result<(), Box<dyn Error>> {
    println!("Connecting to {} ...", POOL);

    let mut stream = TcpStream::connect(POOL)?;

    stream.set_write_timeout(Some(Duration::from_secs(10)))?;

    let reader_stream = stream.try_clone()?;

    let rx = spawn_reader(reader_stream);

    send_json(
        &mut stream,
        json!({
            "id": 1,
            "method": "mining.subscribe",
            "params": [
                "X1331/LIVE-02-MINER"
            ]
        }),
    )?;

    let mut extranonce1 = String::new();

    let mut extranonce2_size = 0usize;

    let mut authorized = false;

    let mut difficulty: Option<f64> = None;

    let mut current_job: Option<LiveJob> = None;

    let mut audit_log = ensure_log()?;

    let mut share_log = ensure_share_log()?;

    loop {
        /*
         * Drain all Stratum traffic before doing
         * another mining batch.
         */
        loop {
            match rx.try_recv() {
                Ok(msg) => {
                    if handle_submit_response(&msg, r, &mut share_log)? {
                        save_runtime(
                            r,
                            "MINING",
                            difficulty,
                            current_job.as_ref().map(|j| j.job_id.as_str()),
                        )?;

                        continue;
                    }

                    let id = msg.get("id").and_then(Value::as_i64);

                    if id == Some(1) {
                        let result = msg
                            .get("result")
                            .and_then(Value::as_array)
                            .ok_or("invalid subscribe response")?;

                        if result.len() < 3 {
                            return Err("short subscribe response".into());
                        }

                        extranonce1 = result[1].as_str().ok_or("missing extranonce1")?.to_string();

                        extranonce2_size =
                            result[2].as_u64().ok_or("missing extranonce2 size")? as usize;

                        println!(
                            "SUBSCRIBE OK | extranonce1={} extranonce2_size={}",
                            extranonce1, extranonce2_size
                        );

                        send_json(
                            &mut stream,
                            json!({
                                "id": 2,
                                "method":
                                    "mining.authorize",
                                "params": [
                                    WORKER,
                                    PASSWORD
                                ]
                            }),
                        )?;

                        continue;
                    }

                    if id == Some(2) {
                        authorized = msg.get("result").and_then(Value::as_bool).unwrap_or(false);

                        println!("AUTHORIZE: {}", if authorized { "OK" } else { "FAILED" });

                        if !authorized {
                            return Err("Stratum authorization failed".into());
                        }

                        continue;
                    }

                    let method = msg.get("method").and_then(Value::as_str);

                    if method == Some("mining.set_difficulty") {
                        if let Some(d) = msg
                            .get("params")
                            .and_then(Value::as_array)
                            .and_then(|p| p.first())
                            .and_then(Value::as_f64)
                        {
                            difficulty = Some(d);

                            println!("DIFFICULTY: {}", d);
                        }

                        continue;
                    }

                    if method == Some("mining.notify") {
                        if !authorized {
                            continue;
                        }

                        let d = match difficulty {
                            Some(v) => v,
                            None => {
                                eprintln!("notify without difficulty; waiting");
                                continue;
                            }
                        };

                        let target = match share_target_from_difficulty(d) {
                            Ok(v) => v,

                            Err(e) => {
                                eprintln!("UNSUPPORTED SHARE TARGET: {}", e);
                                continue;
                            }
                        };

                        let p = msg
                            .get("params")
                            .and_then(Value::as_array)
                            .ok_or("notify params")?;

                        if p.len() < 9 {
                            return Err("short mining.notify".into());
                        }

                        let job_id = p[0].as_str().ok_or("job id")?.to_string();

                        let ntime = p[7].as_str().ok_or("ntime")?.to_string();

                        let clean_jobs = p[8].as_bool().unwrap_or(false);

                        let (header, extranonce2) =
                            build_header(&extranonce1, extranonce2_size, p)?;

                        let offset = job_offset(&job_id, &header);

                        println!(
                            "NEW JOB {} | diff {} | clean={} | offset {:08x}",
                            job_id, d, clean_jobs, offset
                        );

                        /*
                         * Always mine newest work.
                         * clean=true additionally means old
                         * jobs are invalid for submission.
                         */
                        current_job = Some(LiveJob {
                            job_id,
                            header,
                            extranonce2,
                            ntime,
                            clean_jobs,
                            difficulty: d,
                            target,
                            nonce_cursor: 0,
                            nonce_offset: offset,
                        });

                        continue;
                    }
                }

                Err(std::sync::mpsc::TryRecvError::Empty) => break,

                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err("pool closed connection".into());
                }
            }
        }

        if let Some(job) = current_job.as_mut() {
            mine_batch(&mut stream, net, r, job, &mut audit_log, &mut share_log)?;

            continue;
        }

        /*
         * No work yet. Avoid a busy-spin while
         * subscribe/auth/notify arrives.
         */
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(msg) => {
                /*
                 * Put the message through the same
                 * handler on the next iteration by
                 * handling the common early cases here.
                 */
                if handle_submit_response(&msg, r, &mut share_log)? {
                    continue;
                }

                let id = msg.get("id").and_then(Value::as_i64);

                if id == Some(1) {
                    let result = msg
                        .get("result")
                        .and_then(Value::as_array)
                        .ok_or("invalid subscribe response")?;

                    extranonce1 = result
                        .get(1)
                        .and_then(Value::as_str)
                        .ok_or("missing extranonce1")?
                        .to_string();

                    extranonce2_size = result
                        .get(2)
                        .and_then(Value::as_u64)
                        .ok_or("missing extranonce2 size")?
                        as usize;

                    println!(
                        "SUBSCRIBE OK | extranonce1={} extranonce2_size={}",
                        extranonce1, extranonce2_size
                    );

                    send_json(
                        &mut stream,
                        json!({
                            "id": 2,
                            "method":
                                "mining.authorize",
                            "params": [
                                WORKER,
                                PASSWORD
                            ]
                        }),
                    )?;
                } else if id == Some(2) {
                    authorized = msg.get("result").and_then(Value::as_bool).unwrap_or(false);

                    println!("AUTHORIZE: {}", if authorized { "OK" } else { "FAILED" });

                    if !authorized {
                        return Err("authorization failed".into());
                    }
                } else {
                    let method = msg.get("method").and_then(Value::as_str);

                    if method == Some("mining.set_difficulty") {
                        difficulty = msg
                            .get("params")
                            .and_then(Value::as_array)
                            .and_then(|p| p.first())
                            .and_then(Value::as_f64);

                        if let Some(d) = difficulty {
                            println!("DIFFICULTY: {}", d);
                        }
                    }
                }
            }

            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}

            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err("pool closed connection".into());
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    verify_share_target_math()?;

    println!();
    println!("X1331 LIVE-02 — CONTINUOUS ADAPTIVE MINER");
    println!("=========================================");

    println!("Bootstrap       : v0.31 frozen");

    println!("Schedule        : 9 LEARN / 1 AUDIT");

    println!("Candidate pool  : {}", BLIND_POOL);

    println!("SHA/batch       : {} TOP + {} RANDOM", KEEP, KEEP);

    println!("Learning target : SHA256d LZ >= {}", DIFFICULTY_BITS);

    println!("Pool            : {}", POOL);

    println!("Worker          : {}", WORKER);

    println!("mining.submit   : ENABLED");

    println!();

    let reset = std::env::args().any(|a| a == "--reset");

    if reset {
        println!("RESET requested — deleting LIVE model/state");

        let _ = std::fs::remove_file(MODEL_OUTPUT);

        let _ = std::fs::remove_file(STATE_OUTPUT);
    }

    let mut net = if Path::new(MODEL_OUTPUT).exists() {
        println!("RESUME — loading persistent LIVE-02 Observer");

        load_model()?
    } else {
        println!("COLD START — historical education");

        let trained = train_frozen_observer()?;

        save_model(&trained)?;

        trained
    };

    let mut runtime = if reset {
        Runtime {
            submit_id: 1000,
            random_rng: RANDOM_SEED,
            ..Runtime::default()
        }
    } else {
        load_runtime()
    };

    println!(
        "Runtime resume  : sequence={} learn={} audit={} live_hashes={}",
        runtime.sequence, runtime.learn_batches, runtime.audit_batches, runtime.total_live_hashes
    );

    println!(
        "Shares          : found={} submitted={} accepted={} rejected={}",
        runtime.shares_found,
        runtime.shares_submitted,
        runtime.shares_accepted,
        runtime.shares_rejected
    );

    println!();

    /*
     * Reconnect forever without resetting either
     * Observer or runtime counters.
     */
    loop {
        match run_session(&mut net, &mut runtime) {
            Ok(_) => {
                eprintln!("Stratum session ended");
            }

            Err(e) => {
                eprintln!("STRATUM SESSION ERROR: {}", e);
            }
        }

        save_model(&net)?;

        save_runtime(&runtime, "RECONNECTING", None, None)?;

        println!("Reconnect in 5 seconds...");

        std::thread::sleep(Duration::from_secs(5));
    }
}
