use csv::Reader;
use serde_json::{json, Value};
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
const OUTPUT: &str = "data/live01-results.csv";

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

        x[1 + i] =
            ((header[bi] >> bj) & 1) as f32;
    }

    for i in 0..32 {
        x[65 + i] =
            ((nonce >> i) & 1) as f32;
    }

    for i in 0..32 {
        let bi = (i * 37 + 7) % 76;
        let bj = (i * 3 + 1) % 8;

        let hb =
            ((header[bi] >> bj) & 1) as u32;

        let nb =
            (nonce >> i) & 1;

        x[97 + i] =
            (hb ^ nb) as f32;
    }

    for i in 0..32 {
        let bi = (i * 53 + 19) % 76;
        let bj = (i * 7 + 2) % 8;

        let hb =
            ((header[bi] >> bj) & 1) as f32;

        let nb =
            ((nonce >> ((i * 11) % 32)) & 1) as f32;

        x[129 + i] =
            hb * nb;
    }

    x
}

impl Net {
    fn new() -> Self {
        let mut rng = WEIGHT_SEED;

        let mut w1 =
            vec![0.0; INPUTS * H1];

        let mut w2 =
            vec![0.0; H1 * H2];

        let mut w3 =
            vec![0.0; H2];

        let s1 =
            (2.0 / INPUTS as f32).sqrt();

        let s2 =
            (2.0 / H1 as f32).sqrt();

        let s3 =
            (2.0 / H2 as f32).sqrt();

        for w in &mut w1 {
            *w =
                (rand01(&mut rng) * 2.0 - 1.0)
                * s1;
        }

        for w in &mut w2 {
            *w =
                (rand01(&mut rng) * 2.0 - 1.0)
                * s2;
        }

        for w in &mut w3 {
            *w =
                (rand01(&mut rng) * 2.0 - 1.0)
                * s3;
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

    fn forward(
        &self,
        x: &[f32; INPUTS],
    ) -> (Vec<f32>, Vec<f32>, f32) {
        let mut h1 =
            vec![0.0f32; H1];

        for j in 0..H1 {
            let mut z =
                self.b1[j];

            let off =
                j * INPUTS;

            for i in 0..INPUTS {
                z +=
                    self.w1[off + i] * x[i];
            }

            h1[j] =
                z.max(0.0);
        }

        let mut h2 =
            vec![0.0f32; H2];

        for k in 0..H2 {
            let mut z =
                self.b2[k];

            let off =
                k * H1;

            for j in 0..H1 {
                z +=
                    self.w2[off + j] * h1[j];
            }

            h2[k] =
                z.max(0.0);
        }

        let mut z =
            self.b3;

        for k in 0..H2 {
            z +=
                self.w3[k] * h2[k];
        }

        z =
            z.clamp(-20.0, 20.0);

        let p =
            1.0 / (1.0 + (-z).exp());

        (h1, h2, p)
    }

    fn predict(
        &self,
        header: &[u8; 80],
        nonce: u32,
    ) -> f32 {
        let x =
            features(header, nonce);

        self.forward(&x).2
    }

    fn train_one(
        &mut self,
        header: &[u8; 80],
        nonce: u32,
        y: f32,
    ) {
        let x =
            features(header, nonce);

        let (h1, h2, p) =
            self.forward(&x);

        let dz3 =
            p - y;

        let old_w3 =
            self.w3.clone();

        let old_w2 =
            self.w2.clone();

        for k in 0..H2 {
            self.w3[k] -=
                LR * dz3 * h2[k];
        }

        self.b3 -=
            LR * dz3;

        let mut dz2 =
            vec![0.0f32; H2];

        for k in 0..H2 {
            let dh =
                dz3 * old_w3[k];

            dz2[k] =
                if h2[k] > 0.0 {
                    dh
                } else {
                    0.0
                };
        }

        for k in 0..H2 {
            let off =
                k * H1;

            for j in 0..H1 {
                self.w2[off + j] -=
                    LR * dz2[k] * h1[j];
            }

            self.b2[k] -=
                LR * dz2[k];
        }

        let mut dz1 =
            vec![0.0f32; H1];

        for j in 0..H1 {
            let mut dh =
                0.0;

            for k in 0..H2 {
                dh +=
                    dz2[k] *
                    old_w2[k * H1 + j];
            }

            dz1[j] =
                if h1[j] > 0.0 {
                    dh
                } else {
                    0.0
                };
        }

        for j in 0..H1 {
            let off =
                j * INPUTS;

            for i in 0..INPUTS {
                self.w1[off + i] -=
                    LR * dz1[j] * x[i];
            }

            self.b1[j] -=
                LR * dz1[j];
        }
    }
}

fn load_training_headers()
    -> Result<Vec<HeaderTemplate>, Box<dyn Error>>
{
    let mut r =
        Reader::from_path(INPUT)?;

    let cols =
        r.headers()?.clone();

    let hi =
        cols.iter()
            .position(|x| x == "height")
            .ok_or("missing height")?;

    let hh =
        cols.iter()
            .position(|x| x == "header_hex")
            .or_else(|| {
                cols.iter()
                    .position(|x| x == "header")
            })
            .ok_or("missing header")?;

    let vi =
        cols.iter()
            .position(|x| x == "verified");

    let mut out =
        Vec::new();

    for (i, rec) in r.records().enumerate() {
        let rec =
            rec?;

        let height: u64 =
            rec[hi].parse()?;

        if height !=
            START_HEIGHT + i as u64
        {
            return Err(
                format!(
                    "chronology failure at {}",
                    height
                )
                .into(),
            );
        }

        if let Some(v) = vi {
            let s =
                rec[v].to_ascii_lowercase();

            if !matches!(
                s.as_str(),
                "true" | "1" | "yes"
            ) {
                return Err(
                    format!(
                        "unverified {}",
                        height
                    )
                    .into(),
                );
            }
        }

        if i % HEADER_STRIDE != 0 {
            continue;
        }

        if out.len() >= TRAIN_HEADERS {
            break;
        }

        let raw =
            hex::decode(&rec[hh])?;

        if raw.len() != 80 {
            return Err(
                format!(
                    "bad header {}",
                    height
                )
                .into(),
            );
        }

        let mut header =
            [0u8; 80];

        header.copy_from_slice(&raw);

        // Exact v0.31 rule:
        // historical winning nonce is unavailable
        // to the Observer.
        header[76..80].fill(0);

        out.push(
            HeaderTemplate {
                height,
                header,
            }
        );
    }

    if out.len() != TRAIN_HEADERS {
        return Err(
            format!(
                "expected {} training headers, got {}",
                TRAIN_HEADERS,
                out.len()
            )
            .into(),
        );
    }

    Ok(out)
}

fn train_frozen_observer()
    -> Result<Net, Box<dyn Error>>
{
    println!(
        "TRAINING FROZEN v0.31 OBSERVER"
    );
    println!(
        "------------------------------"
    );

    let headers =
        load_training_headers()?;

    println!(
        "Training range : {}..{}",
        headers.first().unwrap().height,
        headers.last().unwrap().height
    );

    println!(
        "Headers        : {}",
        TRAIN_HEADERS
    );

    println!(
        "Candidates/H   : {}",
        TRAIN_CANDIDATES
    );

    println!(
        "Target         : LZ >= {}",
        DIFFICULTY_BITS
    );

    println!();

    let mut net =
        Net::new();

    let mut rng =
        TRAIN_SEED;

    let start =
        Instant::now();

    let mut train_hashes =
        0u64;

    let mut positives =
        0u64;

    for (n, h) in headers.iter().enumerate() {
        for _ in 0..TRAIN_CANDIDATES {
            let nonce =
                next_nonce(&mut rng);

            let hash =
                sha256d(&h.header, nonce);

            let y =
                if leading_zero_bits(&hash)
                    >= DIFFICULTY_BITS
                {
                    positives += 1;
                    1.0
                } else {
                    0.0
                };

            net.train_one(
                &h.header,
                nonce,
                y,
            );

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
    println!(
        "Hashes    : {}",
        train_hashes
    );
    println!(
        "Positives : {}",
        positives
    );
    println!(
        "Rate      : {:.8}",
        positives as f64 /
        train_hashes as f64
    );
    println!(
        "Time      : {:.3}s",
        start.elapsed().as_secs_f64()
    );

    println!();
    println!(
        "OBSERVER FROZEN — ONLINE LEARNING OFF"
    );
    println!();

    Ok(net)
}

fn send_json(
    stream: &mut TcpStream,
    value: Value,
) -> Result<(), Box<dyn Error>> {
    let mut line =
        serde_json::to_string(&value)?;

    line.push('\n');

    stream.write_all(
        line.as_bytes()
    )?;

    stream.flush()?;

    Ok(())
}

fn stratum_prevhash_to_header(
    v: Vec<u8>,
) -> Result<[u8; 32], Box<dyn Error>> {
    if v.len() != 32 {
        return Err(
            format!(
                "expected 32-byte prevhash, got {}",
                v.len()
            )
            .into(),
        );
    }

    let mut out =
        [0u8; 32];

    for word in 0..8 {
        let i =
            word * 4;

        out[i] =
            v[i + 3];

        out[i + 1] =
            v[i + 2];

        out[i + 2] =
            v[i + 1];

        out[i + 3] =
            v[i];
    }

    Ok(out)
}

fn build_header(
    extranonce1: &str,
    extranonce2_size: usize,
    p: &[Value],
) -> Result<([u8; 80], String), Box<dyn Error>> {
    if p.len() < 9 {
        return Err(
            format!(
                "unexpected mining.notify params: {}",
                p.len()
            )
            .into(),
        );
    }

    let prevhash_hex =
        p[1].as_str()
            .ok_or("prevhash")?;

    let coinb1 =
        p[2].as_str()
            .ok_or("coinb1")?;

    let coinb2 =
        p[3].as_str()
            .ok_or("coinb2")?;

    let branches =
        p[4].as_array()
            .ok_or("merkle branches")?;

    let version_hex =
        p[5].as_str()
            .ok_or("version")?;

    let nbits_hex =
        p[6].as_str()
            .ok_or("nbits")?;

    let ntime_hex =
        p[7].as_str()
            .ok_or("ntime")?;

    // Frozen LIVE-01 protocol:
    // exactly one header per mining.notify and
    // extranonce2 = zero.
    let extranonce2 =
        vec![0u8; extranonce2_size];

    let mut coinbase =
        Vec::new();

    coinbase.extend(
        hex::decode(coinb1)?
    );

    coinbase.extend(
        hex::decode(extranonce1)?
    );

    coinbase.extend(
        &extranonce2
    );

    coinbase.extend(
        hex::decode(coinb2)?
    );

    let mut merkle =
        sha256d_bytes(&coinbase);

    for branch in branches {
        let branch_hex =
            branch.as_str()
                .ok_or(
                    "invalid merkle branch"
                )?;

        let branch_bytes =
            hex::decode(branch_hex)?;

        if branch_bytes.len() != 32 {
            return Err(
                "invalid merkle branch length"
                    .into()
            );
        }

        let mut combined =
            Vec::with_capacity(64);

        combined.extend_from_slice(
            &merkle
        );

        combined.extend_from_slice(
            &branch_bytes
        );

        merkle =
            sha256d_bytes(&combined);
    }

    let version =
        hex::decode(version_hex)?;

    let prevhash =
        stratum_prevhash_to_header(
            hex::decode(prevhash_hex)?
        )?;

    let ntime =
        hex::decode(ntime_hex)?;

    let nbits =
        hex::decode(nbits_hex)?;

    if version.len() != 4
        || ntime.len() != 4
        || nbits.len() != 4
    {
        return Err(
            "version/ntime/nbits must each be 4 bytes"
                .into()
        );
    }

    let mut header =
        [0u8; 80];

    let mut v =
        version;

    v.reverse();

    header[0..4]
        .copy_from_slice(&v);

    header[4..36]
        .copy_from_slice(&prevhash);

    // v3 frozen rule:
    // folded binary Merkle digest enters
    // the mining header directly.
    header[36..68]
        .copy_from_slice(&merkle);

    let mut t =
        ntime;

    t.reverse();

    header[68..72]
        .copy_from_slice(&t);

    let mut b =
        nbits;

    b.reverse();

    header[72..76]
        .copy_from_slice(&b);

    header[76..80]
        .copy_from_slice(
            &0u32.to_le_bytes()
        );

    Ok((
        header,
        hex::encode(extranonce2),
    ))
}

fn eval_indices(
    header: &[u8; 80],
    candidates: &[(u32, f32)],
    indices: &[usize],
) -> Score {
    let mut score =
        Score::default();

    let mut best =
        0u32;

    for &idx in indices {
        let nonce =
            candidates[idx].0;

        let hash =
            sha256d(header, nonce);

        let lz =
            leading_zero_bits(&hash);

        score.hashes += 1;

        score.lz_sum +=
            lz as u64;

        best =
            best.max(lz);

        if lz >= DIFFICULTY_BITS {
            score.successes += 1;
        }
    }

    score.best_sum =
        best as u64;

    score
}

fn ensure_log()
    -> Result<File, Box<dyn Error>>
{
    let exists =
        Path::new(OUTPUT).exists();

    let mut file =
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(OUTPUT)?;

    if !exists {
        writeln!(
            file,
            "sequence,job_id,header80,extranonce2,difficulty,clean_jobs,top_hashes,random_hashes,top_lz8,random_lz8,top_lz_sum,random_lz_sum,top_best_lz,random_best_lz,cumulative_top_hashes,cumulative_random_hashes,cumulative_top_lz8,cumulative_random_lz8"
        )?;

        file.flush()?;
    }

    Ok(file)
}

fn main()
    -> Result<(), Box<dyn Error>>
{
    println!(
        "X1331 LIVE-01 — Prospective Binance Pool Test"
    );

    println!(
        "=============================================="
    );

    println!(
        "Observer       : v0.31 frozen"
    );

    println!(
        "Online learning: OFF"
    );

    println!(
        "Candidate pool : {}",
        BLIND_POOL
    );

    println!(
        "TOP / RANDOM   : {} / {}",
        KEEP,
        KEEP
    );

    println!(
        "Primary target : SHA256d LZ >= {}",
        DIFFICULTY_BITS
    );

    println!(
        "Pool           : {}",
        POOL
    );

    println!(
        "Worker         : {}",
        WORKER
    );

    println!(
        "mining.submit  : DISABLED"
    );

    println!(
        "Output         : {}",
        OUTPUT
    );

    println!();

    /*
     * IMPORTANT:
     *
     * Training finishes BEFORE connecting to Binance.
     * Therefore no prospective mining.notify is observed
     * during training.
     */
    let net =
        train_frozen_observer()?;

    println!(
        "Connecting to Binance only after Observer freeze..."
    );

    let mut stream =
        TcpStream::connect(POOL)?;

    stream.set_read_timeout(
        Some(Duration::from_secs(300))
    )?;

    stream.set_write_timeout(
        Some(Duration::from_secs(10))
    )?;

    let reader_stream =
        stream.try_clone()?;

    let mut reader =
        BufReader::new(reader_stream);

    send_json(
        &mut stream,
        json!({
            "id": 1,
            "method": "mining.subscribe",
            "params": ["X1331/LIVE-01"]
        }),
    )?;

    let mut extranonce1 =
        String::new();

    let mut extranonce2_size: usize =
        0;

    let mut authorized =
        false;

    let mut difficulty: Option<f64> =
        None;

    let mut pool_rng =
        BLIND_SEED;

    let mut random_rng =
        RANDOM_SEED;

    let mut sequence =
        0u64;

    let mut cumulative_top =
        Score::default();

    let mut cumulative_random =
        Score::default();

    let mut top_job_wins =
        0u64;

    let mut random_job_wins =
        0u64;

    let mut job_ties =
        0u64;

    let mut log =
        ensure_log()?;

    println!();
    println!(
        "LIVE-01 ARMED — waiting for prospective mining.notify"
    );
    println!();

    loop {
        let mut line =
            String::new();

        let n =
            reader.read_line(&mut line)?;

        if n == 0 {
            return Err(
                "pool closed connection"
                    .into()
            );
        }

        let msg: Value =
            match serde_json::from_str(
                line.trim()
            ) {
                Ok(v) => v,

                Err(e) => {
                    eprintln!(
                        "Invalid JSON: {e}"
                    );
                    continue;
                }
            };

        if msg.get("id")
            .and_then(Value::as_i64)
            == Some(1)
        {
            let result =
                msg.get("result")
                    .and_then(Value::as_array)
                    .ok_or(
                        "invalid mining.subscribe response"
                    )?;

            if result.len() < 3 {
                return Err(
                    "short mining.subscribe result"
                        .into()
                );
            }

            extranonce1 =
                result[1]
                    .as_str()
                    .ok_or(
                        "missing extranonce1"
                    )?
                    .to_string();

            extranonce2_size =
                result[2]
                    .as_u64()
                    .ok_or(
                        "missing extranonce2_size"
                    )?
                    as usize;

            println!(
                "SUBSCRIBE OK | extranonce1={} extranonce2_size={}",
                extranonce1,
                extranonce2_size
            );

            send_json(
                &mut stream,
                json!({
                    "id": 2,
                    "method": "mining.authorize",
                    "params": [WORKER, PASSWORD]
                }),
            )?;

            continue;
        }

        if msg.get("id")
            .and_then(Value::as_i64)
            == Some(2)
        {
            authorized =
                msg.get("result")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);

            println!(
                "AUTHORIZE: {}",
                if authorized {
                    "OK"
                } else {
                    "FAILED"
                }
            );

            if !authorized {
                return Err(
                    "Stratum authorization failed"
                        .into()
                );
            }

            continue;
        }

        let method =
            msg.get("method")
                .and_then(Value::as_str);

        if method ==
            Some("mining.set_difficulty")
        {
            if let Some(params) =
                msg.get("params")
                    .and_then(Value::as_array)
            {
                if let Some(d) =
                    params.first()
                        .and_then(Value::as_f64)
                {
                    difficulty =
                        Some(d);

                    println!(
                        "DIFFICULTY: {}",
                        d
                    );
                }
            }

            continue;
        }

        if method !=
            Some("mining.notify")
        {
            continue;
        }

        if !authorized {
            println!(
                "notify before authorization — ignored"
            );
            continue;
        }

        if extranonce1.is_empty()
            || extranonce2_size == 0
        {
            println!(
                "notify before subscription state — ignored"
            );
            continue;
        }

        let p =
            msg.get("params")
                .and_then(Value::as_array)
                .ok_or(
                    "notify params missing"
                )?;

        if p.len() < 9 {
            return Err(
                "short mining.notify"
                    .into()
            );
        }

        let job_id =
            p[0].as_str()
                .ok_or("job_id")?
                .to_string();

        let clean_jobs =
            p[8].as_bool()
                .unwrap_or(false);

        /*
         * This is the exact moment at which a new
         * prospective experimental observation is
         * constructed.
         */
        let (header, extranonce2) =
            build_header(
                &extranonce1,
                extranonce2_size,
                p,
            )?;

        sequence += 1;

        let mut candidates =
            Vec::with_capacity(
                BLIND_POOL
            );

        /*
         * Frozen v0.31 candidate generation.
         * Prediction occurs BEFORE SHA256d evaluation.
         */
        for _ in 0..BLIND_POOL {
            let nonce =
                next_nonce(
                    &mut pool_rng
                );

            let probability =
                net.predict(
                    &header,
                    nonce
                );

            candidates.push(
                (nonce, probability)
            );
        }

        candidates.sort_by(
            |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(
                        Ordering::Equal
                    )
                    .then_with(
                        || a.0.cmp(&b.0)
                    )
            }
        );

        let top_idx:
            Vec<usize> =
            (0..KEEP).collect();

        /*
         * Preserve the v0.31 RANDOM protocol:
         *
         * sample without replacement from the
         * middle 75%.
         *
         * Although LIVE-01 does not evaluate BOTTOM,
         * we preserve the same middle boundaries
         * used by v0.31.
         */
        let middle_start =
            KEEP;

        let middle_end =
            BLIND_POOL - KEEP;

        let mut middle:
            Vec<usize> =
            (middle_start..middle_end)
                .collect();

        for i in 0..KEEP {
            let remaining =
                middle.len() - i;

            let j =
                i +
                (
                    splitmix64(
                        &mut random_rng
                    ) as usize
                    % remaining
                );

            middle.swap(i, j);
        }

        let random_idx =
            &middle[..KEEP];

        /*
         * Only now are the selected candidates
         * actually SHA256d evaluated.
         */
        let top =
            eval_indices(
                &header,
                &candidates,
                &top_idx,
            );

        let random =
            eval_indices(
                &header,
                &candidates,
                random_idx,
            );

        cumulative_top.hashes +=
            top.hashes;

        cumulative_top.successes +=
            top.successes;

        cumulative_top.lz_sum +=
            top.lz_sum;

        cumulative_top.best_sum +=
            top.best_sum;

        cumulative_random.hashes +=
            random.hashes;

        cumulative_random.successes +=
            random.successes;

        cumulative_random.lz_sum +=
            random.lz_sum;

        cumulative_random.best_sum +=
            random.best_sum;

        if top.successes >
            random.successes
        {
            top_job_wins += 1;
        } else if random.successes >
            top.successes
        {
            random_job_wins += 1;
        } else {
            job_ties += 1;
        }

        let top_rate =
            cumulative_top.successes
                as f64
            /
            cumulative_top.hashes
                as f64;

        let random_rate =
            cumulative_random.successes
                as f64
            /
            cumulative_random.hashes
                as f64;

        let gain =
            if random_rate > 0.0 {
                top_rate /
                random_rate
            } else {
                f64::NAN
            };

        let diff_string =
            difficulty
                .map(
                    |d| d.to_string()
                )
                .unwrap_or_else(
                    || "".to_string()
                );

        writeln!(
            log,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            sequence,
            job_id,
            hex::encode(header),
            extranonce2,
            diff_string,
            clean_jobs,
            top.hashes,
            random.hashes,
            top.successes,
            random.successes,
            top.lz_sum,
            random.lz_sum,
            top.best_sum,
            random.best_sum,
            cumulative_top.hashes,
            cumulative_random.hashes,
            cumulative_top.successes,
            cumulative_random.successes
        )?;

        log.flush()?;

        println!(
            "JOB {:6} {} | TOP {:3} RANDOM {:3} | cumulative {} / {} | G {:.6}",
            sequence,
            job_id,
            top.successes,
            random.successes,
            cumulative_top.successes,
            cumulative_random.successes,
            gain
        );

        println!(
            "           best LZ T/R {} / {} | job W/L/T {} / {} / {}",
            top.best_sum,
            random.best_sum,
            top_job_wins,
            random_job_wins,
            job_ties
        );

        println!(
            "           hashes/arm {} | header {}",
            cumulative_top.hashes,
            hex::encode(header)
        );
    }
}
