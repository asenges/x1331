use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    error::Error,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    path::Path,
    time::{Duration, Instant},
};

const POOL: &str = "sha256.poolbinance.com:443";
const WORKER: &str = "Cl0udB4ck0ff1c3.x1331";
const PASSWORD: &str = "x";

const STATE_OUTPUT: &str = "data/live03-state.json";
const SHARE_LOG: &str = "data/live03-shares.csv";
const MODEL_INPUT: &str = "data/live02-observer.bin";

const INPUTS: usize = 161;
const H1: usize = 128;
const H2: usize = 64;

const POW_CHUNK: u64 = 65_536;
const REGION_COUNT: u32 = 8;
const REGION_SIZE: u64 = (1u64 << 32) / REGION_COUNT as u64;
const REPORT_SECS: u64 = 5;

const DIFF1_TARGET: [u8; 32] = [
    0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

struct Net {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: f32,
}

impl Net {
    fn forward(&self, x: &[f32; INPUTS]) -> f32 {
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
        1.0 / (1.0 + (-z).exp())
    }

    fn predict(&self, header: &[u8; 80], nonce: u32) -> f32 {
        self.forward(&features(header, nonce))
    }
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

fn read_f32(file: &mut File) -> Result<f32, Box<dyn Error>> {
    let mut b = [0u8; 4];
    file.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn load_observer() -> Result<Net, Box<dyn Error>> {
    let mut f = File::open(MODEL_INPUT)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"X1331L02" {
        return Err("invalid LIVE-02 observer model".into());
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

fn extranonce2_hex(counter: u64, size: usize) -> Result<String, Box<dyn Error>> {
    if size == 0 || size > 8 {
        return Err(format!("unsupported extranonce2 size {}", size).into());
    }
    let mut b = vec![0u8; size];
    for i in 0..size {
        b[size - 1 - i] = (counter >> (8 * i)) as u8;
    }
    Ok(hex::encode(b))
}

fn build_header(
    extranonce1: &str,
    extranonce2_hex: &str,
    p: &[Value],
) -> Result<[u8; 80], Box<dyn Error>> {
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

    let mut coinbase = Vec::new();
    coinbase.extend(hex::decode(coinb1)?);
    coinbase.extend(hex::decode(extranonce1)?);
    coinbase.extend(hex::decode(extranonce2_hex)?);
    coinbase.extend(hex::decode(coinb2)?);

    let mut merkle = sha256d_bytes(&coinbase);
    for branch in branches {
        let branch_bytes = hex::decode(branch.as_str().ok_or("invalid merkle branch")?)?;
        if branch_bytes.len() != 32 {
            return Err("invalid merkle branch length".into());
        }
        let mut combined = Vec::with_capacity(64);
        combined.extend_from_slice(&merkle);
        combined.extend_from_slice(&branch_bytes);
        merkle = sha256d_bytes(&combined);
    }

    let mut version = hex::decode(version_hex)?;
    let prevhash = stratum_prevhash_to_header(hex::decode(prevhash_hex)?)?;
    let mut ntime = hex::decode(ntime_hex)?;
    let mut nbits = hex::decode(nbits_hex)?;

    if version.len() != 4 || ntime.len() != 4 || nbits.len() != 4 {
        return Err("version/ntime/nbits must each be 4 bytes".into());
    }

    version.reverse();
    ntime.reverse();
    nbits.reverse();

    let mut header = [0u8; 80];
    header[0..4].copy_from_slice(&version);
    header[4..36].copy_from_slice(&prevhash);
    header[36..68].copy_from_slice(&merkle);
    header[68..72].copy_from_slice(&ntime);
    header[72..76].copy_from_slice(&nbits);
    Ok(header)
}

fn share_target_from_difficulty(difficulty: f64) -> Result<[u8; 32], Box<dyn Error>> {
    if !difficulty.is_finite() || difficulty < 1.0 {
        return Err(format!("invalid pool difficulty: {}", difficulty).into());
    }
    if difficulty.fract() != 0.0 {
        return Err(format!("fractional Stratum difficulty {} unsupported", difficulty).into());
    }
    if difficulty > u64::MAX as f64 {
        return Err("pool difficulty exceeds u64".into());
    }

    let divisor = difficulty as u64;
    let mut target = [0u8; 32];
    let mut remainder = 0u128;

    for i in 0..32 {
        let current = (remainder << 8) | DIFF1_TARGET[i] as u128;
        target[i] = (current / divisor as u128) as u8;
        remainder = current % divisor as u128;
    }
    Ok(target)
}

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
    if share_target_from_difficulty(1.0)? != DIFF1_TARGET {
        return Err("difficulty=1 target self-test failed".into());
    }
    let t = share_target_from_difficulty(131072.0)?;
    println!("SHARE TARGET SELF-TEST");
    println!("difficulty 131072 : {}", hex::encode(t));
    println!("target math       : PASS");
    Ok(())
}

#[derive(Clone)]
struct JobTemplate {
    params: Vec<Value>,
    job_id: String,
    ntime: String,
    clean_jobs: bool,
}

struct LiveJob {
    job_id: String,
    header: [u8; 80],
    extranonce2: String,
    ntime: String,
    clean_jobs: bool,
    difficulty: f64,
    target: [u8; 32],
    region_order: [u8; 8],
    region_index: usize,
    region_cursor: u64,
}

#[derive(Clone)]
struct SubmitMeta {
    job_id: String,
    extranonce2: String,
    ntime: String,
    nonce: String,
    difficulty: f64,
}

#[derive(Default)]
struct Runtime {
    total_hashes: u64,
    session_hashes: u64,
    jobs: u64,
    work_units: u64,
    extranonce2_counter: u64,
    shares_found: u64,
    shares_submitted: u64,
    shares_accepted: u64,
    shares_rejected: u64,
    shares_stale: u64,
    submit_id: u64,
}

fn load_runtime() -> Runtime {
    let mut r = Runtime {
        submit_id: 1000,
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
    let u = |name: &str| v.get(name).and_then(Value::as_u64).unwrap_or(0);
    r.total_hashes = u("total_hashes");
    r.jobs = u("jobs");
    r.work_units = u("work_units");
    r.extranonce2_counter = u("extranonce2_counter");
    r.shares_found = u("shares_found");
    r.shares_submitted = u("shares_submitted");
    r.shares_accepted = u("shares_accepted");
    r.shares_rejected = u("shares_rejected");
    r.shares_stale = u("shares_stale");
    r.submit_id = v.get("submit_id").and_then(Value::as_u64).unwrap_or(1000);
    r
}

fn save_runtime(
    r: &Runtime,
    status: &str,
    difficulty: Option<f64>,
    job: Option<&LiveJob>,
    hashrate: f64,
) -> Result<(), Box<dyn Error>> {
    let state = json!({
        "status": status,
        "pool": POOL,
        "worker": WORKER,
        "engine": "LIVE-03-X1331-POW",
        "total_hashes": r.total_hashes,
        "session_hashes": r.session_hashes,
        "jobs": r.jobs,
        "work_units": r.work_units,
        "extranonce2_counter": r.extranonce2_counter,
        "hashrate_hs": hashrate,
        "shares_found": r.shares_found,
        "shares_submitted": r.shares_submitted,
        "shares_accepted": r.shares_accepted,
        "shares_rejected": r.shares_rejected,
        "shares_stale": r.shares_stale,
        "submit_id": r.submit_id,
        "difficulty": difficulty,
        "job_id": job.map(|j| j.job_id.as_str()),
        "extranonce2": job.map(|j| j.extranonce2.as_str()),
        "region": job.map(|j| j.region_order[j.region_index]),
        "region_cursor": job.map(|j| j.region_cursor),
    });
    let tmp = format!("{}.tmp", STATE_OUTPUT);
    std::fs::write(&tmp, serde_json::to_vec_pretty(&state)?)?;
    std::fs::rename(tmp, STATE_OUTPUT)?;
    Ok(())
}

fn ensure_share_log() -> Result<File, Box<dyn Error>> {
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

fn choose_regions(net: &Net, header: &[u8; 80]) -> [u8; 8] {
    let mut scored = Vec::with_capacity(8);
    for region in 0..8u32 {
        let start = region as u64 * REGION_SIZE;
        let midpoint = start + REGION_SIZE / 2;
        let nonce = midpoint as u32;
        scored.push((region as u8, net.predict(header, nonce)));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut order = [0u8; 8];
    for (i, (r, _)) in scored.into_iter().enumerate() {
        order[i] = r;
    }
    order
}

fn make_live_job(
    observer: &Net,
    template: &JobTemplate,
    extranonce1: &str,
    extranonce2_size: usize,
    extranonce2_counter: u64,
    difficulty: f64,
) -> Result<LiveJob, Box<dyn Error>> {
    let ex2 = extranonce2_hex(extranonce2_counter, extranonce2_size)?;
    let header = build_header(extranonce1, &ex2, &template.params)?;
    let order = choose_regions(observer, &header);
    Ok(LiveJob {
        job_id: template.job_id.clone(),
        header,
        extranonce2: ex2,
        ntime: template.ntime.clone(),
        clean_jobs: template.clean_jobs,
        difficulty,
        target: share_target_from_difficulty(difficulty)?,
        region_order: order,
        region_index: 0,
        region_cursor: 0,
    })
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
                    Err(e) => eprintln!("STRATUM invalid JSON: {}", e),
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

fn submit_share(
    stream: &mut TcpStream,
    r: &mut Runtime,
    pending: &mut HashMap<u64, SubmitMeta>,
    share_log: &mut File,
    job: &LiveJob,
    nonce: u32,
) -> Result<(), Box<dyn Error>> {
    r.submit_id += 1;
    let id = r.submit_id;
    let nonce_hex = nonce_submit_hex(nonce);

    let meta = SubmitMeta {
        job_id: job.job_id.clone(),
        extranonce2: job.extranonce2.clone(),
        ntime: job.ntime.clone(),
        nonce: nonce_hex.clone(),
        difficulty: job.difficulty,
    };

    send_json(
        stream,
        json!({
            "id": id,
            "method": "mining.submit",
            "params": [WORKER, meta.job_id, meta.extranonce2, meta.ntime, meta.nonce]
        }),
    )?;

    r.shares_submitted += 1;
    writeln!(
        share_log,
        "{},{},{},{},{},{},SUBMITTED",
        id, meta.job_id, meta.extranonce2, meta.ntime, meta.nonce, meta.difficulty
    )?;
    share_log.flush()?;
    pending.insert(id, meta);

    println!(
        "*** SHARE SUBMITTED *** id={} job={} nonce={}",
        id, job.job_id, nonce_hex
    );
    Ok(())
}

fn handle_submit_response(
    msg: &Value,
    r: &mut Runtime,
    pending: &mut HashMap<u64, SubmitMeta>,
    share_log: &mut File,
) -> Result<bool, Box<dyn Error>> {
    let id = match msg.get("id").and_then(Value::as_u64) {
        Some(v) => v,
        None => return Ok(false),
    };
    let meta = match pending.remove(&id) {
        Some(v) => v,
        None => return Ok(false),
    };

    let accepted = msg.get("result").and_then(Value::as_bool).unwrap_or(false);
    if accepted {
        r.shares_accepted += 1;
        writeln!(
            share_log,
            "{},{},{},{},{},{},ACCEPTED",
            id, meta.job_id, meta.extranonce2, meta.ntime, meta.nonce, meta.difficulty
        )?;
        println!(
            "*** SHARE ACCEPTED *** id={} accepted={}",
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
        writeln!(
            share_log,
            "{},{},{},{},{},{},REJECTED:{}",
            id, meta.job_id, meta.extranonce2, meta.ntime, meta.nonce, meta.difficulty, error
        )?;
        println!("*** SHARE REJECTED *** id={} {}", id, error);
    }
    share_log.flush()?;
    Ok(true)
}

fn mine_chunk(
    stream: &mut TcpStream,
    r: &mut Runtime,
    pending: &mut HashMap<u64, SubmitMeta>,
    share_log: &mut File,
    job: &mut LiveJob,
) -> Result<bool, Box<dyn Error>> {
    let region = job.region_order[job.region_index] as u64;
    let region_start = region * REGION_SIZE;
    let remaining = REGION_SIZE.saturating_sub(job.region_cursor);
    if remaining == 0 {
        job.region_index += 1;
        job.region_cursor = 0;
        if job.region_index >= 8 {
            return Ok(true);
        }
        return Ok(false);
    }

    let count = POW_CHUNK.min(remaining);
    let start = region_start + job.region_cursor;

    for i in 0..count {
        let nonce = (start + i) as u32;
        let hash = sha256d(&job.header, nonce);
        r.total_hashes += 1;
        r.session_hashes += 1;

        if hash_meets_share_target(&hash, &job.target) {
            r.shares_found += 1;
            println!(
                "*** POOL SHARE FOUND *** job={} ex2={} nonce={} raw_hash={} LZ={}",
                job.job_id,
                job.extranonce2,
                nonce_submit_hex(nonce),
                hex::encode(hash),
                leading_zero_bits(&hash)
            );
            submit_share(stream, r, pending, share_log, job, nonce)?;
        }
    }

    job.region_cursor += count;
    r.work_units += 1;
    Ok(false)
}

fn run_session(observer: &Net, r: &mut Runtime) -> Result<(), Box<dyn Error>> {
    println!("Connecting to {} ...", POOL);
    let mut stream = TcpStream::connect(POOL)?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;

    let rx = spawn_reader(stream.try_clone()?);
    send_json(
        &mut stream,
        json!({
            "id": 1,
            "method": "mining.subscribe",
            "params": ["X1331/LIVE-03-POW"]
        }),
    )?;

    let mut extranonce1 = String::new();
    let mut extranonce2_size = 0usize;
    let mut authorized = false;
    let mut difficulty: Option<f64> = None;
    let mut template: Option<JobTemplate> = None;
    let mut current_job: Option<LiveJob> = None;
    let mut pending: HashMap<u64, SubmitMeta> = HashMap::new();
    let mut share_log = ensure_share_log()?;

    let session_start = Instant::now();
    let mut last_report = Instant::now();
    r.session_hashes = 0;

    loop {
        loop {
            match rx.try_recv() {
                Ok(msg) => {
                    if handle_submit_response(&msg, r, &mut pending, &mut share_log)? {
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
                                "method": "mining.authorize",
                                "params": [WORKER, PASSWORD]
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

                    if method == Some("mining.set_extranonce") {
                        if let Some(p) = msg.get("params").and_then(Value::as_array) {
                            if p.len() >= 2 {
                                extranonce1 = p[0]
                                    .as_str()
                                    .ok_or("set_extranonce extranonce1")?
                                    .to_string();
                                extranonce2_size =
                                    p[1].as_u64().ok_or("set_extranonce size")? as usize;
                                current_job = None;
                                println!(
                                    "SET_EXTRANONCE | extranonce1={} size={}",
                                    extranonce1, extranonce2_size
                                );
                            }
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

                        let p = msg
                            .get("params")
                            .and_then(Value::as_array)
                            .ok_or("notify params")?;
                        if p.len() < 9 {
                            return Err("short mining.notify".into());
                        }

                        let t = JobTemplate {
                            params: p.clone(),
                            job_id: p[0].as_str().ok_or("job id")?.to_string(),
                            ntime: p[7].as_str().ok_or("ntime")?.to_string(),
                            clean_jobs: p[8].as_bool().unwrap_or(false),
                        };

                        r.extranonce2_counter = r.extranonce2_counter.wrapping_add(1);
                        let j = make_live_job(
                            observer,
                            &t,
                            &extranonce1,
                            extranonce2_size,
                            r.extranonce2_counter,
                            d,
                        )?;

                        println!(
                            "NEW JOB {} | diff {} | clean={} | ex2={} | X1331 regions={:?}",
                            j.job_id, d, j.clean_jobs, j.extranonce2, j.region_order
                        );

                        r.jobs += 1;
                        template = Some(t);
                        current_job = Some(j);
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
            let exhausted = mine_chunk(&mut stream, r, &mut pending, &mut share_log, job)?;

            if exhausted {
                let t = template.as_ref().ok_or("missing job template")?;
                let d = difficulty.ok_or("missing difficulty")?;
                r.extranonce2_counter = r.extranonce2_counter.wrapping_add(1);
                let j = make_live_job(
                    observer,
                    t,
                    &extranonce1,
                    extranonce2_size,
                    r.extranonce2_counter,
                    d,
                )?;
                println!(
                    "NONCE SPACE COMPLETE -> new extranonce2={} | X1331 regions={:?}",
                    j.extranonce2, j.region_order
                );
                current_job = Some(j);
            }
        } else {
            std::thread::sleep(Duration::from_millis(50));
        }

        if last_report.elapsed() >= Duration::from_secs(REPORT_SECS) {
            let elapsed = session_start.elapsed().as_secs_f64().max(0.001);
            let hs = r.session_hashes as f64 / elapsed;
            let job_ref = current_job.as_ref();

            println!(
                "POW | {:.0} H/s | session {} | total {} | shares F/S/A/R {}/{}/{}/{} | pending {}",
                hs,
                r.session_hashes,
                r.total_hashes,
                r.shares_found,
                r.shares_submitted,
                r.shares_accepted,
                r.shares_rejected,
                pending.len()
            );

            save_runtime(r, "MINING", difficulty, job_ref, hs)?;
            last_report = Instant::now();
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    verify_share_target_math()?;

    println!();
    println!("X1331 LIVE-03 — CONTINUOUS POW MINER");
    println!("====================================");
    println!("Observer        : LIVE-02 persistent model, inference only");
    println!("Topology        : 8 X1331 nonce regions");
    println!("PoW chunk       : {} hashes", POW_CHUNK);
    println!("Pool            : {}", POOL);
    println!("Worker          : {}", WORKER);
    println!("mining.submit   : ENABLED");
    println!("State           : {}", STATE_OUTPUT);
    println!("Shares          : {}", SHARE_LOG);
    println!();

    let observer = load_observer()?;
    println!("OBSERVER LOADED : {}", MODEL_INPUT);

    let mut runtime = load_runtime();
    println!(
        "Runtime resume  : total_hashes={} jobs={} shares={}/{}/{}",
        runtime.total_hashes,
        runtime.jobs,
        runtime.shares_submitted,
        runtime.shares_accepted,
        runtime.shares_rejected
    );
    println!();

    loop {
        match run_session(&observer, &mut runtime) {
            Ok(_) => eprintln!("Stratum session ended"),
            Err(e) => eprintln!("STRATUM SESSION ERROR: {}", e),
        }

        save_runtime(&runtime, "RECONNECTING", None, None, 0.0)?;
        println!("Reconnect in 5 seconds...");
        std::thread::sleep(Duration::from_secs(5));
    }
}
