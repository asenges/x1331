use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    net::TcpStream,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const POOL: &str = "sha256.poolbinance.com:443";
const WORKER: &str = "Cl0udB4ck0ff1c3.x1331";
const PASSWORD: &str = "x";

const V2_DIR: &str = "data/live10/v2";
const LOG_PATH: &str = "data/live10/v2/live10-v2-experience.jsonl";
const STATE_PATH: &str = "data/live10/v2/live10-v2-state.json";
const STATE_TMP: &str = "data/live10/v2/live10-v2-state.tmp";
const MEM_PATH: &str = "data/live10/v2/live10-v2-mem.json";
const MEM_TMP: &str = "data/live10/v2/live10-v2-mem.tmp";

const STATES: usize = 8;
const CONTEXTS: usize = 512;

// Equal SHA budget.
// 64 candidate nonces per state = 512 SHA256d/job.
const NONCES_PER_STATE: u32 = 64;
const TOTAL_SHA_PER_JOB: u64 = STATES as u64 * NONCES_PER_STATE as u64;

// Do not allow action/collapse until a context has enough
// prospective LIVE experience.
const MIN_CONTEXT_OBS: u64 = 12;

// Even after maturity, require enough observed advantage
// before calling the prediction an operational hypothesis.
const MIN_TOP_PROBABILITY: f64 = 0.30;

#[derive(Clone)]
struct OnlineMem {
    counts: [[u64; STATES]; CONTEXTS],
    totals: [u64; CONTEXTS],
}

impl OnlineMem {
    fn new() -> Self {
        Self {
            counts: [[0u64; STATES]; CONTEXTS],
            totals: [0u64; CONTEXTS],
        }
    }

    fn probabilities(&self, context: usize) -> [f64; STATES] {
        let denominator = self.totals[context] as f64 + STATES as f64;

        let mut p = [0.0f64; STATES];

        for state in 0..STATES {
            p[state] = (self.counts[context][state] as f64 + 1.0) / denominator;
        }

        p
    }

    fn observe(&mut self, context: usize, reality: usize) {
        self.counts[context][reality] = self.counts[context][reality].saturating_add(1);

        self.totals[context] = self.totals[context].saturating_add(1);
    }
}

#[derive(Default)]
struct RuntimeStats {
    jobs: u64,
    sha256d: u64,

    mature_predictions: u64,
    correct_predictions: u64,
    errors: u64,
    passes: u64,

    reward: f64,

    lz8_x1331: u64,
    lz8_control: u64,

    pool_target_x1331: u64,
    pool_target_control: u64,
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn sha256d(data: &[u8]) -> [u8; 32] {
    let first = Sha256::digest(data);
    Sha256::digest(first).into()
}

fn send_json(stream: &mut TcpStream, value: Value) -> Result<(), Box<dyn Error>> {
    let mut line = serde_json::to_string(&value)?;
    line.push('\n');

    stream.write_all(line.as_bytes())?;
    stream.flush()?;

    Ok(())
}

fn stratum_prevhash_to_header(bytes: Vec<u8>) -> Result<[u8; 32], Box<dyn Error>> {
    if bytes.len() != 32 {
        return Err(format!("expected 32-byte prevhash, got {}", bytes.len()).into());
    }

    let mut out = [0u8; 32];

    for word in 0..8 {
        let i = word * 4;

        out[i] = bytes[i + 3];
        out[i + 1] = bytes[i + 2];
        out[i + 2] = bytes[i + 1];
        out[i + 3] = bytes[i];
    }

    Ok(out)
}

#[inline(always)]
fn bit_msb(bytes: &[u8], bit: usize) -> usize {
    let byte = bytes[bit >> 3];
    let shift = 7 - (bit & 7);

    ((byte >> shift) & 1) as usize
}

#[inline(always)]
fn cell3(bytes: &[u8], start_bit: usize) -> usize {
    (bit_msb(bytes, start_bit) << 2)
        | (bit_msb(bytes, start_bit + 1) << 1)
        | bit_msb(bytes, start_bit + 2)
}

// BEFORE-only context.
//
// Three separated real 3-bit figures from the 80-byte job template.
// The nonce bytes are zero at this point and are intentionally excluded.
fn job_context(header: &[u8; 80]) -> usize {
    let a = cell3(header, 0);
    let b = cell3(header, 213);
    let c = cell3(header, 426);

    (a << 6) | (b << 3) | c
}

fn ranked_states(probabilities: &[f64; STATES]) -> [usize; STATES] {
    let mut indices = [0usize, 1, 2, 3, 4, 5, 6, 7];

    indices.sort_by(|a, b| {
        probabilities[*b]
            .partial_cmp(&probabilities[*a])
            .unwrap()
            .then_with(|| a.cmp(b))
    });

    indices
}

fn bitcoin_pow_be(raw_digest: &[u8; 32]) -> [u8; 32] {
    let mut out = *raw_digest;
    out.reverse();
    out
}

fn leading_zero_bits_pow(raw_digest: &[u8; 32]) -> u32 {
    let pow = bitcoin_pow_be(raw_digest);
    let mut total = 0u32;
    for byte in &pow {
        if *byte == 0 {
            total += 8;
        } else {
            total += byte.leading_zeros();
            break;
        }
    }
    total
}

fn pow_hash_less(a_raw: &[u8; 32], b_raw: &[u8; 32]) -> bool {
    bitcoin_pow_be(a_raw) < bitcoin_pow_be(b_raw)
}

// Convert compact nBits to a 32-byte big-endian target.
fn compact_target(bits: u32) -> [u8; 32] {
    let exponent = (bits >> 24) as usize;
    let mantissa = bits & 0x007f_ffff;

    let mut target = [0u8; 32];

    let mantissa_bytes = [
        ((mantissa >> 16) & 0xff) as u8,
        ((mantissa >> 8) & 0xff) as u8,
        (mantissa & 0xff) as u8,
    ];

    if exponent <= 3 {
        let shift = 3 - exponent;
        let value = mantissa >> (8 * shift);

        let bytes = value.to_be_bytes();

        let start = 32 - exponent;

        target[start..].copy_from_slice(&bytes[4 - exponent..]);
    } else if exponent <= 32 {
        let start = 32 - exponent;

        for i in 0..3 {
            if start + i < 32 {
                target[start + i] = mantissa_bytes[i];
            }
        }
    }

    target
}

fn meets_target(raw_digest: &[u8; 32], target_be: &[u8; 32]) -> bool {
    bitcoin_pow_be(raw_digest) <= *target_be
}

fn deterministic_control_state(job_id: &str) -> usize {
    let digest = Sha256::digest(job_id.as_bytes());

    (digest[0] & 7) as usize
}

// Nonces are partitioned by their low 3 bits:
//
// state 000 -> ...,0,8,16,...
// state 001 -> ...,1,9,17,...
//
// Equal number of candidates for every state.
fn nonce_for(state: usize, index: u32) -> u32 {
    index.wrapping_mul(STATES as u32).wrapping_add(state as u32)
}

struct VerifyResult {
    best_state: usize,
    best_hash: [u8; 32],
    best_nonce: u32,

    state_best_hashes: [[u8; 32]; STATES],
    state_best_nonces: [u32; STATES],

    state_lz8: [u64; STATES],
    state_target_hits: [u64; STATES],
}

fn verify_equal_budget(header_template: &[u8; 80], network_target: &[u8; 32]) -> VerifyResult {
    let mut state_best_hashes = [[0xffu8; 32]; STATES];

    let mut state_best_nonces = [0u32; STATES];

    let mut state_lz8 = [0u64; STATES];

    let mut state_target_hits = [0u64; STATES];

    for state in 0..STATES {
        for index in 0..NONCES_PER_STATE {
            let nonce = nonce_for(state, index);

            let mut header = *header_template;

            header[76..80].copy_from_slice(&nonce.to_le_bytes());

            let hash = sha256d(&header);

            if leading_zero_bits_pow(&hash) >= 8 {
                state_lz8[state] += 1;
            }

            if meets_target(&hash, network_target) {
                state_target_hits[state] += 1;
            }

            if pow_hash_less(&hash, &state_best_hashes[state]) {
                state_best_hashes[state] = hash;
                state_best_nonces[state] = nonce;
            }
        }
    }

    let mut best_state = 0usize;

    for state in 1..STATES {
        if pow_hash_less(&state_best_hashes[state], &state_best_hashes[best_state]) {
            best_state = state;
        }
    }

    VerifyResult {
        best_state,
        best_hash: state_best_hashes[best_state],
        best_nonce: state_best_nonces[best_state],
        state_best_hashes,
        state_best_nonces,
        state_lz8,
        state_target_hits,
    }
}

fn append_experience(value: &Value) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = Path::new(LOG_PATH).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_PATH)?;

    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;

    Ok(())
}

fn write_mem(memory: &OnlineMem, journal_events: u64) -> Result<(), Box<dyn Error>> {
    let mut flat = Vec::with_capacity(CONTEXTS * STATES);
    for context in 0..CONTEXTS {
        for state in 0..STATES {
            flat.push(memory.counts[context][state]);
        }
    }
    let value = json!({
        "schema": "x1331-live10-v2-mem",
        "updated_at": unix_time(),
        "journal_events": journal_events,
        "contexts": CONTEXTS,
        "states": STATES,
        "counts": flat,
        "totals": memory.totals.to_vec()
    });
    fs::write(MEM_TMP, serde_json::to_vec_pretty(&value)?)?;
    fs::rename(MEM_TMP, MEM_PATH)?;
    Ok(())
}

fn recover_v2() -> Result<(OnlineMem, RuntimeStats, u64), Box<dyn Error>> {
    let mut memory = OnlineMem::new();
    let mut stats = RuntimeStats::default();
    let mut mismatches = 0u64;

    if !Path::new(LOG_PATH).exists() {
        println!("RECOVERY journal_events=0 mem_observations=0 audit_mismatches=0");
        return Ok((memory, stats, mismatches));
    }

    let file = fs::File::open(LOG_PATH)?;
    let reader = BufReader::new(file);

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let event: Value = serde_json::from_str(&line)
            .map_err(|e| format!("journal line {} invalid JSON: {}", line_no + 1, e))?;

        if event.get("schema").and_then(Value::as_str) != Some("x1331-live10-v2") {
            return Err(format!("journal line {} has wrong schema", line_no + 1).into());
        }

        let context = event
            .get("context")
            .and_then(Value::as_u64)
            .ok_or("journal context missing")? as usize;
        let winner = event
            .pointer("/verify/winner_state")
            .and_then(Value::as_u64)
            .ok_or("journal winner_state missing")? as usize;
        if context >= CONTEXTS || winner >= STATES {
            return Err(format!("journal line {} context/state out of range", line_no + 1).into());
        }

        let before_obs = event
            .get("context_observations_before")
            .and_then(Value::as_u64)
            .ok_or("journal context_observations_before missing")?;
        if memory.totals[context] != before_obs {
            mismatches += 1;
        }

        let expected = memory.probabilities(context);
        if let Some(saved) = event
            .pointer("/before/probabilities")
            .and_then(Value::as_array)
        {
            if saved.len() != STATES {
                mismatches += 1;
            } else {
                for i in 0..STATES {
                    let p = saved[i].as_f64().unwrap_or(f64::NAN);
                    if !p.is_finite() || (p - expected[i]).abs() > 1e-12 {
                        mismatches += 1;
                        break;
                    }
                }
            }
        } else {
            mismatches += 1;
        }

        let collapse = event
            .pointer("/before/collapse")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let correct = event
            .pointer("/learning/correct")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let reward = event
            .pointer("/learning/reward")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);

        stats.jobs += 1;
        stats.sha256d += event
            .pointer("/verify/sha_budget")
            .and_then(Value::as_u64)
            .unwrap_or(TOTAL_SHA_PER_JOB);
        stats.reward += reward;

        if collapse {
            stats.mature_predictions += 1;
            if correct {
                stats.correct_predictions += 1;
            } else {
                stats.errors += 1;
            }
        } else {
            stats.passes += 1;
        }

        let predicted = event
            .pointer("/before/predicted_state")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let control = event
            .pointer("/before/control_state")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;

        if let Some(a) = event.pointer("/verify/state_lz8").and_then(Value::as_array) {
            stats.lz8_x1331 += a.get(predicted).and_then(Value::as_u64).unwrap_or(0);
            stats.lz8_control += a.get(control).and_then(Value::as_u64).unwrap_or(0);
        }
        if let Some(a) = event
            .pointer("/verify/state_target_hits")
            .and_then(Value::as_array)
        {
            stats.pool_target_x1331 += a.get(predicted).and_then(Value::as_u64).unwrap_or(0);
            stats.pool_target_control += a.get(control).and_then(Value::as_u64).unwrap_or(0);
        }

        memory.observe(context, winner);
    }

    let observations: u64 = memory.totals.iter().sum();
    println!("RECOVERY");
    println!("  journal_events    : {}", stats.jobs);
    println!("  mem_observations  : {}", observations);
    println!("  audit_mismatches  : {}", mismatches);

    if mismatches != 0 {
        return Err(format!("recovery audit failed: {} mismatches", mismatches).into());
    }

    write_mem(&memory, stats.jobs)?;
    Ok((memory, stats, mismatches))
}

fn write_state(stats: &RuntimeStats, difficulty: Option<f64>) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = Path::new(STATE_PATH).parent() {
        fs::create_dir_all(parent)?;
    }

    let accuracy = if stats.mature_predictions == 0 {
        0.0
    } else {
        stats.correct_predictions as f64 / stats.mature_predictions as f64
    };

    let value = json!({
        "updated_at": unix_time(),
        "mode": "LIVE10_V2_PERSISTENT_SHADOW",
        "submit": false,
        "jobs": stats.jobs,
        "sha256d": stats.sha256d,
        "sha_per_job": TOTAL_SHA_PER_JOB,
        "mature_predictions": stats.mature_predictions,
        "correct_predictions": stats.correct_predictions,
        "errors": stats.errors,
        "passes": stats.passes,
        "accuracy": accuracy,
        "reward": stats.reward,
        "lz8_x1331": stats.lz8_x1331,
        "lz8_control": stats.lz8_control,
        "pool_target_x1331": stats.pool_target_x1331,
        "pool_target_control": stats.pool_target_control,
        "difficulty": difficulty
    });

    fs::write(STATE_TMP, serde_json::to_vec_pretty(&value)?)?;

    fs::rename(STATE_TMP, STATE_PATH)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("============================================================");
    println!(" X1331 LIVE-10.1 — PERSISTENT ADAPTIVE POOL LEARNING");
    println!(" Binance Stratum V1 / REAL mining.notify");
    println!(" Equal-budget SHA reality");
    println!(" Reward / penalty -> causal MEM update");
    println!(" SUBMIT: DISABLED");
    println!("============================================================");
    println!("Pool              : {POOL}");
    println!("Worker            : {WORKER}");
    println!("States            : {STATES}");
    println!("Nonces/state      : {NONCES_PER_STATE}");
    println!("SHA/job           : {TOTAL_SHA_PER_JOB}");
    println!("Min context obs   : {MIN_CONTEXT_OBS}");
    println!("Action threshold  : {MIN_TOP_PROBABILITY:.3}");
    println!("Experience log    : {LOG_PATH}");
    println!("MEM checkpoint    : {MEM_PATH}");
    println!();

    fs::create_dir_all(V2_DIR)?;

    let (mut memory, mut stats, _) = recover_v2()?;
    println!("  status            : EXACT RECOVERY");
    println!();

    let mut stream = TcpStream::connect(POOL)?;

    stream.set_read_timeout(Some(Duration::from_secs(120)))?;

    stream.set_write_timeout(Some(Duration::from_secs(10)))?;

    let reader_stream = stream.try_clone()?;
    let mut reader = BufReader::new(reader_stream);

    send_json(
        &mut stream,
        json!({
            "id": 1,
            "method": "mining.subscribe",
            "params": ["X1331/LIVE-10.1"]
        }),
    )?;

    let mut extranonce1 = String::new();
    let mut extranonce2_size = 0usize;
    let mut authorized = false;
    let mut difficulty: Option<f64> = None;

    loop {
        let mut line = String::new();

        let bytes = reader.read_line(&mut line)?;

        if bytes == 0 {
            return Err("pool closed connection".into());
        }

        let message: Value = match serde_json::from_str(line.trim()) {
            Ok(value) => value,

            Err(error) => {
                eprintln!("Invalid JSON: {error}");
                continue;
            }
        };

        if message.get("id").and_then(Value::as_i64) == Some(1) {
            let result = message
                .get("result")
                .and_then(Value::as_array)
                .ok_or("invalid mining.subscribe response")?;

            if result.len() < 3 {
                return Err("short mining.subscribe result".into());
            }

            extranonce1 = result[1].as_str().ok_or("missing extranonce1")?.to_string();

            extranonce2_size = result[2].as_u64().ok_or("missing extranonce2_size")? as usize;

            println!("SUBSCRIBE OK");
            println!("  extranonce1      : {}", extranonce1);
            println!("  extranonce2_size : {}", extranonce2_size);

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

        if message.get("id").and_then(Value::as_i64) == Some(2) {
            authorized = message
                .get("result")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            println!("AUTHORIZE: {}", if authorized { "OK" } else { "FAILED" });

            if !authorized {
                return Err("Stratum authorization failed".into());
            }

            continue;
        }

        let method = message.get("method").and_then(Value::as_str);

        if method == Some("mining.set_difficulty") {
            if let Some(params) = message.get("params").and_then(Value::as_array) {
                if let Some(value) = params.first().and_then(Value::as_f64) {
                    difficulty = Some(value);

                    println!("DIFFICULTY: {}", value);
                }
            }

            continue;
        }

        if method != Some("mining.notify") {
            continue;
        }

        if !authorized {
            println!("notify before authorization; ignored");
            continue;
        }

        if extranonce1.is_empty() || extranonce2_size == 0 {
            println!("notify before subscription state complete");
            continue;
        }

        let params = message
            .get("params")
            .and_then(Value::as_array)
            .ok_or("notify params missing")?;

        if params.len() < 9 {
            return Err(format!("unexpected mining.notify params: {}", params.len()).into());
        }

        let job_id = params[0].as_str().ok_or("job_id")?;

        let prevhash_hex = params[1].as_str().ok_or("prevhash")?;

        let coinb1 = params[2].as_str().ok_or("coinb1")?;

        let coinb2 = params[3].as_str().ok_or("coinb2")?;

        let branches = params[4].as_array().ok_or("merkle branches")?;

        let version_hex = params[5].as_str().ok_or("version")?;

        let nbits_hex = params[6].as_str().ok_or("nbits")?;

        let ntime_hex = params[7].as_str().ok_or("ntime")?;

        let clean_jobs = params[8].as_bool().unwrap_or(false);

        // Deterministic extranonce2 in SHADOW mode.
        let extranonce2 = vec![0u8; extranonce2_size];

        let mut coinbase = Vec::new();

        coinbase.extend(hex::decode(coinb1)?);

        coinbase.extend(hex::decode(&extranonce1)?);

        coinbase.extend_from_slice(&extranonce2);

        coinbase.extend(hex::decode(coinb2)?);

        let mut merkle = sha256d(&coinbase);

        for branch in branches {
            let branch_hex = branch.as_str().ok_or("invalid merkle branch")?;

            let branch_bytes = hex::decode(branch_hex)?;

            if branch_bytes.len() != 32 {
                return Err("invalid merkle branch length".into());
            }

            let mut combined = Vec::with_capacity(64);

            combined.extend_from_slice(&merkle);

            combined.extend_from_slice(&branch_bytes);

            merkle = sha256d(&combined);
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

        // BEFORE nonce is explicitly unknown.
        // Zero only represents the template storage value.
        header[76..80].copy_from_slice(&0u32.to_le_bytes());

        // ====================================================
        // BEFORE — ZERO FUTURE KNOWLEDGE
        // ====================================================

        let context = job_context(&header);

        let context_observations = memory.totals[context];

        let probabilities = memory.probabilities(context);

        let ranking = ranked_states(&probabilities);

        let predicted_state = ranking[0];

        let confidence = probabilities[predicted_state];

        let mature = context_observations >= MIN_CONTEXT_OBS;

        let collapse = mature && confidence >= MIN_TOP_PROBABILITY;

        let control_state = deterministic_control_state(job_id);

        // Freeze BEFORE values before any candidate SHA.
        let frozen_probabilities = probabilities;

        let frozen_predicted_state = predicted_state;

        let frozen_control_state = control_state;

        // ====================================================
        // REALITY / VERIFY
        // ====================================================

        let compact = u32::from_be_bytes([
            hex::decode(nbits_hex)?[0],
            hex::decode(nbits_hex)?[1],
            hex::decode(nbits_hex)?[2],
            hex::decode(nbits_hex)?[3],
        ]);

        let network_target = compact_target(compact);

        let verification = verify_equal_budget(&header, &network_target);

        let winner_state = verification.best_state;

        let correct = frozen_predicted_state == winner_state;

        // Proper logarithmic reward:
        //
        // uniform prediction receives ln(1/8).
        // Better calibrated correct probability improves reward.
        // Confident errors are naturally penalized.
        let probability_of_reality = frozen_probabilities[winner_state].max(1e-15);

        let reward = probability_of_reality.ln() - (1.0f64 / STATES as f64).ln();

        // Metrics for X1331 selected arm versus deterministic
        // independent control arm.
        stats.lz8_x1331 += verification.state_lz8[frozen_predicted_state];

        stats.lz8_control += verification.state_lz8[frozen_control_state];

        stats.pool_target_x1331 += verification.state_target_hits[frozen_predicted_state];

        stats.pool_target_control += verification.state_target_hits[frozen_control_state];

        stats.jobs += 1;
        stats.sha256d += TOTAL_SHA_PER_JOB;
        stats.reward += reward;

        if collapse {
            stats.mature_predictions += 1;

            if correct {
                stats.correct_predictions += 1;
            } else {
                stats.errors += 1;
            }
        } else {
            stats.passes += 1;
        }

        // ====================================================
        // EXPERIENCE — LEARN ONLY AFTER REALITY
        // ====================================================

        memory.observe(context, winner_state);

        let event = json!({
            "timestamp": unix_time(),

            "schema": "x1331-live10-v2",
            "job_id": job_id,
            "difficulty": difficulty,
            "clean_jobs": clean_jobs,
            "header80": hex::encode(header),

            "context": context,
            "context_observations_before":
                context_observations,

            "before": {
                "probabilities":
                    frozen_probabilities,
                "predicted_state":
                    frozen_predicted_state,
                "control_state":
                    frozen_control_state,
                "confidence":
                    confidence,
                "mature":
                    mature,
                "collapse":
                    collapse
            },

            "verify": {
                "sha_budget":
                    TOTAL_SHA_PER_JOB,
                "nonces_per_state":
                    NONCES_PER_STATE,

                "winner_state":
                    winner_state,

                "best_nonce":
                    verification.best_nonce,

                "best_hash_raw":
                    hex::encode(verification.best_hash),
                "best_hash_pow_be":
                    hex::encode(bitcoin_pow_be(&verification.best_hash)),

                "state_best_nonces":
                    verification.state_best_nonces,

                "state_best_hashes_raw":
                    verification.state_best_hashes.iter().map(hex::encode).collect::<Vec<_>>(),
                "state_best_hashes_pow_be":
                    verification.state_best_hashes.iter()
                        .map(|h| hex::encode(bitcoin_pow_be(h))).collect::<Vec<_>>(),

                "state_lz8":
                    verification.state_lz8,

                "state_target_hits":
                    verification
                        .state_target_hits
            },

            "learning": {
                "correct":
                    correct,
                "reward":
                    reward,
                "probability_of_reality":
                    probability_of_reality,
                "memory_updated_after_verify":
                    true
            },

            "submit": false
        });

        append_experience(&event)?;
        write_mem(&memory, stats.jobs)?;
        write_state(&stats, difficulty)?;

        let accuracy = if stats.mature_predictions == 0 {
            0.0
        } else {
            stats.correct_predictions as f64 / stats.mature_predictions as f64
        };

        println!();
        println!("JOB {}  #{}", job_id, stats.jobs);

        println!(
            "BEFORE context={} obs={} top={} p={:.4} {}",
            context,
            context_observations,
            frozen_predicted_state,
            confidence,
            if collapse { "COLLAPSE" } else { "PASS" }
        );

        println!(
            "REALITY state={} best_nonce={} LZ={}",
            winner_state,
            verification.best_nonce,
            leading_zero_bits_pow(&verification.best_hash)
        );

        if collapse {
            println!(
                "LEARN COLLAPSE_{} reward={:+.6}",
                if correct { "CORRECT" } else { "ERROR" },
                reward
            );
        } else {
            println!("LEARN PASS_EXPERIENCE reward={:+.6}", reward);
        }

        println!(
            "RUN jobs={} sha={} pass={} mature={} accuracy={:.4}",
            stats.jobs, stats.sha256d, stats.passes, stats.mature_predictions, accuracy
        );

        println!("LZ8 X={} CONTROL={}", stats.lz8_x1331, stats.lz8_control);

        println!(
            "TARGET X={} CONTROL={}",
            stats.pool_target_x1331, stats.pool_target_control
        );

        println!("SUBMIT DISABLED");
    }
}
