use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::Path;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[path = "../hhm/mod.rs"]
mod hhm;
use hhm::{AssociativeObserver, HhmMemory};

const HISTORY: &str = "data/bitcoin-history-v020-frozen.csv";
const CELLS: &str = "data/live05b-cells.csv";

const POOL: &str = "sha256.poolbinance.com:443";
const WORKER: &str = "Cl0udB4ck0ff1c3.x1331";
const PASSWORD: &str = "x";

const FREEZE_FILE: &str = "data/live08d1-freezes.jsonl";
const PROSPECTIVE_FILE: &str = "data/live08d1-prospective.jsonl";
const CHECKPOINT_FILE: &str = "data/live08d1-checkpoint.json";
const CHECKPOINT_TMP: &str = "data/live08d1-checkpoint.json.tmp";
const STATE_FILE: &str = "data/live08d1-state.json";
const STATE_TMP: &str = "data/live08d1-state.json.tmp";

const SCHEMA: &str = "x1331-live08d1-v1";
const BLOCKSTREAM: &str = "https://blockstream.info/api";
const POLL_SECONDS: u64 = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CognitiveSnapshot {
    probabilities: Vec<f64>,
    evidence: Vec<f64>,
    observations: Vec<u64>,
    leader: Option<String>,
    recognition: f64,
    stability: f64,
    confidence_internal: f64,
    cognitive_cycles: u64,
    leader_streak: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HhmSnapshot {
    historical_blocks: u64,
    historical_state_observations: u64,
    microcell_rows: u64,
    microcell_samples: u64,
    min_height: Option<u64>,
    max_height: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingFreeze {
    schema: String,
    freeze_id: u64,
    frozen_at: u64,
    incoming: u64,
    job_id: String,
    pool_difficulty: f64,
    prevhash_stratum: String,
    prevhash_canonical: String,
    version: String,
    nbits: String,
    ntime: String,
    notify_fingerprint: String,
    cognition: CognitiveSnapshot,
    hhm: HhmSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlockInfo {
    id: String,
    height: u64,
    version: i64,
    timestamp: u64,
    mediantime: u64,
    nonce: u64,
    bits: u64,
    difficulty: f64,
    merkle_root: String,
    tx_count: u64,
    size: u64,
    weight: u64,
    previousblockhash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RuntimeCheckpoint {
    schema: String,
    saved_at: u64,
    next_freeze_id: u64,
    outcomes_total: u64,
    last_chain_height: u64,
    last_chain_hash: String,
    observer: AssociativeObserver,
    seen_notify_fingerprints: HashSet<String>,
    pending: Vec<PendingFreeze>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn atomic_write(path: &str, tmp: &str, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    {
        let mut f = File::create(tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}

fn save_checkpoint(cp: &mut RuntimeCheckpoint) -> Result<(), Box<dyn Error>> {
    cp.saved_at = now();
    let bytes = serde_json::to_vec_pretty(cp)?;
    atomic_write(CHECKPOINT_FILE, CHECKPOINT_TMP, &bytes)
}

fn save_state(cp: &RuntimeCheckpoint, hhm: &HhmMemory) -> Result<(), Box<dyn Error>> {
    let obj = json!({
        "schema": SCHEMA,
        "saved_at": now(),
        "mode": "PASS+LEARN",
        "auto_play": false,
        "verify": false,
        "submit": false,
        "next_freeze_id": cp.next_freeze_id,
        "outcomes_total": cp.outcomes_total,
        "pending": cp.pending.len(),
        "seen_notify_fingerprints": cp.seen_notify_fingerprints.len(),
        "chain_height": cp.last_chain_height,
        "chain_hash": cp.last_chain_hash,
        "observer": cp.observer,
        "hhm": {
            "historical_blocks": hhm.summary.historical_blocks,
            "historical_state_observations": hhm.summary.historical_state_observations,
            "microcell_rows": hhm.summary.microcell_rows,
            "microcell_samples": hhm.summary.microcell_samples,
            "min_height": hhm.summary.min_height,
            "max_height": hhm.summary.max_height
        }
    });
    atomic_write(
        STATE_FILE,
        STATE_TMP,
        serde_json::to_string_pretty(&obj)?.as_bytes(),
    )
}

fn append_jsonl<T: Serialize>(path: &str, value: &T) -> Result<(), Box<dyn Error>> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut f, value)?;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}

fn canonical_prevhash(stratum: &str) -> Result<String, Box<dyn Error>> {
    let raw = hex::decode(stratum)?;
    if raw.len() != 32 {
        return Err("prevhash must be 32 bytes".into());
    }
    let mut header_repr = Vec::with_capacity(32);
    for word in raw.chunks_exact(4) {
        header_repr.extend(word.iter().rev());
    }
    header_repr.reverse();
    Ok(hex::encode(header_repr))
}

fn notify_fingerprint(params: &Value) -> String {
    let mut h = Sha256::new();
    h.update(serde_json::to_vec(params).unwrap_or_default());
    hex::encode(h.finalize())
}

fn context_signals(params: &Value) -> [f64; 8] {
    let mut material = Vec::new();
    for idx in [1usize, 5, 6, 7] {
        if let Some(s) = params.get(idx).and_then(Value::as_str) {
            material.extend_from_slice(s.as_bytes());
            material.push(0);
        }
    }
    let branches = params
        .get(4)
        .and_then(Value::as_array)
        .map(|x| x.len())
        .unwrap_or(0);
    material.extend_from_slice(&(branches as u64).to_le_bytes());

    let mut out = [0.0; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut h = Sha256::new();
        h.update(&material);
        h.update([i as u8]);
        let d = h.finalize();
        let x = u64::from_le_bytes(d[0..8].try_into().unwrap());
        *slot = (x as f64 / u64::MAX as f64) * 0.001;
    }
    out
}

fn cognition(observer: &AssociativeObserver) -> CognitiveSnapshot {
    CognitiveSnapshot {
        probabilities: observer.states.iter().map(|s| s.probability).collect(),
        evidence: observer.states.iter().map(|s| s.evidence).collect(),
        observations: observer.states.iter().map(|s| s.observations).collect(),
        leader: observer.leader.clone(),
        recognition: observer.recognition_rate,
        stability: observer.stability,
        confidence_internal: observer.confidence,
        cognitive_cycles: observer.cognitive_cycles,
        leader_streak: observer.leader_streak,
    }
}

fn hhm_snapshot(hhm: &HhmMemory) -> HhmSnapshot {
    HhmSnapshot {
        historical_blocks: hhm.summary.historical_blocks,
        historical_state_observations: hhm.summary.historical_state_observations,
        microcell_rows: hhm.summary.microcell_rows,
        microcell_samples: hhm.summary.microcell_samples,
        min_height: hhm.summary.min_height,
        max_height: hhm.summary.max_height,
    }
}

fn load_hhm() -> Result<HhmMemory, Box<dyn Error>> {
    let mut hhm = HhmMemory::new();

    let mut rdr = csv::Reader::from_path(HISTORY)?;
    for row in rdr.records() {
        let r = row?;
        let height: u64 = r.get(0).ok_or("history height")?.parse()?;
        hhm.observe_height(height);
        hhm.summary.historical_blocks += 1;

        for idx in 9..=19 {
            let state = r.get(idx).ok_or("missing x1331 historical state")?;
            hhm.observe_historical_state(state);
        }
    }

    let mut rdr = csv::Reader::from_path(CELLS)?;
    for row in rdr.records() {
        let r = row?;
        let child: usize = r.get(3).ok_or("cell child")?.parse()?;
        let samples: u64 = r.get(6).ok_or("cell samples")?.parse()?;
        let mean_lz: f64 = r.get(7).ok_or("cell mean_lz")?.parse()?;
        let best_lz: u64 = r.get(8).ok_or("cell best_lz")?.parse()?;
        let lz8: u64 = r.get(9).ok_or("cell lz8")?.parse()?;
        let lz12: u64 = r.get(10).ok_or("cell lz12")?.parse()?;
        let lz16: u64 = r.get(11).ok_or("cell lz16")?.parse()?;
        hhm.observe_microcell(child, samples, mean_lz, best_lz, lz8, lz12, lz16);
    }

    Ok(hhm)
}

fn priors(hhm: &HhmMemory) -> [f64; 8] {
    let mut p = [0.0; 8];
    let total = hhm.summary.historical_state_observations as f64;
    for (i, s) in hhm.states.iter().enumerate().take(8) {
        p[i] = if total > 0.0 {
            s.historical_hits as f64 / total
        } else {
            1.0 / 8.0
        };
    }
    p
}

fn client() -> Result<Client, Box<dyn Error>> {
    Ok(Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent("x1331-live08d1/1.0")
        .build()?)
}

fn get_text(http: &Client, url: &str) -> Result<String, Box<dyn Error>> {
    Ok(http.get(url).send()?.error_for_status()?.text()?)
}

fn tip_height(http: &Client) -> Result<u64, Box<dyn Error>> {
    Ok(get_text(http, &format!("{BLOCKSTREAM}/blocks/tip/height"))?
        .trim()
        .parse()?)
}

fn block_by_height(http: &Client, height: u64) -> Result<BlockInfo, Box<dyn Error>> {
    let hash = get_text(http, &format!("{BLOCKSTREAM}/block-height/{height}"))?
        .trim()
        .to_string();
    Ok(http
        .get(format!("{BLOCKSTREAM}/block/{hash}"))
        .send()?
        .error_for_status()?
        .json()?)
}

fn close_against_block(
    cp: &mut RuntimeCheckpoint,
    block: &BlockInfo,
) -> Result<u64, Box<dyn Error>> {
    let parent = block.previousblockhash.as_deref().unwrap_or("");
    let mut keep = Vec::with_capacity(cp.pending.len());
    let mut closed = 0u64;

    for freeze in cp.pending.drain(..) {
        if freeze.prevhash_canonical == parent {
            let record = json!({
                "schema": SCHEMA,
                "relation": "DIRECT_SUCCESSOR",
                "before": freeze,
                "outcome": block,
                "closed_at": now()
            });
            append_jsonl(PROSPECTIVE_FILE, &record)?;
            cp.outcomes_total += 1;
            closed += 1;
        } else {
            keep.push(freeze);
        }
    }
    cp.pending = keep;
    Ok(closed)
}

fn catch_up_chain(http: &Client, cp: &mut RuntimeCheckpoint) -> Result<(), Box<dyn Error>> {
    let tip = tip_height(http)?;

    if cp.last_chain_height == 0 {
        let b = block_by_height(http, tip)?;
        cp.last_chain_height = b.height;
        cp.last_chain_hash = b.id.clone();
        println!("CHAIN BASELINE | height={} hash={}", b.height, b.id);
        return Ok(());
    }

    if tip < cp.last_chain_height {
        println!(
            "CHAIN WARNING | remote tip {} below checkpoint {} | possible reorg/API lag",
            tip, cp.last_chain_height
        );
        return Ok(());
    }

    for height in (cp.last_chain_height + 1)..=tip {
        let b = block_by_height(http, height)?;
        let expected_parent = cp.last_chain_hash.clone();
        let actual_parent = b.previousblockhash.clone().unwrap_or_default();

        if actual_parent != expected_parent {
            println!(
                "REORG/GAP GUARD | height={} expected_parent={} actual_parent={} | STOP CATCH-UP",
                height, expected_parent, actual_parent
            );
            return Err("chain continuity mismatch".into());
        }

        println!(
            "CHAIN OUTCOME | height={} hash={} parent={}",
            b.height, b.id, actual_parent
        );

        let closed = close_against_block(cp, &b)?;
        cp.last_chain_height = b.height;
        cp.last_chain_hash = b.id.clone();

        save_checkpoint(cp)?;

        println!(
            "DATASET CLOSE | block={} closed={} remain_pending={} total_examples={}",
            b.height,
            closed,
            cp.pending.len(),
            cp.outcomes_total
        );
    }

    Ok(())
}

fn make_freeze(
    cp: &mut RuntimeCheckpoint,
    hhm: &HhmMemory,
    params: &Value,
    difficulty: f64,
) -> Result<Option<PendingFreeze>, Box<dyn Error>> {
    let fp = notify_fingerprint(params);
    if cp.seen_notify_fingerprints.contains(&fp) {
        println!("DUPLICATE NOTIFY | fingerprint={} | ignored", &fp[..16]);
        return Ok(None);
    }

    let job_id = params
        .get(0)
        .and_then(Value::as_str)
        .ok_or("notify job_id")?
        .to_string();
    let prev_stratum = params
        .get(1)
        .and_then(Value::as_str)
        .ok_or("notify prevhash")?
        .to_string();
    let version = params
        .get(5)
        .and_then(Value::as_str)
        .ok_or("notify version")?
        .to_string();
    let nbits = params
        .get(6)
        .and_then(Value::as_str)
        .ok_or("notify nbits")?
        .to_string();
    let ntime = params
        .get(7)
        .and_then(Value::as_str)
        .ok_or("notify ntime")?
        .to_string();

    cp.observer.observe_signals(context_signals(params));

    let freeze = PendingFreeze {
        schema: SCHEMA.to_string(),
        freeze_id: cp.next_freeze_id,
        frozen_at: now(),
        incoming: cp.observer.incoming,
        job_id,
        pool_difficulty: difficulty,
        prevhash_stratum: prev_stratum.clone(),
        prevhash_canonical: canonical_prevhash(&prev_stratum)?,
        version,
        nbits,
        ntime,
        notify_fingerprint: fp.clone(),
        cognition: cognition(&cp.observer),
        hhm: hhm_snapshot(hhm),
    };

    // Durable BEFORE boundary:
    // append the immutable freeze first, then persist pending/observer/dedup.
    append_jsonl(FREEZE_FILE, &freeze)?;

    cp.seen_notify_fingerprints.insert(fp);
    cp.pending.push(freeze.clone());
    cp.next_freeze_id += 1;
    save_checkpoint(cp)?;
    save_state(cp, hhm)?;

    Ok(Some(freeze))
}

fn new_checkpoint(observer: AssociativeObserver) -> RuntimeCheckpoint {
    RuntimeCheckpoint {
        schema: SCHEMA.to_string(),
        saved_at: now(),
        next_freeze_id: 1,
        outcomes_total: 0,
        last_chain_height: 0,
        last_chain_hash: String::new(),
        observer,
        seen_notify_fingerprints: HashSet::new(),
        pending: Vec::new(),
    }
}

fn load_checkpoint() -> Result<Option<RuntimeCheckpoint>, Box<dyn Error>> {
    if !Path::new(CHECKPOINT_FILE).exists() {
        return Ok(None);
    }
    let bytes = fs::read(CHECKPOINT_FILE)?;
    let cp: RuntimeCheckpoint = serde_json::from_slice(&bytes)?;
    if cp.schema != SCHEMA {
        return Err(format!("checkpoint schema mismatch: {}", cp.schema).into());
    }
    Ok(Some(cp))
}

fn send_json(writer: &mut TcpStream, value: Value) -> Result<(), Box<dyn Error>> {
    let mut s = serde_json::to_string(&value)?;
    s.push('\n');
    writer.write_all(s.as_bytes())?;
    writer.flush()?;
    Ok(())
}

fn run_stratum(
    hhm: &HhmMemory,
    http: &Client,
    cp: &mut RuntimeCheckpoint,
) -> Result<(), Box<dyn Error>> {
    println!("Connecting to {POOL} ...");
    let stream = TcpStream::connect(POOL)?;
    stream.set_read_timeout(Some(Duration::from_secs(1)))?;
    let mut writer = stream.try_clone()?;
    let mut reader = BufReader::new(stream);

    send_json(
        &mut writer,
        json!({"id":1,"method":"mining.subscribe","params":[]}),
    )?;

    let mut subscribed = false;
    let mut authorized = false;
    let mut difficulty = 0.0f64;
    let mut line = String::new();
    let mut last_poll = now();

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return Err("Stratum connection closed".into()),
            Ok(_) => {
                let msg: Value = match serde_json::from_str(line.trim()) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if msg.get("id").and_then(Value::as_i64) == Some(1) && !subscribed {
                    subscribed = true;
                    println!("SUBSCRIBE OK");
                    send_json(
                        &mut writer,
                        json!({"id":2,"method":"mining.authorize","params":[WORKER,PASSWORD]}),
                    )?;
                } else if msg.get("id").and_then(Value::as_i64) == Some(2) && !authorized {
                    let ok = msg.get("result").and_then(Value::as_bool).unwrap_or(false);
                    println!("AUTHORIZE: {}", if ok { "OK" } else { "FAILED" });
                    if !ok {
                        return Err("authorization failed".into());
                    }
                    authorized = true;
                }

                match msg.get("method").and_then(Value::as_str) {
                    Some("mining.set_difficulty") => {
                        if let Some(d) = msg
                            .get("params")
                            .and_then(Value::as_array)
                            .and_then(|a| a.first())
                            .and_then(Value::as_f64)
                        {
                            difficulty = d;
                            println!("DIFFICULTY: {difficulty}");
                        }
                    }
                    Some("mining.notify") => {
                        let params = msg.get("params").ok_or("notify params")?;
                        if let Some(f) = make_freeze(cp, hhm, params, difficulty)? {
                            println!(
                                "COGNITIVE FREEZE | id={} job={} parent={} leader={:?} conf={:.6} stab={:.6} recog={:.6} BEFORE-OUTCOME",
                                f.freeze_id,
                                f.job_id,
                                f.prevhash_canonical,
                                f.cognition.leader,
                                f.cognition.confidence_internal,
                                f.cognition.stability,
                                f.cognition.recognition
                            );
                        }
                    }
                    _ => {}
                }
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(e) => return Err(e.into()),
        }

        if now().saturating_sub(last_poll) >= POLL_SECONDS {
            if let Err(e) = catch_up_chain(http, cp) {
                println!("CHAIN POLL ERROR: {e}");
            }
            save_checkpoint(cp)?;
            save_state(cp, hhm)?;
            last_poll = now();
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 LIVE-08D.1");
    println!("PERSISTENT PROSPECTIVE COGNITIVE DATASET");
    println!("RULE: COGNITION IS DURABLE AND FROZEN BEFORE NETWORK OUTCOME");
    println!("AUTO PLAY=OFF | VERIFY=OFF | SUBMIT=OFF");
    println!();

    println!("Loading HHM...");
    let hhm = load_hhm()?;
    println!(
        "HHM READY | blocks={} states={} microcells={} SHA-memory={}",
        hhm.summary.historical_blocks,
        hhm.summary.historical_state_observations,
        hhm.summary.microcell_rows,
        hhm.summary.microcell_samples
    );

    if hhm.summary.historical_blocks != 200_000
        || hhm.summary.historical_state_observations != 2_200_000
        || hhm.summary.microcell_rows != 102_400
        || hhm.summary.microcell_samples != 26_214_400
    {
        return Err("HHM bootstrap invariant failed".into());
    }

    let http = client()?;
    let mut cp = match load_checkpoint()? {
        Some(cp) => {
            println!(
                "RESTORE | next_id={} incoming={} cycles={} leader={:?} streak={} pending={} seen={} outcomes={}",
                cp.next_freeze_id,
                cp.observer.incoming,
                cp.observer.cognitive_cycles,
                cp.observer.leader,
                cp.observer.leader_streak,
                cp.pending.len(),
                cp.seen_notify_fingerprints.len(),
                cp.outcomes_total
            );
            cp
        }
        None => {
            println!("NEW SCHRODINGER CHECKPOINT");
            new_checkpoint(AssociativeObserver::new(priors(&hhm)))
        }
    };

    // Resolve every available height since the persisted chain cursor before
    // reconnecting to Binance. Pending freezes remain open unless their
    // canonical parent is exactly the observed block's previous hash.
    catch_up_chain(&http, &mut cp)?;
    save_checkpoint(&mut cp)?;
    save_state(&cp, &hhm)?;

    loop {
        match run_stratum(&hhm, &http, &mut cp) {
            Ok(()) => {}
            Err(e) => println!("STRATUM SESSION ENDED | {e} | reconnect in 5 seconds"),
        }
        save_checkpoint(&mut cp)?;
        save_state(&cp, &hhm)?;
        thread::sleep(Duration::from_secs(5));

        // Catch up chain while disconnected before opening the next session.
        if let Err(e) = catch_up_chain(&http, &mut cp) {
            println!("CHAIN CATCH-UP ERROR: {e}");
        }
    }
}
