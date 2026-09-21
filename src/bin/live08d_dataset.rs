#[path = "../hhm/mod.rs"]
mod hhm;

use hhm::{AssociativeObserver, HhmMemory};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    error::Error,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    net::TcpStream,
    path::Path,
    sync::mpsc,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const HISTORY: &str = "data/bitcoin-history-v020-frozen.csv";
const CELLS: &str = "data/live05b-cells.csv";
const POOL: &str = "sha256.poolbinance.com:443";
const WORKER: &str = "Cl0udB4ck0ff1c3.x1331";
const PASSWORD: &str = "x";
const ESPLORA: &str = "https://blockstream.info/api";

const FREEZES: &str = "data/live08d-freezes.jsonl";
const DATASET: &str = "data/live08d-prospective.jsonl";
const PENDING: &str = "data/live08d-pending.json";
const PENDING_TMP: &str = "data/live08d-pending.json.tmp";
const STATE: &str = "data/live08d-state.json";
const STATE_TMP: &str = "data/live08d-state.json.tmp";

#[derive(Clone, Serialize, Deserialize)]
struct CognitiveState {
    probabilities: [f64; 8],
    evidence: [f64; 8],
    observations: [u64; 8],
    leader: Option<String>,
    recognition: f64,
    stability: f64,
    confidence_internal: f64,
    cognitive_cycles: u64,
    leader_streak: u64,
    incoming: u64,
}

#[derive(Clone, Serialize, Deserialize)]
struct Freeze {
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
    hhm_historical_blocks: u64,
    hhm_historical_state_observations: u64,
    hhm_microcell_rows: u64,
    hhm_microcell_samples: u64,
    cognition: CognitiveState,
}

#[derive(Clone, Serialize, Deserialize)]
struct ChainOutcome {
    observed_at: u64,
    height: u64,
    hash: String,
    previousblockhash: String,
    version: i64,
    merkle_root: String,
    timestamp: u64,
    mediantime: u64,
    nonce: u64,
    bits: u64,
    difficulty: f64,
    tx_count: u64,
    size: u64,
    weight: u64,
}

#[derive(Deserialize)]
struct Block {
    id: String,
    height: u64,
    version: i64,
    timestamp: u64,
    tx_count: u64,
    size: u64,
    weight: u64,
    merkle_root: String,
    previousblockhash: Option<String>,
    mediantime: u64,
    nonce: u64,
    bits: u64,
    difficulty: f64,
}

#[derive(Serialize)]
struct TrainingExample {
    schema: String,
    before: Freeze,
    outcome: ChainOutcome,
    relation: String,
}

#[derive(Serialize)]
struct RuntimeState {
    version: String,
    timestamp_unix: u64,
    authorized: bool,
    difficulty: Option<f64>,
    incoming: u64,
    freezes: u64,
    outcomes: u64,
    duplicates: u64,
    pending: usize,
    chain_height: u64,
    chain_hash: String,
    observer_confidence_internal: f64,
    observer_stability: f64,
    observer_recognition: f64,
    observer_leader: Option<String>,
    auto_play: bool,
    verify: bool,
    submit: bool,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn atomic<T: Serialize>(tmp: &str, path: &str, value: &T) -> Result<(), Box<dyn Error>> {
    fs::write(tmp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn append_jsonl<T: Serialize>(path: &str, value: &T) -> Result<(), Box<dyn Error>> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut f, value)?;
    writeln!(f)?;
    f.flush()?;
    Ok(())
}

fn send(stream: &mut TcpStream, v: Value) -> Result<(), Box<dyn Error>> {
    let mut line = serde_json::to_string(&v)?;
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn spawn_reader(stream: TcpStream) -> mpsc::Receiver<Value> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut r = BufReader::new(stream);
        loop {
            let mut line = String::new();
            match r.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if let Ok(v) = serde_json::from_str::<Value>(line.trim()) {
                        if tx.send(v).is_err() {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });
    rx
}

fn tip(client: &Client) -> Result<(u64, String), Box<dyn Error>> {
    let h: u64 = client
        .get(format!("{ESPLORA}/blocks/tip/height"))
        .send()?
        .error_for_status()?
        .text()?
        .trim()
        .parse()?;
    let hash = client
        .get(format!("{ESPLORA}/block-height/{h}"))
        .send()?
        .error_for_status()?
        .text()?
        .trim()
        .to_string();
    Ok((h, hash))
}

fn get_block(client: &Client, hash: &str) -> Result<Block, Box<dyn Error>> {
    Ok(client
        .get(format!("{ESPLORA}/block/{hash}"))
        .send()?
        .error_for_status()?
        .json()?)
}

fn canonical_prevhash(s: &str) -> Result<String, Box<dyn Error>> {
    let b = hex::decode(s)?;
    if b.len() != 32 {
        return Err("prevhash must be 32 bytes".into());
    }
    let mut header_order = [0u8; 32];
    for w in 0..8 {
        let i = w * 4;
        header_order[i] = b[i + 3];
        header_order[i + 1] = b[i + 2];
        header_order[i + 2] = b[i + 1];
        header_order[i + 3] = b[i];
    }
    header_order.reverse();
    Ok(hex::encode(header_order))
}

fn fingerprint(params: &[Value]) -> Result<String, Box<dyn Error>> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(params)?)))
}

fn load_pending() -> Result<Vec<Freeze>, Box<dyn Error>> {
    if !Path::new(PENDING).exists() {
        return Ok(Vec::new());
    }
    Ok(serde_json::from_slice(&fs::read(PENDING)?)?)
}

fn save_pending(p: &[Freeze]) -> Result<(), Box<dyn Error>> {
    atomic(PENDING_TMP, PENDING, &p)
}

/* Pre-outcome only. These are associative context fingerprints, not SHA evidence. */
fn context_signals(params: &[Value]) -> [f64; 8] {
    let mut material = Vec::new();
    for &idx in &[1usize, 5, 6, 7] {
        if let Some(s) = params.get(idx).and_then(Value::as_str) {
            material.extend_from_slice(s.as_bytes());
        }
    }
    let branches = params
        .get(4)
        .and_then(Value::as_array)
        .map(|x| x.len())
        .unwrap_or(0);
    material.extend_from_slice(&(branches as u64).to_le_bytes());

    let d = Sha256::digest(&material);
    let mut out = [0.0f64; 8];
    for i in 0..8 {
        let a = u16::from_le_bytes([d[i * 2], d[i * 2 + 1]]) as f64 / 65535.0;
        out[i] = (a - 0.5) * 0.02;
    }
    out
}

fn snapshot(observer: &AssociativeObserver) -> CognitiveState {
    let mut probabilities = [0.0; 8];
    let mut evidence = [0.0; 8];
    let mut observations = [0u64; 8];
    for i in 0..8 {
        probabilities[i] = observer.states[i].probability;
        evidence[i] = observer.states[i].evidence;
        observations[i] = observer.states[i].observations;
    }
    CognitiveState {
        probabilities,
        evidence,
        observations,
        leader: observer.leader.clone(),
        recognition: observer.recognition_rate,
        stability: observer.stability,
        confidence_internal: observer.confidence,
        cognitive_cycles: observer.cognitive_cycles,
        leader_streak: observer.leader_streak,
        incoming: observer.incoming,
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

        // Frozen v0.20 schema:
        // 0=height ... 8=nonce, 9..=19=x1331_l1..x1331_l11, 20=verified
        for idx in 9..=19 {
            let state = r.get(idx).ok_or("missing x1331 historical state")?;
            hhm.observe_historical_state(state);
        }
    }

    let mut cells = csv::Reader::from_path(CELLS)?;
    for row in cells.records() {
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

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 LIVE-08D");
    println!("PROSPECTIVE COGNITIVE DATASET");
    println!("RULE: COGNITION IS FROZEN BEFORE NETWORK OUTCOME");
    println!("AUTO PLAY=OFF | VERIFY=OFF | SUBMIT=OFF\n");

    println!("Loading HHM...");
    let hhm = load_hhm()?;
    println!(
        "HHM READY | blocks={} states={} microcells={} SHA-memory={}",
        hhm.summary.historical_blocks,
        hhm.summary.historical_state_observations,
        hhm.summary.microcell_rows,
        hhm.summary.microcell_samples
    );

    let priors: [f64; 8] = std::array::from_fn(|i| hhm.states[i].historical_hits as f64);
    let mut observer = AssociativeObserver::new(priors);

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent("X1331-LIVE-08D")
        .build()?;
    let (mut chain_height, mut chain_hash) = tip(&client)?;

    let mut pending = load_pending()?;
    let mut seen: HashSet<String> = pending
        .iter()
        .map(|x| x.notify_fingerprint.clone())
        .collect();
    let mut incoming = pending.iter().map(|x| x.incoming).max().unwrap_or(0);
    let mut freezes = pending.iter().map(|x| x.freeze_id).max().unwrap_or(0);
    let mut outcomes = 0u64;
    let mut duplicates = 0u64;

    println!("CHAIN BASELINE | height={chain_height} hash={chain_hash}");
    println!("RESTORED 08D PENDING | {}", pending.len());

    loop {
        println!("Connecting to {POOL} ...");
        let mut stream = match TcpStream::connect(POOL) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("CONNECT ERROR: {e}");
                thread::sleep(Duration::from_secs(5));
                continue;
            }
        };
        let rx = spawn_reader(stream.try_clone()?);
        send(
            &mut stream,
            json!({"id":1,"method":"mining.subscribe","params":["X1331/LIVE-08D"]}),
        )?;

        let mut authorized = false;
        let mut difficulty = None;
        let mut last_poll = now();

        loop {
            match rx.recv_timeout(Duration::from_secs(1)) {
                Ok(msg) => {
                    let id = msg.get("id").and_then(Value::as_i64);
                    if id == Some(1) {
                        println!("SUBSCRIBE OK");
                        send(
                            &mut stream,
                            json!({"id":2,"method":"mining.authorize","params":[WORKER,PASSWORD]}),
                        )?;
                        continue;
                    }
                    if id == Some(2) {
                        authorized = msg.get("result").and_then(Value::as_bool).unwrap_or(false);
                        println!("AUTHORIZE: {}", if authorized { "OK" } else { "FAILED" });
                        if !authorized {
                            break;
                        }
                        continue;
                    }

                    match msg.get("method").and_then(Value::as_str) {
                        Some("mining.set_difficulty") => {
                            difficulty = msg
                                .get("params")
                                .and_then(Value::as_array)
                                .and_then(|p| p.first())
                                .and_then(Value::as_f64);
                            println!("DIFFICULTY: {}", difficulty.unwrap_or(0.0));
                        }
                        Some("mining.notify") if authorized => {
                            let p = msg
                                .get("params")
                                .and_then(Value::as_array)
                                .ok_or("notify params")?;
                            if p.len() < 9 {
                                return Err("short mining.notify".into());
                            }

                            let fp = fingerprint(p)?;
                            if seen.contains(&fp) {
                                duplicates += 1;
                                println!(
                                    "DUPLICATE | job={} ignored",
                                    p[0].as_str().unwrap_or("?")
                                );
                                continue;
                            }

                            let signals = context_signals(p);
                            observer.observe_signals(signals);
                            incoming += 1;
                            freezes += 1;

                            let prev_stratum = p[1].as_str().ok_or("prevhash")?.to_string();
                            let freeze = Freeze {
                                freeze_id: freezes,
                                frozen_at: now(),
                                incoming,
                                job_id: p[0].as_str().ok_or("job")?.to_string(),
                                pool_difficulty: difficulty.unwrap_or(0.0),
                                prevhash_canonical: canonical_prevhash(&prev_stratum)?,
                                prevhash_stratum: prev_stratum,
                                version: p[5].as_str().ok_or("version")?.to_string(),
                                nbits: p[6].as_str().ok_or("nbits")?.to_string(),
                                ntime: p[7].as_str().ok_or("ntime")?.to_string(),
                                notify_fingerprint: fp.clone(),
                                hhm_historical_blocks: hhm.summary.historical_blocks,
                                hhm_historical_state_observations: hhm
                                    .summary
                                    .historical_state_observations,
                                hhm_microcell_rows: hhm.summary.microcell_rows,
                                hhm_microcell_samples: hhm.summary.microcell_samples,
                                cognition: snapshot(&observer),
                            };

                            /* The complete BEFORE record is durable before any outcome lookup. */
                            append_jsonl(FREEZES, &freeze)?;
                            pending.push(freeze.clone());
                            seen.insert(fp);
                            save_pending(&pending)?;

                            println!(
                                "COGNITIVE FREEZE | id={} job={} parent={} leader={:?} conf={:.6} stab={:.6} recog={:.6} BEFORE-OUTCOME",
                                freeze.freeze_id,
                                freeze.job_id,
                                freeze.prevhash_canonical,
                                freeze.cognition.leader,
                                freeze.cognition.confidence_internal,
                                freeze.cognition.stability,
                                freeze.cognition.recognition
                            );
                        }
                        _ => {}
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }

            let t = now();
            if t.saturating_sub(last_poll) >= 10 {
                last_poll = t;
                match tip(&client) {
                    Ok((new_height, new_hash)) if new_hash != chain_hash => {
                        let b = get_block(&client, &new_hash)?;
                        let parent = b.previousblockhash.clone().unwrap_or_default();
                        let outcome = ChainOutcome {
                            observed_at: now(),
                            height: b.height,
                            hash: b.id.clone(),
                            previousblockhash: parent.clone(),
                            version: b.version,
                            merkle_root: b.merkle_root,
                            timestamp: b.timestamp,
                            mediantime: b.mediantime,
                            nonce: b.nonce,
                            bits: b.bits,
                            difficulty: b.difficulty,
                            tx_count: b.tx_count,
                            size: b.size,
                            weight: b.weight,
                        };

                        println!(
                            "CHAIN OUTCOME | height={} hash={} parent={}",
                            outcome.height, outcome.hash, outcome.previousblockhash
                        );

                        let mut keep = Vec::with_capacity(pending.len());
                        let mut closed = 0u64;
                        for freeze in pending.drain(..) {
                            if freeze.prevhash_canonical == parent {
                                let example = TrainingExample {
                                    schema: "x1331-live08d-v1".to_string(),
                                    before: freeze,
                                    outcome: outcome.clone(),
                                    relation: "DIRECT_SUCCESSOR".to_string(),
                                };
                                append_jsonl(DATASET, &example)?;
                                outcomes += 1;
                                closed += 1;
                            } else {
                                keep.push(freeze);
                            }
                        }
                        pending = keep;
                        save_pending(&pending)?;

                        println!(
                            "DATASET CLOSE | block={} closed={} remain_pending={} total_examples={}",
                            outcome.height,
                            closed,
                            pending.len(),
                            outcomes
                        );
                        chain_height = new_height;
                        chain_hash = new_hash;
                    }
                    Ok(_) => {}
                    Err(e) => eprintln!("CHAIN POLL ERROR: {e}"),
                }

                atomic(
                    STATE_TMP,
                    STATE,
                    &RuntimeState {
                        version: "LIVE-08D".to_string(),
                        timestamp_unix: t,
                        authorized,
                        difficulty,
                        incoming,
                        freezes,
                        outcomes,
                        duplicates,
                        pending: pending.len(),
                        chain_height,
                        chain_hash: chain_hash.clone(),
                        observer_confidence_internal: observer.confidence,
                        observer_stability: observer.stability,
                        observer_recognition: observer.recognition_rate,
                        observer_leader: observer.leader.clone(),
                        auto_play: false,
                        verify: false,
                        submit: false,
                    },
                )?;
            }
        }

        eprintln!("STRATUM SESSION ENDED | reconnect in 5 seconds");
        thread::sleep(Duration::from_secs(5));
    }
}
