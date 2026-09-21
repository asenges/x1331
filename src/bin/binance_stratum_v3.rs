use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    io::{BufRead, BufReader, Write},
    net::TcpStream,
    time::Duration,
};

const POOL: &str = "sha256.poolbinance.com:443";
const WORKER: &str = "Cl0udB4ck0ff1c3.x1331";
const PASSWORD: &str = "x";

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

fn stratum_prevhash_to_header(v: Vec<u8>) -> Result<[u8; 32], Box<dyn Error>> {
    if v.len() != 32 {
        return Err(format!("expected 32-byte prevhash, got {}", v.len()).into());
    }

    // Stratum V1 prevhash is encoded as eight 32-bit words whose
    // byte order is reversed relative to Bitcoin header serialization.
    // Reverse bytes INSIDE each u32 word. Do NOT reverse all 32 bytes.
    let mut out = [0u8; 32];

    for word in 0..8 {
        let i = word * 4;
        out[i]     = v[i + 3];
        out[i + 1] = v[i + 2];
        out[i + 2] = v[i + 1];
        out[i + 3] = v[i];
    }

    Ok(out)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 Binance Stratum Adapter v3 — HEADER VALIDATION");
    println!("======================================");
    println!("Pool   : {POOL}");
    println!("Worker : {WORKER}");
    println!("Submit : DISABLED");
    println!();

    let mut stream = TcpStream::connect(POOL)?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;

    let reader_stream = stream.try_clone()?;
    let mut reader = BufReader::new(reader_stream);

    send_json(
        &mut stream,
        json!({
            "id": 1,
            "method": "mining.subscribe",
            "params": ["X1331/0.31"]
        }),
    )?;

    let mut extranonce1 = String::new();
    let mut extranonce2_size: usize = 0;
    let mut authorized = false;
    let mut difficulty: Option<f64> = None;

    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;

        if n == 0 {
            return Err("pool closed connection".into());
        }

        let msg: Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Invalid JSON: {e}");
                continue;
            }
        };

        // Subscription response.
        if msg.get("id").and_then(Value::as_i64) == Some(1) {
            let result = msg
                .get("result")
                .and_then(Value::as_array)
                .ok_or("invalid mining.subscribe response")?;

            if result.len() < 3 {
                return Err("short mining.subscribe result".into());
            }

            extranonce1 = result[1]
                .as_str()
                .ok_or("missing extranonce1")?
                .to_string();

            extranonce2_size = result[2]
                .as_u64()
                .ok_or("missing extranonce2_size")?
                as usize;

            println!("SUBSCRIBE OK");
            println!("  extranonce1      : {extranonce1}");
            println!("  extranonce2_size : {extranonce2_size}");

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

        // Authorization response.
        if msg.get("id").and_then(Value::as_i64) == Some(2) {
            authorized = msg
                .get("result")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            println!("AUTHORIZE: {}", if authorized { "OK" } else { "FAILED" });

            if !authorized {
                return Err("Stratum authorization failed".into());
            }

            continue;
        }

        let method = msg.get("method").and_then(Value::as_str);

        if method == Some("mining.set_difficulty") {
            if let Some(params) = msg.get("params").and_then(Value::as_array) {
                if let Some(d) = params.first().and_then(Value::as_f64) {
                    difficulty = Some(d);
                    println!("DIFFICULTY: {d}");
                }
            }

            continue;
        }

        if method != Some("mining.notify") {
            continue;
        }

        if !authorized {
            println!("Job arrived before authorization; waiting...");
            continue;
        }

        if extranonce1.is_empty() || extranonce2_size == 0 {
            println!("Job arrived before subscription state was complete.");
            continue;
        }

        let p = msg
            .get("params")
            .and_then(Value::as_array)
            .ok_or("notify params missing")?;

        if p.len() < 9 {
            return Err(format!("unexpected mining.notify params: {}", p.len()).into());
        }

        let job_id = p[0].as_str().ok_or("job_id")?;
        let prevhash_hex = p[1].as_str().ok_or("prevhash")?;
        let coinb1 = p[2].as_str().ok_or("coinb1")?;
        let coinb2 = p[3].as_str().ok_or("coinb2")?;
        let branches = p[4].as_array().ok_or("merkle branches")?;
        let version_hex = p[5].as_str().ok_or("version")?;
        let nbits_hex = p[6].as_str().ok_or("nbits")?;
        let ntime_hex = p[7].as_str().ok_or("ntime")?;
        let clean_jobs = p[8].as_bool().unwrap_or(false);

        // Deterministic extranonce2 for DRY-RUN only.
        // We are NOT hashing candidate nonces or submitting this job.
        let extranonce2 = vec![0u8; extranonce2_size];

        let mut coinbase = Vec::new();
        coinbase.extend(hex::decode(coinb1)?);
        coinbase.extend(hex::decode(&extranonce1)?);
        coinbase.extend(&extranonce2);
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

        let version = hex::decode(version_hex)?;
        let prevhash = stratum_prevhash_to_header(hex::decode(prevhash_hex)?)?;
        let ntime = hex::decode(ntime_hex)?;
        let nbits = hex::decode(nbits_hex)?;

        if version.len() != 4 || ntime.len() != 4 || nbits.len() != 4 {
            return Err("version/ntime/nbits must each be 4 bytes".into());
        }

        /*
         * Bitcoin serialized header:
         *
         * version    4 bytes LE
         * prevhash  32 bytes internal byte order
         * merkle    32 bytes internal byte order
         * ntime      4 bytes LE
         * nbits      4 bytes LE
         * nonce      4 bytes LE
         *
         * Stratum transmits version/ntime/nbits as display-order hex,
         * so their four-byte fields are reversed for serialized header.
         */

        let mut header = [0u8; 80];

        let mut v = version;
        v.reverse();
        header[0..4].copy_from_slice(&v);

        header[4..36].copy_from_slice(&prevhash);

        // Stratum V1 merkle construction already produces the byte
        // sequence used in the mining header. Do NOT reverse it here.
        header[36..68].copy_from_slice(&merkle);

        let mut t = ntime;
        t.reverse();
        header[68..72].copy_from_slice(&t);

        let mut b = nbits;
        b.reverse();
        header[72..76].copy_from_slice(&b);

        // nonce = 0 solely to make an 80-byte template.
        header[76..80].copy_from_slice(&0u32.to_le_bytes());

        println!();
        println!("REAL STRATUM JOB PARSED");
        println!("=======================");
        println!("job_id       : {job_id}");
        println!("difficulty   : {:?}", difficulty);
        println!("clean_jobs   : {clean_jobs}");
        println!("version      : {version_hex}");
        println!("prevhash     : {prevhash_hex}");
        println!("ntime        : {ntime_hex}");
        println!("nbits        : {nbits_hex}");
        println!("branches     : {}", branches.len());
        println!("coinbase_len : {} bytes", coinbase.len());
        println!("extranonce1  : {extranonce1}");
        println!("extranonce2  : {}", hex::encode(&extranonce2));
        println!("merkle       : {}", hex::encode(merkle));
        println!("header80     : {}", hex::encode(header));
        println!();
        println!("DRY-RUN COMPLETE");
        println!("SHA candidate search : NOT EXECUTED");
        println!("mining.submit        : NOT EXECUTED");

        break;
    }

    Ok(())
}
