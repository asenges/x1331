use reqwest::blocking::Client;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use std::{
    error::Error,
    fs::{self, OpenOptions},
    path::Path,
    thread,
    time::{Duration, Instant},
};

const API: &str = "https://blockstream.info/api";

// Primera prueba real.
// 10,000 bloques terminando en 899,999.
const START_HEIGHT: u64 = 890_000;
const END_HEIGHT: u64 = 899_999;

const OUTPUT: &str = "data/bitcoin-blocks.csv";

// Pausa conservadora para API pública.
const REQUEST_DELAY_MS: u64 = 75;
const MAX_RETRIES: usize = 6;

#[derive(Debug, Deserialize)]
struct BlockInfo {
    id: String,
    height: u64,
    version: i64,
    timestamp: u64,
    bits: u64,
    nonce: u64,
    merkle_root: String,
    previousblockhash: Option<String>,
}

fn get_text(
    client: &Client,
    url: &str,
) -> Result<String, Box<dyn Error>> {
    for attempt in 1..=MAX_RETRIES {
        match client.get(url).send() {
            Ok(response) => {
                if response.status().is_success() {
                    let text =
                        response.text()?;

                    thread::sleep(
                        Duration::from_millis(
                            REQUEST_DELAY_MS,
                        ),
                    );

                    return Ok(
                        text.trim().to_string()
                    );
                }

                eprintln!(
                    "HTTP {} | attempt {}/{} | {}",
                    response.status(),
                    attempt,
                    MAX_RETRIES,
                    url
                );
            }

            Err(error) => {
                eprintln!(
                    "request error | attempt {}/{} | {}",
                    attempt,
                    MAX_RETRIES,
                    error
                );
            }
        }

        let backoff =
            500 * attempt as u64;

        thread::sleep(
            Duration::from_millis(
                backoff,
            ),
        );
    }

    Err(
        format!(
            "request failed after {} attempts: {}",
            MAX_RETRIES,
            url
        )
        .into(),
    )
}

fn get_json<T>(
    client: &Client,
    url: &str,
) -> Result<T, Box<dyn Error>>
where
    T: for<'de> Deserialize<'de>,
{
    let text =
        get_text(client, url)?;

    Ok(
        serde_json::from_str(
            &text
        )?
    )
}

fn sha256d(
    bytes: &[u8],
) -> [u8; 32] {
    let first =
        Sha256::digest(bytes);

    let second =
        Sha256::digest(first);

    let mut result =
        [0u8; 32];

    result.copy_from_slice(
        &second
    );

    result
}

fn bitcoin_display_hash(
    raw_hash: &[u8; 32],
) -> String {
    let mut reversed =
        *raw_hash;

    reversed.reverse();

    hex::encode(reversed)
}

fn verify_header(
    header_hex: &str,
    expected_block_hash: &str,
) -> Result<bool, Box<dyn Error>> {
    let bytes =
        hex::decode(header_hex)?;

    if bytes.len() != 80 {
        return Err(
            format!(
                "invalid Bitcoin header length: {} bytes",
                bytes.len()
            )
            .into(),
        );
    }

    let raw =
        sha256d(&bytes);

    let calculated =
        bitcoin_display_hash(
            &raw
        );

    Ok(
        calculated
            .eq_ignore_ascii_case(
                expected_block_hash
            )
    )
}

fn nonce_from_header(
    header_hex: &str,
) -> Result<u32, Box<dyn Error>> {
    let bytes =
        hex::decode(header_hex)?;

    if bytes.len() != 80 {
        return Err(
            "header must be 80 bytes"
                .into(),
        );
    }

    // Bitcoin header:
    // bytes 76..79 = nonce,
    // serialized little-endian.
    let nonce =
        u32::from_le_bytes([
            bytes[76],
            bytes[77],
            bytes[78],
            bytes[79],
        ]);

    Ok(nonce)
}

fn x1331_path(
    nonce: u32,
) -> Vec<u8> {
    let mut path =
        Vec::with_capacity(11);

    // 32-bit nonce.
    //
    // First ten levels consume
    // 30 bits as groups of three.
    //
    // Final two bits are represented
    // as a partial terminal state.
    for level in 0..10 {
        let shift =
            32 - ((level + 1) * 3);

        let state =
            ((nonce >> shift) & 0b111)
                as u8;

        path.push(state);
    }

    // Remaining lowest 2 bits.
    path.push(
        (nonce & 0b11) as u8
    );

    path
}

fn existing_last_height()
    -> Result<Option<u64>, Box<dyn Error>>
{
    if !Path::new(OUTPUT).exists() {
        return Ok(None);
    }

    let mut reader =
        csv::Reader::from_path(
            OUTPUT
        )?;

    let mut last =
        None;

    for record in reader.records() {
        let record =
            record?;

        if let Some(value) =
            record.get(0)
        {
            if let Ok(height) =
                value.parse::<u64>()
            {
                last =
                    Some(height);
            }
        }
    }

    Ok(last)
}

fn main()
    -> Result<(), Box<dyn Error>>
{
    let started =
        Instant::now();

    fs::create_dir_all(
        "data"
    )?;

    println!(
        "X1331 Runtime v0.15"
    );

    println!(
        "==================="
    );

    println!(
        "Historical Bitcoin Dataset Builder"
    );

    println!();

    println!(
        "Source       : Blockstream Esplora"
    );

    println!(
        "Start height : {}",
        START_HEIGHT
    );

    println!(
        "End height   : {}",
        END_HEIGHT
    );

    println!(
        "Requested    : {} blocks",
        END_HEIGHT
            - START_HEIGHT
            + 1
    );

    println!(
        "Output       : {}",
        OUTPUT
    );

    println!();

    let client =
        Client::builder()
            .user_agent(
                "X1331-research/0.15"
            )
            .timeout(
                Duration::from_secs(20)
            )
            .build()?;

    let last_existing =
        existing_last_height()?;

    let actual_start =
        match last_existing {
            Some(height)
                if height >= START_HEIGHT
                    && height < END_HEIGHT =>
            {
                println!(
                    "Resume detected at block {}",
                    height
                );

                height + 1
            }

            Some(height)
                if height >= END_HEIGHT =>
            {
                println!(
                    "Dataset already reaches {}.",
                    height
                );

                return Ok(());
            }

            _ =>
                START_HEIGHT,
        };

    let file_exists =
        Path::new(OUTPUT).exists();

    let file =
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(OUTPUT)?;

    let mut writer =
        csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(file);

    if !file_exists {
        let mut header =
            vec![
                "height".to_string(),
                "block_hash".to_string(),
                "header_hex".to_string(),
                "version".to_string(),
                "previous_block_hash".to_string(),
                "merkle_root".to_string(),
                "timestamp".to_string(),
                "bits".to_string(),
                "nonce".to_string(),
            ];

        for level in 1..=11 {
            header.push(
                format!(
                    "x1331_l{}",
                    level
                )
            );
        }

        header.push(
            "verified".to_string()
        );

        writer.write_record(
            &header
        )?;

        writer.flush()?;
    }

    let mut downloaded =
        0u64;

    let mut verified =
        0u64;

    let mut failed =
        0u64;

    for height in
        actual_start..=END_HEIGHT
    {
        let block_result:
            Result<(), Box<dyn Error>> =
            (|| {
                //
                // 1. Height -> block hash
                //
                let hash_url =
                    format!(
                        "{}/block-height/{}",
                        API,
                        height
                    );

                let block_hash =
                    get_text(
                        &client,
                        &hash_url,
                    )?;

                //
                // 2. Block metadata
                //
                let info_url =
                    format!(
                        "{}/block/{}",
                        API,
                        block_hash
                    );

                let info:
                    BlockInfo =
                    get_json(
                        &client,
                        &info_url,
                    )?;

                //
                // 3. Exact 80-byte header
                //
                let header_url =
                    format!(
                        "{}/block/{}/header",
                        API,
                        block_hash
                    );

                let header_hex =
                    get_text(
                        &client,
                        &header_url,
                    )?;

                //
                // 4. Cryptographic verification
                //
                let valid =
                    verify_header(
                        &header_hex,
                        &block_hash,
                    )?;

                //
                // 5. Cross-check nonce.
                //
                let header_nonce =
                    nonce_from_header(
                        &header_hex
                    )?;

                if header_nonce as u64
                    != info.nonce
                {
                    return Err(
                        format!(
                            "nonce mismatch: API={} header={}",
                            info.nonce,
                            header_nonce
                        )
                        .into(),
                    );
                }

                if info.id
                    != block_hash
                {
                    return Err(
                        "block id mismatch"
                            .into(),
                    );
                }

                if info.height
                    != height
                {
                    return Err(
                        "height mismatch"
                            .into(),
                    );
                }

                let path =
                    x1331_path(
                        header_nonce
                    );

                let mut row =
                    vec![
                        info.height
                            .to_string(),

                        block_hash.clone(),

                        header_hex,

                        info.version
                            .to_string(),

                        info.previousblockhash
                            .unwrap_or_default(),

                        info.merkle_root,

                        info.timestamp
                            .to_string(),

                        info.bits
                            .to_string(),

                        info.nonce
                            .to_string(),
                    ];

                for state in path {
                    row.push(
                        format!(
                            "{:03b}",
                            state
                        )
                    );
                }

                row.push(
                    valid.to_string()
                );

                writer.write_record(
                    &row
                )?;

                writer.flush()?;

                downloaded += 1;

                if valid {
                    verified += 1;
                }

                Ok(())
            })();

        if let Err(error) =
            block_result
        {
            failed += 1;

            eprintln!(
                "FAILED height {}: {}",
                height,
                error
            );
        }

        let processed =
            downloaded + failed;

        if processed % 25 == 0 {
            let elapsed =
                started
                    .elapsed()
                    .as_secs_f64();

            let rate =
                if elapsed > 0.0 {
                    processed as f64
                        / elapsed
                } else {
                    0.0
                };

            println!(
                "height {} | processed {} | verified {} | failed {} | {:.2} blocks/s",
                height,
                processed,
                verified,
                failed,
                rate
            );
        }
    }

    let elapsed =
        started.elapsed();

    println!();

    println!(
        "DATASET RESULT"
    );

    println!(
        "=============="
    );

    println!(
        "Downloaded : {}",
        downloaded
    );

    println!(
        "Verified   : {}",
        verified
    );

    println!(
        "Failed     : {}",
        failed
    );

    println!(
        "Verification rate : {:.4}%",
        if downloaded > 0 {
            verified as f64
                / downloaded as f64
                * 100.0
        } else {
            0.0
        }
    );

    println!(
        "Elapsed     : {:.3} s",
        elapsed.as_secs_f64()
    );

    println!(
        "Dataset     : {}",
        OUTPUT
    );

    Ok(())
}
