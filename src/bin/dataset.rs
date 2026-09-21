use reqwest::blocking::{Client, Response};
use reqwest::StatusCode;
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

const START_HEIGHT: u64 = 890_000;
const END_HEIGHT: u64 = 899_999;

const OUTPUT: &str = "data/bitcoin-blocks.csv";

// ~1 request/sec. Three requests/block.
const REQUEST_DELAY_MS: u64 = 1_000;

// We do NOT skip blocks.
const MAX_RETRIES: usize = 20;

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

fn wait_after_success() {
    thread::sleep(Duration::from_millis(REQUEST_DELAY_MS));
}

fn retry_after_seconds(response: &Response) -> Option<u64> {
    response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
}

fn get_text(
    client: &Client,
    url: &str,
) -> Result<String, Box<dyn Error>> {
    for attempt in 1..=MAX_RETRIES {
        match client.get(url).send() {
            Ok(response) => {
                let status = response.status();

                if status.is_success() {
                    let text = response.text()?;
                    wait_after_success();
                    return Ok(text.trim().to_string());
                }

                if status == StatusCode::TOO_MANY_REQUESTS {
                    let server_wait = retry_after_seconds(&response);

                    let exponential =
                        5u64.saturating_mul(
                            2u64.pow(
                                ((attempt - 1).min(4)) as u32
                            )
                        );

                    let wait_seconds =
                        server_wait
                            .unwrap_or(exponential)
                            .max(5);

                    eprintln!(
                        "HTTP 429 | attempt {}/{} | waiting {}s | {}",
                        attempt,
                        MAX_RETRIES,
                        wait_seconds,
                        url
                    );

                    thread::sleep(
                        Duration::from_secs(wait_seconds)
                    );

                    continue;
                }

                eprintln!(
                    "HTTP {} | attempt {}/{} | {}",
                    status,
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

        let wait_seconds =
            (attempt as u64 * 2).min(30);

        eprintln!(
            "retrying in {}s...",
            wait_seconds
        );

        thread::sleep(
            Duration::from_secs(wait_seconds)
        );
    }

    Err(
        format!(
            "request failed after {} attempts: {}",
            MAX_RETRIES,
            url
        )
        .into()
    )
}

fn get_json<T>(
    client: &Client,
    url: &str,
) -> Result<T, Box<dyn Error>>
where
    T: for<'de> Deserialize<'de>,
{
    let text = get_text(client, url)?;
    Ok(serde_json::from_str(&text)?)
}

fn sha256d(bytes: &[u8]) -> [u8; 32] {
    let first = Sha256::digest(bytes);
    let second = Sha256::digest(first);

    let mut result = [0u8; 32];
    result.copy_from_slice(&second);
    result
}

fn bitcoin_display_hash(raw_hash: &[u8; 32]) -> String {
    let mut reversed = *raw_hash;
    reversed.reverse();
    hex::encode(reversed)
}

fn verify_header(
    header_hex: &str,
    expected_block_hash: &str,
) -> Result<bool, Box<dyn Error>> {
    let bytes = hex::decode(header_hex)?;

    if bytes.len() != 80 {
        return Err(
            format!(
                "invalid Bitcoin header length: {} bytes",
                bytes.len()
            )
            .into()
        );
    }

    let raw = sha256d(&bytes);
    let calculated = bitcoin_display_hash(&raw);

    Ok(
        calculated.eq_ignore_ascii_case(
            expected_block_hash
        )
    )
}

fn nonce_from_header(
    header_hex: &str,
) -> Result<u32, Box<dyn Error>> {
    let bytes = hex::decode(header_hex)?;

    if bytes.len() != 80 {
        return Err("header must be 80 bytes".into());
    }

    Ok(
        u32::from_le_bytes([
            bytes[76],
            bytes[77],
            bytes[78],
            bytes[79],
        ])
    )
}

fn x1331_path(nonce: u32) -> Vec<u8> {
    let mut path = Vec::with_capacity(11);

    for level in 0..10 {
        let shift = 32 - ((level + 1) * 3);
        path.push(
            ((nonce >> shift) & 0b111) as u8
        );
    }

    // Remaining two bits.
    path.push((nonce & 0b11) as u8);

    path
}

fn existing_last_height()
    -> Result<Option<u64>, Box<dyn Error>>
{
    if !Path::new(OUTPUT).exists() {
        return Ok(None);
    }

    let mut reader = csv::Reader::from_path(OUTPUT)?;
    let mut last = None;

    for record in reader.records() {
        let record = record?;

        if let Some(value) = record.get(0) {
            if let Ok(height) = value.parse::<u64>() {
                last = Some(height);
            }
        }
    }

    Ok(last)
}

fn main() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();

    fs::create_dir_all("data")?;

    println!("X1331 Runtime v0.15.1");
    println!("=====================");
    println!("Historical Bitcoin Dataset Builder");
    println!("Rate-limit-safe / resumable / no gaps");
    println!();

    println!("Source       : Blockstream Esplora");
    println!("Start height : {}", START_HEIGHT);
    println!("End height   : {}", END_HEIGHT);
    println!(
        "Requested    : {} blocks",
        END_HEIGHT - START_HEIGHT + 1
    );
    println!("Output       : {}", OUTPUT);
    println!("Request delay: {} ms", REQUEST_DELAY_MS);
    println!();

    let client =
        Client::builder()
            .user_agent("X1331-research/0.15.1")
            .timeout(Duration::from_secs(30))
            .build()?;

    let last_existing = existing_last_height()?;

    let actual_start =
        match last_existing {
            Some(height)
                if height >= START_HEIGHT
                    && height < END_HEIGHT =>
            {
                println!(
                    "Resume detected."
                );
                println!(
                    "Last stored block : {}",
                    height
                );
                println!(
                    "Resuming at       : {}",
                    height + 1
                );
                println!();

                height + 1
            }

            Some(height)
                if height >= END_HEIGHT =>
            {
                println!(
                    "Dataset already reaches block {}.",
                    height
                );
                return Ok(());
            }

            _ => START_HEIGHT,
        };

    let file_exists = Path::new(OUTPUT).exists();

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
        let mut header = vec![
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
            header.push(format!("x1331_l{}", level));
        }

        header.push("verified".to_string());

        writer.write_record(&header)?;
        writer.flush()?;
    }

    let mut downloaded = 0u64;
    let mut verified = 0u64;

    for height in actual_start..=END_HEIGHT {
        //
        // IMPORTANT:
        // A block failure terminates the process.
        // We never silently skip a height.
        //

        let hash_url =
            format!(
                "{}/block-height/{}",
                API,
                height
            );

        let block_hash =
            match get_text(&client, &hash_url) {
                Ok(value) => value,

                Err(error) => {
                    eprintln!();
                    eprintln!(
                        "STOPPED at height {}",
                        height
                    );
                    eprintln!("{}", error);
                    eprintln!(
                        "No block was skipped."
                    );
                    eprintln!(
                        "Run the program again to resume."
                    );

                    return Err(error);
                }
            };

        let info_url =
            format!(
                "{}/block/{}",
                API,
                block_hash
            );

        let info: BlockInfo =
            get_json(
                &client,
                &info_url
            )?;

        let header_url =
            format!(
                "{}/block/{}/header",
                API,
                block_hash
            );

        let header_hex =
            get_text(
                &client,
                &header_url
            )?;

        let valid =
            verify_header(
                &header_hex,
                &block_hash
            )?;

        if !valid {
            return Err(
                format!(
                    "SHA256d verification FAILED at height {}",
                    height
                )
                .into()
            );
        }

        let header_nonce =
            nonce_from_header(
                &header_hex
            )?;

        if header_nonce as u64 != info.nonce {
            return Err(
                format!(
                    "nonce mismatch at {}: API={} header={}",
                    height,
                    info.nonce,
                    header_nonce
                )
                .into()
            );
        }

        if info.id != block_hash {
            return Err(
                format!(
                    "block id mismatch at {}",
                    height
                )
                .into()
            );
        }

        if info.height != height {
            return Err(
                format!(
                    "height mismatch: requested {} received {}",
                    height,
                    info.height
                )
                .into()
            );
        }

        let path =
            x1331_path(
                header_nonce
            );

        let mut row = vec![
            info.height.to_string(),
            block_hash,
            header_hex,
            info.version.to_string(),
            info.previousblockhash
                .unwrap_or_default(),
            info.merkle_root,
            info.timestamp.to_string(),
            info.bits.to_string(),
            info.nonce.to_string(),
        ];

        for state in path {
            row.push(
                format!("{:03b}", state)
            );
        }

        row.push("true".to_string());

        writer.write_record(&row)?;
        writer.flush()?;

        downloaded += 1;
        verified += 1;

        if downloaded % 25 == 0 {
            let elapsed =
                started.elapsed().as_secs_f64();

            println!(
                "height {} | this run {} | total {} | verified {} | {:.3} blocks/s",
                height,
                downloaded,
                height - START_HEIGHT + 1,
                verified,
                if elapsed > 0.0 {
                    downloaded as f64 / elapsed
                } else {
                    0.0
                }
            );
        }
    }

    let elapsed = started.elapsed();

    println!();
    println!("DATASET RESULT");
    println!("==============");
    println!(
        "Downloaded this run : {}",
        downloaded
    );
    println!(
        "Verified this run   : {}",
        verified
    );
    println!(
        "Final height        : {}",
        END_HEIGHT
    );
    println!(
        "Expected total rows : {}",
        END_HEIGHT - START_HEIGHT + 1
    );
    println!(
        "Elapsed             : {:.3} s",
        elapsed.as_secs_f64()
    );
    println!(
        "Dataset             : {}",
        OUTPUT
    );

    Ok(())
}
