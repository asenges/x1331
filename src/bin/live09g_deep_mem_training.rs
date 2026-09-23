use csv::ReaderBuilder;
use sha2::{Digest, Sha256};

use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

const DATASET: &str = "data/bitcoin-history-v020-frozen.csv";
const MODEL_OUT: &str = "data/live09/live09g-deep-mem.bin";

const EXPECTED_ROWS: usize = 200_000;

const TRAIN_END: usize = 140_000;
const VALID_END: usize = 170_000;

const INPUT_BITS: usize = 640;
const OUTPUT_BITS: usize = 256;

const INPUT_CELLS: usize = INPUT_BITS / 3; // 213 + 1 residual bit
const OUTPUT_CELLS: usize = OUTPUT_BITS / 3; // 85 + 1 residual bit

const CONTEXTS: usize = 512; // 8^3
const STATES: usize = 8;

// 85 output positions × 512 contexts × 8 possible output figures.
const COUNT_LEN: usize = OUTPUT_CELLS * CONTEXTS * STATES;

#[derive(Default, Clone)]
struct Metrics {
    n: u64,
    top1: u64,
    recall2: u64,
    recall4: u64,
    logloss_sum: f64,
    brier_sum: f64,
}

impl Metrics {
    fn observe(&mut self, probs: &[f64; 8], target: usize) {
        let ranked = ranked_states(probs);

        self.n += 1;

        if ranked[0] == target {
            self.top1 += 1;
        }

        if ranked[..2].contains(&target) {
            self.recall2 += 1;
        }

        if ranked[..4].contains(&target) {
            self.recall4 += 1;
        }

        self.logloss_sum += -probs[target].max(1e-15).ln();

        let mut brier = 0.0;
        for s in 0..8 {
            let y = if s == target { 1.0 } else { 0.0 };
            let d = probs[s] - y;
            brier += d * d;
        }
        self.brier_sum += brier;
    }

    fn print(&self, name: &str) {
        let n = self.n.max(1) as f64;

        println!(
            "{:<10} figures={} top1={:.9} recall2={:.9} recall4={:.9} logloss={:.9} brier={:.9}",
            name,
            self.n,
            self.top1 as f64 / n,
            self.recall2 as f64 / n,
            self.recall4 as f64 / n,
            self.logloss_sum / n,
            self.brier_sum / n
        );
    }
}

struct DeepMem {
    counts: Vec<u32>,
    totals: Vec<u32>,
}

impl DeepMem {
    fn new() -> Self {
        Self {
            counts: vec![0u32; COUNT_LEN],
            totals: vec![0u32; OUTPUT_CELLS * CONTEXTS],
        }
    }

    #[inline(always)]
    fn count_index(output_cell: usize, context: usize, state: usize) -> usize {
        ((output_cell * CONTEXTS + context) * STATES) + state
    }

    #[inline(always)]
    fn total_index(output_cell: usize, context: usize) -> usize {
        output_cell * CONTEXTS + context
    }

    #[inline(always)]
    fn observe(&mut self, output_cell: usize, context: usize, state: usize) {
        let ci = Self::count_index(output_cell, context, state);
        let ti = Self::total_index(output_cell, context);

        self.counts[ci] = self.counts[ci].saturating_add(1);
        self.totals[ti] = self.totals[ti].saturating_add(1);
    }

    #[inline(always)]
    fn probabilities(&self, output_cell: usize, context: usize) -> [f64; 8] {
        let ti = Self::total_index(output_cell, context);
        let total = self.totals[ti] as f64;

        // Laplace smoothing.
        let denom = total + 8.0;

        let mut p = [0.0f64; 8];

        for s in 0..8 {
            let ci = Self::count_index(output_cell, context, s);
            p[s] = (self.counts[ci] as f64 + 1.0) / denom;
        }

        p
    }

    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }

        let tmp = format!("{path}.tmp");
        let file = File::create(&tmp)?;
        let mut w = BufWriter::new(file);

        w.write_all(b"X1331DGM")?;
        w.write_all(&1u32.to_le_bytes())?;

        w.write_all(&(INPUT_BITS as u32).to_le_bytes())?;
        w.write_all(&(OUTPUT_BITS as u32).to_le_bytes())?;
        w.write_all(&(INPUT_CELLS as u32).to_le_bytes())?;
        w.write_all(&(OUTPUT_CELLS as u32).to_le_bytes())?;
        w.write_all(&(CONTEXTS as u32).to_le_bytes())?;
        w.write_all(&(STATES as u32).to_le_bytes())?;

        for v in &self.counts {
            w.write_all(&v.to_le_bytes())?;
        }

        for v in &self.totals {
            w.write_all(&v.to_le_bytes())?;
        }

        w.flush()?;
        drop(w);

        fs::rename(tmp, path)?;

        Ok(())
    }
}

fn decode_hex_80(s: &str) -> Result<[u8; 80], Box<dyn Error>> {
    if s.len() != 160 {
        return Err(format!("header_hex length {} != 160", s.len()).into());
    }

    let mut out = [0u8; 80];

    for i in 0..80 {
        out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)?;
    }

    Ok(out)
}

#[inline(always)]
fn sha256d(input: &[u8; 80]) -> [u8; 32] {
    let first = Sha256::digest(input);
    let second = Sha256::digest(first);

    let mut out = [0u8; 32];
    out.copy_from_slice(&second);
    out
}

#[inline(always)]
fn bit_msb(bytes: &[u8], bit: usize) -> u8 {
    let byte = bytes[bit >> 3];
    let shift = 7 - (bit & 7);
    (byte >> shift) & 1
}

#[inline(always)]
fn cell3(bytes: &[u8], start_bit: usize) -> usize {
    ((bit_msb(bytes, start_bit) as usize) << 2)
        | ((bit_msb(bytes, start_bit + 1) as usize) << 1)
        | bit_msb(bytes, start_bit + 2) as usize
}

// Each output figure gets a deterministic three-figure context distributed
// across the full 640-bit BEFORE header.
//
// This is intentionally frozen geometry, not tuned from outcomes.
#[inline(always)]
fn input_context(header: &[u8; 80], output_cell: usize) -> usize {
    let a = (output_cell * 7) % INPUT_CELLS;
    let b = (output_cell * 73 + 71) % INPUT_CELLS;
    let c = (output_cell * 149 + 137) % INPUT_CELLS;

    let sa = cell3(header, a * 3);
    let sb = cell3(header, b * 3);
    let sc = cell3(header, c * 3);

    (sa << 6) | (sb << 3) | sc
}

#[inline(always)]
fn output_state(digest: &[u8; 32], output_cell: usize) -> usize {
    cell3(digest, output_cell * 3)
}

fn ranked_states(probs: &[f64; 8]) -> [usize; 8] {
    let mut idx = [0usize, 1, 2, 3, 4, 5, 6, 7];

    idx.sort_by(|a, b| {
        probs[*b]
            .partial_cmp(&probs[*a])
            .unwrap()
            .then_with(|| a.cmp(b))
    });

    idx
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("============================================================");
    println!(" X1331 LIVE-09G — DEEP MEM TRAINING");
    println!(" REAL BITCOIN HISTORY");
    println!(" 640 BEFORE bits -> 256 historical SHA256d bits");
    println!(" Full scale: 896 bits = 298 complete 3-bit figures + 2 bits");
    println!("============================================================");
    println!("dataset={DATASET}");
    println!("train_rows=0..{}", TRAIN_END);
    println!("validation_rows={}..{}", TRAIN_END, VALID_END);
    println!("blind_rows={}..{}", VALID_END, EXPECTED_ROWS);
    println!("input_cells={INPUT_CELLS}");
    println!("output_cells={OUTPUT_CELLS}");
    println!("contexts={CONTEXTS}");
    println!("count_slots={COUNT_LEN}");
    println!();

    let total_start = Instant::now();

    let mut reader = ReaderBuilder::new().has_headers(true).from_path(DATASET)?;

    let headers = reader.headers()?.clone();

    let height_idx = headers
        .iter()
        .position(|x| x == "height")
        .ok_or("missing height")?;

    let header_hex_idx = headers
        .iter()
        .position(|x| x == "header_hex")
        .ok_or("missing header_hex")?;

    let verified_idx = headers
        .iter()
        .position(|x| x == "verified")
        .ok_or("missing verified")?;

    let mut mem = DeepMem::new();

    let mut train_prequential = Metrics::default();
    let mut validation = Metrics::default();
    let mut blind = Metrics::default();

    let mut rows = 0usize;
    let mut first_height = 0u64;
    let mut last_height = 0u64;

    let train_start = Instant::now();

    for result in reader.records() {
        let record = result?;

        let height: u64 = record
            .get(height_idx)
            .ok_or("missing height value")?
            .parse()?;

        let verified = record.get(verified_idx).ok_or("missing verified value")?;

        if verified != "true" {
            return Err(format!("height {height}: verified != true").into());
        }

        if rows == 0 {
            first_height = height;
        }

        last_height = height;

        let header_hex = record
            .get(header_hex_idx)
            .ok_or("missing header_hex value")?;

        // BEFORE reality: only the 80-byte historical header is decoded.
        let header = decode_hex_80(header_hex)?;

        // Freeze all BEFORE contexts before SHA result is used.
        let mut contexts = [0usize; OUTPUT_CELLS];

        for output_cell in 0..OUTPUT_CELLS {
            contexts[output_cell] = input_context(&header, output_cell);
        }

        // Historical reality reveal.
        let digest = sha256d(&header);

        if rows < TRAIN_END {
            // Strict prequential training:
            // predict from MEM_{t-1}, score, then learn experience t.
            for output_cell in 0..OUTPUT_CELLS {
                let context = contexts[output_cell];
                let target = output_state(&digest, output_cell);

                let probs = mem.probabilities(output_cell, context);
                train_prequential.observe(&probs, target);

                mem.observe(output_cell, context, target);
            }
        } else if rows < VALID_END {
            // Frozen MEM. No updates.
            for output_cell in 0..OUTPUT_CELLS {
                let context = contexts[output_cell];
                let target = output_state(&digest, output_cell);

                let probs = mem.probabilities(output_cell, context);
                validation.observe(&probs, target);
            }
        } else {
            // Blind chronological tail. MEM remains frozen.
            for output_cell in 0..OUTPUT_CELLS {
                let context = contexts[output_cell];
                let target = output_state(&digest, output_cell);

                let probs = mem.probabilities(output_cell, context);
                blind.observe(&probs, target);
            }
        }

        rows += 1;

        if rows % 10_000 == 0 {
            let elapsed = train_start.elapsed().as_secs_f64();

            println!(
                "rows={} height={} rows_per_s={:.1}",
                rows,
                height,
                rows as f64 / elapsed
            );
        }
    }

    if rows != EXPECTED_ROWS {
        return Err(format!("expected {EXPECTED_ROWS} rows, got {rows}").into());
    }

    if first_height != 700_000 || last_height != 899_999 {
        return Err(format!("unexpected height range {}..{}", first_height, last_height).into());
    }

    mem.save(MODEL_OUT)?;

    let model_bytes = fs::metadata(MODEL_OUT)?.len();

    println!();
    println!("============================================================");
    println!(" LIVE-09G FINAL");
    println!("============================================================");
    println!("rows={rows}");
    println!("height_range={}..{}", first_height, last_height);
    println!("train_rows={TRAIN_END}");
    println!("validation_rows={}", VALID_END - TRAIN_END);
    println!("blind_rows={}", EXPECTED_ROWS - VALID_END);
    println!("model={MODEL_OUT}");
    println!("model_bytes={model_bytes}");
    println!("elapsed_s={:.3}", total_start.elapsed().as_secs_f64());
    println!();

    train_prequential.print("TRAIN-PREQ");
    validation.print("VALIDATION");
    blind.print("BLIND");

    println!();
    println!("NULL REFERENCES");
    println!("top1=0.125000000");
    println!("recall2=0.250000000");
    println!("recall4=0.500000000");
    println!("uniform_logloss={:.9}", (8.0f64).ln());
    println!("uniform_brier=0.875000000");

    println!();
    println!("CAUSAL RULE:");
    println!("  header/context exists BEFORE target use.");
    println!("  TRAIN predicts before each experience is learned.");
    println!("  VALIDATION and BLIND use frozen TRAIN memory.");
    println!("  No validation/blind updates enter MEM.");

    println!();
    println!("SCIENTIFIC RULE:");
    println!("  Historical Bitcoin winners contain mining-process structure.");
    println!("  Any advantage here is NOT automatically SHA predictability.");
    println!("  Generalization must be judged on chronological frozen tails.");

    println!();
    println!("LIVE-09G: COMPLETE");

    Ok(())
}
