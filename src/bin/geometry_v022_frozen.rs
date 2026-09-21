use csv::{Reader, Writer};
use std::error::Error;

const INPUT: &str = "data/bitcoin-history-v020-frozen.csv";
const OUTPUT: &str = "data/v022-nonce-geometry-1k.csv";

const START: u64 = 700_000;
const END: u64 = 899_999;
const EXPECTED_ROWS: usize = 200_000;
const WINDOW: u64 = 1_000;

#[derive(Debug)]
struct Row {
    height: u64,
    nonce: u32,
    timestamp: u64,
    verified: bool,
}

fn bit(nonce: u32, b: usize) -> usize {
    ((nonce >> b) & 1) as usize
}

fn l9(nonce: u32) -> usize {
    ((nonce >> 5) & 0b111) as usize
}

fn pct(n: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        n as f64 / total as f64 * 100.0
    }
}

fn entropy8(counts: &[usize; 8]) -> f64 {
    let total: usize = counts.iter().sum();

    if total == 0 {
        return 0.0;
    }

    let mut h = 0.0;

    for &count in counts {
        if count == 0 {
            continue;
        }

        let p = count as f64 / total as f64;
        h -= p * p.log2();
    }

    h
}

fn l9_uniform_chi2(counts: &[usize; 8]) -> f64 {
    let total: usize = counts.iter().sum();

    if total == 0 {
        return 0.0;
    }

    let expected = total as f64 / 8.0;

    counts
        .iter()
        .map(|&observed| {
            let d = observed as f64 - expected;
            d * d / expected
        })
        .sum()
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 v0.22 Nonce Geometry / Regime Attribution");
    println!("================================================");
    println!("Input       : {}", INPUT);
    println!("Output      : {}", OUTPUT);
    println!("Range       : {} .. {}", START, END);
    println!("Window      : {} blocks", WINDOW);
    println!();

    // ========================================================
    // LOAD DATASET
    // ========================================================

    let mut reader = Reader::from_path(INPUT)?;
    let headers = reader.headers()?.clone();

    let height_idx = headers
        .iter()
        .position(|h| h == "height")
        .ok_or("height column missing")?;

    let nonce_idx = headers
        .iter()
        .position(|h| h == "nonce")
        .ok_or("nonce column missing")?;

    let timestamp_idx = headers
        .iter()
        .position(|h| h == "timestamp")
        .ok_or("timestamp column missing")?;

    let verified_idx = headers
        .iter()
        .position(|h| h == "verified")
        .ok_or("verified column missing")?;

    let mut rows =
        Vec::<Row>::with_capacity(EXPECTED_ROWS);

    for record in reader.records() {
        let record = record?;

        let height: u64 =
            record[height_idx].parse()?;

        let nonce: u32 =
            record[nonce_idx].parse()?;

        let timestamp: u64 =
            record[timestamp_idx].parse()?;

        let verified = matches!(
            record[verified_idx]
                .to_ascii_lowercase()
                .as_str(),
            "true" | "1" | "yes"
        );

        rows.push(Row {
            height,
            nonce,
            timestamp,
            verified,
        });
    }

    // ========================================================
    // VALIDATE
    // ========================================================

    if rows.len() != EXPECTED_ROWS {
        return Err(
            format!(
                "expected {} rows, got {}",
                EXPECTED_ROWS,
                rows.len()
            )
            .into(),
        );
    }

    for (i, row) in rows.iter().enumerate() {
        let expected_height = START + i as u64;

        if row.height != expected_height {
            return Err(
                format!(
                    "height failure: expected {}, got {}",
                    expected_height,
                    row.height
                )
                .into(),
            );
        }

        if !row.verified {
            return Err(
                format!(
                    "unverified row at {}",
                    row.height
                )
                .into(),
            );
        }
    }

    if rows[0].height != START
        || rows[rows.len() - 1].height != END
    {
        return Err("dataset boundary failure".into());
    }

    println!("DATASET VALIDATION");
    println!("------------------");
    println!("Rows        : {}", rows.len());
    println!("First       : {}", rows[0].height);
    println!("Last        : {}", rows[rows.len() - 1].height);
    println!("Consecutive : YES");
    println!("Verified    : ALL");
    println!();

    // ========================================================
    // OUTPUT
    // ========================================================

    let mut writer = Writer::from_path(OUTPUT)?;

    writer.write_record([
        "start_height",
        "end_height",
        "start_timestamp",
        "end_timestamp",
        "n",
        "bit9_ones",
        "bit9_pct",
        "bit8_ones",
        "bit8_pct",
        "bit7_ones",
        "bit7_pct",
        "bit6_ones",
        "bit6_pct",
        "bit5_ones",
        "bit5_pct",
        "l9_000",
        "l9_001",
        "l9_010",
        "l9_011",
        "l9_100",
        "l9_101",
        "l9_110",
        "l9_111",
        "l9_entropy",
        "l9_chi2",
        "bit7_delta_prev_pp",
    ])?;

    println!("1K REGIME MAP");
    println!("-------------");
    println!(
        "{:<15} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9} {:>10}",
        "window",
        "b9%",
        "b8%",
        "b7%",
        "b6%",
        "b5%",
        "L9 H",
        "Δb7(pp)"
    );

    let mut previous_bit7: Option<f64> = None;

    let mut global_bits = [0usize; 32];
    let mut global_l9 = [0usize; 8];

    let mut min_bit7 = (0u64, 101.0f64);
    let mut max_bit7 = (0u64, -1.0f64);

    let mut largest_up =
        (0u64, f64::NEG_INFINITY);

    let mut largest_down =
        (0u64, f64::INFINITY);

    // ========================================================
    // WINDOWS
    // ========================================================

    for chunk in rows.chunks(WINDOW as usize) {
        if chunk.len() != WINDOW as usize {
            return Err(
                format!(
                    "partial window at {}",
                    chunk[0].height
                )
                .into(),
            );
        }

        let start_height = chunk[0].height;
        let end_height = chunk[chunk.len() - 1].height;

        let start_timestamp = chunk[0].timestamp;
        let end_timestamp = chunk[chunk.len() - 1].timestamp;

        let mut bits = [0usize; 32];
        let mut states = [0usize; 8];

        for row in chunk {
            for b in 0..32 {
                if bit(row.nonce, b) == 1 {
                    bits[b] += 1;
                    global_bits[b] += 1;
                }
            }

            let state = l9(row.nonce);
            states[state] += 1;
            global_l9[state] += 1;
        }

        let b9 = pct(bits[9], chunk.len());
        let b8 = pct(bits[8], chunk.len());
        let b7 = pct(bits[7], chunk.len());
        let b6 = pct(bits[6], chunk.len());
        let b5 = pct(bits[5], chunk.len());

        let entropy = entropy8(&states);
        let chi2 = l9_uniform_chi2(&states);

        let delta =
            previous_bit7
                .map(|previous| b7 - previous)
                .unwrap_or(0.0);

        if b7 < min_bit7.1 {
            min_bit7 = (start_height, b7);
        }

        if b7 > max_bit7.1 {
            max_bit7 = (start_height, b7);
        }

        if previous_bit7.is_some() {
            if delta > largest_up.1 {
                largest_up = (start_height, delta);
            }

            if delta < largest_down.1 {
                largest_down = (start_height, delta);
            }
        }

        println!(
            "{}..{} {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}% {:>9.5} {:>+9.2}",
            start_height,
            end_height,
            b9,
            b8,
            b7,
            b6,
            b5,
            entropy,
            delta
        );

        writer.write_record([
            start_height.to_string(),
            end_height.to_string(),
            start_timestamp.to_string(),
            end_timestamp.to_string(),
            chunk.len().to_string(),

            bits[9].to_string(),
            format!("{:.6}", b9),

            bits[8].to_string(),
            format!("{:.6}", b8),

            bits[7].to_string(),
            format!("{:.6}", b7),

            bits[6].to_string(),
            format!("{:.6}", b6),

            bits[5].to_string(),
            format!("{:.6}", b5),

            states[0].to_string(),
            states[1].to_string(),
            states[2].to_string(),
            states[3].to_string(),
            states[4].to_string(),
            states[5].to_string(),
            states[6].to_string(),
            states[7].to_string(),

            format!("{:.9}", entropy),
            format!("{:.9}", chi2),
            format!("{:.6}", delta),
        ])?;

        previous_bit7 = Some(b7);
    }

    writer.flush()?;

    // ========================================================
    // GLOBAL SUMMARY
    // ========================================================

    println!();
    println!("GLOBAL GEOMETRY");
    println!("---------------");

    for b in [9usize, 8, 7, 6, 5] {
        println!(
            "bit {:>2}: {:>7} / {} = {:>9.5}%",
            b,
            global_bits[b],
            rows.len(),
            pct(global_bits[b], rows.len())
        );
    }

    println!();
    println!("L9 GLOBAL");
    println!("---------");

    for state in 0..8 {
        println!(
            "{:03b}: {:>7}  {:>9.5}%",
            state,
            global_l9[state],
            pct(global_l9[state], rows.len())
        );
    }

    println!(
        "Entropy : {:.9} / 3.000000000",
        entropy8(&global_l9)
    );

    println!(
        "Chi²    : {:.9}",
        l9_uniform_chi2(&global_l9)
    );

    // ========================================================
    // TEMPORAL EXTREMES
    // ========================================================

    println!();
    println!("BIT7 TEMPORAL EXTREMES");
    println!("----------------------");

    println!(
        "Minimum 1k window : {}..{} = {:.4}%",
        min_bit7.0,
        min_bit7.0 + WINDOW - 1,
        min_bit7.1
    );

    println!(
        "Maximum 1k window : {}..{} = {:.4}%",
        max_bit7.0,
        max_bit7.0 + WINDOW - 1,
        max_bit7.1
    );

    println!(
        "Largest 1k rise   : starts {} = {:+.4} pp vs previous window",
        largest_up.0,
        largest_up.1
    );

    println!(
        "Largest 1k fall   : starts {} = {:+.4} pp vs previous window",
        largest_down.0,
        largest_down.1
    );

    // ========================================================
    // KNOWN CONSISTENCY
    // ========================================================

    let known: Vec<&Row> = rows
        .iter()
        .filter(|r| r.height >= 890_000)
        .collect();

    let known_bit7 =
        known
            .iter()
            .filter(|r| bit(r.nonce, 7) == 1)
            .count();

    println!();
    println!("V0.19 / V0.21 CONSISTENCY");
    println!("-------------------------");

    println!(
        "890000..899999 bit7 ones: {} / {} = {:.4}%",
        known_bit7,
        known.len(),
        pct(known_bit7, known.len())
    );

    if known_bit7 != 3470
        || known.len() != 10_000
    {
        return Err(
            "known-range consistency failure".into()
        );
    }

    println!("Expected                  : 3470 / 10000 = 34.7000%");
    println!("Integrity                 : PASS");

    println!();
    println!("V0.22 COMPLETE");
    println!("==============");
    println!("Windows             : 200 × 1,000 blocks");
    println!("Primary geometry    : bits 9,8,7,6,5 + L9");
    println!("Adaptive selection  : NONE");
    println!("ML training         : NONE");
    println!("SHA predictability  : NOT CLAIMED");
    println!("CSV                  : {}", OUTPUT);

    Ok(())
}
