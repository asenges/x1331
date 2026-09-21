use csv::Reader;
use std::{
    collections::BTreeMap,
    error::Error,
};

const INPUT: &str = "data/bitcoin-history-v020-frozen.csv";
const EXPECTED_ROWS: usize = 200_000;
const START_HEIGHT: u64 = 700_000;
const END_HEIGHT: u64 = 899_999;

#[derive(Clone)]
struct Block {
    height: u64,
    version: u32,
    timestamp: u64,
    bits: u32,
    nonce: u32,
    l9: usize,
}

#[derive(Default, Clone)]
struct Stats {
    n: u64,
    states: [u64; 8],
    bit7_ones: u64,
}

impl Stats {
    fn add(&mut self, b: &Block) {
        self.n += 1;
        self.states[b.l9] += 1;

        if ((b.nonce >> 7) & 1) == 1 {
            self.bit7_ones += 1;
        }
    }

    fn bit7_rate(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.bit7_ones as f64 / self.n as f64
        }
    }

    fn entropy(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }

        let mut h = 0.0;

        for &count in &self.states {
            if count == 0 {
                continue;
            }

            let p = count as f64 / self.n as f64;
            h -= p * p.log2();
        }

        h
    }

    fn chi_square_uniform(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }

        let expected = self.n as f64 / 8.0;

        self.states
            .iter()
            .map(|&obs| {
                let d = obs as f64 - expected;
                d * d / expected
            })
            .sum()
    }
}

fn parse_u32_auto(s: &str) -> Result<u32, Box<dyn Error>> {
    let s = s.trim();

    if let Some(hex) = s.strip_prefix("0x") {
        Ok(u32::from_str_radix(hex, 16)?)
    } else if s.chars().any(|c| c.is_ascii_alphabetic()) {
        Ok(u32::from_str_radix(s, 16)?)
    } else {
        Ok(s.parse::<u32>()?)
    }
}

fn find_column(
    headers: &csv::StringRecord,
    names: &[&str],
) -> Option<usize> {
    for name in names {
        if let Some(i) = headers.iter().position(|h| h == *name) {
            return Some(i);
        }
    }

    None
}

fn print_stats(label: &str, s: &Stats) {
    println!("{}", label);
    println!(
        "  N={} bit7={:.4}% entropy={:.6}/3 chi2={:.3}",
        s.n,
        s.bit7_rate() * 100.0,
        s.entropy(),
        s.chi_square_uniform()
    );

    print!("  L9:");

    for state in 0..8 {
        let pct = if s.n == 0 {
            0.0
        } else {
            s.states[state] as f64 / s.n as f64 * 100.0
        };

        print!(
            " {:03b}={:.3}%",
            state,
            pct
        );
    }

    println!();
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 v0.27 Challenge Anatomy Laboratory");
    println!("========================================");
    println!("Input           : {}", INPUT);
    println!("Range           : {}..{}", START_HEIGHT, END_HEIGHT);
    println!("Expected rows   : {}", EXPECTED_ROWS);
    println!("Target          : historical winning nonce L9");
    println!("Adaptive ML     : NONE");
    println!("SHA prediction  : NOT TESTED");
    println!("Purpose         : mining-process attribution");
    println!();

    let mut reader = Reader::from_path(INPUT)?;
    let headers = reader.headers()?.clone();

    println!("CSV COLUMNS");
    println!("-----------");
    for (i, h) in headers.iter().enumerate() {
        println!("{:>2}: {}", i, h);
    }
    println!();

    let height_idx =
        find_column(&headers, &["height"])
            .ok_or("missing height")?;

    let header_idx =
        find_column(&headers, &["header_hex", "header"])
            .ok_or("missing header_hex/header")?;

    let verified_idx =
        find_column(&headers, &["verified"]);

    let mut blocks = Vec::with_capacity(EXPECTED_ROWS);

    for (row_i, rec) in reader.records().enumerate() {
        let rec = rec?;

        let height: u64 = rec[height_idx].parse()?;

        let expected_height = START_HEIGHT + row_i as u64;

        if height != expected_height {
            return Err(
                format!(
                    "chronology failure: expected {}, got {}",
                    expected_height,
                    height
                )
                .into()
            );
        }

        if let Some(idx) = verified_idx {
            let v = rec[idx].to_ascii_lowercase();

            if !matches!(v.as_str(), "true" | "1" | "yes") {
                return Err(
                    format!("unverified row {}", height).into()
                );
            }
        }

        let raw = hex::decode(&rec[header_idx])?;

        if raw.len() != 80 {
            return Err(
                format!(
                    "header length {} at {}",
                    raw.len(),
                    height
                )
                .into()
            );
        }

        let version = u32::from_le_bytes(
            raw[0..4].try_into()?
        );

        let timestamp = u32::from_le_bytes(
            raw[68..72].try_into()?
        ) as u64;

        let bits = u32::from_le_bytes(
            raw[72..76].try_into()?
        );

        let nonce = u32::from_le_bytes(
            raw[76..80].try_into()?
        );

        let l9 = ((nonce >> 5) & 0b111) as usize;

        blocks.push(Block {
            height,
            version,
            timestamp,
            bits,
            nonce,
            l9,
        });
    }

    if blocks.len() != EXPECTED_ROWS {
        return Err(
            format!(
                "expected {} rows, got {}",
                EXPECTED_ROWS,
                blocks.len()
            )
            .into()
        );
    }

    if blocks.first().unwrap().height != START_HEIGHT
        || blocks.last().unwrap().height != END_HEIGHT
    {
        return Err("height range mismatch".into());
    }

    println!("VALIDATION");
    println!("----------");
    println!("Rows       : {}", blocks.len());
    println!(
        "Range      : {}..{}",
        blocks.first().unwrap().height,
        blocks.last().unwrap().height
    );
    println!("Chronology : PASS");
    println!("Headers    : 80 bytes");
    println!();

    // =====================================================
    // GLOBAL
    // =====================================================

    let mut global = Stats::default();

    for b in &blocks {
        global.add(b);
    }

    println!("GLOBAL");
    println!("------");
    print_stats("All blocks", &global);
    println!();

    // =====================================================
    // 10K CHRONOLOGICAL WINDOWS
    // =====================================================

    println!("10K CHRONOLOGICAL WINDOWS");
    println!("-------------------------");

    for chunk in blocks.chunks(10_000) {
        let mut s = Stats::default();

        for b in chunk {
            s.add(b);
        }

        let first = chunk.first().unwrap().height;
        let last = chunk.last().unwrap().height;

        print_stats(
            &format!("{}..{}", first, last),
            &s
        );
    }

    println!();

    // =====================================================
    // DIFFICULTY PERIODS
    // =====================================================

    println!("DIFFICULTY PERIODS");
    println!("------------------");

    let mut epochs: BTreeMap<u64, Stats> = BTreeMap::new();

    for b in &blocks {
        let epoch_start =
            (b.height / 2016) * 2016;

        epochs
            .entry(epoch_start)
            .or_default()
            .add(b);
    }

    for (start, s) in &epochs {
        print_stats(
            &format!(
                "{}..{}",
                start,
                start + 2015
            ),
            s
        );
    }

    println!();

    // =====================================================
    // BITS VALUE
    // =====================================================

    println!("NBITS REGIMES");
    println!("-------------");

    let mut bits_map: BTreeMap<u32, Stats> =
        BTreeMap::new();

    for b in &blocks {
        bits_map
            .entry(b.bits)
            .or_default()
            .add(b);
    }

    let mut bits_vec:
        Vec<(u32, Stats)> =
        bits_map.into_iter().collect();

    bits_vec.sort_by(|a, b| {
        b.1.n.cmp(&a.1.n)
    });

    for (bits, s) in &bits_vec {
        println!(
            "bits=0x{:08x}",
            bits
        );
        print_stats(" ", s);
    }

    println!();

    // =====================================================
    // VERSION EXACT
    // =====================================================

    println!("TOP EXACT VERSION VALUES");
    println!("------------------------");

    let mut version_map:
        BTreeMap<u32, Stats> =
        BTreeMap::new();

    for b in &blocks {
        version_map
            .entry(b.version)
            .or_default()
            .add(b);
    }

    let mut versions:
        Vec<(u32, Stats)> =
        version_map.into_iter().collect();

    versions.sort_by(|a, b| {
        b.1.n.cmp(&a.1.n)
    });

    for (version, s) in versions.iter().take(30) {
        println!(
            "version=0x{:08x}",
            version
        );
        print_stats(" ", s);
    }

    println!();

    // =====================================================
    // VERSION LOW BYTE
    // =====================================================

    println!("VERSION LOW-BYTE GROUPS");
    println!("-----------------------");

    let mut version_low:
        BTreeMap<u8, Stats> =
        BTreeMap::new();

    for b in &blocks {
        let key = (b.version & 0xff) as u8;

        version_low
            .entry(key)
            .or_default()
            .add(b);
    }

    let mut low_vec:
        Vec<(u8, Stats)> =
        version_low.into_iter().collect();

    low_vec.sort_by(|a, b| {
        b.1.n.cmp(&a.1.n)
    });

    for (v, s) in low_vec.iter().take(32) {
        if s.n < 100 {
            continue;
        }

        println!("version_low=0x{:02x}", v);
        print_stats(" ", s);
    }

    println!();

    // =====================================================
    // VERSION ROLLING MASK
    // =====================================================

    println!("VERSION ROLLING FIELD");
    println!("---------------------");
    println!("Grouping exploratory version bits 13..28.");
    println!("This is descriptive, not causal attribution.");
    println!();

    let mut rolling:
        BTreeMap<u16, Stats> =
        BTreeMap::new();

    for b in &blocks {
        let key =
            ((b.version >> 13) & 0xffff) as u16;

        rolling
            .entry(key)
            .or_default()
            .add(b);
    }

    let mut rolling_vec:
        Vec<(u16, Stats)> =
        rolling.into_iter().collect();

    rolling_vec.sort_by(|a, b| {
        b.1.n.cmp(&a.1.n)
    });

    for (mask, s) in rolling_vec.iter().take(30) {
        if s.n < 100 {
            continue;
        }

        println!("rolling=0x{:04x}", mask);
        print_stats(" ", s);
    }

    println!();

    // =====================================================
    // INTER-BLOCK TIME
    // =====================================================

    println!("INTER-BLOCK TIME BUCKETS");
    println!("------------------------");

    let labels = [
        "<60s",
        "60-299s",
        "300-599s",
        "600-1199s",
        ">=1200s",
    ];

    let mut dt_stats: [Stats; 5] =
        std::array::from_fn(|_| Stats::default());

    let mut negative_or_equal = 0u64;

    for i in 1..blocks.len() {
        let cur = &blocks[i];
        let prev = &blocks[i - 1];

        if cur.timestamp <= prev.timestamp {
            negative_or_equal += 1;
        }

        let dt =
            cur.timestamp.saturating_sub(prev.timestamp);

        let bucket =
            if dt < 60 {
                0
            } else if dt < 300 {
                1
            } else if dt < 600 {
                2
            } else if dt < 1200 {
                3
            } else {
                4
            };

        dt_stats[bucket].add(cur);
    }

    for i in 0..5 {
        print_stats(labels[i], &dt_stats[i]);
    }

    println!(
        "Non-increasing timestamps: {}",
        negative_or_equal
    );
    println!();

    // =====================================================
    // TIMESTAMP LOW BITS
    // =====================================================

    println!("TIMESTAMP LOW-BIT GROUPS");
    println!("------------------------");

    for width in [1u32, 2, 3, 4] {
        println!("timestamp mod 2^{}", width);

        let size = 1usize << width;
        let mut groups:
            Vec<Stats> =
            (0..size)
                .map(|_| Stats::default())
                .collect();

        let mask = (1u64 << width) - 1;

        for b in &blocks {
            let key =
                (b.timestamp & mask) as usize;

            groups[key].add(b);
        }

        for (key, s) in groups.iter().enumerate() {
            print_stats(
                &format!("  {:0width$b}", key, width = width as usize),
                s
            );
        }

        println!();
    }

    // =====================================================
    // PREVIOUS BLOCK PROCESS CONTINUITY
    // =====================================================

    println!("ADJACENT L9 CONTINUITY");
    println!("----------------------");

    let mut transition =
        [[0u64; 8]; 8];

    for i in 1..blocks.len() {
        let a = blocks[i - 1].l9;
        let b = blocks[i].l9;

        transition[a][b] += 1;
    }

    print!("prev\\next");

    for s in 0..8 {
        print!(" {:>8}", format!("{:03b}", s));
    }

    println!();

    for a in 0..8 {
        print!("{:03b}      ", a);

        let total: u64 =
            transition[a].iter().sum();

        for b in 0..8 {
            let pct =
                if total == 0 {
                    0.0
                } else {
                    transition[a][b] as f64
                        / total as f64
                        * 100.0
                };

            print!(" {:>7.3}", pct);
        }

        println!();
    }

    println!();

    // =====================================================
    // SUMMARY
    // =====================================================

    println!("V0.27 COMPLETE");
    println!("==============");
    println!("Historical winners only.");
    println!("No counterfactual SHA landscape used.");
    println!("No adaptive feature selection.");
    println!("No pool identity inferred.");
    println!("No SHA weakness claimed.");
    println!();
    println!("Interpretation:");
    println!("If L9 geometry changes strongly across");
    println!("version/time/process regimes, that supports");
    println!("investigating mining-work construction.");
    println!();
    println!("The next independent evidence source is");
    println!("coinbase metadata from sampled full blocks.");

    Ok(())
}
