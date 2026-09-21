use csv::Reader;
use std::{
    cmp::Ordering,
    error::Error,
};

const INPUT: &str =
    "data/v022-nonce-geometry-1k-frozen.csv";

const EXPECTED_WINDOWS: usize = 200;
const STATES: usize = 8;

// Warm-up: first 20 windows = 20,000 blocks.
// No scoring during warm-up.
const WARMUP: usize = 20;

// Exponential moving average.
// Higher alpha = more weight on recent regime.
const ALPHA: f64 = 0.20;

#[derive(Debug, Clone)]
struct Window {
    start: u64,
    end: u64,
    n: u64,
    counts: [u64; STATES],
}

fn parse_u64(
    record: &csv::StringRecord,
    idx: usize,
) -> Result<u64, Box<dyn Error>> {
    Ok(record
        .get(idx)
        .ok_or("missing field")?
        .parse::<u64>()?)
}

fn normalize_counts(
    counts: &[u64; STATES],
) -> [f64; STATES] {
    let total: u64 = counts.iter().sum();

    let mut p = [0.0; STATES];

    if total == 0 {
        return p;
    }

    for i in 0..STATES {
        p[i] = counts[i] as f64 / total as f64;
    }

    p
}

fn rank_states(
    prediction: &[f64; STATES],
) -> [usize; STATES] {
    let mut indices =
        [0usize, 1, 2, 3, 4, 5, 6, 7];

    indices.sort_by(|&a, &b| {
        prediction[b]
            .partial_cmp(&prediction[a])
            .unwrap_or(Ordering::Equal)
            // deterministic tie-break
            .then_with(|| a.cmp(&b))
    });

    indices
}

fn recall_for_k(
    actual: &[u64; STATES],
    ranking: &[usize; STATES],
    k: usize,
) -> f64 {
    let total: u64 = actual.iter().sum();

    if total == 0 {
        return 0.0;
    }

    let kept: u64 = ranking[..k]
        .iter()
        .map(|&state| actual[state])
        .sum();

    kept as f64 / total as f64
}

fn selected_count(
    actual: &[u64; STATES],
    ranking: &[usize; STATES],
    k: usize,
) -> u64 {
    ranking[..k]
        .iter()
        .map(|&state| actual[state])
        .sum()
}

fn concentration_gain(
    recall: f64,
    k: usize,
) -> f64 {
    let random_recall =
        k as f64 / STATES as f64;

    recall / random_recall
}

fn brier_score(
    predicted: &[f64; STATES],
    actual: &[f64; STATES],
) -> f64 {
    let mut score = 0.0;

    for i in 0..STATES {
        let d = predicted[i] - actual[i];
        score += d * d;
    }

    score / STATES as f64
}

fn fmt_state(state: usize) -> String {
    format!("{:03b}", state)
}

fn fmt_top(
    ranking: &[usize; STATES],
    k: usize,
) -> String {
    ranking[..k]
        .iter()
        .map(|&s| fmt_state(s))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 v0.23 Walk-Forward Observer");
    println!("=================================");
    println!("Input          : {}", INPUT);
    println!("Windows        : {}", EXPECTED_WINDOWS);
    println!("Window size    : 1,000 blocks");
    println!("Warm-up        : {} windows", WARMUP);
    println!("Observer       : EMA of past L9 distributions");
    println!("Alpha          : {:.3}", ALPHA);
    println!("Future leakage : NONE");
    println!();

    // ========================================================
    // LOAD V0.22 FROZEN GEOMETRY
    // ========================================================

    let mut reader = Reader::from_path(INPUT)?;
    let headers = reader.headers()?.clone();

    let index = |name: &str|
        -> Result<usize, Box<dyn Error>> {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| {
                format!("missing column: {}", name).into()
            })
    };

    let start_idx = index("start_height")?;
    let end_idx = index("end_height")?;
    let n_idx = index("n")?;

    let state_names = [
        "l9_000",
        "l9_001",
        "l9_010",
        "l9_011",
        "l9_100",
        "l9_101",
        "l9_110",
        "l9_111",
    ];

    let mut state_indices = [0usize; STATES];

    for i in 0..STATES {
        state_indices[i] = index(state_names[i])?;
    }

    let mut windows =
        Vec::<Window>::with_capacity(EXPECTED_WINDOWS);

    for record in reader.records() {
        let record = record?;

        let start = parse_u64(&record, start_idx)?;
        let end = parse_u64(&record, end_idx)?;
        let n = parse_u64(&record, n_idx)?;

        let mut counts = [0u64; STATES];

        for i in 0..STATES {
            counts[i] =
                parse_u64(&record, state_indices[i])?;
        }

        let sum: u64 = counts.iter().sum();

        if sum != n {
            return Err(
                format!(
                    "L9 count mismatch {}..{}: {} != {}",
                    start, end, sum, n
                )
                .into(),
            );
        }

        windows.push(Window {
            start,
            end,
            n,
            counts,
        });
    }

    // ========================================================
    // VALIDATE
    // ========================================================

    if windows.len() != EXPECTED_WINDOWS {
        return Err(
            format!(
                "expected {} windows, got {}",
                EXPECTED_WINDOWS,
                windows.len()
            )
            .into(),
        );
    }

    for (i, w) in windows.iter().enumerate() {
        let expected_start =
            700_000 + i as u64 * 1_000;

        let expected_end =
            expected_start + 999;

        if w.start != expected_start
            || w.end != expected_end
            || w.n != 1_000
        {
            return Err(
                format!(
                    "window validation failed at index {}",
                    i
                )
                .into(),
            );
        }
    }

    println!("DATASET VALIDATION");
    println!("------------------");
    println!("Windows     : {}", windows.len());
    println!(
        "Range       : {} .. {}",
        windows[0].start,
        windows[windows.len() - 1].end
    );
    println!("L9 totals   : VALID");
    println!("Chronology  : VALID");
    println!();

    // ========================================================
    // INITIALIZE OBSERVER USING WARM-UP ONLY
    // ========================================================

    let mut ema = [0.0f64; STATES];

    for i in 0..WARMUP {
        let p = normalize_counts(&windows[i].counts);

        if i == 0 {
            ema = p;
        } else {
            for s in 0..STATES {
                ema[s] =
                    ALPHA * p[s]
                    + (1.0 - ALPHA) * ema[s];
            }
        }
    }

    // ========================================================
    // WALK-FORWARD
    // ========================================================

    let scored_windows =
        windows.len() - WARMUP;

    let mut total_hits = [0u64; 5];

    let mut total_blocks = 0u64;

    let mut total_brier = 0.0;

    let mut wins_vs_random = [0usize; 5];
    let mut losses_vs_random = [0usize; 5];
    let mut ties_vs_random = [0usize; 5];

    println!("WALK-FORWARD PREDICTIONS");
    println!("------------------------");

    println!(
        "{:<15} {:<7} {:<11} {:>8} {:>8} {:>8} {:>8} {:>9}",
        "window",
        "top1",
        "top2",
        "R@1",
        "R@2",
        "R@3",
        "R@4",
        "Brier"
    );

    for i in WARMUP..windows.len() {
        let w = &windows[i];

        // IMPORTANT:
        // ranking is produced BEFORE observing this window.
        let ranking = rank_states(&ema);

        let actual_p =
            normalize_counts(&w.counts);

        let brier =
            brier_score(&ema, &actual_p);

        total_brier += brier;
        total_blocks += w.n;

        let mut recalls = [0.0f64; 5];

        for k in 1..=4 {
            let hits =
                selected_count(
                    &w.counts,
                    &ranking,
                    k,
                );

            total_hits[k] += hits;

            let recall =
                hits as f64 / w.n as f64;

            recalls[k] = recall;

            let random =
                k as f64 / STATES as f64;

            if recall > random {
                wins_vs_random[k] += 1;
            } else if recall < random {
                losses_vs_random[k] += 1;
            } else {
                ties_vs_random[k] += 1;
            }
        }

        println!(
            "{}..{} {:<7} {:<11} {:>7.2}% {:>7.2}% {:>7.2}% {:>7.2}% {:>9.6}",
            w.start,
            w.end,
            fmt_top(&ranking, 1),
            fmt_top(&ranking, 2),
            recalls[1] * 100.0,
            recalls[2] * 100.0,
            recalls[3] * 100.0,
            recalls[4] * 100.0,
            brier
        );

        // ====================================================
        // ONLY NOW MAY THE OBSERVER LEARN THIS WINDOW
        // ====================================================

        for s in 0..STATES {
            ema[s] =
                ALPHA * actual_p[s]
                + (1.0 - ALPHA) * ema[s];
        }
    }

    // ========================================================
    // SUMMARY
    // ========================================================

    println!();
    println!("WALK-FORWARD SUMMARY");
    println!("--------------------");

    println!(
        "Scored windows : {}",
        scored_windows
    );

    println!(
        "Scored blocks  : {}",
        total_blocks
    );

    println!(
        "Mean Brier     : {:.9}",
        total_brier / scored_windows as f64
    );

    println!();

    println!(
        "{:<5} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "K",
        "Recall",
        "Random",
        "Conc.Gain",
        "Reduction",
        "W/L/T"
    );

    for k in 1..=4 {
        let recall =
            total_hits[k] as f64
            / total_blocks as f64;

        let random =
            k as f64 / STATES as f64;

        let gain =
            concentration_gain(recall, k);

        let reduction =
            1.0 - random;

        println!(
            "{:<5} {:>11.5}% {:>11.5}% {:>11.6} {:>11.2}% {:>3}/{:<3}/{:<3}",
            k,
            recall * 100.0,
            random * 100.0,
            gain,
            reduction * 100.0,
            wins_vs_random[k],
            losses_vs_random[k],
            ties_vs_random[k]
        );
    }

    // ========================================================
    // FINAL OBSERVER STATE
    // ========================================================

    println!();
    println!("FINAL OBSERVER DISTRIBUTION");
    println!("---------------------------");

    let final_ranking = rank_states(&ema);

    for &state in &final_ranking {
        println!(
            "{}  {:.6}%",
            fmt_state(state),
            ema[state] * 100.0
        );
    }

    println!();

    println!("INTERPRETATION RULES");
    println!("--------------------");
    println!("Random R@1 = 12.5%");
    println!("Random R@2 = 25.0%");
    println!("Random R@3 = 37.5%");
    println!("Random R@4 = 50.0%");
    println!();
    println!("ConcentrationGain > 1 means the");
    println!("past-only Observer concentrated");
    println!("future historical winning nonces");
    println!("better than uniform state choice.");
    println!();
    println!("This DOES NOT establish SHA256");
    println!("predictability or mining advantage.");
    println!("It tests temporal predictability of");
    println!("the historical winning-nonce regime.");

    println!();
    println!("V0.23 COMPLETE");
    println!("==============");
    println!("Future leakage : NONE");
    println!("Adaptive ML    : NONE");
    println!("Observer       : past-only EMA");
    println!("Collapse       : N=8 -> K=1..4");

    Ok(())
}
