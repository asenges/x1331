use csv::Reader;
use std::{
    cmp::Ordering,
    error::Error,
};

const INPUT: &str =
    "data/v022-nonce-geometry-1k-frozen.csv";

const EXPECTED_WINDOWS: usize = 200;
const STATES: usize = 8;
const WARMUP: usize = 20;

// Frozen from v0.23. Do not tune here.
const ALPHA: f64 = 0.20;

#[derive(Debug, Clone)]
struct Window {
    start: u64,
    end: u64,
    n: u64,
    counts: [u64; STATES],
}

#[derive(Clone)]
struct ModelStats {
    name: &'static str,
    hits: [u64; 5],
    wins_vs_uniform: [usize; 5],
    losses_vs_uniform: [usize; 5],
    ties_vs_uniform: [usize; 5],
    brier_sum: f64,
    top1_changes: usize,
    previous_top1: Option<usize>,
}

impl ModelStats {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            hits: [0; 5],
            wins_vs_uniform: [0; 5],
            losses_vs_uniform: [0; 5],
            ties_vs_uniform: [0; 5],
            brier_sum: 0.0,
            top1_changes: 0,
            previous_top1: None,
        }
    }
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

fn normalize_u64(
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

fn normalize_f64(
    values: &[f64; STATES],
) -> [f64; STATES] {
    let total: f64 = values.iter().sum();
    let mut p = [0.0; STATES];

    if total <= 0.0 {
        return [1.0 / STATES as f64; STATES];
    }

    for i in 0..STATES {
        p[i] = values[i] / total;
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
            .then_with(|| a.cmp(&b))
    });

    indices
}

fn selected_count(
    actual: &[u64; STATES],
    ranking: &[usize; STATES],
    k: usize,
) -> u64 {
    ranking[..k]
        .iter()
        .map(|&s| actual[s])
        .sum()
}

fn brier(
    predicted: &[f64; STATES],
    actual: &[f64; STATES],
) -> f64 {
    let mut total = 0.0;

    for i in 0..STATES {
        let d = predicted[i] - actual[i];
        total += d * d;
    }

    total / STATES as f64
}

fn fmt_state(s: usize) -> String {
    format!("{:03b}", s)
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

fn score_model(
    stats: &mut ModelStats,
    prediction: &[f64; STATES],
    actual_counts: &[u64; STATES],
    actual_p: &[f64; STATES],
    n: u64,
) {
    let ranking = rank_states(prediction);

    if let Some(prev) = stats.previous_top1 {
        if prev != ranking[0] {
            stats.top1_changes += 1;
        }
    }

    stats.previous_top1 = Some(ranking[0]);

    stats.brier_sum += brier(prediction, actual_p);

    for k in 1..=4 {
        let hits =
            selected_count(actual_counts, &ranking, k);

        stats.hits[k] += hits;

        let recall = hits as f64 / n as f64;
        let uniform = k as f64 / STATES as f64;

        if recall > uniform {
            stats.wins_vs_uniform[k] += 1;
        } else if recall < uniform {
            stats.losses_vs_uniform[k] += 1;
        } else {
            stats.ties_vs_uniform[k] += 1;
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("X1331 v0.24 Observer Ablation Laboratory");
    println!("=========================================");
    println!("Input          : {}", INPUT);
    println!("Windows        : {}", EXPECTED_WINDOWS);
    println!("Warm-up        : {}", WARMUP);
    println!("EMA alpha      : {:.3} (FROZEN v0.23)", ALPHA);
    println!("Scoring        : same 180 windows as v0.23");
    println!("Future leakage : NONE");
    println!();

    // ========================================================
    // LOAD
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

    let mut state_idx = [0usize; STATES];

    for i in 0..STATES {
        state_idx[i] = index(state_names[i])?;
    }

    let mut windows =
        Vec::<Window>::with_capacity(EXPECTED_WINDOWS);

    for record in reader.records() {
        let record = record?;

        let start = parse_u64(&record, start_idx)?;
        let end = parse_u64(&record, end_idx)?;
        let n = parse_u64(&record, n_idx)?;

        let mut counts = [0u64; STATES];

        for s in 0..STATES {
            counts[s] =
                parse_u64(&record, state_idx[s])?;
        }

        if counts.iter().sum::<u64>() != n {
            return Err(
                format!(
                    "count mismatch at {}..{}",
                    start, end
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

        if w.start != expected_start
            || w.end != expected_start + 999
            || w.n != 1_000
        {
            return Err(
                format!(
                    "window validation failed at {}",
                    i
                )
                .into(),
            );
        }
    }

    println!("DATASET VALIDATION");
    println!("------------------");
    println!("Windows    : {}", windows.len());
    println!(
        "Range      : {}..{}",
        windows[0].start,
        windows.last().unwrap().end
    );
    println!("Chronology : VALID");
    println!("L9 totals  : VALID");
    println!();

    // ========================================================
    // WARM-UP MODELS
    // ========================================================

    let uniform =
        [1.0 / STATES as f64; STATES];

    // Frozen prior: only first 20 windows.
    let mut frozen_counts = [0u64; STATES];

    for w in &windows[..WARMUP] {
        for s in 0..STATES {
            frozen_counts[s] += w.counts[s];
        }
    }

    let frozen_prior =
        normalize_u64(&frozen_counts);

    // EMA initialized identically to v0.23.
    let mut ema = [0.0f64; STATES];

    for i in 0..WARMUP {
        let p = normalize_u64(&windows[i].counts);

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

    // Cumulative past starts with warm-up counts.
    let mut cumulative =
        [0u64; STATES];

    for s in 0..STATES {
        cumulative[s] = frozen_counts[s];
    }

    // Previous-window baseline.
    let mut previous =
        normalize_u64(
            &windows[WARMUP - 1].counts
        );

    let frozen_rank =
        rank_states(&frozen_prior);

    println!("FROZEN PRIOR");
    println!("------------");

    for &s in &frozen_rank {
        println!(
            "{}  {:.6}%",
            fmt_state(s),
            frozen_prior[s] * 100.0
        );
    }

    println!();

    // ========================================================
    // STATS
    // ========================================================

    let mut uniform_stats =
        ModelStats::new("Uniform");

    let mut frozen_stats =
        ModelStats::new("FrozenPrior");

    let mut previous_stats =
        ModelStats::new("Previous");

    let mut cumulative_stats =
        ModelStats::new("Cumulative");

    let mut ema_stats =
        ModelStats::new("EMA");

    let mut total_blocks = 0u64;

    println!("WALK-FORWARD ABLATION");
    println!("---------------------");
    println!(
        "{:<15} {:<7} {:<7} {:<7} {:<7}",
        "window",
        "Frozen",
        "Prev",
        "Cum",
        "EMA"
    );

    // ========================================================
    // WALK FORWARD
    // ========================================================

    for i in WARMUP..windows.len() {
        let w = &windows[i];
        let actual_p =
            normalize_u64(&w.counts);

        let cumulative_f64 = {
            let mut tmp = [0.0f64; STATES];

            for s in 0..STATES {
                tmp[s] = cumulative[s] as f64;
            }

            normalize_f64(&tmp)
        };

        // Predictions exist BEFORE current outcome.
        let frozen_r =
            rank_states(&frozen_prior);

        let prev_r =
            rank_states(&previous);

        let cum_r =
            rank_states(&cumulative_f64);

        let ema_r =
            rank_states(&ema);

        println!(
            "{}..{} {:<7} {:<7} {:<7} {:<7}",
            w.start,
            w.end,
            fmt_top(&frozen_r, 1),
            fmt_top(&prev_r, 1),
            fmt_top(&cum_r, 1),
            fmt_top(&ema_r, 1)
        );

        score_model(
            &mut uniform_stats,
            &uniform,
            &w.counts,
            &actual_p,
            w.n,
        );

        score_model(
            &mut frozen_stats,
            &frozen_prior,
            &w.counts,
            &actual_p,
            w.n,
        );

        score_model(
            &mut previous_stats,
            &previous,
            &w.counts,
            &actual_p,
            w.n,
        );

        score_model(
            &mut cumulative_stats,
            &cumulative_f64,
            &w.counts,
            &actual_p,
            w.n,
        );

        score_model(
            &mut ema_stats,
            &ema,
            &w.counts,
            &actual_p,
            w.n,
        );

        total_blocks += w.n;

        // ====================================================
        // LEARN CURRENT WINDOW ONLY AFTER SCORING
        // ====================================================

        for s in 0..STATES {
            cumulative[s] += w.counts[s];

            ema[s] =
                ALPHA * actual_p[s]
                + (1.0 - ALPHA) * ema[s];
        }

        previous = actual_p;
    }

    let scored =
        EXPECTED_WINDOWS - WARMUP;

    // ========================================================
    // RESULTS
    // ========================================================

    let models = [
        &uniform_stats,
        &frozen_stats,
        &previous_stats,
        &cumulative_stats,
        &ema_stats,
    ];

    println!();
    println!("MODEL SUMMARY");
    println!("-------------");
    println!(
        "{:<13} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Model",
        "R@1",
        "R@2",
        "R@3",
        "R@4",
        "Brier"
    );

    for model in models {
        println!(
            "{:<13} {:>11.5}% {:>11.5}% {:>11.5}% {:>11.5}% {:>12.9}",
            model.name,
            model.hits[1] as f64
                / total_blocks as f64
                * 100.0,
            model.hits[2] as f64
                / total_blocks as f64
                * 100.0,
            model.hits[3] as f64
                / total_blocks as f64
                * 100.0,
            model.hits[4] as f64
                / total_blocks as f64
                * 100.0,
            model.brier_sum / scored as f64
        );
    }

    println!();

    // ========================================================
    // CONCENTRATION VS UNIFORM
    // ========================================================

    println!("CONCENTRATION GAIN VS UNIFORM");
    println!("-----------------------------");
    println!(
        "{:<13} {:>10} {:>10} {:>10} {:>10}",
        "Model",
        "K1",
        "K2",
        "K3",
        "K4"
    );

    for model in [
        &frozen_stats,
        &previous_stats,
        &cumulative_stats,
        &ema_stats,
    ] {
        print!("{:<13}", model.name);

        for k in 1..=4 {
            let recall =
                model.hits[k] as f64
                / total_blocks as f64;

            let random =
                k as f64 / STATES as f64;

            print!(
                " {:>9.5}",
                recall / random
            );
        }

        println!();
    }

    println!();

    // ========================================================
    // EMA VS STATIC
    // ========================================================

    println!("DYNAMIC VALUE — EMA VS FROZEN PRIOR");
    println!("-----------------------------------");
    println!(
        "{:<5} {:>13} {:>13} {:>13} {:>13}",
        "K",
        "FrozenRecall",
        "EMARecall",
        "EMA/Frozen",
        "Delta(pp)"
    );

    for k in 1..=4 {
        let frozen_recall =
            frozen_stats.hits[k] as f64
            / total_blocks as f64;

        let ema_recall =
            ema_stats.hits[k] as f64
            / total_blocks as f64;

        println!(
            "{:<5} {:>12.5}% {:>12.5}% {:>13.6} {:>+12.5}",
            k,
            frozen_recall * 100.0,
            ema_recall * 100.0,
            ema_recall / frozen_recall,
            (ema_recall - frozen_recall) * 100.0
        );
    }

    println!();

    // ========================================================
    // WINDOW-LEVEL ROBUSTNESS
    // ========================================================

    println!("WINDOW ROBUSTNESS VS UNIFORM");
    println!("----------------------------");
    println!(
        "{:<13} {:>12} {:>12} {:>12} {:>12}",
        "Model",
        "K1 W/L/T",
        "K2 W/L/T",
        "K3 W/L/T",
        "K4 W/L/T"
    );

    for model in [
        &frozen_stats,
        &previous_stats,
        &cumulative_stats,
        &ema_stats,
    ] {
        print!("{:<13}", model.name);

        for k in 1..=4 {
            print!(
                " {:>3}/{:<3}/{:<3}",
                model.wins_vs_uniform[k],
                model.losses_vs_uniform[k],
                model.ties_vs_uniform[k]
            );
        }

        println!();
    }

    println!();

    // ========================================================
    // ADAPTATION
    // ========================================================

    println!("TOP1 ADAPTATION");
    println!("---------------");
    println!(
        "Frozen changes     : {}",
        frozen_stats.top1_changes
    );
    println!(
        "Previous changes   : {}",
        previous_stats.top1_changes
    );
    println!(
        "Cumulative changes : {}",
        cumulative_stats.top1_changes
    );
    println!(
        "EMA changes        : {}",
        ema_stats.top1_changes
    );

    println!();

    // ========================================================
    // V0.23 REPRODUCTION
    // ========================================================

    let ema_r1 =
        ema_stats.hits[1] as f64
        / total_blocks as f64
        * 100.0;

    let ema_r2 =
        ema_stats.hits[2] as f64
        / total_blocks as f64
        * 100.0;

    let ema_r3 =
        ema_stats.hits[3] as f64
        / total_blocks as f64
        * 100.0;

    let ema_r4 =
        ema_stats.hits[4] as f64
        / total_blocks as f64
        * 100.0;

    println!("V0.23 REPRODUCTION CHECK");
    println!("------------------------");
    println!("Expected:");
    println!("R@1 = 18.03611%");
    println!("R@2 = 36.13556%");
    println!("R@3 = 53.90611%");
    println!("R@4 = 71.65167%");
    println!();
    println!("Observed:");
    println!("R@1 = {:.5}%", ema_r1);
    println!("R@2 = {:.5}%", ema_r2);
    println!("R@3 = {:.5}%", ema_r3);
    println!("R@4 = {:.5}%", ema_r4);

    let tolerance = 0.00001;

    if (ema_r1 - 18.03611).abs() > tolerance
        || (ema_r2 - 36.13556).abs() > tolerance
        || (ema_r3 - 53.90611).abs() > tolerance
        || (ema_r4 - 71.65167).abs() > tolerance
    {
        return Err(
            "v0.23 reproduction check FAILED".into()
        );
    }

    println!("Integrity : PASS");

    println!();
    println!("V0.24 COMPLETE");
    println!("==============");
    println!("Scored windows : {}", scored);
    println!("Scored blocks  : {}", total_blocks);
    println!("Alpha tuned    : NO");
    println!("Future leakage : NONE");
    println!("Purpose        : diagnostic ablation");
    println!();
    println!("Do NOT interpret this experiment as");
    println!("evidence of SHA256 predictability.");
    println!("It separates static historical bias");
    println!("from temporal adaptation value.");

    Ok(())
}
