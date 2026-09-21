mod core;

use rand::RngExt;

use crate::core::predictor::TinyPredictor;
use crate::core::synthetic::SyntheticContext;

const TRAINING: usize = 20_000;
const BLIND: usize = 10_000;
const RUNS: usize = 30;

const SIGNAL_LEVELS: [f64; 7] = [
    0.20,
    0.15,
    0.10,
    0.075,
    0.05,
    0.025,
    0.00,
];

#[derive(Default, Clone)]
struct RunResult {
    top1: f64,
    top2: f64,
    top4: f64,
    lift: f64,
}

fn noisy_target<R: rand::RngExt>(
    rng: &mut R,
    true_target: u8,
    signal: f64,
) -> u8 {
    if rng.random::<f64>() < signal {
        true_target
    } else {
        rng.random_range(0u8..8u8)
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let m = mean(values);

    let variance =
        values.iter()
            .map(|v| {
                let d = v - m;
                d * d
            })
            .sum::<f64>()
            / (values.len() - 1) as f64;

    variance.sqrt()
}

fn ci95(values: &[f64]) -> (f64, f64) {
    let m = mean(values);
    let sd = sample_sd(values);
    let se = sd / (values.len() as f64).sqrt();

    // n=30 -> t(29), 97.5% ~= 2.045
    let margin = 2.045 * se;

    (m - margin, m + margin)
}

fn run_once(signal: f64) -> RunResult {
    let mut rng = rand::rng();
    let mut model = TinyPredictor::new();

    //
    // Fresh training set.
    //
    for _ in 0..TRAINING {
        let context = SyntheticContext {
            x1: rng.random(),
            x2: rng.random(),
            x3: rng.random(),
        };

        let hidden = context.hidden_target();

        let observed =
            noisy_target(&mut rng, hidden, signal);

        model.train(
            context.x1,
            context.x2,
            context.x3,
            observed,
        );
    }

    //
    // Completely unseen blind set.
    //
    let mut top1 = 0u64;
    let mut top2 = 0u64;
    let mut top4 = 0u64;

    for _ in 0..BLIND {
        let context = SyntheticContext {
            x1: rng.random(),
            x2: rng.random(),
            x3: rng.random(),
        };

        let hidden = context.hidden_target();

        let actual =
            noisy_target(&mut rng, hidden, signal);

        let scores =
            model.scores(
                context.x1,
                context.x2,
                context.x3,
            );

        let mut ranking: Vec<usize> =
            (0..8).collect();

        ranking.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let target = actual as usize;

        if ranking[0] == target {
            top1 += 1;
        }

        if ranking[..2].contains(&target) {
            top2 += 1;
        }

        if ranking[..4].contains(&target) {
            top4 += 1;
        }
    }

    let p1 = top1 as f64 / BLIND as f64;
    let p2 = top2 as f64 / BLIND as f64;
    let p4 = top4 as f64 / BLIND as f64;

    RunResult {
        top1: p1,
        top2: p2,
        top4: p4,
        lift: p1 / 0.125,
    }
}

fn main() {
    println!("X1331 Runtime v0.11");
    println!("===================");
    println!("Signal Detection Threshold");
    println!();

    println!("Independent models per signal : {}", RUNS);
    println!("Training samples per model    : {}", TRAINING);
    println!("Blind samples per model       : {}", BLIND);
    println!("Model parameters              : 32");
    println!();

    println!(
        "{:>7} {:>10} {:>10} {:>10} {:>10} {:>23}",
        "Signal",
        "Top1",
        "Top2",
        "Top4",
        "Lift",
        "95% CI Lift"
    );

    println!(
        "{:-<7} {:-<10} {:-<10} {:-<10} {:-<10} {:-<23}",
        "", "", "", "", "", ""
    );

    for signal in SIGNAL_LEVELS {
        let mut results = Vec::with_capacity(RUNS);

        for _ in 0..RUNS {
            results.push(run_once(signal));
        }

        let top1_values: Vec<f64> =
            results.iter().map(|r| r.top1).collect();

        let top2_values: Vec<f64> =
            results.iter().map(|r| r.top2).collect();

        let top4_values: Vec<f64> =
            results.iter().map(|r| r.top4).collect();

        let lift_values: Vec<f64> =
            results.iter().map(|r| r.lift).collect();

        let avg_top1 = mean(&top1_values);
        let avg_top2 = mean(&top2_values);
        let avg_top4 = mean(&top4_values);
        let avg_lift = mean(&lift_values);

        let (ci_low, ci_high) =
            ci95(&lift_values);

        println!(
            "{:>6.1}% {:>9.2}% {:>9.2}% {:>9.2}% {:>9.3}x [{:>7.3}, {:>7.3}]",
            signal * 100.0,
            avg_top1 * 100.0,
            avg_top2 * 100.0,
            avg_top4 * 100.0,
            avg_lift,
            ci_low,
            ci_high
        );
    }

    println!();
    println!("NULL / RANDOM");
    println!("=============");
    println!("Top1 = 12.50%");
    println!("Top2 = 25.00%");
    println!("Top4 = 50.00%");
    println!("Lift = 1.000x");
    println!();

    println!("Interpretation:");
    println!("  CI entirely > 1.0  -> reproducible positive signal");
    println!("  CI includes 1.0    -> cannot distinguish from random");
    println!("  CI entirely < 1.0  -> reproducible negative effect");
}
