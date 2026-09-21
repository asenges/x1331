mod core;

use rand::RngExt;

use crate::core::predictor::TinyPredictor;
use crate::core::synthetic::SyntheticContext;

const TRAINING: usize = 20_000;
const BLIND: usize = 10_000;

const SIGNAL_LEVELS: [f64; 7] = [
    1.00,
    0.75,
    0.50,
    0.25,
    0.10,
    0.05,
    0.00,
];

#[derive(Default)]
struct ResultStats {
    correct: u64,
    top2: u64,
    top4: u64,
    confidence_sum: f64,
}

fn noisy_target<R: rand::RngExt>(
    rng: &mut R,
    true_target: u8,
    signal: f64,
) -> u8 {
    //
    // signal = 1.0:
    // always return the true hidden target.
    //
    // signal = 0.0:
    // target is completely random.
    //
    // Intermediate values:
    // retain the real relationship with probability SIGNAL,
    // otherwise replace it with a random state.
    //
    if rng.random::<f64>() < signal {
        true_target
    } else {
        rng.random_range(0u8..8u8)
    }
}

fn softmax_confidence(scores: &[f64; 8]) -> f64 {
    let max_score = scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut denominator = 0.0;

    for score in scores {
        denominator +=
            (*score - max_score).exp();
    }

    1.0 / denominator
}

fn main() {
    println!("X1331 Runtime v0.10");
    println!("==================");
    println!("Weak Signal Laboratory");
    println!();

    println!(
        "Training observations per signal : {}",
        TRAINING
    );

    println!(
        "Blind observations per signal    : {}",
        BLIND
    );

    println!();

    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "Signal",
        "Top1",
        "Top2",
        "Top4",
        "Confidence",
        "Lift"
    );

    println!(
        "{:-<8} {:-<10} {:-<10} {:-<10} {:-<12} {:-<12}",
        "", "", "", "", "", ""
    );

    for signal in SIGNAL_LEVELS {
        //
        // Completely new model.
        //
        let mut model =
            TinyPredictor::new();

        let mut rng =
            rand::rng();

        //
        // TRAINING
        //
        for _ in 0..TRAINING {
            let context = SyntheticContext {
                x1: rng.random(),
                x2: rng.random(),
                x3: rng.random(),
            };

            let hidden =
                context.hidden_target();

            let observed =
                noisy_target(
                    &mut rng,
                    hidden,
                    signal,
                );

            model.train(
                context.x1,
                context.x2,
                context.x3,
                observed,
            );
        }

        //
        // BLIND
        //
        let mut stats =
            ResultStats::default();

        for _ in 0..BLIND {
            let context = SyntheticContext {
                x1: rng.random(),
                x2: rng.random(),
                x3: rng.random(),
            };

            let hidden =
                context.hidden_target();

            //
            // IMPORTANT:
            //
            // Blind reality receives the SAME signal
            // degradation as training.
            //
            let actual =
                noisy_target(
                    &mut rng,
                    hidden,
                    signal,
                );

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
                    .unwrap()
            });

            if ranking[0] == actual as usize {
                stats.correct += 1;
            }

            if ranking[..2]
                .contains(&(actual as usize))
            {
                stats.top2 += 1;
            }

            if ranking[..4]
                .contains(&(actual as usize))
            {
                stats.top4 += 1;
            }

            stats.confidence_sum +=
                softmax_confidence(&scores);
        }

        let top1 =
            stats.correct as f64
            / BLIND as f64;

        let top2 =
            stats.top2 as f64
            / BLIND as f64;

        let top4 =
            stats.top4 as f64
            / BLIND as f64;

        let confidence =
            stats.confidence_sum
            / BLIND as f64;

        //
        // Random Top1 baseline = 1/8 = 12.5%
        //
        let lift =
            top1 / 0.125;

        println!(
            "{:>7.0}% {:>9.2}% {:>9.2}% {:>9.2}% {:>11.2}% {:>11.3}x",
            signal * 100.0,
            top1 * 100.0,
            top2 * 100.0,
            top4 * 100.0,
            confidence * 100.0,
            lift
        );
    }

    println!();
    println!("Random baselines");
    println!("================");
    println!("Top1 = 12.50%");
    println!("Top2 = 25.00%");
    println!("Top4 = 50.00%");
    println!("Lift = 1.000x");
}
