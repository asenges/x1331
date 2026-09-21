mod core;

use rand::RngExt;

use crate::core::predictor::TinyPredictor;
use crate::core::synthetic::SyntheticContext;

const TRAINING: usize = 20_000;
const BLIND: usize = 10_000;

const K_VALUES: [usize; 5] = [
    1, 2, 3, 4, 6
];

#[derive(Default, Clone)]
struct Stats {
    target_survived: u64,
    quality_sum: f64,
    random_quality_sum: f64,
}

fn main() {
    println!("X1331 Runtime v0.9");
    println!("==================");
    println!("Hidden Signal Laboratory\n");

    let mut rng = rand::rng();

    let mut model =
        TinyPredictor::new();

    //
    // TRAIN
    //
    println!(
        "Training tiny predictor on {} observations...",
        TRAINING
    );

    for _ in 0..TRAINING {
        let context = SyntheticContext {
            x1: rng.random(),
            x2: rng.random(),
            x3: rng.random(),
        };

        let target =
            context.hidden_target();

        model.train(
            context.x1,
            context.x2,
            context.x3,
            target,
        );
    }

    println!("Training complete.");
    println!("Model parameters: 32\n");

    //
    // BLIND TEST
    //
    println!(
        "Blind test: {} unseen contexts\n",
        BLIND
    );

    let mut stats =
        vec![Stats::default(); K_VALUES.len()];

    let mut top1_correct = 0u64;

    for _ in 0..BLIND {
        let context = SyntheticContext {
            x1: rng.random(),
            x2: rng.random(),
            x3: rng.random(),
        };

        let target =
            context.hidden_target();

        let rewards =
            context.rewards();

        let scores =
            model.scores(
                context.x1,
                context.x2,
                context.x3,
            );

        let prediction =
            model.predict(
                context.x1,
                context.x2,
                context.x3,
            );

        if prediction == target {
            top1_correct += 1;
        }

        let mut ranking: Vec<usize> =
            (0..8).collect();

        ranking.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap()
        });

        let total_quality =
            rewards.iter().sum::<f64>();

        for (index, k) in
            K_VALUES.iter().enumerate()
        {
            let kept =
                &ranking[..*k];

            if kept.contains(&(target as usize)) {
                stats[index].target_survived += 1;
            }

            let selected_quality =
                kept
                    .iter()
                    .map(|i| rewards[*i])
                    .sum::<f64>();

            stats[index].quality_sum +=
                selected_quality;

            stats[index].random_quality_sum +=
                total_quality
                * (*k as f64 / 8.0);
        }
    }

    println!(
        "Top-1 prediction accuracy: {:.2}%",
        top1_correct as f64
            / BLIND as f64
            * 100.0
    );

    println!();
    println!("N -> K BLIND RESULTS");
    println!("====================");

    println!(
        "{:>3} {:>11} {:>12} {:>12} {:>14}",
        "K",
        "Reduction",
        "TargetRecall",
        "RandomRecall",
        "Concentr.Gain"
    );

    for (index, k) in
        K_VALUES.iter().enumerate()
    {
        let s = &stats[index];

        let reduction =
            1.0 - (*k as f64 / 8.0);

        let recall =
            s.target_survived as f64
            / BLIND as f64;

        let random_recall =
            *k as f64 / 8.0;

        let concentration_gain =
            s.quality_sum
            / s.random_quality_sum;

        println!(
            "{:>3} {:>10.2}% {:>11.2}% {:>11.2}% {:>14.6}",
            k,
            reduction * 100.0,
            recall * 100.0,
            random_recall * 100.0,
            concentration_gain
        );
    }
}
