mod core;

use rand::RngExt;
use std::time::Instant;

use crate::core::cell::{
    PossibilityRegion,
    X1331Cell,
};

use crate::core::sha_predictor::ShaPredictor;
use crate::core::verifier::Sha256dVerifier;

const TRAINING: usize = 1_000;
const BLIND: usize = 1_000;
const RUNS: usize = 10;

const CLASSES: usize = 8;
const FEATURES: usize = 9;

const SAMPLES_PER_REGION: u64 = 100;

#[derive(Default)]
struct RunResult {
    top1: f64,
    top2: f64,
    top4: f64,
}

fn mean(
    values: &[f64],
) -> f64 {
    values.iter().sum::<f64>()
        / values.len() as f64
}

fn sample_sd(
    values: &[f64],
) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let m =
        mean(values);

    (
        values
            .iter()
            .map(|x| {
                (x - m).powi(2)
            })
            .sum::<f64>()
            / (values.len() - 1) as f64
    )
    .sqrt()
}

fn ci95(
    values: &[f64],
) -> (f64, f64) {
    let m =
        mean(values);

    let se =
        sample_sd(values)
        / (values.len() as f64).sqrt();

    // Student t:
    // df = 9
    // 97.5 percentile ~= 2.262
    let margin =
        2.262 * se;

    (
        m - margin,
        m + margin,
    )
}

fn evaluate_regions(
    header: &[u8; 32],
    verifier: &mut Sha256dVerifier,
) -> [f64; 8] {
    let root =
        PossibilityRegion::root(24);

    let cell =
        X1331Cell::from_region(root);

    let mut rewards =
        [0.0; 8];

    let mut rng =
        rand::rng();

    for state in &cell.states {
        let mut sum =
            0u64;

        for _ in 0..SAMPLES_PER_REGION {
            let nonce =
                rng.random_range(
                    state.region.start()
                        ..=state.region.end()
                );

            let result =
                verifier.evaluate(
                    header,
                    nonce,
                );

            sum +=
                result.leading_zero_bits
                    as u64;
        }

        rewards[
            state.value as usize
        ] =
            sum as f64
            / SAMPLES_PER_REGION as f64;
    }

    rewards
}

fn best_state(
    rewards: &[f64; 8],
) -> u8 {
    rewards
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1
                .partial_cmp(b.1)
                .unwrap_or(
                    std::cmp::Ordering::Equal
                )
        })
        .map(|(index, _)| {
            index as u8
        })
        .unwrap()
}

fn run_once(
    verifier: &mut Sha256dVerifier,
) -> RunResult {
    let mut rng =
        rand::rng();

    let mut model =
        ShaPredictor::new();

    //
    // TRAINING
    //
    for _ in 0..TRAINING {
        let header: [u8; 32] =
            rng.random();

        let rewards =
            evaluate_regions(
                &header,
                verifier,
            );

        let target =
            best_state(&rewards);

        model.train(
            &header,
            target,
        );
    }

    //
    // BLIND TEST
    //
    let mut top1 =
        0u64;

    let mut top2 =
        0u64;

    let mut top4 =
        0u64;

    for _ in 0..BLIND {
        let header: [u8; 32] =
            rng.random();

        //
        // IMPORTANT:
        //
        // Prediction happens BEFORE
        // ground truth is evaluated.
        //
        let scores =
            model.scores(&header);

        let mut ranking:
            Vec<usize> =
            (0..8).collect();

        ranking.sort_by(
            |a, b| {
                scores[*b]
                    .partial_cmp(
                        &scores[*a]
                    )
                    .unwrap_or(
                        std::cmp::Ordering::Equal
                    )
            },
        );

        //
        // Reveal SHA256d ground truth
        // only AFTER prediction.
        //
        let rewards =
            evaluate_regions(
                &header,
                verifier,
            );

        let target =
            best_state(&rewards)
                as usize;

        if ranking[0] == target {
            top1 += 1;
        }

        if ranking[..2]
            .contains(&target)
        {
            top2 += 1;
        }

        if ranking[..4]
            .contains(&target)
        {
            top4 += 1;
        }
    }

    RunResult {
        top1:
            top1 as f64
            / BLIND as f64,

        top2:
            top2 as f64
            / BLIND as f64,

        top4:
            top4 as f64
            / BLIND as f64,
    }
}

fn main() {
    let runtime_start =
        Instant::now();

    println!(
        "X1331 Runtime v0.12"
    );

    println!(
        "==================="
    );

    println!(
        "SHA256d Predictability Laboratory"
    );

    println!();

    let model_parameters =
        CLASSES * FEATURES;

    let expected_headers =
        RUNS
        * (TRAINING + BLIND);

    let expected_regions =
        expected_headers
        * CLASSES;

    let expected_hashes =
        expected_regions
        * SAMPLES_PER_REGION
            as usize;

    println!(
        "Predictor parameters : {}",
        model_parameters
    );

    println!(
        "Runs                 : {}",
        RUNS
    );

    println!(
        "Training/run         : {}",
        TRAINING
    );

    println!(
        "Blind/run            : {}",
        BLIND
    );

    println!(
        "Hashes/region        : {}",
        SAMPLES_PER_REGION
    );

    println!();

    println!(
        "COMPUTATIONAL COMPLEXITY"
    );

    println!(
        "========================"
    );

    println!(
        "Headers               : {}",
        expected_headers
    );

    println!(
        "Regions/header        : {}",
        CLASSES
    );

    println!(
        "Total regions         : {}",
        expected_regions
    );

    println!(
        "Hashes/region         : {}",
        SAMPLES_PER_REGION
    );

    println!(
        "Expected SHA256d      : {}",
        expected_hashes
    );

    println!(
        "Prediction/header     : O(C x F) = O({} x {})",
        CLASSES,
        FEATURES
    );

    println!(
        "Ground truth/header   : O(C x S) = O({} x {})",
        CLASSES,
        SAMPLES_PER_REGION
    );

    println!();

    let mut verifier =
        Sha256dVerifier::new();

    let mut results =
        Vec::with_capacity(
            RUNS
        );

    for run in 0..RUNS {
        let run_start =
            Instant::now();

        let result =
            run_once(
                &mut verifier
            );

        let run_elapsed =
            run_start
                .elapsed()
                .as_secs_f64();

        println!(
            "run {:>2}/{} | Top1 {:>6.2}% | Top2 {:>6.2}% | Top4 {:>6.2}% | {:>7.3}s",
            run + 1,
            RUNS,
            result.top1 * 100.0,
            result.top2 * 100.0,
            result.top4 * 100.0,
            run_elapsed,
        );

        results.push(
            result
        );
    }

    let top1:
        Vec<f64> =
        results
            .iter()
            .map(|r| r.top1)
            .collect();

    let top2:
        Vec<f64> =
        results
            .iter()
            .map(|r| r.top2)
            .collect();

    let top4:
        Vec<f64> =
        results
            .iter()
            .map(|r| r.top4)
            .collect();

    let lift1:
        Vec<f64> =
        top1
            .iter()
            .map(|p| {
                p / 0.125
            })
            .collect();

    let lift2:
        Vec<f64> =
        top2
            .iter()
            .map(|p| {
                p / 0.25
            })
            .collect();

    let lift4:
        Vec<f64> =
        top4
            .iter()
            .map(|p| {
                p / 0.50
            })
            .collect();

    let (l1_lo, l1_hi) =
        ci95(&lift1);

    let (l2_lo, l2_hi) =
        ci95(&lift2);

    let (l4_lo, l4_hi) =
        ci95(&lift4);

    println!();

    println!(
        "BLIND RESULTS"
    );

    println!(
        "============="
    );

    println!(
        "Top1 : {:>6.2}% | Lift {:.4}x | 95% CI [{:.4}, {:.4}]",
        mean(&top1) * 100.0,
        mean(&lift1),
        l1_lo,
        l1_hi
    );

    println!(
        "Top2 : {:>6.2}% | Lift {:.4}x | 95% CI [{:.4}, {:.4}]",
        mean(&top2) * 100.0,
        mean(&lift2),
        l2_lo,
        l2_hi
    );

    println!(
        "Top4 : {:>6.2}% | Lift {:.4}x | 95% CI [{:.4}, {:.4}]",
        mean(&top4) * 100.0,
        mean(&lift4),
        l4_lo,
        l4_hi
    );

    println!();

    println!(
        "NULL EXPECTATION"
    );

    println!(
        "================"
    );

    println!(
        "Top1 = 12.50% | Lift 1.0000x"
    );

    println!(
        "Top2 = 25.00% | Lift 1.0000x"
    );

    println!(
        "Top4 = 50.00% | Lift 1.0000x"
    );

    //
    // PERFORMANCE REPORT
    //
    let elapsed =
        runtime_start.elapsed();

    let elapsed_seconds =
        elapsed.as_secs_f64();

    let evaluations =
        verifier.evaluations();

    let hashes_per_second =
        if elapsed_seconds > 0.0 {
            evaluations as f64
                / elapsed_seconds
        } else {
            0.0
        };

    let headers_per_second =
        if elapsed_seconds > 0.0 {
            expected_headers as f64
                / elapsed_seconds
        } else {
            0.0
        };

    //
    // Approximate scoring operations.
    //
    // Each prediction scores:
    //
    // C classes * F features
    //
    // Training also performs updates,
    // so this is deliberately reported
    // as predictor scoring work rather
    // than exact CPU instructions.
    //
    let prediction_score_ops =
        expected_headers as u64
        * CLASSES as u64
        * FEATURES as u64;

    let prediction_hash_ratio =
        if evaluations > 0 {
            prediction_score_ops as f64
                / evaluations as f64
        } else {
            0.0
        };

    println!();

    println!(
        "COMPLEXITY / PERFORMANCE"
    );

    println!(
        "========================"
    );

    println!(
        "Elapsed time           : {:.3} s",
        elapsed_seconds
    );

    let total_secs =
        elapsed.as_secs();

    let hours =
        total_secs / 3600;

    let minutes =
        (total_secs % 3600)
        / 60;

    let seconds =
        elapsed_seconds
        - (hours * 3600) as f64
        - (minutes * 60) as f64;

    println!(
        "Elapsed                : {:02}:{:02}:{:06.3}",
        hours,
        minutes,
        seconds
    );

    println!(
        "SHA256d evaluations    : {}",
        evaluations
    );

    println!(
        "SHA256d / second       : {:.2}",
        hashes_per_second
    );

    println!(
        "Headers / second       : {:.2}",
        headers_per_second
    );

    println!(
        "Predictor parameters   : {}",
        model_parameters
    );

    println!(
        "Prediction score ops   : {}",
        prediction_score_ops
    );

    println!(
        "Prediction/hash ratio  : {:.6}",
        prediction_hash_ratio
    );

    println!();

    println!(
        "ASYMPTOTIC MODEL"
    );

    println!(
        "================"
    );

    println!(
        "Prediction/header      : O(C x F)"
    );

    println!(
        "Training update/header : O(C x F)"
    );

    println!(
        "Ground truth/header    : O(C x S)"
    );

    println!(
        "Total experiment       : O(R x (T+B) x C x S)"
    );

    println!();

    println!(
        "R={} T={} B={} C={} F={} S={}",
        RUNS,
        TRAINING,
        BLIND,
        CLASSES,
        FEATURES,
        SAMPLES_PER_REGION
    );

    println!();

    println!(
        "Cost model:"
    );

    println!(
        "C_total = C_prediction + C_learning + C_verification"
    );
}
