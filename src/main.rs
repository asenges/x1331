mod core;

use rand::RngExt;
use std::time::{Duration, Instant};

use crate::core::cell::{
    PossibilityRegion,
    X1331Cell,
};

use crate::core::sha_predictor::ShaPredictor;
use crate::core::verifier::Sha256dVerifier;

// ------------------------------------------------------------
// X1331 v0.14
// Recursive Collapse Laboratory
// ------------------------------------------------------------

const TRAINING: usize = 1_000;
const BLIND: usize = 5_000;
const RUNS: usize = 10;

const TOTAL_BITS: u8 = 24;
const CLASSES: usize = 8;
const FEATURES: usize = 9;

const SAMPLES_PER_REGION: u64 = 100;

// Recursive experiment:
// level 1 = 3 prefix bits
// level 2 = 6 prefix bits
// level 3 = 9 prefix bits
const MAX_LEVELS: usize = 3;

// Keep K children at every collapse.
const KEEP_K: usize = 2;

#[derive(Default, Clone)]
struct Timing {
    prediction: Duration,
    learning: Duration,
    verification: Duration,
}

impl Timing {
    fn total(&self) -> Duration {
        self.prediction
            + self.learning
            + self.verification
    }
}

#[derive(Default, Clone)]
struct LevelStats {
    observations: u64,

    best_survived: u64,

    selected_reward_sum: f64,
    baseline_reward_sum: f64,

    kept_regions: u64,
    possible_regions: u64,
}

impl LevelStats {
    fn best_recall(&self) -> f64 {
        if self.observations == 0 {
            0.0
        } else {
            self.best_survived as f64
                / self.observations as f64
        }
    }

    fn concentration_gain(&self) -> f64 {
        if self.baseline_reward_sum == 0.0 {
            0.0
        } else {
            self.selected_reward_sum
                / self.baseline_reward_sum
        }
    }

    fn reduction(&self) -> f64 {
        if self.possible_regions == 0 {
            0.0
        } else {
            1.0
                - self.kept_regions as f64
                    / self.possible_regions as f64
        }
    }
}

#[derive(Clone)]
struct CandidateRegion {
    region: PossibilityRegion,
    score: f64,
}

fn evaluate_region(
    header: &[u8; 32],
    region: &PossibilityRegion,
    verifier: &mut Sha256dVerifier,
    timing: &mut Timing,
) -> f64 {
    let mut rng = rand::rng();

    let start = Instant::now();

    let mut sum = 0u64;

    for _ in 0..SAMPLES_PER_REGION {
        let nonce =
            rng.random_range(
                region.start()
                    ..=region.end()
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

    timing.verification +=
        start.elapsed();

    sum as f64
        / SAMPLES_PER_REGION as f64
}

fn evaluate_children(
    header: &[u8; 32],
    parent: PossibilityRegion,
    verifier: &mut Sha256dVerifier,
    timing: &mut Timing,
) -> [(PossibilityRegion, f64); 8] {
    let cell =
        X1331Cell::from_region(parent);

    std::array::from_fn(
        |index| {
            let region =
                cell.states[index]
                    .region
                    .clone();

            let reward =
                evaluate_region(
                    header,
                    &region,
                    verifier,
                    timing,
                );

            (
                region,
                reward,
            )
        },
    )
}

fn rank_states(
    model: &ShaPredictor,
    header: &[u8; 32],
    timing: &mut Timing,
) -> Vec<(usize, f64)> {
    let start =
        Instant::now();

    let scores =
        model.scores(header);

    let mut ranking:
        Vec<(usize, f64)> =
        scores
            .iter()
            .enumerate()
            .map(
                |(state, score)| {
                    (
                        state,
                        *score,
                    )
                },
            )
            .collect();

    ranking.sort_by(
        |a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(
                    std::cmp::Ordering::Equal
                )
        },
    );

    timing.prediction +=
        start.elapsed();

    ranking
}

fn best_actual_state(
    rewards: &[(PossibilityRegion, f64); 8],
) -> usize {
    rewards
        .iter()
        .enumerate()
        .max_by(
            |a, b| {
                a.1.1
                    .partial_cmp(&b.1.1)
                    .unwrap_or(
                        std::cmp::Ordering::Equal
                    )
            },
        )
        .map(
            |(index, _)| index
        )
        .unwrap()
}

fn mean_reward(
    rewards: &[(PossibilityRegion, f64); 8],
) -> f64 {
    rewards
        .iter()
        .map(|(_, reward)| {
            *reward
        })
        .sum::<f64>()
        / 8.0
}

fn train_model(
    model: &mut ShaPredictor,
    verifier: &mut Sha256dVerifier,
    timing: &mut Timing,
) {
    let mut rng =
        rand::rng();

    for _ in 0..TRAINING {
        let header: [u8; 32] =
            rng.random();

        let root =
            PossibilityRegion::root(
                TOTAL_BITS
            );

        let rewards =
            evaluate_children(
                &header,
                root,
                verifier,
                timing,
            );

        let target =
            best_actual_state(
                &rewards
            ) as u8;

        let start =
            Instant::now();

        model.train(
            &header,
            target,
        );

        timing.learning +=
            start.elapsed();
    }
}

fn recursive_blind_test(
    model: &ShaPredictor,
    verifier: &mut Sha256dVerifier,
    timing: &mut Timing,
    level_stats: &mut [LevelStats],
) {
    let mut rng =
        rand::rng();

    for _ in 0..BLIND {
        let header: [u8; 32] =
            rng.random();

        let mut active =
            vec![
                CandidateRegion {
                    region:
                        PossibilityRegion::root(
                            TOTAL_BITS
                        ),
                    score: 0.0,
                }
            ];

        for level in 0..MAX_LEVELS {
            let mut next_active:
                Vec<CandidateRegion> =
                Vec::new();

            for candidate in &active {
                if !candidate.region.can_expand() {
                    continue;
                }

                //
                // Prediction FIRST.
                //
                let ranking =
                    rank_states(
                        model,
                        &header,
                        timing,
                    );

                //
                // Ground truth AFTER prediction.
                //
                let rewards =
                    evaluate_children(
                        &header,
                        candidate.region.clone(),
                        verifier,
                        timing,
                    );

                let best_actual =
                    best_actual_state(
                        &rewards
                    );

                let baseline =
                    mean_reward(
                        &rewards
                    );

                let selected_states:
                    Vec<usize> =
                    ranking
                        .iter()
                        .take(KEEP_K)
                        .map(
                            |(state, _)| {
                                *state
                            },
                        )
                        .collect();

                let survived =
                    selected_states
                        .contains(
                            &best_actual
                        );

                let selected_reward =
                    selected_states
                        .iter()
                        .map(
                            |state| {
                                rewards[*state].1
                            },
                        )
                        .sum::<f64>()
                        / KEEP_K as f64;

                level_stats[level]
                    .observations += 1;

                if survived {
                    level_stats[level]
                        .best_survived += 1;
                }

                level_stats[level]
                    .selected_reward_sum +=
                    selected_reward;

                level_stats[level]
                    .baseline_reward_sum +=
                    baseline;

                level_stats[level]
                    .kept_regions +=
                    KEEP_K as u64;

                level_stats[level]
                    .possible_regions +=
                    CLASSES as u64;

                for state in selected_states {
                    next_active.push(
                        CandidateRegion {
                            region:
                                rewards[state]
                                    .0
                                    .clone(),

                            score:
                                ranking
                                    .iter()
                                    .find(
                                        |(s, _)| {
                                            *s
                                                == state
                                        },
                                    )
                                    .map(
                                        |(_, score)| {
                                            *score
                                        },
                                    )
                                    .unwrap_or(0.0),
                        },
                    );
                }
            }

            active =
                next_active;

            if active.is_empty() {
                break;
            }
        }
    }
}

fn main() {
    let total_start =
        Instant::now();

    println!(
        "X1331 Runtime v0.14"
    );

    println!(
        "==================="
    );

    println!(
        "Recursive Collapse Laboratory"
    );

    println!();

    println!(
        "Total possibility bits : {}",
        TOTAL_BITS
    );

    println!(
        "Initial possibilities  : {}",
        1u64 << TOTAL_BITS
    );

    println!(
        "Branches/cell          : {}",
        CLASSES
    );

    println!(
        "Keep K/collapse        : {}",
        KEEP_K
    );

    println!(
        "Recursive levels       : {}",
        MAX_LEVELS
    );

    println!(
        "Training/run           : {}",
        TRAINING
    );

    println!(
        "Blind/run              : {}",
        BLIND
    );

    println!(
        "Runs                   : {}",
        RUNS
    );

    println!(
        "Hashes/region          : {}",
        SAMPLES_PER_REGION
    );

    println!(
        "Predictor parameters   : {}",
        CLASSES * FEATURES
    );

    println!();

    println!(
        "COLLAPSE GEOMETRY"
    );

    println!(
        "================="
    );

    let mut theoretical_active =
        1usize;

    let mut surviving_fraction =
        1.0f64;

    for level in 1..=MAX_LEVELS {
        let candidates =
            theoretical_active
            * CLASSES;

        theoretical_active *=
            KEEP_K;

        surviving_fraction *=
            KEEP_K as f64
            / CLASSES as f64;

        println!(
            "Level {} : inspect {:>3} child regions -> keep {:>3} | cumulative survival {:>8.4}% | cumulative reduction {:>8.4}%",
            level,
            candidates,
            theoretical_active,
            surviving_fraction * 100.0,
            (1.0 - surviving_fraction)
                * 100.0,
        );
    }

    println!();

    let mut verifier =
        Sha256dVerifier::new();

    let mut timing =
        Timing::default();

    let mut global_stats =
        vec![
            LevelStats::default();
            MAX_LEVELS
        ];

    for run in 0..RUNS {
        let run_start =
            Instant::now();

        let before_eval =
            verifier.evaluations();

        let mut model =
            ShaPredictor::new();

        train_model(
            &mut model,
            &mut verifier,
            &mut timing,
        );

        let mut run_stats =
            vec![
                LevelStats::default();
                MAX_LEVELS
            ];

        recursive_blind_test(
            &model,
            &mut verifier,
            &mut timing,
            &mut run_stats,
        );

        for level in 0..MAX_LEVELS {
            global_stats[level]
                .observations +=
                run_stats[level]
                    .observations;

            global_stats[level]
                .best_survived +=
                run_stats[level]
                    .best_survived;

            global_stats[level]
                .selected_reward_sum +=
                run_stats[level]
                    .selected_reward_sum;

            global_stats[level]
                .baseline_reward_sum +=
                run_stats[level]
                    .baseline_reward_sum;

            global_stats[level]
                .kept_regions +=
                run_stats[level]
                    .kept_regions;

            global_stats[level]
                .possible_regions +=
                run_stats[level]
                    .possible_regions;
        }

        let run_hashes =
            verifier.evaluations()
            - before_eval;

        println!(
            "run {:>2}/{} | {:>10} SHA256d | {:>7.3}s",
            run + 1,
            RUNS,
            run_hashes,
            run_start
                .elapsed()
                .as_secs_f64(),
        );
    }

    println!();

    println!(
        "RECURSIVE BLIND RESULTS"
    );

    println!(
        "======================="
    );

    println!(
        "{:>5} {:>12} {:>12} {:>15} {:>15}",
        "Level",
        "Reduction",
        "BestRecall",
        "Concentr.Gain",
        "RandomRecall"
    );

    println!(
        "{:-<5} {:-<12} {:-<12} {:-<15} {:-<15}",
        "", "", "", "", ""
    );

    for level in 0..MAX_LEVELS {
        let stats =
            &global_stats[level];

        let random_recall =
            KEEP_K as f64
            / CLASSES as f64;

        println!(
            "{:>5} {:>11.2}% {:>11.2}% {:>14.6} {:>14.2}%",
            level + 1,
            stats.reduction()
                * 100.0,
            stats.best_recall()
                * 100.0,
            stats.concentration_gain(),
            random_recall
                * 100.0,
        );
    }

    println!();

    println!(
        "NULL EXPECTATION"
    );

    println!(
        "================"
    );

    println!(
        "For K={} of N={}:",
        KEEP_K,
        CLASSES
    );

    println!(
        "Best-region survival per collapse = {:.2}%",
        KEEP_K as f64
            / CLASSES as f64
            * 100.0
    );

    println!(
        "ConcentrationGain = 1.000000"
    );

    println!();

    let elapsed =
        total_start.elapsed();

    let evaluations =
        verifier.evaluations();

    let total_seconds =
        elapsed.as_secs_f64();

    let prediction_seconds =
        timing.prediction
            .as_secs_f64();

    let learning_seconds =
        timing.learning
            .as_secs_f64();

    let verification_seconds =
        timing.verification
            .as_secs_f64();

    let measured_phase_seconds =
        timing.total()
            .as_secs_f64();

    println!(
        "TIME BREAKDOWN"
    );

    println!(
        "=============="
    );

    println!(
        "Prediction    : {:>10.6} s | {:>7.3}%",
        prediction_seconds,
        if total_seconds > 0.0 {
            prediction_seconds
                / total_seconds
                * 100.0
        } else {
            0.0
        }
    );

    println!(
        "Learning      : {:>10.6} s | {:>7.3}%",
        learning_seconds,
        if total_seconds > 0.0 {
            learning_seconds
                / total_seconds
                * 100.0
        } else {
            0.0
        }
    );

    println!(
        "Verification  : {:>10.6} s | {:>7.3}%",
        verification_seconds,
        if total_seconds > 0.0 {
            verification_seconds
                / total_seconds
                * 100.0
        } else {
            0.0
        }
    );

    println!(
        "Measured phase: {:>10.6} s",
        measured_phase_seconds
    );

    println!(
        "Total elapsed : {:>10.6} s",
        total_seconds
    );

    println!();

    println!(
        "PERFORMANCE"
    );

    println!(
        "==========="
    );

    println!(
        "SHA256d evaluations : {}",
        evaluations
    );

    println!(
        "SHA256d / second    : {:.2}",
        if total_seconds > 0.0 {
            evaluations as f64
                / total_seconds
        } else {
            0.0
        }
    );

    println!(
        "Predictor parameters: {}",
        CLASSES * FEATURES
    );

    println!();

    println!(
        "COMPLEXITY MODEL"
    );

    println!(
        "================"
    );

    println!(
        "Prediction/cell       : O(C x F)"
    );

    println!(
        "Verification/cell     : O(C x S)"
    );

    println!(
        "Active cells/level    : K^(L-1)"
    );

    println!(
        "Blind verification    : O(B x C x S x sum(K^(L-1)))"
    );

    println!(
        "Training verification : O(T x C x S)"
    );

    println!();

    println!(
        "C_total = C_prediction + C_learning + C_verification"
    );
}
