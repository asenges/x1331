mod core;

use rand::RngExt;
use std::collections::HashSet;

use crate::core::agent::{
    Agent,
    BinaryAgent,
    UniformAgent,
};
use crate::core::cell::{
    PossibilityRegion,
    X1331Cell,
};
use crate::core::memory::ObserverMemory;
use crate::core::observer::Observer;
use crate::core::verifier::Sha256dVerifier;

const K_VALUES: [usize; 6] = [1, 2, 3, 4, 6, 8];
const VALUABLE_COUNT: usize = 2;

#[derive(Clone, Default)]
struct KStats {
    experiments: u64,

    valuable_found: u64,
    valuable_total: u64,

    quality_sum: f64,
    random_expected_quality_sum: f64,

    best_survived: u64,
}

fn evaluate_region(
    region: PossibilityRegion,
    header: &[u8],
    samples: u64,
    verifier: &mut Sha256dVerifier,
) -> f64 {
    let mut rng = rand::rng();
    let mut reward_sum = 0u64;

    for _ in 0..samples {
        let nonce =
            rng.random_range(region.start()..=region.end());

        let result =
            verifier.evaluate(header, nonce);

        reward_sum +=
            result.leading_zero_bits as u64;
    }

    reward_sum as f64 / samples as f64
}

fn main() {
    println!("X1331 Runtime v0.8");
    println!("==================");
    println!("Predictive reduction experiment: N -> K\n");

    // Load learned method performance from v0.7.
    //
    // We freeze it during this experiment:
    // NO observer.learn().
    let memory =
        ObserverMemory::load("data/observer-v07.json");

    println!(
        "Frozen training experiments: {}",
        memory.experiments
    );

    let observer =
        Observer::new(memory);

    let agents: Vec<Box<dyn Agent>> = vec![
        Box::new(UniformAgent),
        Box::new(BinaryAgent),
    ];

    let experiments = 500u64;
    let samples_per_region = 500u64;

    let mut verifier =
        Sha256dVerifier::new();

    let mut rng = rand::rng();

    let mut stats =
        vec![KStats::default(); K_VALUES.len()];

    for experiment in 0..experiments {
        let header: [u8; 32] = rng.random();

        let root =
            PossibilityRegion::root(24);

        let cell =
            X1331Cell::from_region(root);

        let mut proposals = Vec::new();

        for agent in &agents {
            proposals.extend(
                agent.propose(&cell)
            );
        }

        // Prediction/ranking BEFORE revealing reality.
        let ranking =
            observer.rank(&cell, &proposals);

        // Now reveal all eight outcomes.
        let mut rewards = [0.0_f64; 8];

        for state in &cell.states {
            rewards[state.value as usize] =
                evaluate_region(
                    state.region,
                    &header,
                    samples_per_region,
                    &mut verifier,
                );
        }

        // Ground truth ranking.
        let mut actual_order: Vec<usize> =
            (0..8).collect();

        actual_order.sort_by(|a, b| {
            rewards[*b]
                .partial_cmp(&rewards[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let valuable: HashSet<u8> =
            actual_order
                .iter()
                .take(VALUABLE_COUNT)
                .map(|i| *i as u8)
                .collect();

        let actual_best =
            actual_order[0] as u8;

        let total_reward =
            rewards.iter().sum::<f64>();

        for (stats_index, k) in
            K_VALUES.iter().enumerate()
        {
            let kept: Vec<u8> =
                ranking
                    .iter()
                    .take(*k)
                    .map(|(state, _)| *state)
                    .collect();

            let kept_set: HashSet<u8> =
                kept.iter().copied().collect();

            let valuable_found =
                valuable
                    .intersection(&kept_set)
                    .count() as u64;

            let selected_quality =
                kept
                    .iter()
                    .map(|state| {
                        rewards[*state as usize]
                    })
                    .sum::<f64>();

            // Expected reward mass of a random subset
            // of size K.
            let random_expected_quality =
                total_reward * (*k as f64 / 8.0);

            let s = &mut stats[stats_index];

            s.experiments += 1;
            s.valuable_found += valuable_found;
            s.valuable_total +=
                VALUABLE_COUNT as u64;

            s.quality_sum += selected_quality;

            s.random_expected_quality_sum +=
                random_expected_quality;

            if kept_set.contains(&actual_best) {
                s.best_survived += 1;
            }
        }

        if (experiment + 1) % 100 == 0 {
            println!(
                "completed {:>4}/{} experiments",
                experiment + 1,
                experiments
            );
        }
    }

    println!();
    println!("N -> K RESULTS");
    println!("==============");

    println!(
        "{:>3} {:>11} {:>11} {:>14} {:>12}",
        "K",
        "Reduction",
        "Recall",
        "Concentr.Gain",
        "BestRecall"
    );

    for (index, k) in
        K_VALUES.iter().enumerate()
    {
        let s = &stats[index];

        let reduction =
            1.0 - (*k as f64 / 8.0);

        let recall =
            s.valuable_found as f64
            / s.valuable_total as f64;

        let quality_gain =
            if s.random_expected_quality_sum == 0.0 {
                1.0
            } else {
                s.quality_sum
                    / s.random_expected_quality_sum
            };

        let best_recall =
            s.best_survived as f64
            / s.experiments as f64;

        println!(
            "{:>3} {:>10.2}% {:>10.2}% {:>14.6} {:>11.2}%",
            k,
            reduction * 100.0,
            recall * 100.0,
            quality_gain,
            best_recall * 100.0
        );
    }

    println!();
    println!("Random-reference recall:");
    println!("------------------------");

    for k in K_VALUES {
        let expected_recall =
            k as f64 / 8.0;

        println!(
            "K={} -> {:.2}%",
            k,
            expected_recall * 100.0
        );
    }

    println!(
        "\nSHA256d evaluations: {}",
        verifier.evaluations()
    );
}
