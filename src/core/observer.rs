use std::collections::HashMap;

use crate::core::agent::Proposal;
use crate::core::cell::X1331Cell;
use crate::core::memory::ObserverMemory;

pub struct Observer {
    pub memory: ObserverMemory,
}

impl Observer {
    pub fn new(memory: ObserverMemory) -> Self {
        Self { memory }
    }

    pub fn rank(
        &self,
        cell: &X1331Cell,
        proposals: &[Proposal],
    ) -> Vec<(u8, f64)> {
        let mut scores = [0.0_f64; 8];

        for proposal in proposals {
            let key = proposal.method_key();
            let trust = self.memory.trust(&key);

            scores[proposal.state_value as usize] +=
                proposal.score * trust;
        }

        let mut ranking: Vec<(u8, f64)> =
            cell.states
                .iter()
                .map(|state| {
                    (
                        state.value,
                        scores[state.value as usize],
                    )
                })
                .collect();

        ranking.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ranking
    }

    pub fn top_k(
        &self,
        cell: &X1331Cell,
        proposals: &[Proposal],
        k: usize,
    ) -> Vec<u8> {
        self.rank(cell, proposals)
            .into_iter()
            .take(k.min(8))
            .map(|(state, _)| state)
            .collect()
    }

    pub fn learn(
        &mut self,
        proposals: &[Proposal],
        actual_rewards: &[f64; 8],
    ) {
        let baseline =
            actual_rewards.iter().sum::<f64>() / 8.0;

        let mut methods:
            HashMap<String, [f64; 8]> =
            HashMap::new();

        for proposal in proposals {
            methods
                .entry(proposal.method_key())
                .or_insert([0.0; 8])
                [proposal.state_value as usize] =
                proposal.score;
        }

        for (key, predictions) in methods {
            let correlation =
                pearson(&predictions, actual_rewards);

            let max_prediction = predictions
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            let top_states: Vec<usize> =
                predictions
                    .iter()
                    .enumerate()
                    .filter(|(_, score)| {
                        (**score - max_prediction).abs()
                            < 1e-12
                    })
                    .map(|(index, _)| index)
                    .collect();

            let top_reward =
                top_states
                    .iter()
                    .map(|i| actual_rewards[*i])
                    .sum::<f64>()
                    / top_states.len() as f64;

            self.memory.update_method(
                &key,
                correlation,
                top_reward,
                baseline,
            );
        }
    }
}

fn pearson(
    x: &[f64; 8],
    y: &[f64; 8],
) -> f64 {
    let mean_x = x.iter().sum::<f64>() / 8.0;
    let mean_y = y.iter().sum::<f64>() / 8.0;

    let mut numerator = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;

    for i in 0..8 {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;

        numerator += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }

    let denominator = (dx2 * dy2).sqrt();

    if denominator < 1e-15 {
        0.0
    } else {
        numerator / denominator
    }
}
