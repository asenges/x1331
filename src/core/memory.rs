use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodPerformance {
    pub experiments: u64,

    pub correlation_sum: f64,
    pub correlation_sq_sum: f64,

    pub top_reward_sum: f64,
    pub baseline_reward_sum: f64,

    pub wins: u64,
    pub losses: u64,
    pub ties: u64,
}

impl Default for MethodPerformance {
    fn default() -> Self {
        Self {
            experiments: 0,

            correlation_sum: 0.0,
            correlation_sq_sum: 0.0,

            top_reward_sum: 0.0,
            baseline_reward_sum: 0.0,

            wins: 0,
            losses: 0,
            ties: 0,
        }
    }
}

impl MethodPerformance {
    pub fn update(
        &mut self,
        correlation: f64,
        top_reward: f64,
        baseline_reward: f64,
    ) {
        self.experiments += 1;

        self.correlation_sum += correlation;
        self.correlation_sq_sum += correlation * correlation;

        self.top_reward_sum += top_reward;
        self.baseline_reward_sum += baseline_reward;

        let epsilon = 1e-12;

        if top_reward > baseline_reward + epsilon {
            self.wins += 1;
        } else if top_reward < baseline_reward - epsilon {
            self.losses += 1;
        } else {
            self.ties += 1;
        }
    }

    pub fn mean_correlation(&self) -> f64 {
        if self.experiments == 0 {
            return 0.0;
        }

        self.correlation_sum / self.experiments as f64
    }

    pub fn correlation_variance(&self) -> f64 {
        if self.experiments < 2 {
            return 0.0;
        }

        let n = self.experiments as f64;
        let mean = self.mean_correlation();

        ((self.correlation_sq_sum / n) - mean * mean)
            .max(0.0)
    }

    pub fn top_mean(&self) -> f64 {
        if self.experiments == 0 {
            return 0.0;
        }

        self.top_reward_sum / self.experiments as f64
    }

    pub fn baseline_mean(&self) -> f64 {
        if self.experiments == 0 {
            return 0.0;
        }

        self.baseline_reward_sum / self.experiments as f64
    }

    pub fn top_gain(&self) -> f64 {
        let baseline = self.baseline_mean();

        if baseline == 0.0 {
            return 1.0;
        }

        self.top_mean() / baseline
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ObserverMemory {
    pub methods: HashMap<String, MethodPerformance>,
    pub experiments: u64,
}

impl ObserverMemory {
    pub fn load(path: &str) -> Self {
        if !Path::new(path).exists() {
            return Self::default();
        }

        match fs::read_to_string(path) {
            Ok(data) => serde_json::from_str(&data)
                .unwrap_or_default(),

            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &str) {
        let data =
            serde_json::to_string_pretty(self).unwrap();

        fs::write(path, data).unwrap();
    }

    pub fn update_method(
        &mut self,
        key: &str,
        correlation: f64,
        top_reward: f64,
        baseline_reward: f64,
    ) {
        self.methods
            .entry(key.to_string())
            .or_default()
            .update(
                correlation,
                top_reward,
                baseline_reward,
            );
    }

    pub fn trust(&self, key: &str) -> f64 {
        let Some(stats) = self.methods.get(key) else {
            return 1.0;
        };

        // During early exploration don't let tiny samples
        // strongly influence the Observer.
        if stats.experiments < 30 {
            return 1.0;
        }

        let correlation =
            stats.mean_correlation();

        let gain =
            stats.top_gain() - 1.0;

        // Conservative bounded trust.
        //
        // 1.0 = neutral
        // >1  = historical positive evidence
        // <1  = historical negative evidence
        (1.0 + correlation * 0.25 + gain * 0.25)
            .clamp(0.75, 1.25)
    }
}
