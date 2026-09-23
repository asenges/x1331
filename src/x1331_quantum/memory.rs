use std::f64::consts::PI;

use super::bit_matrix::X1331State;
use super::schrodinger::Psi1331;

pub const CONTEXT_COUNT: usize = 8;
pub const STATE_COUNT: usize = 8;

const EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy)]
pub struct Experience {
    pub context: X1331State,
    pub outcome: X1331State,
}

#[derive(Debug, Clone)]
pub struct ContextMemory {
    counts: [[u64; STATE_COUNT]; CONTEXT_COUNT],
    totals: [u64; CONTEXT_COUNT],
    total_experiences: u64,
}

#[derive(Debug, Clone)]
pub struct MemoryPrediction {
    pub probabilities: [f64; STATE_COUNT],
    pub observations: u64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct ResonanceReport {
    pub observations: u64,
    pub count_probabilities: [f64; STATE_COUNT],
    pub psi_probabilities: [f64; STATE_COUNT],
    pub count_top: X1331State,
    pub psi_top: X1331State,
    pub resonance_strength: f64,
}

impl ContextMemory {
    pub fn new() -> Self {
        Self {
            counts: [[0; STATE_COUNT]; CONTEXT_COUNT],
            totals: [0; CONTEXT_COUNT],
            total_experiences: 0,
        }
    }

    pub fn total_experiences(&self) -> u64 {
        self.total_experiences
    }

    pub fn observations_for(&self, context: X1331State) -> u64 {
        self.totals[context.value() as usize]
    }

    pub fn count(&self, context: X1331State, outcome: X1331State) -> u64 {
        self.counts[context.value() as usize][outcome.value() as usize]
    }

    pub fn observe(&mut self, experience: Experience) {
        let context = experience.context.value() as usize;
        let outcome = experience.outcome.value() as usize;

        self.counts[context][outcome] += 1;
        self.totals[context] += 1;
        self.total_experiences += 1;
    }

    /// Ordinary count-memory prediction.
    ///
    /// Laplace smoothing keeps all eight possibilities represented.
    /// This is the conventional baseline against which X1331 resonance
    /// is compared.
    pub fn predict_counts(&self, context: X1331State) -> MemoryPrediction {
        let context_index = context.value() as usize;
        let observations = self.totals[context_index];

        let alpha = 1.0_f64;
        let denominator = observations as f64 + alpha * STATE_COUNT as f64;

        let probabilities = std::array::from_fn(|state| {
            (self.counts[context_index][state] as f64 + alpha) / denominator
        });

        let uniform = 1.0 / STATE_COUNT as f64;
        let max_probability = probabilities.iter().copied().fold(0.0_f64, f64::max);

        let confidence = if max_probability <= uniform {
            0.0
        } else {
            ((max_probability - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
        };

        MemoryPrediction {
            probabilities,
            observations,
            confidence,
        }
    }

    /// Build an X1331 Psi from causal memory.
    ///
    /// Important:
    /// - only already-observed experiences are used;
    /// - the current outcome is NOT known yet;
    /// - all eight states begin alive;
    /// - historical association changes amplitude and phase;
    /// - sparse memory remains conservative.
    pub fn resonate(&self, context: X1331State) -> (Psi1331, ResonanceReport) {
        let prediction = self.predict_counts(context);
        let observations = prediction.observations;

        let mut psi = Psi1331::uniform();

        if observations == 0 {
            let top = X1331State::S000;
            let psi_probabilities = psi.probabilities();

            return (
                psi,
                ResonanceReport {
                    observations,
                    count_probabilities: prediction.probabilities,
                    psi_probabilities,
                    count_top: top,
                    psi_top: top,
                    resonance_strength: 0.0,
                },
            );
        }

        let uniform = 1.0 / STATE_COUNT as f64;

        // Maturity prevents a tiny number of observations from producing
        // a strong resonance.
        let maturity = 1.0 - (-(observations as f64) / 24.0).exp();

        // Context geometry contributes phase. This is deliberately
        // deterministic and inspectable.
        for state in X1331State::ALL {
            let index = state.value() as usize;
            let probability = prediction.probabilities[index];

            let deviation = probability - uniform;
            let magnitude = deviation.abs() * 1.75 * maturity;

            if magnitude <= EPSILON {
                continue;
            }

            let distance = context.hamming_distance(state) as f64;

            let geometric_phase = distance * PI / 8.0;

            // Above-uniform associations reinforce.
            // Below-uniform associations receive opposite phase.
            let phase = if deviation >= 0.0 {
                geometric_phase
            } else {
                geometric_phase + PI
            };

            psi.interfere(state, magnitude, phase);
        }

        let psi_probabilities = psi.probabilities();

        let count_top = top_state(&prediction.probabilities);
        let psi_top = top_state(&psi_probabilities);

        let resonance_strength = l1_distance_from_uniform(&psi_probabilities);

        let report = ResonanceReport {
            observations,
            count_probabilities: prediction.probabilities,
            psi_probabilities,
            count_top,
            psi_top,
            resonance_strength,
        };

        (psi, report)
    }
}

impl Default for ContextMemory {
    fn default() -> Self {
        Self::new()
    }
}

pub fn top_state(probabilities: &[f64; STATE_COUNT]) -> X1331State {
    let mut best_index = 0usize;
    let mut best_probability = probabilities[0];

    for (index, probability) in probabilities.iter().enumerate().skip(1) {
        if *probability > best_probability {
            best_probability = *probability;
            best_index = index;
        }
    }

    X1331State::from_u8(best_index as u8)
}

pub fn probability_rank(probabilities: &[f64; STATE_COUNT], actual: X1331State) -> usize {
    let actual_index = actual.value() as usize;
    let actual_probability = probabilities[actual_index];

    1 + probabilities
        .iter()
        .enumerate()
        .filter(|(index, probability)| {
            **probability > actual_probability
                || ((**probability - actual_probability).abs() <= EPSILON && *index < actual_index)
        })
        .count()
}

pub fn log_loss(probabilities: &[f64; STATE_COUNT], actual: X1331State) -> f64 {
    let probability = probabilities[actual.value() as usize].clamp(EPSILON, 1.0);

    -probability.ln()
}

pub fn brier_score(probabilities: &[f64; STATE_COUNT], actual: X1331State) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .map(|(index, probability)| {
            let target = if index == actual.value() as usize {
                1.0
            } else {
                0.0
            };

            let error = probability - target;
            error * error
        })
        .sum()
}

pub fn recall_at_k(probabilities: &[f64; STATE_COUNT], actual: X1331State, k: usize) -> bool {
    probability_rank(probabilities, actual) <= k
}

pub fn l1_distance_from_uniform(probabilities: &[f64; STATE_COUNT]) -> f64 {
    let uniform = 1.0 / STATE_COUNT as f64;

    0.5 * probabilities
        .iter()
        .map(|probability| (probability - uniform).abs())
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "{left} not within {tolerance} of {right}"
        );
    }

    #[test]
    fn empty_memory_is_uniform() {
        let memory = ContextMemory::new();

        let prediction = memory.predict_counts(X1331State::S101);

        for probability in prediction.probabilities {
            assert_close(probability, 0.125, 1.0e-12);
        }

        assert_eq!(prediction.observations, 0);
        assert_close(prediction.confidence, 0.0, 1.0e-12);
    }

    #[test]
    fn experience_is_context_specific() {
        let mut memory = ContextMemory::new();

        for _ in 0..20 {
            memory.observe(Experience {
                context: X1331State::S001,
                outcome: X1331State::S111,
            });
        }

        let learned = memory.predict_counts(X1331State::S001);

        let unseen = memory.predict_counts(X1331State::S010);

        assert!(learned.probabilities[X1331State::S111.value() as usize] > 0.125);

        for probability in unseen.probabilities {
            assert_close(probability, 0.125, 1.0e-12);
        }
    }

    #[test]
    fn resonance_preserves_normalization() {
        let mut memory = ContextMemory::new();

        for _ in 0..100 {
            memory.observe(Experience {
                context: X1331State::S101,
                outcome: X1331State::S011,
            });
        }

        let (psi, report) = memory.resonate(X1331State::S101);

        assert_close(psi.total_mass(), 1.0, 1.0e-10);

        assert_eq!(report.count_top, X1331State::S011);

        assert_eq!(report.psi_top, X1331State::S011);
    }

    #[test]
    fn learned_association_changes_future_psi() {
        let mut memory = ContextMemory::new();

        let (_, before) = memory.resonate(X1331State::S100);

        for _ in 0..64 {
            memory.observe(Experience {
                context: X1331State::S100,
                outcome: X1331State::S110,
            });
        }

        let (_, after) = memory.resonate(X1331State::S100);

        assert!(
            after.psi_probabilities[X1331State::S110.value() as usize]
                > before.psi_probabilities[X1331State::S110.value() as usize]
        );
    }

    #[test]
    fn ranking_and_recall_are_consistent() {
        let probabilities = [0.40, 0.20, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04];

        assert_eq!(probability_rank(&probabilities, X1331State::S000), 1);

        assert_eq!(probability_rank(&probabilities, X1331State::S001), 2);

        assert!(recall_at_k(&probabilities, X1331State::S001, 2));

        assert!(!recall_at_k(&probabilities, X1331State::S001, 1));
    }
}
