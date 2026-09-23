use super::bit_matrix::{X1331BitMatrix, X1331State};
use super::schrodinger::Psi1331;

pub const STATE_COUNT: usize = 8;
pub const HISTORY_CELLS: usize = 3;
pub const CONTEXT_COUNT: usize = 512;

const EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryObserverAction {
    Pass,
    Retain4,
    Retain2,
    Collapse1,
}

impl BinaryObserverAction {
    pub fn k(self) -> usize {
        match self {
            Self::Pass => 8,
            Self::Retain4 => 4,
            Self::Retain2 => 2,
            Self::Collapse1 => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Retain4 => "K4",
            Self::Retain2 => "K2",
            Self::Collapse1 => "K1",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinaryPrediction {
    pub probabilities: [f64; STATE_COUNT],
    pub observations: u64,
    pub top_state: X1331State,
    pub top_probability: f64,
    pub second_probability: f64,
    pub entropy_bits: f64,
    pub concentration: f64,
}

#[derive(Debug, Clone)]
pub struct BinaryTemporalMemory {
    counts: Vec<[u64; STATE_COUNT]>,
    totals: Vec<u64>,
    experiences: u64,
}

impl BinaryTemporalMemory {
    pub fn new() -> Self {
        Self {
            counts: vec![[0; STATE_COUNT]; CONTEXT_COUNT],
            totals: vec![0; CONTEXT_COUNT],
            experiences: 0,
        }
    }

    pub fn experiences(&self) -> u64 {
        self.experiences
    }

    pub fn context_index(history: [X1331State; HISTORY_CELLS]) -> usize {
        ((history[0].value() as usize) << 6)
            | ((history[1].value() as usize) << 3)
            | history[2].value() as usize
    }

    pub fn history_from_matrix(
        matrix: &X1331BitMatrix,
        start_bit: usize,
    ) -> Option<[X1331State; HISTORY_CELLS]> {
        let a = matrix.cell_at(start_bit)?.state;
        let b = matrix.cell_at(start_bit + 3)?.state;
        let c = matrix.cell_at(start_bit + 6)?.state;

        Some([a, b, c])
    }

    pub fn target_from_matrix(matrix: &X1331BitMatrix, start_bit: usize) -> Option<X1331State> {
        Some(matrix.cell_at(start_bit + 9)?.state)
    }

    pub fn observe(&mut self, history: [X1331State; HISTORY_CELLS], actual: X1331State) {
        let context = Self::context_index(history);
        let outcome = actual.value() as usize;

        self.counts[context][outcome] += 1;
        self.totals[context] += 1;
        self.experiences += 1;
    }

    pub fn predict(&self, history: [X1331State; HISTORY_CELLS]) -> BinaryPrediction {
        let context = Self::context_index(history);
        let observations = self.totals[context];

        let alpha = 1.0;
        let denominator = observations as f64 + alpha * STATE_COUNT as f64;

        let probabilities =
            std::array::from_fn(|state| (self.counts[context][state] as f64 + alpha) / denominator);

        prediction_from_probabilities(probabilities, observations)
    }

    pub fn resonate(&self, history: [X1331State; HISTORY_CELLS]) -> (Psi1331, BinaryPrediction) {
        let prediction = self.predict(history);
        let mut psi = Psi1331::uniform();

        if prediction.observations == 0 {
            return (psi, prediction);
        }

        let maturity = 1.0 - (-(prediction.observations as f64) / 16.0).exp();

        let uniform = 1.0 / STATE_COUNT as f64;

        for state in X1331State::ALL {
            let index = state.value() as usize;

            let deviation = prediction.probabilities[index] - uniform;

            let magnitude = deviation.abs() * 1.50 * maturity;

            if magnitude <= EPSILON {
                continue;
            }

            let distance = history[HISTORY_CELLS - 1].hamming_distance(state) as f64;

            let geometric_phase = distance * std::f64::consts::PI / 8.0;

            let phase = if deviation >= 0.0 {
                geometric_phase
            } else {
                geometric_phase + std::f64::consts::PI
            };

            psi.interfere(state, magnitude, phase);
        }

        (psi, prediction)
    }
}

impl Default for BinaryTemporalMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ProspectiveCalibration {
    evaluated: u64,
    top1_hits: u64,
    top2_hits: u64,
    top4_hits: u64,
}

impl ProspectiveCalibration {
    pub fn new() -> Self {
        Self {
            evaluated: 0,
            top1_hits: 0,
            top2_hits: 0,
            top4_hits: 0,
        }
    }

    pub fn evaluated(&self) -> u64 {
        self.evaluated
    }

    pub fn top1_rate(&self) -> f64 {
        ratio(self.top1_hits, self.evaluated)
    }

    pub fn top2_rate(&self) -> f64 {
        ratio(self.top2_hits, self.evaluated)
    }

    pub fn top4_rate(&self) -> f64 {
        ratio(self.top4_hits, self.evaluated)
    }

    pub fn observe(&mut self, probabilities: &[f64; STATE_COUNT], actual: X1331State) {
        self.evaluated += 1;

        let ranked = ranked_states(probabilities);

        if ranked[0] == actual {
            self.top1_hits += 1;
        }

        if ranked[..2].contains(&actual) {
            self.top2_hits += 1;
        }

        if ranked[..4].contains(&actual) {
            self.top4_hits += 1;
        }
    }
}

impl Default for ProspectiveCalibration {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct BinaryTemporalObserver {
    calibration: ProspectiveCalibration,
    min_context_observations: u64,
    min_calibration_samples: u64,
}

impl BinaryTemporalObserver {
    pub fn new() -> Self {
        Self {
            calibration: ProspectiveCalibration::new(),
            min_context_observations: 12,
            min_calibration_samples: 256,
        }
    }

    pub fn calibration(&self) -> &ProspectiveCalibration {
        &self.calibration
    }

    pub fn decide(&self, psi: &Psi1331, prediction: &BinaryPrediction) -> BinaryObserverAction {
        if prediction.observations < self.min_context_observations {
            return BinaryObserverAction::Pass;
        }

        if self.calibration.evaluated() < self.min_calibration_samples {
            return BinaryObserverAction::Pass;
        }

        let psi_probabilities = psi.probabilities();

        let current = prediction_from_probabilities(psi_probabilities, prediction.observations);

        let top1 = self.calibration.top1_rate();
        let top2 = self.calibration.top2_rate();
        let top4 = self.calibration.top4_rate();

        if top1 >= 0.90
            && current.concentration >= 0.55
            && (current.top_probability - current.second_probability) >= 0.25
        {
            BinaryObserverAction::Collapse1
        } else if top2 >= 0.80 && current.concentration >= 0.25 {
            BinaryObserverAction::Retain2
        } else if top4 >= 0.75 && current.concentration >= 0.08 {
            BinaryObserverAction::Retain4
        } else {
            BinaryObserverAction::Pass
        }
    }

    pub fn observe_result(&mut self, probabilities: &[f64; STATE_COUNT], actual: X1331State) {
        self.calibration.observe(probabilities, actual);
    }
}

impl Default for BinaryTemporalObserver {
    fn default() -> Self {
        Self::new()
    }
}

pub fn ranked_states(probabilities: &[f64; STATE_COUNT]) -> [X1331State; STATE_COUNT] {
    let mut states = X1331State::ALL;

    states.sort_by(|left, right| {
        probabilities[right.value() as usize]
            .partial_cmp(&probabilities[left.value() as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.value().cmp(&right.value()))
    });

    states
}

pub fn retained(probabilities: &[f64; STATE_COUNT], actual: X1331State, k: usize) -> bool {
    ranked_states(probabilities)[..k].contains(&actual)
}

fn prediction_from_probabilities(
    probabilities: [f64; STATE_COUNT],
    observations: u64,
) -> BinaryPrediction {
    let ranked = ranked_states(&probabilities);

    let top_state = ranked[0];
    let second_state = ranked[1];

    let top_probability = probabilities[top_state.value() as usize];

    let second_probability = probabilities[second_state.value() as usize];

    let entropy_bits: f64 = probabilities
        .iter()
        .filter(|p| **p > EPSILON)
        .map(|p| -p * p.log2())
        .sum();

    let max_entropy = (STATE_COUNT as f64).log2();

    let concentration = (1.0 - entropy_bits / max_entropy).clamp(0.0, 1.0);

    BinaryPrediction {
        probabilities,
        observations,
        top_state,
        top_probability,
        second_probability,
        entropy_bits,
        concentration,
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_bits_produce_exact_temporal_figures() {
        let raw = vec![0b00000101, 0b01110000];

        let matrix = X1331BitMatrix::from_bytes(raw);

        let history = BinaryTemporalMemory::history_from_matrix(&matrix, 0).unwrap();

        let target = BinaryTemporalMemory::target_from_matrix(&matrix, 0).unwrap();

        assert_eq!(history[0], X1331State::S000);
        assert_eq!(history[1], X1331State::S001);
        assert_eq!(history[2], X1331State::S010);
        assert_eq!(target, X1331State::S111);
    }

    #[test]
    fn context_index_covers_512_states() {
        let mut seen = vec![false; CONTEXT_COUNT];

        for a in X1331State::ALL {
            for b in X1331State::ALL {
                for c in X1331State::ALL {
                    let index = BinaryTemporalMemory::context_index([a, b, c]);

                    assert!(!seen[index]);
                    seen[index] = true;
                }
            }
        }

        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn empty_memory_is_uniform() {
        let memory = BinaryTemporalMemory::new();

        let prediction = memory.predict([X1331State::S000, X1331State::S001, X1331State::S010]);

        assert_eq!(prediction.observations, 0);

        for p in prediction.probabilities {
            assert!((p - 0.125).abs() < 1.0e-12);
        }
    }

    #[test]
    fn memory_learns_binary_figure_relation() {
        let mut memory = BinaryTemporalMemory::new();

        let history = [X1331State::S011, X1331State::S101, X1331State::S110];

        for _ in 0..128 {
            memory.observe(history, X1331State::S111);
        }

        let prediction = memory.predict(history);

        assert_eq!(prediction.top_state, X1331State::S111);

        assert!(prediction.top_probability > 0.90);
    }

    #[test]
    fn observer_passes_without_calibration() {
        let memory = BinaryTemporalMemory::new();

        let observer = BinaryTemporalObserver::new();

        let history = [X1331State::S000, X1331State::S001, X1331State::S010];

        let (psi, prediction) = memory.resonate(history);

        assert_eq!(
            observer.decide(&psi, &prediction),
            BinaryObserverAction::Pass
        );
    }
}
