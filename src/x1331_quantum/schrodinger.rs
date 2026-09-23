use std::f64::consts::PI;

use super::bit_matrix::X1331State;

pub const STATE_COUNT: usize = 8;
const EPSILON: f64 = 1.0e-15;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Amplitude {
    pub re: f64,
    pub im: f64,
}

impl Amplitude {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn scale(self, factor: f64) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    pub fn rotate(self, phase: f64) -> Self {
        let c = phase.cos();
        let s = phase.sin();

        Self {
            re: self.re * c - self.im * s,
            im: self.re * s + self.im * c,
        }
    }

    pub fn from_polar(magnitude: f64, phase: f64) -> Self {
        Self {
            re: magnitude * phase.cos(),
            im: magnitude * phase.sin(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Psi1331 {
    amplitudes: [Amplitude; STATE_COUNT],
}

#[derive(Debug, Clone)]
pub struct RankedState {
    pub state: X1331State,
    pub probability: f64,
}

#[derive(Debug, Clone)]
pub struct CollapseDecision {
    pub k: usize,
    pub states: Vec<RankedState>,
    pub retained_mass: f64,
    pub entropy_bits: f64,
    pub concentration: f64,
    pub separation: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct EvolutionConfig {
    /// Fraction of each state's amplitude retained directly each cycle.
    pub self_weight: f64,

    /// Total incoming contribution from the three cube neighbors.
    pub neighbor_weight: f64,

    /// Phase applied per Hamming-distance unit from the stimulus.
    pub phase_step: f64,

    /// Extra coherent amplitude injected into the stimulus state.
    pub stimulus_gain: f64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            self_weight: 0.82,
            neighbor_weight: 0.18,
            phase_step: PI / 8.0,
            stimulus_gain: 0.08,
        }
    }
}

impl Psi1331 {
    pub fn uniform() -> Self {
        let magnitude = 1.0 / (STATE_COUNT as f64).sqrt();

        Self {
            amplitudes: [Amplitude::new(magnitude, 0.0); STATE_COUNT],
        }
    }

    pub fn basis(state: X1331State) -> Self {
        let mut amplitudes = [Amplitude::zero(); STATE_COUNT];
        amplitudes[state.value() as usize] = Amplitude::new(1.0, 0.0);

        Self { amplitudes }
    }

    pub fn amplitudes(&self) -> &[Amplitude; STATE_COUNT] {
        &self.amplitudes
    }

    pub fn total_mass(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_sqr()).sum()
    }

    pub fn normalize(&mut self) {
        let mass = self.total_mass();

        assert!(
            mass.is_finite() && mass > EPSILON,
            "Psi cannot normalize zero or invalid mass"
        );

        let factor = 1.0 / mass.sqrt();

        for amplitude in &mut self.amplitudes {
            *amplitude = amplitude.scale(factor);
        }
    }

    pub fn probability(&self, state: X1331State) -> f64 {
        self.amplitudes[state.value() as usize].norm_sqr()
    }

    pub fn probabilities(&self) -> [f64; STATE_COUNT] {
        std::array::from_fn(|i| self.amplitudes[i].norm_sqr())
    }

    pub fn entropy_bits(&self) -> f64 {
        self.probabilities()
            .iter()
            .filter(|p| **p > EPSILON)
            .map(|p| -p * p.log2())
            .sum()
    }

    /// Normalized concentration relative to a uniform eight-state Psi.
    ///
    /// 0.0 = maximum entropy (uniform)
    /// 1.0 = one-state basis vector
    pub fn concentration(&self) -> f64 {
        let max_entropy = (STATE_COUNT as f64).log2();
        (1.0 - self.entropy_bits() / max_entropy).clamp(0.0, 1.0)
    }

    pub fn ranked(&self) -> Vec<RankedState> {
        let mut ranked: Vec<_> = X1331State::ALL
            .iter()
            .map(|state| RankedState {
                state: *state,
                probability: self.probability(*state),
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.state.value().cmp(&b.state.value()))
        });

        ranked
    }

    pub fn top_separation(&self) -> f64 {
        let ranked = self.ranked();

        if ranked.len() < 2 {
            return 0.0;
        }

        (ranked[0].probability - ranked[1].probability).max(0.0)
    }

    /// Explicit coherent signal injection.
    ///
    /// Positive/negative phase can reinforce or cancel the existing
    /// amplitude. This is a classical complex-amplitude operation.
    pub fn interfere(&mut self, state: X1331State, magnitude: f64, phase: f64) {
        let index = state.value() as usize;
        let signal = Amplitude::from_polar(magnitude, phase);

        self.amplitudes[index] = self.amplitudes[index].add(signal);
        self.normalize();
    }

    /// One deterministic quantum-inspired X1331 evolution cycle.
    ///
    /// Every state remains represented. Each state receives:
    /// - retained self amplitude
    /// - coherent contributions from its three cube neighbors
    /// - phase determined by relation to the incoming 3-bit stimulus
    /// - a small coherent stimulus injection
    ///
    /// No SHA and no randomness occur here.
    pub fn evolve(&mut self, stimulus: X1331State, config: EvolutionConfig) {
        let old = self.amplitudes;
        let mut next = [Amplitude::zero(); STATE_COUNT];

        for state in X1331State::ALL {
            let index = state.value() as usize;

            let distance = state.hamming_distance(stimulus) as f64;
            let local_phase = distance * config.phase_step;

            let mut value = old[index].scale(config.self_weight).rotate(local_phase);

            let neighbors = state.neighbors();
            let per_neighbor = config.neighbor_weight / 3.0;

            for neighbor in neighbors {
                let neighbor_index = neighbor.value() as usize;

                let relation = neighbor.hamming_distance(stimulus) as f64;

                let signed_phase = if neighbor.layer() <= state.layer() {
                    relation * config.phase_step
                } else {
                    -relation * config.phase_step
                };

                let contribution = old[neighbor_index].scale(per_neighbor).rotate(signed_phase);

                value = value.add(contribution);
            }

            next[index] = value;
        }

        let stimulus_index = stimulus.value() as usize;

        next[stimulus_index] =
            next[stimulus_index].add(Amplitude::from_polar(config.stimulus_gain, 0.0));

        self.amplitudes = next;
        self.normalize();
    }

    /// Conservative adaptive reduction.
    ///
    /// This intentionally requires BOTH:
    /// - distribution concentration
    /// - separation between the top two states
    ///
    /// Therefore cumulative probability alone cannot cause collapse.
    ///
    /// Thresholds are experimental mechanics for LIVE-09B, not calibrated
    /// Bitcoin confidence thresholds.
    pub fn decision(&self) -> CollapseDecision {
        let ranked = self.ranked();
        let entropy = self.entropy_bits();
        let concentration = self.concentration();
        let separation = self.top_separation();

        let k = if concentration >= 0.72 && separation >= 0.30 {
            1
        } else if concentration >= 0.38 && separation >= 0.10 {
            2
        } else if concentration >= 0.12 {
            4
        } else {
            8
        };

        let states: Vec<_> = ranked.iter().take(k).cloned().collect();
        let retained_mass = states.iter().map(|item| item.probability).sum();

        CollapseDecision {
            k,
            states,
            retained_mass,
            entropy_bits: entropy,
            concentration,
            separation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tolerance: f64) {
        assert!(
            (a - b).abs() <= tolerance,
            "{a} not within {tolerance} of {b}"
        );
    }

    #[test]
    fn uniform_is_normalized() {
        let psi = Psi1331::uniform();

        assert_close(psi.total_mass(), 1.0, 1.0e-12);

        for state in X1331State::ALL {
            assert_close(psi.probability(state), 0.125, 1.0e-12);
        }
    }

    #[test]
    fn uniform_passes() {
        let psi = Psi1331::uniform();
        let decision = psi.decision();

        assert_eq!(decision.k, 8);
        assert_close(decision.concentration, 0.0, 1.0e-12);
        assert_close(decision.separation, 0.0, 1.0e-12);
    }

    #[test]
    fn basis_state_collapses_to_one() {
        let psi = Psi1331::basis(X1331State::S101);
        let decision = psi.decision();

        assert_eq!(decision.k, 1);
        assert_eq!(decision.states[0].state, X1331State::S101);
        assert_close(decision.retained_mass, 1.0, 1.0e-12);
    }

    #[test]
    fn constructive_interference_increases_mass() {
        let mut psi = Psi1331::uniform();

        let before = psi.probability(X1331State::S101);

        psi.interfere(X1331State::S101, 0.25, 0.0);

        let after = psi.probability(X1331State::S101);

        assert!(after > before);
        assert_close(psi.total_mass(), 1.0, 1.0e-12);
    }

    #[test]
    fn opposite_phase_reduces_target_mass() {
        let mut constructive = Psi1331::uniform();
        let mut destructive = Psi1331::uniform();

        constructive.interfere(X1331State::S011, 0.20, 0.0);
        destructive.interfere(X1331State::S011, 0.20, PI);

        let pc = constructive.probability(X1331State::S011);
        let pd = destructive.probability(X1331State::S011);

        assert!(pc > pd);
        assert_close(constructive.total_mass(), 1.0, 1.0e-12);
        assert_close(destructive.total_mass(), 1.0, 1.0e-12);
    }

    #[test]
    fn evolution_preserves_normalization() {
        let mut psi = Psi1331::uniform();
        let config = EvolutionConfig::default();

        for i in 0..10_000 {
            let stimulus = X1331State::from_u8((i & 7) as u8);
            psi.evolve(stimulus, config);

            assert_close(psi.total_mass(), 1.0, 1.0e-10);

            for p in psi.probabilities() {
                assert!(p.is_finite());
                assert!(p >= 0.0);
                assert!(p <= 1.0 + 1.0e-12);
            }
        }
    }

    #[test]
    fn cube_neighbor_relationship_is_exact() {
        for state in X1331State::ALL {
            for neighbor in state.neighbors() {
                assert_eq!(state.hamming_distance(neighbor), 1);
            }
        }
    }
}
