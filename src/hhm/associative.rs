use serde::{Deserialize, Serialize};

pub const STATES: [&str; 8] = ["000", "001", "010", "011", "100", "101", "110", "111"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalState {
    pub state: String,
    pub evidence: f64,
    pub probability: f64,
    pub observations: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociativeObserver {
    pub states: Vec<SignalState>,

    pub incoming: u64,
    pub cognitive_cycles: u64,

    pub recognition_rate: f64,
    pub confidence: f64,
    pub stability: f64,

    pub leader: Option<String>,
    pub previous_leader: Option<String>,
    pub leader_streak: u64,

    pub collapses: u64,
    pub passes: u64,

    pub last_action: String,
}

impl AssociativeObserver {
    pub fn new(priors: [f64; 8]) -> Self {
        let mut states = Vec::with_capacity(8);

        let sum: f64 = priors.iter().sum();

        for i in 0..8 {
            let p = if sum > 0.0 {
                priors[i] / sum
            } else {
                1.0 / 8.0
            };

            states.push(SignalState {
                state: STATES[i].to_string(),
                evidence: 0.0,
                probability: p,
                observations: 0,
            });
        }

        Self {
            states,
            incoming: 0,
            cognitive_cycles: 0,
            recognition_rate: 0.0,
            confidence: 0.5,
            stability: 0.0,
            leader: None,
            previous_leader: None,
            leader_streak: 0,
            collapses: 0,
            passes: 0,
            last_action: "PASS".to_string(),
        }
    }

    pub fn observe_signals(&mut self, signals: [f64; 8]) {
        self.incoming += 1;

        /*
         * Important:
         *
         * These are associative signals, NOT probabilities that a
         * Bitcoin hash will be valid.
         *
         * LIVE-08B does not SHA256d and does not submit.
         */
        for (state, signal) in self.states.iter_mut().zip(signals) {
            state.evidence += signal;
            state.observations += 1;
        }

        self.recompute_distribution();
        self.cognitive_cycle();
    }

    fn recompute_distribution(&mut self) {
        let max_evidence = self
            .states
            .iter()
            .map(|s| s.evidence)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut weights = [0.0_f64; 8];
        let mut total = 0.0;

        for (i, state) in self.states.iter().enumerate() {
            /*
             * Softmax over average accumulated associative evidence.
             * Temperature intentionally conservative.
             */
            let avg = if state.observations > 0 {
                state.evidence / state.observations as f64
            } else {
                0.0
            };

            let centered = if max_evidence.is_finite() {
                avg.clamp(-8.0, 8.0)
            } else {
                0.0
            };

            let w = (centered / 4.0).exp();

            weights[i] = w;
            total += w;
        }

        if total <= 0.0 {
            return;
        }

        for (state, weight) in self.states.iter_mut().zip(weights) {
            state.probability = weight / total;
        }
    }

    fn cognitive_cycle(&mut self) {
        self.cognitive_cycles += 1;

        let mut ranked: Vec<(usize, f64)> = self
            .states
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.probability))
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best = ranked[0];
        let second = ranked[1];

        let new_leader = self.states[best.0].state.clone();

        self.previous_leader = self.leader.clone();

        if self.leader.as_deref() == Some(new_leader.as_str()) {
            self.leader_streak += 1;
        } else {
            self.leader_streak = 1;
            self.leader = Some(new_leader);
        }

        let margin = (best.1 - second.1).max(0.0);

        /*
         * Stability means persistence of the current interpretation.
         * It is NOT cryptographic certainty.
         */
        self.stability = 1.0 - (-((self.leader_streak as f64) / 32.0)).exp();

        /*
         * Recognition measures separation among the eight active
         * possibilities. 0 means essentially no preference.
         */
        self.recognition_rate = (margin / 0.875).clamp(0.0, 1.0);

        /*
         * Deliberately conservative placeholder confidence.
         *
         * It remains close to neutral until prospective calibration
         * exists. This is NOT allowed to trigger real verification.
         */
        let maturity = 1.0 - (-(self.incoming as f64 / 512.0)).exp();

        let internal_signal = self.recognition_rate * self.stability * maturity;

        self.confidence = 0.5 + 0.49 * internal_signal.clamp(0.0, 1.0);

        /*
         * LIVE-08B is shadow-only.
         *
         * A shadow collapse records what Schrödinger WOULD choose.
         * No SHA and no mining.submit are permitted.
         */
        let shadow_ready = self.incoming >= 128
            && self.confidence >= 0.90
            && self.stability >= 0.90
            && self.recognition_rate >= 0.50;

        if shadow_ready {
            self.collapses += 1;
            self.last_action = format!(
                "SHADOW_COLLAPSE:{}",
                self.leader.as_deref().unwrap_or("UNKNOWN")
            );
        } else {
            self.passes += 1;
            self.last_action = "PASS".to_string();
        }
    }
}
