use serde::{Deserialize, Serialize};

pub const X1331_STATES: [&str; 8] = ["000", "001", "010", "011", "100", "101", "110", "111"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMemory {
    pub state: String,

    pub historical_hits: u64,

    pub cell_children: u64,
    pub cell_samples: u64,

    pub lz8_hits: u64,
    pub lz12_hits: u64,
    pub lz16_hits: u64,

    pub mean_lz_sum: f64,
    pub best_lz_sum: u64,
}

impl StateMemory {
    pub fn new(state: &str) -> Self {
        Self {
            state: state.to_string(),

            historical_hits: 0,

            cell_children: 0,
            cell_samples: 0,

            lz8_hits: 0,
            lz12_hits: 0,
            lz16_hits: 0,

            mean_lz_sum: 0.0,
            best_lz_sum: 0,
        }
    }

    pub fn mean_lz(&self) -> f64 {
        if self.cell_children == 0 {
            0.0
        } else {
            self.mean_lz_sum / self.cell_children as f64
        }
    }

    pub fn mean_best_lz(&self) -> f64 {
        if self.cell_children == 0 {
            0.0
        } else {
            self.best_lz_sum as f64 / self.cell_children as f64
        }
    }

    pub fn lz8_rate(&self) -> f64 {
        if self.cell_samples == 0 {
            0.0
        } else {
            self.lz8_hits as f64 / self.cell_samples as f64
        }
    }

    pub fn lz12_rate(&self) -> f64 {
        if self.cell_samples == 0 {
            0.0
        } else {
            self.lz12_hits as f64 / self.cell_samples as f64
        }
    }

    pub fn lz16_rate(&self) -> f64 {
        if self.cell_samples == 0 {
            0.0
        } else {
            self.lz16_hits as f64 / self.cell_samples as f64
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HhmSummary {
    pub historical_blocks: u64,
    pub historical_state_observations: u64,

    pub microcell_rows: u64,
    pub microcell_samples: u64,

    pub min_height: Option<u64>,
    pub max_height: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HhmMemory {
    pub summary: HhmSummary,
    pub states: Vec<StateMemory>,
}

impl HhmMemory {
    pub fn new() -> Self {
        Self {
            summary: HhmSummary {
                historical_blocks: 0,
                historical_state_observations: 0,

                microcell_rows: 0,
                microcell_samples: 0,

                min_height: None,
                max_height: None,
            },

            states: X1331_STATES
                .iter()
                .map(|state| StateMemory::new(state))
                .collect(),
        }
    }

    pub fn state_index(state: &str) -> Option<usize> {
        X1331_STATES
            .iter()
            .position(|candidate| *candidate == state)
    }

    pub fn observe_historical_state(&mut self, state: &str) {
        if let Some(index) = Self::state_index(state) {
            self.states[index].historical_hits += 1;
            self.summary.historical_state_observations += 1;
        }
    }

    pub fn observe_height(&mut self, height: u64) {
        self.summary.min_height = Some(
            self.summary
                .min_height
                .map_or(height, |current| current.min(height)),
        );

        self.summary.max_height = Some(
            self.summary
                .max_height
                .map_or(height, |current| current.max(height)),
        );
    }

    pub fn observe_microcell(
        &mut self,
        child: usize,
        samples: u64,
        mean_lz: f64,
        best_lz: u64,
        lz8: u64,
        lz12: u64,
        lz16: u64,
    ) {
        if child >= self.states.len() {
            return;
        }

        let state = &mut self.states[child];

        state.cell_children += 1;
        state.cell_samples += samples;

        state.mean_lz_sum += mean_lz;
        state.best_lz_sum += best_lz;

        state.lz8_hits += lz8;
        state.lz12_hits += lz12;
        state.lz16_hits += lz16;

        self.summary.microcell_rows += 1;
        self.summary.microcell_samples += samples;
    }
}
