#[derive(Debug, Clone)]
pub struct SyntheticContext {
    pub x1: f64,
    pub x2: f64,
    pub x3: f64,
}

impl SyntheticContext {
    pub fn hidden_target(&self) -> u8 {
        //
        // HIDDEN RULE
        //
        // Deliberadamente NO la conoce el Observer.
        //
        // Cada feature controla un bit.
        //
        let b2 = if self.x1 > 0.5 { 1 } else { 0 };
        let b1 = if self.x2 > 0.5 { 1 } else { 0 };
        let b0 = if self.x3 > 0.5 { 1 } else { 0 };

        (b2 << 2) | (b1 << 1) | b0
    }

    pub fn rewards(&self) -> [f64; 8] {
        let target = self.hidden_target();

        let mut rewards = [0.0; 8];

        for state in 0..8u8 {
            let distance =
                (state ^ target).count_ones();

            // Best state = 1.0
            // one bit away = 0.5
            // two bits = 0.25
            // three bits = 0.125
            rewards[state as usize] =
                1.0 / 2_f64.powi(distance as i32);
        }

        rewards
    }
}
