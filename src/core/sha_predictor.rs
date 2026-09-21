#[derive(Debug, Clone)]
pub struct ShaPredictor {
    weights: [[f64; 9]; 8],
    learning_rate: f64,
}

impl ShaPredictor {
    pub fn new() -> Self {
        Self {
            weights: [[0.0; 9]; 8],
            learning_rate: 0.03,
        }
    }

    pub fn features(
        header: &[u8; 32],
        state: u8,
    ) -> [f64; 9] {
        let b0 = (state & 1) as f64;
        let b1 = ((state >> 1) & 1) as f64;
        let b2 = ((state >> 2) & 1) as f64;

        // Cheap pre-hash header features.
        // These are available BEFORE SHA256d evaluation.
        let h0 = header[0] as f64 / 255.0;
        let h1 = header[7] as f64 / 255.0;
        let h2 = header[15] as f64 / 255.0;
        let h3 = header[23] as f64 / 255.0;
        let h4 = header[31] as f64 / 255.0;

        [
            1.0,
            b0,
            b1,
            b2,
            h0,
            h1,
            h2,
            h3,
            h4,
        ]
    }

    pub fn scores(
        &self,
        header: &[u8; 32],
    ) -> [f64; 8] {
        let mut scores = [0.0; 8];

        for state in 0..8usize {
            let features =
                Self::features(
                    header,
                    state as u8,
                );

            scores[state] =
                self.weights[state]
                    .iter()
                    .zip(features.iter())
                    .map(|(weight, feature)| {
                        weight * feature
                    })
                    .sum();
        }

        scores
    }

    pub fn train(
        &mut self,
        header: &[u8; 32],
        target: u8,
    ) {
        let scores =
            self.scores(header);

        let max_score =
            scores
                .iter()
                .copied()
                .fold(
                    f64::NEG_INFINITY,
                    f64::max,
                );

        let mut exp_scores =
            [0.0; 8];

        let mut denominator =
            0.0;

        for i in 0..8 {
            exp_scores[i] =
                (scores[i] - max_score).exp();

            denominator +=
                exp_scores[i];
        }

        for state in 0..8usize {
            let probability =
                exp_scores[state]
                / denominator;

            let expected =
                if state == target as usize {
                    1.0
                } else {
                    0.0
                };

            let error =
                expected - probability;

            let features =
                Self::features(
                    header,
                    state as u8,
                );

            for j in 0..9 {
                self.weights[state][j] +=
                    self.learning_rate
                    * error
                    * features[j];
            }
        }
    }
}
