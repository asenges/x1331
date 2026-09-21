#[derive(Debug, Clone)]
pub struct TinyPredictor {
    weights: [[f64; 4]; 8],
    learning_rate: f64,
}

impl TinyPredictor {
    pub fn new() -> Self {
        Self {
            weights: [[0.0; 4]; 8],
            learning_rate: 0.08,
        }
    }

    fn features(
        x1: f64,
        x2: f64,
        x3: f64,
    ) -> [f64; 4] {
        [1.0, x1, x2, x3]
    }

    pub fn scores(
        &self,
        x1: f64,
        x2: f64,
        x3: f64,
    ) -> [f64; 8] {
        let f = Self::features(x1, x2, x3);

        let mut scores = [0.0; 8];

        for state in 0..8 {
            scores[state] =
                self.weights[state]
                    .iter()
                    .zip(f.iter())
                    .map(|(w, x)| w * x)
                    .sum();
        }

        scores
    }

    pub fn predict(
        &self,
        x1: f64,
        x2: f64,
        x3: f64,
    ) -> u8 {
        let scores = self.scores(x1, x2, x3);

        scores
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.partial_cmp(b.1).unwrap()
            })
            .map(|(i, _)| i as u8)
            .unwrap()
    }

    pub fn train(
        &mut self,
        x1: f64,
        x2: f64,
        x3: f64,
        target: u8,
    ) {
        let f = Self::features(x1, x2, x3);
        let scores = self.scores(x1, x2, x3);

        // Softmax
        let max_score =
            scores.iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

        let mut exp_scores = [0.0; 8];
        let mut denominator = 0.0;

        for i in 0..8 {
            exp_scores[i] =
                (scores[i] - max_score).exp();

            denominator += exp_scores[i];
        }

        for state in 0..8 {
            let probability =
                exp_scores[state] / denominator;

            let expected =
                if state == target as usize {
                    1.0
                } else {
                    0.0
                };

            let error =
                expected - probability;

            for j in 0..4 {
                self.weights[state][j] +=
                    self.learning_rate
                    * error
                    * f[j];
            }
        }
    }
}
