use std::f64::consts::FRAC_1_SQRT_2;

#[derive(Clone, Copy, Debug)]
struct Amp {
    re: f64,
    im: f64,
}

impl Amp {
    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

struct QuantumRegister {
    qubits: usize,
    amplitudes: Vec<Amp>,
}

impl QuantumRegister {
    fn basis_zero(qubits: usize) -> Self {
        let size = 1usize << qubits;

        let mut amplitudes = vec![Amp::zero(); size];
        amplitudes[0] = Amp { re: 1.0, im: 0.0 };

        Self { qubits, amplitudes }
    }

    fn hadamard(&mut self, target: usize) {
        assert!(target < self.qubits);

        let mask = 1usize << target;

        for base in 0..self.amplitudes.len() {
            if base & mask != 0 {
                continue;
            }

            let other = base | mask;

            let a = self.amplitudes[base];
            let b = self.amplitudes[other];

            self.amplitudes[base] = Amp {
                re: (a.re + b.re) * FRAC_1_SQRT_2,
                im: (a.im + b.im) * FRAC_1_SQRT_2,
            };

            self.amplitudes[other] = Amp {
                re: (a.re - b.re) * FRAC_1_SQRT_2,
                im: (a.im - b.im) * FRAC_1_SQRT_2,
            };
        }
    }

    fn uniform(&mut self) {
        for q in 0..self.qubits {
            self.hadamard(q);
        }
    }

    // Grover phase oracle.
    fn mark(&mut self, state: usize) {
        self.amplitudes[state].re *= -1.0;
        self.amplitudes[state].im *= -1.0;
    }

    // Reflection around the global mean amplitude.
    fn diffuse(&mut self) {
        let n = self.amplitudes.len() as f64;

        let mean_re =
            self.amplitudes.iter().map(|a| a.re).sum::<f64>() / n;

        let mean_im =
            self.amplitudes.iter().map(|a| a.im).sum::<f64>() / n;

        for a in &mut self.amplitudes {
            a.re = 2.0 * mean_re - a.re;
            a.im = 2.0 * mean_im - a.im;
        }
    }

    fn probability(&self, state: usize) -> f64 {
        self.amplitudes[state].norm_sqr()
    }

    fn total_probability(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_sqr()).sum()
    }

    fn winner(&self) -> (usize, f64) {
        self.amplitudes
            .iter()
            .enumerate()
            .map(|(i, a)| (i, a.norm_sqr()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }
}

fn main() {
    const QUBITS: usize = 8;
    const TARGET: usize = 173;

    let states = 1usize << QUBITS;

    println!("X1331 LIVE-12B — GROVER 8-QUBIT LAB");
    println!("====================================");
    println!("qubits     : {QUBITS}");
    println!("states     : {states}");
    println!("target     : {TARGET} (0b{TARGET:08b})");

    let mut q = QuantumRegister::basis_zero(QUBITS);

    q.uniform();

    println!(
        "\nInitial target probability: {:.9}",
        q.probability(TARGET)
    );

    // Optimal iteration count for one marked item:
    // floor(pi/4 * sqrt(N)).
    let iterations =
        (std::f64::consts::PI / 4.0 * (states as f64).sqrt()).floor() as usize;

    println!("Grover iterations          : {iterations}");

    for i in 0..iterations {
        q.mark(TARGET);
        q.diffuse();

        let p = q.probability(TARGET);

        println!(
            "iteration {:02}: target probability = {:.9}",
            i + 1,
            p
        );
    }

    let (winner, probability) = q.winner();

    println!();
    println!("total probability : {:.12}", q.total_probability());
    println!("winner            : {winner} (0b{winner:08b})");
    println!("winner probability: {:.9}", probability);

    assert!((q.total_probability() - 1.0).abs() < 1.0e-10);
    assert_eq!(winner, TARGET);

    println!();
    println!("PASS — 8-qubit Grover amplified the marked state.");
}
