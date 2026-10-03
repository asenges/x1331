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
        let mut amplitudes = vec![Amp::zero(); 1usize << qubits];
        amplitudes[0] = Amp { re: 1.0, im: 0.0 };

        Self { qubits, amplitudes }
    }

    fn x(&mut self, target: usize) {
        let mask = 1usize << target;

        for i in 0..self.amplitudes.len() {
            if i & mask == 0 {
                self.amplitudes.swap(i, i | mask);
            }
        }
    }

    fn h(&mut self, target: usize) {
        let mask = 1usize << target;

        for i in 0..self.amplitudes.len() {
            if i & mask != 0 {
                continue;
            }

            let j = i | mask;
            let a = self.amplitudes[i];
            let b = self.amplitudes[j];

            self.amplitudes[i] = Amp {
                re: (a.re + b.re) * FRAC_1_SQRT_2,
                im: (a.im + b.im) * FRAC_1_SQRT_2,
            };

            self.amplitudes[j] = Amp {
                re: (a.re - b.re) * FRAC_1_SQRT_2,
                im: (a.im - b.im) * FRAC_1_SQRT_2,
            };
        }
    }

    fn cnot(&mut self, control: usize, target: usize) {
        let cm = 1usize << control;
        let tm = 1usize << target;

        for i in 0..self.amplitudes.len() {
            if i & cm != 0 && i & tm == 0 {
                self.amplitudes.swap(i, i | tm);
            }
        }
    }

    fn toffoli(&mut self, c1: usize, c2: usize, target: usize) {
        let m1 = 1usize << c1;
        let m2 = 1usize << c2;
        let tm = 1usize << target;

        for i in 0..self.amplitudes.len() {
            if i & m1 != 0 && i & m2 != 0 && i & tm == 0 {
                self.amplitudes.swap(i, i | tm);
            }
        }
    }

    fn phase_flip_when_all_one(&mut self, controls: &[usize]) {
        let mask = controls
            .iter()
            .fold(0usize, |acc, &q| acc | (1usize << q));

        for i in 0..self.amplitudes.len() {
            if i & mask == mask {
                self.amplitudes[i].re *= -1.0;
                self.amplitudes[i].im *= -1.0;
            }
        }
    }

    fn probability_nonce(&self, nonce: usize) -> f64 {
        let nonce_mask = 0xffusize;

        self.amplitudes
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i & nonce_mask) == nonce)
            .map(|(_, a)| a.norm_sqr())
            .sum()
    }

    fn total_probability(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_sqr()).sum()
    }

    fn winner_nonce(&self) -> (usize, f64) {
        (0..256)
            .map(|n| (n, self.probability_nonce(n)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }
}

/*
 * Reversible oracle.
 *
 * nonce qubits = q0..q7
 * ancilla       = q8
 *
 * Predicate:
 *
 *     low 3 nonce bits == 101
 *
 * AND upper five bits == 10101
 *
 * Therefore the circuit recognizes 0b10101101 = 173,
 * but Grover itself is never handed integer 173.
 *
 * We transform the desired bit pattern into |11111111>,
 * compute a reversible partial predicate into ancilla,
 * phase-mark only the satisfying branch, then uncompute.
 */
fn oracle(q: &mut QuantumRegister) {
    const ANC: usize = 8;

    /*
     * Desired pattern, q7..q0:
     *
     * 10101101
     *
     * Zero positions: q1, q4, q6
     *
     * X turns those required-zero controls into required-one controls.
     */
    q.x(1);
    q.x(4);
    q.x(6);

    /*
     * Reversibly compute q0 AND q1 into ancilla.
     */
    q.toffoli(0, 1, ANC);

    /*
     * Phase flip requires:
     *
     * ancilla = 1
     * q2..q7 = 1
     *
     * Together that means all original 8 predicate bits matched.
     */
    q.phase_flip_when_all_one(&[ANC, 2, 3, 4, 5, 6, 7]);

    /*
     * Uncompute ancilla.
     */
    q.toffoli(0, 1, ANC);

    /*
     * Undo temporary X gates.
     */
    q.x(6);
    q.x(4);
    q.x(1);
}

fn diffusion_nonce(q: &mut QuantumRegister) {
    /*
     * H^8
     */
    for bit in 0..8 {
        q.h(bit);
    }

    /*
     * X^8
     */
    for bit in 0..8 {
        q.x(bit);
    }

    /*
     * Phase flip |11111111>.
     *
     * Ancilla is NOT included and should have returned to |0>.
     */
    q.phase_flip_when_all_one(&[0, 1, 2, 3, 4, 5, 6, 7]);

    /*
     * X^8
     */
    for bit in 0..8 {
        q.x(bit);
    }

    /*
     * H^8
     */
    for bit in 0..8 {
        q.h(bit);
    }
}

fn main() {
    const NONCE_QUBITS: usize = 8;
    const TOTAL_QUBITS: usize = 9;
    const STATES: usize = 1 << NONCE_QUBITS;
    const ITERATIONS: usize = 12;

    println!("X1331 LIVE-12D — REVERSIBLE GROVER ORACLE");
    println!("==========================================");
    println!("nonce qubits : {NONCE_QUBITS}");
    println!("ancillas     : {}", TOTAL_QUBITS - NONCE_QUBITS);
    println!("total qubits : {TOTAL_QUBITS}");
    println!("nonce space  : {STATES}");
    println!("state vector : {}", 1usize << TOTAL_QUBITS);

    let mut q = QuantumRegister::basis_zero(TOTAL_QUBITS);

    // Superposition only over nonce register.
    for bit in 0..NONCE_QUBITS {
        q.h(bit);
    }

    println!("\nRunning Grover with circuit-computed oracle...");

    for iteration in 1..=ITERATIONS {
        oracle(&mut q);
        diffusion_nonce(&mut q);

        let (winner, probability) = q.winner_nonce();

        println!(
            "iteration {:02}: winner={:3} ({:08b}) P={:.9}",
            iteration,
            winner,
            winner,
            probability
        );
    }

    let (winner, probability) = q.winner_nonce();

    println!();
    println!("winner            : {winner}");
    println!("winner binary     : {winner:08b}");
    println!("winner probability: {:.9}", probability);
    println!("total probability : {:.12}", q.total_probability());

    assert!((q.total_probability() - 1.0).abs() < 1.0e-10);
    assert_eq!(winner, 0b10101101);

    println!();
    println!("PASS — reversible oracle discovered and amplified its matching nonce.");
}
