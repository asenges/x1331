use std::f64::consts::FRAC_1_SQRT_2;

const NONCE_BITS: usize = 4;
const HASH_BITS: usize = 4;

const NONCE_START: usize = 0; // q0..q3
const HASH_START: usize = 4;  // q4..q7
const FLAG: usize = 8;        // q8

const TOTAL_QUBITS: usize = 9;
const NONCES: usize = 1 << NONCE_BITS;

// Reduced mining target.
const TARGET: u8 = 3;

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
    amplitudes: Vec<Amp>,
}

impl QuantumRegister {
    fn zero() -> Self {
        let mut amplitudes = vec![Amp::zero(); 1usize << TOTAL_QUBITS];

        amplitudes[0] = Amp {
            re: 1.0,
            im: 0.0,
        };

        Self { amplitudes }
    }

    fn x(&mut self, target: usize) {
        let tm = 1usize << target;

        for i in 0..self.amplitudes.len() {
            if i & tm == 0 {
                self.amplitudes.swap(i, i | tm);
            }
        }
    }

    fn h(&mut self, target: usize) {
        let tm = 1usize << target;

        for i in 0..self.amplitudes.len() {
            if i & tm != 0 {
                continue;
            }

            let j = i | tm;

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

    fn phase_flip_basis(&mut self, required_one: &[usize]) {
        let mask = required_one
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
        let nonce_mask = (1usize << NONCE_BITS) - 1;

        self.amplitudes
            .iter()
            .enumerate()
            .filter(|(i, _)| ((*i >> NONCE_START) & nonce_mask) == nonce)
            .map(|(_, a)| a.norm_sqr())
            .sum()
    }

    fn probability_dirty_workspace(&self) -> f64 {
        let hash_mask =
            ((1usize << HASH_BITS) - 1) << HASH_START;

        let flag_mask = 1usize << FLAG;

        self.amplitudes
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                (i & hash_mask) != 0 || (i & flag_mask) != 0
            })
            .map(|(_, a)| a.norm_sqr())
            .sum()
    }

    fn total_probability(&self) -> f64 {
        self.amplitudes
            .iter()
            .map(|a| a.norm_sqr())
            .sum()
    }
}

/*
 * H(x) = x XOR ROTL4(x,1) XOR ROTL4(x,2)
 *
 * Input q0..q3 is preserved.
 * Result is XORed into q4..q7.
 *
 * Since every operation is CNOT, applying this function
 * twice uncomputes it.
 */
fn reduced_hash(q: &mut QuantumRegister) {
    for out_bit in 0..HASH_BITS {
        let hq = HASH_START + out_bit;

        let x0 = NONCE_START + out_bit;
        let x1 = NONCE_START + ((out_bit + 3) % NONCE_BITS);
        let x2 = NONCE_START + ((out_bit + 2) % NONCE_BITS);

        q.cnot(x0, hq);
        q.cnot(x1, hq);
        q.cnot(x2, hq);
    }
}

/*
 * Phase predicate H(x) <= TARGET.
 *
 * For this first reduced experiment we implement the
 * comparison as reversible basis-pattern phase controls.
 *
 * Importantly, the host does NOT inspect the nonce amplitudes
 * to decide which nonce to mark.
 *
 * The predicate operates only on the computed hash register.
 */
fn phase_if_hash_le_target(q: &mut QuantumRegister) {
    for value in 0..=TARGET {
        /*
         * Convert desired hash bit pattern into all-ones.
         */
        for bit in 0..HASH_BITS {
            if ((value >> bit) & 1) == 0 {
                q.x(HASH_START + bit);
            }
        }

        /*
         * FLAG is temporarily turned on so the phase control
         * includes a dedicated predicate/phase qubit.
         */
        q.x(FLAG);

        let controls = [
            HASH_START,
            HASH_START + 1,
            HASH_START + 2,
            HASH_START + 3,
            FLAG,
        ];

        q.phase_flip_basis(&controls);

        q.x(FLAG);

        /*
         * Restore hash register representation.
         */
        for bit in (0..HASH_BITS).rev() {
            if ((value >> bit) & 1) == 0 {
                q.x(HASH_START + bit);
            }
        }
    }
}

/*
 * Complete compute → predicate → uncompute oracle.
 */
fn oracle(q: &mut QuantumRegister) {
    reduced_hash(q);

    phase_if_hash_le_target(q);

    // H is XOR-based, therefore H^-1 = H.
    reduced_hash(q);
}

fn diffusion_nonce(q: &mut QuantumRegister) {
    for bit in 0..NONCE_BITS {
        q.h(NONCE_START + bit);
    }

    for bit in 0..NONCE_BITS {
        q.x(NONCE_START + bit);
    }

    q.phase_flip_basis(&[
        NONCE_START,
        NONCE_START + 1,
        NONCE_START + 2,
        NONCE_START + 3,
    ]);

    for bit in (0..NONCE_BITS).rev() {
        q.x(NONCE_START + bit);
    }

    for bit in 0..NONCE_BITS {
        q.h(NONCE_START + bit);
    }
}

/*
 * Classical reference ONLY for validation/reporting.
 * It does not participate in the quantum oracle.
 */
fn reduced_hash_classical(x: u8) -> u8 {
    let x = x & 0x0f;

    let rot1 = ((x << 1) | (x >> 3)) & 0x0f;
    let rot2 = ((x << 2) | (x >> 2)) & 0x0f;

    x ^ rot1 ^ rot2
}

fn main() {
    println!("X1331 LIVE-12E — REDUCED HASH GROVER ORACLE");
    println!("============================================");
    println!("nonce qubits : {NONCE_BITS}");
    println!("hash qubits  : {HASH_BITS}");
    println!("flag qubits  : 1");
    println!("total qubits : {TOTAL_QUBITS}");
    println!("nonce space  : {NONCES}");
    println!("target       : H(x) <= {TARGET}");

    /*
     * Reference truth table.
     * This is printed for verification only.
     */
    println!("\nClassical reference:");
    let mut solutions = Vec::new();

    for x in 0..NONCES {
        let h = reduced_hash_classical(x as u8);

        let valid = h <= TARGET;

        println!(
            "x={:02} ({:04b}) -> H={:02} ({:04b}) {}",
            x,
            x,
            h,
            h,
            if valid { "<= TARGET" } else { "" }
        );

        if valid {
            solutions.push(x);
        }
    }

    println!("\nsolutions = {:?}", solutions);
    println!("M = {}", solutions.len());

    assert!(!solutions.is_empty());

    let n = NONCES as f64;
    let m = solutions.len() as f64;

    let theta = (m / n).sqrt().asin();

    let iterations =
        ((std::f64::consts::PI / (4.0 * theta)) - 0.5)
            .round()
            .max(1.0) as usize;

    println!("Grover iterations = {iterations}");

    let mut q = QuantumRegister::zero();

    /*
     * Superposition over nonce only.
     */
    for bit in 0..NONCE_BITS {
        q.h(bit);
    }

    println!(
        "\nworkspace dirty probability before oracle = {:.12}",
        q.probability_dirty_workspace()
    );

    for iteration in 1..=iterations {
        oracle(&mut q);

        let dirty_after_oracle =
            q.probability_dirty_workspace();

        println!(
            "iteration {:02}: dirty after oracle = {:.12}",
            iteration,
            dirty_after_oracle
        );

        /*
         * This is critical:
         * compute/predicate/uncompute must restore all workspace.
         */
        assert!(dirty_after_oracle < 1.0e-12);

        diffusion_nonce(&mut q);
    }

    println!("\nFinal nonce probabilities:");

    let mut solution_probability = 0.0;

    for x in 0..NONCES {
        let p = q.probability_nonce(x);

        let valid =
            reduced_hash_classical(x as u8) <= TARGET;

        if valid {
            solution_probability += p;
        }

        println!(
            "x={:02} ({:04b}) P={:.9} {}",
            x,
            x,
            p,
            if valid { "<-- VALID" } else { "" }
        );
    }

    println!();
    println!(
        "total valid probability : {:.9}",
        solution_probability
    );

    println!(
        "workspace dirty prob    : {:.12}",
        q.probability_dirty_workspace()
    );

    println!(
        "total probability       : {:.12}",
        q.total_probability()
    );

    assert!(
        q.probability_dirty_workspace() < 1.0e-12
    );

    assert!(
        (q.total_probability() - 1.0).abs() < 1.0e-10
    );

    println!();
    println!(
        "PASS — reduced reversible hash predicate amplified valid mining states."
    );
}
