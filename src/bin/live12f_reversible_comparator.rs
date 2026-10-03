use std::f64::consts::FRAC_1_SQRT_2;

const NONCE_BITS: usize = 4;
const HASH_BITS: usize = 4;

const NONCE_START: usize = 0; // q0..q3
const HASH_START: usize = 4;  // q4..q7

const TOTAL_QUBITS: usize = 8;
const NONCES: usize = 1 << NONCE_BITS;

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
        let mut amplitudes =
            vec![Amp::zero(); 1usize << TOTAL_QUBITS];

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

    /*
     * General multi-controlled phase.
     *
     * Each tuple is:
     *
     *     (qubit, required_value)
     *
     * This is a reversible diagonal operation. It does not
     * enumerate valid hash values; it tests a Boolean condition
     * directly against each computational basis state.
     */
    fn phase_if(&mut self, controls: &[(usize, bool)]) {
        for i in 0..self.amplitudes.len() {
            let matches = controls.iter().all(|&(q, required)| {
                let bit = ((i >> q) & 1) != 0;
                bit == required
            });

            if matches {
                self.amplitudes[i].re *= -1.0;
                self.amplitudes[i].im *= -1.0;
            }
        }
    }

    fn probability_nonce(&self, nonce: usize) -> f64 {
        let mask = (1usize << NONCE_BITS) - 1;

        self.amplitudes
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i & mask) == nonce)
            .map(|(_, a)| a.norm_sqr())
            .sum()
    }

    fn probability_dirty_hash(&self) -> f64 {
        let mask =
            ((1usize << HASH_BITS) - 1) << HASH_START;

        self.amplitudes
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i & mask) != 0)
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
 * Same reversible reduced hash as LIVE-12E:
 *
 * H(x) = x XOR ROTL4(x,1) XOR ROTL4(x,2)
 *
 * |x>|0000> -> |x>|H(x)>
 *
 * Since it consists only of CNOTs:
 *
 * H^-1 = H
 */
fn reduced_hash(q: &mut QuantumRegister) {
    for out_bit in 0..HASH_BITS {
        let hq = HASH_START + out_bit;

        let x0 = NONCE_START + out_bit;
        let x1 =
            NONCE_START + ((out_bit + 3) % NONCE_BITS);
        let x2 =
            NONCE_START + ((out_bit + 2) % NONCE_BITS);

        q.cnot(x0, hq);
        q.cnot(x1, hq);
        q.cnot(x2, hq);
    }
}

/*
 * Reversible phase comparator for:
 *
 *            hash <= TARGET
 *
 * TARGET is a compile-time classical constant.
 *
 * Instead of enumerating 0,1,2,...TARGET, we implement
 * unsigned lexicographic comparison from MSB to LSB.
 *
 * h < T iff at the first differing bit:
 *
 *             h_bit = 0
 *             T_bit = 1
 *
 * while every more-significant bit is equal.
 *
 * Equality h == T is also valid because predicate is <=.
 *
 * Each mutually-exclusive condition receives one phase flip.
 */
fn phase_if_hash_le_target(q: &mut QuantumRegister) {
    /*
     * Cases where hash < target.
     */
    for bit in (0..HASH_BITS).rev() {
        let target_bit = ((TARGET >> bit) & 1) != 0;

        /*
         * A first-difference "less" case can only occur
         * where target has a 1 and hash has a 0.
         */
        if !target_bit {
            continue;
        }

        let mut controls = Vec::new();

        /*
         * All more-significant bits must equal TARGET.
         */
        for higher in ((bit + 1)..HASH_BITS).rev() {
            let t =
                ((TARGET >> higher) & 1) != 0;

            controls.push((HASH_START + higher, t));
        }

        /*
         * First differing bit:
         *
         * hash = 0, target = 1
         */
        controls.push((HASH_START + bit, false));

        q.phase_if(&controls);
    }

    /*
     * Equality case: hash == TARGET.
     */
    let equality: Vec<(usize, bool)> =
        (0..HASH_BITS)
            .map(|bit| {
                (
                    HASH_START + bit,
                    ((TARGET >> bit) & 1) != 0,
                )
            })
            .collect();

    q.phase_if(&equality);
}

fn oracle(q: &mut QuantumRegister) {
    // compute
    reduced_hash(q);

    // predicate + phase
    phase_if_hash_le_target(q);

    // uncompute
    reduced_hash(q);
}

fn diffusion_nonce(q: &mut QuantumRegister) {
    for bit in 0..NONCE_BITS {
        q.h(bit);
    }

    for bit in 0..NONCE_BITS {
        q.x(bit);
    }

    let controls: Vec<(usize, bool)> =
        (0..NONCE_BITS)
            .map(|bit| (bit, true))
            .collect();

    q.phase_if(&controls);

    for bit in (0..NONCE_BITS).rev() {
        q.x(bit);
    }

    for bit in 0..NONCE_BITS {
        q.h(bit);
    }
}

/*
 * Classical reference only.
 * Never used by the oracle.
 */
fn reduced_hash_classical(x: u8) -> u8 {
    let x = x & 0x0f;

    let rot1 =
        ((x << 1) | (x >> 3)) & 0x0f;

    let rot2 =
        ((x << 2) | (x >> 2)) & 0x0f;

    x ^ rot1 ^ rot2
}

fn main() {
    println!(
        "X1331 LIVE-12F — REVERSIBLE TARGET COMPARATOR"
    );

    println!(
        "=============================================="
    );

    println!("nonce qubits : {NONCE_BITS}");
    println!("hash qubits  : {HASH_BITS}");
    println!("total qubits : {TOTAL_QUBITS}");
    println!("nonce space  : {NONCES}");
    println!("predicate    : H(x) <= {TARGET}");

    /*
     * Reference expected answers.
     */
    let solutions: Vec<usize> =
        (0..NONCES)
            .filter(|&x| {
                reduced_hash_classical(x as u8)
                    <= TARGET
            })
            .collect();

    println!("\nreference solutions = {:?}", solutions);

    let n = NONCES as f64;
    let m = solutions.len() as f64;

    assert!(m > 0.0);

    let theta = (m / n).sqrt().asin();

    let iterations =
        ((std::f64::consts::PI / (4.0 * theta))
            - 0.5)
            .round()
            .max(1.0) as usize;

    println!("M                 = {}", solutions.len());
    println!("Grover iterations = {iterations}");

    let mut q = QuantumRegister::zero();

    /*
     * Uniform nonce superposition.
     */
    for bit in 0..NONCE_BITS {
        q.h(bit);
    }

    println!(
        "\ndirty hash before = {:.12}",
        q.probability_dirty_hash()
    );

    for iteration in 1..=iterations {
        oracle(&mut q);

        let dirty =
            q.probability_dirty_hash();

        println!(
            "iteration {:02}: dirty after oracle = {:.12}",
            iteration,
            dirty
        );

        assert!(dirty < 1.0e-12);

        diffusion_nonce(&mut q);
    }

    println!("\nFinal probabilities:");

    let mut valid_probability = 0.0;

    for x in 0..NONCES {
        let p = q.probability_nonce(x);

        let hash =
            reduced_hash_classical(x as u8);

        let valid = hash <= TARGET;

        if valid {
            valid_probability += p;
        }

        println!(
            "x={:02} ({:04b}) H={:02} P={:.9} {}",
            x,
            x,
            hash,
            p,
            if valid { "<-- VALID" } else { "" }
        );
    }

    println!();

    println!(
        "valid probability : {:.9}",
        valid_probability
    );

    println!(
        "dirty hash        : {:.12}",
        q.probability_dirty_hash()
    );

    println!(
        "total probability : {:.12}",
        q.total_probability()
    );

    assert!(
        q.probability_dirty_hash() < 1.0e-12
    );

    assert!(
        (q.total_probability() - 1.0).abs()
            < 1.0e-10
    );

    assert!(
        valid_probability > 0.999999
    );

    println!();

    println!(
        "PASS — reversible comparator amplified H(x) <= TARGET."
    );
}
