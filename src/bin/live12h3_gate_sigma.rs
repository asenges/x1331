use std::f64::consts::FRAC_1_SQRT_2;

const BITS: usize = 4;

const X_START: usize = 0; // q0..q3
const T_START: usize = 4; // q4..q7

const TOTAL_QUBITS: usize = 8;
const DIM: usize = 1 << TOTAL_QUBITS;

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
        let mut amplitudes = vec![Amp::zero(); DIM];

        amplitudes[0] = Amp {
            re: 1.0,
            im: 0.0,
        };

        Self { amplitudes }
    }

    fn basis(x: u8, t: u8) -> Self {
        let mut q = Self::zero();
        q.amplitudes.fill(Amp::zero());

        let index =
            ((x as usize) << X_START)
            | ((t as usize) << T_START);

        q.amplitudes[index] = Amp {
            re: 1.0,
            im: 0.0,
        };

        q
    }

    fn h(&mut self, target: usize) {
        let tm = 1usize << target;

        for i in 0..DIM {
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

        for i in 0..DIM {
            if i & cm != 0 && i & tm == 0 {
                self.amplitudes.swap(i, i | tm);
            }
        }
    }

    fn probability_basis(&self, x: u8, t: u8) -> f64 {
        let index =
            ((x as usize) << X_START)
            | ((t as usize) << T_START);

        self.amplitudes[index].norm_sqr()
    }

    fn probability_target_zero(&self) -> f64 {
        let mask =
            ((1usize << BITS) - 1) << T_START;

        self.amplitudes
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i & mask) == 0)
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
 * Bit mapping for ROTR on a 4-bit word.
 *
 * If output bit j receives input bit k for ROTR(x,r):
 *
 *     k = (j + r) mod 4
 *
 * We never physically rotate X.
 * We wire the corresponding X bit into target via CNOT.
 */
fn xor_rotr_into_target(
    q: &mut QuantumRegister,
    rotation: usize,
) {
    for out_bit in 0..BITS {
        let input_bit =
            (out_bit + rotation) % BITS;

        q.cnot(
            X_START + input_bit,
            T_START + out_bit,
        );
    }
}

/*
 * Reduced-width Σ0:
 *
 * T ^= ROTR(X,1)
 * T ^= ROTR(X,2)
 * T ^= ROTR(X,3)
 */
fn compute_sigma0(q: &mut QuantumRegister) {
    xor_rotr_into_target(q, 1);
    xor_rotr_into_target(q, 2);
    xor_rotr_into_target(q, 3);
}

/*
 * Reverse order = exact inverse.
 */
fn uncompute_sigma0(q: &mut QuantumRegister) {
    xor_rotr_into_target(q, 3);
    xor_rotr_into_target(q, 2);
    xor_rotr_into_target(q, 1);
}

/*
 * Reduced-width Σ1:
 *
 * T ^= ROTR(X,1)
 * T ^= ROTR(X,2)
 * T ^= X
 */
fn compute_sigma1(q: &mut QuantumRegister) {
    xor_rotr_into_target(q, 1);
    xor_rotr_into_target(q, 2);
    xor_rotr_into_target(q, 0);
}

fn uncompute_sigma1(q: &mut QuantumRegister) {
    xor_rotr_into_target(q, 0);
    xor_rotr_into_target(q, 2);
    xor_rotr_into_target(q, 1);
}

/*
 * Classical references used ONLY by tests.
 */
fn rotr4(x: u8, n: u32) -> u8 {
    let n = n % 4;

    if n == 0 {
        x & 0x0f
    } else {
        ((x >> n) | (x << (4 - n))) & 0x0f
    }
}

fn sigma0_reference(x: u8) -> u8 {
    rotr4(x, 1)
        ^ rotr4(x, 2)
        ^ rotr4(x, 3)
}

fn sigma1_reference(x: u8) -> u8 {
    rotr4(x, 1)
        ^ rotr4(x, 2)
        ^ x
}

fn main() {
    println!(
        "X1331 LIVE-12H3 — GATE-LEVEL SIGMA NETWORKS"
    );
    println!(
        "============================================="
    );

    println!("word width   : 4 qubits");
    println!("X register   : q0..q3");
    println!("T register   : q4..q7");
    println!("total qubits : {TOTAL_QUBITS}");
    println!("state vector : {DIM}");
    println!("gate         : CNOT");
    println!("claim        : reduced-width SHA-like Σ networks");

    /*
     * TEST 1 — Σ0 exhaustive.
     */
    println!();
    println!("TEST 1 — gate-level Σ0");
    println!("----------------------");

    let mut sigma0_ok = 0usize;
    let mut sigma0_inverse_ok = 0usize;

    for x in 0u8..16 {
        let expected = sigma0_reference(x);

        let mut q =
            QuantumRegister::basis(x, 0);

        compute_sigma0(&mut q);

        let p =
            q.probability_basis(x, expected);

        assert!(
            (p - 1.0).abs() < 1.0e-12,
            "Σ0 failed for x={x:04b}"
        );

        sigma0_ok += 1;

        uncompute_sigma0(&mut q);

        let restored =
            q.probability_basis(x, 0);

        assert!(
            (restored - 1.0).abs() < 1.0e-12,
            "Σ0 uncompute failed"
        );

        sigma0_inverse_ok += 1;
    }

    println!(
        "forward   : {sigma0_ok}/16"
    );

    println!(
        "uncompute : {sigma0_inverse_ok}/16"
    );

    /*
     * TEST 2 — Σ1 exhaustive.
     */
    println!();
    println!("TEST 2 — gate-level Σ1");
    println!("----------------------");

    let mut sigma1_ok = 0usize;
    let mut sigma1_inverse_ok = 0usize;

    for x in 0u8..16 {
        let expected = sigma1_reference(x);

        let mut q =
            QuantumRegister::basis(x, 0);

        compute_sigma1(&mut q);

        let p =
            q.probability_basis(x, expected);

        assert!(
            (p - 1.0).abs() < 1.0e-12,
            "Σ1 failed for x={x:04b}"
        );

        sigma1_ok += 1;

        uncompute_sigma1(&mut q);

        let restored =
            q.probability_basis(x, 0);

        assert!(
            (restored - 1.0).abs() < 1.0e-12,
            "Σ1 uncompute failed"
        );

        sigma1_inverse_ok += 1;
    }

    println!(
        "forward   : {sigma1_ok}/16"
    );

    println!(
        "uncompute : {sigma1_inverse_ok}/16"
    );

    /*
     * TEST 3 — coherent Σ0.
     */
    println!();
    println!("TEST 3 — coherent Σ0");
    println!("--------------------");

    let mut q = QuantumRegister::zero();

    for bit in 0..BITS {
        q.h(X_START + bit);
    }

    compute_sigma0(&mut q);

    let mut correct_probability = 0.0;

    for x in 0u8..16 {
        let expected =
            sigma0_reference(x);

        let p =
            q.probability_basis(x, expected);

        correct_probability += p;

        println!(
            "X={:04b} -> Σ0={:04b} P={:.9}",
            x,
            expected,
            p
        );
    }

    println!();
    println!(
        "P(correct Σ0) = {:.12}",
        correct_probability
    );

    assert!(
        (correct_probability - 1.0).abs()
            < 1.0e-10
    );

    uncompute_sigma0(&mut q);

    println!(
        "P(T=0000 after uncompute) = {:.12}",
        q.probability_target_zero()
    );

    assert!(
        (q.probability_target_zero() - 1.0).abs()
            < 1.0e-10
    );

    /*
     * TEST 4 — coherent Σ1.
     */
    println!();
    println!("TEST 4 — coherent Σ1");
    println!("--------------------");

    let mut q = QuantumRegister::zero();

    for bit in 0..BITS {
        q.h(X_START + bit);
    }

    compute_sigma1(&mut q);

    let mut correct_probability = 0.0;

    for x in 0u8..16 {
        let expected =
            sigma1_reference(x);

        correct_probability +=
            q.probability_basis(x, expected);
    }

    println!(
        "P(correct Σ1) = {:.12}",
        correct_probability
    );

    assert!(
        (correct_probability - 1.0).abs()
            < 1.0e-10
    );

    uncompute_sigma1(&mut q);

    println!(
        "P(T=0000 after uncompute) = {:.12}",
        q.probability_target_zero()
    );

    println!(
        "total probability         = {:.12}",
        q.total_probability()
    );

    assert!(
        (q.probability_target_zero() - 1.0).abs()
            < 1.0e-10
    );

    assert!(
        (q.total_probability() - 1.0).abs()
            < 1.0e-10
    );

    println!();
    println!("SUMMARY");
    println!("-------");
    println!("Σ0 : 16/16 + coherent + uncompute");
    println!("Σ1 : 16/16 + coherent + uncompute");

    println!();
    println!(
        "PASS — reduced Σ networks were computed entirely by reversible CNOT wiring."
    );
}
