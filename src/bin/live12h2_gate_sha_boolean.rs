use std::f64::consts::FRAC_1_SQRT_2;

const BITS: usize = 4;

const X_START: usize = 0;  // q0..q3
const Y_START: usize = 4;  // q4..q7
const Z_START: usize = 8;  // q8..q11
const T_START: usize = 12; // q12..q15

const TOTAL_QUBITS: usize = 16;
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

    fn basis(x: u8, y: u8, z: u8, t: u8) -> Self {
        let mut q = Self::zero();
        q.amplitudes.fill(Amp::zero());

        let index =
            ((x as usize) << X_START)
            | ((y as usize) << Y_START)
            | ((z as usize) << Z_START)
            | ((t as usize) << T_START);

        q.amplitudes[index] = Amp {
            re: 1.0,
            im: 0.0,
        };

        q
    }

    fn x(&mut self, target: usize) {
        let tm = 1usize << target;

        for i in 0..DIM {
            if i & tm == 0 {
                self.amplitudes.swap(i, i | tm);
            }
        }
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

    fn toffoli(
        &mut self,
        control1: usize,
        control2: usize,
        target: usize,
    ) {
        let c1 = 1usize << control1;
        let c2 = 1usize << control2;
        let tm = 1usize << target;

        for i in 0..DIM {
            if i & tm != 0 {
                continue;
            }

            if i & c1 != 0 && i & c2 != 0 {
                self.amplitudes.swap(i, i | tm);
            }
        }
    }

    fn probability_basis(
        &self,
        x: u8,
        y: u8,
        z: u8,
        t: u8,
    ) -> f64 {
        let index =
            ((x as usize) << X_START)
            | ((y as usize) << Y_START)
            | ((z as usize) << Z_START)
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
 * Gate-level:
 *
 * Ch(x,y,z) = (x & y) ^ (!x & z)
 *
 * For each bit:
 *
 * t ^= x & y
 * t ^= !x & z
 *
 * Negative control !x is implemented:
 *
 * X(x)
 * Toffoli(x,z,t)
 * X(x)
 */
fn compute_ch(q: &mut QuantumRegister) {
    for bit in 0..BITS {
        let x = X_START + bit;
        let y = Y_START + bit;
        let z = Z_START + bit;
        let t = T_START + bit;

        q.toffoli(x, y, t);

        q.x(x);
        q.toffoli(x, z, t);
        q.x(x);
    }
}

/*
 * Because every constituent gate is self-inverse,
 * reverse gate order gives exact uncompute.
 */
fn uncompute_ch(q: &mut QuantumRegister) {
    for bit in (0..BITS).rev() {
        let x = X_START + bit;
        let y = Y_START + bit;
        let z = Z_START + bit;
        let t = T_START + bit;

        q.x(x);
        q.toffoli(x, z, t);
        q.x(x);

        q.toffoli(x, y, t);
    }
}

/*
 * Gate-level:
 *
 * Maj(x,y,z) =
 *     (x & y) ^ (x & z) ^ (y & z)
 */
fn compute_maj(q: &mut QuantumRegister) {
    for bit in 0..BITS {
        let x = X_START + bit;
        let y = Y_START + bit;
        let z = Z_START + bit;
        let t = T_START + bit;

        q.toffoli(x, y, t);
        q.toffoli(x, z, t);
        q.toffoli(y, z, t);
    }
}

fn uncompute_maj(q: &mut QuantumRegister) {
    for bit in (0..BITS).rev() {
        let x = X_START + bit;
        let y = Y_START + bit;
        let z = Z_START + bit;
        let t = T_START + bit;

        q.toffoli(y, z, t);
        q.toffoli(x, z, t);
        q.toffoli(x, y, t);
    }
}

/*
 * Classical references ONLY for verification.
 */
fn ch_reference(x: u8, y: u8, z: u8) -> u8 {
    ((x & y) ^ ((!x & 0x0f) & z)) & 0x0f
}

fn maj_reference(x: u8, y: u8, z: u8) -> u8 {
    ((x & y) ^ (x & z) ^ (y & z)) & 0x0f
}

fn main() {
    println!(
        "X1331 LIVE-12H2 — GATE-LEVEL SHA BOOLEAN PRIMITIVES"
    );
    println!(
        "==================================================="
    );

    println!("word width   : 4 qubits");
    println!("registers    : X,Y,Z,T");
    println!("total qubits : {TOTAL_QUBITS}");
    println!("state vector : {DIM}");
    println!("gates        : X + Toffoli");
    println!("claim        : reduced-width SHA Boolean primitives");

    /*
     * TEST 1 — Ch exhaustive.
     *
     * 16^3 = 4096 inputs.
     */
    println!();
    println!("TEST 1 — gate-level Ch");
    println!("----------------------");

    let mut ch_ok = 0usize;
    let mut ch_uncompute_ok = 0usize;

    for x in 0u8..16 {
        for y in 0u8..16 {
            for z in 0u8..16 {
                let expected =
                    ch_reference(x, y, z);

                let mut q =
                    QuantumRegister::basis(x, y, z, 0);

                compute_ch(&mut q);

                let p = q.probability_basis(
                    x,
                    y,
                    z,
                    expected,
                );

                assert!(
                    (p - 1.0).abs() < 1.0e-12,
                    "Ch failure x={x} y={y} z={z}"
                );

                ch_ok += 1;

                uncompute_ch(&mut q);

                let restored =
                    q.probability_basis(x, y, z, 0);

                assert!(
                    (restored - 1.0).abs() < 1.0e-12,
                    "Ch uncompute failure"
                );

                ch_uncompute_ok += 1;
            }
        }
    }

    println!("forward   : {ch_ok}/4096");
    println!("uncompute : {ch_uncompute_ok}/4096");

    /*
     * TEST 2 — Maj exhaustive.
     */
    println!();
    println!("TEST 2 — gate-level Maj");
    println!("-----------------------");

    let mut maj_ok = 0usize;
    let mut maj_uncompute_ok = 0usize;

    for x in 0u8..16 {
        for y in 0u8..16 {
            for z in 0u8..16 {
                let expected =
                    maj_reference(x, y, z);

                let mut q =
                    QuantumRegister::basis(x, y, z, 0);

                compute_maj(&mut q);

                let p = q.probability_basis(
                    x,
                    y,
                    z,
                    expected,
                );

                assert!(
                    (p - 1.0).abs() < 1.0e-12,
                    "Maj failure x={x} y={y} z={z}"
                );

                maj_ok += 1;

                uncompute_maj(&mut q);

                let restored =
                    q.probability_basis(x, y, z, 0);

                assert!(
                    (restored - 1.0).abs() < 1.0e-12,
                    "Maj uncompute failure"
                );

                maj_uncompute_ok += 1;
            }
        }
    }

    println!("forward   : {maj_ok}/4096");
    println!("uncompute : {maj_uncompute_ok}/4096");

    /*
     * TEST 3 — coherent Ch.
     *
     * Put X in all 16 values simultaneously.
     * Keep:
     *
     * Y = 1010
     * Z = 0101
     * T = 0000
     */
    println!();
    println!("TEST 3 — coherent Ch superposition");
    println!("----------------------------------");

    let y = 0b1010u8;
    let z = 0b0101u8;

    let mut q =
        QuantumRegister::basis(0, y, z, 0);

    for bit in 0..BITS {
        q.h(X_START + bit);
    }

    compute_ch(&mut q);

    let mut correct_probability = 0.0;

    for x in 0u8..16 {
        let expected =
            ch_reference(x, y, z);

        let p =
            q.probability_basis(
                x,
                y,
                z,
                expected,
            );

        correct_probability += p;

        println!(
            "X={:04b} -> Ch={:04b} P={:.9}",
            x,
            expected,
            p
        );
    }

    println!();
    println!(
        "P(correct Ch) = {:.12}",
        correct_probability
    );

    assert!(
        (correct_probability - 1.0).abs()
            < 1.0e-10
    );

    /*
     * Coherent uncompute.
     */
    uncompute_ch(&mut q);

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

    /*
     * TEST 4 — coherent Maj.
     */
    println!();
    println!("TEST 4 — coherent Maj superposition");
    println!("-----------------------------------");

    let mut q =
        QuantumRegister::basis(0, y, z, 0);

    for bit in 0..BITS {
        q.h(X_START + bit);
    }

    compute_maj(&mut q);

    let mut correct_probability = 0.0;

    for x in 0u8..16 {
        let expected =
            maj_reference(x, y, z);

        correct_probability +=
            q.probability_basis(
                x,
                y,
                z,
                expected,
            );
    }

    println!(
        "P(correct Maj) = {:.12}",
        correct_probability
    );

    assert!(
        (correct_probability - 1.0).abs()
            < 1.0e-10
    );

    uncompute_maj(&mut q);

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
    println!("Ch  : 4096/4096 + coherent + uncompute");
    println!("Maj : 4096/4096 + coherent + uncompute");

    println!();
    println!(
        "PASS — Ch and Maj were computed by reversible X/Toffoli gates."
    );
}
