use std::f64::consts::FRAC_1_SQRT_2;

const E: usize = 0;   // q0..q3
const F: usize = 4;   // q4..q7
const G: usize = 8;   // q8..q11
const T: usize = 12;  // q12..q15

const WORD: usize = 4;
const QUBITS: usize = 16;
const DIM: usize = 1 << QUBITS;

#[derive(Clone, Copy)]
struct Amp {
    re: f64,
    im: f64,
}

impl Amp {
    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    fn norm2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

#[derive(Default)]
struct GateCount {
    x: u64,
    h: u64,
    cnot: u64,
    toffoli: u64,
}

struct QState {
    a: Vec<Amp>,
    gates: GateCount,
}

impl QState {
    fn zero() -> Self {
        let mut a = vec![Amp::zero(); DIM];
        a[0].re = 1.0;

        Self {
            a,
            gates: GateCount::default(),
        }
    }

    fn x(&mut self, q: usize) {
        self.gates.x += 1;

        let m = 1usize << q;

        for i in 0..DIM {
            if i & m == 0 {
                self.a.swap(i, i | m);
            }
        }
    }

    fn h(&mut self, q: usize) {
        self.gates.h += 1;

        let m = 1usize << q;

        for i in 0..DIM {
            if i & m != 0 {
                continue;
            }

            let j = i | m;

            let a = self.a[i];
            let b = self.a[j];

            self.a[i] = Amp {
                re: (a.re + b.re) * FRAC_1_SQRT_2,
                im: (a.im + b.im) * FRAC_1_SQRT_2,
            };

            self.a[j] = Amp {
                re: (a.re - b.re) * FRAC_1_SQRT_2,
                im: (a.im - b.im) * FRAC_1_SQRT_2,
            };
        }
    }

    fn cnot(&mut self, c: usize, t: usize) {
        self.gates.cnot += 1;

        let cm = 1usize << c;
        let tm = 1usize << t;

        for i in 0..DIM {
            if i & cm != 0 && i & tm == 0 {
                self.a.swap(i, i | tm);
            }
        }
    }

    fn toffoli(&mut self, c1: usize, c2: usize, t: usize) {
        self.gates.toffoli += 1;

        let m1 = 1usize << c1;
        let m2 = 1usize << c2;
        let tm = 1usize << t;

        for i in 0..DIM {
            if i & m1 != 0
                && i & m2 != 0
                && i & tm == 0
            {
                self.a.swap(i, i | tm);
            }
        }
    }

    fn probability_target(
        &self,
        e: u8,
        f: u8,
        g: u8,
        target: u8,
    ) -> f64 {
        let mut p = 0.0;

        for (i, amp) in self.a.iter().enumerate() {
            let ee = ((i >> E) & 0x0f) as u8;
            let ff = ((i >> F) & 0x0f) as u8;
            let gg = ((i >> G) & 0x0f) as u8;
            let tt = ((i >> T) & 0x0f) as u8;

            if ee == e
                && ff == f
                && gg == g
                && tt == target
            {
                p += amp.norm2();
            }
        }

        p
    }

    fn probability_t_zero(&self) -> f64 {
        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| ((i >> T) & 0x0f) == 0)
            .map(|(_, a)| a.norm2())
            .sum()
    }

    fn norm(&self) -> f64 {
        self.a.iter().map(|x| x.norm2()).sum()
    }
}

/*
 * Ch(e,f,g) = (e & f) XOR (~e & g)
 *
 * XOR result into T.
 */
fn ch_into_t(q: &mut QState) {
    for bit in 0..WORD {
        q.toffoli(
            E + bit,
            F + bit,
            T + bit,
        );

        q.x(E + bit);

        q.toffoli(
            E + bit,
            G + bit,
            T + bit,
        );

        q.x(E + bit);
    }
}

/*
 * Reduced SHA-like:
 *
 * Sigma1(e) =
 * ROTR1(e) XOR ROTR2(e) XOR e
 *
 * XOR into T using CNOTs.
 */
fn sigma1_into_t(q: &mut QState) {
    for out in 0..WORD {
        /*
         * ROTR1 output[out] = input[(out+1)%4]
         * ROTR2 output[out] = input[(out+2)%4]
         * identity          = input[out]
         */
        q.cnot(
            E + ((out + 1) % WORD),
            T + out,
        );

        q.cnot(
            E + ((out + 2) % WORD),
            T + out,
        );

        q.cnot(
            E + out,
            T + out,
        );
    }
}

/*
 * Forward coherent SHA-like slice:
 *
 * T ^= Ch(E,F,G)
 * T ^= Sigma1(E)
 */
fn sha_slice(q: &mut QState) {
    ch_into_t(q);
    sigma1_into_t(q);
}

/*
 * Reverse gate order.
 *
 * Each X/CNOT/Toffoli is self-inverse.
 */
fn uncompute_sha_slice(q: &mut QState) {
    /*
     * Reverse Sigma1 CNOT sequence.
     */
    for out in (0..WORD).rev() {
        q.cnot(
            E + out,
            T + out,
        );

        q.cnot(
            E + ((out + 2) % WORD),
            T + out,
        );

        q.cnot(
            E + ((out + 1) % WORD),
            T + out,
        );
    }

    /*
     * Reverse Ch.
     */
    for bit in (0..WORD).rev() {
        q.x(E + bit);

        q.toffoli(
            E + bit,
            G + bit,
            T + bit,
        );

        q.x(E + bit);

        q.toffoli(
            E + bit,
            F + bit,
            T + bit,
        );
    }
}

fn rotl4(x: u8, n: u32) -> u8 {
    let n = n % 4;

    if n == 0 {
        x & 0x0f
    } else {
        ((x << n) | (x >> (4 - n))) & 0x0f
    }
}

fn rotr4(x: u8, n: u32) -> u8 {
    rotl4(x, 4 - (n % 4))
}

fn ch_ref(e: u8, f: u8, g: u8) -> u8 {
    ((e & f) ^ ((!e) & g)) & 0x0f
}

fn sigma1_ref(e: u8) -> u8 {
    (
        rotr4(e, 1)
        ^ rotr4(e, 2)
        ^ e
    ) & 0x0f
}

fn slice_ref(e: u8, f: u8, g: u8) -> u8 {
    ch_ref(e, f, g) ^ sigma1_ref(e)
}

fn set_register_basis(
    q: &mut QState,
    offset: usize,
    value: u8,
) {
    for bit in 0..WORD {
        if ((value >> bit) & 1) != 0 {
            q.x(offset + bit);
        }
    }
}

fn main() {
    println!("X1331 LIVE-13B — COHERENT SHA-LIKE SLICE");
    println!("==========================================");
    println!();

    println!("register E : q0..q3");
    println!("register F : q4..q7");
    println!("register G : q8..q11");
    println!("register T : q12..q15");
    println!("qubits     : {QUBITS}");
    println!("amplitudes : {DIM}");
    println!("operations : X + CNOT + Toffoli");
    println!("slice      : Ch(E,F,G) XOR Sigma1(E)");
    println!("NOT        : full SHA-256");
    println!();

    /*
     * Exhaustive computational-basis validation.
     *
     * 16^3 = 4096 input combinations.
     */
    let mut passed = 0usize;

    for e in 0u8..16 {
        for f in 0u8..16 {
            for g in 0u8..16 {
                let expected =
                    slice_ref(e, f, g);

                let mut q = QState::zero();

                set_register_basis(&mut q, E, e);
                set_register_basis(&mut q, F, f);
                set_register_basis(&mut q, G, g);

                sha_slice(&mut q);

                let p = q.probability_target(
                    e,
                    f,
                    g,
                    expected,
                );

                if (p - 1.0).abs() > 1e-10 {
                    panic!(
                        "forward mismatch e={e:x} f={f:x} g={g:x} expected={expected:x} p={p}"
                    );
                }

                uncompute_sha_slice(&mut q);

                if (q.probability_t_zero() - 1.0)
                    .abs()
                    > 1e-10
                {
                    panic!(
                        "uncompute mismatch e={e:x} f={f:x} g={g:x}"
                    );
                }

                passed += 1;
            }
        }
    }

    println!(
        "basis validation : {passed}/4096 PASS"
    );

    /*
     * Coherent test:
     *
     * E in uniform superposition.
     * F=1010
     * G=0101
     * T=0000
     */
    let mut q = QState::zero();

    let fixed_f = 0b1010u8;
    let fixed_g = 0b0101u8;

    set_register_basis(
        &mut q,
        F,
        fixed_f,
    );

    set_register_basis(
        &mut q,
        G,
        fixed_g,
    );

    for bit in 0..WORD {
        q.h(E + bit);
    }

    sha_slice(&mut q);

    let mut correct_probability = 0.0;

    for e in 0u8..16 {
        let expected =
            slice_ref(
                e,
                fixed_f,
                fixed_g,
            );

        correct_probability +=
            q.probability_target(
                e,
                fixed_f,
                fixed_g,
                expected,
            );
    }

    println!(
        "coherent P(correct) = {:.12}",
        correct_probability
    );

    println!(
        "norm after forward  = {:.12}",
        q.norm()
    );

    assert!(
        (correct_probability - 1.0).abs()
            < 1e-10
    );

    /*
     * Capture forward circuit gate count.
     */
    println!();
    println!("FORWARD GATE COUNT");
    println!("------------------");

    println!("X        : {}", q.gates.x);
    println!("H        : {}", q.gates.h);
    println!("CNOT     : {}", q.gates.cnot);
    println!("Toffoli  : {}", q.gates.toffoli);

    /*
     * Now uncompute.
     */
    uncompute_sha_slice(&mut q);

    println!();

    println!(
        "P(T=0000) after uncompute = {:.12}",
        q.probability_t_zero()
    );

    println!(
        "norm after uncompute      = {:.12}",
        q.norm()
    );

    assert!(
        (q.probability_t_zero() - 1.0).abs()
            < 1e-10
    );

    assert!(
        (q.norm() - 1.0).abs()
            < 1e-10
    );

    /*
     * Structural 32-bit extrapolation.
     *
     * Ch per bit:
     *   2 Toffoli
     *   2 X
     *
     * Sigma1 per output bit:
     *   3 CNOT
     *
     * This extrapolation is ONLY for this Boolean/sigma
     * slice, not an entire SHA-256 round.
     */
    let ch_toffoli_32 = 2 * 32;
    let ch_x_32 = 2 * 32;
    let sigma_cnot_32 = 3 * 32;

    println!();
    println!("32-BIT STRUCTURAL EXTRAPOLATION");
    println!("-------------------------------");

    println!(
        "Ch       : {} Toffoli + {} X",
        ch_toffoli_32,
        ch_x_32
    );

    println!(
        "Sigma1   : {} CNOT",
        sigma_cnot_32
    );

    println!();
    println!(
        "This is NOT a full SHA-256 round estimate:"
    );
    println!(
        "modular additions and remaining round state"
    );
    println!(
        "transformations are not included."
    );

    println!();
    println!("SUMMARY");
    println!("-------");
    println!("basis validation : PASS");
    println!("coherent circuit : PASS");
    println!("uncompute        : PASS");
    println!("gate counting    : PASS");
    println!("full SHA-256     : NO");

    println!();
    println!(
        "PASS — Ch + reduced Sigma1 executed coherently in one reversible state-vector circuit."
    );
}
