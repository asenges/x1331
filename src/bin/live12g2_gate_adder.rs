use std::f64::consts::FRAC_1_SQRT_2;

const BITS: usize = 4;

const A_START: usize = 0; // q0..q3
const B_START: usize = 4; // q4..q7

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

    fn basis(a: u8, b: u8) -> Self {
        let mut q = Self::zero();

        q.amplitudes.fill(Amp::zero());

        let index =
            ((a as usize) << A_START)
            | ((b as usize) << B_START);

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

    /*
     * Reversible modular adder:
     *
     * |A>|B> -> |A>|B+A mod 16>
     *
     * IMPORTANT:
     * This acts directly as a permutation of the quantum
     * state vector. No measurement occurs.
     */
    fn add_a_into_b(&mut self) {
        let old = self.amplitudes.clone();
        let mut next = vec![Amp::zero(); DIM];

        for index in 0..DIM {
            let a = (index >> A_START) & 0x0f;
            let b = (index >> B_START) & 0x0f;

            /*
             * Explicit modular arithmetic without
             * wrapping_add().
             */
            let sum = (a + b) & 0x0f;

            let new_index =
                (a << A_START)
                | (sum << B_START);

            next[new_index] = old[index];
        }

        self.amplitudes = next;
    }

    /*
     * Exact inverse permutation:
     *
     * |A>|B> -> |A>|B-A mod 16>
     */
    fn sub_a_from_b(&mut self) {
        let old = self.amplitudes.clone();
        let mut next = vec![Amp::zero(); DIM];

        for index in 0..DIM {
            let a = (index >> A_START) & 0x0f;
            let b = (index >> B_START) & 0x0f;

            let difference =
                (b + 16 - a) & 0x0f;

            let new_index =
                (a << A_START)
                | (difference << B_START);

            next[new_index] = old[index];
        }

        self.amplitudes = next;
    }

    fn probability_basis(&self, a: u8, b: u8) -> f64 {
        let index =
            ((a as usize) << A_START)
            | ((b as usize) << B_START);

        self.amplitudes[index].norm_sqr()
    }

    fn total_probability(&self) -> f64 {
        self.amplitudes
            .iter()
            .map(|x| x.norm_sqr())
            .sum()
    }
}

fn main() {
    println!(
        "X1331 LIVE-12G2 — QUANTUM STATE-VECTOR MODULAR ADDER"
    );
    println!(
        "===================================================="
    );

    println!("A register    : 4 qubits");
    println!("B register    : 4 qubits");
    println!("total qubits  : {TOTAL_QUBITS}");
    println!("state vector  : {DIM}");
    println!("operation     : |A>|B> -> |A>|A+B mod 16>");

    /*
     * TEST 1
     *
     * Every computational basis state.
     */
    println!();
    println!("TEST 1 — exhaustive basis states");
    println!("--------------------------------");

    let mut forward_ok = 0usize;
    let mut inverse_ok = 0usize;

    for a in 0u8..16 {
        for b in 0u8..16 {
            let mut q = QuantumRegister::basis(a, b);

            q.add_a_into_b();

            let expected =
                ((a as usize + b as usize) & 0x0f) as u8;

            let p =
                q.probability_basis(a, expected);

            assert!(
                (p - 1.0).abs() < 1.0e-12,
                "forward failure A={a} B={b}"
            );

            forward_ok += 1;

            q.sub_a_from_b();

            let p_restored =
                q.probability_basis(a, b);

            assert!(
                (p_restored - 1.0).abs() < 1.0e-12,
                "inverse failure A={a} B={b}"
            );

            inverse_ok += 1;
        }
    }

    println!("forward correct   : {forward_ok}/256");
    println!("uncompute correct : {inverse_ok}/256");

    /*
     * TEST 2
     *
     * Put A into a genuine 16-state superposition while
     * keeping B=0.
     *
     * Then:
     *
     * Σ |A>|0>
     *
     * becomes:
     *
     * Σ |A>|A>
     */
    println!();
    println!("TEST 2 — coherent superposition");
    println!("-------------------------------");

    let mut q = QuantumRegister::zero();

    for bit in 0..BITS {
        q.h(A_START + bit);
    }

    for a in 0u8..16 {
        let p = q.probability_basis(a, 0);

        assert!(
            (p - 1.0 / 16.0).abs() < 1.0e-12
        );
    }

    q.add_a_into_b();

    let mut correlated_probability = 0.0;

    for a in 0u8..16 {
        let p = q.probability_basis(a, a);

        correlated_probability += p;

        println!(
            "A={:02} ({:04b})  B={:02} ({:04b})  P={:.9}",
            a,
            a,
            a,
            a,
            p
        );
    }

    println!(
        "\nP(B=A) after addition = {:.12}",
        correlated_probability
    );

    assert!(
        (correlated_probability - 1.0).abs()
            < 1.0e-10
    );

    /*
     * TEST 3
     *
     * Exact uncompute while state is in superposition.
     */
    println!();
    println!("TEST 3 — coherent uncompute");
    println!("---------------------------");

    q.sub_a_from_b();

    let mut restored_probability = 0.0;

    for a in 0u8..16 {
        restored_probability +=
            q.probability_basis(a, 0);
    }

    println!(
        "P(B=0000) after inverse = {:.12}",
        restored_probability
    );

    println!(
        "total probability       = {:.12}",
        q.total_probability()
    );

    assert!(
        (restored_probability - 1.0).abs()
            < 1.0e-10
    );

    assert!(
        (q.total_probability() - 1.0).abs()
            < 1.0e-10
    );

    println!();
    println!(
        "PASS — modular addition acted coherently and was exactly uncomputed."
    );
}
