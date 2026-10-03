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
        let mask = 1usize << target;

        for i in 0..DIM {
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

    /*
     * General controlled-X.
     *
     * Flip target iff every control qubit is 1.
     *
     * controls.len() == 0 -> X
     * controls.len() == 1 -> CNOT
     * controls.len() == 2 -> Toffoli
     * controls.len() > 2  -> multi-controlled X
     */
    fn mcx(&mut self, controls: &[usize], target: usize) {
        let target_mask = 1usize << target;

        let control_mask = controls
            .iter()
            .fold(0usize, |m, &q| m | (1usize << q));

        for i in 0..DIM {
            if i & target_mask != 0 {
                continue;
            }

            if i & control_mask == control_mask {
                self.amplitudes.swap(
                    i,
                    i | target_mask,
                );
            }
        }
    }

    fn cnot(&mut self, control: usize, target: usize) {
        self.mcx(&[control], target);
    }

    /*
     * Controlled increment of the suffix:
     *
     *     B <- B + 2^start_bit  (mod 16)
     *
     * iff control == 1.
     *
     * Example start_bit=0:
     *
     * b3 ^= control & b2 & b1 & b0
     * b2 ^= control & b1 & b0
     * b1 ^= control & b0
     * b0 ^= control
     *
     * IMPORTANT:
     * execute from MSB toward LSB so controls observe
     * the ORIGINAL lower bits.
     */
    fn controlled_increment_b(
        &mut self,
        control: usize,
        start_bit: usize,
    ) {
        for target_bit in
            ((start_bit + 1)..BITS).rev()
        {
            let mut controls = Vec::new();

            controls.push(control);

            for lower in start_bit..target_bit {
                controls.push(B_START + lower);
            }

            self.mcx(
                &controls,
                B_START + target_bit,
            );
        }

        self.cnot(
            control,
            B_START + start_bit,
        );
    }

    /*
     * Exact inverse of controlled_increment_b.
     *
     * Reverse the gate order.
     */
    fn controlled_decrement_b(
        &mut self,
        control: usize,
        start_bit: usize,
    ) {
        self.cnot(
            control,
            B_START + start_bit,
        );

        for target_bit in
            (start_bit + 1)..BITS
        {
            let mut controls = Vec::new();

            controls.push(control);

            for lower in start_bit..target_bit {
                controls.push(B_START + lower);
            }

            self.mcx(
                &controls,
                B_START + target_bit,
            );
        }
    }

    /*
     * Gate-level modular addition:
     *
     * For each bit A_i:
     *
     * if A_i == 1:
     *      B += 2^i
     *
     * Therefore:
     *
     *      B += A mod 16
     *
     * There is NO host-side computation of A+B here.
     */
    fn add_a_into_b(&mut self) {
        for bit in 0..BITS {
            self.controlled_increment_b(
                A_START + bit,
                bit,
            );
        }
    }

    /*
     * Exact circuit inverse.
     */
    fn uncompute_add_a_into_b(&mut self) {
        for bit in (0..BITS).rev() {
            self.controlled_decrement_b(
                A_START + bit,
                bit,
            );
        }
    }

    fn probability_basis(
        &self,
        a: u8,
        b: u8,
    ) -> f64 {
        let index =
            ((a as usize) << A_START)
            | ((b as usize) << B_START);

        self.amplitudes[index].norm_sqr()
    }

    fn total_probability(&self) -> f64 {
        self.amplitudes
            .iter()
            .map(|a| a.norm_sqr())
            .sum()
    }
}

fn main() {
    println!(
        "X1331 LIVE-12G3 — GATE-LEVEL REVERSIBLE ADDER"
    );
    println!(
        "=============================================="
    );

    println!("A register   : 4 qubits");
    println!("B register   : 4 qubits");
    println!("total qubits : {TOTAL_QUBITS}");
    println!("state vector : {DIM}");

    println!();
    println!("Circuit operations:");
    println!("  CNOT");
    println!("  Toffoli / multi-controlled X");
    println!("  NO direct state-vector addition permutation");

    /*
     * TEST 1:
     * exhaustive forward computation.
     */
    println!();
    println!("TEST 1 — exhaustive gate-level addition");
    println!("----------------------------------------");

    let mut forward_ok = 0usize;
    let mut inverse_ok = 0usize;

    for a in 0u8..16 {
        for b in 0u8..16 {
            let mut q =
                QuantumRegister::basis(a, b);

            q.add_a_into_b();

            /*
             * Host arithmetic appears ONLY here,
             * as the independent expected result.
             */
            let expected =
                ((a as usize + b as usize) & 0x0f)
                    as u8;

            let p =
                q.probability_basis(a, expected);

            assert!(
                (p - 1.0).abs() < 1.0e-12,
                "forward failed: A={a} B={b}, \
                 expected={expected}, P={p}"
            );

            forward_ok += 1;

            q.uncompute_add_a_into_b();

            let restored =
                q.probability_basis(a, b);

            assert!(
                (restored - 1.0).abs()
                    < 1.0e-12,
                "inverse failed: A={a} B={b}"
            );

            inverse_ok += 1;
        }
    }

    println!(
        "forward correct   : {forward_ok}/256"
    );

    println!(
        "uncompute correct : {inverse_ok}/256"
    );

    /*
     * TEST 2:
     * coherent A superposition.
     */
    println!();
    println!("TEST 2 — coherent gate-level addition");
    println!("-------------------------------------");

    let mut q = QuantumRegister::zero();

    for bit in 0..BITS {
        q.h(A_START + bit);
    }

    q.add_a_into_b();

    let mut correlation = 0.0;

    for a in 0u8..16 {
        let p =
            q.probability_basis(a, a);

        correlation += p;

        println!(
            "A={:02} ({:04b}) -> \
             B={:02} ({:04b}) P={:.9}",
            a,
            a,
            a,
            a,
            p
        );
    }

    println!();
    println!(
        "P(B=A) = {:.12}",
        correlation
    );

    assert!(
        (correlation - 1.0).abs()
            < 1.0e-10
    );

    /*
     * TEST 3:
     * coherent inverse.
     */
    println!();
    println!("TEST 3 — gate-level uncompute");
    println!("-----------------------------");

    q.uncompute_add_a_into_b();

    let mut restored = 0.0;

    for a in 0u8..16 {
        restored +=
            q.probability_basis(a, 0);
    }

    println!(
        "P(B=0000)          = {:.12}",
        restored
    );

    println!(
        "total probability = {:.12}",
        q.total_probability()
    );

    assert!(
        (restored - 1.0).abs()
            < 1.0e-10
    );

    assert!(
        (q.total_probability() - 1.0).abs()
            < 1.0e-10
    );

    println!();
    println!(
        "PASS — modular addition was produced by reversible controlled gates."
    );
}
