use std::f64::consts::FRAC_1_SQRT_2;

const A: usize = 0;     // q0..q3
const B: usize = 4;     // q4..q7
const W0: usize = 8;    // clean ancilla
const W1: usize = 9;    // clean ancilla

const WORD: usize = 4;
const QUBITS: usize = 10;
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

    fn x_raw(&mut self, q: usize) {
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
            let x = self.a[i];
            let y = self.a[j];

            self.a[i] = Amp {
                re: (x.re + y.re) * FRAC_1_SQRT_2,
                im: (x.im + y.im) * FRAC_1_SQRT_2,
            };

            self.a[j] = Amp {
                re: (x.re - y.re) * FRAC_1_SQRT_2,
                im: (x.im - y.im) * FRAC_1_SQRT_2,
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

    fn probability_pair(&self, a: u8, b: u8) -> f64 {
        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                (((*i >> A) & 0x0f) as u8 == a)
                    && (((*i >> B) & 0x0f) as u8 == b)
            })
            .map(|(_, amp)| amp.norm2())
            .sum()
    }

    fn probability_b_zero(&self) -> f64 {
        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| ((*i >> B) & 0x0f) == 0)
            .map(|(_, amp)| amp.norm2())
            .sum()
    }

    fn probability_work_zero(&self) -> f64 {
        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                ((*i >> W0) & 1) == 0
                    && ((*i >> W1) & 1) == 0
            })
            .map(|(_, amp)| amp.norm2())
            .sum()
    }

    fn norm(&self) -> f64 {
        self.a.iter().map(|x| x.norm2()).sum()
    }
}

/*
 * Controlled X with 1..4 controls.
 *
 * Crucially, controls > 2 are decomposed using
 * Toffoli and CLEAN ancillas.
 */
fn controlled_x(
    q: &mut QState,
    controls: &[usize],
    target: usize,
) {
    match controls.len() {
        1 => {
            q.cnot(controls[0], target);
        }

        2 => {
            q.toffoli(
                controls[0],
                controls[1],
                target,
            );
        }

        3 => {
            /*
             * W0 = c0 & c1
             * target ^= W0 & c2
             * clean W0
             */
            q.toffoli(
                controls[0],
                controls[1],
                W0,
            );

            q.toffoli(
                W0,
                controls[2],
                target,
            );

            q.toffoli(
                controls[0],
                controls[1],
                W0,
            );
        }

        4 => {
            /*
             * W0 = c0 & c1
             * W1 = W0 & c2
             * target ^= W1 & c3
             * clean W1
             * clean W0
             */
            q.toffoli(
                controls[0],
                controls[1],
                W0,
            );

            q.toffoli(
                W0,
                controls[2],
                W1,
            );

            q.toffoli(
                W1,
                controls[3],
                target,
            );

            q.toffoli(
                W0,
                controls[2],
                W1,
            );

            q.toffoli(
                controls[0],
                controls[1],
                W0,
            );
        }

        _ => panic!(
            "unsupported control count {}",
            controls.len()
        ),
    }
}

/*
 * if control:
 *     B += 2^start mod 16
 */
fn controlled_increment(
    q: &mut QState,
    control: usize,
    start: usize,
) {
    for target_bit in ((start + 1)..WORD).rev() {
        let mut controls =
            Vec::with_capacity(target_bit - start + 1);

        controls.push(control);

        for lower in start..target_bit {
            controls.push(B + lower);
        }

        controlled_x(
            q,
            &controls,
            B + target_bit,
        );
    }

    q.cnot(
        control,
        B + start,
    );
}

/*
 * Exact inverse.
 */
fn controlled_decrement(
    q: &mut QState,
    control: usize,
    start: usize,
) {
    q.cnot(
        control,
        B + start,
    );

    for target_bit in (start + 1)..WORD {
        let mut controls =
            Vec::with_capacity(target_bit - start + 1);

        controls.push(control);

        for lower in start..target_bit {
            controls.push(B + lower);
        }

        controlled_x(
            q,
            &controls,
            B + target_bit,
        );
    }
}

fn add_a_into_b(q: &mut QState) {
    for bit in 0..WORD {
        controlled_increment(
            q,
            A + bit,
            bit,
        );
    }
}

fn uncompute_add(q: &mut QState) {
    for bit in (0..WORD).rev() {
        controlled_decrement(
            q,
            A + bit,
            bit,
        );
    }
}

fn set_basis_raw(
    q: &mut QState,
    offset: usize,
    value: u8,
) {
    for bit in 0..WORD {
        if ((value >> bit) & 1) != 0 {
            q.x_raw(offset + bit);
        }
    }
}

fn main() {
    println!(
        "X1331 LIVE-13D — DECOMPOSED MODULAR ADDER"
    );
    println!(
        "==========================================="
    );
    println!();

    println!("A register : q0..q3");
    println!("B register : q4..q7");
    println!("work       : q8..q9");
    println!("qubits     : {QUBITS}");
    println!("amplitudes : {DIM}");
    println!("operation  : B <- B + A mod 16");
    println!("adder gates: CNOT + Toffoli");
    println!("MCX(3/4)   : NONE as simulator primitive");
    println!("NOT        : full SHA-256");
    println!();

    /*
     * Exhaustive basis validation.
     */
    let mut forward_pass = 0usize;
    let mut inverse_pass = 0usize;

    for a in 0u8..16 {
        for b in 0u8..16 {
            let mut q = QState::zero();

            set_basis_raw(&mut q, A, a);
            set_basis_raw(&mut q, B, b);

            add_a_into_b(&mut q);

            let expected =
                b.wrapping_add(a) & 0x0f;

            let p =
                q.probability_pair(a, expected);

            if (p - 1.0).abs() > 1e-10 {
                panic!(
                    "forward mismatch A={a} B={b} expected={expected} p={p}"
                );
            }

            if (q.probability_work_zero() - 1.0)
                .abs()
                > 1e-10
            {
                panic!(
                    "dirty workspace forward A={a} B={b}"
                );
            }

            forward_pass += 1;

            uncompute_add(&mut q);

            let p_back =
                q.probability_pair(a, b);

            if (p_back - 1.0).abs() > 1e-10 {
                panic!(
                    "inverse mismatch A={a} B={b} p={p_back}"
                );
            }

            if (q.probability_work_zero() - 1.0)
                .abs()
                > 1e-10
            {
                panic!(
                    "dirty workspace inverse A={a} B={b}"
                );
            }

            inverse_pass += 1;
        }
    }

    println!(
        "basis forward : {forward_pass}/256 PASS"
    );

    println!(
        "basis inverse : {inverse_pass}/256 PASS"
    );

    /*
     * Coherent test:
     *
     * A uniform, B=0, work=00.
     */
    let mut q = QState::zero();

    for bit in 0..WORD {
        q.h(A + bit);
    }

    add_a_into_b(&mut q);

    let mut correlated = 0.0;

    for a in 0u8..16 {
        correlated +=
            q.probability_pair(a, a);
    }

    println!();

    println!(
        "coherent P(B=A) = {:.12}",
        correlated
    );

    println!(
        "work P(00)      = {:.12}",
        q.probability_work_zero()
    );

    println!(
        "forward norm    = {:.12}",
        q.norm()
    );

    assert!(
        (correlated - 1.0).abs()
            < 1e-10
    );

    assert!(
        (q.probability_work_zero() - 1.0).abs()
            < 1e-10
    );

    println!();
    println!("FORWARD ADDER GATE COUNT");
    println!("------------------------");

    println!(
        "H preparation : {}",
        q.gates.h
    );

    println!(
        "CNOT          : {}",
        q.gates.cnot
    );

    println!(
        "Toffoli       : {}",
        q.gates.toffoli
    );

    /*
     * Expected forward network:
     *
     * CNOT = 4
     *
     * original two-control operations:
     *   3 direct Toffoli
     *
     * 2 x MCX3 -> 2 * 3 = 6 Toffoli
     * 1 x MCX4 -> 5 Toffoli
     *
     * total = 14 Toffoli
     */
    assert_eq!(q.gates.cnot, 4);
    assert_eq!(q.gates.toffoli, 14);

    println!();
    println!(
        "expected CNOT    : 4"
    );
    println!(
        "expected Toffoli : 14"
    );

    /*
     * Uncompute.
     */
    uncompute_add(&mut q);

    println!();

    println!(
        "P(B=0000) after uncompute = {:.12}",
        q.probability_b_zero()
    );

    println!(
        "work P(00) after uncompute = {:.12}",
        q.probability_work_zero()
    );

    println!(
        "norm after uncompute       = {:.12}",
        q.norm()
    );

    assert!(
        (q.probability_b_zero() - 1.0).abs()
            < 1e-10
    );

    assert!(
        (q.probability_work_zero() - 1.0).abs()
            < 1e-10
    );

    assert!(
        (q.norm() - 1.0).abs()
            < 1e-10
    );

    println!();
    println!("RESULT");
    println!("------");

    println!(
        "generalized MCX primitive : ELIMINATED"
    );

    println!(
        "clean ancillas            : 2"
    );

    println!(
        "forward adder CNOT        : 4"
    );

    println!(
        "forward adder Toffoli     : 14"
    );

    println!(
        "workspace after operation : CLEAN"
    );

    println!(
        "workspace after inverse   : CLEAN"
    );

    println!();

    println!("SUMMARY");
    println!("-------");
    println!("basis forward       : PASS");
    println!("basis inverse       : PASS");
    println!("coherent addition   : PASS");
    println!("MCX decomposition   : PASS");
    println!("ancilla cleanup     : PASS");
    println!("full SHA-256        : NO");

    println!();

    println!(
        "PASS — reduced modular adder decomposed to CNOT + Toffoli with clean ancillas."
    );
}
