use std::f64::consts::FRAC_1_SQRT_2;

const A: usize = 0; // q0..q3
const B: usize = 4; // q4..q7

const WORD: usize = 4;
const QUBITS: usize = 8;
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
    mcx3: u64,
    mcx4: u64,
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

    /*
     * Generalized controlled-X simulator primitive.
     *
     * IMPORTANT:
     * controls > 2 are not yet decomposed into
     * Toffoli + clean ancillas.
     */
    fn mcx(&mut self, controls: &[usize], target: usize) {
        match controls.len() {
            1 => self.gates.cnot += 1,
            2 => self.gates.toffoli += 1,
            3 => self.gates.mcx3 += 1,
            4 => self.gates.mcx4 += 1,
            _ => {}
        }

        let tm = 1usize << target;

        for i in 0..DIM {
            if i & tm != 0 {
                continue;
            }

            let enabled = controls
                .iter()
                .all(|&c| i & (1usize << c) != 0);

            if enabled {
                self.a.swap(i, i | tm);
            }
        }
    }

    fn cnot(&mut self, c: usize, t: usize) {
        self.mcx(&[c], t);
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

    fn norm(&self) -> f64 {
        self.a.iter().map(|x| x.norm2()).sum()
    }
}

/*
 * Controlled increment:
 *
 * if control == 1:
 *     B += 2^start mod 16
 *
 * Carry propagation is performed from high target
 * bits downward so lower B bits still contain their
 * original values when used as controls.
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

        q.mcx(
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
 * Exact inverse of controlled_increment.
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

        q.mcx(
            &controls,
            B + target_bit,
        );
    }
}

/*
 * B <- B + A mod 16
 */
fn add_a_into_b(q: &mut QState) {
    for bit in 0..WORD {
        controlled_increment(
            q,
            A + bit,
            bit,
        );
    }
}

/*
 * B <- B - A mod 16
 */
fn uncompute_add(q: &mut QState) {
    for bit in (0..WORD).rev() {
        controlled_decrement(
            q,
            A + bit,
            bit,
        );
    }
}

fn set_basis(
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
    println!("X1331 LIVE-13C — COHERENT MODULAR ADDER");
    println!("=========================================");
    println!();

    println!("A register : q0..q3");
    println!("B register : q4..q7");
    println!("qubits     : {QUBITS}");
    println!("amplitudes : {DIM}");
    println!("operation  : B <- B + A mod 16");
    println!("claim      : coherent reversible arithmetic");
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

            set_basis(&mut q, A, a);
            set_basis(&mut q, B, b);

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

            forward_pass += 1;

            uncompute_add(&mut q);

            let p_back =
                q.probability_pair(a, b);

            if (p_back - 1.0).abs() > 1e-10 {
                panic!(
                    "inverse mismatch A={a} B={b} p={p_back}"
                );
            }

            inverse_pass += 1;
        }
    }

    println!(
        "basis forward   : {forward_pass}/256 PASS"
    );

    println!(
        "basis inverse   : {inverse_pass}/256 PASS"
    );

    /*
     * Coherent test:
     *
     * A = uniform superposition
     * B = 0000
     *
     * After addition:
     *
     * sum_a |A=a>|B=a>
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
        "forward norm    = {:.12}",
        q.norm()
    );

    assert!(
        (correlated - 1.0).abs()
            < 1e-10
    );

    /*
     * Gate count for the coherent forward run.
     * H belongs to preparation, not the adder.
     */
    println!();
    println!("FORWARD NETWORK COUNT");
    println!("---------------------");

    println!("H prep   : {}", q.gates.h);
    println!("CNOT     : {}", q.gates.cnot);
    println!("Toffoli  : {}", q.gates.toffoli);
    println!("MCX(3)   : {}", q.gates.mcx3);
    println!("MCX(4)   : {}", q.gates.mcx4);

    println!();

    println!(
        "NOTE: MCX(3/4) are simulator primitives here;"
    );
    println!(
        "they are not yet decomposed into Toffoli + ancillas."
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
        "norm after uncompute      = {:.12}",
        q.norm()
    );

    assert!(
        (q.probability_b_zero() - 1.0).abs()
            < 1e-10
    );

    assert!(
        (q.norm() - 1.0).abs()
            < 1e-10
    );

    /*
     * Structural scaling information.
     */
    println!();
    println!("SHA BRIDGE");
    println!("----------");

    println!(
        "A SHA-256 round requires several modular"
    );
    println!(
        "32-bit additions for T1, T2 and state update."
    );

    println!();

    println!(
        "13C proves coherent modular arithmetic behavior"
    );
    println!(
        "at reduced width, but generalized MCX must be"
    );
    println!(
        "decomposed before making a hardware-level gate"
    );
    println!(
        "cost estimate."
    );

    println!();
    println!("SUMMARY");
    println!("-------");
    println!("basis forward       : PASS");
    println!("basis inverse       : PASS");
    println!("coherent addition   : PASS");
    println!("uncompute           : PASS");
    println!("MCX decomposition   : PENDING");
    println!("full SHA-256        : NO");

    println!();

    println!(
        "PASS — coherent modular addition validated for the reduced SHA arithmetic bridge."
    );
}
