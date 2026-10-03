use std::f64::consts::FRAC_1_SQRT_2;

/*
 * LIVE-13E
 *
 * Reduced coherent SHA-like T1:
 *
 * T1 = H + Sigma1(E) + Ch(E,F,G) + K + W mod 4
 *
 * WORD = 2 bits deliberately, so the complete combined
 * circuit fits comfortably in the VPS state-vector simulator.
 *
 * Registers:
 * E   q0..q1
 * F   q2..q3
 * G   q4..q5
 * H   q6..q7
 * K   q8..q9
 * W   q10..q11
 * CH  q12..q13
 * SIG q14..q15
 * T   q16..q17
 *
 * 18 qubits = 262144 amplitudes.
 */

const WORD: usize = 2;

const E: usize = 0;
const F: usize = 2;
const G: usize = 4;
const HR: usize = 6;
const K: usize = 8;
const W: usize = 10;
const CH: usize = 12;
const SIG: usize = 14;
const T: usize = 16;

const QUBITS: usize = 18;
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
    x: u64,
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

    fn norm(&self) -> f64 {
        self.a.iter().map(|x| x.norm2()).sum()
    }

    fn probability_register_zero(&self, offset: usize) -> f64 {
        let mask = 0b11usize << offset;

        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i & mask) == 0)
            .map(|(_, x)| x.norm2())
            .sum()
    }
}

/*
 * Ch(E,F,G) into clean CH:
 *
 * CH ^= (E & F) ^ ((!E) & G)
 */
fn ch_into(q: &mut QState) {
    for bit in 0..WORD {
        q.toffoli(E + bit, F + bit, CH + bit);

        q.x(E + bit);
        q.toffoli(E + bit, G + bit, CH + bit);
        q.x(E + bit);
    }
}

fn uncompute_ch(q: &mut QState) {
    for bit in (0..WORD).rev() {
        q.x(E + bit);
        q.toffoli(E + bit, G + bit, CH + bit);
        q.x(E + bit);

        q.toffoli(E + bit, F + bit, CH + bit);
    }
}

/*
 * Reduced 2-bit Sigma1:
 *
 * SIG ^= E XOR ROTR1(E)
 *
 * This is deliberately SHA-like at reduced width.
 * It is NOT the literal 32-bit SHA-256 Sigma1.
 */
fn sigma1_into(q: &mut QState) {
    for out in 0..WORD {
        q.cnot(E + out, SIG + out);
        q.cnot(E + ((out + 1) % WORD), SIG + out);
    }
}

fn uncompute_sigma1(q: &mut QState) {
    for out in (0..WORD).rev() {
        q.cnot(E + ((out + 1) % WORD), SIG + out);
        q.cnot(E + out, SIG + out);
    }
}

/*
 * Exact 2-bit modular addition:
 *
 * dst <- dst + src mod 4
 *
 * Important order:
 * carry uses old dst bit 0 before bit 0 is changed.
 *
 * b1 ^= a0 & b0
 * b1 ^= a1
 * b0 ^= a0
 */
fn add2(q: &mut QState, src: usize, dst: usize) {
    q.toffoli(src, dst, dst + 1);
    q.cnot(src + 1, dst + 1);
    q.cnot(src, dst);
}

/*
 * Exact inverse: reverse gate order.
 */
fn unadd2(q: &mut QState, src: usize, dst: usize) {
    q.cnot(src, dst);
    q.cnot(src + 1, dst + 1);
    q.toffoli(src, dst, dst + 1);
}

fn compute_t1(q: &mut QState) {
    ch_into(q);
    sigma1_into(q);

    add2(q, HR, T);
    add2(q, SIG, T);
    add2(q, CH, T);
    add2(q, K, T);
    add2(q, W, T);
}

/*
 * Reverse entire T1 computation and clean CH/SIG.
 */
fn uncompute_t1(q: &mut QState) {
    unadd2(q, W, T);
    unadd2(q, K, T);
    unadd2(q, CH, T);
    unadd2(q, SIG, T);
    unadd2(q, HR, T);

    uncompute_sigma1(q);
    uncompute_ch(q);
}

fn set_word(q: &mut QState, offset: usize, value: u8) {
    for bit in 0..WORD {
        if ((value >> bit) & 1) != 0 {
            q.x(offset + bit);
        }
    }
}

fn word_from_index(i: usize, offset: usize) -> u8 {
    ((i >> offset) & 0b11) as u8
}

fn ch_ref(e: u8, f: u8, g: u8) -> u8 {
    ((e & f) ^ ((!e) & g)) & 0b11
}

fn sigma1_ref(e: u8) -> u8 {
    let e = e & 0b11;
    let r = ((e >> 1) | ((e & 1) << 1)) & 0b11;

    (e ^ r) & 0b11
}

fn t1_ref(
    e: u8,
    f: u8,
    g: u8,
    h: u8,
    k: u8,
    w: u8,
) -> u8 {
    h.wrapping_add(sigma1_ref(e))
        .wrapping_add(ch_ref(e, f, g))
        .wrapping_add(k)
        .wrapping_add(w)
        & 0b11
}

fn probability_correct_t1(q: &QState) -> f64 {
    let mut p = 0.0;

    for (i, amp) in q.a.iter().enumerate() {
        let e = word_from_index(i, E);
        let f = word_from_index(i, F);
        let g = word_from_index(i, G);
        let h = word_from_index(i, HR);
        let k = word_from_index(i, K);
        let w = word_from_index(i, W);
        let t = word_from_index(i, T);

        if t == t1_ref(e, f, g, h, k, w) {
            p += amp.norm2();
        }
    }

    p
}

fn main() {
    println!("X1331 LIVE-13E — COHERENT REDUCED T1");
    println!("======================================");
    println!();

    println!("word width : 2 bits");
    println!("qubits     : {QUBITS}");
    println!("amplitudes : {DIM}");
    println!("gates      : X + CNOT + Toffoli");
    println!();

    println!("T1 = H + Sigma1(E) + Ch(E,F,G) + K + W mod 4");
    println!("Sigma1     : reduced SHA-like 2-bit network");
    println!("NOT        : literal SHA-256");
    println!();

    /*
     * Fixed context:
     *
     * F=01
     * G=10
     * H=11
     * K=01
     * W=10
     *
     * E is placed in coherent uniform superposition.
     */
    let mut q = QState::zero();

    set_word(&mut q, F, 0b01);
    set_word(&mut q, G, 0b10);
    set_word(&mut q, HR, 0b11);
    set_word(&mut q, K, 0b01);
    set_word(&mut q, W, 0b10);

    for bit in 0..WORD {
        q.h(E + bit);
    }

    println!("input context:");
    println!("F = 01");
    println!("G = 10");
    println!("H = 11");
    println!("K = 01");
    println!("W = 10");
    println!("E = uniform superposition");
    println!();

    compute_t1(&mut q);

    let p_correct = probability_correct_t1(&q);

    println!(
        "coherent P(T = reduced T1) = {:.12}",
        p_correct
    );

    println!(
        "norm after forward         = {:.12}",
        q.norm()
    );

    println!(
        "CH workspace populated     = {:.12}",
        1.0 - q.probability_register_zero(CH)
    );

    println!(
        "SIG workspace populated    = {:.12}",
        1.0 - q.probability_register_zero(SIG)
    );

    assert!((p_correct - 1.0).abs() < 1e-10);
    assert!((q.norm() - 1.0).abs() < 1e-10);

    println!();
    println!("FORWARD CIRCUIT COUNT");
    println!("---------------------");
    println!("X        : {}", q.gates.x);
    println!("H        : {}", q.gates.h);
    println!("CNOT     : {}", q.gates.cnot);
    println!("Toffoli  : {}", q.gates.toffoli);

    println!();
    println!("E BRANCHES");
    println!("----------");

    for e in 0u8..4 {
        println!(
            "E={:02b} -> T1={:02b}",
            e,
            t1_ref(
                e,
                0b01,
                0b10,
                0b11,
                0b01,
                0b10
            )
        );
    }

    /*
     * Reverse everything.
     */
    uncompute_t1(&mut q);

    let p_t_zero = q.probability_register_zero(T);
    let p_ch_zero = q.probability_register_zero(CH);
    let p_sig_zero = q.probability_register_zero(SIG);

    println!();
    println!("UNCOMPUTE");
    println!("---------");

    println!(
        "P(T=00)   = {:.12}",
        p_t_zero
    );

    println!(
        "P(CH=00)  = {:.12}",
        p_ch_zero
    );

    println!(
        "P(SIG=00) = {:.12}",
        p_sig_zero
    );

    println!(
        "norm       = {:.12}",
        q.norm()
    );

    assert!((p_t_zero - 1.0).abs() < 1e-10);
    assert!((p_ch_zero - 1.0).abs() < 1e-10);
    assert!((p_sig_zero - 1.0).abs() < 1e-10);
    assert!((q.norm() - 1.0).abs() < 1e-10);

    println!();
    println!("SHA BRIDGE");
    println!("----------");
    println!("Ch               : coherent reversible");
    println!("reduced Sigma1   : coherent reversible");
    println!("modular additions: coherent reversible");
    println!("T1 assembly      : coherent reversible");
    println!("workspace cleanup: verified");
    println!("full SHA-256      : NO");

    println!();
    println!("SUMMARY");
    println!("-------");
    println!("reduced T1 forward : PASS");
    println!("coherent branches  : PASS");
    println!("uncompute           : PASS");
    println!("norm                : PASS");
    println!("full SHA-256        : NO");

    println!();
    println!(
        "PASS — reduced SHA-like T1 assembled coherently from reversible circuit components."
    );
}
