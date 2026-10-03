use std::f64::consts::FRAC_1_SQRT_2;

const NONCE: usize = 0; // q0..q3
const HASH: usize = 4;  // q4..q7
const PHASE: usize = 8; // q8 = |->
const W0: usize = 9;    // work
const W1: usize = 10;   // work
const W2: usize = 11;   // work / phase target helper

const BITS: usize = 4;
const QUBITS: usize = 12;
const DIM: usize = 1 << QUBITS;

const TARGET: u8 = 3;

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

struct QState {
    a: Vec<Amp>,
}

impl QState {
    fn zero() -> Self {
        let mut a = vec![Amp::zero(); DIM];
        a[0].re = 1.0;
        Self { a }
    }

    fn x(&mut self, q: usize) {
        let m = 1usize << q;

        for i in 0..DIM {
            if i & m == 0 {
                self.a.swap(i, i | m);
            }
        }
    }

    fn h(&mut self, q: usize) {
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
        let cm = 1usize << c;
        let tm = 1usize << t;

        for i in 0..DIM {
            if i & cm != 0 && i & tm == 0 {
                self.a.swap(i, i | tm);
            }
        }
    }

    fn toffoli(&mut self, c1: usize, c2: usize, t: usize) {
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

    fn probability_nonce(&self, nonce: u8) -> f64 {
        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| ((*i >> NONCE) & 0x0f) == nonce as usize)
            .map(|(_, a)| a.norm2())
            .sum()
    }

    fn probability_hash_zero(&self) -> f64 {
        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| ((*i >> HASH) & 0x0f) == 0)
            .map(|(_, a)| a.norm2())
            .sum()
    }

    fn probability_work_zero(&self) -> f64 {
        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let w =
                    (((*i >> W0) & 1) << 0)
                    | (((*i >> W1) & 1) << 1)
                    | (((*i >> W2) & 1) << 2);

                w == 0
            })
            .map(|(_, a)| a.norm2())
            .sum()
    }

    fn probability_phase_minus(&self) -> f64 {
        let m = 1usize << PHASE;
        let mut p = 0.0;

        for i in 0..DIM {
            if i & m != 0 {
                continue;
            }

            let j = i | m;

            let re =
                (self.a[i].re - self.a[j].re)
                    * FRAC_1_SQRT_2;

            let im =
                (self.a[i].im - self.a[j].im)
                    * FRAC_1_SQRT_2;

            p += re * re + im * im;
        }

        p
    }

    fn norm(&self) -> f64 {
        self.a.iter().map(|x| x.norm2()).sum()
    }
}

/*
 * Reduced reversible hash.
 */
fn compute_hash(q: &mut QState) {
    for out in 0..BITS {
        let x0 = out;
        let x1 = (out + 3) % 4;
        let x2 = (out + 2) % 4;

        q.cnot(NONCE + x0, HASH + out);
        q.cnot(NONCE + x1, HASH + out);
        q.cnot(NONCE + x2, HASH + out);
    }
}

fn uncompute_hash(q: &mut QState) {
    for out in (0..BITS).rev() {
        let x0 = out;
        let x1 = (out + 3) % 4;
        let x2 = (out + 2) % 4;

        q.cnot(NONCE + x2, HASH + out);
        q.cnot(NONCE + x1, HASH + out);
        q.cnot(NONCE + x0, HASH + out);
    }
}

/*
 * TARGET = 0011.
 *
 * HASH <= 0011 iff h3=0 AND h2=0.
 *
 * Phase kickback through PHASE = |->.
 */
fn mining_phase(q: &mut QState) {
    let h3 = HASH + 3;
    let h2 = HASH + 2;

    q.x(h3);
    q.x(h2);

    q.toffoli(h3, h2, PHASE);

    q.x(h2);
    q.x(h3);
}

fn oracle(q: &mut QState) {
    compute_hash(q);
    mining_phase(q);
    uncompute_hash(q);
}

/*
 * Four-controlled X on PHASE, decomposed using
 * W0 and W1.
 *
 * controls: c0,c1,c2,c3
 *
 * W0 = c0 AND c1
 * W1 = W0 AND c2
 * PHASE ^= W1 AND c3
 *
 * Then uncompute W1,W0.
 */
fn c4x_phase(
    q: &mut QState,
    c0: usize,
    c1: usize,
    c2: usize,
    c3: usize,
) {
    q.toffoli(c0, c1, W0);
    q.toffoli(W0, c2, W1);

    q.toffoli(W1, c3, PHASE);

    q.toffoli(W0, c2, W1);
    q.toffoli(c0, c1, W0);
}

/*
 * Gate-level Grover diffusion:
 *
 * D = H^n X^n (phase on |1111>) X^n H^n
 *
 * Global sign is irrelevant.
 *
 * PHASE remains |->, so c4x_phase generates
 * phase kickback on |1111>.
 */
fn gate_diffusion(q: &mut QState) {
    for bit in 0..BITS {
        q.h(NONCE + bit);
    }

    for bit in 0..BITS {
        q.x(NONCE + bit);
    }

    c4x_phase(
        q,
        NONCE + 0,
        NONCE + 1,
        NONCE + 2,
        NONCE + 3,
    );

    for bit in 0..BITS {
        q.x(NONCE + bit);
    }

    for bit in 0..BITS {
        q.h(NONCE + bit);
    }
}

/*
 * Classical validation only.
 */
fn rotl4(x: u8, n: u32) -> u8 {
    let n = n % 4;

    if n == 0 {
        x & 0x0f
    } else {
        ((x << n) | (x >> (4 - n))) & 0x0f
    }
}

fn hash_reference(x: u8) -> u8 {
    (
        x
        ^ rotl4(x, 1)
        ^ rotl4(x, 2)
    ) & 0x0f
}

fn main() {
    println!("X1331 LIVE-12L — GATE-LEVEL GROVER");
    println!("===================================");

    println!("nonce       : q0..q3");
    println!("hash        : q4..q7");
    println!("phase anc   : q8 = |->");
    println!("work        : q9..q11");
    println!("total       : {QUBITS} qubits");
    println!("state       : {DIM} amplitudes");
    println!("target      : {TARGET} ({TARGET:04b})");
    println!("oracle      : X/CNOT/Toffoli");
    println!("diffusion   : H/X/Toffoli");
    println!("mean shortcut: NONE");
    println!("phase_if     : NONE");
    println!("NOT          : SHA-256");
    println!();

    let solutions: Vec<u8> =
        (0u8..16)
            .filter(|&n| hash_reference(n) <= TARGET)
            .collect();

    println!("reference solutions : {:?}", solutions);

    assert_eq!(solutions, vec![0, 6, 11, 13]);

    /*
     * |nonce> = uniform
     * |hash>  = 0000
     * |phase> = |->
     * |work>  = 000
     */
    let mut q = QState::zero();

    for bit in 0..BITS {
        q.h(NONCE + bit);
    }

    q.x(PHASE);
    q.h(PHASE);

    println!(
        "initial hash workspace = {:.12}",
        q.probability_hash_zero()
    );

    println!(
        "initial phase |->      = {:.12}",
        q.probability_phase_minus()
    );

    println!(
        "initial work |000>     = {:.12}",
        q.probability_work_zero()
    );

    /*
     * One Grover iteration.
     *
     * N=16, M=4 -> exact.
     */
    oracle(&mut q);

    println!(
        "after oracle hash      = {:.12}",
        q.probability_hash_zero()
    );

    gate_diffusion(&mut q);

    println!(
        "after diffusion work   = {:.12}",
        q.probability_work_zero()
    );

    println!();
    println!("FINAL");
    println!("-----");

    let mut valid_p = 0.0;
    let mut invalid_p = 0.0;

    let mut winner = 0u8;
    let mut winner_p = -1.0;

    for nonce in 0u8..16 {
        let hash = hash_reference(nonce);
        let p = q.probability_nonce(nonce);
        let valid = hash <= TARGET;

        if valid {
            valid_p += p;
        } else {
            invalid_p += p;
        }

        if p > winner_p {
            winner_p = p;
            winner = nonce;
        }

        println!(
            "nonce={:2} hash={:2} P={:.9} {}",
            nonce,
            hash,
            p,
            if valid { "VALID" } else { "" }
        );
    }

    println!();

    println!("P(valid)       = {:.12}", valid_p);
    println!("P(invalid)     = {:.12}", invalid_p);

    println!("winner         = {}", winner);
    println!("winner hash    = {}", hash_reference(winner));

    println!(
        "hash workspace = {:.12}",
        q.probability_hash_zero()
    );

    println!(
        "phase anc |->  = {:.12}",
        q.probability_phase_minus()
    );

    println!(
        "work |000>     = {:.12}",
        q.probability_work_zero()
    );

    println!("norm            = {:.12}", q.norm());

    assert!(solutions.contains(&winner));

    assert!((valid_p - 1.0).abs() < 1e-10);
    assert!(invalid_p.abs() < 1e-10);

    assert!(
        (q.probability_hash_zero() - 1.0).abs()
            < 1e-10
    );

    assert!(
        (q.probability_phase_minus() - 1.0).abs()
            < 1e-10
    );

    assert!(
        (q.probability_work_zero() - 1.0).abs()
            < 1e-10
    );

    assert!((q.norm() - 1.0).abs() < 1e-10);

    println!();
    println!("SUMMARY");
    println!("-------");
    println!("nonce enumeration oracle : NO");
    println!("hash enumeration oracle  : NO");
    println!("phase_if                  : NO");
    println!("mean diffusion shortcut   : NO");
    println!("hash                       : CNOT");
    println!("target phase               : X + Toffoli + |->");
    println!("diffusion                  : H + X + Toffoli + |->");
    println!("work uncompute             : PASS");
    println!("Grover                     : PASS");

    println!();
    println!(
        "PASS — complete reduced Grover cycle executed through reversible gates."
    );
}
