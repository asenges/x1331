use std::f64::consts::FRAC_1_SQRT_2;

const NONCE: usize = 0; // q0..q3
const HASH: usize = 4;  // q4..q7
const ANC: usize = 8;   // q8

const BITS: usize = 4;
const QUBITS: usize = 9;
const DIM: usize = 1 << QUBITS;

const TARGET: u8 = 3; // 0011

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

    fn probability_anc_minus(&self) -> f64 {
        /*
         * Probability of ancilla being |->.
         *
         * For every basis state of the other 8 qubits:
         *
         * |-> = (|0> - |1>)/sqrt(2)
         */
        let m = 1usize << ANC;
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
 * Reduced reversible hash:
 *
 * H(x) = x XOR ROTL4(x,1) XOR ROTL4(x,2)
 *
 * HASH starts |0000>.
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
 * For a 4-bit unsigned HASH:
 *
 * HASH <= 0011
 *
 * iff:
 *
 * h3 = 0 AND h2 = 0
 *
 * h1,h0 are don't-care.
 *
 * Convert negative controls to positive:
 *
 * X(h3)
 * X(h2)
 *
 * Then Toffoli(h3,h2,ANC).
 *
 * ANC is |->, therefore controlled-X on ANC
 * produces phase kickback:
 *
 * X|-> = -|->
 *
 * Then restore h2,h3.
 *
 * This is a real X/Toffoli phase oracle at our
 * simulator gate abstraction. No phase_if().
 */
fn phase_hash_le_target(q: &mut QState) {
    assert_eq!(TARGET, 0b0011);

    let h3 = HASH + 3;
    let h2 = HASH + 2;

    q.x(h3);
    q.x(h2);

    q.toffoli(h3, h2, ANC);

    q.x(h2);
    q.x(h3);
}

fn oracle(q: &mut QState) {
    compute_hash(q);

    phase_hash_le_target(q);

    uncompute_hash(q);
}

/*
 * Grover diffusion over nonce only.
 *
 * We implement H^4 X^4 MCZ X^4 H^4.
 *
 * The MCZ for four nonce qubits is implemented
 * here using a direct four-control phase primitive
 * would reintroduce an abstraction, so instead for
 * this experiment we retain inversion-about-mean.
 *
 * K's claim concerns the MINING COMPARATOR oracle,
 * not decomposition of diffusion.
 */
fn diffusion(q: &mut QState) {
    /*
     * Each fixed HASH+ANC sector gets independent
     * inversion about the nonce mean.
     */
    for anc in 0usize..2 {
        for hash in 0usize..16 {
            let mut mean_re = 0.0;
            let mut mean_im = 0.0;

            for nonce in 0usize..16 {
                let i =
                    nonce
                    | (hash << HASH)
                    | (anc << ANC);

                mean_re += q.a[i].re;
                mean_im += q.a[i].im;
            }

            mean_re /= 16.0;
            mean_im /= 16.0;

            for nonce in 0usize..16 {
                let i =
                    nonce
                    | (hash << HASH)
                    | (anc << ANC);

                q.a[i].re =
                    2.0 * mean_re - q.a[i].re;

                q.a[i].im =
                    2.0 * mean_im - q.a[i].im;
            }
        }
    }
}

/*
 * Classical reference — validation only.
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
    println!(
        "X1331 LIVE-12K — GATE COMPARATOR MINING ORACLE"
    );
    println!(
        "==============================================="
    );

    println!("nonce       : q0..q3");
    println!("hash        : q4..q7");
    println!("phase anc   : q8 = |->");
    println!("total       : {QUBITS} qubits");
    println!("state       : {DIM} amplitudes");
    println!("target      : {TARGET} ({TARGET:04b})");
    println!("oracle gates: X + CNOT + Toffoli");
    println!("phase_if    : NONE");
    println!("NOT         : SHA-256");
    println!();

    let solutions: Vec<u8> =
        (0u8..16)
            .filter(|&nonce| {
                hash_reference(nonce) <= TARGET
            })
            .collect();

    println!(
        "reference solutions : {:?}",
        solutions
    );

    assert_eq!(
        solutions,
        vec![0, 6, 11, 13]
    );

    /*
     * Prepare:
     *
     * nonce = uniform
     * hash  = |0000>
     * anc   = |->
     */
    let mut q = QState::zero();

    for bit in 0..BITS {
        q.h(NONCE + bit);
    }

    /*
     * |0> --X--H--> |->
     */
    q.x(ANC);
    q.h(ANC);

    println!(
        "initial P(hash=0000) = {:.12}",
        q.probability_hash_zero()
    );

    println!(
        "initial P(anc=|->)   = {:.12}",
        q.probability_anc_minus()
    );

    /*
     * Test oracle separately.
     */
    let mut oracle_test = QState {
        a: q.a.clone(),
    };

    oracle(&mut oracle_test);

    println!(
        "after oracle hash workspace = {:.12}",
        oracle_test.probability_hash_zero()
    );

    println!(
        "after oracle ancilla |->    = {:.12}",
        oracle_test.probability_anc_minus()
    );

    assert!(
        (oracle_test.probability_hash_zero() - 1.0)
            .abs() < 1e-10
    );

    assert!(
        (oracle_test.probability_anc_minus() - 1.0)
            .abs() < 1e-10
    );

    /*
     * M=4, N=16 -> one exact Grover iteration.
     */
    oracle(&mut q);
    diffusion(&mut q);

    println!();
    println!("FINAL");
    println!("-----");

    let mut valid_p = 0.0;
    let mut invalid_p = 0.0;

    let mut winner = 0u8;
    let mut winner_p = -1.0;

    for nonce in 0u8..16 {
        let hash =
            hash_reference(nonce);

        let p =
            q.probability_nonce(nonce);

        let valid =
            hash <= TARGET;

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

    println!(
        "P(valid)       = {:.12}",
        valid_p
    );

    println!(
        "P(invalid)     = {:.12}",
        invalid_p
    );

    println!(
        "winner         = {}",
        winner
    );

    println!(
        "winner hash    = {}",
        hash_reference(winner)
    );

    println!(
        "hash workspace = {:.12}",
        q.probability_hash_zero()
    );

    println!(
        "phase anc |->  = {:.12}",
        q.probability_anc_minus()
    );

    println!(
        "norm            = {:.12}",
        q.norm()
    );

    assert!(
        solutions.contains(&winner)
    );

    assert!(
        (valid_p - 1.0).abs()
            < 1e-10
    );

    assert!(
        invalid_p.abs()
            < 1e-10
    );

    assert!(
        (q.probability_hash_zero() - 1.0)
            .abs() < 1e-10
    );

    assert!(
        (q.probability_anc_minus() - 1.0)
            .abs() < 1e-10
    );

    assert!(
        (q.norm() - 1.0).abs()
            < 1e-10
    );

    println!();
    println!("SUMMARY");
    println!("-------");
    println!("nonce enumeration in oracle : NO");
    println!("hash enumeration in oracle  : NO");
    println!("phase_if                     : NO");
    println!("hash circuit                 : CNOT");
    println!("target comparator            : X + Toffoli");
    println!("phase mechanism              : |-> kickback");
    println!("hash uncompute               : PASS");
    println!("Grover                       : PASS");

    println!();
    println!(
        "PASS — target phase was generated by reversible gates and phase kickback."
    );
}
