use std::f64::consts::FRAC_1_SQRT_2;

const N: usize = 4;
const NONCE: usize = 0; // q0..q3
const HASH: usize = 4;  // q4..q7
const QUBITS: usize = 8;
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

    /*
     * Phase flip when HASH register equals a specific value.
     *
     * This operates only on the hash register.
     * It never inspects the nonce value.
     */
    fn phase_hash_equals(&mut self, value: u8) {
        let mask = 0x0fusize << HASH;
        let wanted = (value as usize) << HASH;

        for i in 0..DIM {
            if (i & mask) == wanted {
                self.a[i].re = -self.a[i].re;
                self.a[i].im = -self.a[i].im;
            }
        }
    }

    fn probability_nonce(&self, nonce: u8) -> f64 {
        let nm = 0x0fusize << NONCE;
        let wanted = (nonce as usize) << NONCE;

        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i & nm) == wanted)
            .map(|(_, a)| a.norm2())
            .sum()
    }

    fn probability_hash_zero(&self) -> f64 {
        let hm = 0x0fusize << HASH;

        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i & hm) == 0)
            .map(|(_, a)| a.norm2())
            .sum()
    }

    fn norm(&self) -> f64 {
        self.a.iter().map(|x| x.norm2()).sum()
    }
}

/*
 * Reduced reversible hash:
 *
 * H(x) = x XOR ROTL(x,1) XOR ROTL(x,2)
 *
 * HASH starts at zero.
 *
 * Everything is CNOT wiring.
 */
fn compute_hash(q: &mut QState) {
    for out in 0..N {
        let x0 = out;
        let x1 = (out + 3) % 4;
        let x2 = (out + 2) % 4;

        q.cnot(NONCE + x0, HASH + out);
        q.cnot(NONCE + x1, HASH + out);
        q.cnot(NONCE + x2, HASH + out);
    }
}

fn uncompute_hash(q: &mut QState) {
    for out in (0..N).rev() {
        let x0 = out;
        let x1 = (out + 3) % 4;
        let x2 = (out + 2) % 4;

        q.cnot(NONCE + x2, HASH + out);
        q.cnot(NONCE + x1, HASH + out);
        q.cnot(NONCE + x0, HASH + out);
    }
}

/*
 * Mining predicate:
 *
 * H(nonce) <= TARGET
 *
 * IMPORTANT:
 * We enumerate acceptable HASH values,
 * not nonce values.
 *
 * The oracle therefore does not know which
 * nonce is a solution.
 */
fn target_phase(q: &mut QState) {
    for hash in 0..=TARGET {
        q.phase_hash_equals(hash);
    }
}

fn oracle(q: &mut QState) {
    compute_hash(q);
    target_phase(q);
    uncompute_hash(q);
}

/*
 * Diffusion only over nonce register.
 *
 * Since hash workspace has been uncomputed to |0000>,
 * this is ordinary Grover diffusion on 16 nonce states.
 */
fn diffusion(q: &mut QState) {
    /*
     * For every fixed hash-workspace sector,
     * perform inversion about the mean over nonce.
     *
     * In the actual Grover path only HASH=0000
     * carries amplitude after uncompute.
     */
    for hash in 0usize..16 {
        let mut mean_re = 0.0;
        let mut mean_im = 0.0;

        for nonce in 0usize..16 {
            let i =
                (nonce << NONCE)
                | (hash << HASH);

            mean_re += q.a[i].re;
            mean_im += q.a[i].im;
        }

        mean_re /= 16.0;
        mean_im /= 16.0;

        for nonce in 0usize..16 {
            let i =
                (nonce << NONCE)
                | (hash << HASH);

            q.a[i].re =
                2.0 * mean_re - q.a[i].re;

            q.a[i].im =
                2.0 * mean_im - q.a[i].im;
        }
    }
}

/*
 * Classical reference ONLY for validation.
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
    println!("X1331 LIVE-12I — REDUCED MINING ORACLE");
    println!("========================================");

    println!("nonce width  : 4 qubits");
    println!("hash width   : 4 qubits");
    println!("total qubits : {QUBITS}");
    println!("state vector : {DIM}");
    println!("target       : H(nonce) <= {TARGET}");
    println!("hash circuit : CNOT reversible network");
    println!("claim        : reduced mining oracle");
    println!("NOT          : SHA-256");
    println!();

    /*
     * Classical reference tells us ONLY what
     * answer should exist for validation.
     */
    let mut solutions = Vec::new();

    for nonce in 0u8..16 {
        let h = hash_reference(nonce);

        println!(
            "nonce {:2} ({:04b}) -> hash {:2} ({:04b}) {}",
            nonce,
            nonce,
            h,
            h,
            if h <= TARGET { "<= TARGET" } else { "" }
        );

        if h <= TARGET {
            solutions.push(nonce);
        }
    }

    println!();
    println!("reference solutions : {:?}", solutions);

    let m = solutions.len();
    let n = 16usize;

    assert!(m > 0);
    assert!(m < n);

    /*
     * Optimal Grover iteration estimate.
     */
    let theta =
        ((m as f64 / n as f64).sqrt()).asin();

    let iterations =
        (
            std::f64::consts::PI
                / (4.0 * theta)
        )
        .floor() as usize;

    println!("solutions M         : {m}");
    println!("search space N      : {n}");
    println!("Grover iterations   : {iterations}");
    println!();

    /*
     * Prepare nonce uniform superposition.
     */
    let mut q = QState::zero();

    for bit in 0..N {
        q.h(NONCE + bit);
    }

    println!(
        "workspace before oracle P(hash=0000) = {:.12}",
        q.probability_hash_zero()
    );

    /*
     * Show oracle cleanliness independently.
     */
    let mut oracle_test = QState {
        a: q.a.clone(),
    };

    oracle(&mut oracle_test);

    println!(
        "workspace after oracle  P(hash=0000) = {:.12}",
        oracle_test.probability_hash_zero()
    );

    assert!(
        (oracle_test.probability_hash_zero() - 1.0).abs()
            < 1.0e-10
    );

    /*
     * Grover.
     */
    for r in 0..iterations {
        oracle(&mut q);
        diffusion(&mut q);

        let valid_p: f64 =
            solutions
                .iter()
                .map(|&x| q.probability_nonce(x))
                .sum();

        println!(
            "iteration {:2} : P(valid nonce) = {:.12}",
            r + 1,
            valid_p
        );
    }

    println!();
    println!("FINAL DISTRIBUTION");
    println!("------------------");

    let mut valid_probability = 0.0;
    let mut winner = 0u8;
    let mut winner_p = -1.0f64;

    for nonce in 0u8..16 {
        let p = q.probability_nonce(nonce);
        let h = hash_reference(nonce);
        let valid = h <= TARGET;

        if valid {
            valid_probability += p;
        }

        if p > winner_p {
            winner_p = p;
            winner = nonce;
        }

        println!(
            "nonce {:2} hash={:2} P={:.9} {}",
            nonce,
            h,
            p,
            if valid { "VALID" } else { "" }
        );
    }

    println!();
    println!(
        "P(valid nonce) = {:.12}",
        valid_probability
    );

    println!(
        "winner         = {} ({:04b})",
        winner,
        winner
    );

    println!(
        "winner hash    = {}",
        hash_reference(winner)
    );

    println!(
        "winner valid   = {}",
        hash_reference(winner) <= TARGET
    );

    println!(
        "workspace clean = {:.12}",
        q.probability_hash_zero()
    );

    println!(
        "norm            = {:.12}",
        q.norm()
    );

    assert!(
        solutions.contains(&winner),
        "Grover winner is not a valid mining solution"
    );

    assert!(
        (q.probability_hash_zero() - 1.0).abs()
            < 1.0e-10
    );

    assert!(
        (q.norm() - 1.0).abs()
            < 1.0e-10
    );

    println!();
    println!("SUMMARY");
    println!("-------");
    println!("hash compute     : reversible CNOT network");
    println!("target predicate : hash <= target");
    println!("uncompute        : PASS");
    println!("Grover           : PASS");
    println!("winner           : VALID");

    println!();
    println!(
        "PASS — reduced mining solutions were selected through compute → target phase → uncompute → Grover."
    );
}
