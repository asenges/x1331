use std::f64::consts::FRAC_1_SQRT_2;

const NONCE: usize = 0;
const HASH: usize = 4;
const BITS: usize = 4;
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
     * Reversible diagonal Boolean phase primitive.
     *
     * controls = [(qubit, required_bit), ...]
     *
     * No nonce values and no hash values are enumerated.
     */
    fn phase_if(&mut self, controls: &[(usize, bool)]) {
        for i in 0..DIM {
            let matches = controls.iter().all(|&(q, required)| {
                let bit = ((i >> q) & 1) != 0;
                bit == required
            });

            if matches {
                self.a[i].re = -self.a[i].re;
                self.a[i].im = -self.a[i].im;
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

    fn norm(&self) -> f64 {
        self.a.iter().map(|x| x.norm2()).sum()
    }
}

/*
 * H(x) = x XOR ROTL4(x,1) XOR ROTL4(x,2)
 *
 * Reversible CNOT network.
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
 * Phase flip when HASH <= TARGET.
 *
 * Binary lexicographic comparison.
 *
 * For each TARGET bit that is 1:
 *
 *   all more-significant HASH bits equal TARGET
 *   AND current HASH bit = 0
 *
 * means HASH is already strictly smaller.
 *
 * Finally mark exact equality.
 *
 * These conditions are mutually exclusive.
 */
fn phase_hash_le_target(q: &mut QState) {
    /*
     * Strictly-less branches.
     */
    for bit in (0..BITS).rev() {
        let target_bit =
            ((TARGET >> bit) & 1) != 0;

        if !target_bit {
            continue;
        }

        let mut controls = Vec::new();

        /*
         * Higher bits must equal TARGET.
         */
        for higher in ((bit + 1)..BITS).rev() {
            let required =
                ((TARGET >> higher) & 1) != 0;

            controls.push((
                HASH + higher,
                required,
            ));
        }

        /*
         * Current hash bit is 0 while target is 1.
         */
        controls.push((HASH + bit, false));

        q.phase_if(&controls);
    }

    /*
     * Equality HASH == TARGET.
     */
    let mut equality = Vec::new();

    for bit in 0..BITS {
        equality.push((
            HASH + bit,
            ((TARGET >> bit) & 1) != 0,
        ));
    }

    q.phase_if(&equality);
}

fn oracle(q: &mut QState) {
    compute_hash(q);
    phase_hash_le_target(q);
    uncompute_hash(q);
}

fn diffusion(q: &mut QState) {
    for hash in 0usize..16 {
        let mut mean_re = 0.0;
        let mut mean_im = 0.0;

        for nonce in 0usize..16 {
            let i = nonce | (hash << HASH);

            mean_re += q.a[i].re;
            mean_im += q.a[i].im;
        }

        mean_re /= 16.0;
        mean_im /= 16.0;

        for nonce in 0usize..16 {
            let i = nonce | (hash << HASH);

            q.a[i].re =
                2.0 * mean_re - q.a[i].re;

            q.a[i].im =
                2.0 * mean_im - q.a[i].im;
        }
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
    println!("X1331 LIVE-12J — COMPARATOR MINING ORACLE");
    println!("==========================================");

    println!("nonce       : 4 qubits");
    println!("hash        : 4 qubits");
    println!("target      : {TARGET} ({TARGET:04b})");
    println!("state       : {DIM} amplitudes");
    println!("hash        : reversible CNOT network");
    println!("comparison  : binary HASH <= TARGET");
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

    println!(
        "M={} N=16",
        solutions.len()
    );

    /*
     * For M=4,N=16 one Grover iteration is exact.
     */
    let theta =
        ((solutions.len() as f64 / 16.0).sqrt())
            .asin();

    let iterations =
        (
            std::f64::consts::PI
                / (4.0 * theta)
        )
        .floor() as usize;

    println!(
        "Grover iterations   : {}",
        iterations
    );

    /*
     * Uniform nonce superposition.
     */
    let mut q = QState::zero();

    for bit in 0..BITS {
        q.h(NONCE + bit);
    }

    /*
     * Oracle cleanliness.
     */
    let mut test = QState {
        a: q.a.clone(),
    };

    oracle(&mut test);

    println!(
        "workspace after oracle = {:.12}",
        test.probability_hash_zero()
    );

    assert!(
        (test.probability_hash_zero() - 1.0).abs()
            < 1e-10
    );

    /*
     * Grover.
     */
    for r in 0..iterations {
        oracle(&mut q);
        diffusion(&mut q);

        let p: f64 =
            solutions
                .iter()
                .map(|&n| q.probability_nonce(n))
                .sum();

        println!(
            "iteration {} P(valid) = {:.12}",
            r + 1,
            p
        );
    }

    println!();
    println!("FINAL");
    println!("-----");

    let mut valid_probability = 0.0;
    let mut invalid_probability = 0.0;

    let mut winner = 0;
    let mut winner_p = -1.0;

    for nonce in 0u8..16 {
        let hash = hash_reference(nonce);
        let p = q.probability_nonce(nonce);

        if hash <= TARGET {
            valid_probability += p;
        } else {
            invalid_probability += p;
        }

        if p > winner_p {
            winner = nonce;
            winner_p = p;
        }

        println!(
            "nonce={:2} hash={:2} P={:.9} {}",
            nonce,
            hash,
            p,
            if hash <= TARGET {
                "VALID"
            } else {
                ""
            }
        );
    }

    println!();

    println!(
        "P(valid)       = {:.12}",
        valid_probability
    );

    println!(
        "P(invalid)     = {:.12}",
        invalid_probability
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
        "workspace      = {:.12}",
        q.probability_hash_zero()
    );

    println!(
        "norm           = {:.12}",
        q.norm()
    );

    assert!(solutions.contains(&winner));

    assert!(
        (valid_probability - 1.0).abs()
            < 1e-10
    );

    assert!(
        invalid_probability.abs()
            < 1e-10
    );

    assert!(
        (q.probability_hash_zero() - 1.0).abs()
            < 1e-10
    );

    assert!(
        (q.norm() - 1.0).abs()
            < 1e-10
    );

    println!();
    println!("SUMMARY");
    println!("-------");
    println!("nonce enumeration in oracle : NO");
    println!("hash-value enumeration       : NO");
    println!("hash compute                 : CNOT");
    println!("comparison                   : HASH <= TARGET");
    println!("hash uncompute               : PASS");
    println!("Grover                       : PASS");

    println!();
    println!(
        "PASS — reduced mining oracle selected solutions using hash computation + binary target comparison + uncompute + Grover."
    );
}
