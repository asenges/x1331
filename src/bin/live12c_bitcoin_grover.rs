use sha2::{Digest, Sha256};
use std::f64::consts::FRAC_1_SQRT_2;

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
    qubits: usize,
    amplitudes: Vec<Amp>,
}

impl QuantumRegister {
    fn basis_zero(qubits: usize) -> Self {
        let size = 1usize << qubits;
        let mut amplitudes = vec![Amp::zero(); size];

        amplitudes[0] = Amp { re: 1.0, im: 0.0 };

        Self { qubits, amplitudes }
    }

    fn hadamard(&mut self, target: usize) {
        let mask = 1usize << target;

        for base in 0..self.amplitudes.len() {
            if base & mask != 0 {
                continue;
            }

            let other = base | mask;

            let a = self.amplitudes[base];
            let b = self.amplitudes[other];

            self.amplitudes[base] = Amp {
                re: (a.re + b.re) * FRAC_1_SQRT_2,
                im: (a.im + b.im) * FRAC_1_SQRT_2,
            };

            self.amplitudes[other] = Amp {
                re: (a.re - b.re) * FRAC_1_SQRT_2,
                im: (a.im - b.im) * FRAC_1_SQRT_2,
            };
        }
    }

    fn uniform(&mut self) {
        for q in 0..self.qubits {
            self.hadamard(q);
        }
    }

    fn mark(&mut self, state: usize) {
        self.amplitudes[state].re *= -1.0;
        self.amplitudes[state].im *= -1.0;
    }

    fn diffuse(&mut self) {
        let n = self.amplitudes.len() as f64;

        let mean_re =
            self.amplitudes.iter().map(|a| a.re).sum::<f64>() / n;

        let mean_im =
            self.amplitudes.iter().map(|a| a.im).sum::<f64>() / n;

        for a in &mut self.amplitudes {
            a.re = 2.0 * mean_re - a.re;
            a.im = 2.0 * mean_im - a.im;
        }
    }

    fn probability(&self, state: usize) -> f64 {
        self.amplitudes[state].norm_sqr()
    }

    fn total_probability(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_sqr()).sum()
    }

    fn winner(&self) -> (usize, f64) {
        self.amplitudes
            .iter()
            .enumerate()
            .map(|(i, a)| (i, a.norm_sqr()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }
}

fn sha256d(data: &[u8]) -> [u8; 32] {
    let first = Sha256::digest(data);
    Sha256::digest(first).into()
}

fn hash_numeric_be(raw_digest: [u8; 32]) -> [u8; 32] {
    let mut x = raw_digest;
    x.reverse();
    x
}

fn main() {
    const QUBITS: usize = 8;
    const STATES: usize = 1 << QUBITS;

    println!("X1331 LIVE-12C — BITCOIN SHA256d × GROVER");
    println!("=========================================");

    /*
     * Deterministic 80-byte Bitcoin-style header.
     *
     * Bytes 0..76 are frozen.
     * Bytes 76..80 are the nonce in little-endian,
     * exactly as in x1331_miner.
     */
    let mut base_header = [0u8; 80];

    // Give the frozen part deterministic non-zero structure.
    for (i, b) in base_header[..76].iter_mut().enumerate() {
        *b = ((i * 37 + 19) & 0xff) as u8;
    }

    let mut best_nonce = 0usize;
    let mut best_numeric = [0xffu8; 32];
    let mut best_raw = [0u8; 32];

    println!("Classical SHA256d scan of {STATES} candidate nonces...");

    for nonce in 0..STATES {
        let mut header = base_header;

        header[76..80]
            .copy_from_slice(&(nonce as u32).to_le_bytes());

        let raw = sha256d(&header);
        let numeric = hash_numeric_be(raw);

        if numeric < best_numeric {
            best_numeric = numeric;
            best_raw = raw;
            best_nonce = nonce;
        }
    }

    println!("SHA-selected nonce : {best_nonce}");
    println!("nonce binary       : {best_nonce:08b}");
    println!("numeric hash       : {}", hex::encode(best_numeric));
    println!("raw SHA256d        : {}", hex::encode(best_raw));

    /*
     * Quantum simulation starts here.
     *
     * IMPORTANT:
     * SHA above is classical. We use its result to construct
     * the phase oracle. This is a hybrid Grover simulation,
     * NOT a reversible quantum SHA-256 circuit.
     */

    let mut q = QuantumRegister::basis_zero(QUBITS);
    q.uniform();

    println!(
        "\nInitial probability of SHA-selected nonce: {:.9}",
        q.probability(best_nonce)
    );

    let iterations =
        (std::f64::consts::PI / 4.0 * (STATES as f64).sqrt()).floor() as usize;

    println!("Grover iterations: {iterations}");

    for i in 0..iterations {
        q.mark(best_nonce);
        q.diffuse();

        println!(
            "iteration {:02}: P(SHA winner) = {:.9}",
            i + 1,
            q.probability(best_nonce)
        );
    }

    let (winner, probability) = q.winner();

    println!();
    println!("Grover winner      : {winner}");
    println!("winner binary      : {winner:08b}");
    println!("winner probability : {:.9}", probability);
    println!("total probability  : {:.12}", q.total_probability());

    // Independently recompute the winning header/hash.
    let mut verification_header = base_header;

    verification_header[76..80]
        .copy_from_slice(&(winner as u32).to_le_bytes());

    let verification_raw = sha256d(&verification_header);
    let verification_numeric = hash_numeric_be(verification_raw);

    println!("verified hash      : {}", hex::encode(verification_numeric));

    assert_eq!(winner, best_nonce);
    assert_eq!(verification_numeric, best_numeric);
    assert!((q.total_probability() - 1.0).abs() < 1.0e-10);

    println!();
    println!("PASS — SHA256d selected reality and Grover amplified it.");
}
