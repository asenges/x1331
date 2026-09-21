use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct HashResult {
    pub nonce: u64,
    pub hash: [u8; 32],
    pub leading_zero_bits: u32,
}

pub struct Sha256dVerifier {
    evaluations: u64,
}

impl Sha256dVerifier {
    pub fn new() -> Self {
        Self { evaluations: 0 }
    }

    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    pub fn evaluate(&mut self, header: &[u8], nonce: u64) -> HashResult {
        let mut input = Vec::with_capacity(header.len() + 8);
        input.extend_from_slice(header);
        input.extend_from_slice(&nonce.to_le_bytes());

        let first = Sha256::digest(&input);
        let second = Sha256::digest(first);

        self.evaluations += 1;

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&second);

        let leading_zero_bits = count_leading_zero_bits(&hash);

        HashResult {
            nonce,
            hash,
            leading_zero_bits,
        }
    }
}

fn count_leading_zero_bits(hash: &[u8; 32]) -> u32 {
    let mut total = 0;

    for byte in hash {
        if *byte == 0 {
            total += 8;
        } else {
            total += byte.leading_zeros();
            break;
        }
    }

    total
}
