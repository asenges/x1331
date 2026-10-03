const BYTES_PER_AMPLITUDE: u128 = 16; // Complex<f64>

fn state_vector_bytes(qubits: u32) -> Option<u128> {
    if qubits >= 124 {
        return None;
    }

    Some((1u128 << qubits) * BYTES_PER_AMPLITUDE)
}

fn human_bytes(bytes: u128) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    const TIB: f64 = GIB * 1024.0;
    const PIB: f64 = TIB * 1024.0;
    const EIB: f64 = PIB * 1024.0;

    let b = bytes as f64;

    if b >= EIB {
        format!("{:.3} EiB", b / EIB)
    } else if b >= PIB {
        format!("{:.3} PiB", b / PIB)
    } else if b >= TIB {
        format!("{:.3} TiB", b / TIB)
    } else if b >= GIB {
        format!("{:.3} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.3} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.3} KiB", b / KIB)
    } else {
        format!("{} B", bytes)
    }
}

fn print_dense_cost(qubits: u32) {
    match state_vector_bytes(qubits) {
        Some(bytes) => {
            println!(
                "{:>4} qubits : {:>20}",
                qubits,
                human_bytes(bytes)
            );
        }
        None => {
            println!(
                "{:>4} qubits : beyond u128 byte model",
                qubits
            );
        }
    }
}

fn main() {
    println!("X1331 LIVE-13A — SHA-256 RESOURCE BRIDGE");
    println!("=========================================");
    println!();

    println!("PURPOSE");
    println!("-------");
    println!("Quantify the resource boundary between:");
    println!("  LIVE-12 reduced reversible Grover");
    println!("and");
    println!("  a reversible Bitcoin SHA-256d oracle.");
    println!();

    println!("This experiment does NOT claim to simulate SHA-256.");
    println!();

    /*
     * Our current VPS practical state-vector region.
     */
    println!("DENSE STATE-VECTOR COST");
    println!("-----------------------");

    for q in [
        8u32, 9, 12, 16, 20, 22, 24, 25, 26, 27,
        32, 40, 64, 80, 96, 112, 120,
    ] {
        print_dense_cost(q);
    }

    println!();

    /*
     * Bitcoin/SHA-256 structural widths.
     */
    const NONCE_BITS: u32 = 32;

    const SHA_WORD_BITS: u32 = 32;
    const SHA_STATE_WORDS: u32 = 8;
    const SHA_STATE_BITS: u32 =
        SHA_WORD_BITS * SHA_STATE_WORDS;

    const SHA_BLOCK_BITS: u32 = 512;
    const SHA_DIGEST_BITS: u32 = 256;

    println!("BITCOIN / SHA-256 WIDTHS");
    println!("------------------------");
    println!(
        "Bitcoin nonce                  : {} bits",
        NONCE_BITS
    );
    println!(
        "SHA-256 word                   : {} bits",
        SHA_WORD_BITS
    );
    println!(
        "SHA-256 working state          : {} x {} = {} bits",
        SHA_STATE_WORDS,
        SHA_WORD_BITS,
        SHA_STATE_BITS
    );
    println!(
        "SHA-256 message block          : {} bits",
        SHA_BLOCK_BITS
    );
    println!(
        "SHA-256 digest                 : {} bits",
        SHA_DIGEST_BITS
    );
    println!();

    /*
     * Lower-bound conceptual register counts.
     *
     * These are NOT full circuit estimates.
     * They deliberately exclude arithmetic ancillas,
     * message-schedule workspace, comparator workspace,
     * reversible cleanup, etc.
     */
    let nonce_plus_state =
        NONCE_BITS + SHA_STATE_BITS;

    let nonce_plus_state_digest =
        NONCE_BITS
        + SHA_STATE_BITS
        + SHA_DIGEST_BITS;

    println!("CONCEPTUAL LOWER BOUNDS");
    println!("-----------------------");

    println!(
        "nonce + one SHA working state  : {} qubits",
        nonce_plus_state
    );

    println!(
        "nonce + state + digest         : {} qubits",
        nonce_plus_state_digest
    );

    println!();

    println!("These are LOWER BOUNDS only.");
    println!(
        "A real reversible SHA circuit also needs temporary"
    );
    println!(
        "workspace for additions, Ch/Maj, sigma networks,"
    );
    println!(
        "message scheduling, target comparison and uncompute."
    );

    println!();

    /*
     * Show why a dense simulator cannot represent even
     * the lower bound.
     */
    println!("DENSE SIMULATION CONSEQUENCE");
    println!("----------------------------");

    println!(
        "{} qubits would require 2^{} complex amplitudes.",
        nonce_plus_state,
        nonce_plus_state
    );

    println!(
        "{} qubits would require 2^{} complex amplitudes.",
        nonce_plus_state_digest,
        nonce_plus_state_digest
    );

    println!();

    println!(
        "For comparison, LIVE-12L used 12 qubits = 4096 amplitudes."
    );

    println!();

    /*
     * Important distinction:
     *
     * Grover search width != SHA workspace width.
     */
    println!("SEARCH SPACE VS ORACLE WIDTH");
    println!("----------------------------");

    println!(
        "Bitcoin nonce search register : {} qubits",
        NONCE_BITS
    );

    println!(
        "Nonce search states           : 2^{} = {}",
        NONCE_BITS,
        1u64 << NONCE_BITS
    );

    let classical_queries =
        (1u64 << NONCE_BITS) as f64;

    let grover_scale =
        classical_queries.sqrt();

    println!(
        "Classical exhaustive scale    : ~2^32 evaluations"
    );

    println!(
        "Ideal Grover query scale      : ~sqrt(2^32) = ~{:.0}",
        grover_scale
    );

    println!();

    println!(
        "That query reduction does NOT make SHA-256 itself"
    );
    println!(
        "cheap: every Grover oracle call must coherently"
    );
    println!(
        "compute the mining predicate and then uncompute it."
    );

    println!();

    /*
     * SHA-256d adds another important constraint.
     */
    println!("SHA-256d ORACLE STRUCTURE");
    println!("-------------------------");

    println!("Conceptually:");
    println!();
    println!("  |nonce>");
    println!("      |");
    println!("      v");
    println!("  build 80-byte Bitcoin header");
    println!("      |");
    println!("      v");
    println!("  SHA-256 #1");
    println!("      |");
    println!("      v");
    println!("  256-bit digest");
    println!("      |");
    println!("      v");
    println!("  SHA-256 #2");
    println!("      |");
    println!("      v");
    println!("  hash <= target");
    println!("      |");
    println!("      v");
    println!("  phase kickback");
    println!("      |");
    println!("      v");
    println!("  reverse SHA-256 #2");
    println!("      |");
    println!("      v");
    println!("  reverse SHA-256 #1");
    println!("      |");
    println!("      v");
    println!("  clean workspace");
    println!();

    println!("X1331 STRATEGY");
    println!("--------------");

    println!("1. Do NOT dense-simulate full SHA-256d.");
    println!(
        "2. Validate reversible SHA components at reduced width."
    );
    println!(
        "3. Measure gate/ancilla cost of those components."
    );
    println!(
        "4. Extrapolate/resource-estimate the 32-bit SHA design."
    );
    println!(
        "5. Keep real SHA256d as the independent final verifier."
    );
    println!(
        "6. Never call classical state-vector execution a physical"
    );
    println!(
        "   quantum speedup."
    );

    println!();

    println!("LIVE-13 DECISION");
    println!("----------------");
    println!(
        "Full SHA-256d dense state-vector simulation: INFEASIBLE"
    );
    println!(
        "Reduced reversible SHA circuit experiments : FEASIBLE"
    );
    println!(
        "Full SHA-256d resource estimation          : FEASIBLE"
    );
    println!(
        "Real Bitcoin SHA256d verification          : FEASIBLE"
    );

    println!();

    println!(
        "PASS — SHA-256d boundary identified without overclaiming."
    );
}
