/*
 * X1331 LIVE-13G
 *
 * FINAL RESOURCE MODEL
 * ====================
 *
 * Purpose:
 *
 * Consolidate LIVE-12 + LIVE-13 into one defensible
 * experimental conclusion.
 *
 * This program DOES NOT claim:
 *
 * - physical quantum execution
 * - a full reversible SHA-256 implementation
 * - quantum wall-clock speedup on this VPS
 * - a working quantum Bitcoin miner
 *
 * It DOES document:
 *
 * - the real Bitcoin mining predicate
 * - the Grover oracle architecture
 * - experimentally validated reduced components
 * - dense state-vector limitations
 * - ideal Grover query scaling
 * - the remaining engineering gap
 */

fn state_vector_bytes(qubits: u32) -> f64 {
    16.0 * 2f64.powi(qubits as i32)
}

fn human_bytes(bytes: f64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    const TIB: f64 = GIB * 1024.0;
    const PIB: f64 = TIB * 1024.0;
    const EIB: f64 = PIB * 1024.0;

    if bytes >= EIB {
        format!("{:.3e} EiB", bytes / EIB)
    } else if bytes >= PIB {
        format!("{:.3e} PiB", bytes / PIB)
    } else if bytes >= TIB {
        format!("{:.3} TiB", bytes / TIB)
    } else if bytes >= GIB {
        format!("{:.3} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.3} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.3} KiB", bytes / KIB)
    } else {
        format!("{:.0} B", bytes)
    }
}

fn grover_queries(n: f64, solutions: f64) -> f64 {
    /*
     * Approximate optimal Grover iteration count:
     *
     * pi/4 * sqrt(N/M)
     */
    std::f64::consts::FRAC_PI_4
        * (n / solutions).sqrt()
}

fn main() {
    println!("X1331 LIVE-13G — FINAL RESOURCE MODEL");
    println!("======================================");
    println!();

    println!("EXPERIMENTAL STATUS");
    println!("-------------------");
    println!();

    println!("LIVE-12");
    println!("  Grover amplitude amplification       : PASS");
    println!("  multiple marked solutions            : PASS");
    println!("  reversible reduced hash oracle       : PASS");
    println!("  reversible target predicate          : PASS");
    println!("  phase kickback                       : PASS");
    println!("  gate-level reduced Grover diffusion  : PASS");
    println!("  oracle uncompute                     : PASS");
    println!();

    println!("LIVE-13");
    println!("  SHA Ch circuit                       : PASS");
    println!("  reduced Sigma circuits               : PASS");
    println!("  coherent SHA slice                   : PASS");
    println!("  modular addition                     : PASS");
    println!("  MCX decomposition to CNOT/Toffoli    : PASS");
    println!("  coherent reduced T1                  : PASS");
    println!("  complete reduced round structure     : PASS");
    println!("  exhaustive round reversibility       : PASS");
    println!();

    println!("BITCOIN PREDICATE");
    println!("-----------------");
    println!();

    println!("For an 80-byte Bitcoin block header:");
    println!();
    println!("  header[76..80] = nonce in little-endian");
    println!("  raw = SHA256(SHA256(header))");
    println!("  numeric_hash = reverse(raw)");
    println!("  valid = numeric_hash <= target");
    println!();

    println!("This is the predicate a Bitcoin Grover oracle");
    println!("would have to evaluate coherently.");
    println!();

    println!("CONCEPTUAL REVERSIBLE ORACLE");
    println!("----------------------------");
    println!();

    println!(" |nonce>");
    println!("    |");
    println!("    v");
    println!(" construct Bitcoin header");
    println!("    |");
    println!("    v");
    println!(" reversible SHA-256 #1");
    println!("    |");
    println!("    v");
    println!(" reversible SHA-256 #2");
    println!("    |");
    println!("    v");
    println!(" reversible hash <= target");
    println!("    |");
    println!("    v");
    println!(" phase kickback");
    println!("    |");
    println!("    v");
    println!(" uncompute comparator");
    println!("    |");
    println!("    v");
    println!(" uncompute SHA-256 #2");
    println!("    |");
    println!("    v");
    println!(" uncompute SHA-256 #1");
    println!("    |");
    println!("    v");
    println!(" clean workspace");
    println!();

    println!("DENSE STATE-VECTOR BOUNDARY");
    println!("---------------------------");
    println!();

    for q in [8u32, 12, 18, 20, 22, 24, 25, 26, 27, 32] {
        println!(
            "{:>2} qubits : {:>15}",
            q,
            human_bytes(state_vector_bytes(q))
        );
    }

    println!();

    /*
     * These are conceptual register counts, NOT a proof
     * of the minimum possible qubit count.
     */
    let nonce_bits = 32u32;
    let sha_state_bits = 256u32;
    let digest_bits = 256u32;

    let conceptual_a =
        nonce_bits + sha_state_bits;

    let conceptual_b =
        nonce_bits + sha_state_bits + digest_bits;

    println!("CONCEPTUAL SHA256d REGISTER WIDTH");
    println!("---------------------------------");
    println!();

    println!("Bitcoin nonce                 : 32 bits");
    println!("SHA-256 working state         : 256 bits");
    println!("SHA-256 digest                : 256 bits");
    println!();

    println!(
        "nonce + one SHA state         : {} qubits",
        conceptual_a
    );

    println!(
        "nonce + state + digest        : {} qubits",
        conceptual_b
    );

    println!();

    println!("IMPORTANT:");
    println!("These are conceptual register counts only.");
    println!("They are NOT formal minimum-qubit bounds.");
    println!("A real reversible implementation also needs");
    println!("workspace for additions, message schedule,");
    println!("comparison and uncomputation.");
    println!();

    println!(
        "{}-qubit dense state vector:",
        conceptual_a
    );

    println!(
        "  amplitudes = 2^{}",
        conceptual_a
    );

    println!(
        "  memory     = {}",
        human_bytes(
            state_vector_bytes(conceptual_a)
        )
    );

    println!();

    println!(
        "{}-qubit dense state vector:",
        conceptual_b
    );

    println!(
        "  amplitudes = 2^{}",
        conceptual_b
    );

    println!(
        "  memory     = {}",
        human_bytes(
            state_vector_bytes(conceptual_b)
        )
    );

    println!();

    println!("BITCOIN NONCE SEARCH");
    println!("--------------------");
    println!();

    let nonce_space =
        2f64.powi(32);

    println!(
        "32-bit nonce states N = {:.0}",
        nonce_space
    );

    println!();

    println!("Ideal Grover query counts:");
    println!();

    for solutions in [1.0, 2.0, 4.0, 16.0, 256.0] {
        println!(
            "M={:<3.0} -> ~{:>10.1} oracle iterations",
            solutions,
            grover_queries(
                nonce_space,
                solutions
            )
        );
    }

    println!();

    let classical_one_solution =
        nonce_space;

    let grover_one_solution =
        grover_queries(
            nonce_space,
            1.0,
        );

    println!(
        "Classical worst-case nonce evaluations, M=1 : {:.0}",
        classical_one_solution
    );

    println!(
        "Ideal Grover iterations, M=1               : {:.1}",
        grover_one_solution
    );

    println!();

    println!("CRITICAL INTERPRETATION");
    println!("-----------------------");
    println!();

    println!("The quadratic Grover result is a QUERY-COMPLEXITY");
    println!("property of an ideal quantum oracle.");
    println!();

    println!("Each Grover iteration would still require a");
    println!("coherent reversible SHA256d predicate plus its");
    println!("uncomputation.");
    println!();

    println!("Running these circuits in X1331 on a classical");
    println!("CPU state-vector simulator does NOT provide");
    println!("physical quantum speedup.");
    println!();

    println!("WHAT X1331 ACTUALLY ESTABLISHED");
    println!("-------------------------------");
    println!();

    println!("1. A Bitcoin-style mining predicate can be placed");
    println!("   behind a Grover-style oracle architecture.");
    println!();

    println!("2. X1331 experimentally demonstrated the Grover");
    println!("   mechanics on reduced reversible mining models.");
    println!();

    println!("3. Ch, reduced Sigma, modular arithmetic,");
    println!("   comparator behavior, phase kickback and");
    println!("   uncomputation were independently validated.");
    println!();

    println!("4. A reduced SHA-like T1 was executed coherently.");
    println!();

    println!("5. A complete reduced SHA-like round was shown");
    println!("   exhaustively to be a reversible permutation.");
    println!();

    println!("6. Dense classical simulation cannot scale to");
    println!("   the register widths required by SHA256d.");
    println!();

    println!("7. Therefore full Bitcoin SHA256d should move");
    println!("   from dense simulation to reversible-circuit");
    println!("   resource estimation.");
    println!();

    println!("WHAT X1331 DID NOT ESTABLISH");
    println!("----------------------------");
    println!();

    println!("  physical quantum execution       : NO");
    println!("  full reversible SHA-256 circuit  : NO");
    println!("  full reversible SHA256d circuit  : NO");
    println!("  Bitcoin quantum speedup measured : NO");
    println!("  faster CPU mining                : NO");
    println!("  broken SHA-256                   : NO");
    println!("  broken Bitcoin PoW               : NO");
    println!();

    println!("NEXT ENGINEERING STEP");
    println!("---------------------");
    println!();

    println!("Do NOT increase dense simulator width.");
    println!();

    println!("Instead:");
    println!("  - specify a scalable 32-bit reversible adder");
    println!("  - specify the SHA-256 message schedule");
    println!("  - count gates/ancillas per SHA-256 round");
    println!("  - multiply across 64 rounds");
    println!("  - model the second SHA-256");
    println!("  - add the target comparator");
    println!("  - include compute/uncompute cost");
    println!("  - estimate Grover iterations separately");
    println!();

    println!("FINAL RESULT");
    println!("------------");
    println!();

    println!("X1331 has a validated classical state-vector");
    println!("laboratory for studying reduced reversible");
    println!("Bitcoin/Grover circuits.");
    println!();

    println!("It provides an experimental bridge from the");
    println!("real Bitcoin SHA256d <= target predicate to a");
    println!("conceptual reversible Grover oracle.");
    println!();

    println!("The remaining full-SHA256d problem is now a");
    println!("RESOURCE-ESTIMATION problem, not something this");
    println!("VPS should attempt to dense-simulate.");
    println!();

    println!("PASS — LIVE-12/LIVE-13 experimental bridge complete.");
}
