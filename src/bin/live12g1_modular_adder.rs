/*
 * X1331 LIVE-12G1
 * Reversible 4-bit modular adder validation.
 *
 * Goal:
 *
 *     |a>|b> -> |a>|(a+b) mod 16>
 *
 * followed by the exact inverse:
 *
 *     |a>|(a+b) mod 16> -> |a>|b>
 *
 * This experiment deliberately does NOT use Grover yet.
 * We first validate the arithmetic primitive exhaustively.
 */

const BITS: usize = 4;
const MODULUS: u8 = 1 << BITS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BasisState {
    a: u8,
    b: u8,
}

/*
 * Reversible modular addition.
 *
 * For fixed a:
 *
 *      b -> b + a (mod 16)
 *
 * is a permutation of the 4-bit state space and therefore
 * reversible.
 *
 * This function models the logical reversible transformation.
 */
fn add_mod_16(state: BasisState) -> BasisState {
    BasisState {
        a: state.a,
        b: state.b.wrapping_add(state.a) & 0x0f,
    }
}

/*
 * Exact inverse:
 *
 *      b -> b - a (mod 16)
 */
fn sub_mod_16(state: BasisState) -> BasisState {
    BasisState {
        a: state.a,
        b: state.b.wrapping_sub(state.a) & 0x0f,
    }
}

fn main() {
    println!("X1331 LIVE-12G1 — REVERSIBLE MODULAR ADDER");
    println!("============================================");
    println!("register A : {BITS} bits");
    println!("register B : {BITS} bits");
    println!("operation  : B <- (A + B) mod {MODULUS}");
    println!();

    let mut tested = 0usize;

    /*
     * Exhaustively test every possible 4-bit pair.
     *
     * 16 × 16 = 256 basis states.
     */
    for a in 0u8..MODULUS {
        for b in 0u8..MODULUS {
            let original = BasisState { a, b };

            let added = add_mod_16(original);

            let expected =
                b.wrapping_add(a) & 0x0f;

            assert_eq!(
                added.a,
                a,
                "A register changed"
            );

            assert_eq!(
                added.b,
                expected,
                "addition failed for A={a} B={b}"
            );

            /*
             * Critical reversibility test.
             */
            let restored = sub_mod_16(added);

            assert_eq!(
                restored,
                original,
                "uncompute failed for A={a} B={b}"
            );

            tested += 1;
        }
    }

    println!("Exhaustive basis-state test");
    println!("---------------------------");
    println!("tested states : {tested}");
    println!("failures      : 0");

    println!();
    println!("Examples:");
    println!("---------");

    let examples = [
        BasisState { a: 1, b: 2 },
        BasisState { a: 7, b: 5 },
        BasisState { a: 15, b: 1 },
        BasisState { a: 15, b: 15 },
    ];

    for original in examples {
        let added = add_mod_16(original);
        let restored = sub_mod_16(added);

        println!(
            "A={:02} ({:04b})  B={:02} ({:04b})  \
             -> SUM={:02} ({:04b})  \
             -> UNCOMPUTE={:02}",
            original.a,
            original.a,
            original.b,
            original.b,
            added.b,
            added.b,
            restored.b,
        );
    }

    /*
     * Verify permutation property independently.
     *
     * For every fixed A, all sixteen B outputs must occur
     * exactly once.
     */
    println!();
    println!("Permutation test");
    println!("----------------");

    for a in 0u8..MODULUS {
        let mut seen = [false; 16];

        for b in 0u8..MODULUS {
            let out =
                add_mod_16(BasisState { a, b });

            assert!(
                !seen[out.b as usize],
                "non-bijective mapping for A={a}"
            );

            seen[out.b as usize] = true;
        }

        assert!(
            seen.iter().all(|&x| x),
            "incomplete permutation for A={a}"
        );
    }

    println!("A values tested : 16");
    println!("permutations    : 16/16");

    println!();
    println!(
        "PASS — 4-bit modular addition is bijective and exactly uncomputable."
    );
}
