/*
 * X1331 LIVE-12H1
 *
 * Reduced-width SHA-like reversible primitives.
 *
 * 4-bit words are used so every possible input can be
 * exhaustively verified on the VPS.
 *
 * This is NOT SHA-256.
 */

const MASK: u8 = 0x0f;

fn rotr4(x: u8, n: u32) -> u8 {
    let n = n % 4;

    if n == 0 {
        x & MASK
    } else {
        ((x >> n) | (x << (4 - n))) & MASK
    }
}

/*
 * SHA-256 Boolean function:
 *
 * Ch(x,y,z) = (x & y) ^ (!x & z)
 *
 * restricted to four bits.
 */
fn ch4(x: u8, y: u8, z: u8) -> u8 {
    ((x & y) ^ ((!x & MASK) & z)) & MASK
}

/*
 * SHA-256 Boolean function:
 *
 * Maj(x,y,z) =
 *     (x & y) ^ (x & z) ^ (y & z)
 */
fn maj4(x: u8, y: u8, z: u8) -> u8 {
    ((x & y) ^ (x & z) ^ (y & z)) & MASK
}

/*
 * Reduced-width analogues of SHA-256's
 * big sigma functions.
 *
 * The rotation distances are adapted to four-bit words.
 */
fn sigma0_4(x: u8) -> u8 {
    rotr4(x, 1)
        ^ rotr4(x, 2)
        ^ rotr4(x, 3)
}

fn sigma1_4(x: u8) -> u8 {
    rotr4(x, 1)
        ^ rotr4(x, 2)
        ^ x
}

/*
 * Reversible compute form:
 *
 * |inputs>|target>
 *
 * target ^= f(inputs)
 *
 * Applying the same transformation twice restores target.
 */

fn compute_ch(
    x: u8,
    y: u8,
    z: u8,
    target: &mut u8,
) {
    *target ^= ch4(x, y, z);
    *target &= MASK;
}

fn compute_maj(
    x: u8,
    y: u8,
    z: u8,
    target: &mut u8,
) {
    *target ^= maj4(x, y, z);
    *target &= MASK;
}

fn compute_sigma0(
    x: u8,
    target: &mut u8,
) {
    *target ^= sigma0_4(x);
    *target &= MASK;
}

fn compute_sigma1(
    x: u8,
    target: &mut u8,
) {
    *target ^= sigma1_4(x);
    *target &= MASK;
}

fn main() {
    println!(
        "X1331 LIVE-12H1 — REDUCED SHA-LIKE PRIMITIVES"
    );
    println!(
        "==============================================="
    );

    println!("word width : 4 bits");
    println!("domain     : 0..15");
    println!("claim      : reduced-width SHA-like only");

    /*
     * TEST 1 — rotations.
     */
    println!();
    println!("TEST 1 — ROTR");
    println!("-------------");

    let mut rotation_tests = 0usize;

    for x in 0u8..16 {
        for n in 0..4 {
            let y = rotr4(x, n);

            assert!(y <= MASK);

            /*
             * Rotate back.
             */
            let restored =
                rotr4(y, (4 - n) % 4);

            assert_eq!(
                restored,
                x,
                "rotation inverse failed x={x} n={n}"
            );

            rotation_tests += 1;
        }
    }

    println!(
        "rotation tests : {rotation_tests}/64"
    );

    /*
     * TEST 2 — Ch exhaustive.
     */
    println!();
    println!("TEST 2 — Ch(x,y,z)");
    println!("------------------");

    let mut ch_tests = 0usize;

    for x in 0u8..16 {
        for y in 0u8..16 {
            for z in 0u8..16 {
                let expected =
                    ((x & y)
                        ^ ((!x & MASK) & z))
                        & MASK;

                assert_eq!(
                    ch4(x, y, z),
                    expected
                );

                let mut work = 0u8;

                compute_ch(
                    x,
                    y,
                    z,
                    &mut work,
                );

                assert_eq!(work, expected);

                /*
                 * Exact uncompute.
                 */
                compute_ch(
                    x,
                    y,
                    z,
                    &mut work,
                );

                assert_eq!(
                    work,
                    0,
                    "Ch uncompute failed"
                );

                ch_tests += 1;
            }
        }
    }

    println!(
        "Ch tests       : {ch_tests}/4096"
    );

    /*
     * TEST 3 — Majority exhaustive.
     */
    println!();
    println!("TEST 3 — Maj(x,y,z)");
    println!("-------------------");

    let mut maj_tests = 0usize;

    for x in 0u8..16 {
        for y in 0u8..16 {
            for z in 0u8..16 {
                let expected =
                    ((x & y)
                        ^ (x & z)
                        ^ (y & z))
                        & MASK;

                assert_eq!(
                    maj4(x, y, z),
                    expected
                );

                let mut work = 0u8;

                compute_maj(
                    x,
                    y,
                    z,
                    &mut work,
                );

                assert_eq!(work, expected);

                compute_maj(
                    x,
                    y,
                    z,
                    &mut work,
                );

                assert_eq!(
                    work,
                    0,
                    "Maj uncompute failed"
                );

                maj_tests += 1;
            }
        }
    }

    println!(
        "Maj tests      : {maj_tests}/4096"
    );

    /*
     * TEST 4 — sigma transforms.
     */
    println!();
    println!("TEST 4 — Σ-like transforms");
    println!("--------------------------");

    let mut sigma_tests = 0usize;

    for x in 0u8..16 {
        let s0 = sigma0_4(x);
        let s1 = sigma1_4(x);

        let mut work0 = 0u8;
        let mut work1 = 0u8;

        compute_sigma0(
            x,
            &mut work0,
        );

        compute_sigma1(
            x,
            &mut work1,
        );

        assert_eq!(work0, s0);
        assert_eq!(work1, s1);

        /*
         * XOR-compute again = uncompute.
         */
        compute_sigma0(
            x,
            &mut work0,
        );

        compute_sigma1(
            x,
            &mut work1,
        );

        assert_eq!(work0, 0);
        assert_eq!(work1, 0);

        sigma_tests += 1;
    }

    println!(
        "sigma tests    : {sigma_tests}/16"
    );

    /*
     * Show complete 4-bit truth table.
     */
    println!();
    println!("Reduced Σ truth table");
    println!("---------------------");

    for x in 0u8..16 {
        println!(
            "x={:02} ({:04b})  Σ0={:04b}  Σ1={:04b}",
            x,
            x,
            sigma0_4(x),
            sigma1_4(x)
        );
    }

    println!();
    println!("SUMMARY");
    println!("-------");

    println!("ROTR       : PASS");
    println!("Ch         : PASS + uncompute");
    println!("Maj        : PASS + uncompute");
    println!("Σ0 / Σ1    : PASS + uncompute");

    println!();
    println!(
        "PASS — reduced SHA-like logical primitives validated exhaustively."
    );
}
