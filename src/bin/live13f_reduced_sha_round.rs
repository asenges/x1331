/*
 * X1331 LIVE-13F
 *
 * Complete reduced SHA-like reversible round.
 *
 * WORD WIDTH = 2 bits
 *
 * This validates the STRUCTURE of a SHA-256-style round:
 *
 * T1 = h + Sigma1(e) + Ch(e,f,g) + K + W
 * T2 = Sigma0(a) + Maj(a,b,c)
 *
 * a' = T1 + T2
 * b' = a
 * c' = b
 * d' = c
 * e' = d + T1
 * f' = e
 * g' = f
 * h' = g
 *
 * All arithmetic is modulo 4.
 *
 * IMPORTANT:
 * This is a reduced logical reversible round.
 * It is NOT full SHA-256 and NOT a full state-vector
 * gate decomposition of the entire round.
 */

const MASK: u8 = 0b11;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct State {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
    e: u8,
    f: u8,
    g: u8,
    h: u8,
}

fn add2(x: u8, y: u8) -> u8 {
    x.wrapping_add(y) & MASK
}

fn sub2(x: u8, y: u8) -> u8 {
    x.wrapping_sub(y) & MASK
}

fn rotr2(x: u8, n: usize) -> u8 {
    let n = n % 2;
    let x = x & MASK;

    if n == 0 {
        x
    } else {
        ((x >> 1) | ((x & 1) << 1)) & MASK
    }
}

fn ch(x: u8, y: u8, z: u8) -> u8 {
    ((x & y) ^ ((!x) & z)) & MASK
}

fn maj(x: u8, y: u8, z: u8) -> u8 {
    ((x & y) ^ (x & z) ^ (y & z)) & MASK
}

/*
 * Reduced SHA-like sigma functions.
 *
 * At width 2 these are structural analogues,
 * not literal SHA-256 rotation constants.
 */
fn sigma0(x: u8) -> u8 {
    (x ^ rotr2(x, 1)) & MASK
}

fn sigma1(x: u8) -> u8 {
    (x ^ rotr2(x, 1)) & MASK
}

fn t1(s: State, k: u8, w: u8) -> u8 {
    let mut x = s.h;

    x = add2(x, sigma1(s.e));
    x = add2(x, ch(s.e, s.f, s.g));
    x = add2(x, k);
    x = add2(x, w);

    x
}

fn t2(s: State) -> u8 {
    add2(
        sigma0(s.a),
        maj(s.a, s.b, s.c),
    )
}

fn round_forward(s: State, k: u8, w: u8) -> State {
    let x1 = t1(s, k, w);
    let x2 = t2(s);

    State {
        a: add2(x1, x2),
        b: s.a,
        c: s.b,
        d: s.c,
        e: add2(s.d, x1),
        f: s.e,
        g: s.f,
        h: s.g,
    }
}

/*
 * Exact inverse.
 *
 * From output:
 *
 * old a = b'
 * old b = c'
 * old c = d'
 * old e = f'
 * old f = g'
 * old g = h'
 *
 * T2 can therefore be reconstructed.
 *
 * T1 = a' - T2
 *
 * old d = e' - T1
 *
 * old h =
 *   T1 - Sigma1(old e)
 *      - Ch(old e,old f,old g)
 *      - K
 *      - W
 */
fn round_inverse(out: State, k: u8, w: u8) -> State {
    let old_a = out.b;
    let old_b = out.c;
    let old_c = out.d;

    let old_e = out.f;
    let old_f = out.g;
    let old_g = out.h;

    let x2 = add2(
        sigma0(old_a),
        maj(old_a, old_b, old_c),
    );

    let x1 = sub2(out.a, x2);

    let old_d = sub2(out.e, x1);

    let mut old_h = x1;

    old_h = sub2(old_h, sigma1(old_e));
    old_h = sub2(
        old_h,
        ch(old_e, old_f, old_g),
    );
    old_h = sub2(old_h, k);
    old_h = sub2(old_h, w);

    State {
        a: old_a,
        b: old_b,
        c: old_c,
        d: old_d,
        e: old_e,
        f: old_f,
        g: old_g,
        h: old_h,
    }
}

fn state_from_u16(mut x: u16) -> State {
    let take = |v: &mut u16| -> u8 {
        let r = (*v & 0b11) as u8;
        *v >>= 2;
        r
    };

    State {
        a: take(&mut x),
        b: take(&mut x),
        c: take(&mut x),
        d: take(&mut x),
        e: take(&mut x),
        f: take(&mut x),
        g: take(&mut x),
        h: take(&mut x),
    }
}

fn state_to_u16(s: State) -> u16 {
    (s.a as u16)
        | ((s.b as u16) << 2)
        | ((s.c as u16) << 4)
        | ((s.d as u16) << 6)
        | ((s.e as u16) << 8)
        | ((s.f as u16) << 10)
        | ((s.g as u16) << 12)
        | ((s.h as u16) << 14)
}

fn bit_distance(a: State, b: State) -> u32 {
    (state_to_u16(a) ^ state_to_u16(b))
        .count_ones()
}

fn print_state(label: &str, s: State) {
    println!(
        "{} {:02b} {:02b} {:02b} {:02b} {:02b} {:02b} {:02b} {:02b}",
        label,
        s.a,
        s.b,
        s.c,
        s.d,
        s.e,
        s.f,
        s.g,
        s.h
    );
}

fn main() {
    println!(
        "X1331 LIVE-13F — COMPLETE REDUCED SHA-LIKE ROUND"
    );
    println!(
        "================================================="
    );
    println!();

    println!("word width     : 2 bits");
    println!("state width    : 16 logical bits");
    println!("arithmetic     : modulo 4");
    println!("round inverse  : explicit");
    println!("full SHA-256   : NO");
    println!();

    println!("ROUND");
    println!("-----");
    println!(
        "T1 = h + Sigma1(e) + Ch(e,f,g) + K + W"
    );
    println!(
        "T2 = Sigma0(a) + Maj(a,b,c)"
    );
    println!();

    /*
     * Transparent example.
     */
    let input = State {
        a: 0b00,
        b: 0b01,
        c: 0b10,
        d: 0b11,
        e: 0b01,
        f: 0b10,
        g: 0b11,
        h: 0b00,
    };

    let k = 0b01;
    let w = 0b10;

    let x1 = t1(input, k, w);
    let x2 = t2(input);

    let output = round_forward(input, k, w);
    let recovered = round_inverse(output, k, w);

    print_state("input     :", input);

    println!(
        "K/W       : {:02b} {:02b}",
        k, w
    );

    println!(
        "T1/T2     : {:02b} {:02b}",
        x1, x2
    );

    print_state("output    :", output);
    print_state("recovered :", recovered);

    assert_eq!(input, recovered);

    println!("example inverse : PASS");
    println!();

    /*
     * Exhaust every possible 16-bit reduced state
     * for all 4 possible K values and 4 W values.
     *
     * 65536 * 16 = 1,048,576 round/inverse tests.
     */
    let mut tests: u64 = 0;
    let mut failures: u64 = 0;

    for raw in 0u32..=0xffff {
        let s = state_from_u16(raw as u16);

        for k in 0u8..4 {
            for w in 0u8..4 {
                let out =
                    round_forward(s, k, w);

                let back =
                    round_inverse(out, k, w);

                tests += 1;

                if back != s {
                    failures += 1;

                    if failures <= 5 {
                        println!(
                            "FAIL raw={raw:04x} K={k} W={w}"
                        );
                    }
                }
            }
        }
    }

    println!(
        "exhaustive inverse : {}/{} PASS",
        tests - failures,
        tests
    );

    assert_eq!(failures, 0);

    /*
     * Verify permutation/bijection for one fixed K/W.
     *
     * Since every input has an exact inverse this is
     * already implied, but checking collisions gives
     * us another independent sanity check.
     */
    let fixed_k = 0b01;
    let fixed_w = 0b10;

    let mut seen = vec![false; 1 << 16];
    let mut collisions = 0u64;

    for raw in 0u32..=0xffff {
        let s = state_from_u16(raw as u16);

        let out =
            round_forward(s, fixed_k, fixed_w);

        let index =
            state_to_u16(out) as usize;

        if seen[index] {
            collisions += 1;
        }

        seen[index] = true;
    }

    let outputs_seen =
        seen.iter().filter(|&&x| x).count();

    println!(
        "unique outputs     : {}/65536",
        outputs_seen
    );

    println!(
        "collisions         : {}",
        collisions
    );

    assert_eq!(outputs_seen, 65536);
    assert_eq!(collisions, 0);

    /*
     * Simple avalanche observation.
     *
     * Flip each of the 16 input bits independently
     * and compare one-round output.
     *
     * This is diagnostic only, NOT a cryptographic
     * security claim.
     */
    let base = State {
        a: 0b01,
        b: 0b10,
        c: 0b11,
        d: 0b00,
        e: 0b10,
        f: 0b01,
        g: 0b11,
        h: 0b01,
    };

    let base_raw = state_to_u16(base);

    let base_out =
        round_forward(base, fixed_k, fixed_w);

    let mut total_changed = 0u32;

    for bit in 0..16 {
        let changed_raw =
            base_raw ^ (1u16 << bit);

        let changed =
            state_from_u16(changed_raw);

        let changed_out =
            round_forward(
                changed,
                fixed_k,
                fixed_w,
            );

        total_changed +=
            bit_distance(base_out, changed_out);
    }

    let average_changed =
        total_changed as f64 / 16.0;

    println!();

    println!("DIFFUSION OBSERVATION");
    println!("---------------------");

    println!(
        "average changed output bits after one round: {:.3}/16",
        average_changed
    );

    println!(
        "security claim: NONE"
    );

    /*
     * Bridge to the coherent components already
     * validated in 13B-13E.
     */
    println!();
    println!("SHA BRIDGE STATUS");
    println!("-----------------");

    println!(
        "Ch reversible circuit             : validated earlier"
    );

    println!(
        "Sigma reversible circuit          : validated earlier"
    );

    println!(
        "modular adder CNOT+Toffoli        : validated 13D"
    );

    println!(
        "coherent reduced T1               : validated 13E"
    );

    println!(
        "T1 + T2 round structure           : validated 13F"
    );

    println!(
        "round bijection                   : PASS"
    );

    println!(
        "full 32-bit gate-level SHA round  : NOT CLAIMED"
    );

    println!();
    println!("SUMMARY");
    println!("-------");

    println!(
        "round forward/inverse : PASS"
    );

    println!(
        "exhaustive states     : PASS"
    );

    println!(
        "bijection             : PASS"
    );

    println!(
        "reduced SHA structure : PASS"
    );

    println!(
        "full SHA-256          : NO"
    );

    println!();

    println!(
        "PASS — complete reduced SHA-like round is an exact reversible permutation."
    );
}
