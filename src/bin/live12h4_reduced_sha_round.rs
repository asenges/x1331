/*
 * X1331 LIVE-12H4
 *
 * Reduced-width SHA-256-like compression round.
 *
 * IMPORTANT:
 * - 4-bit words, not 32-bit words.
 * - This is NOT SHA-256.
 * - It preserves the structural equations of a SHA-256 round:
 *
 * T1 = h + Σ1(e) + Ch(e,f,g) + K + W
 * T2 = Σ0(a) + Maj(a,b,c)
 *
 * e' = d + T1
 * a' = T1 + T2
 *
 * Words b,c,d,f,g,h shift as in SHA-256.
 */

const MASK: u8 = 0x0f;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

fn add4(x: u8, y: u8) -> u8 {
    (x + y) & MASK
}

fn sub4(x: u8, y: u8) -> u8 {
    x.wrapping_sub(y) & MASK
}

fn rotr4(x: u8, n: u32) -> u8 {
    let n = n % 4;

    if n == 0 {
        x & MASK
    } else {
        ((x >> n) | (x << (4 - n))) & MASK
    }
}

fn ch4(x: u8, y: u8, z: u8) -> u8 {
    ((x & y) ^ ((!x & MASK) & z)) & MASK
}

fn maj4(x: u8, y: u8, z: u8) -> u8 {
    ((x & y) ^ (x & z) ^ (y & z)) & MASK
}

fn sigma0(x: u8) -> u8 {
    (
        rotr4(x, 1)
        ^ rotr4(x, 2)
        ^ rotr4(x, 3)
    ) & MASK
}

fn sigma1(x: u8) -> u8 {
    (
        rotr4(x, 1)
        ^ rotr4(x, 2)
        ^ x
    ) & MASK
}

fn t1(s: State, k: u8, w: u8) -> u8 {
    let mut x = s.h;

    x = add4(x, sigma1(s.e));
    x = add4(x, ch4(s.e, s.f, s.g));
    x = add4(x, k);
    x = add4(x, w);

    x
}

fn t2(s: State) -> u8 {
    add4(
        sigma0(s.a),
        maj4(s.a, s.b, s.c),
    )
}

/*
 * Reduced SHA-like forward round.
 */
fn round(s: State, k: u8, w: u8) -> State {
    let t1v = t1(s, k, w);
    let t2v = t2(s);

    State {
        a: add4(t1v, t2v),
        b: s.a,
        c: s.b,
        d: s.c,

        e: add4(s.d, t1v),
        f: s.e,
        g: s.f,
        h: s.g,
    }
}

/*
 * Exact inverse of the reduced round.
 *
 * From output:
 *
 * old a = new b
 * old b = new c
 * old c = new d
 * old e = new f
 * old f = new g
 * old g = new h
 *
 * Therefore T2 can be reconstructed.
 *
 * T1 = new_a - T2
 *
 * old d = new_e - T1
 *
 * Finally:
 *
 * T1 =
 * old_h + Σ1(old_e) + Ch(old_e,old_f,old_g)
 *       + K + W
 *
 * so old_h can also be reconstructed.
 */
fn inverse_round(
    out: State,
    k: u8,
    w: u8,
) -> State {
    let old_a = out.b;
    let old_b = out.c;
    let old_c = out.d;

    let old_e = out.f;
    let old_f = out.g;
    let old_g = out.h;

    let t2v = add4(
        sigma0(old_a),
        maj4(old_a, old_b, old_c),
    );

    let t1v =
        sub4(out.a, t2v);

    let old_d =
        sub4(out.e, t1v);

    let mut known = sigma1(old_e);

    known = add4(
        known,
        ch4(old_e, old_f, old_g),
    );

    known = add4(known, k);
    known = add4(known, w);

    let old_h =
        sub4(t1v, known);

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

fn print_state(label: &str, s: State) {
    println!(
        "{} a={:x} b={:x} c={:x} d={:x} \
e={:x} f={:x} g={:x} h={:x}",
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
        "X1331 LIVE-12H4 — REDUCED SHA-LIKE ROUND"
    );
    println!(
        "=========================================="
    );

    println!("word width : 4 bits");
    println!("state      : 8 × 4-bit words");
    println!("claim      : structural SHA-256-like round");
    println!("NOT        : full SHA-256");
    println!();

    /*
     * TEST 1 — one transparent example.
     */
    let initial = State {
        a: 0x1,
        b: 0x2,
        c: 0x3,
        d: 0x4,
        e: 0x5,
        f: 0x6,
        g: 0x7,
        h: 0x8,
    };

    let k = 0x9;
    let w = 0xa;

    println!("TEST 1 — transparent round");
    println!("--------------------------");

    print_state("input :", initial);

    let t1v = t1(initial, k, w);
    let t2v = t2(initial);

    println!("K     : {:x}", k);
    println!("W     : {:x}", w);
    println!("T1    : {:x}", t1v);
    println!("T2    : {:x}", t2v);

    let output =
        round(initial, k, w);

    print_state("output:", output);

    let restored =
        inverse_round(output, k, w);

    print_state("undo  :", restored);

    assert_eq!(restored, initial);

    /*
     * TEST 2
     *
     * Exhaustively vary A and E.
     *
     * Other state words remain fixed.
     *
     * 16 × 16 × 16 K × 16 W
     * = 65,536 complete forward/inverse rounds.
     */
    println!();
    println!("TEST 2 — exhaustive A/E/K/W");
    println!("---------------------------");

    let mut tested = 0usize;

    for a in 0u8..16 {
        for e in 0u8..16 {
            for k in 0u8..16 {
                for w in 0u8..16 {
                    let s = State {
                        a,
                        b: 0x2,
                        c: 0x7,
                        d: 0xb,
                        e,
                        f: 0x3,
                        g: 0xc,
                        h: 0x5,
                    };

                    let out =
                        round(s, k, w);

                    let back =
                        inverse_round(out, k, w);

                    assert_eq!(
                        back,
                        s,
                        "inverse failure a={a} e={e} k={k} w={w}"
                    );

                    tested += 1;
                }
            }
        }
    }

    println!(
        "forward/inverse rounds : {tested}/65536"
    );

    /*
     * TEST 3
     *
     * Avalanche sanity check.
     *
     * Flip each individual input bit and count output
     * differences. This is descriptive, not a cryptographic
     * proof.
     */
    println!();
    println!("TEST 3 — reduced avalanche sanity");
    println!("---------------------------------");

    let baseline =
        round(initial, k, w);

    let base_words = [
        initial.a,
        initial.b,
        initial.c,
        initial.d,
        initial.e,
        initial.f,
        initial.g,
        initial.h,
    ];

    let mut total_changed = 0u32;
    let mut cases = 0u32;

    for word_index in 0..8 {
        for bit in 0..4 {
            let mut words = base_words;

            words[word_index] ^=
                1u8 << bit;

            let modified = State {
                a: words[0],
                b: words[1],
                c: words[2],
                d: words[3],
                e: words[4],
                f: words[5],
                g: words[6],
                h: words[7],
            };

            let out =
                round(modified, k, w);

            let lhs = [
                baseline.a,
                baseline.b,
                baseline.c,
                baseline.d,
                baseline.e,
                baseline.f,
                baseline.g,
                baseline.h,
            ];

            let rhs = [
                out.a,
                out.b,
                out.c,
                out.d,
                out.e,
                out.f,
                out.g,
                out.h,
            ];

            let mut changed = 0u32;

            for i in 0..8 {
                changed +=
                    (lhs[i] ^ rhs[i])
                        .count_ones();
            }

            println!(
                "flip word={} bit={} -> changed output bits={}",
                word_index,
                bit,
                changed
            );

            total_changed += changed;
            cases += 1;
        }
    }

    let average =
        total_changed as f64
            / cases as f64;

    println!();
    println!(
        "average changed output bits = {:.3} / 32",
        average
    );

    println!();
    println!("SUMMARY");
    println!("-------");
    println!("round structure : PASS");
    println!("inverse         : PASS");
    println!(
        "tested          : {tested} forward/inverse pairs"
    );

    println!();
    println!(
        "PASS — first reduced SHA-256-like compression round assembled and inverted."
    );
}
