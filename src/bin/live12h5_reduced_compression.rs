/*
 * X1331 LIVE-12H5
 *
 * Multi-round reduced SHA-256-like compression experiment.
 *
 * 4-bit words.
 * NOT SHA-256.
 *
 * Goals:
 * 1. Chain multiple reduced compression rounds.
 * 2. Reverse the complete chain exactly.
 * 3. Measure avalanche growth as rounds increase.
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
    x.wrapping_add(y) & MASK
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

    let mut known =
        sigma1(old_e);

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

/*
 * Deterministic reduced round constants.
 *
 * These are experimental 4-bit constants.
 * They are NOT SHA-256 K constants.
 */
fn round_k(round: usize) -> u8 {
    const K: [u8; 16] = [
        0x2, 0x7, 0xb, 0xe,
        0x3, 0xd, 0x5, 0x9,
        0xc, 0x1, 0xf, 0x6,
        0xa, 0x4, 0x8, 0x0,
    ];

    K[round % K.len()]
}

/*
 * Deterministic reduced message schedule.
 *
 * Again: experimental, not SHA-256.
 */
fn round_w(seed: u8, round: usize) -> u8 {
    let r = round as u8;

    (
        seed
        .wrapping_add(r.wrapping_mul(7))
        ^ r.rotate_left(1)
    ) & MASK
}

fn compress(
    mut s: State,
    rounds: usize,
    seed: u8,
) -> State {
    for r in 0..rounds {
        let k = round_k(r);
        let w = round_w(seed, r);

        s = round(s, k, w);
    }

    s
}

fn uncompress(
    mut s: State,
    rounds: usize,
    seed: u8,
) -> State {
    for r in (0..rounds).rev() {
        let k = round_k(r);
        let w = round_w(seed, r);

        s = inverse_round(s, k, w);
    }

    s
}

fn state_words(s: State) -> [u8; 8] {
    [
        s.a, s.b, s.c, s.d,
        s.e, s.f, s.g, s.h,
    ]
}

fn from_words(w: [u8; 8]) -> State {
    State {
        a: w[0],
        b: w[1],
        c: w[2],
        d: w[3],
        e: w[4],
        f: w[5],
        g: w[6],
        h: w[7],
    }
}

fn hamming_distance(a: State, b: State) -> u32 {
    let x = state_words(a);
    let y = state_words(b);

    let mut distance = 0u32;

    for i in 0..8 {
        distance +=
            (x[i] ^ y[i]).count_ones();
    }

    distance
}

fn print_state(label: &str, s: State) {
    println!(
        "{} {:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x}",
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

fn avalanche(
    initial: State,
    rounds: usize,
    seed: u8,
) -> f64 {
    let baseline =
        compress(initial, rounds, seed);

    let original =
        state_words(initial);

    let mut total_changed = 0u32;
    let mut cases = 0u32;

    /*
     * Flip every one of the 32 input bits.
     */
    for word in 0..8 {
        for bit in 0..4 {
            let mut modified = original;

            modified[word] ^=
                1u8 << bit;

            let altered =
                compress(
                    from_words(modified),
                    rounds,
                    seed,
                );

            total_changed +=
                hamming_distance(
                    baseline,
                    altered,
                );

            cases += 1;
        }
    }

    total_changed as f64
        / cases as f64
}

fn main() {
    println!(
        "X1331 LIVE-12H5 — REDUCED COMPRESSION CHAIN"
    );
    println!(
        "============================================"
    );

    println!("word width : 4 bits");
    println!("state      : 32 logical bits");
    println!("rounds     : 1 / 2 / 4 / 8 / 16");
    println!("claim      : reduced SHA-256-like experiment");
    println!("NOT        : SHA-256");
    println!();

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

    let seed = 0xa;

    /*
     * TEST 1 — complete chain inversion.
     */
    println!("TEST 1 — chain reversibility");
    println!("----------------------------");

    print_state("initial :", initial);

    for rounds in [1usize, 2, 4, 8, 16] {
        let out =
            compress(initial, rounds, seed);

        let restored =
            uncompress(out, rounds, seed);

        assert_eq!(
            restored,
            initial,
            "chain inverse failed at {rounds} rounds"
        );

        print!(
            "{:2} rounds : ",
            rounds
        );

        print_state("", out);

        println!(
            "           inverse = PASS"
        );
    }

    /*
     * TEST 2 — avalanche progression.
     */
    println!();
    println!("TEST 2 — avalanche progression");
    println!("------------------------------");

    let mut previous = 0.0;

    for rounds in [1usize, 2, 4, 8, 16] {
        let avg =
            avalanche(
                initial,
                rounds,
                seed,
            );

        println!(
            "{:2} rounds : {:6.3} / 32 bits ({:5.1}%)",
            rounds,
            avg,
            avg / 32.0 * 100.0
        );

        previous = avg;
    }

    /*
     * TEST 3 — many deterministic initial states.
     *
     * Verify complete forward/backward chain.
     */
    println!();
    println!("TEST 3 — multi-state reversibility");
    println!("----------------------------------");

    let mut tested = 0usize;

    for x in 0u8..16 {
        for y in 0u8..16 {
            let s = State {
                a: x,
                b: y,
                c: x ^ y,
                d: add4(x, y),

                e: x ^ 0x0f,
                f: y ^ 0x0f,
                g: add4(x, 3),
                h: add4(y, 5),
            };

            for rounds in [1usize, 2, 4, 8, 16] {
                let out =
                    compress(
                        s,
                        rounds,
                        seed,
                    );

                let restored =
                    uncompress(
                        out,
                        rounds,
                        seed,
                    );

                assert_eq!(
                    restored,
                    s,
                    "inverse failed x={x} y={y} rounds={rounds}"
                );

                tested += 1;
            }
        }
    }

    println!(
        "forward/inverse chains : {tested}/{tested}"
    );

    /*
     * TEST 4 — message sensitivity.
     *
     * Change the reduced message seed by one bit.
     */
    println!();
    println!("TEST 4 — message-seed sensitivity");
    println!("---------------------------------");

    let baseline =
        compress(initial, 16, seed);

    for bit in 0..4 {
        let changed_seed =
            seed ^ (1u8 << bit);

        let altered =
            compress(
                initial,
                16,
                changed_seed,
            );

        let distance =
            hamming_distance(
                baseline,
                altered,
            );

        println!(
            "seed {:04b} -> {:04b} : changed bits = {}/32",
            seed,
            changed_seed,
            distance
        );
    }

    println!();
    println!("SUMMARY");
    println!("-------");
    println!("1 round  inverse : PASS");
    println!("2 rounds inverse : PASS");
    println!("4 rounds inverse : PASS");
    println!("8 rounds inverse : PASS");
    println!("16 rounds inverse: PASS");

    println!();
    println!(
        "PASS — reduced compression chain remained exactly reversible."
    );
}
