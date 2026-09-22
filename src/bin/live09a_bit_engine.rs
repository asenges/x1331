#[path = "../x1331_quantum/mod.rs"]
mod x1331_quantum;

use std::hint::black_box;
use std::time::Instant;

use x1331_quantum::{BitOrder, FigureTransition, X1331BitMatrix, X1331State};

const BENCH_ITERATIONS: usize = 5_000_000;

fn main() {
    println!("============================================================");
    println!(" X1331 LIVE-09A — PURE BIT / FIGURE ENGINE");
    println!(" Classical quantum-inspired substrate");
    println!(" No SHA256d. No prediction. No physical quantum claim.");
    println!("============================================================");

    validate_topology();
    validate_raw_matrix();
    benchmark_cells();
    benchmark_progression();

    println!();
    println!("LIVE-09A: PASS");
}

fn validate_topology() {
    println!();
    println!("--- X1331 1|3|3|1 TOPOLOGY ---");

    let mut layer_counts = [0usize; 4];

    for state in X1331State::ALL {
        let layer = state.layer();
        layer_counts[layer as usize] += 1;

        let neighbors = state.neighbors();

        println!(
            "state={} bits={:?} layer={} complement={} neighbors=[{}, {}, {}]",
            state,
            state.bits(),
            layer,
            state.complement(),
            neighbors[0],
            neighbors[1],
            neighbors[2],
        );

        for neighbor in neighbors {
            assert_eq!(state.hamming_distance(neighbor), 1);
        }
    }

    assert_eq!(layer_counts, [1, 3, 3, 1]);

    println!("layers={:?} => 1|3|3|1 PASS", layer_counts);
}

fn validate_raw_matrix() {
    println!();
    println!("--- RAW MATRIX / REVERSIBILITY ---");

    let raw = vec![
        0x13, 0x31, 0xa5, 0x5a, 0x00, 0xff, 0x81, 0x7e, 0x42, 0x24, 0xc3, 0x3c,
    ];

    let matrix = X1331BitMatrix::from_bytes(raw.clone());

    assert_eq!(matrix.bit_order(), BitOrder::MsbFirst);
    assert_eq!(matrix.bit_len(), raw.len() * 8);
    assert_eq!(matrix.raw_bytes(), raw.as_slice());

    let bit_view = matrix.full_bit_view();

    let reconstructed =
        X1331BitMatrix::bytes_from_bits(&bit_view).expect("complete raw bit view must reconstruct");

    assert_eq!(reconstructed, raw);

    let cells = matrix.cells_from(0);
    let transitions = matrix.transitions_from(0);

    println!("raw_bytes={}", matrix.raw_bytes().len());
    println!("raw_bits={}", matrix.bit_len());
    println!("complete_3bit_cells={}", cells.len());
    println!("figure_transitions={}", transitions.len());

    println!();
    println!("first cells:");

    for cell in cells.iter().take(12) {
        println!(
            "bit={:03}..{:03} figure={:?} state={} layer={}",
            cell.start_bit,
            cell.end_bit_exclusive() - 1,
            cell.bits,
            cell.state,
            cell.state.layer(),
        );
    }

    println!();
    println!("first progressions:");

    for transition in transitions.iter().take(10) {
        println!(
            "{} -> {} | xor={} distance={} layer {}->{}",
            transition.from,
            transition.to,
            transition.xor,
            transition.hamming_distance,
            transition.from_layer,
            transition.to_layer,
        );
    }

    println!();
    println!("raw byte -> bit matrix -> raw byte: PASS");
}

fn benchmark_cells() {
    println!();
    println!("--- CELL BENCHMARK ---");

    let matrix = X1331BitMatrix::from_bytes(vec![0x13, 0x31, 0xa5, 0x5a, 0x00, 0xff, 0x81, 0x7e]);

    let available_starts = matrix.bit_len() - 2;
    let mut checksum = 0u64;

    let start = Instant::now();

    for i in 0..BENCH_ITERATIONS {
        let bit_index = i % available_starts;

        let cell = black_box(
            matrix
                .cell_at(bit_index)
                .expect("benchmark coordinate must contain three bits"),
        );

        checksum = checksum.wrapping_add(cell.state.value() as u64);
        checksum = checksum.wrapping_add(cell.start_bit as u64);
    }

    let elapsed = start.elapsed();

    let ns_total = elapsed.as_nanos();
    let ns_per_cell = ns_total as f64 / BENCH_ITERATIONS as f64;

    println!("iterations={}", BENCH_ITERATIONS);
    println!("elapsed_ns={}", ns_total);
    println!("ns_per_cell={:.3}", ns_per_cell);
    println!("checksum={}", black_box(checksum));
}

fn benchmark_progression() {
    println!();
    println!("--- FIGURE PROGRESSION BENCHMARK ---");

    let mut state = X1331State::S000;
    let mut checksum = 0u64;

    let start = Instant::now();

    for i in 0..BENCH_ITERATIONS {
        let stimulus = X1331State::from_u8(((i as u64 * 5 + 3) & 0b111) as u8);

        let transition = black_box(FigureTransition::new(state, stimulus));

        checksum = checksum
            .wrapping_add(transition.xor.value() as u64)
            .wrapping_add(transition.hamming_distance as u64)
            .wrapping_add(transition.to_layer as u64);

        state = stimulus;
    }

    let elapsed = start.elapsed();

    let ns_total = elapsed.as_nanos();
    let ns_per_transition = ns_total as f64 / BENCH_ITERATIONS as f64;

    println!("iterations={}", BENCH_ITERATIONS);
    println!("elapsed_ns={}", ns_total);
    println!("ns_per_transition={:.3}", ns_per_transition);
    println!("final_state={}", state);
    println!("checksum={}", black_box(checksum));
}
