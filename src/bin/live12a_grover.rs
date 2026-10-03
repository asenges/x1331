use x1331::x1331_quantum::{
    Gate1331, Psi1331, X1331State, apply_program,
};

fn main() {
    println!("X1331 LIVE-12A — GROVER 3-QUBIT LAB");
    println!("===================================");

    // |000>
    let psi = Psi1331::basis(X1331State::S000);

    // H⊗H⊗H -> uniform superposition
    let mut psi = apply_program(
        &psi,
        &[
            Gate1331::H { target: 0 },
            Gate1331::H { target: 1 },
            Gate1331::H { target: 2 },
        ],
    );

    println!("\nInitial superposition:");
    print_probabilities(&psi);

    // Oracle: mark |101> by phase inversion.
    let marked = X1331State::S101;
    let marked_index = marked.value() as usize;

    let mut amplitudes = *psi.amplitudes();
    amplitudes[marked_index] = amplitudes[marked_index].scale(-1.0);
    psi = Psi1331::from_amplitudes(amplitudes);

    println!("\nAfter oracle marking |101>:");
    print_probabilities(&psi);

    // Grover diffusion:
    // reflection of each amplitude around the mean.
    let amplitudes = *psi.amplitudes();

    let mean_re =
        amplitudes.iter().map(|a| a.re).sum::<f64>() / amplitudes.len() as f64;

    let mean_im =
        amplitudes.iter().map(|a| a.im).sum::<f64>() / amplitudes.len() as f64;

    let reflected = std::array::from_fn(|i| {
        let a = amplitudes[i];

        x1331::x1331_quantum::Amplitude::new(
            2.0 * mean_re - a.re,
            2.0 * mean_im - a.im,
        )
    });

    psi = Psi1331::from_amplitudes(reflected);

    println!("\nAfter Grover diffusion:");
    print_probabilities(&psi);

    let winner = psi.ranked()[0].clone();

    println!(
        "\nWINNER: {:03b}  probability={:.6}",
        winner.state.value(),
        winner.probability
    );

    assert_eq!(winner.state, marked);

    println!("\nPASS — Grover amplified the marked state.");
}

fn print_probabilities(psi: &Psi1331) {
    for state in X1331State::ALL {
        println!(
            "|{:03b}>  {:.6}",
            state.value(),
            psi.probability(state)
        );
    }
}
