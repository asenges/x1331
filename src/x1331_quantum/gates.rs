use super::schrodinger::{Amplitude, Psi1331};

pub type Matrix8 = [[Amplitude; 8]; 8];

#[derive(Clone, Copy, Debug)]
pub enum Gate1331 {
    X {
        target: usize,
    },
    H {
        target: usize,
    },
    Phase {
        target: usize,
        phi: f64,
    },
    CNot {
        control: usize,
        target: usize,
    },
    ControlledPhase {
        control: usize,
        target: usize,
        phi: f64,
    },
    Swap {
        a: usize,
        b: usize,
    },
    Toffoli {
        control_a: usize,
        control_b: usize,
        target: usize,
    },
}

fn mul(a: Amplitude, b: Amplitude) -> Amplitude {
    Amplitude::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

fn bit_mask(bit: usize) -> usize {
    assert!(bit < 3, "X1331 gate bit must be 0..2");
    1usize << (2 - bit)
}

fn validate_distinct(a: usize, b: usize) {
    assert!(a < 3 && b < 3, "X1331 gate bit must be 0..2");
    assert!(a != b, "gate bits must be distinct");
}

pub fn identity_matrix() -> Matrix8 {
    let mut matrix = [[Amplitude::zero(); 8]; 8];

    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = Amplitude::new(1.0, 0.0);
    }

    matrix
}

pub fn gate_matrix(gate: Gate1331) -> Matrix8 {
    let mut matrix = [[Amplitude::zero(); 8]; 8];

    match gate {
        Gate1331::X { target } => {
            let mask = bit_mask(target);

            for input in 0..8 {
                matrix[input ^ mask][input] = Amplitude::new(1.0, 0.0);
            }
        }

        Gate1331::H { target } => {
            let mask = bit_mask(target);
            let k = std::f64::consts::FRAC_1_SQRT_2;

            for input in 0..8 {
                let zero_state = input & !mask;
                let one_state = zero_state | mask;

                if input & mask == 0 {
                    matrix[zero_state][input] = Amplitude::new(k, 0.0);
                    matrix[one_state][input] = Amplitude::new(k, 0.0);
                } else {
                    matrix[zero_state][input] = Amplitude::new(k, 0.0);
                    matrix[one_state][input] = Amplitude::new(-k, 0.0);
                }
            }
        }

        Gate1331::Phase { target, phi } => {
            let mask = bit_mask(target);
            let phase = Amplitude::from_polar(1.0, phi);

            for input in 0..8 {
                matrix[input][input] = if input & mask != 0 {
                    phase
                } else {
                    Amplitude::new(1.0, 0.0)
                };
            }
        }

        Gate1331::CNot { control, target } => {
            validate_distinct(control, target);

            let control_mask = bit_mask(control);
            let target_mask = bit_mask(target);

            for input in 0..8 {
                let output = if input & control_mask != 0 {
                    input ^ target_mask
                } else {
                    input
                };

                matrix[output][input] = Amplitude::new(1.0, 0.0);
            }
        }

        Gate1331::ControlledPhase {
            control,
            target,
            phi,
        } => {
            validate_distinct(control, target);

            let control_mask = bit_mask(control);
            let target_mask = bit_mask(target);
            let phase = Amplitude::from_polar(1.0, phi);

            for input in 0..8 {
                matrix[input][input] = if input & control_mask != 0 && input & target_mask != 0 {
                    phase
                } else {
                    Amplitude::new(1.0, 0.0)
                };
            }
        }

        Gate1331::Swap { a, b } => {
            validate_distinct(a, b);

            let a_mask = bit_mask(a);
            let b_mask = bit_mask(b);

            for input in 0..8 {
                let a_bit = input & a_mask != 0;
                let b_bit = input & b_mask != 0;

                let output = if a_bit != b_bit {
                    input ^ a_mask ^ b_mask
                } else {
                    input
                };

                matrix[output][input] = Amplitude::new(1.0, 0.0);
            }
        }

        Gate1331::Toffoli {
            control_a,
            control_b,
            target,
        } => {
            assert!(
                control_a < 3 && control_b < 3 && target < 3,
                "X1331 gate bit must be 0..2"
            );

            assert!(
                control_a != control_b && control_a != target && control_b != target,
                "Toffoli requires three distinct bits"
            );

            let control_a_mask = bit_mask(control_a);
            let control_b_mask = bit_mask(control_b);
            let target_mask = bit_mask(target);

            for input in 0..8 {
                let output = if input & control_a_mask != 0 && input & control_b_mask != 0 {
                    input ^ target_mask
                } else {
                    input
                };

                matrix[output][input] = Amplitude::new(1.0, 0.0);
            }
        }
    }

    matrix
}

pub fn apply_matrix(psi: &Psi1331, matrix: &Matrix8) -> Psi1331 {
    let input = psi.amplitudes();
    let mut output = [Amplitude::zero(); 8];

    for row in 0..8 {
        let mut value = Amplitude::zero();

        for (col, input_amplitude) in input.iter().enumerate() {
            value = value.add(mul(matrix[row][col], *input_amplitude));
        }

        output[row] = value;
    }

    Psi1331::from_amplitudes(output)
}

pub fn apply_gate_reference(psi: &Psi1331, gate: Gate1331) -> Psi1331 {
    apply_matrix(psi, &gate_matrix(gate))
}

pub fn apply_gate_fast(psi: &Psi1331, gate: Gate1331) -> Psi1331 {
    let input = *psi.amplitudes();
    let mut output = [Amplitude::zero(); 8];

    match gate {
        Gate1331::X { target } => {
            let mask = bit_mask(target);

            for i in 0..8 {
                output[i ^ mask] = input[i];
            }
        }

        Gate1331::H { target } => {
            let mask = bit_mask(target);
            let k = std::f64::consts::FRAC_1_SQRT_2;

            for base in 0..8 {
                if base & mask != 0 {
                    continue;
                }

                let other = base | mask;
                let a = input[base];
                let b = input[other];

                output[base] = a.add(b).scale(k);
                output[other] = a.add(b.scale(-1.0)).scale(k);
            }
        }

        Gate1331::Phase { target, phi } => {
            let mask = bit_mask(target);
            output = input;

            for i in 0..8 {
                if i & mask != 0 {
                    output[i] = input[i].rotate(phi);
                }
            }
        }

        Gate1331::CNot { control, target } => {
            validate_distinct(control, target);

            let control_mask = bit_mask(control);
            let target_mask = bit_mask(target);

            for i in 0..8 {
                let output_index = if i & control_mask != 0 {
                    i ^ target_mask
                } else {
                    i
                };

                output[output_index] = input[i];
            }
        }

        Gate1331::ControlledPhase {
            control,
            target,
            phi,
        } => {
            validate_distinct(control, target);

            let control_mask = bit_mask(control);
            let target_mask = bit_mask(target);

            output = input;

            for i in 0..8 {
                if i & control_mask != 0 && i & target_mask != 0 {
                    output[i] = input[i].rotate(phi);
                }
            }
        }

        Gate1331::Swap { a, b } => {
            validate_distinct(a, b);

            let a_mask = bit_mask(a);
            let b_mask = bit_mask(b);

            for i in 0..8 {
                let a_bit = i & a_mask != 0;
                let b_bit = i & b_mask != 0;

                let output_index = if a_bit != b_bit {
                    i ^ a_mask ^ b_mask
                } else {
                    i
                };

                output[output_index] = input[i];
            }
        }

        Gate1331::Toffoli {
            control_a,
            control_b,
            target,
        } => {
            assert!(
                control_a < 3 && control_b < 3 && target < 3,
                "X1331 gate bit must be 0..2"
            );

            assert!(
                control_a != control_b && control_a != target && control_b != target,
                "Toffoli requires three distinct bits"
            );

            let control_a_mask = bit_mask(control_a);
            let control_b_mask = bit_mask(control_b);
            let target_mask = bit_mask(target);

            for i in 0..8 {
                let output_index = if i & control_a_mask != 0 && i & control_b_mask != 0 {
                    i ^ target_mask
                } else {
                    i
                };

                output[output_index] = input[i];
            }
        }
    }

    Psi1331::from_amplitudes(output)
}

pub fn apply_program(psi: &Psi1331, gates: &[Gate1331]) -> Psi1331 {
    let mut state = psi.clone();

    for gate in gates {
        state = apply_gate_fast(&state, *gate);
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1.0e-10;

    fn assert_close(a: &Psi1331, b: &Psi1331) {
        for i in 0..8 {
            let left = a.amplitudes()[i];
            let right = b.amplitudes()[i];

            assert!(
                (left.re - right.re).abs() < EPS,
                "real mismatch at {i}: {} != {}",
                left.re,
                right.re
            );

            assert!(
                (left.im - right.im).abs() < EPS,
                "imag mismatch at {i}: {} != {}",
                left.im,
                right.im
            );
        }
    }

    fn reference_state() -> Psi1331 {
        Psi1331::from_amplitudes([
            Amplitude::new(1.0, 0.0),
            Amplitude::new(0.3, 0.7),
            Amplitude::new(-0.4, 0.2),
            Amplitude::new(0.8, -0.1),
            Amplitude::new(-0.2, -0.6),
            Amplitude::new(0.5, 0.4),
            Amplitude::new(-0.7, 0.3),
            Amplitude::new(0.1, -0.9),
        ])
    }

    #[test]
    fn x_fast_matches_matrix() {
        let psi = reference_state();

        for target in 0..3 {
            let gate = Gate1331::X { target };
            assert_close(
                &apply_gate_fast(&psi, gate),
                &apply_gate_reference(&psi, gate),
            );
        }
    }

    #[test]
    fn h_fast_matches_matrix() {
        let psi = reference_state();

        for target in 0..3 {
            let gate = Gate1331::H { target };
            assert_close(
                &apply_gate_fast(&psi, gate),
                &apply_gate_reference(&psi, gate),
            );
        }
    }

    #[test]
    fn phase_fast_matches_matrix() {
        let psi = reference_state();

        for target in 0..3 {
            let gate = Gate1331::Phase { target, phi: 0.731 };

            assert_close(
                &apply_gate_fast(&psi, gate),
                &apply_gate_reference(&psi, gate),
            );
        }
    }

    #[test]
    fn cnot_fast_matches_matrix() {
        let psi = reference_state();

        for control in 0..3 {
            for target in 0..3 {
                if control == target {
                    continue;
                }

                let gate = Gate1331::CNot { control, target };

                assert_close(
                    &apply_gate_fast(&psi, gate),
                    &apply_gate_reference(&psi, gate),
                );
            }
        }
    }

    #[test]
    fn controlled_phase_fast_matches_matrix() {
        let psi = reference_state();

        let gate = Gate1331::ControlledPhase {
            control: 0,
            target: 2,
            phi: -0.417,
        };

        assert_close(
            &apply_gate_fast(&psi, gate),
            &apply_gate_reference(&psi, gate),
        );
    }

    #[test]
    fn swap_fast_matches_matrix() {
        let psi = reference_state();
        let gate = Gate1331::Swap { a: 0, b: 2 };

        assert_close(
            &apply_gate_fast(&psi, gate),
            &apply_gate_reference(&psi, gate),
        );
    }

    #[test]
    fn toffoli_fast_matches_matrix() {
        let psi = reference_state();

        let gate = Gate1331::Toffoli {
            control_a: 0,
            control_b: 1,
            target: 2,
        };

        assert_close(
            &apply_gate_fast(&psi, gate),
            &apply_gate_reference(&psi, gate),
        );
    }

    #[test]
    fn gates_preserve_norm() {
        let psi = reference_state();

        let result = apply_program(
            &psi,
            &[
                Gate1331::H { target: 0 },
                Gate1331::Phase {
                    target: 2,
                    phi: 0.37,
                },
                Gate1331::CNot {
                    control: 0,
                    target: 1,
                },
                Gate1331::ControlledPhase {
                    control: 1,
                    target: 2,
                    phi: -0.91,
                },
                Gate1331::Swap { a: 0, b: 2 },
                Gate1331::Toffoli {
                    control_a: 0,
                    control_b: 1,
                    target: 2,
                },
            ],
        );

        assert!((result.total_mass() - 1.0).abs() < EPS);
    }

    #[test]
    fn hadamard_is_self_inverse() {
        let psi = reference_state();

        for target in 0..3 {
            let gate = Gate1331::H { target };

            let once = apply_gate_fast(&psi, gate);
            let twice = apply_gate_fast(&once, gate);

            assert_close(&psi, &twice);
        }
    }

    #[test]
    fn x_is_self_inverse() {
        let psi = reference_state();

        for target in 0..3 {
            let gate = Gate1331::X { target };

            let once = apply_gate_fast(&psi, gate);
            let twice = apply_gate_fast(&once, gate);

            assert_close(&psi, &twice);
        }
    }

    #[test]
    fn hadamard_three_bits_creates_uniform_superposition() {
        let psi = Psi1331::basis(super::super::bit_matrix::X1331State::S000);

        let result = apply_program(
            &psi,
            &[
                Gate1331::H { target: 0 },
                Gate1331::H { target: 1 },
                Gate1331::H { target: 2 },
            ],
        );

        for probability in result.probabilities() {
            assert!((probability - 0.125).abs() < EPS);
        }
    }

    #[test]
    fn phase_changes_interference_without_initial_probability_change() {
        let k = std::f64::consts::FRAC_1_SQRT_2;

        let psi = Psi1331::from_amplitudes([
            Amplitude::new(k, 0.0),
            Amplitude::zero(),
            Amplitude::zero(),
            Amplitude::zero(),
            Amplitude::new(k, 0.0),
            Amplitude::zero(),
            Amplitude::zero(),
            Amplitude::zero(),
        ]);

        let phased = apply_gate_fast(
            &psi,
            Gate1331::Phase {
                target: 0,
                phi: std::f64::consts::PI,
            },
        );

        let before = psi.probabilities();
        let after_phase = phased.probabilities();

        for i in 0..8 {
            assert!((before[i] - after_phase[i]).abs() < EPS);
        }

        let normal_h = apply_gate_fast(&psi, Gate1331::H { target: 0 });
        let phased_h = apply_gate_fast(&phased, Gate1331::H { target: 0 });

        let normal_p = normal_h.probabilities();
        let phased_p = phased_h.probabilities();

        assert!(normal_p[0] > 1.0 - EPS);
        assert!(phased_p[4] > 1.0 - EPS);
    }
}
