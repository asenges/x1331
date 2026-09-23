use std::f64::consts::PI;

use x1331::x1331_quantum::{apply_gate_fast, Gate1331, Psi1331, X1331State};

const MAX_ROUNDS: usize = 4096;
const STABLE_ROUNDS_REQUIRED: usize = 8;

/*
LIVE-11Q-B
==========

Purpose:
    Detect a prospective internal state transition that can eventually
    become the trigger for spending SHA work.

Important:
    This laboratory DOES NOT claim SHA predictability.
    It does not execute SHA.
    It does not inspect a future digest.

Pipeline:

    raw binary stimulus
          |
          v
        Psi1331
          |
    gates/interference
          |
          v
       Signal
          |
       PASS / TRY

TRY means:
    "the internal observer reached a sufficiently concentrated,
     separated and stable state."

Whether TRY has any predictive value must be established later by a
strict prospective SHA experiment.
*/

#[derive(Debug, Clone, Copy)]
struct SignalSnapshot {
    round: usize,

    entropy_bits: f64,
    concentration: f64,
    separation: f64,

    coherence: f64,
    phase_motion: f64,

    signal: f64,
    delta_signal: f64,

    stable_rounds: usize,
    k: usize,
}

fn phase(a: x1331::x1331_quantum::Amplitude) -> f64 {
    a.im.atan2(a.re)
}

fn wrapped_phase_delta(a: f64, b: f64) -> f64 {
    let mut d = a - b;

    while d > PI {
        d -= 2.0 * PI;
    }

    while d < -PI {
        d += 2.0 * PI;
    }

    d.abs()
}

fn phase_coherence(psi: &Psi1331) -> f64 {
    let amplitudes = psi.amplitudes();

    let mut x = 0.0;
    let mut y = 0.0;
    let mut weight = 0.0;

    for a in amplitudes {
        let p = a.norm_sqr();

        if p <= 1.0e-15 {
            continue;
        }

        let ph = phase(*a);

        x += p * ph.cos();
        y += p * ph.sin();
        weight += p;
    }

    if weight <= 1.0e-15 {
        return 0.0;
    }

    ((x / weight).powi(2) + (y / weight).powi(2))
        .sqrt()
        .clamp(0.0, 1.0)
}

fn phase_motion(previous: &Psi1331, current: &Psi1331) -> f64 {
    let a = previous.amplitudes();
    let b = current.amplitudes();

    let mut weighted_motion = 0.0;
    let mut weight = 0.0;

    for i in 0..8 {
        let p = 0.5 * (a[i].norm_sqr() + b[i].norm_sqr());

        if p <= 1.0e-15 {
            continue;
        }

        let d = wrapped_phase_delta(phase(b[i]), phase(a[i]));

        weighted_motion += p * (d / PI);
        weight += p;
    }

    if weight <= 1.0e-15 {
        0.0
    } else {
        (weighted_motion / weight).clamp(0.0, 1.0)
    }
}

fn observer_signal(concentration: f64, separation: f64, coherence: f64, phase_motion: f64) -> f64 {
    /*
    Experimental signal only.

    High signal requires:
      - probability concentration
      - top-state separation
      - coherent phase structure
      - low current phase motion

    This is NOT probability of SHA success.
    */

    let phase_stability = 1.0 - phase_motion;

    (concentration * separation.sqrt() * coherence.sqrt() * phase_stability.sqrt()).clamp(0.0, 1.0)
}

fn gate_round(psi: &Psi1331, stimulus: X1331State, round: usize) -> Psi1331 {
    let bits = stimulus.bits();

    let mut state = psi.clone();

    /*
    H mixes possibilities.
    Raw stimulus bits control reversible operations.
    Phase evolves with round number.
    */

    let h_target = round % 3;

    state = apply_gate_fast(&state, Gate1331::H { target: h_target });

    if bits[0] == 1 {
        state = apply_gate_fast(&state, Gate1331::X { target: 0 });
    }

    if bits[1] == 1 {
        state = apply_gate_fast(
            &state,
            Gate1331::CNot {
                control: 0,
                target: 1,
            },
        );
    }

    if bits[2] == 1 {
        state = apply_gate_fast(
            &state,
            Gate1331::CNot {
                control: 1,
                target: 2,
            },
        );
    }

    let phi = ((round + 1) as f64 * PI / 32.0)
        * if bits[(round + 1) % 3] == 1 {
            1.0
        } else {
            -1.0
        };

    state = apply_gate_fast(
        &state,
        Gate1331::Phase {
            target: (round + 1) % 3,
            phi,
        },
    );

    if bits[0] ^ bits[2] == 1 {
        state = apply_gate_fast(
            &state,
            Gate1331::ControlledPhase {
                control: 0,
                target: 2,
                phi: PI / 8.0,
            },
        );
    }

    /*
    Small coherent resonance injection.

    This remains classical simulation.
    */

    let resonance_phase = if round % 2 == 0 { 0.0 } else { PI / 4.0 };

    state.interfere(stimulus, 0.025, resonance_phase);

    state
}

fn run_stimulus(stimulus: X1331State) {
    println!();
    println!("============================================================");
    println!(
        "STIMULUS {:03b}  bits={:?}",
        stimulus.value(),
        stimulus.bits()
    );
    println!("============================================================");

    let mut psi = Psi1331::uniform();

    let mut previous_signal = 0.0;
    let mut stable_rounds = 0usize;

    /*
    Development threshold only.
    It is NOT calibrated against SHA outcomes.
    */

    const SIGNAL_THRESHOLD: f64 = 0.18;
    const DELTA_TOLERANCE: f64 = 0.015;

    println!(
        "{:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7} {:>4} {:>6}",
        "round", "entropy", "conc", "sep", "coher", "motion", "signal", "delta", "K", "stable"
    );

    let mut try_round: Option<usize> = None;

    for round in 0..MAX_ROUNDS {
        let previous = psi.clone();

        psi = gate_round(&psi, stimulus, round);

        let entropy_bits = psi.entropy_bits();
        let concentration = psi.concentration();
        let separation = psi.top_separation();
        let coherence = phase_coherence(&psi);
        let motion = phase_motion(&previous, &psi);

        let signal = observer_signal(concentration, separation, coherence, motion);

        let delta_signal = signal - previous_signal;

        if signal >= SIGNAL_THRESHOLD && delta_signal.abs() <= DELTA_TOLERANCE {
            stable_rounds += 1;
        } else {
            stable_rounds = 0;
        }

        let decision = psi.decision();

        let snapshot = SignalSnapshot {
            round,
            entropy_bits,
            concentration,
            separation,
            coherence,
            phase_motion: motion,
            signal,
            delta_signal,
            stable_rounds,
            k: decision.k,
        };

        if round < 32 || round % 128 == 0 || stable_rounds >= STABLE_ROUNDS_REQUIRED {
            println!(
                "{:6} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:+7.4} {:4} {:6}",
                snapshot.round,
                snapshot.entropy_bits,
                snapshot.concentration,
                snapshot.separation,
                snapshot.coherence,
                snapshot.phase_motion,
                snapshot.signal,
                snapshot.delta_signal,
                snapshot.k,
                snapshot.stable_rounds,
            );
        }

        if stable_rounds >= STABLE_ROUNDS_REQUIRED {
            try_round = Some(round);

            println!();
            println!("*** SIGNAL TRANSITION DETECTED ***");
            println!("action          = TRY");
            println!("round           = {}", round);
            println!("stimulus        = {:03b}", stimulus.value());
            println!("signal          = {:.8}", signal);
            println!("concentration   = {:.8}", concentration);
            println!("separation      = {:.8}", separation);
            println!("coherence       = {:.8}", coherence);
            println!("phase_motion    = {:.8}", motion);
            println!("stable_rounds   = {}", stable_rounds);
            println!("collapse_k      = {}", decision.k);
            println!("retained_mass   = {:.8}", decision.retained_mass);

            print!("ranked_states   =");

            for item in decision.states {
                print!(" {:03b}:{:.6}", item.state.value(), item.probability);
            }

            println!();
            break;
        }

        previous_signal = signal;
    }

    if try_round.is_none() {
        println!();
        println!("action          = PASS");
        println!("reason          = no stable signal transition");
        println!("round_budget    = {}", MAX_ROUNDS);
    }
}

fn main() {
    println!("LIVE-11Q-B — X1331 SIGNAL OBSERVER");
    println!("Classical quantum-inspired runtime");
    println!("No SHA execution");
    println!("No future digest knowledge");
    println!("PASS / TRY signal laboratory");

    for value in 0u8..8 {
        run_stimulus(X1331State::from_u8(value));
    }
}
