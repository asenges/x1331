#[path = "../x1331_quantum/mod.rs"]
mod x1331_quantum;

use std::f64::consts::PI;
use std::hint::black_box;
use std::time::{Duration, Instant};

use x1331_quantum::{EvolutionConfig, Psi1331, X1331State, STATE_COUNT};

const BENCH_CYCLES: usize = 2_000_000;
const DEMO_MAX_CYCLES: usize = 32;
const DEMO_BUDGET: Duration = Duration::from_micros(50);

fn main() {
    println!("============================================================");
    println!(" X1331 LIVE-09B — SCHRODINGER PRIMITIVE");
    println!(" Classical amplitude / phase / interference engine");
    println!(" No SHA256d. No Bitcoin prediction.");
    println!("============================================================");

    demonstrate_interference();
    demonstrate_resource_aware_evolution();
    benchmark_evolution();

    println!();
    println!("LIVE-09B: PASS");
}

fn print_psi(label: &str, psi: &Psi1331) {
    println!();
    println!("{label}");

    for state in X1331State::ALL {
        let amplitude = psi.amplitudes()[state.value() as usize];

        println!(
            "{} re={:+.6} im={:+.6} p={:.6}",
            state,
            amplitude.re,
            amplitude.im,
            psi.probability(state),
        );
    }

    let decision = psi.decision();

    let retained: Vec<String> = decision
        .states
        .iter()
        .map(|item| format!("{}:{:.4}", item.state, item.probability))
        .collect();

    println!(
        "mass={:.12} entropy={:.6} concentration={:.6} separation={:.6}",
        psi.total_mass(),
        decision.entropy_bits,
        decision.concentration,
        decision.separation,
    );

    println!(
        "decision={} K={} retained_mass={:.6} states=[{}]",
        if decision.k == STATE_COUNT {
            "PASS"
        } else {
            "COLLAPSE"
        },
        decision.k,
        decision.retained_mass,
        retained.join(", "),
    );
}

fn demonstrate_interference() {
    println!();
    println!("--- INTERFERENCE DEMONSTRATION ---");

    let target = X1331State::S101;

    let uniform = Psi1331::uniform();
    print_psi("initial uniform Psi", &uniform);

    let mut constructive = uniform.clone();
    constructive.interfere(target, 0.30, 0.0);

    print_psi("after constructive signal on 101", &constructive);

    let mut destructive = uniform.clone();
    destructive.interfere(target, 0.30, PI);

    print_psi("after opposite-phase signal on 101", &destructive);

    let pc = constructive.probability(target);
    let pd = destructive.probability(target);

    assert!(pc > pd);

    println!();
    println!(
        "target=101 constructive_p={:.6} destructive_p={:.6}",
        pc, pd
    );

    println!("constructive/destructive interference: PASS");
}

fn demonstrate_resource_aware_evolution() {
    println!();
    println!("--- RESOURCE-AWARE EVOLUTION ---");

    let config = EvolutionConfig::default();
    let mut psi = Psi1331::uniform();

    let stimuli = [
        X1331State::S101,
        X1331State::S101,
        X1331State::S111,
        X1331State::S101,
        X1331State::S100,
        X1331State::S101,
        X1331State::S111,
        X1331State::S101,
    ];

    let start = Instant::now();

    let mut previous_concentration = psi.concentration();
    let mut stagnant_cycles = 0usize;
    let mut executed = 0usize;

    for cycle in 0..DEMO_MAX_CYCLES {
        if start.elapsed() >= DEMO_BUDGET {
            println!(
                "STOP reason=resource_budget cycle={} elapsed_ns={}",
                cycle,
                start.elapsed().as_nanos()
            );
            break;
        }

        let stimulus = stimuli[cycle % stimuli.len()];

        psi.evolve(stimulus, config);
        executed += 1;

        let decision = psi.decision();
        let delta = decision.concentration - previous_concentration;

        println!(
            "cycle={:02} stimulus={} concentration={:.6} delta={:+.6} entropy={:.6} K={} elapsed_ns={}",
            cycle + 1,
            stimulus,
            decision.concentration,
            delta,
            decision.entropy_bits,
            decision.k,
            start.elapsed().as_nanos(),
        );

        if delta.abs() < 1.0e-5 {
            stagnant_cycles += 1;
        } else {
            stagnant_cycles = 0;
        }

        previous_concentration = decision.concentration;

        if stagnant_cycles >= 4 {
            println!("STOP reason=marginal_information cycle={}", cycle + 1);
            break;
        }
    }

    let elapsed = start.elapsed();
    let decision = psi.decision();

    println!();
    println!("cycles_executed={}", executed);
    println!("reasoning_ns={}", elapsed.as_nanos());
    println!(
        "final_decision={} K={}",
        if decision.k == STATE_COUNT {
            "PASS"
        } else {
            "COLLAPSE"
        },
        decision.k
    );

    println!(
        "retained_states={}",
        decision
            .states
            .iter()
            .map(|item| item.state.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    assert!((psi.total_mass() - 1.0).abs() < 1.0e-10);
}

fn benchmark_evolution() {
    println!();
    println!("--- PSI EVOLUTION BENCHMARK ---");

    let config = EvolutionConfig::default();
    let mut psi = Psi1331::uniform();

    let start = Instant::now();

    for i in 0..BENCH_CYCLES {
        let stimulus = X1331State::from_u8(((i * 5 + 3) & 7) as u8);

        psi.evolve(black_box(stimulus), black_box(config));
        black_box(&psi);
    }

    let elapsed = start.elapsed();

    let total_ns = elapsed.as_nanos();
    let ns_per_cycle = total_ns as f64 / BENCH_CYCLES as f64;

    let cycles_per_us = 1000.0 / ns_per_cycle;

    println!("cycles={}", BENCH_CYCLES);
    println!("elapsed_ns={}", total_ns);
    println!("ns_per_cycle={:.3}", ns_per_cycle);
    println!("cycles_per_us={:.3}", cycles_per_us);
    println!("final_mass={:.12}", psi.total_mass());

    assert!((psi.total_mass() - 1.0).abs() < 1.0e-10);
}
