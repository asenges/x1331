#[path = "../x1331_quantum/mod.rs"]
mod x1331_quantum;

use sha2::{Digest, Sha256};
use std::env;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::time::Instant;

use x1331_quantum::{
    ranked_states, retained, BinaryObserverAction, BinaryTemporalMemory, BinaryTemporalObserver,
    X1331BitMatrix, X1331State,
};

const DEFAULT_ITERATIONS: u64 = 10_000_000;
const WARMUP: u64 = 100_000;
const REPORT_EVERY: u64 = 100_000;

const INPUT_SEED: u64 = 0x1331_09E0_2026_0923;
const CONTROL_SEED: u64 = 0x1331_09E0_CAFE_0001;

const LOG_PATH: &str = "data/live09/live09e-sha-zero-knowledge.csv";

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);

        let mut z = self.state;

        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);

        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }

    fn fill_bytes(&mut self, bytes: &mut [u8]) {
        for chunk in bytes.chunks_mut(8) {
            let value = self.next_u64().to_le_bytes();

            for (dst, src) in chunk.iter_mut().zip(value.iter()) {
                *dst = *src;
            }
        }
    }

    fn random_probabilities(&mut self) -> [f64; 8] {
        let selected = (self.next_u64() & 7) as usize;

        let mut probabilities = [0.0; 8];

        probabilities[selected] = 1.0;

        probabilities
    }
}

#[derive(Debug, Clone, Default)]
struct Metrics {
    evaluated: u64,

    x_top1: u64,
    x_recall2: u64,
    x_recall4: u64,

    control_top1: u64,
    control_recall2: u64,
    control_recall4: u64,

    pass: u64,
    k4: u64,
    k2: u64,
    k1: u64,

    retained_actual: u64,
    k_sum: u64,

    x_logloss_sum: f64,
    uniform_logloss_sum: f64,
}

impl Metrics {
    fn observe(
        &mut self,
        probabilities: &[f64; 8],
        control: &[f64; 8],
        actual: X1331State,
        action: BinaryObserverAction,
    ) {
        self.evaluated += 1;

        let x_ranked = ranked_states(probabilities);
        let control_ranked = ranked_states(control);

        if x_ranked[0] == actual {
            self.x_top1 += 1;
        }

        if x_ranked[..2].contains(&actual) {
            self.x_recall2 += 1;
        }

        if x_ranked[..4].contains(&actual) {
            self.x_recall4 += 1;
        }

        if control_ranked[0] == actual {
            self.control_top1 += 1;
        }

        if control_ranked[..2].contains(&actual) {
            self.control_recall2 += 1;
        }

        if control_ranked[..4].contains(&actual) {
            self.control_recall4 += 1;
        }

        let k = action.k();

        self.k_sum += k as u64;

        match action {
            BinaryObserverAction::Pass => self.pass += 1,
            BinaryObserverAction::Retain4 => self.k4 += 1,
            BinaryObserverAction::Retain2 => self.k2 += 1,
            BinaryObserverAction::Collapse1 => self.k1 += 1,
        }

        if retained(probabilities, actual, k) {
            self.retained_actual += 1;
        }

        let actual_index = actual.value() as usize;

        let p = probabilities[actual_index].clamp(1.0e-12, 1.0);

        self.x_logloss_sum += -p.ln();

        self.uniform_logloss_sum += -(0.125_f64).ln();
    }

    fn x_top1_rate(&self) -> f64 {
        ratio(self.x_top1, self.evaluated)
    }

    fn x_recall2_rate(&self) -> f64 {
        ratio(self.x_recall2, self.evaluated)
    }

    fn x_recall4_rate(&self) -> f64 {
        ratio(self.x_recall4, self.evaluated)
    }

    fn control_top1_rate(&self) -> f64 {
        ratio(self.control_top1, self.evaluated)
    }

    fn control_recall2_rate(&self) -> f64 {
        ratio(self.control_recall2, self.evaluated)
    }

    fn control_recall4_rate(&self) -> f64 {
        ratio(self.control_recall4, self.evaluated)
    }

    fn pass_rate(&self) -> f64 {
        ratio(self.pass, self.evaluated)
    }

    fn coverage(&self) -> f64 {
        1.0 - self.pass_rate()
    }

    fn retained_recall(&self) -> f64 {
        ratio(self.retained_actual, self.evaluated)
    }

    fn mean_k(&self) -> f64 {
        if self.evaluated == 0 {
            8.0
        } else {
            self.k_sum as f64 / self.evaluated as f64
        }
    }

    fn reduction(&self) -> f64 {
        1.0 - self.mean_k() / 8.0
    }

    fn x_logloss(&self) -> f64 {
        if self.evaluated == 0 {
            0.0
        } else {
            self.x_logloss_sum / self.evaluated as f64
        }
    }

    fn uniform_logloss(&self) -> f64 {
        if self.evaluated == 0 {
            0.0
        } else {
            self.uniform_logloss_sum / self.evaluated as f64
        }
    }
}

fn sha256d(input: &[u8; 80]) -> [u8; 32] {
    let first = Sha256::digest(input);
    Sha256::digest(first).into()
}

fn context_from_input(matrix: &X1331BitMatrix) -> [X1331State; 3] {
    /*
     * Three separated 3-bit figures from the 640-bit input.
     *
     * All coordinates exist BEFORE SHA256d.
     *
     * 0..2
     * 213..215
     * 426..428
     */
    [
        matrix.cell_at(0).expect("input cell 0 missing").state,
        matrix.cell_at(213).expect("input cell 213 missing").state,
        matrix.cell_at(426).expect("input cell 426 missing").state,
    ]
}

fn first_digest_figure(digest: &[u8; 32]) -> X1331State {
    /*
     * MSB-first, matching X1331BitMatrix.
     *
     * This reads digest bits 0..2 only AFTER SHA256d.
     */
    X1331State::from_u8(digest[0] >> 5)
}

fn main() {
    let iterations = env::args()
        .nth(1)
        .map(|value| value.parse::<u64>().expect("iterations must be an integer"))
        .unwrap_or(DEFAULT_ITERATIONS);

    assert!(iterations > WARMUP, "iterations must exceed WARMUP");

    std::fs::create_dir_all("data/live09").expect("cannot create data/live09");

    let new_file = !std::path::Path::new(LOG_PATH).exists();

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_PATH)
        .expect("cannot open LIVE-09E log");

    let mut log = BufWriter::new(file);

    if new_file {
        writeln!(
            log,
            "iteration,evaluated,x_top1,x_recall2,x_recall4,control_top1,control_recall2,control_recall4,pass_rate,coverage,mean_k,reduction,retained_recall,x_logloss,uniform_logloss,hashes_per_second"
        )
        .expect("cannot write CSV header");

        log.flush().expect("cannot flush CSV header");
    }

    println!("============================================================");
    println!(" X1331 LIVE-09E — SHA ZERO-FUTURE-KNOWLEDGE");
    println!(" 640 RAW INPUT BITS -> X1331 -> PSI -> FREEZE");
    println!(" FREEZE -> SHA256d -> 256 REAL OUTPUT BITS -> EXPERIENCE");
    println!(" No future digest exists during BEFORE.");
    println!(" Controlled SHA experiment. Not Bitcoin prediction.");
    println!("============================================================");
    println!("iterations={}", iterations);
    println!("warmup={}", WARMUP);
    println!("report_every={}", REPORT_EVERY);
    println!("log={}", LOG_PATH);
    println!();

    let mut input_rng = Rng::new(INPUT_SEED);
    let mut control_rng = Rng::new(CONTROL_SEED);

    let mut memory = BinaryTemporalMemory::new();
    let mut observer = BinaryTemporalObserver::new();

    let mut metrics = Metrics::default();

    let global_start = Instant::now();
    let mut interval_start = Instant::now();
    let mut interval_hashes = 0_u64;

    for iteration in 1..=iterations {
        /*
         * ========================================================
         * BEFORE
         * ========================================================
         *
         * Only the 80-byte input exists here.
         *
         * There is deliberately no digest variable yet.
         */
        let mut input = [0_u8; 80];
        input_rng.fill_bytes(&mut input);

        let input_matrix = X1331BitMatrix::from_bytes(input.to_vec());

        assert_eq!(input_matrix.bit_len(), 640);

        let context = context_from_input(&input_matrix);

        let (psi, prediction) = memory.resonate(context);

        let frozen_probabilities = psi.probabilities();

        let frozen_action = observer.decide(&psi, &prediction);

        let frozen_control = control_rng.random_probabilities();

        /*
         * ========================================================
         * CAUSAL / KNOWLEDGE BOUNDARY
         * ========================================================
         *
         * The SHA digest is created for the first time HERE.
         *
         * Everything above has already been frozen.
         */
        let digest = sha256d(&input);

        interval_hashes += 1;

        /*
         * ========================================================
         * AFTER
         * ========================================================
         */
        let actual = first_digest_figure(&digest);

        /*
         * Warmup builds causal experience and prospective
         * calibration but is excluded from reported evaluation.
         */
        if iteration > WARMUP {
            metrics.observe(
                &frozen_probabilities,
                &frozen_control,
                actual,
                frozen_action,
            );
        }

        /*
         * EXPERIENCE COMMIT occurs strictly AFTER scoring.
         */
        observer.observe_result(&frozen_probabilities, actual);

        memory.observe(context, actual);

        if iteration % REPORT_EVERY == 0 {
            let interval_elapsed = interval_start.elapsed().as_secs_f64();

            let hashes_per_second = interval_hashes as f64 / interval_elapsed.max(1.0e-12);

            println!(
                "iter={} eval={} x_top1={:.6} ctrl_top1={:.6} pass={:.6} K={:.4} reduction={:.6} recall={:.6} xLL={:.6} uniformLL={:.6} rate={:.0} SHA/s",
                iteration,
                metrics.evaluated,
                metrics.x_top1_rate(),
                metrics.control_top1_rate(),
                metrics.pass_rate(),
                metrics.mean_k(),
                metrics.reduction(),
                metrics.retained_recall(),
                metrics.x_logloss(),
                metrics.uniform_logloss(),
                hashes_per_second,
            );

            writeln!(
                log,
                "{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.3}",
                iteration,
                metrics.evaluated,
                metrics.x_top1_rate(),
                metrics.x_recall2_rate(),
                metrics.x_recall4_rate(),
                metrics.control_top1_rate(),
                metrics.control_recall2_rate(),
                metrics.control_recall4_rate(),
                metrics.pass_rate(),
                metrics.coverage(),
                metrics.mean_k(),
                metrics.reduction(),
                metrics.retained_recall(),
                metrics.x_logloss(),
                metrics.uniform_logloss(),
                hashes_per_second,
            )
            .expect("cannot append LIVE-09E CSV");

            log.flush().expect("cannot flush LIVE-09E CSV");

            interval_start = Instant::now();
            interval_hashes = 0;
        }
    }

    let elapsed = global_start.elapsed();

    println!();
    println!("============================================================");
    println!(" LIVE-09E FINAL");
    println!("============================================================");

    println!(
        "iterations={} evaluated={} elapsed_s={:.3}",
        iterations,
        metrics.evaluated,
        elapsed.as_secs_f64(),
    );

    println!(
        "X1331   top1={:.9} recall2={:.9} recall4={:.9} logloss={:.9}",
        metrics.x_top1_rate(),
        metrics.x_recall2_rate(),
        metrics.x_recall4_rate(),
        metrics.x_logloss(),
    );

    println!(
        "CONTROL top1={:.9} recall2={:.9} recall4={:.9}",
        metrics.control_top1_rate(),
        metrics.control_recall2_rate(),
        metrics.control_recall4_rate(),
    );

    println!("UNIFORM logloss={:.9}", metrics.uniform_logloss(),);

    println!(
        "OBSERVER pass={:.9} coverage={:.9} K8={} K4={} K2={} K1={}",
        metrics.pass_rate(),
        metrics.coverage(),
        metrics.pass,
        metrics.k4,
        metrics.k2,
        metrics.k1,
    );

    println!(
        "meanK={:.6} reduction={:.9} retained_recall={:.9}",
        metrics.mean_k(),
        metrics.reduction(),
        metrics.retained_recall(),
    );

    println!("memory_experiences={}", memory.experiences(),);

    println!();
    println!("Interpretation rule:");
    println!("  PASS on SHA noise is valid behavior.");
    println!("  Collapse is interesting ONLY if prospective Recall@K");
    println!("  exceeds the corresponding equal-budget baseline.");
    println!("  No threshold tuning is allowed from this run.");
    println!();
    println!("LIVE-09E: COMPLETE");
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}
