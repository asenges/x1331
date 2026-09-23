#[path = "../x1331_quantum/mod.rs"]
mod x1331_quantum;

use std::time::Instant;

use x1331_quantum::{
    ranked_states, retained, BinaryObserverAction, BinaryTemporalMemory, BinaryTemporalObserver,
    X1331BitMatrix, X1331State,
};

const TRAIN_STEPS: usize = 24_000;
const EVAL_STEPS: usize = 16_000;

const STRONG_SEED: u64 = 0x1331_09D0_0000_0001;
const WEAK_SEED: u64 = 0x1331_09D0_0000_0002;
const NULL_SEED: u64 = 0x1331_09D0_0000_0003;

#[derive(Debug, Clone, Copy)]
enum Regime {
    Strong,
    Weak,
    Null,
}

impl Regime {
    fn name(self) -> &'static str {
        match self {
            Self::Strong => "STRONG",
            Self::Weak => "WEAK",
            Self::Null => "NULL",
        }
    }

    fn seed(self) -> u64 {
        match self {
            Self::Strong => STRONG_SEED,
            Self::Weak => WEAK_SEED,
            Self::Null => NULL_SEED,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Metrics {
    evaluated: u64,
    top1_hits: u64,
    pass: u64,
    k4: u64,
    k2: u64,
    k1: u64,
    retained_actual: u64,
    k_sum: u64,
    log_loss_sum: f64,
}

impl Metrics {
    fn observe(
        &mut self,
        probabilities: &[f64; 8],
        actual: X1331State,
        action: BinaryObserverAction,
    ) {
        self.evaluated += 1;

        let ranked = ranked_states(probabilities);

        if ranked[0] == actual {
            self.top1_hits += 1;
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

        let p = probabilities[actual.value() as usize].clamp(1.0e-12, 1.0);

        self.log_loss_sum += -p.ln();
    }

    fn top1(&self) -> f64 {
        ratio(self.top1_hits, self.evaluated)
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

    fn log_loss(&self) -> f64 {
        if self.evaluated == 0 {
            0.0
        } else {
            self.log_loss_sum / self.evaluated as f64
        }
    }
}

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

    fn next_state(&mut self) -> X1331State {
        X1331State::from_u8((self.next_u64() & 7) as u8)
    }

    fn chance_256(&mut self, threshold: u8) -> bool {
        (self.next_u64() & 0xff) < threshold as u64
    }
}

fn main() {
    println!("============================================================");
    println!(" X1331 LIVE-09D — BINARY TEMPORAL OBSERVER");
    println!(" RAW BITS -> FIGURES -> MEM -> PSI -> OBSERVER");
    println!(" Zero-future-knowledge causal boundary");
    println!(" No SHA256d. No Bitcoin prediction.");
    println!("============================================================");

    binary_roundtrip_gate();

    let strong = run(Regime::Strong);
    let weak = run(Regime::Weak);
    let null = run(Regime::Null);

    print_metrics(Regime::Strong, &strong);

    print_metrics(Regime::Weak, &weak);

    print_metrics(Regime::Null, &null);

    println!();
    println!("------------------------------------------------------------");
    println!("ACCEPTANCE CHECKS");
    println!("------------------------------------------------------------");

    let strong_learns = strong.top1() > 0.90;

    let strong_reduces = strong.coverage() > 0.50 && strong.retained_recall() > 0.90;

    let weak_detected = weak.top1() > 0.40;

    let weak_cautious = weak.reduction() < strong.reduction();

    let null_random = (null.top1() - 0.125).abs() < 0.03;

    let null_pass = null.pass_rate() > 0.95;

    check("strong binary relation learned", strong_learns);

    check("strong safely reduces N -> K", strong_reduces);

    check("weak binary signal detected", weak_detected);

    check("weak more cautious than strong", weak_cautious);

    check("null remains near uniform", null_random);

    check("null returns PASS", null_pass);

    assert!(strong_learns);
    assert!(strong_reduces);
    assert!(weak_detected);
    assert!(weak_cautious);
    assert!(null_random);
    assert!(null_pass);

    println!();
    println!("LIVE-09D: PASS");
}

fn binary_roundtrip_gate() {
    let bytes = vec![0x13, 0x31, 0xa5, 0x5a, 0xff, 0x00, 0x81, 0x7e];

    let matrix = X1331BitMatrix::from_bytes(bytes.clone());

    let bits = matrix.full_bit_view();

    let rebuilt = X1331BitMatrix::bytes_from_bits(&bits).expect("binary reconstruction failed");

    assert_eq!(bytes, rebuilt);

    println!(
        "binary gate: bytes={} bits={} roundtrip=PASS",
        bytes.len(),
        bits.len(),
    );
}

fn run(regime: Regime) -> Metrics {
    let mut memory = BinaryTemporalMemory::new();

    let mut observer = BinaryTemporalObserver::new();

    let mut rng = Rng::new(regime.seed());

    let mut history = [X1331State::S000, X1331State::S001, X1331State::S010];

    for _ in 0..TRAIN_STEPS {
        let frame = build_binary_frame(regime, history, &mut rng);

        let matrix = X1331BitMatrix::from_bytes(frame);

        let extracted_history = BinaryTemporalMemory::history_from_matrix(&matrix, 0)
            .expect("history extraction failed");

        assert_eq!(extracted_history, history);

        let (psi, _prediction) = memory.resonate(extracted_history);

        let before = psi.probabilities();

        /*
         * CAUSAL BOUNDARY.
         *
         * The target exists in the raw binary frame,
         * but it is not read until AFTER the BEFORE
         * prediction has been frozen.
         */
        let actual =
            BinaryTemporalMemory::target_from_matrix(&matrix, 0).expect("target extraction failed");

        observer.observe_result(&before, actual);

        memory.observe(extracted_history, actual);

        history = [history[1], history[2], actual];
    }

    let start = Instant::now();

    let mut metrics = Metrics::default();

    for _ in 0..EVAL_STEPS {
        let frame = build_binary_frame(regime, history, &mut rng);

        let matrix = X1331BitMatrix::from_bytes(frame);

        /*
         * BEFORE:
         * only first 9 bits are extracted.
         */
        let extracted_history = BinaryTemporalMemory::history_from_matrix(&matrix, 0)
            .expect("history extraction failed");

        let (psi, prediction) = memory.resonate(extracted_history);

        let probabilities = psi.probabilities();

        let action = observer.decide(&psi, &prediction);

        /*
         * AFTER:
         * target bits 9..11 are revealed only now.
         */
        let actual =
            BinaryTemporalMemory::target_from_matrix(&matrix, 0).expect("target extraction failed");

        metrics.observe(&probabilities, actual, action);

        observer.observe_result(&probabilities, actual);

        memory.observe(extracted_history, actual);

        history = [history[1], history[2], actual];
    }

    let elapsed = start.elapsed();

    println!(
        "{} runtime_ns={} ns_per_eval={:.3} experiences={}",
        regime.name(),
        elapsed.as_nanos(),
        elapsed.as_nanos() as f64 / EVAL_STEPS as f64,
        memory.experiences(),
    );

    metrics
}

fn build_binary_frame(regime: Regime, history: [X1331State; 3], rng: &mut Rng) -> Vec<u8> {
    let actual = generate_next(regime, history, rng);

    let mut bits = Vec::with_capacity(16);

    for state in history {
        bits.extend_from_slice(&state.bits());
    }

    bits.extend_from_slice(&actual.bits());

    /*
     * Pad to a whole number of bytes.
     * Padding is transport only and is not part of
     * the four X1331 figures under test.
     */
    while bits.len() % 8 != 0 {
        bits.push(0);
    }

    X1331BitMatrix::bytes_from_bits(&bits).expect("failed to encode binary frame")
}

fn generate_next(regime: Regime, history: [X1331State; 3], rng: &mut Rng) -> X1331State {
    let deterministic = temporal_rule(history);

    match regime {
        Regime::Strong => deterministic,

        Regime::Weak => {
            if rng.chance_256(160) {
                deterministic
            } else {
                rng.next_state()
            }
        }

        Regime::Null => rng.next_state(),
    }
}

fn temporal_rule(history: [X1331State; 3]) -> X1331State {
    let mixed = history[0].value() ^ history[1].value() ^ history[2].value();

    let rotated = ((mixed << 1) | (mixed >> 2)) & 0b111;

    X1331State::from_u8(rotated)
}

fn print_metrics(regime: Regime, metrics: &Metrics) {
    println!();
    println!("------------------------------------------------------------");
    println!("REGIME {}", regime.name(),);
    println!("------------------------------------------------------------");

    println!(
        "evaluated={} top1={:.6} logloss={:.6}",
        metrics.evaluated,
        metrics.top1(),
        metrics.log_loss(),
    );

    println!(
        "PASS={} K4={} K2={} K1={}",
        metrics.pass, metrics.k4, metrics.k2, metrics.k1,
    );

    println!(
        "pass_rate={:.6} coverage={:.6} meanK={:.4} reduction={:.6} retained_recall={:.6}",
        metrics.pass_rate(),
        metrics.coverage(),
        metrics.mean_k(),
        metrics.reduction(),
        metrics.retained_recall(),
    );
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn check(label: &str, value: bool) {
    println!("{:<38} {}", label, if value { "PASS" } else { "FAIL" },);
}
