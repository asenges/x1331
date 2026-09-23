#[path = "../x1331_quantum/mod.rs"]
mod x1331_quantum;

use std::time::Instant;

use x1331_quantum::{
    brier_score, log_loss, recall_at_k, top_state, ContextMemory, Experience, Psi1331, X1331State,
    STATE_COUNT,
};

const STEPS_PER_REGIME: usize = 16_000;
const WARMUP_STEPS: usize = 1_000;

const NULL_SEED: u64 = 0x1331_09C0_0000_0001;
const WEAK_SEED: u64 = 0x1331_09C0_0000_0002;

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
}

#[derive(Debug, Clone, Default)]
struct ArmMetrics {
    evaluated: u64,
    top1_hits: u64,
    recall2_hits: u64,
    recall4_hits: u64,
    log_loss_sum: f64,
    brier_sum: f64,
    rank_sum: u64,
}

impl ArmMetrics {
    fn observe(&mut self, probabilities: &[f64; STATE_COUNT], actual: X1331State) {
        self.evaluated += 1;

        let top = top_state(probabilities);

        if top == actual {
            self.top1_hits += 1;
        }

        if recall_at_k(probabilities, actual, 2) {
            self.recall2_hits += 1;
        }

        if recall_at_k(probabilities, actual, 4) {
            self.recall4_hits += 1;
        }

        self.log_loss_sum += log_loss(probabilities, actual);

        self.brier_sum += brier_score(probabilities, actual);

        self.rank_sum += x1331_quantum::probability_rank(probabilities, actual) as u64;
    }

    fn top1_rate(&self) -> f64 {
        ratio(self.top1_hits, self.evaluated)
    }

    fn recall2(&self) -> f64 {
        ratio(self.recall2_hits, self.evaluated)
    }

    fn recall4(&self) -> f64 {
        ratio(self.recall4_hits, self.evaluated)
    }

    fn mean_log_loss(&self) -> f64 {
        divide(self.log_loss_sum, self.evaluated)
    }

    fn mean_brier(&self) -> f64 {
        divide(self.brier_sum, self.evaluated)
    }

    fn mean_rank(&self) -> f64 {
        divide(self.rank_sum as f64, self.evaluated)
    }
}

#[derive(Debug, Clone, Default)]
struct CollapseMetrics {
    evaluated: u64,
    pass: u64,
    k4: u64,
    k2: u64,
    k1: u64,
    retained_actual: u64,
    k_sum: u64,
}

impl CollapseMetrics {
    fn observe(&mut self, psi: &Psi1331, actual: X1331State) {
        self.evaluated += 1;

        let decision = psi.decision();

        self.k_sum += decision.k as u64;

        match decision.k {
            8 => self.pass += 1,
            4 => self.k4 += 1,
            2 => self.k2 += 1,
            1 => self.k1 += 1,
            other => panic!("unexpected K={other}"),
        }

        if decision.states.iter().any(|item| item.state == actual) {
            self.retained_actual += 1;
        }
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
        divide(self.k_sum as f64, self.evaluated)
    }

    fn mean_reduction(&self) -> f64 {
        1.0 - self.mean_k() / 8.0
    }
}

#[derive(Debug, Clone)]
struct ExperimentResult {
    regime: Regime,
    control: ArmMetrics,
    count_mem: ArmMetrics,
    x1331_mem: ArmMetrics,
    collapse: CollapseMetrics,
    experiences: u64,
    elapsed_ns: u128,
}

#[derive(Debug, Clone)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
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
    println!(" X1331 LIVE-09C — MEM + EXPERIENCE");
    println!(" Prospective causal associative-memory experiment");
    println!(" CONTROL vs COUNT-MEM vs X1331-MEM");
    println!(" No SHA256d. No Bitcoin prediction.");
    println!("============================================================");

    let strong = run_regime(Regime::Strong);
    let weak = run_regime(Regime::Weak);
    let null = run_regime(Regime::Null);

    print_result(&strong);
    print_result(&weak);
    print_result(&null);

    validate(&strong, &weak, &null);

    println!();
    println!("LIVE-09C: PASS");
}

fn run_regime(regime: Regime) -> ExperimentResult {
    let mut memory = ContextMemory::new();

    let mut control = ArmMetrics::default();
    let mut count_mem = ArmMetrics::default();
    let mut x1331_mem = ArmMetrics::default();
    let mut collapse = CollapseMetrics::default();

    let mut weak_rng = DeterministicRng::new(WEAK_SEED ^ regime_tag(regime));

    let mut null_rng = DeterministicRng::new(NULL_SEED ^ regime_tag(regime));

    let start = Instant::now();

    for step in 0..STEPS_PER_REGIME {
        let context = X1331State::from_u8((step & 7) as u8);

        /*
         * CRITICAL CAUSAL ORDER:
         *
         * 1. prediction from past memory
         * 2. freeze predictions
         * 3. reveal current outcome
         * 4. score prediction
         * 5. only then update memory
         */

        let count_prediction = memory.predict_counts(context);

        let (psi, resonance) = memory.resonate(context);

        let control_probabilities = [1.0 / 8.0; STATE_COUNT];

        let count_probabilities = count_prediction.probabilities;

        let x1331_probabilities = resonance.psi_probabilities;

        // Outcome is generated only after all BEFORE
        // predictions have been produced.
        let actual = generate_outcome(regime, context, &mut weak_rng, &mut null_rng);

        if step >= WARMUP_STEPS {
            control.observe(&control_probabilities, actual);

            count_mem.observe(&count_probabilities, actual);

            x1331_mem.observe(&x1331_probabilities, actual);

            collapse.observe(&psi, actual);
        }

        memory.observe(Experience {
            context,
            outcome: actual,
        });
    }

    ExperimentResult {
        regime,
        control,
        count_mem,
        x1331_mem,
        collapse,
        experiences: memory.total_experiences(),
        elapsed_ns: start.elapsed().as_nanos(),
    }
}

fn generate_outcome(
    regime: Regime,
    context: X1331State,
    weak_rng: &mut DeterministicRng,
    null_rng: &mut DeterministicRng,
) -> X1331State {
    match regime {
        Regime::Strong => strong_mapping(context),

        Regime::Weak => {
            // Approximately 62.5% follows the planted
            // relationship; the rest is deterministic noise.
            if weak_rng.chance_256(160) {
                strong_mapping(context)
            } else {
                weak_rng.next_state()
            }
        }

        Regime::Null => null_rng.next_state(),
    }
}

fn strong_mapping(context: X1331State) -> X1331State {
    // Non-trivial but deterministic permutation.
    //
    // 000 -> 011
    // 001 -> 110
    // 010 -> 001
    // 011 -> 100
    // 100 -> 111
    // 101 -> 010
    // 110 -> 101
    // 111 -> 000
    const MAP: [u8; 8] = [0b011, 0b110, 0b001, 0b100, 0b111, 0b010, 0b101, 0b000];

    X1331State::from_u8(MAP[context.value() as usize])
}

fn regime_tag(regime: Regime) -> u64 {
    match regime {
        Regime::Strong => 0x11,
        Regime::Weak => 0x22,
        Regime::Null => 0x33,
    }
}

fn print_arm(name: &str, metrics: &ArmMetrics) {
    println!(
        "{:<11} top1={:.6} recall2={:.6} recall4={:.6} logloss={:.6} brier={:.6} mean_rank={:.4}",
        name,
        metrics.top1_rate(),
        metrics.recall2(),
        metrics.recall4(),
        metrics.mean_log_loss(),
        metrics.mean_brier(),
        metrics.mean_rank(),
    );
}

fn print_result(result: &ExperimentResult) {
    println!();
    println!("------------------------------------------------------------");
    println!("REGIME {}", result.regime.name());
    println!("------------------------------------------------------------");

    println!(
        "experiences={} evaluated={} elapsed_ns={} ns_per_experience={:.3}",
        result.experiences,
        result.control.evaluated,
        result.elapsed_ns,
        result.elapsed_ns as f64 / result.experiences as f64,
    );

    print_arm("CONTROL", &result.control);
    print_arm("COUNT-MEM", &result.count_mem);
    print_arm("X1331-MEM", &result.x1331_mem);

    println!(
        "SCHRODINGER pass={:.6} coverage={:.6} K8={} K4={} K2={} K1={} meanK={:.4} reduction={:.6} retained_recall={:.6}",
        result.collapse.pass_rate(),
        result.collapse.coverage(),
        result.collapse.pass,
        result.collapse.k4,
        result.collapse.k2,
        result.collapse.k1,
        result.collapse.mean_k(),
        result.collapse.mean_reduction(),
        result.collapse.retained_recall(),
    );
}

fn validate(strong: &ExperimentResult, weak: &ExperimentResult, null: &ExperimentResult) {
    println!();
    println!("------------------------------------------------------------");
    println!("ACCEPTANCE CHECKS");
    println!("------------------------------------------------------------");

    let uniform_log_loss = (8.0_f64).ln();

    let strong_learns = strong.count_mem.top1_rate() > 0.95 && strong.x1331_mem.top1_rate() > 0.95;

    println!(
        "strong relationship learned       {}",
        pass_fail(strong_learns)
    );

    let weak_above_control = weak.count_mem.top1_rate() > weak.control.top1_rate()
        && weak.x1331_mem.top1_rate() > weak.control.top1_rate();

    println!(
        "weak relationship above control   {}",
        pass_fail(weak_above_control)
    );

    let null_near_uniform = (null.count_mem.mean_log_loss() - uniform_log_loss).abs() < 0.05
        && (null.x1331_mem.mean_log_loss() - uniform_log_loss).abs() < 0.10;

    println!(
        "null remains near uniform          {}",
        pass_fail(null_near_uniform)
    );

    let null_mostly_passes = null.collapse.pass_rate() > 0.95;

    println!(
        "null returns PASS                  {}",
        pass_fail(null_mostly_passes)
    );

    let strong_beats_null = strong.x1331_mem.mean_log_loss() < null.x1331_mem.mean_log_loss();

    println!(
        "X1331 strong beats X1331 null      {}",
        pass_fail(strong_beats_null)
    );

    let causal_counts = strong.experiences == STEPS_PER_REGIME as u64
        && weak.experiences == STEPS_PER_REGIME as u64
        && null.experiences == STEPS_PER_REGIME as u64;

    println!(
        "all experiences committed AFTER    {}",
        pass_fail(causal_counts)
    );

    assert!(strong_learns);
    assert!(weak_above_control);
    assert!(null_near_uniform);
    assert!(null_mostly_passes);
    assert!(strong_beats_null);
    assert!(causal_counts);
}

fn pass_fail(value: bool) -> &'static str {
    if value {
        "PASS"
    } else {
        "FAIL"
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn divide(numerator: f64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator / denominator as f64
    }
}
