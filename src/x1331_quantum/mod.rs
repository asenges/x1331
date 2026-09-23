//! X1331 quantum-inspired classical computation primitives.
//!
//! RAW BINARY is authoritative.
//! X1331 states are derived figures over real bit coordinates.
//!
//! LIVE-09A:
//! - reversible raw binary matrix
//! - 3-bit figures
//! - Boolean cube / 1|3|3|1 topology
//!
//! LIVE-09B:
//! - classical complex Psi
//! - phase and interference
//! - adaptive 8 -> K mechanics
//!
//! LIVE-09C:
//! - causal MEM + EXPERIENCE
//! - prospective learning
//!
//! LIVE-09D:
//! - binary-first temporal memory
//! - states derived from X1331BitMatrix
//! - BEFORE expectation
//! - prospective calibration
//! - PASS / K4 / K2 / K1
//!
//! LIVE-11Q:
//! - explicit 3-bit quantum-inspired gate engine
//! - 8-dimensional complex state vector
//! - reference 8x8 unitary transformations
//! - optimized direct amplitude operations
//! - X / H / Phase / CNOT / ControlledPhase / SWAP / Toffoli
//!
//! No physical quantum-computing claim is made.

pub mod binary_temporal;
pub mod bit_matrix;
pub mod gates;
pub mod memory;
pub mod schrodinger;

pub use binary_temporal::{
    ranked_states, retained, BinaryObserverAction, BinaryPrediction, BinaryTemporalMemory,
    BinaryTemporalObserver, ProspectiveCalibration,
};

pub use bit_matrix::{BitOrder, FigureTransition, X1331BitMatrix, X1331Cell, X1331State};

pub use gates::{
    apply_gate_fast, apply_gate_reference, apply_matrix, apply_program, gate_matrix,
    identity_matrix, Gate1331, Matrix8,
};

pub use memory::{
    brier_score, l1_distance_from_uniform, log_loss, probability_rank, recall_at_k, top_state,
    ContextMemory, Experience, MemoryPrediction, ResonanceReport,
};

pub use schrodinger::{
    Amplitude, CollapseDecision, EvolutionConfig, Psi1331, RankedState, STATE_COUNT,
};
