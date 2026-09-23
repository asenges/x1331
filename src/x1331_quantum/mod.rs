//! X1331 quantum-inspired classical computation primitives.
//!
//! LIVE-09A:
//! - exact packed raw bytes remain authoritative
//! - reversible bit coordinates
//! - 3-bit X1331 cells
//! - 8-state Boolean cube
//! - 1|3|3|1 Hamming topology
//! - figure progression
//!
//! LIVE-09B:
//! - eight-state classical complex Psi
//! - phase and interference
//! - cube-mediated evolution
//! - adaptive 8 -> K reduction
//!
//! LIVE-09C:
//! - causal MEM + EXPERIENCE
//! - conventional count-memory baseline
//! - X1331 resonance from accumulated experience
//! - prospective evaluation before memory update
//!
//! No SHA256d is performed in this module.
//! No physical quantum-computing claim is made.

pub mod bit_matrix;
pub mod memory;
pub mod schrodinger;

pub use bit_matrix::{BitOrder, FigureTransition, X1331BitMatrix, X1331Cell, X1331State};

pub use memory::{
    brier_score, l1_distance_from_uniform, log_loss, probability_rank, recall_at_k, top_state,
    ContextMemory, Experience, MemoryPrediction, ResonanceReport,
};

pub use schrodinger::{
    Amplitude, CollapseDecision, EvolutionConfig, Psi1331, RankedState, STATE_COUNT,
};
