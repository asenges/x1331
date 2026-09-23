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
//! No SHA256d is performed in this module.
//! No physical quantum-computing claim is made.

pub mod bit_matrix;
pub mod schrodinger;

pub use bit_matrix::{BitOrder, FigureTransition, X1331BitMatrix, X1331Cell, X1331State};

pub use schrodinger::{
    Amplitude, CollapseDecision, EvolutionConfig, Psi1331, RankedState, STATE_COUNT,
};
