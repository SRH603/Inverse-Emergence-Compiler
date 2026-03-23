//! Synthesizer — automatic generation of local agent rules from emergence specifications.
//!
//! Two synthesis engines:
//! - [`enumerate`] — brute-force enumeration of FSTs (baseline, exhaustive but slow)
//! - [`cegis`] — Counter-Example Guided Inductive Synthesis using Z3 SMT solver
//!
//! The [`properties`] module connects EmergeLang specifications to simulation-based
//! property checkers, enabling generic synthesis for any spec.

pub mod cegis;
pub mod enumerate;
pub mod properties;
