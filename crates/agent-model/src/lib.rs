//! Agent model — representations of agents, their local rules, and communication topologies.
//!
//! This crate provides:
//! - [`fst::Fst`] — Finite State Transducer, the simplest local rule representation
//! - [`topology::Topology`] — how agents are connected (complete, ring, grid, etc.)
//! - [`interpreter`] — multi-agent simulation engine that runs FSTs on topologies

pub mod fst;
pub mod interpreter;
pub mod topology;
