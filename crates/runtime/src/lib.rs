//! Runtime — execution engine for running verified local rules in simulation.
//!
//! Provides [`simulation::run_scenario`] to tie together FST execution,
//! topology setup, and property monitoring into a single scenario runner.

pub mod simulation;
