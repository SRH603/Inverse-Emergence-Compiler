//! Verifier — verification of synthesized local rules against emergence specifications.
//!
//! - [`statistical`] — Monte Carlo verification with Bayesian confidence bounds
//! - [`monitor`] — runtime monitoring of ETL properties on simulation traces

pub mod monitor;
pub mod statistical;
