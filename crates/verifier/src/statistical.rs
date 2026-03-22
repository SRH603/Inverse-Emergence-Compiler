//! Statistical verification via Monte Carlo simulation.
//!
//! When formal verification is infeasible, run many simulations with random
//! initial conditions and compute statistical guarantees.

use agent_model::fst::Fst;
use agent_model::interpreter::{simulate, Trace};
use agent_model::topology::Topology;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for statistical verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConfig {
    /// Number of simulation runs.
    pub num_runs: u32,
    /// Number of agents per run.
    pub num_agents: usize,
    /// Maximum simulation steps per run.
    pub max_steps: u64,
    /// Number of possible initial values.
    pub value_range: i64,
    /// Topology to test on.
    pub topology: Topology,
    /// Number of agents to remove for fault tolerance testing (0 = no faults).
    pub fault_removals: usize,
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            num_runs: 1000,
            num_agents: 10,
            max_steps: 100,
            value_range: 2,
            topology: Topology::Complete,
            fault_removals: 0,
        }
    }
}

/// Result of statistical verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalResult {
    pub total_runs: u32,
    pub successful_runs: u32,
    pub success_rate: f64,
    /// Bayesian 99% lower confidence bound on the true success probability.
    pub confidence_lower_bound: f64,
    /// Average convergence step (among successful runs).
    pub avg_convergence_step: Option<f64>,
    /// Maximum convergence step (among successful runs).
    pub max_convergence_step: Option<u64>,
    /// Failure modes: initial conditions that led to failure.
    pub failure_examples: Vec<Vec<i64>>,
}

/// Run statistical verification of an FST against a property.
pub fn verify_statistically(
    fst: &Fst,
    config: &StatisticalConfig,
    checker: impl Fn(&Trace) -> bool + Sync,
) -> StatisticalResult {
    let results: Vec<(bool, Option<u64>, Vec<i64>)> = (0..config.num_runs)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();

            // Random initial values
            let initial_values: Vec<i64> = (0..config.num_agents)
                .map(|_| rng.gen_range(0..config.value_range))
                .collect();

            let trace = simulate(
                fst,
                config.num_agents,
                &config.topology,
                &initial_values,
                config.max_steps,
            );

            let success = checker(&trace);
            let convergence = trace.convergence_step();

            (success, convergence, initial_values)
        })
        .collect();

    let total = config.num_runs;
    let successful = results.iter().filter(|(s, _, _)| *s).count() as u32;
    let success_rate = successful as f64 / total as f64;

    // Bayesian confidence bound: Beta(successful + 1, failed + 1), 1% quantile
    let confidence_lower_bound = beta_quantile(
        successful as f64 + 1.0,
        (total - successful) as f64 + 1.0,
        0.01,
    );

    let successful_convergences: Vec<u64> = results
        .iter()
        .filter_map(|(s, c, _)| if *s { *c } else { None })
        .collect();

    let avg_convergence_step = if successful_convergences.is_empty() {
        None
    } else {
        Some(
            successful_convergences.iter().sum::<u64>() as f64
                / successful_convergences.len() as f64,
        )
    };

    let max_convergence_step = successful_convergences.iter().max().copied();

    let failure_examples: Vec<Vec<i64>> = results
        .iter()
        .filter(|(s, _, _)| !*s)
        .take(5)
        .map(|(_, _, vals)| vals.clone())
        .collect();

    StatisticalResult {
        total_runs: total,
        successful_runs: successful,
        success_rate,
        confidence_lower_bound,
        avg_convergence_step,
        max_convergence_step,
        failure_examples,
    }
}

/// Approximate quantile of the Beta distribution using the normal approximation.
/// For Beta(a, b), the p-th quantile.
fn beta_quantile(a: f64, b: f64, p: f64) -> f64 {
    // Normal approximation to Beta distribution
    let mean = a / (a + b);
    let variance = (a * b) / ((a + b).powi(2) * (a + b + 1.0));
    let std_dev = variance.sqrt();

    // Approximate z-score for p
    let z = normal_quantile(p);

    let result = mean + z * std_dev;
    result.clamp(0.0, 1.0)
}

/// Approximate quantile of the standard normal distribution (Abramowitz & Stegun).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Rational approximation (Abramowitz & Stegun 26.2.23)
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -result
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_majority_rule_statistical() {
        // Simple majority rule FST
        let mut fst = Fst::new("majority", 2, 2, 1);
        fst.add_transition(0, 0, 0, 0);
        fst.add_transition(0, 1, 1, 0);
        fst.add_transition(1, 0, 0, 0);
        fst.add_transition(1, 1, 1, 0);
        fst.set_output(0, 0);
        fst.set_output(1, 1);

        let config = StatisticalConfig {
            num_runs: 100,
            num_agents: 5,
            max_steps: 50,
            value_range: 2,
            topology: Topology::Complete,
            fault_removals: 0,
        };

        let result = verify_statistically(&fst, &config, |trace| {
            trace.eventually_globally_agree()
        });

        // Majority rule on complete graph should work most of the time
        assert!(result.success_rate > 0.5);
    }
}
