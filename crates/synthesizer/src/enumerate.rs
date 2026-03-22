//! Brute-force enumerative synthesizer.
//!
//! Enumerates all FSTs up to a given size and tests each one against
//! the specification via simulation. This is the simplest possible
//! synthesizer — a baseline to build upon.

use agent_model::fst::{Fst, FstEnumerator};
use agent_model::interpreter::{simulate, Trace};
use agent_model::topology::Topology;
use rand::Rng;
use tracing::{debug, info};

/// Configuration for the enumerative synthesizer.
#[derive(Debug, Clone)]
pub struct EnumerateConfig {
    /// Maximum number of FST states to try.
    pub max_states: u32,
    /// Number of observation symbols.
    pub num_observations: u32,
    /// Number of action symbols.
    pub num_actions: u32,
    /// Topology to test on.
    pub topology: Topology,
    /// Number of agents for testing.
    pub test_agents: usize,
    /// Number of random initial conditions to test per candidate.
    pub test_runs: u32,
    /// Maximum simulation steps.
    pub max_steps: u64,
    /// Range of initial values.
    pub value_range: i64,
    /// Minimum success rate to accept a candidate.
    pub min_success_rate: f64,
}

impl Default for EnumerateConfig {
    fn default() -> Self {
        Self {
            max_states: 3,
            num_observations: 2,
            num_actions: 1,
            topology: Topology::Complete,
            test_agents: 5,
            test_runs: 50,
            max_steps: 50,
            value_range: 2,
            min_success_rate: 0.95,
        }
    }
}

/// Result of synthesis.
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// The synthesized FST (if found).
    pub fst: Option<Fst>,
    /// Number of candidates evaluated.
    pub candidates_evaluated: u64,
    /// Success rate of the best candidate.
    pub best_success_rate: f64,
    /// Total enumeration time in seconds.
    pub elapsed_seconds: f64,
}

/// Enumerate FSTs and find one that satisfies the property.
pub fn synthesize(config: &EnumerateConfig, checker: impl Fn(&Trace) -> bool) -> SynthesisResult {
    let start = std::time::Instant::now();

    let mut best_fst: Option<Fst> = None;
    let mut best_rate = 0.0;
    let mut total_candidates = 0u64;

    for num_states in 1..=config.max_states {
        let mut enumerator =
            FstEnumerator::new(num_states, config.num_observations, config.num_actions);

        let total = enumerator.total_count();
        info!(
            "Enumerating {} FSTs with {} states, {} observations, {} actions",
            total, num_states, config.num_observations, config.num_actions
        );

        if total > 1_000_000 {
            info!("Skipping: search space too large ({})", total);
            continue;
        }

        loop {
            let fst = match enumerator.current_fst() {
                Some(f) => f,
                None => break,
            };

            total_candidates += 1;

            // Skip degenerate FSTs
            if fst.is_degenerate() {
                if !enumerator.advance() {
                    break;
                }
                continue;
            }

            // Test this FST
            let success_rate = test_fst(&fst, config, &checker);

            if success_rate > best_rate {
                best_rate = success_rate;
                best_fst = Some(fst.clone());
                debug!(
                    "New best: {} states, rate {:.2}%",
                    num_states,
                    success_rate * 100.0
                );
            }

            if success_rate >= config.min_success_rate {
                info!(
                    "Found satisfying FST with {} states (rate {:.2}%)",
                    num_states,
                    success_rate * 100.0
                );
                return SynthesisResult {
                    fst: Some(fst),
                    candidates_evaluated: total_candidates,
                    best_success_rate: success_rate,
                    elapsed_seconds: start.elapsed().as_secs_f64(),
                };
            }

            if !enumerator.advance() {
                break;
            }
        }
    }

    SynthesisResult {
        fst: best_fst,
        candidates_evaluated: total_candidates,
        best_success_rate: best_rate,
        elapsed_seconds: start.elapsed().as_secs_f64(),
    }
}

/// Test an FST on multiple random initial conditions.
fn test_fst(fst: &Fst, config: &EnumerateConfig, checker: &impl Fn(&Trace) -> bool) -> f64 {
    let mut rng = rand::thread_rng();
    let mut successes = 0u32;

    for _ in 0..config.test_runs {
        let initial_values: Vec<i64> = (0..config.test_agents)
            .map(|_| rng.gen_range(0..config.value_range))
            .collect();

        let trace = simulate(
            fst,
            config.test_agents,
            &config.topology,
            &initial_values,
            config.max_steps,
        );

        if checker(&trace) {
            successes += 1;
        }
    }

    successes as f64 / config.test_runs as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesize_consensus() {
        let config = EnumerateConfig {
            max_states: 2,
            num_observations: 2,
            num_actions: 1,
            topology: Topology::Complete,
            test_agents: 3,
            test_runs: 20,
            max_steps: 30,
            value_range: 2,
            min_success_rate: 0.8,
        };

        let result = synthesize(&config, |trace| trace.eventually_globally_agree());

        println!(
            "Evaluated {} candidates in {:.2}s, best rate: {:.2}%",
            result.candidates_evaluated,
            result.elapsed_seconds,
            result.best_success_rate * 100.0
        );

        assert!(result.fst.is_some(), "Should find at least one FST");
        assert!(result.best_success_rate >= 0.8);
    }
}
