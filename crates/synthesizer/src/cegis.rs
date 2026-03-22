//! CEGIS (Counter-Example Guided Inductive Synthesis) synthesizer.
//!
//! Uses Z3 SMT solver to synthesize FST transitions, guided by
//! counterexamples from simulation-based testing.
//!
//! Algorithm:
//! 1. Ask Z3 to propose FST transitions satisfying structural constraints
//! 2. Simulate the proposed FST on random initial conditions
//! 3. If simulation finds a counterexample (spec violation):
//!    - Extract the failing initial condition
//!    - Add it as a constraint: "the FST must work on this initial condition"
//!    - Go to step 1
//! 4. If no counterexample found after many runs: return the FST

use agent_model::fst::{Fst, StateId};
use agent_model::interpreter::{simulate, Trace};
use agent_model::topology::Topology;
use rand::Rng;
use tracing::{debug, info, warn};
use z3::ast::{Ast, Int};
use z3::{Config, Context, Optimize, SatResult};

/// Configuration for the CEGIS synthesizer.
#[derive(Debug, Clone)]
pub struct CegisConfig {
    /// Number of FST states.
    pub num_states: u32,
    /// Number of observation symbols.
    pub num_observations: u32,
    /// Number of action symbols.
    pub num_actions: u32,
    /// Topology for simulation testing.
    pub topology: Topology,
    /// Number of agents for testing.
    pub test_agents: usize,
    /// Maximum simulation steps.
    pub max_steps: u64,
    /// Range of initial values.
    pub value_range: i64,
    /// Maximum CEGIS iterations.
    pub max_iterations: u32,
    /// Number of random tests per iteration.
    pub tests_per_iteration: u32,
    /// Minimum success rate to accept.
    pub min_success_rate: f64,
}

impl Default for CegisConfig {
    fn default() -> Self {
        Self {
            num_states: 3,
            num_observations: 2,
            num_actions: 1,
            topology: Topology::Complete,
            test_agents: 5,
            max_steps: 50,
            value_range: 2,
            max_iterations: 50,
            tests_per_iteration: 100,
            min_success_rate: 0.95,
        }
    }
}

/// Result of CEGIS synthesis.
#[derive(Debug, Clone)]
pub struct CegisResult {
    pub fst: Option<Fst>,
    pub iterations: u32,
    pub counterexamples_used: usize,
    pub final_success_rate: f64,
    pub elapsed_seconds: f64,
}

/// Run CEGIS synthesis.
pub fn synthesize_cegis(config: &CegisConfig, checker: impl Fn(&Trace) -> bool) -> CegisResult {
    let start = std::time::Instant::now();

    let z3_cfg = Config::new();
    let ctx = Context::new(&z3_cfg);
    let solver = Optimize::new(&ctx);

    let ns = config.num_states as usize;
    let no = config.num_observations as usize;

    // Create Z3 variables for the transition table:
    // next_state[s][o] = the next state when in state s and observing o
    // action[s][o] = the action taken
    let mut next_state_vars = Vec::new();
    let mut action_vars = Vec::new();

    for s in 0..ns {
        let mut row_ns = Vec::new();
        let mut row_act = Vec::new();
        for o in 0..no {
            let ns_var = Int::new_const(&ctx, format!("ns_{}_{}", s, o));
            let act_var = Int::new_const(&ctx, format!("act_{}_{}", s, o));

            // Bound constraints: 0 <= next_state < num_states, 0 <= action < num_actions
            solver.assert(&ns_var.ge(&Int::from_i64(&ctx, 0)));
            solver.assert(&ns_var.lt(&Int::from_i64(&ctx, config.num_states as i64)));
            solver.assert(&act_var.ge(&Int::from_i64(&ctx, 0)));
            solver.assert(&act_var.lt(&Int::from_i64(&ctx, config.num_actions as i64)));

            row_ns.push(ns_var);
            row_act.push(act_var);
        }
        next_state_vars.push(row_ns);
        action_vars.push(row_act);
    }

    // Output values for each state
    let mut output_vars = Vec::new();
    for s in 0..ns {
        let out_var = Int::new_const(&ctx, format!("out_{}", s));
        solver.assert(&out_var.ge(&Int::from_i64(&ctx, 0)));
        solver.assert(&out_var.lt(&Int::from_i64(&ctx, config.value_range)));
        output_vars.push(out_var);
    }

    // Anti-degeneracy constraints:

    // 1. Not all outputs are the same
    if ns >= 2 {
        let mut any_different = z3::ast::Bool::from_bool(&ctx, false);
        for s in 1..ns {
            let diff = output_vars[s]._eq(&output_vars[0]).not();
            any_different = z3::ast::Bool::or(&ctx, &[&any_different, &diff]);
        }
        solver.assert(&any_different);
    }

    // 2. Observations matter: for at least one state, different observations lead to different next states
    if no >= 2 {
        let mut obs_matters = z3::ast::Bool::from_bool(&ctx, false);
        for s in 0..ns {
            let diff = next_state_vars[s][0]._eq(&next_state_vars[s][1]).not();
            obs_matters = z3::ast::Bool::or(&ctx, &[&obs_matters, &diff]);
        }
        solver.assert(&obs_matters);
    }

    // 3. At least two states are reachable from initial state (state 0)
    // Soft constraint: maximize the number of distinct reachable states
    if ns >= 2 {
        let mut any_reaches_other = z3::ast::Bool::from_bool(&ctx, false);
        for o in 0..no {
            let reaches = next_state_vars[0][o]._eq(&Int::from_i64(&ctx, 0)).not();
            any_reaches_other = z3::ast::Bool::or(&ctx, &[&any_reaches_other, &reaches]);
        }
        solver.assert(&any_reaches_other);
    }

    let mut counterexamples: Vec<Vec<i64>> = Vec::new();
    let mut best_fst: Option<Fst> = None;
    let mut best_rate = 0.0;

    for iteration in 0..config.max_iterations {
        debug!("CEGIS iteration {}", iteration);

        // Solve
        match solver.check(&[]) {
            SatResult::Sat => {
                let model = solver.get_model().unwrap();

                // Extract FST from Z3 model
                let mut fst = Fst::new(
                    "cegis_synthesized",
                    config.num_states,
                    config.num_observations,
                    config.num_actions,
                );

                for s in 0..ns {
                    for o in 0..no {
                        let next = model
                            .eval(&next_state_vars[s][o], true)
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0) as StateId;
                        let act = model
                            .eval(&action_vars[s][o], true)
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0) as u32;
                        fst.add_transition(s as u32, o as u32, next, act);
                    }

                    let out = model
                        .eval(&output_vars[s], true)
                        .and_then(|v| v.as_i64())
                        .unwrap_or(s as i64);
                    fst.set_output(s as u32, out);
                }

                // Test the FST
                let rate = test_fst_rate(&fst, config, &checker);
                debug!("Candidate FST success rate: {:.1}%", rate * 100.0);

                if rate > best_rate {
                    best_rate = rate;
                    best_fst = Some(fst.clone());
                }

                if rate >= config.min_success_rate {
                    info!(
                        "CEGIS converged after {} iterations with {:.1}% success rate",
                        iteration + 1,
                        rate * 100.0
                    );
                    return CegisResult {
                        fst: Some(fst),
                        iterations: iteration + 1,
                        counterexamples_used: counterexamples.len(),
                        final_success_rate: rate,
                        elapsed_seconds: start.elapsed().as_secs_f64(),
                    };
                }

                // Find counterexamples
                let new_cex = find_counterexamples(&fst, config, &checker);
                if new_cex.is_empty() {
                    info!("No more counterexamples found but rate below threshold");
                    // Try with more random tests
                    continue;
                }

                // Add counterexample constraints
                for cex in &new_cex {
                    add_counterexample_constraint(
                        &ctx,
                        &solver,
                        &next_state_vars,
                        &output_vars,
                        config,
                        cex,
                    );
                    counterexamples.push(cex.clone());
                }

                debug!(
                    "Added {} counterexamples (total: {})",
                    new_cex.len(),
                    counterexamples.len()
                );
            }
            SatResult::Unsat => {
                warn!("Z3 returned UNSAT — constraints are unsatisfiable");
                break;
            }
            SatResult::Unknown => {
                warn!("Z3 returned UNKNOWN");
                break;
            }
        }
    }

    CegisResult {
        fst: best_fst,
        iterations: config.max_iterations,
        counterexamples_used: counterexamples.len(),
        final_success_rate: best_rate,
        elapsed_seconds: start.elapsed().as_secs_f64(),
    }
}

/// Add a counterexample constraint to the solver.
///
/// The constraint encodes: "running the FST on this initial condition
/// must eventually reach consensus" — but since we can't encode full
/// simulation in SMT, we use a bounded unrolling approach.
///
/// We unroll the simulation for a fixed number of steps and require
/// that by the last step all agents have the same output value.
fn add_counterexample_constraint(
    ctx: &Context,
    solver: &Optimize,
    next_state_vars: &[Vec<Int>],
    output_vars: &[Int],
    config: &CegisConfig,
    initial_values: &[i64],
) {
    let n = initial_values.len();
    let ns = config.num_states as usize;
    let unroll_steps = 5.min(config.max_steps as usize); // bounded unrolling

    // Create state variables for each agent at each step
    // agent_state[t][i] = state of agent i at time t
    let mut agent_states: Vec<Vec<Int>> = Vec::new();

    // Initial states: map initial values to FST states
    let step0: Vec<Int> = (0..n)
        .map(|i| {
            let init_state = (initial_values[i].unsigned_abs() % config.num_states as u64) as i64;
            Int::from_i64(ctx, init_state)
        })
        .collect();
    agent_states.push(step0);

    // Unroll simulation
    for t in 0..unroll_steps {
        let mut next_states = Vec::new();
        for i in 0..n {
            // Compute observation for agent i: majority output among neighbors
            // For complete graph: majority of all other agents
            // Simplified: use the output value of agent (i+1)%n as observation proxy
            let neighbor = (i + 1) % n;

            // Build ITE chain: for each possible (state, obs) pair,
            // select the corresponding next_state
            let cur_state = &agent_states[t][i];
            let neighbor_state = &agent_states[t][neighbor];

            // Get neighbor's output as observation
            let mut obs_expr = Int::from_i64(ctx, 0);
            for s in 0..ns {
                let is_this_state = neighbor_state._eq(&Int::from_i64(ctx, s as i64));
                obs_expr = is_this_state.ite(
                    &Int::from_i64(
                        ctx,
                        s as i64 % config.num_observations as i64,
                    ),
                    &obs_expr,
                );
            }

            // Now select next_state based on (cur_state, obs)
            let mut next_expr = Int::from_i64(ctx, 0);
            for s in (0..ns).rev() {
                for o in (0..config.num_observations as usize).rev() {
                    let is_state = cur_state._eq(&Int::from_i64(ctx, s as i64));
                    let is_obs = obs_expr._eq(&Int::from_i64(ctx, o as i64));
                    let both = z3::ast::Bool::and(ctx, &[&is_state, &is_obs]);
                    next_expr = both.ite(&next_state_vars[s][o], &next_expr);
                }
            }

            next_states.push(next_expr);
        }
        agent_states.push(next_states);
    }

    // Final constraint: at the last step, all agents should have the same output
    let last = &agent_states[unroll_steps];
    if n >= 2 {
        // Get output of first agent
        let mut out0 = Int::from_i64(ctx, 0);
        for s in (0..ns).rev() {
            let is_state = last[0]._eq(&Int::from_i64(ctx, s as i64));
            out0 = is_state.ite(&output_vars[s], &out0);
        }

        // All other agents must have same output
        for i in 1..n {
            let mut out_i = Int::from_i64(ctx, 0);
            for s in (0..ns).rev() {
                let is_state = last[i]._eq(&Int::from_i64(ctx, s as i64));
                out_i = is_state.ite(&output_vars[s], &out_i);
            }
            solver.assert(&out0._eq(&out_i));
        }
    }
}

/// Test an FST on random initial conditions and return success rate.
fn test_fst_rate(fst: &Fst, config: &CegisConfig, checker: &impl Fn(&Trace) -> bool) -> f64 {
    let mut rng = rand::thread_rng();
    let mut successes = 0u32;

    for _ in 0..config.tests_per_iteration {
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

    successes as f64 / config.tests_per_iteration as f64
}

/// Find counterexamples: initial conditions where the FST fails.
fn find_counterexamples(
    fst: &Fst,
    config: &CegisConfig,
    checker: &impl Fn(&Trace) -> bool,
) -> Vec<Vec<i64>> {
    let mut rng = rand::thread_rng();
    let mut counterexamples = Vec::new();

    for _ in 0..config.tests_per_iteration {
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

        if !checker(&trace) {
            counterexamples.push(initial_values);
            if counterexamples.len() >= 5 {
                break; // Don't add too many at once
            }
        }
    }

    counterexamples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cegis_consensus() {
        let config = CegisConfig {
            num_states: 3,
            num_observations: 2,
            num_actions: 1,
            topology: Topology::Complete,
            test_agents: 3,
            max_steps: 30,
            value_range: 2,
            max_iterations: 20,
            tests_per_iteration: 50,
            min_success_rate: 0.80,
        };

        let result = synthesize_cegis(&config, |trace| trace.eventually_globally_agree());

        println!(
            "CEGIS: {} iterations, {} counterexamples, rate {:.1}%",
            result.iterations, result.counterexamples_used, result.final_success_rate * 100.0
        );

        assert!(result.fst.is_some(), "CEGIS should find an FST");
        // The FST should not be degenerate
        if let Some(fst) = &result.fst {
            assert!(!fst.is_degenerate(), "CEGIS result should not be degenerate");
        }
    }
}
