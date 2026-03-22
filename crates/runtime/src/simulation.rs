//! High-level simulation runner that ties together FST + topology + verification.

use agent_model::fst::Fst;
use agent_model::interpreter::{simulate, Trace};
use agent_model::topology::Topology;
use serde::{Deserialize, Serialize};
use verifier::monitor::{self, MonitorProperty, MonitorResult};

/// A simulation scenario to run and verify.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub name: String,
    pub num_agents: usize,
    pub topology: Topology,
    pub initial_values: Vec<i64>,
    pub max_steps: u64,
    pub properties: Vec<MonitorProperty>,
}

/// Result of running a scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub name: String,
    pub trace_length: usize,
    pub property_results: Vec<MonitorResult>,
    pub all_satisfied: bool,
}

/// Run a scenario: simulate the FST and check all properties.
pub fn run_scenario(fst: &Fst, scenario: &Scenario) -> ScenarioResult {
    let trace = simulate(
        fst,
        scenario.num_agents,
        &scenario.topology,
        &scenario.initial_values,
        scenario.max_steps,
    );

    let property_results: Vec<MonitorResult> = scenario
        .properties
        .iter()
        .map(|prop| monitor::check_property(&trace, prop))
        .collect();

    let all_satisfied = property_results.iter().all(|r| r.satisfied);

    ScenarioResult {
        name: scenario.name.clone(),
        trace_length: trace.snapshots.len(),
        property_results,
        all_satisfied,
    }
}

/// Print a trace in human-readable format.
pub fn print_trace(trace: &Trace) {
    for snap in &trace.snapshots {
        let values: Vec<String> = snap.agents.iter().map(|a| a.value.to_string()).collect();
        println!("  Step {}: [{}]", snap.step, values.join(", "));
    }
}
