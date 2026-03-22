//! Multi-agent simulation interpreter.
//!
//! Runs N agents, each executing the same FST, on a given topology.
//! Records the trace of global states for verification.

use crate::fst::{Fst, ObservationId, StateId};
use crate::topology::Topology;
use serde::{Deserialize, Serialize};

/// The state of one agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub id: usize,
    pub fst_state: StateId,
    pub value: i64,
}

/// A snapshot of the entire system at one time step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalSnapshot {
    pub step: u64,
    pub agents: Vec<AgentState>,
}

impl GlobalSnapshot {
    /// Check if all agents have the same output value (consensus).
    pub fn all_agree(&self) -> bool {
        if self.agents.is_empty() {
            return true;
        }
        let first = self.agents[0].value;
        self.agents.iter().all(|a| a.value == first)
    }

    /// Get the majority value (most common output).
    pub fn majority_value(&self) -> Option<i64> {
        if self.agents.is_empty() {
            return None;
        }
        let mut counts = std::collections::HashMap::new();
        for a in &self.agents {
            *counts.entry(a.value).or_insert(0u32) += 1;
        }
        counts.into_iter().max_by_key(|&(_, c)| c).map(|(v, _)| v)
    }

    /// Fraction of agents with a given value.
    pub fn fraction_with_value(&self, value: i64) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        let count = self.agents.iter().filter(|a| a.value == value).count();
        count as f64 / self.agents.len() as f64
    }
}

/// A complete simulation trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub snapshots: Vec<GlobalSnapshot>,
}

impl Trace {
    /// Check if consensus is eventually reached and maintained (eventually globally all_agree).
    pub fn eventually_globally_agree(&self) -> bool {
        for (i, snap) in self.snapshots.iter().enumerate() {
            if snap.all_agree() {
                // Check that all subsequent snapshots also agree
                if self.snapshots[i..].iter().all(|s| s.all_agree()) {
                    return true;
                }
            }
        }
        false
    }

    /// Find the first step at which consensus is reached (and maintained).
    pub fn convergence_step(&self) -> Option<u64> {
        for (i, snap) in self.snapshots.iter().enumerate() {
            if snap.all_agree() && self.snapshots[i..].iter().all(|s| s.all_agree()) {
                return Some(snap.step);
            }
        }
        None
    }
}

/// The observation function: how an agent perceives its neighbors.
/// For consensus: returns the majority value among neighbors.
pub fn compute_observation(
    agent_id: usize,
    agents: &[AgentState],
    topology: &Topology,
    num_observations: u32,
) -> ObservationId {
    let neighbors = topology.neighbors(agent_id, agents.len());
    if neighbors.is_empty() {
        return 0;
    }

    // For consensus-type problems:
    // observation = majority value among neighbors, clamped to observation space
    let mut counts = std::collections::HashMap::new();
    for &n in &neighbors {
        *counts.entry(agents[n].value).or_insert(0u32) += 1;
    }

    let majority = counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(v, _)| v)
        .unwrap_or(0);

    // Map value to observation id (modular)
    (majority.unsigned_abs() % num_observations as u64) as ObservationId
}

/// Run a simulation: N agents running the same FST on a topology for max_steps.
pub fn simulate(
    fst: &Fst,
    n: usize,
    topology: &Topology,
    initial_values: &[i64],
    max_steps: u64,
) -> Trace {
    assert_eq!(initial_values.len(), n);

    // Initialize agents: map initial values to FST states
    let mut agents: Vec<AgentState> = initial_values
        .iter()
        .enumerate()
        .map(|(id, &val)| {
            let fst_state = (val.unsigned_abs() % fst.num_states as u64) as StateId;
            AgentState {
                id,
                fst_state,
                value: val,
            }
        })
        .collect();

    let mut snapshots = Vec::new();
    snapshots.push(GlobalSnapshot {
        step: 0,
        agents: agents.clone(),
    });

    for step in 1..=max_steps {
        let mut new_agents = agents.clone();

        for i in 0..n {
            let obs = compute_observation(i, &agents, topology, fst.num_observations);
            if let Some((next_state, _action)) = fst.step(agents[i].fst_state, obs) {
                new_agents[i].fst_state = next_state;
                new_agents[i].value = fst.get_output(next_state).unwrap_or(next_state as i64);
            }
        }

        agents = new_agents;
        snapshots.push(GlobalSnapshot {
            step,
            agents: agents.clone(),
        });

        // Early termination if consensus reached
        if snapshots.last().unwrap().all_agree() {
            // Run a few more steps to check stability
            let mut stable = true;
            for extra in 1..=5 {
                let mut extra_agents = agents.clone();
                for i in 0..n {
                    let obs =
                        compute_observation(i, &agents, topology, fst.num_observations);
                    if let Some((next_state, _)) = fst.step(agents[i].fst_state, obs) {
                        extra_agents[i].fst_state = next_state;
                        extra_agents[i].value =
                            fst.get_output(next_state).unwrap_or(next_state as i64);
                    }
                }
                let extra_snap = GlobalSnapshot {
                    step: step + extra,
                    agents: extra_agents.clone(),
                };
                if !extra_snap.all_agree()
                {
                    stable = false;
                    break;
                }
                agents = extra_agents;
                snapshots.push(GlobalSnapshot {
                    step: step + extra,
                    agents: agents.clone(),
                });
            }
            if stable {
                break;
            }
        }
    }

    Trace { snapshots }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fst::Fst;

    #[test]
    fn test_trivial_consensus() {
        // All agents start in same state → should agree immediately
        let mut fst = Fst::new("trivial", 2, 2, 1);
        // Stay in current state regardless of observation
        fst.add_transition(0, 0, 0, 0);
        fst.add_transition(0, 1, 0, 0);
        fst.add_transition(1, 0, 1, 0);
        fst.add_transition(1, 1, 1, 0);
        fst.set_output(0, 0);
        fst.set_output(1, 1);

        let topology = Topology::Complete;
        let trace = simulate(&fst, 3, &topology, &[0, 0, 0], 10);
        assert!(trace.snapshots[0].all_agree());
    }

    #[test]
    fn test_majority_consensus() {
        // Simple majority rule: adopt the majority value of neighbors
        let mut fst = Fst::new("majority", 2, 2, 1);
        // If observation is 0 (majority of neighbors have value 0), go to state 0
        fst.add_transition(0, 0, 0, 0);
        fst.add_transition(0, 1, 1, 0);
        fst.add_transition(1, 0, 0, 0);
        fst.add_transition(1, 1, 1, 0);
        fst.set_output(0, 0);
        fst.set_output(1, 1);

        let topology = Topology::Complete;
        // 2 agents with value 0, 1 with value 1 → should converge to 0
        let trace = simulate(&fst, 3, &topology, &[0, 0, 1], 20);
        assert!(trace.eventually_globally_agree());
    }
}
