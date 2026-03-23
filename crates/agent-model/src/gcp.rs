//! Guarded Command Program (GCP) — a richer agent rule representation.
//!
//! A GCP extends FSTs with:
//! - Integer variables (bounded domain)
//! - Guards: boolean conditions on local state + neighbor observations
//! - Actions: variable assignments + message sends
//!
//! GCPs are more expressive than FSTs and can encode threshold-based protocols
//! (e.g., "if more than 2/3 of neighbors agree, adopt their value").

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A bounded integer variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub min: i64,
    pub max: i64,
    pub initial: i64,
}

/// A guard: a boolean condition on local state and observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Guard {
    /// Variable equals a constant.
    Eq(String, i64),
    /// Variable not equal to a constant.
    Ne(String, i64),
    /// Variable greater than or equal to a constant.
    Ge(String, i64),
    /// Variable less than a constant.
    Lt(String, i64),
    /// Count of neighbors satisfying a condition >= threshold.
    NeighborCount {
        /// Which neighbor variable to check.
        var: String,
        /// Value to compare against.
        value: i64,
        /// Minimum count required.
        threshold: usize,
    },
    /// Fraction of neighbors satisfying a condition >= threshold.
    NeighborFraction {
        var: String,
        value: i64,
        threshold: f64,
    },
    /// Conjunction of guards.
    And(Vec<Guard>),
    /// Disjunction of guards.
    Or(Vec<Guard>),
    /// Always true.
    True,
}

/// An action: what to do when a guard is satisfied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    /// Set a variable to a constant.
    Set(String, i64),
    /// Set a variable to the majority value of a neighbor variable.
    SetToNeighborMajority(String, String),
    /// Set a variable to the value of another local variable.
    Copy(String, String),
    /// No-op.
    Skip,
}

/// A guarded command: if guard then actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardedCommand {
    pub guard: Guard,
    pub actions: Vec<Action>,
    /// Priority: higher priority commands are evaluated first.
    pub priority: u32,
}

/// A Guarded Command Program defining an agent's local rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gcp {
    pub name: String,
    pub variables: Vec<Variable>,
    pub commands: Vec<GuardedCommand>,
}

/// Runtime state of one agent executing a GCP.
#[derive(Debug, Clone)]
pub struct GcpAgentState {
    pub id: usize,
    pub vars: HashMap<String, i64>,
}

impl Gcp {
    pub fn new(name: impl Into<String>, variables: Vec<Variable>) -> Self {
        Self {
            name: name.into(),
            variables,
            commands: Vec::new(),
        }
    }

    pub fn add_command(&mut self, cmd: GuardedCommand) {
        self.commands.push(cmd);
    }

    /// Create initial state for an agent with a given initial value override.
    pub fn initial_state(&self, id: usize, value_override: Option<i64>) -> GcpAgentState {
        let mut vars = HashMap::new();
        for v in &self.variables {
            vars.insert(v.name.clone(), v.initial);
        }
        if let Some(val) = value_override {
            if let Some(first_var) = self.variables.first() {
                vars.insert(first_var.name.clone(), val);
            }
        }
        GcpAgentState { id, vars }
    }

    /// Execute one step: evaluate guards and apply actions.
    pub fn step(
        &self,
        agent: &GcpAgentState,
        neighbor_states: &[&GcpAgentState],
    ) -> GcpAgentState {
        let mut new_state = agent.clone();

        // Sort commands by priority (highest first)
        let mut sorted: Vec<&GuardedCommand> = self.commands.iter().collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Execute the first matching command
        for cmd in &sorted {
            if self.evaluate_guard(&cmd.guard, agent, neighbor_states) {
                for action in &cmd.actions {
                    self.execute_action(action, &mut new_state, neighbor_states);
                }
                break; // only execute first matching command
            }
        }

        // Clamp variables to bounds
        for v in &self.variables {
            if let Some(val) = new_state.vars.get_mut(&v.name) {
                *val = (*val).clamp(v.min, v.max);
            }
        }

        new_state
    }

    fn evaluate_guard(
        &self,
        guard: &Guard,
        agent: &GcpAgentState,
        neighbors: &[&GcpAgentState],
    ) -> bool {
        match guard {
            Guard::True => true,
            Guard::Eq(var, val) => agent.vars.get(var).copied().unwrap_or(0) == *val,
            Guard::Ne(var, val) => agent.vars.get(var).copied().unwrap_or(0) != *val,
            Guard::Ge(var, val) => agent.vars.get(var).copied().unwrap_or(0) >= *val,
            Guard::Lt(var, val) => agent.vars.get(var).copied().unwrap_or(0) < *val,
            Guard::NeighborCount {
                var,
                value,
                threshold,
            } => {
                let count = neighbors
                    .iter()
                    .filter(|n| n.vars.get(var).copied().unwrap_or(0) == *value)
                    .count();
                count >= *threshold
            }
            Guard::NeighborFraction {
                var,
                value,
                threshold,
            } => {
                if neighbors.is_empty() {
                    return false;
                }
                let count = neighbors
                    .iter()
                    .filter(|n| n.vars.get(var).copied().unwrap_or(0) == *value)
                    .count();
                (count as f64 / neighbors.len() as f64) >= *threshold
            }
            Guard::And(guards) => guards.iter().all(|g| self.evaluate_guard(g, agent, neighbors)),
            Guard::Or(guards) => guards.iter().any(|g| self.evaluate_guard(g, agent, neighbors)),
        }
    }

    fn execute_action(
        &self,
        action: &Action,
        state: &mut GcpAgentState,
        neighbors: &[&GcpAgentState],
    ) {
        match action {
            Action::Set(var, val) => {
                state.vars.insert(var.clone(), *val);
            }
            Action::SetToNeighborMajority(target, source_var) => {
                if neighbors.is_empty() {
                    return;
                }
                // Count occurrences of each value
                let mut counts: HashMap<i64, usize> = HashMap::new();
                for n in neighbors {
                    let v = n.vars.get(source_var).copied().unwrap_or(0);
                    *counts.entry(v).or_insert(0) += 1;
                }
                if let Some((majority_val, _)) = counts.into_iter().max_by_key(|&(_, c)| c) {
                    state.vars.insert(target.clone(), majority_val);
                }
            }
            Action::Copy(target, source) => {
                if let Some(val) = state.vars.get(source).copied() {
                    state.vars.insert(target.clone(), val);
                }
            }
            Action::Skip => {}
        }
    }

    /// Get the "output value" of an agent (first variable's value).
    pub fn output_value(&self, agent: &GcpAgentState) -> i64 {
        self.variables
            .first()
            .and_then(|v| agent.vars.get(&v.name))
            .copied()
            .unwrap_or(0)
    }
}

/// Create a simple majority-vote GCP for consensus.
pub fn majority_consensus_gcp() -> Gcp {
    let mut gcp = Gcp::new(
        "majority_consensus",
        vec![Variable {
            name: "value".to_string(),
            min: 0,
            max: 1,
            initial: 0,
        }],
    );

    // Rule: adopt the majority value among neighbors
    gcp.add_command(GuardedCommand {
        guard: Guard::NeighborFraction {
            var: "value".to_string(),
            value: 1,
            threshold: 0.5,
        },
        actions: vec![Action::Set("value".to_string(), 1)],
        priority: 1,
    });

    gcp.add_command(GuardedCommand {
        guard: Guard::NeighborFraction {
            var: "value".to_string(),
            value: 0,
            threshold: 0.5,
        },
        actions: vec![Action::Set("value".to_string(), 0)],
        priority: 1,
    });

    // Default: keep current value
    gcp.add_command(GuardedCommand {
        guard: Guard::True,
        actions: vec![Action::Skip],
        priority: 0,
    });

    gcp
}

/// Create a threshold-based consensus GCP (Byzantine-tolerant style).
pub fn threshold_consensus_gcp(threshold_fraction: f64) -> Gcp {
    let mut gcp = Gcp::new(
        "threshold_consensus",
        vec![
            Variable {
                name: "value".to_string(),
                min: 0,
                max: 1,
                initial: 0,
            },
            Variable {
                name: "decided".to_string(),
                min: 0,
                max: 1,
                initial: 0,
            },
        ],
    );

    // If already decided, don't change
    gcp.add_command(GuardedCommand {
        guard: Guard::Eq("decided".to_string(), 1),
        actions: vec![Action::Skip],
        priority: 10,
    });

    // If threshold fraction of neighbors agree on value 1, decide 1
    gcp.add_command(GuardedCommand {
        guard: Guard::NeighborFraction {
            var: "value".to_string(),
            value: 1,
            threshold: threshold_fraction,
        },
        actions: vec![
            Action::Set("value".to_string(), 1),
            Action::Set("decided".to_string(), 1),
        ],
        priority: 5,
    });

    // If threshold fraction agree on 0, decide 0
    gcp.add_command(GuardedCommand {
        guard: Guard::NeighborFraction {
            var: "value".to_string(),
            value: 0,
            threshold: threshold_fraction,
        },
        actions: vec![
            Action::Set("value".to_string(), 0),
            Action::Set("decided".to_string(), 1),
        ],
        priority: 5,
    });

    // Otherwise, adopt majority
    gcp.add_command(GuardedCommand {
        guard: Guard::True,
        actions: vec![Action::SetToNeighborMajority(
            "value".to_string(),
            "value".to_string(),
        )],
        priority: 0,
    });

    gcp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_majority_consensus_gcp() {
        let gcp = majority_consensus_gcp();

        // 3 agents: two with value 0, one with value 1
        let agents: Vec<GcpAgentState> = vec![
            gcp.initial_state(0, Some(0)),
            gcp.initial_state(1, Some(0)),
            gcp.initial_state(2, Some(1)),
        ];

        // Agent 2 should adopt value 0 (majority of neighbors)
        let neighbors: Vec<&GcpAgentState> = vec![&agents[0], &agents[1]];
        let new_state = gcp.step(&agents[2], &neighbors);
        assert_eq!(gcp.output_value(&new_state), 0);
    }

    #[test]
    fn test_threshold_consensus() {
        let gcp = threshold_consensus_gcp(0.6);
        assert_eq!(gcp.commands.len(), 4);
        assert_eq!(gcp.variables.len(), 2);

        let agent = gcp.initial_state(0, Some(0));
        assert_eq!(gcp.output_value(&agent), 0);
    }

    #[test]
    fn test_guard_evaluation() {
        let gcp = majority_consensus_gcp();
        let agent = gcp.initial_state(0, Some(0));
        let n1 = gcp.initial_state(1, Some(1));
        let n2 = gcp.initial_state(2, Some(1));
        let neighbors: Vec<&GcpAgentState> = vec![&n1, &n2];

        // Both neighbors have value 1, so fraction >= 0.5 for value=1 should be true
        assert!(gcp.evaluate_guard(
            &Guard::NeighborFraction {
                var: "value".to_string(),
                value: 1,
                threshold: 0.5,
            },
            &agent,
            &neighbors,
        ));
    }

    #[test]
    fn test_gcp_convergence() {
        let gcp = majority_consensus_gcp();

        // 5 agents: 3 with value 0, 2 with value 1 → should converge to 0
        let mut agents: Vec<GcpAgentState> = (0..5)
            .map(|i| gcp.initial_state(i, Some(if i < 3 { 0 } else { 1 })))
            .collect();

        for _step in 0..20 {
            let mut new_agents = agents.clone();
            for i in 0..5 {
                let neighbors: Vec<&GcpAgentState> =
                    (0..5).filter(|&j| j != i).map(|j| &agents[j]).collect();
                new_agents[i] = gcp.step(&agents[i], &neighbors);
            }
            agents = new_agents;
        }

        // All should have value 0
        let values: Vec<i64> = agents.iter().map(|a| gcp.output_value(a)).collect();
        assert!(
            values.iter().all(|&v| v == values[0]),
            "should converge, got {:?}",
            values
        );
    }
}
