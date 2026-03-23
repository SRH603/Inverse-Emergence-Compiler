//! Runtime monitoring of ETL properties on simulation traces.

use agent_model::interpreter::Trace;
use serde::{Deserialize, Serialize};

/// A property to monitor during simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitorProperty {
    /// All agents agree on the same value.
    AllAgree,
    /// At least `fraction` of agents have the same value.
    FractionAgree(f64),
    /// The system has converged (stable for `window` steps).
    Stable { window: u64 },
    /// No agent has a specific forbidden value.
    NeverValue(i64),
}

/// Result of monitoring a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorResult {
    pub property: MonitorProperty,
    pub satisfied: bool,
    /// Step at which the property was first satisfied (or violated).
    pub relevant_step: Option<u64>,
    pub details: String,
}

/// Check a property against a trace.
pub fn check_property(trace: &Trace, property: &MonitorProperty) -> MonitorResult {
    match property {
        MonitorProperty::AllAgree => check_eventually_globally_agree(trace),
        MonitorProperty::FractionAgree(f) => check_fraction_agree(trace, *f),
        MonitorProperty::Stable { window } => check_stable(trace, *window),
        MonitorProperty::NeverValue(v) => check_never_value(trace, *v),
    }
}

fn check_eventually_globally_agree(trace: &Trace) -> MonitorResult {
    for (i, snap) in trace.snapshots.iter().enumerate() {
        if snap.all_agree() && trace.snapshots[i..].iter().all(|s| s.all_agree()) {
            return MonitorResult {
                property: MonitorProperty::AllAgree,
                satisfied: true,
                relevant_step: Some(snap.step),
                details: format!(
                    "Consensus reached at step {} with value {}",
                    snap.step,
                    snap.agents.first().map_or(0, |a| a.value)
                ),
            };
        }
    }
    MonitorResult {
        property: MonitorProperty::AllAgree,
        satisfied: false,
        relevant_step: None,
        details: "Consensus never reached".into(),
    }
}

fn check_fraction_agree(trace: &Trace, threshold: f64) -> MonitorResult {
    for snap in &trace.snapshots {
        if let Some(majority) = snap.majority_value() {
            let frac = snap.fraction_with_value(majority);
            if frac >= threshold {
                return MonitorResult {
                    property: MonitorProperty::FractionAgree(threshold),
                    satisfied: true,
                    relevant_step: Some(snap.step),
                    details: format!(
                        "Fraction {:.2} >= {:.2} at step {} (value {})",
                        frac, threshold, snap.step, majority
                    ),
                };
            }
        }
    }
    MonitorResult {
        property: MonitorProperty::FractionAgree(threshold),
        satisfied: false,
        relevant_step: None,
        details: format!("Fraction never reached {:.2}", threshold),
    }
}

fn check_stable(trace: &Trace, window: u64) -> MonitorResult {
    if trace.snapshots.len() < window as usize {
        return MonitorResult {
            property: MonitorProperty::Stable { window },
            satisfied: false,
            relevant_step: None,
            details: format!("Trace too short for window {}", window),
        };
    }

    for i in 0..trace.snapshots.len().saturating_sub(window as usize) {
        let ref_values: Vec<i64> = trace.snapshots[i].agents.iter().map(|a| a.value).collect();
        let mut stable = true;
        for j in 1..=(window as usize) {
            let cur_values: Vec<i64> = trace.snapshots[i + j]
                .agents
                .iter()
                .map(|a| a.value)
                .collect();
            if cur_values != ref_values {
                stable = false;
                break;
            }
        }
        if stable {
            return MonitorResult {
                property: MonitorProperty::Stable { window },
                satisfied: true,
                relevant_step: Some(trace.snapshots[i].step),
                details: format!("Stable for {} steps starting at step {}", window, i),
            };
        }
    }

    MonitorResult {
        property: MonitorProperty::Stable { window },
        satisfied: false,
        relevant_step: None,
        details: format!("Never stable for {} consecutive steps", window),
    }
}

fn check_never_value(trace: &Trace, forbidden: i64) -> MonitorResult {
    for snap in &trace.snapshots {
        for agent in &snap.agents {
            if agent.value == forbidden {
                return MonitorResult {
                    property: MonitorProperty::NeverValue(forbidden),
                    satisfied: false,
                    relevant_step: Some(snap.step),
                    details: format!(
                        "Agent {} had forbidden value {} at step {}",
                        agent.id, forbidden, snap.step
                    ),
                };
            }
        }
    }
    MonitorResult {
        property: MonitorProperty::NeverValue(forbidden),
        satisfied: true,
        relevant_step: None,
        details: format!("Value {} never appeared", forbidden),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_model::interpreter::{AgentState, GlobalSnapshot, Trace};

    fn make_trace(steps: Vec<Vec<i64>>) -> Trace {
        let snapshots = steps
            .into_iter()
            .enumerate()
            .map(|(step, values)| GlobalSnapshot {
                step: step as u64,
                agents: values
                    .into_iter()
                    .enumerate()
                    .map(|(id, value)| AgentState {
                        id,
                        fst_state: value.unsigned_abs() as u32,
                        value,
                    })
                    .collect(),
            })
            .collect();
        Trace { snapshots }
    }

    #[test]
    fn test_all_agree_satisfied() {
        let trace = make_trace(vec![
            vec![0, 1, 0],
            vec![0, 0, 0],
            vec![0, 0, 0],
        ]);
        let result = check_property(&trace, &MonitorProperty::AllAgree);
        assert!(result.satisfied);
        assert_eq!(result.relevant_step, Some(1));
    }

    #[test]
    fn test_all_agree_not_satisfied() {
        let trace = make_trace(vec![
            vec![0, 1, 0],
            vec![0, 1, 0],
            vec![1, 0, 1],
        ]);
        let result = check_property(&trace, &MonitorProperty::AllAgree);
        assert!(!result.satisfied);
    }

    #[test]
    fn test_fraction_agree() {
        let trace = make_trace(vec![vec![0, 0, 0, 1]]);
        let result = check_property(&trace, &MonitorProperty::FractionAgree(0.7));
        assert!(result.satisfied); // 75% have value 0
    }

    #[test]
    fn test_fraction_agree_fails() {
        let trace = make_trace(vec![vec![0, 1, 2, 3]]);
        let result = check_property(&trace, &MonitorProperty::FractionAgree(0.5));
        assert!(!result.satisfied); // only 25% per value
    }

    #[test]
    fn test_stable() {
        let trace = make_trace(vec![
            vec![0, 1],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
        ]);
        let result = check_property(&trace, &MonitorProperty::Stable { window: 2 });
        assert!(result.satisfied);
    }

    #[test]
    fn test_never_value_satisfied() {
        let trace = make_trace(vec![vec![0, 1, 2], vec![0, 1, 2]]);
        let result = check_property(&trace, &MonitorProperty::NeverValue(5));
        assert!(result.satisfied);
    }

    #[test]
    fn test_never_value_violated() {
        let trace = make_trace(vec![vec![0, 1, 5]]);
        let result = check_property(&trace, &MonitorProperty::NeverValue(5));
        assert!(!result.satisfied);
        assert_eq!(result.relevant_step, Some(0));
    }

    #[test]
    fn test_empty_trace() {
        let trace = Trace {
            snapshots: Vec::new(),
        };
        let result = check_property(&trace, &MonitorProperty::AllAgree);
        assert!(!result.satisfied);
    }

    #[test]
    fn test_single_agent() {
        let trace = make_trace(vec![vec![42], vec![42]]);
        let result = check_property(&trace, &MonitorProperty::AllAgree);
        assert!(result.satisfied);
    }
}
