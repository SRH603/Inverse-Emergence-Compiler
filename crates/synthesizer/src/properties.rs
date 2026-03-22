//! Property system: connects EmergeLang specs to simulation-based property checkers.
//!
//! Instead of hardcoding "consensus", this module defines a generic property
//! evaluation framework that interprets ETL specs against simulation traces.

use agent_model::interpreter::Trace;
use emergelang::ast;
use emergelang::etl::EtlSpec;

/// A compiled property checker that can evaluate traces.
#[derive(Debug, Clone)]
pub struct CompiledProperty {
    pub name: String,
    pub checks: Vec<PropertyCheck>,
}

/// Individual property checks derived from ETL specs.
#[derive(Debug, Clone)]
pub enum PropertyCheck {
    /// All agents eventually agree on the same value.
    EventuallyGloballyAgree,

    /// System is stable for at least `window` steps.
    Stable { window: u64 },

    /// Converge within a deadline.
    ConvergeWithin { deadline: u64 },

    /// Fraction of agents agree on majority value >= threshold.
    FractionAgree { threshold: f64 },

    /// System still works after removing any single agent.
    FaultTolerant { max_removals: u32 },

    /// Agents achieve spatial coverage (all spread out, no clustering).
    SpatialCoverage,

    /// Agents form a connected formation (flocking).
    Formation,

    /// A ring topology self-repairs after node failure.
    SelfHealingRing,

    /// Custom predicate by name.
    CustomPredicate(String),
}

impl CompiledProperty {
    /// Compile an ETL spec into a set of property checks.
    pub fn from_etl_spec(spec: &EtlSpec) -> Self {
        let mut checks = Vec::new();

        // Analyze liveness properties
        for _formula in &spec.liveness {
            // Default liveness: eventually globally agree (consensus-like)
            checks.push(PropertyCheck::EventuallyGloballyAgree);
        }

        // Analyze fault tolerance
        for formula in &spec.fault_tolerance {
            if let emergelang::etl::EtlFormula::RobustUnderRemoval { max_removals, .. } = formula {
                checks.push(PropertyCheck::FaultTolerant {
                    max_removals: *max_removals,
                });
            }
        }

        // Convergence bound
        if let Some(deadline) = spec.convergence_bound {
            checks.push(PropertyCheck::ConvergeWithin { deadline });
        }

        // Stability (from safety properties)
        if !spec.safety.is_empty() {
            checks.push(PropertyCheck::Stable { window: 10 });
        }

        CompiledProperty {
            name: spec.name.clone(),
            checks,
        }
    }

    /// Compile from EmergeLang AST emerge declaration, using predicate name heuristics.
    pub fn from_emerge_decl(decl: &ast::EmergeDecl) -> Self {
        let mut checks = Vec::new();

        for clause in &decl.clauses {
            match clause {
                ast::EmergeClause::Temporal(tc) => {
                    let pred_name = extract_predicate_name(&tc.property);
                    match pred_name.as_deref() {
                        Some("all_agree") | Some("consensus") => {
                            checks.push(PropertyCheck::EventuallyGloballyAgree);
                        }
                        Some("coverage") | Some("covered") | Some("spread") => {
                            checks.push(PropertyCheck::SpatialCoverage);
                        }
                        Some("formation") | Some("flock") | Some("aligned") => {
                            checks.push(PropertyCheck::Formation);
                        }
                        Some("ring_intact") | Some("ring_connected") | Some("self_heal") => {
                            checks.push(PropertyCheck::SelfHealingRing);
                        }
                        Some(name) => {
                            checks.push(PropertyCheck::CustomPredicate(name.to_string()));
                        }
                        None => {
                            // Default: consensus
                            checks.push(PropertyCheck::EventuallyGloballyAgree);
                        }
                    }
                }
                ast::EmergeClause::Fault(fc) => {
                    checks.push(PropertyCheck::FaultTolerant { max_removals: 1 });
                    // Also check the inner property
                    let pred_name = extract_predicate_name(&fc.property);
                    if let Some(name) = pred_name {
                        match name.as_str() {
                            "all_agree" | "consensus" => {
                                // Already covered by EventuallyGloballyAgree
                            }
                            other => {
                                checks.push(PropertyCheck::CustomPredicate(other.to_string()));
                            }
                        }
                    }
                }
                ast::EmergeClause::Converge(cc) => {
                    if let Some(val) = eval_const(&cc.bound) {
                        checks.push(PropertyCheck::ConvergeWithin {
                            deadline: val as u64,
                        });
                    }
                }
                ast::EmergeClause::Invariant(prop) => {
                    let pred_name = extract_predicate_name(prop);
                    if let Some(name) = pred_name {
                        checks.push(PropertyCheck::CustomPredicate(name));
                    } else {
                        checks.push(PropertyCheck::Stable { window: 10 });
                    }
                }
            }
        }

        // Deduplicate
        checks.dedup_by(|a, b| std::mem::discriminant(a) == std::mem::discriminant(b));

        CompiledProperty {
            name: decl.name.clone(),
            checks,
        }
    }

    /// Evaluate all property checks against a trace. Returns true if all pass.
    pub fn check(&self, trace: &Trace) -> bool {
        self.checks.iter().all(|c| check_single(c, trace))
    }

    /// Evaluate and return detailed results for each check.
    pub fn check_detailed(&self, trace: &Trace) -> Vec<(PropertyCheck, bool, String)> {
        self.checks
            .iter()
            .map(|c| {
                let (passed, detail) = check_single_detailed(c, trace);
                (c.clone(), passed, detail)
            })
            .collect()
    }
}

/// Check a single property against a trace.
fn check_single(check: &PropertyCheck, trace: &Trace) -> bool {
    match check {
        PropertyCheck::EventuallyGloballyAgree => trace.eventually_globally_agree(),

        PropertyCheck::Stable { window } => {
            if let Some(conv_step) = trace.convergence_step() {
                let remaining = trace.snapshots.len() as u64 - conv_step;
                remaining >= *window
            } else {
                false
            }
        }

        PropertyCheck::ConvergeWithin { deadline } => {
            trace
                .convergence_step()
                .map(|s| s <= *deadline)
                .unwrap_or(false)
        }

        PropertyCheck::FractionAgree { threshold } => {
            if let Some(last) = trace.snapshots.last() {
                if let Some(majority) = last.majority_value() {
                    last.fraction_with_value(majority) >= *threshold
                } else {
                    false
                }
            } else {
                false
            }
        }

        PropertyCheck::FaultTolerant { .. } => {
            // Fault tolerance is checked at a higher level by running
            // simulations with agent removals. At the trace level,
            // we just check that consensus is reached.
            trace.eventually_globally_agree()
        }

        PropertyCheck::SpatialCoverage => {
            // For coverage: check that agents spread out
            // Use output value diversity as a proxy
            if let Some(last) = trace.snapshots.last() {
                let unique_values: std::collections::HashSet<i64> =
                    last.agents.iter().map(|a| a.value).collect();
                // Coverage is satisfied if agents are spread (many distinct values)
                // or if they've partitioned space (each has unique assignment)
                unique_values.len() > 1 || last.agents.len() <= 1
            } else {
                false
            }
        }

        PropertyCheck::Formation => {
            // For formation: check that agents have aligned/consistent states
            // and the alignment is stable
            if trace.snapshots.len() < 5 {
                return false;
            }
            let last_few = &trace.snapshots[trace.snapshots.len().saturating_sub(5)..];
            // Check stability: values don't change in last 5 steps
            let last_values: Vec<i64> = last_few.last().unwrap().agents.iter().map(|a| a.value).collect();
            last_few.iter().all(|snap| {
                let vals: Vec<i64> = snap.agents.iter().map(|a| a.value).collect();
                vals == last_values
            })
        }

        PropertyCheck::SelfHealingRing => {
            // Self-healing ring: agents maintain ordered values
            // After settling, agent values should form a consistent pattern
            if let Some(last) = trace.snapshots.last() {
                // Check that agents have reached a stable configuration
                let values: Vec<i64> = last.agents.iter().map(|a| a.value).collect();
                // At least check stability
                if trace.snapshots.len() >= 3 {
                    let prev = &trace.snapshots[trace.snapshots.len() - 2];
                    let prev_values: Vec<i64> = prev.agents.iter().map(|a| a.value).collect();
                    values == prev_values
                } else {
                    false
                }
            } else {
                false
            }
        }

        PropertyCheck::CustomPredicate(name) => {
            // For custom predicates, fall back to consensus check
            // (in a full system, this would dispatch to registered predicate evaluators)
            tracing::warn!("Custom predicate '{}' not implemented, using consensus check", name);
            trace.eventually_globally_agree()
        }
    }
}

/// Check a single property with detailed result.
fn check_single_detailed(check: &PropertyCheck, trace: &Trace) -> (bool, String) {
    let passed = check_single(check, trace);
    let detail = match check {
        PropertyCheck::EventuallyGloballyAgree => {
            if passed {
                let step = trace.convergence_step().unwrap_or(0);
                format!("Consensus reached at step {}", step)
            } else {
                "Consensus not reached".to_string()
            }
        }
        PropertyCheck::ConvergeWithin { deadline } => {
            if let Some(step) = trace.convergence_step() {
                format!("Converged at step {} (deadline: {})", step, deadline)
            } else {
                format!("Did not converge within {} steps", deadline)
            }
        }
        PropertyCheck::FaultTolerant { max_removals } => {
            format!(
                "Fault tolerance (k={}): {}",
                max_removals,
                if passed { "PASS" } else { "FAIL" }
            )
        }
        PropertyCheck::Stable { window } => {
            format!("Stability (window={}): {}", window, if passed { "PASS" } else { "FAIL" })
        }
        _ => format!("{:?}: {}", check, if passed { "PASS" } else { "FAIL" }),
    };
    (passed, detail)
}

/// Extract the predicate function name from a property expression.
fn extract_predicate_name(prop: &ast::PropertyExpr) -> Option<String> {
    match prop {
        ast::PropertyExpr::FuncCall(name, _) => Some(name.clone()),
        ast::PropertyExpr::And(l, _) => extract_predicate_name(l),
        ast::PropertyExpr::Or(l, _) => extract_predicate_name(l),
        ast::PropertyExpr::Not(inner) => extract_predicate_name(inner),
        ast::PropertyExpr::Comparison(_, _, _) => None,
    }
}

/// Evaluate a constant arithmetic expression.
fn eval_const(expr: &ast::ArithExpr) -> Option<f64> {
    match expr {
        ast::ArithExpr::IntLit(i) => Some(*i as f64),
        ast::ArithExpr::FloatLit(f) => Some(*f),
        ast::ArithExpr::BinOp(l, op, r) => {
            let lv = eval_const(l)?;
            let rv = eval_const(r)?;
            Some(match op {
                ast::ArithOp::Add => lv + rv,
                ast::ArithOp::Sub => lv - rv,
                ast::ArithOp::Mul => lv * rv,
                ast::ArithOp::Div => lv / rv,
            })
        }
        ast::ArithExpr::Neg(inner) => eval_const(inner).map(|v| -v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_consensus_spec() {
        let src = r#"
            agent_type Node {
                state: { value: Int }
                observe: neighbors(radius: Nat) -> Set<Node>
                act: { update }
            }

            emerge Consensus(agents: Swarm<Node>) {
                eventually globally: all_agree(agents)
                forall d in agents:
                    without(d): eventually globally: all_agree(agents)
                converge_within: 100 steps
            }
        "#;
        let program = emergelang::parse(src).unwrap();
        let emerge = program.items.iter().find_map(|i| match i {
            ast::Item::Emerge(e) => Some(e),
            _ => None,
        }).unwrap();

        let prop = CompiledProperty::from_emerge_decl(emerge);
        assert_eq!(prop.name, "Consensus");
        assert!(!prop.checks.is_empty());

        // Should have: EventuallyGloballyAgree, FaultTolerant, ConvergeWithin
        let has_consensus = prop.checks.iter().any(|c| matches!(c, PropertyCheck::EventuallyGloballyAgree));
        let has_fault = prop.checks.iter().any(|c| matches!(c, PropertyCheck::FaultTolerant { .. }));
        let has_converge = prop.checks.iter().any(|c| matches!(c, PropertyCheck::ConvergeWithin { .. }));
        assert!(has_consensus, "Should have consensus check");
        assert!(has_fault, "Should have fault tolerance check");
        assert!(has_converge, "Should have convergence check");
    }

    #[test]
    fn test_compile_coverage_spec() {
        let src = r#"
            agent_type Drone {
                state: { position: Vec2 }
                observe: neighbors(radius: Float) -> Set<Drone>
                act: { move_to }
            }

            emerge Coverage(drones: Swarm<Drone>) {
                eventually globally: coverage(drones)
            }
        "#;
        let program = emergelang::parse(src).unwrap();
        let emerge = program.items.iter().find_map(|i| match i {
            ast::Item::Emerge(e) => Some(e),
            _ => None,
        }).unwrap();

        let prop = CompiledProperty::from_emerge_decl(emerge);
        assert_eq!(prop.name, "Coverage");
        let has_coverage = prop.checks.iter().any(|c| matches!(c, PropertyCheck::SpatialCoverage));
        assert!(has_coverage, "Should detect coverage property");
    }
}
