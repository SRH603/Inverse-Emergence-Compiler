//! Emergence Temporal Logic (ETL) — the intermediate representation.
//!
//! EmergeLang surface syntax compiles down to ETL formulas.
//! ETL extends CTL* with:
//!   - Aggregate quantifiers (fraction, count, forall/exists over agents)
//!   - Topological modalities (nearby, connected)
//!   - Convergence operators
//!   - Perturbation quantifiers (robust under agent removal)

use serde::{Deserialize, Serialize};

/// An ETL formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EtlFormula {
    // --- Atomic ---
    /// A predicate on global state (e.g., all_agree, coverage >= 0.9)
    Predicate(Predicate),

    // --- Boolean ---
    And(Box<EtlFormula>, Box<EtlFormula>),
    Or(Box<EtlFormula>, Box<EtlFormula>),
    Not(Box<EtlFormula>),
    Implies(Box<EtlFormula>, Box<EtlFormula>),

    // --- Temporal (CTL* path quantifiers) ---
    /// On all paths, φ holds
    ForAllPaths(Box<EtlFormula>),
    /// There exists a path where φ holds
    ExistsPath(Box<EtlFormula>),

    // --- Temporal (LTL operators on a path) ---
    /// φ holds at the next step
    Next(Box<EtlFormula>),
    /// φ holds at some future step
    Eventually(Box<EtlFormula>),
    /// φ holds at all future steps
    Globally(Box<EtlFormula>),
    /// φ holds until ψ holds
    Until(Box<EtlFormula>, Box<EtlFormula>),

    // --- Emergence-specific extensions ---

    /// Aggregate quantifier: fraction of agents satisfying φ >= threshold
    FractionAtLeast {
        property: Box<EtlFormula>,
        threshold: f64,
    },

    /// Convergence: a real-valued measure converges to target within epsilon by time T
    ConvergesTo {
        measure: Measure,
        target: f64,
        epsilon: f64,
        deadline: u64,
    },

    /// Fault tolerance: φ holds even after removing any k agents
    RobustUnderRemoval {
        property: Box<EtlFormula>,
        max_removals: u32,
    },

    /// Topological: all agents within radius r of an agent satisfy φ
    Neighborhood {
        radius: f64,
        property: Box<EtlFormula>,
    },
}

/// A predicate on the global or local state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Predicate {
    /// Named predicate from the spec (e.g., "all_agree", "coverage")
    Named(String, Vec<PredicateArg>),
    /// Comparison: expr op expr
    Comparison(MeasureExpr, CompOp, MeasureExpr),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredicateArg {
    Var(String),
    Lit(f64),
}

/// A real-valued measure on the system state (for convergence specs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Measure {
    /// Variance of a field across agents
    Variance(String),
    /// Coverage fraction
    Coverage,
    /// Custom named measure
    Named(String),
}

/// An expression that evaluates to a real number.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasureExpr {
    Literal(f64),
    Measure(Measure),
    BinOp(Box<MeasureExpr>, ArithOp, Box<MeasureExpr>),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CompOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// A compiled emergence specification in ETL form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EtlSpec {
    /// The name of the emergence spec
    pub name: String,
    /// Safety properties: must hold at all times (AG φ)
    pub safety: Vec<EtlFormula>,
    /// Liveness properties: must eventually hold (AF φ, or convergence)
    pub liveness: Vec<EtlFormula>,
    /// Fault tolerance requirements
    pub fault_tolerance: Vec<EtlFormula>,
    /// Convergence bounds (optional)
    pub convergence_bound: Option<u64>,
}
