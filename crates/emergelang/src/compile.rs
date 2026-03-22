//! Compiles EmergeLang AST into ETL (Emergence Temporal Logic) specs.

use crate::ast;
use crate::etl::{self, EtlFormula, EtlSpec, Measure, Predicate, PredicateArg};

/// Compile an EmergeLang program's emerge declarations into ETL specs.
pub fn compile_to_etl(program: &ast::Program) -> Vec<EtlSpec> {
    program
        .items
        .iter()
        .filter_map(|item| match item {
            ast::Item::Emerge(e) => Some(compile_emerge(e)),
            _ => None,
        })
        .collect()
}

fn compile_emerge(decl: &ast::EmergeDecl) -> EtlSpec {
    let mut safety = Vec::new();
    let mut liveness = Vec::new();
    let mut fault_tolerance = Vec::new();
    let mut convergence_bound = None;

    for clause in &decl.clauses {
        match clause {
            ast::EmergeClause::Temporal(tc) => {
                let prop = compile_property(&tc.property);
                let formula = match tc.op {
                    ast::TemporalOp::Globally => {
                        // Safety: AG φ
                        EtlFormula::ForAllPaths(Box::new(EtlFormula::Globally(Box::new(prop))))
                    }
                    ast::TemporalOp::Eventually => {
                        // Liveness: AF φ
                        EtlFormula::ForAllPaths(Box::new(EtlFormula::Eventually(Box::new(prop))))
                    }
                    ast::TemporalOp::EventuallyGlobally => {
                        // AF AG φ — convergence to stable state
                        EtlFormula::ForAllPaths(Box::new(EtlFormula::Eventually(Box::new(
                            EtlFormula::Globally(Box::new(prop)),
                        ))))
                    }
                    ast::TemporalOp::GloballyEventually => {
                        // AG AF φ — recurrence
                        EtlFormula::ForAllPaths(Box::new(EtlFormula::Globally(Box::new(
                            EtlFormula::Eventually(Box::new(prop)),
                        ))))
                    }
                };

                match tc.op {
                    ast::TemporalOp::Globally => safety.push(formula),
                    _ => liveness.push(formula),
                }
            }

            ast::EmergeClause::Fault(fc) => {
                let prop = compile_property(&fc.property);
                let temporal = match fc.temporal_op {
                    ast::TemporalOp::EventuallyGlobally => {
                        EtlFormula::ForAllPaths(Box::new(EtlFormula::Eventually(Box::new(
                            EtlFormula::Globally(Box::new(prop)),
                        ))))
                    }
                    ast::TemporalOp::Globally => {
                        EtlFormula::ForAllPaths(Box::new(EtlFormula::Globally(Box::new(prop))))
                    }
                    ast::TemporalOp::Eventually => {
                        EtlFormula::ForAllPaths(Box::new(EtlFormula::Eventually(Box::new(prop))))
                    }
                    _ => prop,
                };

                fault_tolerance.push(EtlFormula::RobustUnderRemoval {
                    property: Box::new(temporal),
                    max_removals: 1, // "forall d ... without(d)" means k=1
                });
            }

            ast::EmergeClause::Converge(cc) => {
                if let Some(bound) = eval_const_arith(&cc.bound) {
                    convergence_bound = Some(bound as u64);
                }
            }

            ast::EmergeClause::Invariant(prop) => {
                let formula = compile_property(prop);
                safety.push(EtlFormula::ForAllPaths(Box::new(EtlFormula::Globally(
                    Box::new(formula),
                ))));
            }
        }
    }

    EtlSpec {
        name: decl.name.clone(),
        safety,
        liveness,
        fault_tolerance,
        convergence_bound,
    }
}

fn compile_property(prop: &ast::PropertyExpr) -> EtlFormula {
    match prop {
        ast::PropertyExpr::Comparison(left, op, right) => {
            let l = compile_measure_expr(left);
            let r = compile_measure_expr(right);
            let etl_op = match op {
                ast::CompOp::Eq => etl::CompOp::Eq,
                ast::CompOp::Ne => etl::CompOp::Ne,
                ast::CompOp::Lt => etl::CompOp::Lt,
                ast::CompOp::Le => etl::CompOp::Le,
                ast::CompOp::Gt => etl::CompOp::Gt,
                ast::CompOp::Ge => etl::CompOp::Ge,
            };
            EtlFormula::Predicate(Predicate::Comparison(l, etl_op, r))
        }
        ast::PropertyExpr::FuncCall(name, args) => {
            let pred_args: Vec<PredicateArg> = args
                .iter()
                .map(|a| match a {
                    ast::ArithExpr::Var(v) => PredicateArg::Var(v.clone()),
                    ast::ArithExpr::FloatLit(f) => PredicateArg::Lit(*f),
                    ast::ArithExpr::IntLit(i) => PredicateArg::Lit(*i as f64),
                    _ => PredicateArg::Var(format!("{:?}", a)),
                })
                .collect();
            EtlFormula::Predicate(Predicate::Named(name.clone(), pred_args))
        }
        ast::PropertyExpr::And(l, r) => {
            EtlFormula::And(Box::new(compile_property(l)), Box::new(compile_property(r)))
        }
        ast::PropertyExpr::Or(l, r) => {
            EtlFormula::Or(Box::new(compile_property(l)), Box::new(compile_property(r)))
        }
        ast::PropertyExpr::Not(inner) => EtlFormula::Not(Box::new(compile_property(inner))),
    }
}

fn compile_measure_expr(expr: &ast::ArithExpr) -> etl::MeasureExpr {
    match expr {
        ast::ArithExpr::FloatLit(f) => etl::MeasureExpr::Literal(*f),
        ast::ArithExpr::IntLit(i) => etl::MeasureExpr::Literal(*i as f64),
        ast::ArithExpr::Var(name) => {
            etl::MeasureExpr::Measure(Measure::Named(name.clone()))
        }
        ast::ArithExpr::FuncCall(name, _) => {
            etl::MeasureExpr::Measure(Measure::Named(name.clone()))
        }
        ast::ArithExpr::BinOp(l, op, r) => {
            let left = compile_measure_expr(l);
            let right = compile_measure_expr(r);
            let etl_op = match op {
                ast::ArithOp::Add => etl::ArithOp::Add,
                ast::ArithOp::Sub => etl::ArithOp::Sub,
                ast::ArithOp::Mul => etl::ArithOp::Mul,
                ast::ArithOp::Div => etl::ArithOp::Div,
            };
            etl::MeasureExpr::BinOp(Box::new(left), etl_op, Box::new(right))
        }
        ast::ArithExpr::Neg(inner) => {
            let compiled = compile_measure_expr(inner);
            etl::MeasureExpr::BinOp(
                Box::new(etl::MeasureExpr::Literal(-1.0)),
                etl::ArithOp::Mul,
                Box::new(compiled),
            )
        }
        ast::ArithExpr::Cardinality(name) => {
            etl::MeasureExpr::Measure(Measure::Named(format!("|{}|", name)))
        }
        ast::ArithExpr::SetComprehension { .. } => {
            etl::MeasureExpr::Measure(Measure::Named("set_comprehension".into()))
        }
    }
}

/// Try to evaluate a constant arithmetic expression.
fn eval_const_arith(expr: &ast::ArithExpr) -> Option<f64> {
    match expr {
        ast::ArithExpr::IntLit(i) => Some(*i as f64),
        ast::ArithExpr::FloatLit(f) => Some(*f),
        ast::ArithExpr::BinOp(l, op, r) => {
            let lv = eval_const_arith(l)?;
            let rv = eval_const_arith(r)?;
            Some(match op {
                ast::ArithOp::Add => lv + rv,
                ast::ArithOp::Sub => lv - rv,
                ast::ArithOp::Mul => lv * rv,
                ast::ArithOp::Div => lv / rv,
            })
        }
        ast::ArithExpr::Neg(inner) => eval_const_arith(inner).map(|v| -v),
        _ => None, // non-constant
    }
}
