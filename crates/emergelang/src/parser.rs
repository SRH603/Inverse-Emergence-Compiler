use pest::Parser;
use pest_derive::Parser;

use crate::ast::*;

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct EmergeLangParser;

/// Parse an EmergeLang source string into an AST Program.
///
/// Returns a `ParseError` if the input doesn't conform to the grammar
/// or if the AST construction fails.
pub fn parse(source: &str) -> Result<Program, ParseError> {
    let pairs = EmergeLangParser::parse(Rule::program, source)
        .map_err(|e| ParseError::Grammar(e.to_string()))?;

    let mut items = Vec::new();
    for pair in pairs {
        if pair.as_rule() == Rule::program {
            for inner in pair.into_inner() {
                if inner.as_rule() == Rule::item {
                    items.push(parse_item(inner)?);
                }
            }
        }
    }

    Ok(Program { items })
}

/// Helper to get the next inner pair or return an error.
fn next_inner<'a>(
    iter: &mut pest::iterators::Pairs<'a, Rule>,
    context: &str,
) -> Result<pest::iterators::Pair<'a, Rule>, ParseError> {
    iter.next()
        .ok_or_else(|| ParseError::Unexpected(format!("expected inner pair in {}", context)))
}

fn parse_item(pair: pest::iterators::Pair<Rule>) -> Result<Item, ParseError> {
    let inner = next_inner(&mut pair.into_inner(), "item")?;
    match inner.as_rule() {
        Rule::agent_type_decl => parse_agent_type(inner).map(Item::AgentType),
        Rule::topology_decl => parse_topology(inner).map(Item::Topology),
        Rule::emerge_decl => parse_emerge(inner).map(Item::Emerge),
        r => Err(ParseError::Unexpected(format!("item: {:?}", r))),
    }
}

fn parse_agent_type(pair: pest::iterators::Pair<Rule>) -> Result<AgentTypeDecl, ParseError> {
    let mut inner = pair.into_inner();
    let name = next_inner(&mut inner, "agent_type name")?.as_str().to_string();

    let mut state_fields = Vec::new();
    let mut observe = None;
    let mut actions = Vec::new();

    let body = next_inner(&mut inner, "agent_body")?;
    for field_pair in body.into_inner() {
        let mut field_inner = field_pair.into_inner();
        let agent_field = match field_inner.next() {
            Some(f) => f,
            None => continue,
        };
        match agent_field.as_rule() {
            Rule::state_field => {
                if let Some(field_list) = agent_field.into_inner().next() {
                    state_fields = parse_field_list(field_list)?;
                }
            }
            Rule::observe_field => {
                if let Some(obs) = agent_field.into_inner().next() {
                    observe = Some(parse_observe_func(obs)?);
                }
            }
            Rule::act_field => {
                if let Some(ident_list) = agent_field.into_inner().next() {
                    for id in ident_list.into_inner() {
                        actions.push(id.as_str().to_string());
                    }
                }
            }
            _ => {}
        }
    }

    Ok(AgentTypeDecl {
        name,
        state_fields,
        observe,
        actions,
    })
}

fn parse_observe_func(pair: pest::iterators::Pair<Rule>) -> Result<ObserveFunc, ParseError> {
    let mut inner = pair.into_inner();
    let name = next_inner(&mut inner, "observe func name")?.as_str().to_string();

    let mut params = Vec::new();
    let mut return_type = TypeExpr::Bool;

    for p in inner {
        match p.as_rule() {
            Rule::param_list => params = parse_param_list(p)?,
            Rule::type_expr => return_type = parse_type_expr(p)?,
            _ => {}
        }
    }

    Ok(ObserveFunc {
        name,
        params,
        return_type,
    })
}

fn parse_field_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Field>, ParseError> {
    let mut fields = Vec::new();
    for f in pair.into_inner() {
        if f.as_rule() == Rule::field {
            let mut inner = f.into_inner();
            let name = next_inner(&mut inner, "field name")?.as_str().to_string();
            let ty = parse_type_expr(next_inner(&mut inner, "field type")?)?;
            fields.push(Field { name, ty });
        }
    }
    Ok(fields)
}

fn parse_param_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Param>, ParseError> {
    let mut params = Vec::new();
    for p in pair.into_inner() {
        if p.as_rule() == Rule::param {
            let mut inner = p.into_inner();
            let name = next_inner(&mut inner, "param name")?.as_str().to_string();
            let ty = parse_type_expr(next_inner(&mut inner, "param type")?)?;
            params.push(Param { name, ty });
        }
    }
    Ok(params)
}

fn parse_type_expr(pair: pest::iterators::Pair<Rule>) -> Result<TypeExpr, ParseError> {
    let text = pair.as_str().trim();
    let mut inner = pair.into_inner();
    if let Some(first) = inner.next() {
        match first.as_str() {
            "Set" => {
                let inner_type = parse_type_expr(
                    next_inner(&mut inner, "Set inner type")?,
                )?;
                Ok(TypeExpr::Set(Box::new(inner_type)))
            }
            "Swarm" => {
                let agent_name = next_inner(&mut inner, "Swarm agent name")?
                    .as_str()
                    .to_string();
                Ok(TypeExpr::Swarm(agent_name))
            }
            other => parse_type_keyword(other),
        }
    } else {
        parse_type_keyword(text)
    }
}

fn parse_type_keyword(s: &str) -> Result<TypeExpr, ParseError> {
    match s {
        "Bool" => Ok(TypeExpr::Bool),
        "Int" => Ok(TypeExpr::Int),
        "Nat" => Ok(TypeExpr::Nat),
        "Float" => Ok(TypeExpr::Float),
        "Vec2" => Ok(TypeExpr::Vec2),
        "Vec3" => Ok(TypeExpr::Vec3),
        "Message" => Ok(TypeExpr::Message),
        "Region" => Ok(TypeExpr::Region),
        other => Ok(TypeExpr::Named(other.to_string())),
    }
}

fn parse_topology(pair: pest::iterators::Pair<Rule>) -> Result<TopologyDecl, ParseError> {
    let mut inner = pair.into_inner();
    let name = next_inner(&mut inner, "topology name")?.as_str().to_string();
    let params = match inner.next() {
        Some(param_list) => parse_param_list(param_list)?,
        None => Vec::new(),
    };
    Ok(TopologyDecl { name, params })
}

fn parse_emerge(pair: pest::iterators::Pair<Rule>) -> Result<EmergeDecl, ParseError> {
    let mut inner = pair.into_inner();
    let name = next_inner(&mut inner, "emerge name")?.as_str().to_string();

    let mut params = Vec::new();
    let mut clauses = Vec::new();

    for p in inner {
        match p.as_rule() {
            Rule::param_list => params = parse_param_list(p)?,
            Rule::emerge_body => {
                for clause in p.into_inner() {
                    clauses.push(parse_emerge_clause(clause)?);
                }
            }
            _ => {}
        }
    }

    Ok(EmergeDecl {
        name,
        params,
        clauses,
    })
}

fn parse_emerge_clause(pair: pest::iterators::Pair<Rule>) -> Result<EmergeClause, ParseError> {
    let inner = next_inner(&mut pair.into_inner(), "emerge_clause inner")?;
    match inner.as_rule() {
        Rule::temporal_clause => {
            let mut parts = inner.into_inner();
            let op = parse_temporal_op(next_inner(&mut parts, "temporal_op")?)?;
            let property = parse_property_expr(next_inner(&mut parts, "temporal property")?)?;
            Ok(EmergeClause::Temporal(TemporalClause { op, property }))
        }
        Rule::fault_clause => {
            let mut parts = inner.into_inner();
            let agent_var = next_inner(&mut parts, "fault agent_var")?.as_str().to_string();
            let swarm_name = next_inner(&mut parts, "fault swarm")?.as_str().to_string();
            let removed_var = next_inner(&mut parts, "fault removed")?.as_str().to_string();
            let temporal_op = parse_temporal_op(next_inner(&mut parts, "fault temporal_op")?)?;
            let property = parse_property_expr(next_inner(&mut parts, "fault property")?)?;
            Ok(EmergeClause::Fault(FaultClause {
                agent_var,
                swarm_name,
                removed_var,
                temporal_op,
                property,
            }))
        }
        Rule::converge_clause => {
            let expr_pair = next_inner(&mut inner.into_inner(), "converge expr")?;
            let bound = parse_arith_expr(expr_pair)?;
            Ok(EmergeClause::Converge(ConvergeClause { bound }))
        }
        Rule::invariant_clause => {
            let prop = next_inner(&mut inner.into_inner(), "invariant property")?;
            Ok(EmergeClause::Invariant(parse_property_expr(prop)?))
        }
        r => Err(ParseError::Unexpected(format!("emerge_clause: {:?}", r))),
    }
}

fn parse_temporal_op(pair: pest::iterators::Pair<Rule>) -> Result<TemporalOp, ParseError> {
    match pair.as_str().trim() {
        "eventually globally" => Ok(TemporalOp::EventuallyGlobally),
        "globally eventually" => Ok(TemporalOp::GloballyEventually),
        "eventually" => Ok(TemporalOp::Eventually),
        "globally" => Ok(TemporalOp::Globally),
        other => Err(ParseError::Unexpected(format!("temporal op: {}", other))),
    }
}

fn parse_property_expr(pair: pest::iterators::Pair<Rule>) -> Result<PropertyExpr, ParseError> {
    parse_property_or(pair)
}

fn parse_property_or(pair: pest::iterators::Pair<Rule>) -> Result<PropertyExpr, ParseError> {
    let inner: Vec<_> = pair.into_inner().collect();
    if inner.is_empty() {
        return Err(ParseError::Unexpected("empty property_or".into()));
    }
    let mut result = parse_property_and(inner[0].clone())?;
    for part in &inner[1..] {
        let right = parse_property_and(part.clone())?;
        result = PropertyExpr::Or(Box::new(result), Box::new(right));
    }
    Ok(result)
}

fn parse_property_and(pair: pest::iterators::Pair<Rule>) -> Result<PropertyExpr, ParseError> {
    let inner: Vec<_> = pair.into_inner().collect();
    if inner.is_empty() {
        return Err(ParseError::Unexpected("empty property_and".into()));
    }
    let mut result = parse_property_atom(inner[0].clone())?;
    for part in &inner[1..] {
        let right = parse_property_atom(part.clone())?;
        result = PropertyExpr::And(Box::new(result), Box::new(right));
    }
    Ok(result)
}

fn parse_property_atom(pair: pest::iterators::Pair<Rule>) -> Result<PropertyExpr, ParseError> {
    let inner_pairs: Vec<_> = pair.into_inner().collect();
    if inner_pairs.is_empty() {
        return Err(ParseError::Unexpected("empty property_atom".into()));
    }

    let first = &inner_pairs[0];
    match first.as_rule() {
        Rule::property_expr => parse_property_expr(first.clone()),
        Rule::comparison => {
            let mut parts = first.clone().into_inner();
            let left = parse_arith_expr(next_inner(&mut parts, "comparison left")?)?;
            let op = parse_comp_op(next_inner(&mut parts, "comparison op")?)?;
            let right = parse_arith_expr(next_inner(&mut parts, "comparison right")?)?;
            Ok(PropertyExpr::Comparison(
                Box::new(left),
                op,
                Box::new(right),
            ))
        }
        Rule::func_call => {
            let (name, args) = parse_func_call(first.clone())?;
            Ok(PropertyExpr::FuncCall(name, args))
        }
        Rule::property_atom => {
            parse_property_atom(first.clone()).map(|p| PropertyExpr::Not(Box::new(p)))
        }
        Rule::ident => {
            let name = first.as_str().to_string();
            if inner_pairs.len() > 1 {
                let mut args = Vec::new();
                for p in &inner_pairs[1..] {
                    if p.as_rule() == Rule::arg_list {
                        for arg in p.clone().into_inner() {
                            if arg.as_rule() == Rule::arith_expr {
                                args.push(parse_arith_expr(arg)?);
                            }
                        }
                    }
                }
                Ok(PropertyExpr::FuncCall(name, args))
            } else {
                Ok(PropertyExpr::FuncCall(name, Vec::new()))
            }
        }
        r => Err(ParseError::Unexpected(format!(
            "property_atom: {:?} = '{}'",
            r,
            first.as_str()
        ))),
    }
}

fn parse_comp_op(pair: pest::iterators::Pair<Rule>) -> Result<CompOp, ParseError> {
    match pair.as_str() {
        "==" => Ok(CompOp::Eq),
        "!=" => Ok(CompOp::Ne),
        "<" => Ok(CompOp::Lt),
        "<=" => Ok(CompOp::Le),
        ">" => Ok(CompOp::Gt),
        ">=" => Ok(CompOp::Ge),
        other => Err(ParseError::Unexpected(format!("comp op: {}", other))),
    }
}

fn parse_arith_expr(pair: pest::iterators::Pair<Rule>) -> Result<ArithExpr, ParseError> {
    let mut inner: Vec<_> = pair.into_inner().collect();
    if inner.is_empty() {
        return Err(ParseError::Unexpected("empty arith_expr".into()));
    }
    let mut result = parse_arith_term(inner.remove(0))?;
    let mut i = 0;
    while i < inner.len() {
        let op = match inner[i].as_str() {
            "+" => ArithOp::Add,
            "-" => ArithOp::Sub,
            other => {
                return Err(ParseError::Unexpected(format!("add_op: {}", other)));
            }
        };
        i += 1;
        if i >= inner.len() {
            return Err(ParseError::Unexpected("missing operand after add_op".into()));
        }
        let right = parse_arith_term(inner.remove(i))?;
        result = ArithExpr::BinOp(Box::new(result), op, Box::new(right));
    }
    Ok(result)
}

fn parse_arith_term(pair: pest::iterators::Pair<Rule>) -> Result<ArithExpr, ParseError> {
    let mut inner: Vec<_> = pair.into_inner().collect();
    if inner.is_empty() {
        return Err(ParseError::Unexpected("empty arith_term".into()));
    }
    let mut result = parse_arith_unary(inner.remove(0))?;
    let mut i = 0;
    while i < inner.len() {
        let op = match inner[i].as_str() {
            "*" => ArithOp::Mul,
            "/" => ArithOp::Div,
            other => {
                return Err(ParseError::Unexpected(format!("mul_op: {}", other)));
            }
        };
        i += 1;
        if i >= inner.len() {
            return Err(ParseError::Unexpected("missing operand after mul_op".into()));
        }
        let right = parse_arith_unary(inner.remove(i))?;
        result = ArithExpr::BinOp(Box::new(result), op, Box::new(right));
    }
    Ok(result)
}

fn parse_arith_unary(pair: pest::iterators::Pair<Rule>) -> Result<ArithExpr, ParseError> {
    let s = pair.as_str().trim();
    let inner: Vec<_> = pair.into_inner().collect();

    if inner.is_empty() {
        return Err(ParseError::Unexpected("empty arith_unary".into()));
    }

    if s.starts_with('-') {
        let atom = parse_arith_atom(
            inner
                .into_iter()
                .last()
                .ok_or_else(|| ParseError::Unexpected("empty negation".into()))?,
        )?;
        Ok(ArithExpr::Neg(Box::new(atom)))
    } else {
        parse_arith_atom(
            inner
                .into_iter()
                .next()
                .ok_or_else(|| ParseError::Unexpected("empty arith_unary".into()))?,
        )
    }
}

fn parse_arith_atom(pair: pest::iterators::Pair<Rule>) -> Result<ArithExpr, ParseError> {
    match pair.as_rule() {
        Rule::arith_atom => {
            let inner = next_inner(&mut pair.into_inner(), "arith_atom inner")?;
            parse_arith_atom(inner)
        }
        Rule::arith_expr => parse_arith_expr(pair),
        Rule::func_call => {
            let (name, args) = parse_func_call(pair)?;
            Ok(ArithExpr::FuncCall(name, args))
        }
        Rule::set_comprehension => {
            let mut inner = pair.into_inner();
            let first = next_inner(&mut inner, "set_comprehension")?;
            if first.as_rule() == Rule::arith_expr {
                let expr = parse_arith_expr(first)?;
                let var = next_inner(&mut inner, "comprehension var")?
                    .as_str()
                    .to_string();
                let collection = next_inner(&mut inner, "comprehension collection")?
                    .as_str()
                    .to_string();
                Ok(ArithExpr::SetComprehension {
                    expr: Box::new(expr),
                    var,
                    collection,
                })
            } else {
                Ok(ArithExpr::Cardinality(first.as_str().to_string()))
            }
        }
        Rule::float_lit => {
            let val: f64 = pair
                .as_str()
                .parse()
                .map_err(|e| ParseError::Unexpected(format!("float: {}", e)))?;
            Ok(ArithExpr::FloatLit(val))
        }
        Rule::int_lit => {
            let val: i64 = pair
                .as_str()
                .parse()
                .map_err(|e| ParseError::Unexpected(format!("int: {}", e)))?;
            Ok(ArithExpr::IntLit(val))
        }
        Rule::qualified_ident | Rule::ident => Ok(ArithExpr::Var(pair.as_str().to_string())),
        r => Err(ParseError::Unexpected(format!(
            "arith_atom: {:?} = '{}'",
            r,
            pair.as_str()
        ))),
    }
}

fn parse_func_call(
    pair: pest::iterators::Pair<Rule>,
) -> Result<(String, Vec<ArithExpr>), ParseError> {
    let mut inner = pair.into_inner();
    let name = next_inner(&mut inner, "func_call name")?.as_str().to_string();
    let mut args = Vec::new();
    if let Some(arg_list) = inner.next() {
        for arg in arg_list.into_inner() {
            if arg.as_rule() == Rule::arith_expr {
                args.push(parse_arith_expr(arg)?);
            }
        }
    }
    Ok((name, args))
}

/// Errors that can occur during parsing.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// The input didn't match the PEG grammar.
    #[error("Grammar error: {0}")]
    Grammar(String),
    /// An unexpected token or structure was encountered during AST construction.
    #[error("Unexpected: {0}")]
    Unexpected(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_agent_type() {
        let src = r#"
            agent_type Drone {
                state: { position: Vec3, velocity: Vec3, battery: Float }
                observe: neighbors_within(radius: Float) -> Set<Drone>
                act: { move_to, broadcast }
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.items.len(), 1);
        let agent = match &program.items[0] {
            Item::AgentType(a) => a,
            other => panic!("expected AgentType, got {:?}", other),
        };
        assert_eq!(agent.name, "Drone");
        assert_eq!(agent.state_fields.len(), 3);
        assert_eq!(agent.actions.len(), 2);
    }

    #[test]
    fn test_parse_topology() {
        let src = "topology KNearestGraph(k: Nat)";
        let program = parse(src).unwrap();
        assert_eq!(program.items.len(), 1);
        let topo = match &program.items[0] {
            Item::Topology(t) => t,
            other => panic!("expected Topology, got {:?}", other),
        };
        assert_eq!(topo.name, "KNearestGraph");
        assert_eq!(topo.params.len(), 1);
    }

    #[test]
    fn test_parse_emerge() {
        let src = r#"
            emerge Consensus(agents: Swarm<Node>) {
                eventually globally: all_agree(agents)
                converge_within: 100 * |agents| steps
            }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.items.len(), 1);
        let emerge = match &program.items[0] {
            Item::Emerge(e) => e,
            other => panic!("expected Emerge, got {:?}", other),
        };
        assert_eq!(emerge.name, "Consensus");
        assert_eq!(emerge.clauses.len(), 2);
    }

    #[test]
    fn test_parse_error_on_malformed_input() {
        let result = parse("agent_type {{{");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_program() {
        let program = parse("").unwrap();
        assert!(program.items.is_empty());
    }

    #[test]
    fn test_parse_multiple_items() {
        let src = r#"
            agent_type A { state: { x: Int } act: { do_thing } }
            topology Ring()
            emerge E(a: Swarm<A>) { eventually: all_agree(a) }
        "#;
        let program = parse(src).unwrap();
        assert_eq!(program.items.len(), 3);
    }
}
