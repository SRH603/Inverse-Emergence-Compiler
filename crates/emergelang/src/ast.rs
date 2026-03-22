use serde::{Deserialize, Serialize};

/// Top-level program: a collection of declarations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Item {
    AgentType(AgentTypeDecl),
    Topology(TopologyDecl),
    Emerge(EmergeDecl),
}

/// Agent type: defines local state, observation, and actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTypeDecl {
    pub name: String,
    pub state_fields: Vec<Field>,
    pub observe: Option<ObserveFunc>,
    pub actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub ty: TypeExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserveFunc {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: TypeExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub ty: TypeExpr,
}

/// Type expressions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TypeExpr {
    Bool,
    Int,
    Nat,
    Float,
    Vec2,
    Vec3,
    Message,
    Region,
    Set(Box<TypeExpr>),
    Swarm(String),
    Named(String),
}

/// Topology declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyDecl {
    pub name: String,
    pub params: Vec<Param>,
}

/// Emergence specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergeDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub clauses: Vec<EmergeClause>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergeClause {
    Temporal(TemporalClause),
    Fault(FaultClause),
    Converge(ConvergeClause),
    Invariant(PropertyExpr),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalClause {
    pub op: TemporalOp,
    pub property: PropertyExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalOp {
    Eventually,
    Globally,
    EventuallyGlobally,
    GloballyEventually,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultClause {
    pub agent_var: String,
    pub swarm_name: String,
    pub removed_var: String,
    pub temporal_op: TemporalOp,
    pub property: PropertyExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergeClause {
    pub bound: ArithExpr,
}

/// Property expressions (boolean-valued).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyExpr {
    Comparison(Box<ArithExpr>, CompOp, Box<ArithExpr>),
    FuncCall(String, Vec<ArithExpr>),
    And(Box<PropertyExpr>, Box<PropertyExpr>),
    Or(Box<PropertyExpr>, Box<PropertyExpr>),
    Not(Box<PropertyExpr>),
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

/// Arithmetic expressions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArithExpr {
    IntLit(i64),
    FloatLit(f64),
    Var(String),          // simple or qualified (a.b.c)
    FuncCall(String, Vec<ArithExpr>),
    BinOp(Box<ArithExpr>, ArithOp, Box<ArithExpr>),
    Neg(Box<ArithExpr>),
    SetComprehension {
        expr: Box<ArithExpr>,
        var: String,
        collection: String,
    },
    Cardinality(String), // |set|
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}
