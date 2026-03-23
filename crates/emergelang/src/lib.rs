//! EmergeLang — a specification language for emergent behaviors.
//!
//! EmergeLang lets you describe *what* global behavior you want a multi-agent
//! system to exhibit, without specifying *how* individual agents should behave.
//!
//! The language compiles to Emergence Temporal Logic (ETL), an extension of CTL*
//! with aggregate quantifiers, topological modalities, and convergence operators.
//!
//! # Example
//!
//! ```text
//! agent_type Node {
//!     state: { value: Int }
//!     observe: neighbors_majority(radius: Nat) -> Int
//!     act: { update_value }
//! }
//!
//! emerge Consensus(agents: Swarm<Node>) {
//!     eventually globally: all_agree(agents)
//! }
//! ```

pub mod ast;
pub mod compile;
pub mod etl;
pub mod parser;

pub use compile::compile_to_etl;
pub use parser::parse;
