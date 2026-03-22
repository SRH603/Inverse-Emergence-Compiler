//! Topology definitions for agent communication graphs.

use petgraph::graph::UnGraph;
use serde::{Deserialize, Serialize};

/// A topology defines how agents are connected (who can observe whom).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Topology {
    /// Complete graph: every agent can observe every other agent.
    Complete,
    /// Ring: each agent can observe its two neighbors.
    Ring,
    /// Grid: 2D grid, each agent observes 4 neighbors (von Neumann).
    Grid { rows: usize, cols: usize },
    /// K-nearest: each agent observes its k nearest neighbors (by index).
    KNearest { k: usize },
    /// Star: one central agent connected to all others.
    Star,
    /// Random geometric: agents in 2D space, connected if within radius r.
    RandomGeometric { radius: f64 },
}

impl Topology {
    /// Build the adjacency graph for n agents.
    pub fn build_graph(&self, n: usize) -> UnGraph<usize, ()> {
        let mut graph = UnGraph::new_undirected();
        let nodes: Vec<_> = (0..n).map(|i| graph.add_node(i)).collect();

        match self {
            Topology::Complete => {
                for i in 0..n {
                    for j in (i + 1)..n {
                        graph.add_edge(nodes[i], nodes[j], ());
                    }
                }
            }
            Topology::Ring => {
                for i in 0..n {
                    let j = (i + 1) % n;
                    graph.add_edge(nodes[i], nodes[j], ());
                }
            }
            Topology::Grid { rows, cols } => {
                let idx = |r: usize, c: usize| r * cols + c;
                for r in 0..*rows {
                    for c in 0..*cols {
                        let i = idx(r, c);
                        if i >= n {
                            continue;
                        }
                        if c + 1 < *cols && idx(r, c + 1) < n {
                            graph.add_edge(nodes[i], nodes[idx(r, c + 1)], ());
                        }
                        if r + 1 < *rows && idx(r + 1, c) < n {
                            graph.add_edge(nodes[i], nodes[idx(r + 1, c)], ());
                        }
                    }
                }
            }
            Topology::KNearest { k } => {
                for i in 0..n {
                    for d in 1..=(*k / 2).max(1) {
                        let j = (i + d) % n;
                        graph.add_edge(nodes[i], nodes[j], ());
                    }
                }
            }
            Topology::Star => {
                for i in 1..n {
                    graph.add_edge(nodes[0], nodes[i], ());
                }
            }
            Topology::RandomGeometric { .. } => {
                // For now, fall back to complete graph
                // TODO: use actual 2D positions with RNG
                for i in 0..n {
                    for j in (i + 1)..n {
                        graph.add_edge(nodes[i], nodes[j], ());
                    }
                }
            }
        }

        graph
    }

    /// Get the neighbors of agent i in a graph of n agents.
    pub fn neighbors(&self, i: usize, n: usize) -> Vec<usize> {
        match self {
            Topology::Complete => (0..n).filter(|&j| j != i).collect(),
            Topology::Ring => {
                let mut result = Vec::new();
                if n > 1 {
                    result.push((i + n - 1) % n);
                    result.push((i + 1) % n);
                }
                result
            }
            Topology::Star => {
                if i == 0 {
                    (1..n).collect()
                } else {
                    vec![0]
                }
            }
            _ => {
                // Build graph and query
                let graph = self.build_graph(n);
                let node = petgraph::graph::NodeIndex::new(i);
                graph.neighbors(node).map(|n| n.index()).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_neighbors() {
        let topo = Topology::Complete;
        assert_eq!(topo.neighbors(0, 4), vec![1, 2, 3]);
        assert_eq!(topo.neighbors(2, 4), vec![0, 1, 3]);
    }

    #[test]
    fn test_ring_neighbors() {
        let topo = Topology::Ring;
        assert_eq!(topo.neighbors(0, 5), vec![4, 1]);
        assert_eq!(topo.neighbors(2, 5), vec![1, 3]);
        assert_eq!(topo.neighbors(4, 5), vec![3, 0]);
    }

    #[test]
    fn test_star_neighbors() {
        let topo = Topology::Star;
        assert_eq!(topo.neighbors(0, 4), vec![1, 2, 3]);
        assert_eq!(topo.neighbors(1, 4), vec![0]);
    }

    #[test]
    fn test_grid_graph() {
        let topo = Topology::Grid { rows: 2, cols: 3 };
        let graph = topo.build_graph(6);
        assert_eq!(graph.node_count(), 6);
        // In a 2x3 grid: 7 edges (3 horizontal + 2 vertical per row... actually 2+2+3=7)
        assert_eq!(graph.edge_count(), 7);
    }
}
