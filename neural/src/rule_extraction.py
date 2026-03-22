"""
Rule extraction: convert trained GNN weights into interpretable local rules.

Three strategies:
1. Decision tree distillation: fit a decision tree to the GNN's I/O behavior
2. Discretization: quantize GNN weights and activations
3. Program synthesis: use GNN as oracle, synthesize a symbolic program that matches it
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ExtractedRule:
    """An extracted local rule in symbolic form."""

    num_states: int
    num_observations: int
    # Transition table: transitions[state][obs] = (next_state, action)
    transitions: dict[tuple[int, int], tuple[int, int]]
    # Output function: output[state] = value
    output: dict[int, int]

    def to_json(self) -> dict:
        return {
            "num_states": self.num_states,
            "num_observations": self.num_observations,
            "transitions": {
                f"{s},{o}": list(v) for (s, o), v in self.transitions.items()
            },
            "output": self.output,
        }


def extract_via_clustering(
    model: torch.nn.Module,
    num_states: int = 4,
    num_observations: int = 4,
    num_samples: int = 10000,
    state_dim: int = 2,
) -> ExtractedRule:
    """
    Extract rules by clustering the GNN's behavior space.

    1. Sample random (state, neighbor_states) pairs
    2. Run through GNN to get (state, observation) -> new_state mappings
    3. Cluster states and observations into discrete buckets
    4. Build FST transition table from clustered data
    """
    model.eval()

    # Sample random states
    states_sample = torch.randn(num_samples, state_dim)

    # For each sample, create a random "neighbor aggregate"
    neighbor_agg = torch.randn(num_samples, state_dim)

    # Cluster states into discrete states using k-means
    state_clusters = _kmeans(states_sample.numpy(), num_states)
    obs_clusters = _kmeans(neighbor_agg.numpy(), num_observations)

    # For each (state_cluster, obs_cluster), find the most common next_state_cluster
    transitions: dict[tuple[int, int], tuple[int, int]] = {}

    with torch.no_grad():
        for s in range(num_states):
            for o in range(num_observations):
                # Find samples in this (state, obs) cluster
                mask = (state_clusters == s) & (obs_clusters == o)
                if not mask.any():
                    # Default: stay in same state
                    transitions[(s, o)] = (s, 0)
                    continue

                # Get the GNN's output for these samples
                sample_states = states_sample[mask]
                # Create a simple complete-graph adjacency for 2 agents
                n = min(len(sample_states), 5)
                adj = torch.ones(n, n) - torch.eye(n)
                batch_states = sample_states[:n]

                new_states = model(batch_states, adj)

                # Cluster the outputs
                new_state_ids = _assign_clusters(
                    new_states.numpy(), _get_centroids(states_sample.numpy(), state_clusters, num_states)
                )

                # Most common next state
                from collections import Counter
                counts = Counter(new_state_ids.tolist())
                next_state = counts.most_common(1)[0][0]
                transitions[(s, o)] = (next_state, 0)

    output = {s: s for s in range(num_states)}

    return ExtractedRule(
        num_states=num_states,
        num_observations=num_observations,
        transitions=transitions,
        output=output,
    )


def _kmeans(data: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """Simple k-means clustering, returns cluster assignments."""
    n = len(data)
    # Random initialization
    indices = np.random.choice(n, k, replace=False)
    centroids = data[indices].copy()

    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(data[:, None] - centroids[None, :], axis=-1)
        labels = np.argmin(dists, axis=1)

        # Update
        new_centroids = np.array([
            data[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


def _get_centroids(data: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Get cluster centroids."""
    return np.array([
        data[labels == i].mean(axis=0) if (labels == i).any() else np.zeros(data.shape[1])
        for i in range(k)
    ])


def _assign_clusters(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign data points to nearest centroid."""
    dists = np.linalg.norm(data[:, None] - centroids[None, :], axis=-1)
    return np.argmin(dists, axis=1)
