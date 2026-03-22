"""
GNN-based rule synthesizer.

Uses a Graph Neural Network to learn local agent rules that produce
desired emergent behavior. The GNN acts as the agent's update function:
    new_state = GNN(local_state, messages_from_neighbors)

All agents share the same GNN weights (parameter sharing = uniform rules).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SynthesizerConfig:
    """Configuration for the GNN synthesizer."""

    state_dim: int = 2  # dimension of each agent's state
    hidden_dim: int = 32  # GNN hidden layer size
    message_dim: int = 16  # message passing dimension
    num_layers: int = 3  # number of message passing rounds
    learning_rate: float = 1e-3
    num_epochs: int = 1000
    num_agents: int = 20  # agents per training simulation
    sim_steps: int = 50  # simulation steps per training episode


class AgentRuleGNN(nn.Module):
    """
    A GNN that defines local agent rules via message passing.

    Each agent has a state vector. At each step:
    1. Each agent sends a message to its neighbors (message function)
    2. Each agent aggregates received messages (aggregation)
    3. Each agent updates its state based on current state + aggregated messages (update function)
    """

    def __init__(self, config: SynthesizerConfig):
        super().__init__()
        self.config = config

        # Message function: (sender_state, receiver_state) -> message
        self.message_fn = nn.Sequential(
            nn.Linear(config.state_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.message_dim),
        )

        # Update function: (current_state, aggregated_messages) -> new_state
        self.update_fn = nn.Sequential(
            nn.Linear(config.state_dim + config.message_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.state_dim),
        )

    def forward(
        self,
        states: torch.Tensor,  # (N, state_dim)
        adj: torch.Tensor,  # (N, N) adjacency matrix
    ) -> torch.Tensor:
        """One step of the local rule: compute new states for all agents."""
        n = states.shape[0]

        # Compute messages for all pairs
        # Expand states for pairwise computation
        sender = states.unsqueeze(0).expand(n, -1, -1)  # (N, N, state_dim)
        receiver = states.unsqueeze(1).expand(-1, n, -1)  # (N, N, state_dim)
        pairs = torch.cat([sender, receiver], dim=-1)  # (N, N, 2*state_dim)

        messages = self.message_fn(pairs)  # (N, N, message_dim)

        # Mask messages by adjacency and aggregate (mean)
        mask = adj.unsqueeze(-1)  # (N, N, 1)
        masked_messages = messages * mask
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)  # (N, 1)
        aggregated = masked_messages.sum(dim=1) / degree  # (N, message_dim)

        # Update states
        update_input = torch.cat([states, aggregated], dim=-1)  # (N, state_dim + message_dim)
        new_states = self.update_fn(update_input)  # (N, state_dim)

        return new_states

    def simulate(
        self,
        initial_states: torch.Tensor,
        adj: torch.Tensor,
        steps: int,
    ) -> list[torch.Tensor]:
        """Run the local rule for multiple steps, returning trajectory."""
        trajectory = [initial_states]
        states = initial_states
        for _ in range(steps):
            states = self.forward(states, adj)
            trajectory.append(states)
        return trajectory


def make_complete_adj(n: int) -> torch.Tensor:
    """Create adjacency matrix for complete graph (no self-loops)."""
    return torch.ones(n, n) - torch.eye(n)


def make_ring_adj(n: int) -> torch.Tensor:
    """Create adjacency matrix for ring topology."""
    adj = torch.zeros(n, n)
    for i in range(n):
        adj[i, (i + 1) % n] = 1.0
        adj[i, (i - 1) % n] = 1.0
    return adj
