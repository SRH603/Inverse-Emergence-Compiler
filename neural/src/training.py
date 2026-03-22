"""
Training pipeline for the GNN synthesizer.

Trains a GNN to produce local rules that achieve a target emergent property,
using differentiable ETL loss functions.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from .diff_etl_loss import consensus_loss, convergence_loss, fault_tolerance_loss
from .gnn_synthesizer import AgentRuleGNN, SynthesizerConfig, make_complete_adj


def train_consensus(
    config: SynthesizerConfig | None = None,
    verbose: bool = True,
) -> AgentRuleGNN:
    """Train a GNN to produce consensus behavior."""
    if config is None:
        config = SynthesizerConfig()

    model = AgentRuleGNN(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    adj = make_complete_adj(config.num_agents)

    for epoch in range(config.num_epochs):
        optimizer.zero_grad()

        # Random initial states (binary-ish: values near 0 or 1)
        initial_states = torch.randint(0, 2, (config.num_agents, config.state_dim)).float()
        initial_states += torch.randn_like(initial_states) * 0.1

        # Simulate
        trajectory = model.simulate(initial_states, adj, config.sim_steps)

        # Compute loss
        loss_consensus = consensus_loss(trajectory)
        loss_converge = convergence_loss(trajectory) * 0.1

        loss = loss_consensus + loss_converge

        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {loss.item():.4f} "
                  f"(consensus: {loss_consensus.item():.4f}, "
                  f"convergence: {loss_converge.item():.4f})")

    return model


def train_fault_tolerant_consensus(
    config: SynthesizerConfig | None = None,
    verbose: bool = True,
) -> AgentRuleGNN:
    """Train a GNN for fault-tolerant consensus."""
    if config is None:
        config = SynthesizerConfig()

    model = AgentRuleGNN(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    adj = make_complete_adj(config.num_agents)

    for epoch in range(config.num_epochs):
        optimizer.zero_grad()

        initial_states = torch.randint(0, 2, (config.num_agents, config.state_dim)).float()
        initial_states += torch.randn_like(initial_states) * 0.1

        # Normal consensus loss
        trajectory = model.simulate(initial_states, adj, config.sim_steps)
        loss_consensus = consensus_loss(trajectory)

        # Fault tolerance loss
        loss_fault = fault_tolerance_loss(
            model, initial_states, adj, config.sim_steps,
            num_removals=1, base_loss_fn=consensus_loss,
        )

        loss = loss_consensus + loss_fault * 0.5

        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {loss.item():.4f}")

    return model


if __name__ == "__main__":
    print("Training consensus GNN...")
    config = SynthesizerConfig(
        num_epochs=500,
        num_agents=10,
        sim_steps=30,
    )
    model = train_consensus(config)
    print("\nDone! Model parameters:", sum(p.numel() for p in model.parameters()))
