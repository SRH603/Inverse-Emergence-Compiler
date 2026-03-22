"""
Differentiable ETL loss functions.

Converts ETL (Emergence Temporal Logic) properties into differentiable
loss functions that can be used to train the GNN synthesizer via
gradient descent.

Key insight: temporal logic operators have natural smooth relaxations:
  - eventually(P) ≈ max_t sigmoid(P(t))
  - globally(P) ≈ min_t sigmoid(P(t))
  - convergence ≈ ||f(T) - target||²
"""

from __future__ import annotations

import torch


def consensus_loss(trajectory: list[torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    """
    Loss for consensus: all agents should converge to the same state.

    Uses variance across agents as a differentiable proxy:
    - Low variance = agents agree
    - High variance = agents disagree

    The loss encourages "eventually globally" consensus:
    minimize the variance at the final steps of the trajectory.
    """
    # Focus on the last 20% of the trajectory for "eventually globally"
    t_start = max(1, int(len(trajectory) * 0.8))
    final_states = trajectory[t_start:]

    # Compute variance across agents at each final step
    variances = []
    for states in final_states:
        # states: (N, state_dim)
        var = states.var(dim=0).sum()  # scalar: total variance across agents
        variances.append(var)

    # "Globally" over final steps: all should have low variance
    # Use smooth-min (log-sum-exp with negative sign)
    var_tensor = torch.stack(variances)
    loss = var_tensor.mean()  # average variance over final steps

    return loss


def convergence_loss(
    trajectory: list[torch.Tensor],
    deadline: int | None = None,
) -> torch.Tensor:
    """
    Loss for convergence speed: the system should converge quickly.

    Penalizes high variance at each step, weighted by time
    (later steps get higher penalty).
    """
    total_loss = torch.tensor(0.0)
    n_steps = len(trajectory)
    deadline = deadline or n_steps

    for t, states in enumerate(trajectory):
        var = states.var(dim=0).sum()
        # Weight increases with time: want convergence early
        weight = (t / deadline) ** 2 if t < deadline else 1.0
        total_loss = total_loss + weight * var

    return total_loss / n_steps


def fault_tolerance_loss(
    model: torch.nn.Module,
    initial_states: torch.Tensor,
    adj: torch.Tensor,
    steps: int,
    num_removals: int = 1,
    base_loss_fn=consensus_loss,
) -> torch.Tensor:
    """
    Loss for fault tolerance: consensus should hold even after removing agents.

    Averages the base loss over all single-agent removals.
    """
    n = initial_states.shape[0]
    total_loss = torch.tensor(0.0)

    for remove_idx in range(min(n, num_removals * 3)):
        # Remove agent i
        mask = torch.ones(n, dtype=torch.bool)
        mask[remove_idx] = False

        reduced_states = initial_states[mask]
        reduced_adj = adj[mask][:, mask]

        # Simulate without the removed agent
        trajectory = model.simulate(reduced_states, reduced_adj, steps)
        total_loss = total_loss + base_loss_fn(trajectory)

    return total_loss / min(n, num_removals * 3)


def coverage_loss(
    trajectory: list[torch.Tensor],
    target_area: tuple[float, float, float, float],  # (x_min, y_min, x_max, y_max)
    grid_resolution: int = 10,
) -> torch.Tensor:
    """
    Loss for spatial coverage: agents should spread to cover an area.

    Uses a differentiable grid-based coverage approximation.
    """
    x_min, y_min, x_max, y_max = target_area
    final_states = trajectory[-1]  # (N, state_dim), assume first 2 dims are position

    positions = final_states[:, :2]  # (N, 2)

    # Create grid points
    xs = torch.linspace(x_min, x_max, grid_resolution)
    ys = torch.linspace(y_min, y_max, grid_resolution)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (G, 2)

    # For each grid point, compute soft coverage by nearest agent
    # coverage(p) = max_agent sigmoid(-||p - agent_pos||² / temperature)
    dists = torch.cdist(grid_points, positions)  # (G, N)
    min_dists = dists.min(dim=1).values  # (G,)

    # Coverage: fraction of grid points within sensing radius
    sensing_radius = (x_max - x_min) / grid_resolution
    covered = torch.sigmoid(-(min_dists - sensing_radius) * 5.0)

    # Loss: want all grid points covered (minimize 1 - mean coverage)
    return 1.0 - covered.mean()
