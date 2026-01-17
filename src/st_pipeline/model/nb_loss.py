from __future__ import annotations

import torch


def nb_negative_log_likelihood(
    target: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if target.shape != mu.shape:
        raise ValueError("target and mu must have same shape")
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    if theta.shape[-1] != mu.shape[-1]:
        raise ValueError("theta must have shape (n_genes,) or match mu")

    log_theta_mu = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu)
        + target * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(target + theta + eps)
        - torch.lgamma(theta + eps)
        - torch.lgamma(target + 1.0)
    )
    return -res.mean()
