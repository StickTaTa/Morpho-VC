from __future__ import annotations

import torch
from torch import nn

from st_pipeline.constants import KEYS
from st_pipeline.model.adapter import VisualAdapter
from st_pipeline.model.nb_loss import nb_negative_log_likelihood
from st_pipeline.model.scgpt_wrapper import ScgptWrapper


class MorphoScgptMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_genes: int,
        scgpt_dim: int,
        scgpt_checkpoint: str | None,
        freeze_scgpt: bool,
        use_mock: bool,
        dropout: float = 0.1,
        aggregation: str = "mean",
        dispersion: str = "gene",
    ) -> None:
        super().__init__()
        if aggregation not in {"mean", "sum"}:
            raise ValueError("aggregation must be mean or sum")
        if dispersion != "gene":
            raise ValueError("dispersion must be 'gene'")

        self.n_genes = n_genes
        self.aggregation = aggregation

        self.adapter = VisualAdapter(input_dim=input_dim, output_dim=scgpt_dim, dropout=dropout)
        self.scgpt = ScgptWrapper(
            n_genes=n_genes,
            d_model=scgpt_dim,
            checkpoint=scgpt_checkpoint,
            use_mock=use_mock,
            freeze=freeze_scgpt,
        )
        self.output_head = nn.Linear(scgpt_dim, 1)
        self.gene_dispersion = nn.Parameter(torch.zeros(n_genes))
        self.softplus = nn.Softplus()

    def _aggregate(self, inst_pred: torch.Tensor, ptr: torch.Tensor, num_bags: int) -> torch.Tensor:
        bag = torch.zeros(num_bags, inst_pred.size(1), device=inst_pred.device)
        bag.index_add_(0, ptr, inst_pred)
        if self.aggregation == "mean":
            counts = torch.bincount(ptr, minlength=num_bags).clamp_min(1).unsqueeze(1)
            bag = bag / counts
        return bag

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        X = batch[KEYS.X]
        ptr = batch[KEYS.PTR_BAG_INSTANCE]
        size_factor = batch[KEYS.SIZE_FACTOR]

        prompt = self.adapter(X).unsqueeze(1)
        seq_out = self.scgpt(prompt)
        gene_out = seq_out[:, 1:, :]
        mu_inst = self.softplus(self.output_head(gene_out)).squeeze(-1)

        num_bags = int(ptr.max().item()) + 1
        mu_bag = self._aggregate(mu_inst, ptr, num_bags)
        mu_bag = mu_bag * size_factor.unsqueeze(1)

        return mu_bag, mu_inst

    def loss(self, batch: dict[str, torch.Tensor], mu_bag: torch.Tensor) -> torch.Tensor:
        theta = torch.exp(self.gene_dispersion)
        return nb_negative_log_likelihood(batch[KEYS.Y_BAG], mu_bag, theta)
