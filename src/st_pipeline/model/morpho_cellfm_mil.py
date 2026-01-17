from __future__ import annotations

import torch
from torch import nn

from st_pipeline.constants import KEYS
from st_pipeline.model.adapter import VisualAdapter
from st_pipeline.model.nb_loss import nb_negative_log_likelihood
from st_pipeline.model.cellfm_wrapper import CellfmWrapper


class MorphoCellfmMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_genes: int,
        cellfm_dim: int,
        cellfm_layers: int,
        cellfm_heads: int,
        cellfm_checkpoint: str | None,
        freeze_cellfm: bool,
        use_mock: bool,
        use_retention: bool,
        vocab_size: int | None = None,
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
        self.adapter = VisualAdapter(input_dim=input_dim, output_dim=cellfm_dim, dropout=dropout)
        self.cellfm = CellfmWrapper(
            vocab_size=vocab_size or n_genes,
            d_model=cellfm_dim,
            n_layers=cellfm_layers,
            n_heads=cellfm_heads,
            checkpoint=cellfm_checkpoint,
            use_mock=use_mock,
            freeze=freeze_cellfm,
            use_retention=use_retention,
        )
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
        gene_ids = batch.get(KEYS.GENE_IDS)
        if gene_ids is None:
            raise ValueError("CellFM requires gene_ids in batch. Set data.gene_vocab_path.")
        if gene_ids.dim() == 1:
            gene_ids = gene_ids.unsqueeze(0).expand(X.size(0), -1)

        prompt = self.adapter(X)
        cell_emb, gene_emb = self.cellfm(prompt, gene_ids)
        pred = self.cellfm.cellwise_decode(cell_emb, gene_emb)
        mu_inst = self.softplus(pred)

        num_bags = int(ptr.max().item()) + 1
        mu_bag = self._aggregate(mu_inst, ptr, num_bags)
        mu_bag = mu_bag * size_factor.unsqueeze(1)
        return mu_bag, mu_inst

    def loss(self, batch: dict[str, torch.Tensor], mu_bag: torch.Tensor) -> torch.Tensor:
        theta = torch.exp(self.gene_dispersion)
        return nb_negative_log_likelihood(batch[KEYS.Y_BAG], mu_bag, theta)
