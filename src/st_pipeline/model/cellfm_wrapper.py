from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from st_pipeline.model.cellfm_model import CellFMBackbone, CellwiseDecoder


logger = logging.getLogger(__name__)


class MockCellfm(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.gene_emb = nn.Embedding(vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.cellwise_dec = CellwiseDecoder(d_model, d_model, zero=False)

    def forward_prompt(
        self,
        prompt_token: torch.Tensor,
        gene_ids: torch.Tensor,
        use_retention: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_token.dim() == 2:
            prompt_token = prompt_token.unsqueeze(1)
        gene_emb = self.gene_emb(gene_ids)
        tokens = torch.cat([prompt_token + self.cls_token, gene_emb], dim=1)
        tokens = self.encoder(tokens)
        cls_out = tokens[:, 0]
        gene_out = tokens[:, 1:]
        return cls_out, gene_out


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            cleaned[key[len(prefix) :]] = value
        else:
            cleaned[key] = value
    return cleaned


def _filter_state_dict(state_dict: dict, model: nn.Module) -> dict:
    model_state = model.state_dict()
    filtered = {}
    skipped = 0
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if hasattr(value, "shape") and model_state[key].shape != value.shape:
            skipped += 1
            continue
        filtered[key] = value
    if skipped:
        logger.warning("Skipped %d CellFM keys due to shape mismatch.", skipped)
    return filtered


class CellfmWrapper(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        checkpoint: str | None = None,
        use_mock: bool = False,
        freeze: bool = True,
        use_retention: bool = True,
    ) -> None:
        super().__init__()
        self.use_retention = use_retention
        if use_mock:
            self.model = MockCellfm(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        else:
            self.model = CellFMBackbone(
                n_genes=vocab_size,
                emb_dims=d_model,
                num_layers=n_layers,
                num_heads=n_heads,
            )
            if checkpoint:
                ckpt_path = Path(checkpoint)
                if ckpt_path.suffix.lower() in {".pt", ".pth"}:
                    state = torch.load(ckpt_path, map_location="cpu")
                    state_dict = state.get("state_dict", state)
                    for prefix in ("module.", "model."):
                        state_dict = _strip_prefix(state_dict, prefix)
                    filtered = _filter_state_dict(state_dict, self.model)
                    missing, unexpected = self.model.load_state_dict(filtered, strict=False)
                    if missing:
                        logger.warning("Missing CellFM keys: %s", missing)
                    if unexpected:
                        logger.warning("Unexpected CellFM keys: %s", unexpected)
                else:
                    raise ValueError("CellFM checkpoint must be a .pt/.pth file. Convert .ckpt first.")

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, prompt_token: torch.Tensor, gene_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_prompt(prompt_token, gene_ids, use_retention=self.use_retention)

    def cellwise_decode(self, cell_emb: torch.Tensor, gene_emb: torch.Tensor) -> torch.Tensor:
        return self.model.cellwise_dec(cell_emb, gene_emb)
