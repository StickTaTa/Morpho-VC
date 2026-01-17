from __future__ import annotations

import inspect
import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class MockScgpt(nn.Module):
    def __init__(self, d_model: int, n_genes: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.encoder = nn.Embedding(n_genes, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(inputs_embeds)


def _make_min_vocab(n_genes: int, pad_token: str) -> tuple[int, dict]:
    pad_idx = n_genes
    vocab = {pad_token: pad_idx}
    return pad_idx, vocab


def _filter_state_dict(state_dict: dict, model: nn.Module) -> dict:
    model_state = model.state_dict()
    filtered = {}
    skipped = 0
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if hasattr(value, "shape") and hasattr(model_state[key], "shape"):
            if model_state[key].shape != value.shape:
                skipped += 1
                continue
        filtered[key] = value
    if skipped:
        logger.warning("Skipped %d scgpt checkpoint keys due to shape mismatch.", skipped)
    return filtered


def _load_scgpt_model(d_model: int, n_genes: int, checkpoint: str | Path | None):
    try:
        from scgpt.model import TransformerModel
    except Exception as exc:  # pragma: no cover - depends on scgpt install
        raise ImportError("scgpt is not installed in this environment") from exc

    sig = inspect.signature(TransformerModel)
    params = sig.parameters
    kwargs = {
        "ntoken": n_genes,
        "d_model": d_model,
        "nhead": 8,
        "d_hid": d_model,
        "nlayers": 4,
        "dropout": 0.1,
    }
    if "vocab" in params:
        pad_param = params.get("pad_token")
        pad_token = "<pad>"
        if pad_param is not None and pad_param.default not in (inspect._empty, None):
            pad_token = pad_param.default
        pad_idx, vocab = _make_min_vocab(n_genes, pad_token)
        kwargs["ntoken"] = n_genes + 1
        kwargs["vocab"] = vocab
        if "pad_token" in params:
            kwargs["pad_token"] = pad_token
        if "pad_value" in params:
            pad_value = params["pad_value"].default
            if pad_value is inspect._empty:
                pad_value = 0.0
            kwargs["pad_value"] = pad_value

    model = TransformerModel(**kwargs)
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            state_dict = state.get("state_dict", state)
            filtered = _filter_state_dict(state_dict, model)
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            if missing:
                logger.warning("Missing keys when loading scgpt: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys when loading scgpt: %s", unexpected)
        else:
            raise FileNotFoundError(f"scgpt checkpoint not found: {checkpoint}")
    return model


def _supports_inputs_embeds(model: nn.Module) -> bool:
    try:
        sig = inspect.signature(model.forward)
    except (ValueError, TypeError):
        return False
    return "inputs_embeds" in sig.parameters


class ScgptWrapper(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_model: int,
        checkpoint: str | None = None,
        use_mock: bool = False,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model

        if use_mock:
            self.scgpt = MockScgpt(d_model=d_model, n_genes=n_genes)
        else:
            self.scgpt = _load_scgpt_model(d_model=d_model, n_genes=n_genes, checkpoint=checkpoint)

        if hasattr(self.scgpt, "encoder"):
            self.encoder = self.scgpt.encoder
        else:
            self.encoder = nn.Embedding(n_genes, d_model)

        if freeze:
            for p in self.scgpt.parameters():
                p.requires_grad = False

    def forward(self, prompt_token: torch.Tensor, gene_ids: torch.Tensor | None = None) -> torch.Tensor:
        if gene_ids is None:
            gene_ids = torch.arange(self.n_genes, device=prompt_token.device)
            gene_ids = gene_ids.unsqueeze(0).expand(prompt_token.size(0), -1)

        gene_emb = self.encoder(gene_ids)
        tokens = torch.cat([prompt_token, gene_emb], dim=1)

        if _supports_inputs_embeds(self.scgpt):
            out = self.scgpt(inputs_embeds=tokens)
        elif hasattr(self.scgpt, "transformer_encoder"):
            out = self.scgpt.transformer_encoder(tokens)
        elif hasattr(self.scgpt, "transformer"):
            out = self.scgpt.transformer(tokens)
        else:
            raise RuntimeError("scgpt model does not expose a transformer encoder")

        return out
