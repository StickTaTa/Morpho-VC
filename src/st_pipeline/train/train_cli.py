from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, random_split
import yaml

from st_pipeline.data.h5ad_loader import load_h5ad
from st_pipeline.data.mil_dataset import MilSpotDataset
from st_pipeline.data.collate import mil_collate
from st_pipeline.data.gene_vocab import load_gene_vocab
from st_pipeline.model.morpho_cellfm_mil import MorphoCellfmMIL
from st_pipeline.model.morpho_scgpt_mil import MorphoScgptMIL


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("st_mil_train")


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Morpho-VC MIL model with NB loss")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mapping_json", default="", help="Optional spot->cell mapping JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    foundation = str(model_cfg.get("foundation", "scgpt")).lower()
    gene_vocab_path = data_cfg.get("gene_vocab_path") if foundation == "cellfm" else None

    h5ad = load_h5ad(
        h5ad_path=data_cfg["h5ad_path"],
        genes=data_cfg["genes"],
        spot_radius_px=float(data_cfg.get("spot_radius_px", 0)),
        gene_vocab_path=gene_vocab_path or None,
    )

    emb_path = Path(data_cfg["cell_emb_h5"])
    if not emb_path.exists():
        raise FileNotFoundError("cell_emb_h5 must exist before training")

    with h5py.File(emb_path, "r") as f:
        input_dim = int(f["embedding"].shape[1])

    mapping_json = Path(args.mapping_json) if args.mapping_json else None

    dataset = MilSpotDataset(
        adata=h5ad.adata,
        embedding_h5=emb_path,
        spot_radius_px=h5ad.spot_radius_px,
        gene_ids=h5ad.gene_ids,
        mapping_json=mapping_json,
    )

    train_size = int(len(dataset) * data_cfg["frac_train"])
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(data_cfg["seed"])
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        collate_fn=mil_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        collate_fn=mil_collate,
    )

    if foundation == "cellfm":
        if h5ad.gene_ids is None:
            raise ValueError("cellfm requires gene_vocab_path to map gene ids.")
        vocab_path = data_cfg.get("gene_vocab_path")
        if not vocab_path:
            raise ValueError("cellfm requires data.gene_vocab_path for vocab size.")
        vocab_size = len(load_gene_vocab(vocab_path))
        model = MorphoCellfmMIL(
            input_dim=input_dim,
            n_genes=len(h5ad.genes),
            cellfm_dim=int(model_cfg.get("cellfm_dim", 1536)),
            cellfm_layers=int(model_cfg.get("cellfm_layers", 2)),
            cellfm_heads=int(model_cfg.get("cellfm_heads", 48)),
            cellfm_checkpoint=model_cfg.get("cellfm_checkpoint") or None,
            freeze_cellfm=bool(model_cfg.get("freeze_cellfm", True)),
            use_mock=bool(model_cfg.get("use_mock_cellfm", False)),
            use_retention=bool(model_cfg.get("cellfm_use_retention", True)),
            vocab_size=vocab_size,
            dropout=float(model_cfg.get("dropout", 0.1)),
            aggregation=model_cfg.get("aggregation", "mean"),
            dispersion=model_cfg.get("dispersion", "gene"),
        )
    else:
        model = MorphoScgptMIL(
            input_dim=input_dim,
            n_genes=len(h5ad.genes),
            scgpt_dim=int(model_cfg["scgpt_dim"]),
            scgpt_checkpoint=model_cfg.get("scgpt_checkpoint") or None,
            freeze_scgpt=bool(model_cfg["freeze_scgpt"]),
            use_mock=bool(model_cfg["use_mock"]),
            dropout=float(model_cfg["dropout"]),
            aggregation=model_cfg.get("aggregation", "mean"),
            dispersion=model_cfg.get("dispersion", "gene"),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(train_cfg["lr"]))

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            mu_bag, _ = model(batch)
            loss = model.loss(batch, mu_bag)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                mu_bag, _ = model(batch)
                loss = model.loss(batch, mu_bag)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        logger.info("Epoch %d | train loss %.4f | val loss %.4f", epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            with (output_dir / "genes.json").open("w") as f:
                json.dump(h5ad.genes, f)
            with (output_dir / "config.json").open("w") as f:
                json.dump(cfg, f)
            logger.info("Saved checkpoint to %s", ckpt_path)


if __name__ == "__main__":
    main()
