from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from st_pipeline.data.h5ad_loader import load_h5ad
from st_pipeline.data.mil_dataset import MilSpotDataset
from st_pipeline.data.collate import mil_collate
from st_pipeline.data.gene_vocab import load_gene_vocab
from st_pipeline.model.morpho_cellfm_mil import MorphoCellfmMIL
from st_pipeline.model.morpho_scgpt_mil import MorphoScgptMIL


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("st_mil_predict")


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Predict with Morpho-VC MIL model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mapping_json", default="")
    parser.add_argument("--save_instance", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    infer_cfg = cfg["infer"]

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
        raise FileNotFoundError("cell_emb_h5 must exist before prediction")

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

    loader = DataLoader(
        dataset,
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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    preds_bag = []
    preds_inst = []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            mu_bag, mu_inst = model(batch)
            preds_bag.append(mu_bag.cpu().numpy())
            if args.save_instance:
                preds_inst.append(mu_inst.cpu().numpy())

    output_dir = Path(infer_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_bag = np.concatenate(preds_bag, axis=0)
    np.save(output_dir / "pred_bag.npy", pred_bag)
    np.save(output_dir / "spot_ids.npy", np.array(dataset.spot_ids))

    if args.save_instance:
        pred_inst = np.concatenate(preds_inst, axis=0)
        np.save(output_dir / "pred_inst.npy", pred_inst)
        with h5py.File(emb_path, "r") as f:
            np.save(output_dir / "cell_ids.npy", f["barcode"][:])

    logger.info("Saved predictions to %s", output_dir)


if __name__ == "__main__":
    main()
