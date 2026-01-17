from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _safe_corr(func, x, y):
    try:
        return func(x, y)[0]
    except Exception:
        return float("nan")


def compute_gene_metrics(y_true: np.ndarray, y_pred: np.ndarray, gene_names: list[str]):
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    if y_true.shape[1] != len(gene_names):
        raise ValueError("gene_names length must match number of genes")

    metrics = {}
    pcc = []
    scc = []
    for i, gene in enumerate(gene_names):
        p = _safe_corr(pearsonr, y_true[:, i], y_pred[:, i])
        s = _safe_corr(spearmanr, y_true[:, i], y_pred[:, i])
        metrics[f"pcc/{gene}"] = p
        metrics[f"scc/{gene}"] = s
        pcc.append(p)
        scc.append(s)

    metrics["pcc_mean"] = float(np.nanmean(pcc))
    metrics["scc_mean"] = float(np.nanmean(scc))
    return metrics
