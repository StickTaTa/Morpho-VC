
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
# import scanpy as sc # Optional if we need to load h5ad direct

class SpatialMorphoDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir (str): Path to processed data directory containing .npy files.
            split (str): 'train' or 'val'.
        """
        self.data_dir = data_dir
        self.split = split
        
        # Load features and coords
        self.features = np.load(os.path.join(data_dir, "features.npy"))
        self.coords = np.load(os.path.join(data_dir, "coords.npy"))
        
        # Prepare gene expression data
        # In a real scenario, this would align spot-by-spot with features.
        # Here we assume pre-aligned arrays or we load them.
        # For prototype, we'll assume `gene_exp.npy` exists and matches `features.npy` order.
        if os.path.exists(os.path.join(data_dir, "gene_exp.npy")):
            self.gene_exp = np.load(os.path.join(data_dir, "gene_exp.npy"))
        else:
            # Create dummy gene exp for verifying the pipeline if file is missing
            print("Warning: gene_exp.npy not found, creating dummy labels")
            self.gene_exp = np.random.rand(len(self.features), 100).astype(np.float32) # Assume 100 genes

        # Perturbation labels
        if os.path.exists(os.path.join(data_dir, "perturbations.csv")):
            self.pert_df = pd.read_csv(os.path.join(data_dir, "perturbations.csv"))
            self.pert_ids = self.pert_df['perturbation_id'].values
        else:
            print("Warning: perturbations.csv not found, using all Control (0)")
            self.pert_ids = np.zeros(len(self.features), dtype=int)
            
        # Simple split logic (first 80% train, last 20% val) - can be improved
        n_total = len(self.features)
        split_idx = int(n_total * 0.8)
        
        if split == 'train':
            self.indices = range(0, split_idx)
        else:
            self.indices = range(split_idx, n_total)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        feature = torch.from_numpy(self.features[real_idx]).float()
        gene_exp = torch.from_numpy(self.gene_exp[real_idx]).float()
        pert_id = torch.tensor(self.pert_ids[real_idx], dtype=torch.long)
        
        return feature, gene_exp, pert_id
