
import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

try:
    import lazyslide
    LAZYSLIDE_AVAILABLE = True
except ImportError:
    LAZYSLIDE_AVAILABLE = False

# Fallback feature extractor if LazySlide incomplete or custom patch logic needed
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def extract_patch_features(img, coords, patch_size=256, model_name='resnet50'):
    """
    Extracts features for patches centered at coords.
    coords: (N, 2) array of [x, y] in pixel space.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init model
    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        # Remove fc
        model = torch.nn.Sequential(*list(model.children())[:-1])
        embedding_dim = 2048
    elif model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        embedding_dim = 512
    
    model.to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    h, w, c = img.shape
    features = []
    valid_indices = []
    
    print(f"Extracting features for {len(coords)} spots...")
    
    batch_size = 32
    batch_tensors = []
    
    for i, (x, y) in enumerate(tqdm(coords)):
        # x, y are centers. Calculate top-left.
        # Assuming coords are (x_pixel, y_pixel)
        x = int(x)
        y = int(y)
        
        x1 = x - patch_size // 2
        y1 = y - patch_size // 2
        x2 = x + patch_size // 2
        y2 = y + patch_size // 2
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue
            
        patch = img[y1:y2, x1:x2] # Numpy is H, W (y, x)
        
        tensor = preprocess(patch)
        batch_tensors.append(tensor)
        valid_indices.append(i)
        
        if len(batch_tensors) >= batch_size:
            batch_stack = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                res = model(batch_stack).squeeze()
                if len(res.shape) == 1: res = res.unsqueeze(0)
                features.append(res.cpu().numpy())
            batch_tensors = []
            
    # Process remaining
    if batch_tensors:
        batch_stack = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            res = model(batch_stack).squeeze()
            if len(res.shape) == 1: res = res.unsqueeze(0)
            features.append(res.cpu().numpy())
            
    if not features:
        return None, []
        
    features = np.concatenate(features, axis=0)
    
    # Project to 512 if using ResNet50 (2048) to match our architecture preference
    # Prototyping: just keeping it as is, but Model expects 512 input.
    # User can adjust --input_dim arguement in pipelines.
    
    return features, valid_indices

def main():
    parser = argparse.ArgumentParser(description="Prepare training data from H5AD + Image")
    parser.add_argument("--h5ad", type=str, required=True, help="Path to AnnData file (.h5ad)")
    parser.add_argument("--image", type=str, required=True, help="Path to high-res H&E image")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Scaling from H5AD coords to Image pixels (if needed)")
    parser.add_argument("--top_k_genes", type=int, default=1000, help="Select top K highly variable genes")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Transcriptomics
    print(f"Loading H5AD: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    
    # Preprocessing Genes
    # sc.pp.normalize_total(adata)
    # sc.pp.log1p(adata)
    print("Selecting highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=args.top_k_genes, subset=True)
    
    # Get Coords
    # Assuming spatial coords are in obsm['spatial']
    if 'spatial' not in adata.obsm:
        raise ValueError("H5AD is missing .obsm['spatial']")
        
    coords_raw = adata.obsm['spatial'] * args.scale_factor
    
    # 2. Load Image
    print(f"Loading Image: {args.image}")
    # OpenCV loads huge images might fail if RAM limited, but standard for prototype
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError("Failed to load image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Extract Patches at Spots
    # Use ResNet18 for 512-dim features to match our default config
    features, valid_indices = extract_patch_features(img, coords_raw, model_name='resnet18')
    
    if features is None:
        print("No valid patches extracted.")
        return
        
    # 4. Filter AnnData to match valid patches
    # valid_indices is list of indices in original adata that were successfully extracted
    adata_filtered = adata[valid_indices].copy()
    features_filtered = features
    coords_filtered = coords_raw[valid_indices]
    
    # Get Gene Expression Matrix (Dense)
    if scipy.sparse.issparse(adata_filtered.X):
        gene_exp = adata_filtered.X.toarray()
    else:
        gene_exp = adata_filtered.X
        
    # 5. Save .npy pairs
    print(f"Saving processing files to {args.output_dir}...")
    print(f"Final Count: {len(features_filtered)} spots.")
    print(f"Features Shape: {features_filtered.shape}")
    print(f"Gene Exp Shape: {gene_exp.shape}")
    
    np.save(os.path.join(args.output_dir, "features.npy"), features_filtered)
    np.save(os.path.join(args.output_dir, "coords.npy"), coords_filtered)
    np.save(os.path.join(args.output_dir, "gene_exp.npy"), gene_exp)
    
    # Save gene names for reference
    pd.DataFrame({'gene_name': adata_filtered.var_names}).to_csv(os.path.join(args.output_dir, "gene_names.csv"), index=False)
    
    # Dummy perturbations (all 0)
    pert_df = pd.DataFrame({'perturbation_id': np.zeros(len(features_filtered), dtype=int)})
    pert_df.to_csv(os.path.join(args.output_dir, "perturbations.csv"), index=False)
    
    print("Done!")

import scipy.sparse # Delayed import

if __name__ == "__main__":
    main()
