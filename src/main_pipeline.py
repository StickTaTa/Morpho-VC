
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_extract import extract_features_lazyslide, extract_features_resnet50
from src.model import MorphoScGPT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Input H&E image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--perturbation_id", type=int, default=1, help="Target perturbation ID")
    parser.add_argument("--control_id", type=int, default=0, help="Control perturbation ID")
    parser.add_argument("--n_genes", type=int, default=100) 
    parser.add_argument("--input_dim", type=int, default=512)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Feature Extraction
    print("Extracting features...")
    temp_dir = os.path.join(args.output_dir, "temp_features")
    os.makedirs(temp_dir, exist_ok=True)
    
    feats, coords = extract_features_lazyslide(args.image_path, temp_dir)
    if feats is None:
        feats, coords = extract_features_resnet50(args.image_path, temp_dir)
        
    if feats is None:
        print("Error: Feature extraction failed.")
        return

    # 2. Load Model
    print("Loading MorphoScGPT model...")
    model = MorphoScGPT(visual_dim=args.input_dim, n_genes=args.n_genes, n_perts=20, use_mock=True).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 3. Predict
    feats_tensor = torch.from_numpy(feats).float().to(device)
    ctrl_idx = torch.full((len(feats),), args.control_id, dtype=torch.long).to(device)
    pert_idx = torch.full((len(feats),), args.perturbation_id, dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Predict Control
        # scGPT architecture is generative, so we just prompt it 
        pred_ctrl = model(feats_tensor, ctrl_idx).cpu().numpy()
        
        # Predict Perturbed
        pred_pert = model(feats_tensor, pert_idx).cpu().numpy()
        
    # 4. Analysis
    lfc = pred_pert - pred_ctrl
    mean_lfc = np.mean(lfc, axis=1)
    
    # 5. Visualization
    print("Generating visualization...")
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], -coords[:, 1], c=mean_lfc, cmap='coolwarm', s=10)
    plt.colorbar(label='Mean LFC (Perturbed - Control)')
    plt.title(f"scGPT Predicted Response to Perturbation {args.perturbation_id}")
    plt.axis('equal')
    plt.savefig(os.path.join(args.output_dir, "spatial_response_map_scgpt.png"))
    plt.close()
    
    np.save(os.path.join(args.output_dir, "pred_control.npy"), pred_ctrl)
    np.save(os.path.join(args.output_dir, "pred_perturbed.npy"), pred_pert)
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
