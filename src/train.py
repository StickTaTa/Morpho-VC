
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SpatialMorphoDataset
from model import MorphoScGPT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    train_dataset = SpatialMorphoDataset(args.data_dir, split='train')
    val_dataset = SpatialMorphoDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model Setup
    # Infer n_genes from data
    sample_feat, sample_gene, sample_pert = train_dataset[0]
    n_genes = sample_gene.shape[0]
    n_perts = 20 # fixed ample buffer or derived
    
    # Initialize scGPT-based model
    # Note: scgpt_model_path would be actual path if user has it
    model = MorphoScGPT(
        visual_dim=sample_feat.shape[0], 
        n_genes=n_genes, 
        n_perts=n_perts,
        freeze_scgpt=True, # Fine-tune only adapter
        use_mock=True # Default to Mock for safety unless user specifies
    ).to(device)
    
    # Optimize only trainable parameters (Adapter + PertEmbedding + Head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr)
    criterion = nn.MSELoss()
    
    logger.info(f"Training started. Trainable params: {len(trainable_params)}")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for feat, gene, pert in train_loader:
            feat, gene, pert = feat.to(device), gene.to(device), pert.to(device)
            
            optimizer.zero_grad()
            pred_gene = model(feat, pert)
            loss = criterion(pred_gene, gene)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for feat, gene, pert in val_loader:
                feat, gene, pert = feat.to(device), gene.to(device), pert.to(device)
                pred_gene = model(feat, pert)
                loss = criterion(pred_gene, gene)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model_scgpt.pth")
            logger.info("Saved best model.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4) # Lower LR for finetuning
    parser.add_argument("--scgpt_path", type=str, default=None)
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()
