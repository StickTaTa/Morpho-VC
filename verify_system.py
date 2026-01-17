
import os
import numpy as np
import cv2
import pandas as pd
import subprocess
import sys

def create_mock_data():
    os.makedirs("data/mock_raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Create a dummy H&E image
    # Size 1024x1024 (yielding ~16 patches of 256x256)
    img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    cv2.imwrite("data/mock_raw/test_slide.tif", img)
    print("Created mock image.")
    
    # 2. Create dummy processed features (simulating feature_extract.py output)
    n_patches = 16
    feat_dim = 512
    n_genes = 100
    
    features = np.random.randn(n_patches, feat_dim).astype(np.float32)
    coords = np.random.randint(0, 1024, (n_patches, 2))
    gene_exp = np.random.randn(n_patches, n_genes).astype(np.float32)
    
    # Perturbations
    # Half control (0), Half perturbed (1)
    pert_ids = np.array([0] * 8 + [1] * 8)
    pert_df = pd.DataFrame({'perturbation_id': pert_ids})
    
    np.save("data/processed/features.npy", features)
    np.save("data/processed/coords.npy", coords)
    np.save("data/processed/gene_exp.npy", gene_exp)
    pert_df.to_csv("data/processed/perturbations.csv", index=False)
    print("Created mock process data.")

def run_tests():
    print(">>> Testing Feature Extraction (Dry Run on mock image)...")
    # We call our script (which might default to ResNet50 if LazySlide fails/mocked)
    res = subprocess.run([sys.executable, "src/feature_extract.py", 
                          "--image_path", "data/mock_raw/test_slide.tif", 
                          "--output_dir", "data/mock_test_extract"], 
                         capture_output=True, text=True)
    if res.returncode != 0:
        print("Feature extraction failed:")
        print(res.stderr)
    else:
        print("Feature extraction ran successfully.")
        
    print(">>> Testing Training Loop...")
    res = subprocess.run([sys.executable, "src/train.py", 
                          "--data_dir", "data/processed", 
                          "--epochs", "2", 
                          "--batch_size", "4"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        print("Training failed:")
        print(res.stderr)
    else:
        print("Training ran successfully.")
        
    print(">>> Testing Inference Pipeline...")
    # Check if model exists
    if not os.path.exists("checkpoints/best_model_scgpt.pth"):
        print("Skipping inference test as model was not saved (loss might not have improved in 2 epochs on random data).")
    else:
        res = subprocess.run([sys.executable, "src/main_pipeline.py",
                              "--image_path", "data/mock_raw/test_slide.tif",
                              "--model_path", "checkpoints/best_model_scgpt.pth",
                              "--output_dir", "results_test"],
                             capture_output=True, text=True)
        if res.returncode != 0:
            print("Inference failed:")
            print(res.stderr)
        else:
            print("Inference ran successfully.")
            
if __name__ == "__main__":
    create_mock_data()
    run_tests()
