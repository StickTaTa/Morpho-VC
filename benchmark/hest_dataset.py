import os
import sys
from pathlib import Path
import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset
import scanpy as sc
import scanpy as sc
from PIL import Image
from scipy.spatial import distance

# 尝试修复 Windows 下 OpenSlide 找不到 DLL 的问题
if os.name == 'nt':
    # 尝试寻找当前 Conda 环境下的 Library/bin
    conda_prefix = os.environ.get('CONDA_PREFIX')
    curr_python = sys.executable
    # 简单的推断策略
    possible_paths = []
    if conda_prefix:
        possible_paths.append(Path(conda_prefix) / 'Library' / 'bin')
    if curr_python:
        possible_paths.append(Path(curr_python).parent / 'Library' / 'bin')
        possible_paths.append(Path(curr_python).parent / '..' / 'Library' / 'bin')
    
    # 显式添加找到的路径
    for p in possible_paths:
        # Check for either v1 or v0 DLLs
        if p.exists() and ((p / 'libopenslide-1.dll').exists() or (p / 'libopenslide-0.dll').exists()):
            try:
                os.add_dll_directory(str(p))
            except AttributeError:
                pass # Python < 3.8
            os.environ['PATH'] = str(p) + os.pathsep + os.environ['PATH']
            break

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

class HESTHisToGeneDataset(Dataset):
    def __init__(self, slide_ids, hest_dir, spatial_dir, genes, patch_size=112, sr=False, n_pos=64, train=True):
        super().__init__()
        self.slide_ids = slide_ids
        self.hest_dir = Path(hest_dir)
        self.spatial_dir = Path(spatial_dir)
        self.genes = list(genes)
        self.patch_size = patch_size
        self.r = patch_size // 2
        self.sr = sr
        self.n_pos = n_pos
        self.train = train

        self.data_list = []  # Store (slide_id, index_in_adata) tuples
        self.adatas = {}
        # Lazy loading: store paths instead of open objects to avoid pickling issues
        self.wsi_paths = {} 
        self.wsis = {} # Cache for opened WSIs (process-local)

        print(f"Loading {len(slide_ids)} slides...")
        for sid in slide_ids:
            # Load h5ad
            h5ad_path = self.spatial_dir / f'{sid}.h5ad'
            adata = sc.read_h5ad(h5ad_path)
            
            # Filter genes
            if self.genes:
                adata = adata[:, self.genes].copy()
            
            # Preprocess
            if 'counts' in adata.layers:
                adata.X = adata.layers['counts'].copy()
            elif scipy.sparse.issparse(adata.X):
                adata.X = adata.X.toarray()
                
            # Standard ST preprocessing
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            # Identify WSI path
            wsi_path = list((self.hest_dir / 'wsis').glob(f"{sid}.*"))[0]
            self.wsi_paths[sid] = str(wsi_path)

            self.adatas[sid] = adata
            
            # Compute grid coordinates for this slide
            coords = adata.obsm['spatial'] # (N, 2) [x, y] in pixels
            min_x, min_y = coords.min(axis=0)
            max_x, max_y = coords.max(axis=0)
            
            span_x = max_x - min_x + 1e-6
            span_y = max_y - min_y + 1e-6
            
            grid_x = np.floor((coords[:, 0] - min_x) / span_x * (self.n_pos - 1)).astype(int)
            grid_y = np.floor((coords[:, 1] - min_y) / span_y * (self.n_pos - 1)).astype(int)
            
            self.adatas[sid].obsm['grid_coords'] = np.stack([grid_x, grid_y], axis=1)

            for i in range(adata.shape[0]):
                self.data_list.append((sid, i))
                
        print(f"Total spots: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sid, i = self.data_list[idx]
        adata = self.adatas[sid]
        
        # Lazy open WSI if not present in this process/thread
        if sid not in self.wsis:
            wsi_path = self.wsi_paths[sid]
            try:
                if HAS_OPENSLIDE:
                    self.wsis[sid] = openslide.OpenSlide(wsi_path)
                else:
                    self.wsis[sid] = Image.open(wsi_path)
            except Exception as e:
                print(f"Error opening WSI for {sid}: {e}, using PIL fallback")
                self.wsis[sid] = Image.open(wsi_path)
        
        wsi = self.wsis[sid]
        
        center = adata.obsm['spatial'][i].astype(int)
        grid_pos = adata.obsm['grid_coords'][i]
        exp = adata.X[i]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        
        cx, cy = center
        # Crop patch
        x_tl = cx - self.r
        y_tl = cy - self.r
        
        try:
            if HAS_OPENSLIDE and isinstance(wsi, openslide.OpenSlide):
                patch = wsi.read_region((int(x_tl), int(y_tl)), 0, (self.patch_size, self.patch_size)).convert('RGB')
            else:
                # PIL Image
                patch = wsi.crop((x_tl, y_tl, x_tl + self.patch_size, y_tl + self.patch_size)).convert('RGB')
        except Exception as e:
            # Fill with zeros if out of bounds or error
            # print(f"Error reading patch for {sid} at {center}: {e}")
            patch = Image.new('RGB', (self.patch_size, self.patch_size))

        # Transform to tensor and normalize
        patch_tensor = torch.from_numpy(np.array(patch)).float() / 255.0
        if patch_tensor.shape[0] != self.patch_size or patch_tensor.shape[1] != self.patch_size:
             patch_tensor = torch.zeros((self.patch_size, self.patch_size, 3))
        
        patch_tensor = patch_tensor.permute(2, 0, 1) # [C, H, W]
        patch_flat = patch_tensor.flatten()
        # Transform to [1, ...] sequence format for HisToGene (ViT expects inputs as sequences)
        patch_flat = patch_flat.unsqueeze(0) # [1, 3*H*W]
        grid_pos = torch.LongTensor(grid_pos).unsqueeze(0) # [1, 2]
        exp = torch.Tensor(exp).unsqueeze(0) # [1, n_genes]

        return patch_flat, grid_pos, exp


def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='NA'):
    spatialMatrix = coord
    nodes = spatialMatrix.shape[0]
    Adj = torch.zeros((nodes, nodes))
    for i in np.arange(spatialMatrix.shape[0]):
        tmp = spatialMatrix[i, :].reshape(1, -1)
        distMat = distance.cdist(tmp, spatialMatrix, distanceType)
        if k == 0:
            k = spatialMatrix.shape[0] - 1
        res = distMat.argsort()[:k + 1]
        tmpdist = distMat[0, res[0][1:k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k + 1):
            if pruneTag == 'NA':
                Adj[i][res[0][j]] = 1.0
            elif pruneTag == 'STD':
                if distMat[0, res[0][j]] <= boundary:
                    Adj[i][res[0][j]] = 1.0
            elif pruneTag == 'Grid':
                if distMat[0, res[0][j]] <= 2.0:
                    Adj[i][res[0][j]] = 1.0
    return Adj


def _safe_corr(a, b):
    a = a.astype(float)
    b = b.astype(float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return -1e9
    return float((a * b).sum() / denom)


def _align_array_to_spatial(array_col, array_row, spatial):
    # Choose swap/flip that best aligns array indices with spatial x/y axes.
    x = spatial[:, 0]
    y = spatial[:, 1]
    candidates = []
    for swap in (False, True):
        ax = array_row if swap else array_col
        ay = array_col if swap else array_row
        for flip_x in (1, -1):
            for flip_y in (1, -1):
                gx = ax * flip_x
                gy = ay * flip_y
                score = _safe_corr(gx, x) + _safe_corr(gy, y)
                candidates.append((score, gx, gy))
    best = max(candidates, key=lambda item: item[0])
    return best[1], best[2]


def _get_grid_coords(adata, n_pos):
    # Prefer array coords and shift to zero-based grid for each slide.
    if "array_col" in adata.obs and "array_row" in adata.obs:
        array_col = adata.obs["array_col"].to_numpy().astype(int)
        array_row = adata.obs["array_row"].to_numpy().astype(int)
        if "spatial" in adata.obsm:
            gx, gy = _align_array_to_spatial(array_col, array_row, adata.obsm["spatial"])
            grid = np.stack([gx, gy], axis=1)
        else:
            grid = np.stack([array_col, array_row], axis=1)
        grid = grid - grid.min(axis=0, keepdims=True)
        max_pos = grid.max(axis=0)
        if n_pos is not None and (max_pos[0] >= n_pos or max_pos[1] >= n_pos):
            raise ValueError(
                f"n_pos={n_pos} is too small for grid coords "
                f"(max col={max_pos[0]}, max row={max_pos[1]})."
            )
        return grid

    # Fallback to min-max normalized spatial coords when array coords are missing.
    if n_pos is None:
        raise ValueError("n_pos must be provided when array coords are unavailable.")
    coords = adata.obsm["spatial"]
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    span_x = max_x - min_x + 1e-6
    span_y = max_y - min_y + 1e-6
    grid_x = np.floor((coords[:, 0] - min_x) / span_x * (n_pos - 1)).astype(int)
    grid_y = np.floor((coords[:, 1] - min_y) / span_y * (n_pos - 1)).astype(int)
    return np.stack([grid_x, grid_y], axis=1)



class HESTTHItoGeneDataset(Dataset):
    def __init__(self, slide_ids, hest_dir, spatial_dir, genes, patch_size=112, n_pos=64, train=True):
        super().__init__()
        self.slide_ids = slide_ids
        self.hest_dir = Path(hest_dir)
        self.spatial_dir = Path(spatial_dir)
        self.genes = list(genes)
        self.patch_size = patch_size
        self.r = patch_size // 2
        self.n_pos = n_pos
        self.train = train

        self.adatas = {}
        self.wsi_paths = {}
        self.wsis = {}

        print(f"Loading {len(slide_ids)} slides for THItoGene...")
        for sid in slide_ids:
            h5ad_path = self.spatial_dir / f'{sid}.h5ad'
            adata = sc.read_h5ad(h5ad_path)
            
            if self.genes:
                adata = adata[:, self.genes].copy()
            
            if 'counts' in adata.layers:
                adata.X = adata.layers['counts'].copy()
            elif scipy.sparse.issparse(adata.X):
                adata.X = adata.X.toarray()
            
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            wsi_path = list((self.hest_dir / 'wsis').glob(f"{sid}.*"))[0]
            self.wsi_paths[sid] = str(wsi_path)
            
            # Grid coords
            adata.obsm['grid_coords'] = _get_grid_coords(adata, self.n_pos)  # (N, 2)
            
            self.adatas[sid] = adata

    def __len__(self):
        return len(self.slide_ids)
    
    def __getitem__(self, idx):
        sid = self.slide_ids[idx]
        adata = self.adatas[sid]
        
        # Open WSI
        if sid not in self.wsis:
            wsi_path = self.wsi_paths[sid]
            try:
                if HAS_OPENSLIDE:
                    self.wsis[sid] = openslide.OpenSlide(wsi_path)
                else:
                    self.wsis[sid] = Image.open(wsi_path)
            except Exception as e:
                print(f"Error opening WSI for {sid}: {e}, using PIL fallback")
                self.wsis[sid] = Image.open(wsi_path)
        wsi = self.wsis[sid]
        
        # Prepare batch data (Whole Slide)
        N = adata.shape[0]
        
        # 1. Patches
        # Note: Loading all patches at once might be slow, but necessary for THItoGene structure if batch_size=1
        patches = torch.zeros((N, 3, self.patch_size, self.patch_size))
        
        spatial_coords = adata.obsm['spatial'].astype(int)
        
        for i in range(N):
            cx, cy = spatial_coords[i]
            x_tl = cx - self.r
            y_tl = cy - self.r
            
            try:
                if HAS_OPENSLIDE and isinstance(wsi, openslide.OpenSlide):
                    patch = wsi.read_region((int(x_tl), int(y_tl)), 0, (self.patch_size, self.patch_size)).convert('RGB')
                else:
                    patch = wsi.crop((x_tl, y_tl, x_tl + self.patch_size, y_tl + self.patch_size)).convert('RGB')
            except:
                patch = Image.new('RGB', (self.patch_size, self.patch_size))
            
            p_tensor = torch.from_numpy(np.array(patch)).float() / 255.0
            if p_tensor.shape[0] != self.patch_size or p_tensor.shape[1] != self.patch_size:
                 p_tensor = torch.zeros((self.patch_size, self.patch_size, 3))
            
            patches[i] = p_tensor.permute(2, 0, 1) # [C, H, W]
            
        # 2. Positions (Grid coords)
        grid_pos = torch.LongTensor(adata.obsm['grid_coords']) # (N, 2)
        
        # 3. Expressions
        exp_np = adata.X
        if hasattr(exp_np, 'toarray'):
            exp_np = exp_np.toarray()
        exp = torch.Tensor(exp_np) # (N, n_genes)

        # 4. Adjacency
        # Build adjacency on grid coords to match positional embedding space.
        adj = calcADJ(adata.obsm["grid_coords"], k=4)
        adj = torch.Tensor(adj)
        
        if self.train:
             return patches, grid_pos, exp, adj
        else:
             # Return center coords (pixel) as well for prediction saving
             centers = torch.Tensor(spatial_coords)
             return patches, grid_pos, exp, centers, adj

