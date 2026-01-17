<div align="center">

# Morpho-VC: Morphological Virtual Cell
### 形态学虚拟细胞系统（以 Morpho-VC 创新流程为核心）

[English](#english) | [中文](#中文)

</div>

---

<a name="english"></a>
## 🇬🇧 English

**Morpho-VC** is a virtual cell system that predicts **spatial transcriptomic gene expression** from H&E whole-slide images. The project emphasizes a **cell-to-spot ST-MIL pipeline**, gene-aware supervision, and scalable training, with external components plugged in as needed.

### Key Features (Our Innovations)
- **Cell-to-spot ST-MIL pipeline** with explicit spot aggregation and NB (Negative Binomial) loss.
- **Gene-aware training strategy** with chunked supervision to scale to large gene sets.
- **Multi-slice training + evaluation workflow** (train/val/test split across slides).
- **Notebook-first reproducibility** with a single main workflow (`notebooks/st_mil_hest_multi.ipynb`).

### User Guide
- [User Guide (Chinese)](docs/User_Guide_CN.md)

### Example Notebooks & Scripts
- [notebooks/st_mil_hest_multi.ipynb](notebooks/st_mil_hest_multi.ipynb) (main training + prediction)
- [notebooks/st_mil_hest_validate.ipynb](notebooks/st_mil_hest_validate.ipynb) (evaluation only)
- [scripts/convert_cellfm_ckpt.py](scripts/convert_cellfm_ckpt.py) (CellFM ckpt -> pt)
- [configs/st_mil.yaml](configs/st_mil.yaml) (CLI config)

### Required Packages
Core (minimum to run notebooks):
```bash
pip install torch torchvision numpy pandas scipy h5py scanpy anndata matplotlib
pip install timm safetensors opencv-python openslide-python
```
Optional (HEST download / geometry support):
```bash
pip install datasets huggingface_hub
pip install geopandas pyogrio shapely
```

### External Components (not tracked in git)
If you use external toolkits/models, place them under `third_party/` (examples below):
- `CellFM`: https://github.com/biomed-AI/CellFM
- `LazySlide`: https://github.com/rendeirolab/LazySlide
- `HEST`: https://github.com/mahmoodlab/hest/

### Checkpoints + Vocab
- Some external weights are MindSpore `.ckpt`.
- Convert to PyTorch `.pt` via:
```bash
python scripts/convert_cellfm_ckpt.py --ckpt /path/to/CellFM_80M_weight.ckpt --out /path/to/CellFM_80M_weight.pt
```
- **80M weights must use** `expand_gene_info.csv` (not `gene_info.csv`).

### Main Workflow
1. **Training + Prediction**: open `notebooks/st_mil_hest_multi.ipynb`
2. **Evaluation** (reads saved results): `notebooks/st_mil_hest_validate.ipynb`

### Optional CLI (advanced)
```bash
PYTHONPATH=src python src/st_pipeline/train/train_cli.py --config configs/st_mil.yaml
PYTHONPATH=src python src/st_pipeline/infer/predict_cli.py --config configs/st_mil.yaml --checkpoint checkpoints/st_mil/best_model.pt
```

> Data, checkpoints, results are intentionally excluded from git. See `.gitignore` rules in your local repo.

---

<a name="中文"></a>
## 🇨🇳 中文

**Morpho-VC** 是一个“看图预测基因表达”的虚拟细胞系统。核心是 **Morpho-VC 自身的 ST-MIL 训练流程**，并支持按需接入外部组件。

### 核心特点（我们的创新点）
- **细胞→spot 的 ST-MIL 管线**：显式聚合 + NB 损失。
- **大规模基因监督**：分块训练策略，降低显存占用。
- **多切片训练/验证/测试**：更接近真实数据评估。
- **Notebook 主流程**：`notebooks/st_mil_hest_multi.ipynb`。

### 使用指南
- [中文使用指南](docs/User_Guide_CN.md)

### 示例脚本与 Notebook
- [notebooks/st_mil_hest_multi.ipynb](notebooks/st_mil_hest_multi.ipynb)（主流程训练+预测）
- [notebooks/st_mil_hest_validate.ipynb](notebooks/st_mil_hest_validate.ipynb)（仅评估）
- [scripts/convert_cellfm_ckpt.py](scripts/convert_cellfm_ckpt.py)（权重转换）
- [configs/st_mil.yaml](configs/st_mil.yaml)（CLI 配置）

### 必备依赖
核心依赖：
```bash
pip install torch torchvision numpy pandas scipy h5py scanpy anndata matplotlib
pip install timm safetensors opencv-python openslide-python
```
可选依赖（下载 HEST / 空间几何）：
```bash
pip install datasets huggingface_hub
pip install geopandas pyogrio shapely
```

### 外部组件（不随 git 跟踪）
如需外部组件，请手动放到 `third_party/`：
- `CellFM`
- `LazySlide`
- `HEST`
参考链接：
- HEST: https://github.com/mahmoodlab/hest/
- CellFM: https://github.com/biomed-AI/CellFM
- LazySlide: https://github.com/rendeirolab/LazySlide

### 权重与词表
- 外部权重通常是 MindSpore `.ckpt`，需转换成 `.pt`。
- 80M 权重必须使用 **`expand_gene_info.csv`**。

### 推荐流程
1) 打开 `notebooks/st_mil_hest_multi.ipynb` 进行训练和预测
2) 打开 `notebooks/st_mil_hest_validate.ipynb` 做评估（读取已保存结果）

### 可选命令行
```bash
PYTHONPATH=src python src/st_pipeline/train/train_cli.py --config configs/st_mil.yaml
PYTHONPATH=src python src/st_pipeline/infer/predict_cli.py --config configs/st_mil.yaml --checkpoint checkpoints/st_mil/best_model.pt
```

> 数据、权重、结果目录不会上传到 GitHub，请保持本地存储。
