<div align="center">

# Morpho-VC: Morphological Virtual Cell
### 形态学虚拟细胞系统 (LazySlide + CellFM)

[English](#english) | [中文](#中文)

</div>

---

<a name="english"></a>
## 🇬🇧 English

**Morpho-VC** is a virtual cell system that predicts **spatial transcriptomic gene expression** from H&E whole-slide images. It combines **LazySlide** for cell-level morphology embeddings and **CellFM** as the gene foundation model, trained with a **ST-MIL (Multi-Instance Learning)** pipeline.

### Key Features
- **LazySlide vision**: cell patch extraction + morphology embeddings.
- **CellFM backbone**: gene embedding space for biologically plausible prediction.
- **ST-MIL training**: cell-to-spot mapping with NB (Negative Binomial) loss.
- **Notebook-first workflow**: the main guide is `notebooks/st_mil_hest_multi.ipynb`.

### User Guide
- [User Guide (Chinese)](docs/User_Guide_CN.md)

### Example Notebooks & Scripts
- [notebooks/st_mil_hest_multi.ipynb](notebooks/st_mil_hest_multi.ipynb) (main training + prediction)
- [notebooks/st_mil_hest_validate.ipynb](notebooks/st_mil_hest_validate.ipynb) (evaluation only)
- [notebooks/st_mil_with_hest_raw.ipynb](notebooks/st_mil_with_hest_raw.ipynb) (single-slice sanity check)
- [notebooks/st_mil_validation.ipynb](notebooks/st_mil_validation.ipynb) (legacy validation)
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

### Third-Party Repos (not tracked in git)
Place these under `third_party/`:
- `CellFM`
- `LazySlide`
- `HEST`

### Checkpoints + Vocab
- CellFM official weights are MindSpore `.ckpt`.
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

**Morpho-VC** 是一个“看图预测基因表达”的虚拟细胞系统。核心流程是：
**LazySlide 提取细胞形态特征 + CellFM 基因基础模型 + ST-MIL 训练（负二项损失）**。

### 核心特点
- **LazySlide 视觉端**：细胞 patch + 形态学嵌入。
- **CellFM 基因端**：稳定的基因嵌入空间。
- **ST-MIL 管线**：细胞→spot 映射 + 包级监督（NB 损失）。
- **Notebook 作为主流程**：`notebooks/st_mil_hest_multi.ipynb`。

### 使用指南
- [中文使用指南](docs/User_Guide_CN.md)

### 示例脚本与 Notebook
- [notebooks/st_mil_hest_multi.ipynb](notebooks/st_mil_hest_multi.ipynb)（主流程训练+预测）
- [notebooks/st_mil_hest_validate.ipynb](notebooks/st_mil_hest_validate.ipynb)（仅评估）
- [notebooks/st_mil_with_hest_raw.ipynb](notebooks/st_mil_with_hest_raw.ipynb)（单切片检查）
- [notebooks/st_mil_validation.ipynb](notebooks/st_mil_validation.ipynb)（旧版验证）
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

### 第三方仓库（不随 git 跟踪）
请手动放到 `third_party/`：
- `CellFM`
- `LazySlide`
- `HEST`

### 权重与词表
- CellFM 官方权重是 MindSpore `.ckpt`，需转换成 `.pt`。
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
