<div align="center">

# Morpho-VC: Morphological Virtual Cell
### 形态学虚拟细胞系统 (CellFM-Powered)

[English](#english) | [中文](#中文)

</div>

---

<a name="english"></a>
## 🇬🇧 English

**Morpho-VC** is a "Morphological Constraint + Spatial Perturbation Condition" virtual cell system. It leverages **LazySlide** for feature extraction and **CellFM** as a foundational generative backbone to predict spatial transcriptomic responses.

### Key Features
*   **CellFM Foundational Model**: Uses pre-trained single-cell knowledge to generate biologically plausible gene expression.
*   **LazySlide Vison**: Extracts high-fidelity morphological embeddings from H&E images.
*   **Adapter Tuning**: Efficiently aligns visual features to scGPT's token space via light training.

### Installation
```bash
conda create -n morpho-vc python=3.10
conda activate morpho-vc
pip install torch scanpy opencv-python
pip install -e third_party/LazySlide
```
*(Note: CellFM weights are provided as MindSpore `.ckpt`; convert to PyTorch `.pt` via `scripts/convert_cellfm_ckpt.py`.)*

### Usage
Please refer to the [User Guide (Chinese)](docs/User_Guide_CN.md) for detailed instructions.

1.  **Extract Features**: `python src/feature_extract.py ...`
2.  **Train Model**: `python src/train.py ...`
3.  **Inference**: `python src/main_pipeline.py ...`

### ST-MIL Pipeline (sCellST-style, fully reimplemented)
This pipeline mirrors the sCellST idea (cell -> spot mapping + bag supervision) but is fully implemented inside this repo.

0. **Optional: LazySlide cell segmentation to CSV**:
    ```bash
    python src/st_pipeline/data/lazyslide_cells_to_csv.py \
      --wsi /path/to/slide.tif \
      --output_csv /path/to/cells.csv \
      --model instanseg
    ```
1. **Export cell patches** (requires cell centers CSV):
    ```bash
    python src/st_pipeline/data/cell_patch_export.py \
      --wsi /path/to/slide.tif \
      --cell_csv /path/to/cells.csv \
      --output_h5 data/cell_images/sample_cell_patches.h5
    ```
2. **Embed cells with LazySlide**:
    ```bash
    python src/st_pipeline/data/cell_embed_lazyslide.py \
      --cell_patch_h5 data/cell_images/sample_cell_patches.h5 \
      --output_h5 data/cell_embeddings/sample_cell_emb.h5 \
      --model_name resnet50
    ```
3. **Train with NB loss**:
    ```bash
    PYTHONPATH=src python src/st_pipeline/train/train_cli.py --config configs/st_mil.yaml
    ```
4. **Predict**:
    ```bash
    PYTHONPATH=src python src/st_pipeline/infer/predict_cli.py \
      --config configs/st_mil.yaml \
      --checkpoint checkpoints/st_mil/best_model.pt
    ```

Notes:
*   For CellFM, set `data.gene_vocab_path` and `model.cellfm_checkpoint` in `configs/st_mil.yaml`.
*   Ensure the vocab file matches the checkpoint gene list.

> Legacy scripts in `src/` are kept for reference but are no longer the primary pipeline.

---

<a name="中文"></a>
## 🇨🇳 中文

**Morpho-VC** 是一个基于大模型微调的虚拟细胞系统。它利用 **LazySlide** 提取 H&E 形态特征，并通过 Adapter 模块驱动 **CellFM** 单细胞基础模型生成在特定空间位置的基因表达谱。

### 核心架构升级
*   **Eye (视觉)**: 使用 LazySlide 提取 512维 图像特征。
*   **Brain (生成)**: 引入 CellFM，利用其预训练的超大规模细胞知识，进行“看图作诗”式的基因生成。
*   **Bridge (连接)**: 使用轻量级 Projector 将视觉信号翻译为 scGPT 可理解的提示符 (Prompts)。

### 安装指南
```bash
conda create -n morpho-vc python=3.10
conda activate morpho-vc
# 安装依赖
# 安装依赖
pip install torch scanpy opencv-python
# CellFM 权重为 MindSpore .ckpt，使用 scripts/convert_cellfm_ckpt.py 转成 .pt
# 安装 LazySlide 
pip install -e third_party/LazySlide
```

### 快速开始

#### 1. 运行流程
详细操作请查看 [中文使用指南 (User Guide)](docs/User_Guide_CN.md)。

*   **特征提取**:
    ```bash
    python src/feature_extract.py --image_path data/raw/slide.tif --output_dir data/processed
    ```
*   **模型微调 (LoRA/Adapter)**:
    ```bash
    python src/train.py --data_dir data/processed --epochs 20
    ```
    *注：由于使用了预训练模型，不仅收敛更快，所需数据量也更少。*
*   **推断与生成**:
    ```bash
    python src/main_pipeline.py --image_path data/raw/new_slide.tif --model_path checkpoints/best_model_scgpt.pth
    ```

### ST-MIL 管线（sCellST 思路，完全重写）
该流程在本仓库内完整重写了 sCellST 思路（细胞->spot 映射 + 包级监督），不依赖第三方实现。

配置提示：
*   在 `configs/st_mil.yaml` 中设置 `gene_vocab_path` 和 `cellfm_checkpoint`。
*   词表需与 CellFM 权重匹配。

0. **可选：LazySlide 细胞分割导出 CSV**:
    ```bash
    python src/st_pipeline/data/lazyslide_cells_to_csv.py \
      --wsi /path/to/slide.tif \
      --output_csv /path/to/cells.csv \
      --model instanseg
    ```
1. **导出细胞 patch**（需要细胞中心 CSV）:
    ```bash
    python src/st_pipeline/data/cell_patch_export.py \
      --wsi /path/to/slide.tif \
      --cell_csv /path/to/cells.csv \
      --output_h5 data/cell_images/sample_cell_patches.h5
    ```
2. **使用 LazySlide 提取细胞特征**:
    ```bash
    python src/st_pipeline/data/cell_embed_lazyslide.py \
      --cell_patch_h5 data/cell_images/sample_cell_patches.h5 \
      --output_h5 data/cell_embeddings/sample_cell_emb.h5 \
      --model_name resnet50
    ```
3. **NB 损失训练**:
    ```bash
    PYTHONPATH=src python src/st_pipeline/train/train_cli.py --config configs/st_mil.yaml
    ```
4. **推断**:
    ```bash
    PYTHONPATH=src python src/st_pipeline/infer/predict_cli.py \
      --config configs/st_mil.yaml \
      --checkpoint checkpoints/st_mil/best_model.pt
    ```

> `src/` 下旧脚本保留作参考，但不再是主流程。

### 更多文档
*   [中文使用指南](docs/User_Guide_CN.md): 包含从数据准备到 scGPT 微调的完整教程。
