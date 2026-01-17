# Morpho-VC 使用指南（以 `st_mil_hest_multi.ipynb` 为主）

本指南聚焦当前**主流程**：以 **Morpho-VC 的 ST-MIL 训练流程**为核心，外部组件按需接入。
请以 `notebooks/st_mil_hest_multi.ipynb` 为主入口，`notebooks/st_mil_hest_validate.ipynb` 为评估入口。

---

## 0. 示例脚本与 Notebook

- [notebooks/st_mil_hest_multi.ipynb](../notebooks/st_mil_hest_multi.ipynb)（主流程训练+预测）
- [notebooks/st_mil_hest_validate.ipynb](../notebooks/st_mil_hest_validate.ipynb)（仅评估）
- [scripts/convert_cellfm_ckpt.py](../scripts/convert_cellfm_ckpt.py)（权重转换）
- [configs/st_mil.yaml](../configs/st_mil.yaml)（CLI 配置）

---

## 1. 环境与依赖

### 1.1 建议环境
```bash
conda create -n morpho-vc python=3.10
conda activate morpho-vc
```

### 1.2 必备依赖
```bash
pip install torch torchvision numpy pandas scipy h5py scanpy anndata matplotlib
pip install timm safetensors opencv-python openslide-python
```

### 1.3 可选依赖（HEST 下载 / 空间几何）
```bash
pip install datasets huggingface_hub
pip install geopandas pyogrio shapely
```

> 如果你使用 HEST 的原始切片 + 组织轮廓文件，建议用 conda-forge 安装 `geopandas/pyogrio/gdal/proj`，避免版本冲突。

---

## 2. 外部组件准备（不随 git 跟踪）

如需外部组件，请将以下仓库放到 `third_party/` 目录：
- `third_party/CellFM`
- `third_party/LazySlide`
- `third_party/HEST`

**LazySlide（可选）安装**：
```bash
pip install -e third_party/LazySlide
```

**HEST（可选）说明**：
需要保证 `third_party/HEST/src/hest` 存在，Notebook 会自动 `sys.path` 引入。

---

## 3. 权重与词表（非常关键）

- CellFM 官方权重为 MindSpore `.ckpt`，需转换为 PyTorch `.pt`：
```bash
python scripts/convert_cellfm_ckpt.py --ckpt /path/to/CellFM_80M_weight.ckpt --out /path/to/CellFM_80M_weight.pt
```

- **80M 权重必须使用 `expand_gene_info.csv`**（行数 27855），否则 `gene_emb` 无法加载，预测会塌缩。

推荐词表路径：
```
third_party/CellFM/csv/expand_gene_info.csv
```

---

## 4. 数据目录约定

主流程默认使用以下目录：
```
Morpho-VC/
  data/
    hest_data/          # HEST 原始数据
    spatial_data/       # h5ad / common_genes.txt
    cell_centers/       # 细胞中心 CSV
    cell_images/        # 细胞 patch h5
    cell_embeddings/    # 细胞 embedding h5
    spot_cell_maps/     # spot->cell 映射 json
  results/
    st_mil_hest/
      val/
      test/
```

这些目录不会上传到 GitHub（见 `.gitignore`）。

---

## 5. 主流程：`st_mil_hest_multi.ipynb`

**你应该只关注这个 Notebook。**
建议按顺序执行每个 cell：

1) **配置路径与切片 ID**
- 确认 `ROOT`、`data_dir`、`cellfm_checkpoint`、`gene_vocab_path`。

2) **准备 HEST 数据**
- 有网：使用 `download_hest` 下载。
- 无网：手动放入 `data/hest_data/`。

3) **导出 ST h5ad**
- 由 HEST 原始数据导出 `spatial_data/{slide_id}.h5ad`。

4) **细胞分割与 patch 导出**
- 可用 LazySlide 或已有分割结果。
- 关键：必须生成细胞中心 CSV（`cell_id,x,y`）。

5) **细胞 embedding**
- 运行 LazySlide embedding，保存到 `cell_embeddings`。

6) **构建 MIL 数据集**
- `MilSpotDataset` 会自动生成 `spot_cell_maps`，如果文件已存在会复用。

7) **训练（NB 损失 + gene chunking）**
- 支持多卡 DDP（分布式数据并行）：
```bash
torchrun --standalone --nproc_per_node=4 st_mil_hest_multi.py
```
- 显存不足时降低 `batch_size` 或 `gene_chunk_size`。

8) **预测并保存**
- 输出 `pred_bag.npy` / `true_bag.npy` 到 `results/st_mil_hest/{split}`。

---

## 6. 评估流程：`st_mil_hest_validate.ipynb`

该 Notebook **不再重新预测**，只读取保存结果进行评估：
- MAE / RMSE
- 按基因 Pearson
- **真实高变基因（HVG）评估**
- 预测 HVG 与真实 HVG 的重叠度

推荐评估策略：
- 主评估基于 **真实高变基因**，避免低表达噪声影响。

---

## 7. 常见问题

### Q1: 报错 “Missing CellFM keys: ['gene_emb', ...]”
**原因**：词表与权重不匹配。
**解决**：使用 `expand_gene_info.csv` 并重新训练。

### Q2: `value_enc.value_enc.a` 缺失要紧吗？
**不影响主流程**。这部分是 CellFM 的表达值编码器参数，当前 pipeline 走的是 prompt 路径，可忽略。

### Q3: 预测的基因分布都很像
**原因**：CellFM 编码器冻结过多，只训练 gene_emb/decoder。
**建议**：解冻最后一层 encoder 或分阶段微调。

### Q4: 显存不足（OOM）
- 降低 `batch_size` 或 `gene_chunk_size`
- 使用 DDP 多卡训练
- 开启 AMP

---

如果你希望我把 Notebook 的默认参数、路径和运行脚本都进一步固化（比如 Slurm 脚本、统一配置文件），告诉我即可。
