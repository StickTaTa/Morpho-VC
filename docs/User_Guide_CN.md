# Morpho-VC (CellFM Edition) 深度使用指南

欢迎使用 **Morpho-VC**。这是一个前沿的“AI 虚拟细胞”系统，旨在通过计算机视觉与单细胞大模型技术，从普通的病理图像（H&E）中“还原”出昂贵的空间转录组信息。

本文档包含 **原理详解** 与 **实战操作** 两大部分，旨在帮助您不仅“会用”，还能“懂原理”。

---

# 第一部分：原理详解 (The "Why" & "How")

## 1. 核心科学问题
我们面临的挑战是：**只有一张普通的组织切片照片（H&E），能否预测出每个细胞在表达什么基因？**
*   **传统方法**：训练一个简单的回归模型（CNN），输入图像，强制输出基因数值。缺点是基因之间互不相关，结果往往不符合生物学规律（比如它可能预测出两个功能相反的基因同时高表达）。
*   **Morpho-VC 的思路**：我们不再让 AI “死记硬背”，而是请一位“懂生物学的专家”来看图。

## 2. 系统架构：Vision + Brain
本系统采用了类似于 GPT-4V 的多模态架构，由三个核心组件构成：

### (1) 眼睛：LazySlide (Visual Encoder)
*   **作用**：看懂图像。
*   **原理**：LazySlide 是一个专为病理图像优化的视觉模型。它将一张 H&E 切片分割成无数个 256x256 的小方块（Patch），并将每个方块转化成一个 512 维的数学向量。这个向量浓缩了细胞的形态、纹理、排列等信息。

### (2) 大脑：CellFM (Foundation Backbone)
*   **作用**：基于预训练知识生成基因表达。
*   **原理**：CellFM 是在超大规模单细胞转录组数据上预训练的基础模型，擅长建模基因共表达与调控结构。
*   **为什么用它？** 因为它提供稳定的基因语义空间，我们只需要训练 Adapter，把图像特征映射到 CellFM 的细胞表示即可。

### (3) 桥梁：Modality Adapter (The Driver)
*   **作用**：翻译。
*   **原理**：这是我们需要**重点训练**的部分。CellFM 只懂“基因语言”，不懂“图像语言”。Adapter 负责把 LazySlide 看到的“图像特征”翻译成 CellFM 可用的“细胞提示符 (Cell Token)”。
    *   *形象比喻*：Adapter 告诉 CellFM：“这里是一群密集的淋巴细胞（图像特征），请生成对应的免疫相关基因表达谱。”

---

# 第二部分：环境与准备 (Preparation)

## 1. 硬件要求
*   **GPU**：强烈建议使用 NVIDIA GPU (显存 >= 16GB)。CellFM 是大模型，CPU 运行会非常慢。
*   **内存**：建议 32GB 以上。

## 2. 软件环境
我们推荐使用 Conda 管理环境。

```bash
# 1. 创建环境
conda create -n morpho-vc python=3.10
conda activate morpho-vc

# 2. 安装基础依赖
pip install torch torchvision scanpy opencv-python pandas matplotlib seaborn

# 3. 准备 CellFM 权重
# 官方权重为 MindSpore .ckpt，需要转换为 PyTorch .pt
python scripts/convert_cellfm_ckpt.py --ckpt /path/to/CellFM_80M_weight.ckpt --out /path/to/CellFM_80M_weight.pt

# 4. 安装 LazySlide (本项目自带)
cd /path/to/Morpho-VC
pip install -e third_party/LazySlide

# 5. 旧版 scGPT 依赖（可选）
# 仅在使用 legacy pipeline（src/train.py）时需要
# pip install scgpt
```

## 3. 数据准备规范
数据质量决定模型上限。请严格按照以下结构整理您的数据：

*   **原始数据 (`data/raw/`)**: 放 .tif / .svs 等病理大图。
*   **训练数据 (`data/processed/`)**:
    *   你需要为训练集准备“标准答案”（Ground Truth）。
    *   对于每个样本（如 `sample_A`），需要生成以下 `.npy` 文件：
        1.  `features.npy`: [N, 512] - 图像特征（由脚本生成）。
        2.  `gene_exp.npy`: [N, Genes] - 真实的空间转录组表达矩阵（需您提供并预处理）。
        3.  `coords.npy`: [N, 2] - 空间坐标。
        4.  `perturbations.csv`: [N, 1] - 扰动标签（如 0=Control, 1=DrugA）。

> **关键点**：`gene_exp.npy` 的行顺序必须与 `features.npy` 的行顺序严格一一对应（即第 i 行的图像特征对应第 i 行的基因表达）。

---

# 第三部分：实战操作指南 (Step-by-Step)

## 步骤 1：特征提取 (Feature Extraction)
这一步将图像转化为机器可读的向量。

**命令**：
```bash
python src/feature_extract.py \
  --image_path data/raw/cancer_slice_01.tif \
  --output_dir data/processed/cancer_slice_01
```

**发生了什么？**
*   程序会自动切分大图。
*   LazySlide 模型读取每个小块。
*   生成 `features.npy` 和 `coords.npy`。
*   *注意：此时还没有 gene_exp.npy，如果您是用作训练集，需要手动将对应的转录组矩阵保存为 gene_exp.npy 放入该目录。*

## 步骤 2：模型微调 (Fine-tuning)
本节为 **legacy scGPT pipeline** 的说明（对应 `src/train.py`）。如果你只使用 ST-MIL + CellFM，可以直接跳到后面的 ST-MIL 管线。
这是最核心的一步。我们将训练 Adapter 让 scGPT 学会“看图”。

**参数详解**：
*   `--data_dir`: 数据目录。
*   `--scgpt_path`: **重要！** 指向您下载的 scGPT 预训练权重文件（如 `best_model.pt`）。如果留空，系统会使用随机初始化的 Mock 模型（仅用于测试代码跑通，**无实际预测能力**）。
*   `--epochs`: 训练轮数。因为是微调，通常 20-50 轮即可。
*   `--lr`: 学习率。微调时宜小，推荐 1e-4。

**命令**：
```bash
python src/train.py \
  --data_dir data/processed/cancer_slice_01 \
  --scgpt_path /path/to/models/scgpt_human_blood.pt \
  --epochs 20 \
  --batch_size 16 \
  --lr 0.0001
```

**如何判断训练好了？**
*   观察终端输出的 `Train Loss` 和 `Val Loss`。
*   如果 Loss 持续下降并趋于平稳，说明模型正在学习。
*   如果是 Mock 模式，Loss 可能很难降得很低，这是正常的。

## 步骤 3：推断与生成 (Inference)
模型训练好后（保存为 `checkpoints/best_model_scgpt.pth`），就可以用来处理新数据了。

**场景**：
手里有一张新的病人切片 `patient_new.tif`，没有做过空间转录组测序，想预测其基因表达。

**流程**：
1.  **先提特征**：对新图运行 `feature_extract.py`。
2.  **再生成**：运行 `main_pipeline.py`。

**命令**：
```bash
python src/main_pipeline.py \
  --image_path data/raw/patient_new.tif \
  --model_path checkpoints/best_model_scgpt.pth \
  --output_dir results/patient_new_pred \
  --input_dim 512 \
  --n_genes 100 \
  --perturbation_id 1
```

**结果解读**：
打开 `results/patient_new_pred` 目录：
*   `spatial_response_map_scgpt.png`: 这是一张热图。颜色越红，代表在该区域模型预测的基因表达变化越剧烈（或目标基因表达越高）。
*   `pred_perturbed.npy`: 这是原本不存在的、AI 生成的完整基因表达矩阵。您可以直接用 Python 加载它进行后续的 Scanpy 分析。

---

## 新增：ST-MIL 管线（sCellST 思路，完全重写）
该流程复刻 sCellST 的训练逻辑（细胞->spot 映射 + 包级监督），但实现全部在本项目中完成，不依赖 sCellST 的代码。损失函数使用负二项分布（Negative Binomial, NB），适合原始计数数据。

**你需要准备的输入**：
1.  **H&E 全切片图像（WSI）**  
2.  **空间转录组 h5ad**：要求 `obsm["spatial"]` 有 spot 坐标。  
3.  **细胞分割结果**：至少包含每个细胞的中心坐标（CSV 格式，`cell_id,x,y`）。  

**如果使用 HEST 原始数据（本地仓库）**：
请将 HEST 仓库放到 `third_party/HEST`，并确保存在 `third_party/HEST/src/hest`。Notebook 会自动将该路径加入 `sys.path`，无需 pip 安装。

**可选：用 LazySlide 生成细胞中心 CSV**
```bash
python src/st_pipeline/data/lazyslide_cells_to_csv.py \
  --wsi /path/to/slide.tif \
  --output_csv /path/to/cells.csv \
  --model instanseg
```

**步骤 A：导出细胞 patch**
```bash
python src/st_pipeline/data/cell_patch_export.py \
  --wsi /path/to/slide.tif \
  --cell_csv /path/to/cells.csv \
  --output_h5 data/cell_images/sample_cell_patches.h5
```

**步骤 B：使用 LazySlide 提取细胞特征**
```bash
python src/st_pipeline/data/cell_embed_lazyslide.py \
  --cell_patch_h5 data/cell_images/sample_cell_patches.h5 \
  --output_h5 data/cell_embeddings/sample_cell_emb.h5 \
  --model_name resnet50
```

**步骤 C：NB 损失训练**
```bash
PYTHONPATH=src python src/st_pipeline/train/train_cli.py --config configs/st_mil.yaml
```

**步骤 D：推断**
```bash
PYTHONPATH=src python src/st_pipeline/infer/predict_cli.py \
  --config configs/st_mil.yaml \
  --checkpoint checkpoints/st_mil/best_model.pt
```

**配置提示**：
*   `configs/st_mil.yaml` 里需要配置 `h5ad_path`、`cell_emb_h5`、`spot_radius_px` 等路径和参数。  
*   如果 `h5ad` 里不带 spot 直径信息，务必在配置里手动填写 `spot_radius_px`。  
*   在服务器上运行时，请在 `morpho_vc` 环境中执行，CellFM 权重由 `cellfm_checkpoint` 指定，基因词表由 `gene_vocab_path` 指定。  
*   `gene_vocab_path` 必须与 CellFM 权重使用的词表一致。  

---

# 第四部分：进阶技巧与常见问题

## Q1: 这里提到的 "Adapter" 到底是什么？
A: 在代码 `src/st_pipeline/model/adapter.py` 中，您会看到 `VisualAdapter` 类。它本质上是一个两层的全连接神经网络（Linear -> Relu -> Linear）。
*   它的输入是图像特征向量。
*   它的输出是 CellFM 的细胞提示符（Cell Token）。
*   训练它，就是训练这个“翻译官”把图像信号转译成大模型能理解的信号。

## Q2: 如何提高预测准确度？
1.  **使用真实的 CellFM 权重**：不要使用默认的 Mock 模式，先把官方 .ckpt 转成 .pt 再加载。
2.  **数据对齐**：确保训练时的 H&E 图像切片（Patch）和空间转录组的 Spot 是通过物理坐标严格对齐的。如果错位，模型学不到正确关系。
3.  **增加数据量**：大模型喜欢大数据。如果只有一张切片训练，容易过拟合。尝试混合多张切片的数据。
4.  **解冻部分 CellFM**：在 `configs/st_mil.yaml` 中，将 `freeze_cellfm` 改为 `false`，或者只解冻最后几层 Retention Layer（需要额外代码控制）。

## Q3: 报错 "RuntimeError: CUDA out of memory"？
A: CellFM 显存占用较大。
*   尝试减小 `--batch_size` (如设为 4 或 1)。
*   开启梯度累积 (Gradient Accumulation) - *注：当前脚本需自行修改实现此功能*。
*   使用 `--use_mock` 在 CPU 上先跑通流程调试代码。
