# Landslide4Sense MWG-UNet++-style Toolkit (PyTorch)

## 项目介绍

这个参考项目提供了一个用于**滑坡分割**任务（如 Landslide4Sense 数据集）的完整工作流程，结合了以下关键技术：

- 基于**U-Net**的主干网络，融合轻量级**Transformer**块以增强全局上下文理解
- **条件 WGAN-GP**（掩码条件生成对抗网络）用于合成真实的图像变体，实现高效数据增强
- 标准的**BCE + Dice**分割损失函数和评估指标

> ⚠️ 这是一个模板质量的实现，旨在清晰易懂且易于修改，而非直接复现最先进的结果。请根据您的环境和数据特点调整超参数和数据增强策略。

## 项目结构

```
MWGUNetpp_Toolkit/
├── LICENSE                   # 许可证文件
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖包列表
├── scripts/                  # 执行脚本目录
│   ├── augment_with_gan.py   # 使用GAN生成增强数据
│   ├── predict.py            # 推理预测脚本
│   ├── train_gan.py          # 训练条件WGAN-GP模型
│   └── train_seg.py          # 训练分割模型
└── src/                      # 源代码目录
    ├── __init__.py
    ├── data/                 # 数据处理模块
    │   ├── __init__.py
    │   └── landslide4sense.py # Landslide4Sense风格数据集读取器
    ├── models/               # 模型定义
    │   ├── __init__.py
    │   ├── blocks.py         # 网络基础模块
    │   ├── gan.py            # 条件WGAN-GP实现
    │   └── unet_transformer.py # U-Net+Transformer分割模型
    └── utils/                # 工具函数
        ├── __init__.py
        ├── losses.py         # 损失函数实现
        ├── metrics.py        # 评估指标实现
        └── wgan_gp.py        # WGAN-GP梯度惩罚实现
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练条件 WGAN-GP（可选，用于数据增强）

```bash
python scripts/train_gan.py --data_root ./data --epochs 50 --batch_size 8
```

### 3. 用生成器合成增强数据（图像会变化；掩码沿用原始）

```bash
python scripts/augment_with_gan.py \
 --data_root ./data/train \
 --ckpt ./checkpoints/gan/latest.pt \
 --out_dir ./data/train_gan_aug \
 --variants_per_image 2
```

### 4. 训练分割模型（可同时喂原始与 GAN 增强数据）

```bash
python scripts/train_seg.py \
 --data_root ./data \
 --extra_train_dir ./data/train_gan_aug \
 --epochs 60 --batch_size 6 --lr 3e-4
```

### 5. 推理预测

```bash
python scripts/predict.py \
 --ckpt ./checkpoints/seg/latest.pt \
 --in_dir ./data/test/images \
 --out_dir ./pred_masks
```

## 设计要点详解

### Transformer 融合设计

在 U-Net 的 bottleneck 层插入轻量级 TransformerBlock，包含多头注意力机制和简单的 2D 位置编码，有效增强模型对全局上下文的理解能力。这种设计在保持 U-Net 强大细节捕获能力的同时，提升了对远距离特征依赖关系的建模能力。

### WGAN-GP 条件数据增强

- **生成器设计**：以(mask, 噪声)为条件（可选 image=0 占位），合成与掩码形状一致的新图像
- **判别器设计**：采用[image, mask] PatchGAN 结构
- **增强原理**：为每个标注掩码生成多个"风格不同"的影像样本，直接成对用于训练分割模型，显著扩充训练数据多样性

### 数据格式规范

- **默认输入**：3 通道图像与二值掩码（支持 0/255 或 0/1 格式）
- **多光谱扩展**：如需处理多光谱数据，可修改 in_channels 参数并在数据读取模块中进行相应处理

### 损失函数与评估指标

- **训练损失**：采用 BCE+Dice 组合损失函数，权重可在`src/utils/losses.py`中调整
- **评估指标**：验证过程输出 IoU（交并比）和 Dice 系数作为主要性能指标

## 扩展可能性

该项目架构具有良好的扩展性，可根据需求进行以下扩展：

- 支持**多通道输入**（多光谱遥感数据）
- 扩展为**多类分割**任务
- 集成**更强的数据增强库**（如 albumentations）
- 调整 Transformer 模块位置（如放置在跳连或解码阶段）

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA 支持（推荐）
