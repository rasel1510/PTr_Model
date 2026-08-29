# Enhanced Pyramid Transformer–CNN Architecture for Accurate Diabetic Foot Ulcer Detection

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Journal](https://img.shields.io/badge/Submitted%20To-Elsevier%20Q1%20Journal-007398.svg?logo=elsevier&logoColor=white)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-success.svg)](#experimental-results)
[![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM%20%7C%20SAM-orange.svg)](#explainable-ai--segmentation)


 
**"Enhanced Pyramid Transformer–CNN Architecture for Accurate Diabetic Foot Ulcer Detection"**  



## 📖 Abstract

Quick recognition of Diabetic Foot Ulcers (DFUs) significantly improves survival outcomes patients. Whatever, accurate clinical diagnosis remains challenging.
While numerous deep learning (DL) driven approaches have developed, many lack extensive accuracy also accessible boundary annotations. This study introduces an enhanced transformer model using transfer learning (TL) with a novel Pyramid Transformer (PTr) architecture to address these limitations. Unlike
traditional CNNs which are constrained by fixed receptive fields, our proposed PTr leverages spatial-reduction attention (SRA) to adaptively capture native as
well as universal contextual reliance’s. It is especially crucial to recognize DFU, where ulcer sizes, textures, and shapes vary significantly, demanding flexible and scale-aware feature extraction. Extensive evaluations were conducted across two datasets, achieving remarkable performance including accuracy,precision of 99.98% and 99.99% respectively. Further validation on an additional dataset of 3,026 annotated DFU images confirmed the robustness of the model, with results showing 97.00% recall, 97.50% precision, 98.00% F1-score, and 99.20% AUC—outperforming all compared models. Additionally, the Segment Anything Model (SAM) delivered excellent segmentation contributions. In order to enhance explainability, Grad-CAM was employed into Xception model, providing visual insights into key decision regions. Overall, the proposed PTr-based methodology demonstrates significant improvements in feature extraction, detection accuracy, and clinical applicability. This framework holds strong potential to support healthcare professionals in achieving early and reliable DFU diagnosis, ultimately improving patient care.


---

## 🌟 Key Highlights & Major Contributions

1. **Enhanced PTr Architecture with Transfer Learning**: We propose an improved transformer-based model (Pyramid Transformer - PTr) integrated with transfer learning, achieving superior feature extraction, robust generalization, and state-of-the-art accuracy for DFU classification.
2. **Scale-Aware Spatial-Reduction Attention (SRA)**: Our PTr with Spatial-Reduction Attention (SRA) adaptively captures native local and universal global contextual dependencies, enabling highly robust DFU detection across diverse ulcer sizes, irregular textures, and morphological shapes.
3. **Precise Lesion Boundary Segmentation via SAM**: We employ the Segment Anything Model (SAM) for segmentation, providing precise pixel-level localization and boundary delineation of DFU-affected regions to support accurate clinical diagnosis and treatment planning.
4. **Transparent Explainable AI (XAI) via Grad-CAM**: To enhance clinical trust and interpretability, we integrate Grad-CAM visualizations that highlight key ulcer and normal decision regions, offering complete transparency in the model's diagnostic decision-making process.

---

## 🏗️ Methodology & Architecture Overview

### 1. Overall Methodology & Clinical Pipeline
![DFU Methodology Pipeline](Methodology%20Structures/DFU_methodology.png)

### 2. Proposed Pyramid Transformer (PTr) Architecture with Transfer Learning
![Proposed Pyramid Transformer with Transfer Learning](Methodology%20Structures/DFU_PyramidTran_TL.png)

### 3. Baseline CNN Architecture (DenseNet-201)
![DenseNet201 Architecture](Methodology%20Structures/DFU_DenseNet201.png)

### 4. Segment Anything Model (SAM) Architecture
![Segment Anything Model Architecture](Methodology%20Structures/DFU_SamArchitect.png)

---

## 🔬 Explainable AI & Segmentation (SAM + Grad-CAM)


### Qualitative SAM Segmentation Performance
![SAM Qualitative Segmentation Output](Methodology%20Structures/SamSegmentation%20Result.png)

---

## 📁 Repository Structure

```tree
.
├── Methodology Structures/            # Architecture diagrams and qualitative visual figures
│   ├── DFU_DenseNet201.png            # DenseNet-201 baseline architecture diagram
│   ├── DFU_PyramidTran_TL.png         # Proposed Pyramid Transformer + TL architecture
│   ├── DFU_SamArchitect.png           # SAM segmentation pipeline
│   ├── DFU_methodology.png            # Overall study workflow & methodology diagram
│   └── SamSegmentation Result.png     # Qualitative SAM segmentation outputs
├── main_method.py                     # Core PyTorch source code (Model, SRA, Dataloaders, Trainer)
├── requirements.txt                   # Complete dependency specifications
└── README.md                          # Comprehensive documentation & reviewer guide
```


## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- NVIDIA CUDA 11.8+ / 12.x enabled GPU (recommended for training)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/DFU-Pyramid-Transformer.git
cd DFU-Pyramid-Transformer
```

### Step 2: Create a Virtual Environment
```bash
# Using Conda
conda create -n dfu_ptr python=3.10 -y
conda activate dfu_ptr

# Or using Python venv
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

*(If `requirements.txt` is not yet created, install the core packages directly: `pip install torch torchvision timm scikit-learn numpy pillow opencv-python matplotlib`)*

---

## 🗂️ Data Preparation

The pipeline supports both pre-split directory structures (`train/val/test`) and unified directories with automatic stratified 70/30 splitting:

### Expected Directory Layout:
```
DFU_dataset/
├── train/
│   ├── Healthy_Foot/
│   │   ├── normal_001.jpg
│   │   └── ...
│   └── Ulcer_Foot/
│       ├── ulcer_001.jpg
│       └── ...
├── val/
│   ├── Healthy_Foot/
│   │   └── ...
│   └── Ulcer_Foot/
│       └── ...
└── test/ (optional)
    ├── Healthy_Foot/
    └── Ulcer_Foot/
```
*Alternatively, place class folders (`Healthy_Foot`, `Ulcer_Foot`) directly inside `DFU_dataset/` and the dataloader will perform the 70% train / 30% validation split automatically with fixed random seeding (`seed=42`).*

---

## 🚀 Usage & Execution Guide

### 1. Synthetic Architecture Verification (No Dataset Required)
Verify tensor shapes, spatial reduction operations, and parameter counts:
```bash
python main_method.py
```
**Expected Output:**
```
[ENVIRONMENT] Execution target: cuda
[MODEL] Proposed PTr loaded. Total Trainable Parameters: 13.24M
[FORWARD CHECK] Input: torch.Size([2, 3, 224, 224]) -> Logits Output: torch.Size([2, 2]) (Valid)
```

### 2. Train on Live DFU Dataset
Configure the dataset path in `main_method.py` or run directly:
```python
# In main_method.py
DFU_DATA_DIR = "./path/to/your/DFU_dataset"
```
Execute training:
```bash
python main_method.py
```


---

## 💻 Reproducibility & Hardware Specifications

All experiments reported in the paper were conducted under the following standardized experimental setup:

- **Operating System**: Ubuntu 22.04 LTS / Windows 11 Pro 64-bit
- **CPU**: Intel Xeon / Core i9 (16 Cores, 3.20 GHz)
- **GPU**: NVIDIA RTX 3090 (24 GB VRAM) / NVIDIA A100 (40 GB VRAM)
- **Framework**: PyTorch 2.x with CUDA 11.8 / 12.1 and cuDNN backend
- **Random Seed**: Fixed to `42` across NumPy, PyTorch, and CUDA generators for deterministic execution.

---




## 📄 License & Acknowledgements

- **License**: This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.
- **Ethics Statement**: All clinical photographic data analyzed adhere strictly to ethical guidelines and institutional review approvals for biomedical image analysis.
- **Acknowledgements**: We acknowledge the open-source community, the developers of PyTorch, timm, and the Segment Anything Model (Meta AI Research).

---
*For questions, inquiries, or peer-review clarifications, please open an issue in this repository or contact the corresponding author directly.*
