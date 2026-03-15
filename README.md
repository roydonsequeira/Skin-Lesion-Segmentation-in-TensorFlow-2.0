# 🩺 Skin Lesion Segmentation — Final Year Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Semantic segmentation of skin lesions** on the **ISIC 2018** dataset using deep learning. This repository implements **U-Net** (PyTorch) and **ResU-Net** (TensorFlow 2.0) for automated lesion boundary detection to support dermatological analysis.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Architectures](#-architectures)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Metrics](#-results--metrics)
- [Screenshots & Visualizations](#-screenshots--visualizations)
- [Future Work](#-future-work)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🎯 Overview

Skin lesion segmentation is a key step in computer-aided diagnosis (CAD) for melanoma and other skin conditions. This project provides:

- **Two model implementations**: U-Net (PyTorch) and ResU-Net (TensorFlow 2.0)
- **End-to-end pipeline**: data loading, training, evaluation, and visualization
- **Interactive demo**: Streamlit app for uploading images and viewing predicted masks and lesion dimensions
- **Standard metrics**: Dice coefficient, IoU, accuracy, F1, precision, recall

| **Input image** | **Ground truth mask** |
|:----------------|:----------------------|
| ![Sample input](img/image.jpg) | ![Sample mask](img/mask.png) |

---

## ✨ Features

- **U-Net** (PyTorch): encoder–decoder with skip connections; training with BCE + Dice-style metrics
- **ResU-Net** (TensorFlow 2.0): residual U-Net with residual blocks in encoder/decoder
- **ISIC 2018**-compatible data loader (images + ground truth masks)
- **Training**: checkpointing, CSV logging, TensorBoard, learning rate scheduling, early stopping
- **Evaluation**: batch prediction, per-image and aggregate metrics (Accuracy, F1, Jaccard, Recall, Precision)
- **Visualization**: training curves, metric scatter plots, ROC–AUC (U-Net)
- **Streamlit app**: upload a dermoscopic image → view segmentation mask and lesion dimensions (bounding box, area)

---

## 📂 Dataset

We use the **[ISIC 2018 Challenge](https://challenge.isic-archive.com/data/)** dataset:

- **Task 1 – Lesion boundary segmentation**
  - **Input**: `ISIC2018_Task1-2_Training_Input/*.jpg` — dermoscopic images
  - **Ground truth**: `ISIC2018_Task1_Training_GroundTruth/*.png` — binary masks
- **Size**: 2,594 image–mask pairs (variable resolution; resized to 256×256 in the code)

### Expected directory layout

Place the dataset so that paths look like:

```
your_dataset_path/
├── ISIC2018_Task1-2_Training_Input/
│   ├── ISIC_0000001.jpg
│   └── ...
└── ISIC2018_Task1_Training_GroundTruth/
    ├── ISIC_0000001_segmentation.png
    └── ...
```

*(Exact filenames may differ; ensure image and mask lists align by index or naming convention.)*

---

## 📁 Project Structure

```
Skin-Lesion-Segmentation-in-TensorFlow-2.0-main/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── img/                      # Sample images for README
│   ├── image.jpg             # Sample input
│   └── mask.png              # Sample mask
├── UNET/                     # U-Net (PyTorch)
│   ├── model.py              # U-Net architecture
│   ├── train.py              # Training script
│   ├── eval.py               # Evaluation & metrics
│   ├── metrics.py            # Dice, IoU
│   ├── loss.py               # Loss definitions
│   ├── visualization.py      # Metric plots
│   ├── roc-auc.py            # ROC-AUC curves
│   ├── app.py                # Streamlit demo
│   ├── app2.py / appx.py     # Alternative app versions
│   ├── img/                  # U-Net figures (e.g. architecture)
│   ├── results/              # Prediction visualizations
│   └── files/                # Checkpoints, CSV logs
└── RESUNET/                  # ResU-Net (TensorFlow 2.0)
    ├── model.py              # ResU-Net architecture
    ├── train.py              # Training script
    ├── eval.py               # Evaluation
    ├── metrics.py            # Dice, IoU (Keras)
    ├── img/                  # ResU-Net figures
    ├── results/              # Predictions
    └── files/                # model.h5, logs
```

---

## 🏗 Architectures

### U-Net (PyTorch)

Classic U-Net: symmetric encoder–decoder with skip connections. Encoder: 4 blocks (64→128→256→512 channels) with max pooling. Bottleneck: 1024 channels. Decoder: upsampling + skip concatenation + convolutions. Output: 1 channel (binary mask).

| ![U-Net architecture](UNET/img/u-net-architecture.png) |
|:--------------------------------------------------------:|
| *U-Net architecture (from [original paper](https://arxiv.org/abs/1505.04597))* |

*Place the file `u-net-architecture.png` in `UNET/img/` if you have a diagram.*

### ResU-Net (TensorFlow 2.0)

ResU-Net adds **residual blocks** inside the U-Net: each block has two 3×3 convs with batch norm and ReLU, plus a 1×1 shortcut. Encoder: 64→128→256, bridge 512, decoder 256→128→64. Final 1×1 conv with sigmoid for binary segmentation.

| ![ResU-Net architecture](RESUNET/img/RESUNET_ARCH.png) |
|:------------------------------------------------------:|
| *ResU-Net architecture (from [paper](https://arxiv.org/pdf/1711.10684.pdf))* |

*Place the file `RESUNET_ARCH.png` in `RESUNET/img/` if you have a diagram.*

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Skin-Lesion-Segmentation-in-TensorFlow-2.0.git
cd Skin-Lesion-Segmentation-in-TensorFlow-2.0
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

- **For U-Net (PyTorch) only:**

  ```bash
  pip install -r requirements-unet.txt
  ```

- **For ResU-Net (TensorFlow) only:**

  ```bash
  pip install -r requirements-resunet.txt
  ```

- **For both (and Streamlit):**

  ```bash
  pip install -r requirements.txt
  ```

  Use `requirements-unet.txt` for U-Net only, or `requirements-resunet.txt` for ResU-Net only.

### 4. Download ISIC 2018 data

Download Task 1 data from the [ISIC 2018 challenge](https://challenge.isic-archive.com/data/) and extract so that paths match the [expected layout](#expected-directory-layout). Set `dataset_path` in the training scripts to your folder.

---

## 🚀 Usage

### U-Net (PyTorch)

1. **Set dataset path**  
   In `UNET/train.py` and `UNET/eval.py`, set:

   ```python
   dataset_path = "path/to/your/ISIC2018_dataset"
   ```

2. **Train**

   ```bash
   cd UNET
   python train.py
   ```

   Checkpoints and CSV logs are saved under `UNET/files/` (e.g. `model.pth`).

3. **Evaluate**

   ```bash
   python eval.py
   ```

   Predictions are saved in `UNET/results/`; metrics in `UNET/files/score.csv`.

4. **Streamlit app** (upload image → segmentation + dimensions)

   ```bash
   streamlit run app.py
   ```

5. **Visualizations**

   ```bash
   python visualization.py   # Metric plots from score.csv
   python loss.py           # Training loss curve (adjust CSV path if needed)
   python roc-auc.py        # ROC-AUC (if you have scores/labels set up)
   ```

### ResU-Net (TensorFlow 2.0)

1. **Set dataset path**  
   In `RESUNET/train.py` and `RESUNET/eval.py`, set:

   ```python
   dataset_path = "path/to/your/ISIC2018_dataset"
   ```

2. **Train**

   ```bash
   cd RESUNET
   python train.py
   ```

   Best model and logs go to `RESUNET/files/` (e.g. `model.h5`, `data.csv`).

3. **Evaluate**

   ```bash
   python eval.py
   ```

   Outputs in `RESUNET/results/` and `RESUNET/files/score.csv`.

---

## 📊 Results & Metrics

Both pipelines report:

- **Accuracy**
- **F1 score** (binary)
- **Jaccard (IoU)**
- **Recall**
- **Precision**

Training uses **Dice-based loss** (and BCE for U-Net). You can compare runs via TensorBoard (U-Net/ResU-Net) and the generated CSV files.

### Example result layout (from evaluation)

Each row: **Input image | Ground truth mask | Predicted mask**

| Input / GT / Prediction |
|:-----------------------:|
| ![Result 1](UNET/results/ISIC_0000012.jpg) |
| ![Result 2](UNET/results/ISIC_0000016.jpg) |
| ![Result 3](UNET/results/ISIC_0000018.jpg) |
| ![Result 4](UNET/results/ISIC_0000019.jpg) |

*Add your own result images under `UNET/results/` or `RESUNET/results/` and reference them here. The names above match the existing README placeholders.*

---

## 📸 Screenshots & Visualizations

### 1. Sample input and mask (repo root)

| Original image | Ground truth mask |
|:--------------:|:-----------------:|
| ![Input](img/image.jpg) | ![Mask](img/mask.png) |

### 2. Streamlit app (U-Net)

- **Upload**: Sidebar “Drag and drop your custom image here”.
- **Output**: Predicted segmentation mask and lesion dimensions (bounding box, area).

*You can add a screenshot here, e.g.:*

<!--
![Streamlit app](docs/screenshots/streamlit_app.png)
-->

### 3. Training curves

- **U-Net**: `UNET/loss.py` and `UNET/visualization.py` plot training loss and metric scatter from `files/score.csv` and `training_losses.csv`.
- **ResU-Net**: Use TensorBoard logs in `RESUNET/files/` (or CSV) and add a screenshot of loss/metrics if you like.

*Example placeholder:*

<!--
![Training loss](docs/screenshots/training_loss.png)
-->

### 4. ROC-AUC (U-Net)

`UNET/roc-auc.py` can generate ROC curves once you wire it to your predicted scores and labels. Add a figure under `docs/screenshots/` and reference it here.

---

## 🔮 Future Work

- [ ] **DeepLabV3+** and other backbones (e.g. ResNet, EfficientNet)
- [ ] **Attention** (e.g. attention U-Net)
- [ ] **Larger resolution** and multi-scale training
- [ ] **Unified config** (YAML/JSON) for paths and hyperparameters
- [ ] **Docker** image for reproducible runs
- [ ] **Export** to ONNX / TFLite for deployment

---

## 🙏 Acknowledgments

- **ISIC** — [International Skin Imaging Collaboration](https://www.isic-archive.com/) and ISIC 2018 challenge organizers
- **U-Net** — [Ronneberger et al.](https://arxiv.org/abs/1505.04597)
- **ResU-Net** — [Zhang et al.](https://arxiv.org/pdf/1711.10684.pdf)
- **TensorFlow** and **PyTorch** communities

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

**Final Year Project** — Skin Lesion Segmentation with Deep Learning.  
For questions or suggestions, open an issue or reach out via your preferred contact.
