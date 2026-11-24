---
title: PetVision AI Breed Identifier
emoji: 🐾
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.33.0
app_file: app.py
pinned: false
license: mit
---

# PetVision AI: Cat & Dog Breed Identification System 🐾

[![Course: AIC354](https://img.shields.io/badge/Course-AIC354%20Machine%20Learning-blue.svg)](https://lahore.comsats.edu.pk/)
[![Instructor: Dr. Rao Muhammad Adeel Nawab](https://img.shields.io/badge/Instructor-Dr.%20Rao%20Muhammad%20Adeel%20Nawab-green.svg)](https://lahore.comsats.edu.pk/)
[![Institution: COMSATS Lahore](https://img.shields.io/badge/COMSATS-Lahore%20Campus-red.svg)](https://lahore.comsats.edu.pk/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.x-orange.svg)](https://pytorch.org/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

> **Academic Project:** Machine Learning Fundamentals (AIC354)  
> **Institution:** Department of Computer Science, COMSATS University Islamabad, Lahore Campus  
> **Instructor:** Dr. Rao Muhammad Adeel Nawab  
> **Semester:** Fall 2026 (5th Semester BS Software Engineering)  
> **Deadline:** Sunday 29 Nov 2026 23:59 (Full Marks Scope: 40/40 Marks)

---

## 🌐 Live Web Application & Repository Links
- **🚀 Live Cloud Web Application:** [https://petvision-ai.streamlit.app/](https://petvision-ai.streamlit.app/) *(Hosted on Streamlit Community Cloud)*
- **📂 GitHub Code Repository:** [https://github.com/usmanhassan05/Pet-Vision](https://github.com/usmanhassan05/Pet-Vision)

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Scan & Sanity Analysis](#-dataset-scan--sanity-analysis)
3. [Stratified Splitting vs. 10-Fold CV](#-stratified-splitting-vs-10-fold-cv)
4. [Model Architectures Compared](#-model-architectures-compared)
5. [Empirical Evaluation & Results](#-empirical-evaluation--results)
6. [Error Analysis & Confused Breeds](#-error-analysis--confused-breeds)
7. [Extra Marks Features](#-extra-marks-features)
8. [Streamlit Web Application](#-streamlit-web-application)
9. [Google Colab GPU Guide](#-google-colab-gpu-guide)
10. [Submission & Viva Preparation](#-submission--viva-preparation)

---

## 🎯 Project Overview
Fine-grained image classification involves identifying subtle differences between sub-categories. **PetVision AI** is an autonomous deep learning system trained on the **Oxford-IIIT Pet Dataset** to recognize **37 distinct breeds of cats and dogs** from arbitrary photographs, identify animal species (**Cat** vs. **Dog**), and present **Top-3 confidence probabilities**.

---

## 📊 Dataset Scan & Sanity Analysis
A strict scan of `dataset/images/images/` was conducted without sampling or altering raw files:
- **Total Images:** Exactly **7,390 valid `.jpg` images** (100% full dataset utilized).
- **Number of Classes:** **37 Breeds** (12 Cat breeds with 2,400 images; 25 Dog breeds with 4,990 images).
- **Class Balance:** Approximately **200 images per breed** (~2.7% per class).
- **Corrupt Files:** **0 corrupt images** verified via Pillow.
- **Color Mode Normalization:** 7,378 RGB, 6 Palette (`P`), 3 RGBA, 3 Grayscale (`L`). All dynamically converted to 3-channel RGB via `.convert('RGB')` to prevent batch tensor dimension crashes.
- **Label Extraction:** Derived using regex `^([A-Za-z_]+)_\d+\.jpg$`. By Oxford convention, initial capital letter indicates **Cat** (e.g., `Persian`), while lowercase indicates **Dog** (e.g., `beagle`).

---

## ✂️ Stratified Splitting vs. 10-Fold CV
Data is partitioned using a **Stratified 70% Train / 15% Validation / 15% Test Split** with fixed seed (`42`):
- **Train Set:** **5,173 images (70.00%)**
- **Validation Set:** **1,108 images (14.99%)**
- **Test Set:** **1,109 images (15.01%)**

### Academic Justification (Why Stratified Split over 10-Fold CV?)
1. **Computational Feasibility:** Training 3 deep neural architectures (Custom CNN, MobileNetV2, ViT-B/16) across 10 folds requires **30 full training runs**, which is computationally prohibitive and environmentally inefficient.
2. **Statistical Adequacy:** With 7,390 images, a 15% test set yields **1,109 unseen test images (~30 images per breed)**, providing high statistical confidence without data leakage.
3. **Zero Class Skew:** Stratification guarantees identical class proportions across all three subsets.

---

## 🧠 Model Architectures Compared
All models are evaluated on the exact same unseen test set (1,109 images):

1. **Custom CNN from Scratch:** 4-stage convolutional network (Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout2d) with AdaptiveAvgPool2d and dense classification head.
2. **MobileNetV2 Transfer Learning:** Depthwise separable convolutions pre-trained on ImageNet. Two-stage training:
   - *Stage 1:* Frozen backbone + newly trained classification head ($LR = 10^{-3}$, 5 epochs).
   - *Stage 2:* Unfrozen top inverted residual blocks for fine-tuning ($LR = 10^{-4}$).
3. **Vision Transformer (ViT-B/16):** 16x16 patch projection (196 tokens), 12 multi-head self-attention transformer encoder layers, fine-tuned classification head.

---

## 📈 Empirical Evaluation & Results

### Final Comparison Table
| Metric | 1. Custom CNN (Scratch) | 2. MobileNetV2 (Fine-Tuned) ⭐️ | 3. Vision Transformer (ViT-B/16) |
| :--- | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | 58.4% | **92.4%** | **93.1%** |
| **Top-3 Test Accuracy** | 79.2% | **98.6%** | **98.9%** |
| **Macro F1-Score** | 0.578 | **0.923** | **0.930** |
| **Weighted F1-Score** | 0.582 | **0.924** | **0.931** |
| **Inference Latency (ms)** | **6.2 ms** | **14.1 ms** (70+ FPS) | 44.8 ms |
| **Total Parameters** | 1.4 M | **2.8 M** | 86.2 M |
| **Model Disk Size** | **5.6 MB** | **14.2 MB** | 340.5 MB |

### 🏆 Verdict: Best Model Selection
**MobileNetV2 (Fine-Tuned)** is chosen as the superior model. While ViT achieves a marginal 0.7% higher Top-1 accuracy, MobileNetV2:
- Operates over **$3\times$ faster** (14.1 ms vs 44.8 ms).
- Consumes **$24\times$ less disk space** (14.2 MB vs 340.5 MB).
- Achieves an outstanding **98.6% Top-3 Accuracy**.
This delivers the optimal accuracy-to-compute ratio for cloud and mobile deployment.

---

## 🔍 Error Analysis & Confused Breeds
Fine-grained misclassifications reflect genetic and morphological proximity:
1. `american_pit_bull_terrier` vs. `staffordshire_bull_terrier` (Identical muscular bone structure and short coats).
2. `Ragdoll` vs. `Birman` (Pointed color schemes with blue eyes and semi-longhair coats).
3. `Bengal` vs. `Egyptian_Mau` (Spotted tabby patterns).

---

## 🌟 Extra Marks Features
- **Data Augmentation Ablation:** Demonstrates invariance to scale, rotation, mirroring, and lighting variations (`results/augmentation_impact.png`).
- **Grad-CAM Explainability:** Class activation mapping highlighting discriminative pet regions (eyes, ears, snout) (`src/explainability.py`).
- **Top-3 Accuracy Metrics:** Evaluated and integrated directly into the web application.

---

## 💻 Streamlit Web Application

To run the interactive web application locally:
```bash
streamlit run app/app.py
```
**App Features:**
- Upload JPG, JPEG, PNG or select built-in 1-click sample images.
- Instant Top-1 prediction with dynamic species badge (🐱 Cat vs 🐶 Dog).
- Interactive Top-3 confidence progress bars.
- Cached model loading (`@st.cache_resource`).

---

## 🚀 Google Colab GPU Guide

If you wish to re-train the heavy models on free cloud GPUs:
1. Upload this project folder to your Google Drive under `MyDrive/ML-LAB-PROJECT`.
2. Open Google Colab and open `notebooks/PetVision_AI.ipynb`.
3. In Colab, select: **Runtime -> Change runtime type -> T4 GPU**.
4. Run the first cell to mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
5. Run all cells sequentially. Checkpoints are automatically saved to `models/best_model.pth`.

---

## 📁 Repository Structure
```text
ML-LAB-PROJECT/
├── dataset/                    # Oxford-IIIT Pet Dataset (Raw, untouched)
│   ├── annotations/
│   │   └── annotations/        # xmls, trimaps, list.txt
│   └── images/
│       └── images/             # 7,390 JPG images + 3 .mat files
├── notebooks/
│   └── PetVision_AI.ipynb      # Complete Colab-ready Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── data.py                 # Scanning, stratified splitting, PyTorch Dataset
│   ├── models.py               # Custom CNN, MobileNetV2, ViT-B/16
│   ├── train.py                # Training loop, EarlyStopping, LR decay
│   ├── evaluate.py             # Metrics calculation (Acc, F1, Latency, Top-3)
│   ├── utils.py                # Visualizations & seed utilities
│   ├── explainability.py       # Grad-CAM implementation
│   ├── augmentation_analysis.py# Data augmentation visualizer
│   └── run_eda_and_split.py    # Automated pipeline runner
├── models/
│   ├── model_metadata.json     # Architecture specs & metrics
│   └── best_model.pth          # Fine-tuned model checkpoint
├── results/                    # 13 high-resolution plots, matrices & tables
├── app/
│   ├── app.py                  # Streamlit web application
│   └── requirements.txt        # Deployment dependencies
├── presentation_content.md     # Fixed 3-slide presentation text & speaking points
├── viva_questions.md           # 15 key viva questions & answers
├── deployment_guide.md         # Free cloud deployment tutorial (Hugging Face / Streamlit)
├── requirements.txt            # Local development dependencies
└── README.md                   # Full documentation
```

---

## 🎓 Viva Preparation & Presentation
- **Presentation Slides:** Formatted in `presentation_content.md` according to instructor Dr. Adeel's strict 3-slide template (Title, PetVision AI System, Course Feedback) with verbatim speaking points.
- **Viva Questions:** Complete answers to 15 anticipated viva questions provided in `viva_questions.md`.
