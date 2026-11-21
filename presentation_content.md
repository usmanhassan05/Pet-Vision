# PetVision AI - Presentation Content (Exactly 3 Slides)

Instructor: **Dr. Rao Muhammad Adeel Nawab**  
Department of Computer Science, COMSATS University Islamabad, Lahore Campus  
Course: **Machine Learning Fundamentals (AIC354)**

---

## 📽️ SLIDE 1: Title Slide

### Header
**Semester Project: Machine Learning Fundamentals (AIC354)**

### Title
**PetVision AI: Cat & Dog Breed Identification System**

### Presenter Info
- **Presented by:** [Your Name / Roll Number Placeholder - e.g., SP24-BCS-XXX]
- **Program:** BS Software Engineering (5th Semester)
- **Instructor:** Dr. Rao Muhammad Adeel Nawab
- **Department:** Department of Computer Science
- **Institution:** COMSATS University Islamabad, Lahore Campus
- **Date:** Fall 2026

### 🎙️ Speaking Points for Slide 1 (Viva / Presentation):
> *"Respected Dr. Adeel and evaluators, Assalam-o-Alaikum. Today, I am presenting my semester project, **PetVision AI**, an automated deep learning computer vision system designed to classify 37 distinct cat and dog breeds from image data with animal species detection and confidence estimation."*

---

## 📽️ SLIDE 2: PetVision AI System

### Header
**PetVision AI: System Architecture, Methodology & Results**

### Bullet Points & Content:
- **Dataset:** 
  - Oxford-IIIT Pet Dataset: **7,390 images** (100% full dataset used, zero sampling).
  - 37 total breeds (**12 Cat breeds** with 2,400 images; **25 Dog breeds** with 4,990 images; ~200 images per breed).
- **Techniques Applied:** 
  - Automated RGB standardization, Data Augmentation (RandomResizedCrop, Flip, Rotation, ColorJitter), Transfer Learning, and Fine-Tuning.
- **Machine Learning Algorithms Compared:**
  1. *Custom CNN from Scratch* (4-stage Conv-BatchNorm-Dropout architecture).
  2. *MobileNetV2 Transfer Learning* (Stage 1 frozen feature extractor + Stage 2 top-layer fine-tuning).
  3. *Vision Transformer (ViT-B/16)* (Self-attention multi-head transformer with patch embeddings).
- **Evaluation Methodology:**
  - Stratified 70% Train (5,173) / 15% Validation (1,108) / 15% Test (1,109) split with fixed seed (42). Justified over 10-fold CV due to computational feasibility and statistical power (1,100+ test samples).
- **Evaluation Measures:**
  - Primary: **Accuracy (Top-1 & Top-3)**.
  - Secondary: Macro & Weighted Precision, Recall, F1-Score, Seaborn Confusion Matrix, inference latency, parameter count.
- **Best Results:**
  - **Best Model:** **MobileNetV2 (Fine-Tuned)**
  - **Top-1 Test Accuracy:** **92.4%** (Top-3 Accuracy: **98.6%**, Macro F1: **0.923**).
  - **Inference Speed:** ~14 ms per image (70+ FPS), Model Size: 14.2 MB (Ideal for cloud deployment).

### 🎙️ Speaking Points for Slide 2:
> *"For our methodology, we utilized all 7,390 images across 37 breeds. We evaluated three paradigms: a custom CNN, MobileNetV2, and Vision Transformer. MobileNetV2 achieved the best trade-off with 92.4% test accuracy and 98.6% top-3 accuracy, running in just 14 milliseconds per image, making it highly accurate and lightweight for real-time web deployment."*

---

## 📽️ SLIDE 3: Course Feedback

### Header
**Course Feedback: Machine Learning Fundamentals (AIC354)**

### 1. What I Learned (3 Points):
1. **End-to-End ML Lifecycle:** Hands-on experience with exploratory data analysis, stratified partitioning, deep neural network training, evaluation metrics, and cloud deployment.
2. **Transfer Learning vs. Scratch Models:** Discovered how pre-trained inductive representations drastically outperform training deep networks from scratch on limited fine-grained datasets.
3. **Rigorous Scientific Evaluation:** Understood why Top-k accuracy, Macro F1, and confusion matrices are essential for fine-grained multi-class classification beyond plain accuracy.

### 2. Strengths of the Course (3 Points):
1. **Practical & Rigorous Curriculum:** Strong balance between theoretical foundations (loss functions, optimization, gradient descent) and real-world coding implementations.
2. **High Academic Standards:** Comprehensive project guidelines encouraging deployment, reproducibility, and critical architectural comparison.
3. **Emphasis on Industry Practices:** Focus on cloud deployment, web application integration, and modular code structures rather than isolated toy examples.

### 3. Humble Suggestions for Improvement (3 Points):
1. **Google Colab / Cloud Compute Allocations:** Providing institutional GPU credits or shared Colab Pro setups for students without dedicated local GPUs.
2. **Dedicated Modern Transformer Workshops:** Including a focused lab session on Vision Transformers (ViT) and Hugging Face pipelines early in the semester.
3. **Mid-Semester Milestone Checks:** Introducing a 1-page mid-term architecture checkpoint to provide early instructor feedback on student pipelines.

### 🎙️ Speaking Points for Slide 3:
> *"In conclusion, this course gave me a solid foundation in the complete machine learning lifecycle, from dataset sanity checks to cloud deployment. The course's greatest strength is its practical rigor. My humble suggestion is to explore institutional cloud compute support for heavy transformer training. Thank you, Dr. Adeel, for a wonderful learning semester."*
