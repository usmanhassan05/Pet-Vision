# PetVision AI - 15 Viva Questions & Simple Answers (With Speaking Points)

Course: **Machine Learning Fundamentals (AIC354)**  
Instructor: **Dr. Rao Muhammad Adeel Nawab**  
Institution: **COMSATS University Islamabad, Lahore Campus**

---

### Q1: What is the exact problem your project solves?
**Answer:** PetVision AI solves fine-grained multi-class image classification on the Oxford-IIIT Pet dataset. It takes any user-provided cat or dog image, predicts which of the 37 breeds it belongs to, determines whether it is a Cat or a Dog, and outputs top-3 candidate predictions with calibrated confidence scores.

### Q2: Why did you use a Stratified Train/Val/Test Split instead of 10-Fold Cross-Validation?
**Answer:** 
1. **Computational Infeasibility:** 10-fold CV on 3 deep neural networks (Custom CNN, MobileNetV2, ViT) requires 30 separate training runs, which is computationally prohibitive.
2. **Statistical Adequacy:** With 7,390 images, a 15% test split reserves 1,109 completely unseen images (~30 images per class), which is statistically more than sufficient for evaluating generalization.
3. **Stratification Guarantee:** Stratification ensures identical class proportions (70% train, 15% val, 15% test) for each of the 37 classes, avoiding data leakage and bias.

### Q3: How did you extract labels and species from the dataset?
**Answer:** We derived labels directly from image filenames using the regular expression `^([A-Za-z_]+)_\d+\.jpg$`. By standard Oxford-IIIT convention, any breed starting with a capital letter is a **Cat** (e.g., `Abyssinian`, `Persian`), while lowercase breeds are **Dogs** (e.g., `beagle`, `boxer`). This captured all 7,390 images, whereas `list.txt` only listed 7,349 images.

### Q4: Why did you convert all images using `.convert('RGB')`?
**Answer:** The dataset scan revealed that while 7,378 images are standard RGB, 6 are Palette (`P`), 3 are RGBA (4 channels), and 3 are Grayscale (`L`, 1 channel). If unhandled, stacking them into mini-batch PyTorch tensors would trigger fatal channel dimension mismatch exceptions `[B, 4, H, W]` vs `[B, 3, H, W]`. `.convert('RGB')` ensures a uniform 3-channel input shape.

### Q5: What is the architectural difference between Custom CNN and MobileNetV2?
**Answer:** 
- The **Custom CNN** uses standard full-rank 2D convolutions trained from scratch with randomly initialized weights, which struggle on small fine-grained datasets.
- **MobileNetV2** uses *Depthwise Separable Convolutions* and *Inverted Residual Blocks with Linear Bottlenecks*, pre-trained on ImageNet (1.4 million images). It has learned generic visual primitives (edges, textures, shapes), requiring far fewer parameters and training much faster.

### Q6: How did you perform Two-Stage Transfer Learning on MobileNetV2?
**Answer:**
- **Stage 1 (Feature Extraction):** The pre-trained backbone was completely frozen (`requires_grad = False`). Only the custom 37-class classification head was trained with an initial learning rate ($1 \times 10^{-3}$) for 5 epochs.
- **Stage 2 (Fine-Tuning):** The top inverted residual blocks were unfrozen, and the network was trained end-to-end with a much lower learning rate ($1 \times 10^{-4}$) to adapt high-level feature maps without destroying pre-trained weights (avoiding catastrophic forgetting).

### Q7: What are the key architectural features of a Vision Transformer (ViT)?
**Answer:** ViT eliminates convolutional filters. It divides a 224x224 image into a grid of 16x16 non-overlapping patches (196 tokens), adds 1D learnable position embeddings, appends a `[CLS]` classification token, and passes them through Multi-Head Self-Attention (MHSA) blocks. Self-attention captures global dependencies across the entire image from the very first layer.

### Q8: What callbacks did you use during training and why?
**Answer:**
1. **EarlyStopping:** Monitored `val_loss` with patience = 5 to prevent overfitting and stop training when validation performance plateaued.
2. **ReduceLROnPlateau:** Dynamically reduced learning rate by a factor of 0.5 when validation loss stagnated, enabling fine convergence in optimization minima.
3. **Model Checkpointing:** Saved the best model weights based on minimum validation loss so the final model was never degraded by later overfitted epochs.

### Q9: Why is Top-3 Accuracy particularly important for pet breed identification?
**Answer:** Fine-grained pet recognition involves visually ambiguous breeds that even human experts confuse (e.g., *Staffordshire Bull Terrier* vs *American Pit Bull Terrier*, or *Ragdoll* vs *Birman*). Top-3 accuracy measures whether the true breed is among the model's top 3 highest-probability guesses, achieving >98% accuracy and providing realistic utility in practical clinical/veterinary applications.

### Q10: What is the difference between Macro F1 and Weighted F1?
**Answer:**
- **Macro F1:** Computes the F1-score independently for each class and averages them unweighted. It gives equal importance to every class regardless of size.
- **Weighted F1:** Averages F1-scores weighted by the support (number of true instances) of each class.
Since our 37 breeds are virtually balanced (~30 test images each), Macro and Weighted F1 are both close (~0.92).

### Q11: What are the most confused breed pairs and why?
**Answer:** The confusion matrix revealed the highest confusion between:
1. `american_pit_bull_terrier` and `staffordshire_bull_terrier` (both share muscular builds, short coats, and similar facial bone structures).
2. `Ragdoll` and `Birman` (both are pointed cat breeds with semi-longhair blue coats and blue eyes).
3. `Bengal` and `Egyptian_Mau` (both have distinct spotted tabby coats).

### Q12: How does data augmentation help prevent overfitting?
**Answer:** Data augmentation artificially expands the training distribution by applying stochastic transformations (`RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation(15)`, and `ColorJitter`). This prevents the deep neural network from memorizing exact pixel configurations, forcing it to learn invariant features such as ear shapes, facial morphology, and coat markings.

### Q13: What model is chosen as the BEST model and why?
**Answer:** **MobileNetV2 (Fine-Tuned)** is chosen as the best model. While Vision Transformer achieves strong raw capacity, MobileNetV2 achieves a virtually identical Top-1 accuracy (**92.4%**) with **only 2.8 million parameters (~14 MB file size)** and an ultra-fast inference latency of **14 ms per image**, compared to ViT's 86 million parameters (340 MB) and 45 ms latency. This makes MobileNetV2 the ideal production-grade model for web and mobile deployment.

### Q14: How is your Streamlit web application optimized for user experience?
**Answer:**
1. Uses `@st.cache_resource` so model weights and transforms load only once into memory.
2. Provides Top-1 prediction with dynamic species badge (🐱 Cat vs 🐶 Dog) and interactive Top-3 confidence progress bars.
3. Features both file upload and built-in 1-click sample images so evaluators can test instantly without uploading their own photos.
4. Robust error handling prevents crashes if invalid files are submitted.

### Q15: If given more time and computational resources, how would you improve PetVision AI?
**Answer:**
1. **Object Detection / Crop Preprocessing:** Integrate a lightweight YOLOv8 head detector to crop pet heads prior to classification, eliminating distracting background clutter.
2. **Ensemble Modeling:** Combine soft-voting predictions from ViT and MobileNetV2 to capture both global attention and local convolutional features.
3. **Deployment as Mobile App:** Export the model to ONNX / CoreML / TFLite for real-time on-device inference without needing cloud servers.
