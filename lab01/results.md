# Results — Skin Lesion Classification on HAM10000

Pipeline: 8 ImageNet-pretrained CNN backbones fine-tuned on the leakage-free
(lesion-level) HAM10000 split, weighted cross-entropy for class imbalance,
best backbone's deep features re-used with 7 classical ML classifiers, plus
a computational efficiency comparison across all backbones.

- Best backbone (by macro F1): **ResNet101**
- Best classifier on ResNet101 deep features (by macro F1): **XGBoost**

## Table 1. Comparison of Transfer Learning Models

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|
| AlexNet | 70.40 | 50.11 | 59.46 | 53.36 | 93.10 |
| VGG16 | 76.77 | 62.14 | 63.19 | 60.98 | 94.84 |
| VGG19 | 76.97 | 59.34 | 65.49 | 61.31 | 94.39 |
| ResNet18 | 76.97 | 62.01 | 66.50 | 62.53 | 93.95 |
| ResNet50 | 80.20 | 61.34 | 66.49 | 63.52 | 94.01 |
| ResNet101 | 80.91 | 65.09 | 73.60 | **68.59** | 92.50 |
| DenseNet121 | **82.22** | **67.54** | 70.15 | 68.53 | **94.98** |
| EfficientNet-B0 | 78.99 | 62.73 | **73.18** | 65.89 | 93.05 |

*Metrics are macro-averaged, on the held-out test set. DenseNet121 has the
highest raw accuracy/precision/AUC, but ResNet101 edges it out on macro F1
(the metric used to pick the backbone) by balancing precision/recall
better across the minority classes.*

## Table 2. Comparison of Different Classifiers (on ResNet101 deep features)

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|---|
| Deep Features (ResNet101) | Logistic Regression | 81.72 | 68.14 | 67.68 | 67.64 | 90.41 |
| Deep Features (ResNet101) | Decision Tree | 76.16 | 62.09 | 60.15 | 60.52 | 76.85 |
| Deep Features (ResNet101) | Random Forest | 82.12 | 72.65 | 65.15 | 68.34 | 94.94 |
| Deep Features (ResNet101) | K-Nearest Neighbors (KNN) | 82.22 | 69.34 | 65.80 | 67.31 | 87.31 |
| Deep Features (ResNet101) | Linear SVM | 82.32 | 67.96 | 68.06 | 67.70 | 94.48 |
| Deep Features (ResNet101) | RBF-SVM | 81.31 | 65.88 | 68.75 | 66.95 | 95.17 |
| Deep Features (ResNet101) | XGBoost | **82.93** | 71.08 | 67.59 | **69.10** | **95.33** |

*XGBoost on top of frozen ResNet101 features slightly outperforms end-to-end
fine-tuned ResNet101 itself (69.10 vs 68.59 macro F1), at far lower training
cost since only the classifier head needs retraining.*

## Table 3. Computational Efficiency Comparison

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
|---|---|---|---|---|---|
| AlexNet | 57.03 | 228.14 | 0.71 | 2.10 | 70.40 |
| VGG16 | 134.29 | 537.17 | 15.47 | 9.27 | 76.77 |
| VGG19 | 139.60 | 558.41 | 19.63 | 10.79 | 76.97 |
| ResNet18 | 11.18 | 44.80 | 1.82 | 3.27 | 76.97 |
| ResNet50 | 23.52 | 94.41 | 4.13 | 12.34 | 80.20 |
| ResNet101 | 42.51 | 170.68 | 7.86 | 28.66 | 80.91 |
| DenseNet121 | 6.96 | 28.45 | 2.90 | 36.95 | **82.22** |
| EfficientNet-B0 | **4.02** | **16.36** | **0.41** | 20.82 | 78.99 |

*EfficientNet-B0 is by far the lightest (4M params, 16MB, 0.41 GFLOPs) while
still beating AlexNet/VGG16/VGG19 on accuracy — best accuracy-per-FLOP
trade-off. DenseNet121 gives the best raw accuracy at a fraction of
VGG16/19's size. Inference time doesn't track FLOPs cleanly (DenseNet121 is
slower per-image than ResNet101 despite fewer FLOPs) because dense
concatenation isn't as parallel-friendly as sequential conv blocks on this
GPU/batch-size-1 setup.*

---
**Setup notes:** ImageNet-pretrained backbones, fine-tuned end-to-end with
`Adam` (lr 5e-5, weight decay 1e-5), class-weighted cross-entropy loss,
mixed precision (AMP), gradient clipping, `ReduceLROnPlateau` on val macro
F1, early stopping (patience 6), up to 15 epochs. All splits are lesion-level
(no image from the same lesion appears in more than one split) to avoid
data leakage. Classical classifiers were trained on standardized deep
features extracted from the frozen best-backbone penultimate layer,
fit on train+val, evaluated on the untouched test set.
