# Lab 05: HOG-Based Industrial Defect Detection and Classification

## 1. Introduction
This report details the implementation and analysis of Histogram of Oriented Gradients (HOG) feature extraction for classifying industrial steel surface defects using the NEU-DET dataset. It explores image preprocessing, feature extraction, the performance of traditional machine learning classifiers (SVM and Random Forest), and evaluates the robustness of the system against various image degradations (brightness, noise, rotation, and blur). Finally, it demonstrates a simulated industrial quality-control decision module.

## 2. Experimental Setup
*   **Dataset**: NEU Steel Surface Defect Detect (Classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches).
*   **Preprocessing**: Grayscale conversion, resized to a common resolution of 64x64.
*   **Feature Extraction**: HOG (Orientations: 9, Pixels per cell: 8x8, Cells per block: 2x2).
*   **Classifiers**: Support Vector Machine (Linear Kernel), Random Forest (100 estimators).
*   **Robustness Tests**: Brightness scaling, Gaussian Noise, Rotation (15 degrees), Gaussian Blur.

## 3. Results and Observations

### Table 1: Classification Performance Comparison
| Classifier | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Support Vector Machine (SVM) | 76.00% | 77.34% | 76.00% | 75.32% |
| Random Forest | 82.00% | 83.79% | 82.00% | 81.00% |

*Observation*: Random Forest outperformed the linear SVM across all metrics, achieving an accuracy of 82.00%. Both models demonstrated that HOG features provide a strong baseline for texture-based defect classification, effectively capturing the structural edges of the various surface defects.

### Table 2: Model Robustness Analysis (SVM)
| Condition | Accuracy | Change in Accuracy | F1-Score | Change in F1-Score |
|---|---|---|---|---|
| Baseline | 76.00% | - | 75.32% | - |
| Brightness | 55.00% | -21.00% | 53.59% | -21.73% |
| Rotation (15°) | 57.00% | -19.00% | 55.09% | -20.24% |
| Gaussian Noise | 32.00% | -44.00% | 27.34% | -47.98% |
| Gaussian Blur | 25.00% | -51.00% | 16.39% | -58.93% |

*Observation*: The HOG-based SVM classifier is highly sensitive to image degradations that disrupt local gradient information. Gaussian Blur had the most catastrophic impact (reducing accuracy by 51.00%), as it smooths out the high-frequency edges that HOG relies upon. Gaussian noise also severely corrupted the gradient histograms, leading to a 44.00% drop in accuracy. While HOG is theoretically somewhat invariant to illumination changes (due to block normalization), extreme brightness scaling still degraded performance.

## 4. Discussion & Conclusion
This laboratory demonstrated the complete pipeline for a traditional computer vision classification system. While hand-crafted features like HOG are computationally efficient and provide interpretable gradient representations, they lack the invariant properties of learned deep features. The robustness experiments highlight a critical limitation of traditional CV approaches in industrial settings: they require strictly controlled environments (consistent lighting, sharp focus, minimal noise) to maintain reliability. The final quality-control module successfully integrated the model to output actionable industrial decisions (e.g., "DEFECTIVE - REJECT PRODUCT"), demonstrating the practical application of this pipeline.
